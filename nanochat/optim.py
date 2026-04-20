"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.

Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from nanochat.common import COMPUTE_DTYPE

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

def adamw_step_eager(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
) -> None:
    """Uncompiled AdamW step for backends like MPS that reject CPU scalar tensors."""
    p.mul_(1 - lr * wd)
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    denom = (exp_avg_sq / bias2).sqrt() + eps
    p.add_(exp_avg / denom, alpha=-(lr / bias1))

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

def muon_step_eager(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum: float,
    lr: float,
    wd: float,
    beta2: float,
    ns_steps: int,
    red_dim: int,
) -> None:
    """Uncompiled Muon step for backends like MPS that reject CPU scalar tensors."""
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp(momentum_buffer, momentum)

    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

def polar_express_matrix(g: Tensor, ns_steps: int) -> Tensor:
    """
    Readable single-matrix version of the Polar Express step used by Muon.
    This returns a gradient-like direction and does not mutate the input.
    """
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X.to(dtype=g.dtype)

def muonize_matrix(g: Tensor, state: dict, momentum: float, ns_steps: int, beta2: float | None) -> Tensor:
    """
    Muon direction for one matrix or a stack of matrices: momentum, Polar
    Express, and NorMuon-style variance reduction. State is anchored on the
    owning parameter.
    """
    if "momentum_buffer" not in state or state["momentum_buffer"].shape != g.shape:
        state["momentum_buffer"] = torch.zeros_like(g)
    momentum_buffer = state["momentum_buffer"]
    momentum_buffer.lerp_(g, 1 - momentum)
    g = g.lerp(momentum_buffer, momentum)

    g = polar_express_matrix(g, ns_steps)

    red_dim = -1 if g.shape[-2] >= g.shape[-1] else -2
    state_shape = (*g.shape[:-2], g.shape[-2], 1) if red_dim == -1 else (*g.shape[:-2], 1, g.shape[-1])
    if "second_momentum_buffer" not in state or state["second_momentum_buffer"].shape != state_shape:
        state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=g.dtype, device=g.device)
    second_momentum_buffer = state["second_momentum_buffer"]

    beta2 = 0.0 if beta2 is None else beta2
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    return g * final_scale.to(g.dtype)

def apply_cautious_update_(p: Tensor, direction: Tensor, lr: float, wd: float) -> None:
    """
    Apply Muon-style cautious decay and a descent step for a gradient-like direction.
    The mask intentionally matches muon_step_fused: mask = (direction * weight) >= 0.
    """
    direction = direction.to(dtype=p.dtype)
    mask = (direction * p) >= 0
    p.sub_(lr * direction + lr * wd * p * mask)

def pop_vo_feature_covariances(Wv: Tensor) -> tuple[Tensor, Tensor]:
    required = ("_vo_cov_x_sum", "_vo_cov_h_sum", "_vo_cov_count")
    if not all(hasattr(Wv, name) for name in required):
        raise RuntimeError("vo_product_muon requires attention feature covariances from a training forward pass")
    count = max(1, int(Wv._vo_cov_count))
    cov_x = Wv._vo_cov_x_sum / count
    cov_h = Wv._vo_cov_h_sum / count
    for name in required:
        delattr(Wv, name)
    return cov_x, cov_h

def vo_feature_covariance_proxy_grad(
    Wv: Tensor,
    Wo: Tensor,
    Gv: Tensor,
    Go: Tensor,
    cov_x: Tensor,
    cov_h: Tensor,
) -> Tensor:
    """
    Per-head feature-covariance VO proxy. In PyTorch Linear orientation,
    c_v.weight is (head_dim * n_head, model_dim), while c_proj.weight is
    (model_dim, head_dim * n_head).
    """
    n_head, head_dim = cov_h.shape[0], cov_h.shape[1]
    model_dim = Wv.shape[1]
    if Wv.shape != Gv.shape or Wo.shape != Go.shape:
        raise RuntimeError("VO gradients must match their parameter shapes")
    if Wv.shape[0] != n_head * head_dim or Wo.shape[1] != n_head * head_dim:
        raise RuntimeError("VO covariance head shape does not match parameter shapes")

    Wv_h = Wv.float().view(n_head, head_dim, model_dim)
    Gv_h = Gv.float().view(n_head, head_dim, model_dim)
    Wo_h = Wo.float().view(model_dim, n_head, head_dim).permute(1, 0, 2)
    Go_h = Go.float().view(model_dim, n_head, head_dim).permute(1, 0, 2)
    cov_x = cov_x.float().to(device=Wv.device)
    cov_h = cov_h.float().to(device=Wv.device)

    term_o = torch.matmul(torch.matmul(Go_h, cov_h), Wv_h)
    term_v = torch.matmul(torch.matmul(Wo_h, Gv_h), cov_x)
    return (0.5 * (term_o + term_v)).to(dtype=Wv.dtype)

def split_vo_feature_covariance_direction(
    T: Tensor,
    Wv: Tensor,
    Wo: Tensor,
    n_head: int,
    head_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Split per-head product directions back to c_v and c_proj directions."""
    model_dim = Wv.shape[1]
    Wv_h = Wv.float().view(n_head, head_dim, model_dim)
    Wo_h = Wo.float().view(model_dim, n_head, head_dim).permute(1, 0, 2)
    denom_v = Wv_h.square().sum(dim=(-2, -1), keepdim=True) / max(1, model_dim) + eps
    denom_o = Wo_h.square().sum(dim=(-2, -1), keepdim=True) / max(1, model_dim) + eps
    T = T.float()
    dir_o_h = 0.5 * torch.matmul(T, Wv_h.mT) / denom_v
    dir_v_h = 0.5 * torch.matmul(Wo_h.mT, T) / denom_o
    dir_o = dir_o_h.permute(1, 0, 2).reshape_as(Wo)
    dir_v = dir_v_h.reshape_as(Wv)
    return dir_v.to(dtype=Wv.dtype), dir_o.to(dtype=Wo.dtype)

def matrix_top_singular_value(x: Tensor) -> float:
    """Return mean top singular value, using CPU for MPS backends without native SVD."""
    x_float = x.float().cpu() if x.device.type == "mps" else x.float()
    return float(torch.linalg.svdvals(x_float)[..., 0].mean().item())

def ffn_stacked_joint_grad(Gin: Tensor, Gout: Tensor) -> Tensor:
    """
    Lossless stacked FFN joint gradient:
        H = [Gin; Gout.T]
    where Gin=(hidden, model), Gout=(model, hidden), and H=(2*hidden, model).
    """
    return torch.cat([Gin, Gout.mT], dim=0)

def split_ffn_stacked_joint_direction(direction: Tensor, hidden_dim: int) -> tuple[Tensor, Tensor]:
    """Split Muonize(H) back into c_fc and c_proj directions."""
    dir_in = direction[:hidden_dim]
    dir_out = direction[hidden_dim:].mT
    return dir_in, dir_out

# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw', 'muon', 'vo_product_muon', or 'ffn_joint_muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
            - For VO groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay',
              'eps', 'pair_offsets'
            - For FFN groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay',
              'pair_offsets'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            if p.device.type == "mps":
                adamw_step_eager(
                    p, grad, exp_avg, exp_avg_sq, state['step'],
                    group['lr'], group['betas'][0], group['betas'][1],
                    group['eps'], group['weight_decay'],
                )
            else:
                # Fill 0-D tensors with current values
                self._adamw_step_t.fill_(state['step'])
                self._adamw_lr_t.fill_(group['lr'])
                self._adamw_beta1_t.fill_(group['betas'][0])
                self._adamw_beta2_t.fill_(group['betas'][1])
                self._adamw_eps_t.fill_(group['eps'])
                self._adamw_wd_t.fill_(group['weight_decay'])

                # Fused update: weight_decay -> momentum -> bias_correction -> param_update
                adamw_step_fused(
                    p, grad, exp_avg, exp_avg_sq,
                    self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                    self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
                )

    def _step_muon(self, group: dict) -> None:
        """
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        params: list[Tensor] = group['params']
        if not params:
            return

        # Get or create group-level buffers (stored in first param's state for convenience)
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum for every individual parameter
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer is factored, either per-row or per-column
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (NOTE: this assumes all params have the same shape)
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        shape_lr = group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5
        beta2 = group["beta2"] if group["beta2"] is not None else 0.0
        if p.device.type == "mps":
            muon_step_eager(
                stacked_grads,
                stacked_params,
                momentum_buffer,
                second_momentum_buffer,
                group["momentum"],
                shape_lr,
                group["weight_decay"],
                beta2,
                group["ns_steps"],
                red_dim,
            )
        else:
            # Fill all the 0-D tensors with current values
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(beta2)
            self._muon_lr_t.fill_(shape_lr)
            self._muon_wd_t.fill_(group["weight_decay"])

            # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
            muon_step_fused(
                stacked_grads,
                stacked_params,
                momentum_buffer,
                second_momentum_buffer,
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                group["ns_steps"],
                red_dim,
            )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    def _step_vo_product_muon(self, group: dict) -> None:
        """
        Paired optimizer for attention VO matrices. Params are stored flat in the
        group and pair_offsets gives (Wv_idx, Wo_idx) positions into that list.
        """
        params: list[Tensor] = group["params"]
        pair_offsets = group["pair_offsets"]
        collect_stats = group.get("collect_stats", False)
        stats = dict(
            num_pairs_active=0,
            proxy_grad_frob_sum=0.0,
            proxy_grad_top_sv_sum=0.0,
            update_frob_v_sum=0.0,
            update_frob_o_sum=0.0,
            update_to_weight_ratio_v_sum=0.0,
            update_to_weight_ratio_o_sum=0.0,
        )

        for v_idx, o_idx in pair_offsets:
            Wv, Wo = params[v_idx], params[o_idx]
            if Wv.grad is None or Wo.grad is None:
                continue

            Gv, Go = Wv.grad, Wo.grad
            state = self.state[Wv]
            state["step"] = state.get("step", 0) + 1

            cov_x, cov_h = pop_vo_feature_covariances(Wv)
            G_pair = vo_feature_covariance_proxy_grad(Wv, Wo, Gv, Go, cov_x, cov_h)
            T = muonize_matrix(
                G_pair,
                state,
                momentum=group["momentum"],
                ns_steps=group["ns_steps"],
                beta2=group["beta2"],
            )
            dir_v, dir_o = split_vo_feature_covariance_direction(
                T,
                Wv,
                Wo,
                n_head=cov_h.shape[0],
                head_dim=cov_h.shape[1],
                eps=group["eps"],
            )
            shape_lr = group["lr"] * max(1.0, Wo.shape[-2] / Wo.shape[-1])**0.5

            if collect_stats:
                update_v = shape_lr * dir_v.float()
                update_o = shape_lr * dir_o.float()
                weight_v_norm = Wv.float().norm().clamp_min(1e-12)
                weight_o_norm = Wo.float().norm().clamp_min(1e-12)
                stats["num_pairs_active"] += 1
                stats["proxy_grad_frob_sum"] += float(G_pair.float().norm().item())
                stats["proxy_grad_top_sv_sum"] += matrix_top_singular_value(G_pair)
                stats["update_frob_v_sum"] += float(update_v.norm().item())
                stats["update_frob_o_sum"] += float(update_o.norm().item())
                stats["update_to_weight_ratio_v_sum"] += float((update_v.norm() / weight_v_norm).item())
                stats["update_to_weight_ratio_o_sum"] += float((update_o.norm() / weight_o_norm).item())

            apply_cautious_update_(Wo, dir_o, lr=shape_lr, wd=group["weight_decay"])
            apply_cautious_update_(Wv, dir_v, lr=shape_lr, wd=group["weight_decay"])

        if collect_stats:
            n = max(1, stats["num_pairs_active"])
            group["last_stats"] = {
                "vo/num_pairs_active": stats["num_pairs_active"],
                "vo/proxy_grad_frob_mean": stats["proxy_grad_frob_sum"] / n,
                "vo/proxy_grad_top_sv_mean": stats["proxy_grad_top_sv_sum"] / n,
                "vo/update_frob_v_mean": stats["update_frob_v_sum"] / n,
                "vo/update_frob_o_mean": stats["update_frob_o_sum"] / n,
                "vo/update_to_weight_ratio_v": stats["update_to_weight_ratio_v_sum"] / n,
                "vo/update_to_weight_ratio_o": stats["update_to_weight_ratio_o_sum"] / n,
            }
        else:
            group["last_stats"] = None

    def _step_ffn_joint_muon(self, group: dict) -> None:
        """
        Lossless stacked joint Muon for FFN matrices. Params are stored flat as
        [Win0, Wout0, Win1, Wout1, ...], and pair_offsets gives those positions.
        Win is c_fc.weight with shape (hidden, model); Wout is c_proj.weight with
        shape (model, hidden).
        """
        params: list[Tensor] = group["params"]
        pair_offsets = group["pair_offsets"]
        collect_stats = group.get("collect_stats", False)
        stats = dict(
            num_pairs_active=0,
            stacked_grad_frob_sum=0.0,
            stacked_grad_top_sv_sum=0.0,
            update_frob_in_sum=0.0,
            update_frob_out_sum=0.0,
            update_to_weight_ratio_in_sum=0.0,
            update_to_weight_ratio_out_sum=0.0,
        )

        for in_idx, out_idx in pair_offsets:
            Win, Wout = params[in_idx], params[out_idx]
            if Win.grad is None or Wout.grad is None:
                continue

            Gin, Gout = Win.grad, Wout.grad
            state = self.state[Win]
            state["step"] = state.get("step", 0) + 1

            H = ffn_stacked_joint_grad(Gin, Gout)
            T = muonize_matrix(
                H,
                state,
                momentum=group["momentum"],
                ns_steps=group["ns_steps"],
                beta2=group["beta2"],
            )
            dir_in, dir_out = split_ffn_stacked_joint_direction(T, Win.shape[0])
            shape_lr = group["lr"]

            if collect_stats:
                update_in = shape_lr * dir_in.float()
                update_out = shape_lr * dir_out.float()
                weight_in_norm = Win.float().norm().clamp_min(1e-12)
                weight_out_norm = Wout.float().norm().clamp_min(1e-12)
                stats["num_pairs_active"] += 1
                stats["stacked_grad_frob_sum"] += float(H.float().norm().item())
                stats["stacked_grad_top_sv_sum"] += matrix_top_singular_value(H)
                stats["update_frob_in_sum"] += float(update_in.norm().item())
                stats["update_frob_out_sum"] += float(update_out.norm().item())
                stats["update_to_weight_ratio_in_sum"] += float((update_in.norm() / weight_in_norm).item())
                stats["update_to_weight_ratio_out_sum"] += float((update_out.norm() / weight_out_norm).item())

            apply_cautious_update_(Wout, dir_out, lr=shape_lr, wd=group["weight_decay"])
            apply_cautious_update_(Win, dir_in, lr=shape_lr, wd=group["weight_decay"])

        if collect_stats:
            n = max(1, stats["num_pairs_active"])
            group["last_stats"] = {
                "ffn/num_pairs_active": stats["num_pairs_active"],
                "ffn/stacked_grad_frob_mean": stats["stacked_grad_frob_sum"] / n,
                "ffn/stacked_grad_top_sv_mean": stats["stacked_grad_top_sv_sum"] / n,
                "ffn/update_frob_in_mean": stats["update_frob_in_sum"] / n,
                "ffn/update_frob_out_mean": stats["update_frob_out_sum"] / n,
                "ffn/update_to_weight_ratio_in": stats["update_to_weight_ratio_in_sum"] / n,
                "ffn/update_to_weight_ratio_out": stats["update_to_weight_ratio_out_sum"] / n,
            }
        else:
            group["last_stats"] = None

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            elif group['kind'] == 'vo_product_muon':
                self._step_vo_product_muon(group)
            elif group['kind'] == 'ffn_joint_muon':
                self._step_ffn_joint_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    See MuonAdamW for the algorithmic details of each optimizer. This class adds
    distributed communication to enable multi-GPU training without PyTorch DDP.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (caller's responsibility).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer, second_momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        unsupported_kinds = {"vo_product_muon", "ffn_joint_muon"}
        unsupported = sorted({group.get("kind") for group in param_groups} & unsupported_kinds)
        if unsupported:
            raise NotImplementedError(f"{', '.join(unsupported)} is only implemented for single-GPU training in v1")
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)
