import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import MuonAdamW


def _make_vo_optimizer(Wv, Wo, collect_stats=False):
    return MuonAdamW([
        dict(
            kind="vo_product_muon",
            params=[Wv, Wo],
            pair_offsets=[(0, 1)],
            lr=0.01,
            momentum=0.95,
            ns_steps=2,
            beta2=0.9,
            weight_decay=0.01,
            eps=1e-8,
            collect_stats=collect_stats,
        )
    ])


def _set_random_grads(*params):
    for p in params:
        p.grad = torch.randn_like(p)


def _set_vo_covariances(Wv, n_head=2, count=5):
    model_dim = Wv.shape[1]
    assert Wv.shape[0] % n_head == 0
    head_dim = Wv.shape[0] // n_head
    Wv._vo_cov_x_sum = torch.eye(model_dim, dtype=torch.float32, device=Wv.device) * count
    Wv._vo_cov_h_sum = torch.eye(head_dim, dtype=torch.float32, device=Wv.device).expand(n_head, head_dim, head_dim).clone() * count
    Wv._vo_cov_count = count


def test_vo_pair_one_step_shapes_and_finite():
    torch.manual_seed(123)
    Wv = torch.nn.Parameter(torch.randn(8, 8))
    Wo = torch.nn.Parameter(torch.randn(8, 8))
    _set_random_grads(Wv, Wo)
    _set_vo_covariances(Wv)
    opt = _make_vo_optimizer(Wv, Wo, collect_stats=True)

    opt.step()

    assert Wv.shape == (8, 8)
    assert Wo.shape == (8, 8)
    assert torch.isfinite(Wv).all()
    assert torch.isfinite(Wo).all()
    assert opt.param_groups[0]["last_stats"]["vo/num_pairs_active"] == 1


def test_vo_pair_state_dict_roundtrip_runs_another_step():
    torch.manual_seed(456)
    Wv = torch.nn.Parameter(torch.randn(6, 6))
    Wo = torch.nn.Parameter(torch.randn(6, 6))
    opt = _make_vo_optimizer(Wv, Wo)

    _set_random_grads(Wv, Wo)
    _set_vo_covariances(Wv)
    opt.step()
    state_dict = opt.state_dict()

    opt2 = _make_vo_optimizer(Wv, Wo)
    opt2.load_state_dict(state_dict)
    _set_random_grads(Wv, Wo)
    _set_vo_covariances(Wv)
    opt2.step()

    assert torch.isfinite(Wv).all()
    assert torch.isfinite(Wo).all()


def test_vo_pair_requires_feature_covariances():
    torch.manual_seed(654)
    Wv = torch.nn.Parameter(torch.randn(8, 8))
    Wo = torch.nn.Parameter(torch.randn(8, 8))
    _set_random_grads(Wv, Wo)
    opt = _make_vo_optimizer(Wv, Wo)

    with pytest.raises(RuntimeError, match="feature covariances"):
        opt.step()


def test_gqa_mismatched_vo_shapes_fall_back_to_muon():
    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=4, n_kv_head=2, n_embd=16))
    opt = model.setup_optimizer(optimizer_kind="vo_product_muon")

    assert not any(group["kind"] == "vo_product_muon" for group in opt.param_groups)
    muon_ids = {id(p) for group in opt.param_groups if group["kind"] == "muon" for p in group["params"]}
    for block in model.transformer.h:
        assert id(block.attn.c_v.weight) in muon_ids
        assert id(block.attn.c_proj.weight) in muon_ids


def test_vo_product_muon_groups_square_attention_vo_pairs():
    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    opt = model.setup_optimizer(optimizer_kind="vo_product_muon")

    vo_groups = [group for group in opt.param_groups if group["kind"] == "vo_product_muon"]
    assert len(vo_groups) == 1
    assert len(vo_groups[0]["pair_offsets"]) == model.config.n_layer

    vo_ids = {id(p) for p in vo_groups[0]["params"]}
    muon_ids = {id(p) for group in opt.param_groups if group["kind"] == "muon" for p in group["params"]}
    for block in model.transformer.h:
        assert id(block.attn.c_v.weight) in vo_ids
        assert id(block.attn.c_proj.weight) in vo_ids
        assert id(block.attn.c_v.weight) not in muon_ids
        assert id(block.attn.c_proj.weight) not in muon_ids


def test_attention_c_proj_gaussian_init_is_balanced_with_value_projection():
    torch.manual_seed(789)
    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    model.init_weights(attn_c_proj_init="gaussian")

    for block in model.transformer.h:
        wv = block.attn.c_v.weight
        wo = block.attn.c_proj.weight
        denom_v = wv.float().square().sum() / wv.shape[-1]
        denom_o = wo.float().square().sum() / wo.shape[-1]
        assert torch.count_nonzero(wo).item() > 0
        assert 0.25 <= float(denom_o / denom_v) <= 4.0


def test_attention_c_proj_zero_init_remains_available():
    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    model.init_weights(attn_c_proj_init="zero")

    for block in model.transformer.h:
        assert torch.count_nonzero(block.attn.c_proj.weight).item() == 0


def test_default_muon_grouping_covers_legacy_transformer_params():
    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    opt = model.setup_optimizer(optimizer_kind="muon")

    assert not any(group["kind"] == "vo_product_muon" for group in opt.param_groups)
    assert not any(group["kind"] == "ffn_joint_muon" for group in opt.param_groups)
    assert [group["kind"] for group in opt.param_groups[:6]] == ["adamw"] * 6
    muon_params = [p for group in opt.param_groups if group["kind"] == "muon" for p in group["params"]]
    assert {id(p) for p in muon_params} == {id(p) for p in model.transformer.h.parameters()}


def test_adamw_all_routes_transformer_matrices_to_matrix_adamw():
    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    opt = model.setup_optimizer(optimizer_kind="adamw_all")

    assert not any(group["kind"] == "muon" for group in opt.param_groups)
    assert not any(group["kind"] == "vo_product_muon" for group in opt.param_groups)
    assert not any(group["kind"] == "ffn_joint_muon" for group in opt.param_groups)
    matrix_groups = [group for group in opt.param_groups if group.get("matrix_group", False)]
    matrix_params = [p for group in matrix_groups for p in group["params"]]
    assert {id(p) for p in matrix_params} == {id(p) for p in model.transformer.h.parameters()}
    assert all(group["betas"] == (0.9, 0.95) for group in matrix_groups)


def test_vo_product_muon_ddp_is_rejected(monkeypatch):
    import nanochat.gpt as gpt_module

    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    monkeypatch.setattr(gpt_module, "get_dist_info", lambda: (True, 0, 0, 2))

    with pytest.raises(NotImplementedError, match="single-GPU"):
        model.setup_optimizer(optimizer_kind="vo_product_muon")


# ---- Paired FFN optimizer tests ----

def _make_paired_ffn_optimizer(Win, Wout, alpha_max=0.2, warmup_steps=0, collect_stats=False):
    return MuonAdamW([
        dict(
            kind="paired_ffn",
            params=[Win, Wout],
            pair_offsets=[(0, 1)],
            lr=0.01,
            momentum=0.95,
            ns_steps=2,
            beta2=0.9,
            weight_decay=0.01,
            ffn_pair_alpha_max=alpha_max,
            ffn_pair_warmup_steps=warmup_steps,
            ffn_pair_eps=1e-8,
            ffn_pair_g_floor=1e-3,
            collect_stats=collect_stats,
        )
    ])


def _set_ffn_pair_gate(Win, hidden_dim, activation_kind="relu"):
    """Set a fake FFN gate sum/count on Win for testing."""
    if activation_kind == "relu":
        gate = (torch.rand(hidden_dim) > 0.3).float()
    elif activation_kind == "relu2":
        gate = 2.0 * torch.relu(torch.randn(hidden_dim))
    else:
        gate = torch.sigmoid(torch.randn(hidden_dim)) * (torch.rand(hidden_dim) > 0.3).float()
    gate = gate.clamp_min(1e-3)
    Win._ffn_pair_gate_sum = gate * 100  # simulate 100 samples
    Win._ffn_pair_gate_count = 100


def test_paired_ffn_one_step_shapes_and_finite():
    torch.manual_seed(100)
    Win = torch.nn.Parameter(torch.randn(32, 8))   # hidden=32, model=8
    Wout = torch.nn.Parameter(torch.randn(8, 32))   # model=8, hidden=32
    _set_random_grads(Win, Wout)
    _set_ffn_pair_gate(Win, 32)
    opt = _make_paired_ffn_optimizer(Win, Wout, warmup_steps=0, collect_stats=True)

    opt.step()

    assert Win.shape == (32, 8)
    assert Wout.shape == (8, 32)
    assert torch.isfinite(Win).all()
    assert torch.isfinite(Wout).all()
    assert opt.param_groups[0]["last_stats"]["ffn/num_pairs_active"] == 1


def test_paired_ffn_relu2_gate():
    torch.manual_seed(101)
    Win = torch.nn.Parameter(torch.randn(16, 8))
    Wout = torch.nn.Parameter(torch.randn(8, 16))
    _set_random_grads(Win, Wout)
    _set_ffn_pair_gate(Win, 16, activation_kind="relu2")
    opt = _make_paired_ffn_optimizer(Win, Wout, warmup_steps=0)

    opt.step()

    assert torch.isfinite(Win).all()
    assert torch.isfinite(Wout).all()


def test_paired_ffn_warmup_uses_plain_muon():
    """Before warmup_steps, the paired correction should not be used."""
    torch.manual_seed(102)
    Win = torch.nn.Parameter(torch.randn(16, 8))
    Wout = torch.nn.Parameter(torch.randn(8, 16))

    # Step 1: warmup_steps=10, so step 1 is before warmup => pure Muon
    _set_random_grads(Win, Wout)
    # Set gate but it should be discarded during warmup
    _set_ffn_pair_gate(Win, 16)
    opt = _make_paired_ffn_optimizer(Win, Wout, warmup_steps=10)

    opt.step()
    assert torch.isfinite(Win).all()
    assert torch.isfinite(Wout).all()
    # Gate should have been cleaned up even during warmup
    assert not hasattr(Win, "_ffn_pair_gate_sum")
    assert not hasattr(Win, "_ffn_pair_gate_count")


def test_paired_ffn_grouping_covers_ffn_matrices():
    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    opt = model.setup_optimizer(optimizer_kind="paired_ffn")

    paired_groups = [group for group in opt.param_groups if group["kind"] == "paired_ffn"]
    assert len(paired_groups) == 1
    assert len(paired_groups[0]["pair_offsets"]) == model.config.n_layer

    paired_ids = {id(p) for p in paired_groups[0]["params"]}
    muon_ids = {id(p) for group in opt.param_groups if group["kind"] == "muon" for p in group["params"]}
    for block in model.transformer.h:
        assert id(block.mlp.c_fc.weight) in paired_ids
        assert id(block.mlp.c_proj.weight) in paired_ids
        assert id(block.mlp.c_fc.weight) not in muon_ids
        assert id(block.mlp.c_proj.weight) not in muon_ids
        # Verify gate recording was enabled
        assert block.mlp.record_ffn_pair_gate is True


def test_paired_ffn_requires_gate_mean():
    torch.manual_seed(103)
    Win = torch.nn.Parameter(torch.randn(16, 8))
    Wout = torch.nn.Parameter(torch.randn(8, 16))
    _set_random_grads(Win, Wout)
    # Don't set gate mean — should raise after warmup
    opt = _make_paired_ffn_optimizer(Win, Wout, warmup_steps=0)

    with pytest.raises(RuntimeError, match="ffn_pair_gate_mean"):
        opt.step()

