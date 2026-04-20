import pytest
import torch

from nanochat.checkpoint_manager import _patch_missing_config_keys
from nanochat.gpt import GPT, GPTConfig, MLP
from nanochat.optim import MuonAdamW, ffn_stacked_joint_grad, split_ffn_stacked_joint_direction


def _make_ffn_optimizer(Win, Wout, collect_stats=False):
    return MuonAdamW([
        dict(
            kind="ffn_joint_muon",
            params=[Win, Wout],
            pair_offsets=[(0, 1)],
            lr=0.01,
            momentum=0.95,
            ns_steps=2,
            beta2=0.9,
            weight_decay=0.01,
            collect_stats=collect_stats,
        )
    ])


def _set_random_grads(*params):
    for p in params:
        p.grad = torch.randn_like(p)


def test_ffn_stacked_joint_grad_is_lossless():
    torch.manual_seed(123)
    Gin = torch.randn(12, 4)
    Gout = torch.randn(4, 12)

    H = ffn_stacked_joint_grad(Gin, Gout)
    dir_in, dir_out = split_ffn_stacked_joint_direction(H, Gin.shape[0])

    assert H.shape == (24, 4)
    assert torch.equal(dir_in, Gin)
    assert torch.equal(dir_out, Gout)
    assert torch.isfinite(H).all()


def test_ffn_pair_one_step_shapes_and_finite():
    torch.manual_seed(234)
    Win = torch.nn.Parameter(torch.randn(12, 4))
    Wout = torch.nn.Parameter(torch.randn(4, 12))
    _set_random_grads(Win, Wout)
    opt = _make_ffn_optimizer(Win, Wout, collect_stats=True)

    opt.step()

    assert Win.shape == (12, 4)
    assert Wout.shape == (4, 12)
    assert torch.isfinite(Win).all()
    assert torch.isfinite(Wout).all()
    stats = opt.param_groups[0]["last_stats"]
    assert stats["ffn/num_pairs_active"] == 1
    assert "ffn/stacked_grad_frob_mean" in stats


def test_ffn_pair_state_dict_roundtrip_runs_another_step():
    torch.manual_seed(345)
    Win = torch.nn.Parameter(torch.randn(10, 5))
    Wout = torch.nn.Parameter(torch.randn(5, 10))
    opt = _make_ffn_optimizer(Win, Wout)

    _set_random_grads(Win, Wout)
    opt.step()
    state_dict = opt.state_dict()

    opt2 = _make_ffn_optimizer(Win, Wout)
    opt2.load_state_dict(state_dict)
    _set_random_grads(Win, Wout)
    opt2.step()

    assert torch.isfinite(Win).all()
    assert torch.isfinite(Wout).all()


def test_paired_ffn_groups_one_pair_per_block():
    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    opt = model.setup_optimizer(optimizer_kind="paired_ffn")

    ffn_groups = [group for group in opt.param_groups if group["kind"] == "ffn_joint_muon"]
    assert len(ffn_groups) == 1
    assert len(ffn_groups[0]["pair_offsets"]) == model.config.n_layer

    ffn_ids = {id(p) for p in ffn_groups[0]["params"]}
    muon_ids = {id(p) for group in opt.param_groups if group["kind"] == "muon" for p in group["params"]}
    assert not any(group["kind"] == "vo_product_muon" for group in opt.param_groups)
    for block in model.transformer.h:
        assert id(block.mlp.c_fc.weight) in ffn_ids
        assert id(block.mlp.c_proj.weight) in ffn_ids
        assert id(block.mlp.c_fc.weight) not in muon_ids
        assert id(block.mlp.c_proj.weight) not in muon_ids
        assert id(block.attn.c_q.weight) in muon_ids
        assert id(block.attn.c_proj.weight) in muon_ids


def test_mismatched_ffn_shapes_fall_back_to_muon():
    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    model.transformer.h[0].mlp.c_proj.weight = torch.nn.Parameter(torch.randn(17, 64))
    opt = model.setup_optimizer(optimizer_kind="paired_ffn")

    ffn_groups = [group for group in opt.param_groups if group["kind"] == "ffn_joint_muon"]
    assert len(ffn_groups) == 1
    assert len(ffn_groups[0]["pair_offsets"]) == 1
    muon_ids = {id(p) for group in opt.param_groups if group["kind"] == "muon" for p in group["params"]}
    assert id(model.transformer.h[0].mlp.c_fc.weight) in muon_ids
    assert id(model.transformer.h[0].mlp.c_proj.weight) in muon_ids


def test_paired_ffn_ddp_is_rejected(monkeypatch):
    import nanochat.gpt as gpt_module

    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    monkeypatch.setattr(gpt_module, "get_dist_info", lambda: (True, 0, 0, 2))

    with pytest.raises(NotImplementedError, match="single-GPU"):
        model.setup_optimizer(optimizer_kind="paired_ffn")


def test_mlp_activation_relu_and_relu_squared_differ():
    relu_mlp = MLP(GPTConfig(n_embd=2, n_head=1, n_kv_head=1, mlp_activation="relu"))
    squared_mlp = MLP(GPTConfig(n_embd=2, n_head=1, n_kv_head=1, mlp_activation="relu_squared"))
    for mlp in (relu_mlp, squared_mlp):
        with torch.no_grad():
            mlp.c_fc.weight.zero_()
            mlp.c_proj.weight.zero_()
            mlp.c_fc.weight[0, 0] = 1
            mlp.c_fc.weight[1, 1] = 1
            mlp.c_proj.weight[0, 0] = 1
            mlp.c_proj.weight[1, 1] = 1

    x = torch.tensor([[[2.0, 3.0]]])

    assert torch.allclose(relu_mlp(x), torch.tensor([[[2.0, 3.0]]]))
    assert torch.allclose(squared_mlp(x), torch.tensor([[[4.0, 9.0]]]))


def test_old_checkpoint_config_patches_mlp_activation_to_relu_squared():
    config = {
        "sequence_len": 16,
        "vocab_size": 128,
        "n_layer": 2,
        "n_head": 2,
        "n_kv_head": 2,
        "n_embd": 16,
        "window_pattern": "L",
    }

    _patch_missing_config_keys(config)

    assert config["mlp_activation"] == "relu_squared"


def test_mlp_c_proj_gaussian_init_and_zero_init():
    torch.manual_seed(456)
    model = GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=16))
    model.init_weights(mlp_c_proj_init="gaussian")
    assert all(torch.count_nonzero(block.mlp.c_proj.weight).item() > 0 for block in model.transformer.h)

    model.init_weights(mlp_c_proj_init="zero")
    assert all(torch.count_nonzero(block.mlp.c_proj.weight).item() == 0 for block in model.transformer.h)


def test_invalid_mlp_activation_is_rejected():
    with pytest.raises(ValueError, match="Unknown MLP activation"):
        MLP(GPTConfig(n_embd=2, n_head=1, n_kv_head=1, mlp_activation="gelu"))
