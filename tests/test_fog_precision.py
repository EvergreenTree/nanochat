import pytest
import torch

from nanochat.fog import FOGConfig, FogSelfAttention
from nanochat.model_factory import build_model_config, instantiate_model, patch_model_config_kwargs, strip_backend_extra_state
from nanochat import precision as precision_mod
from nanochat.precision import (
    is_full_context_window_pattern,
    precision_recipe_requires_full_context_window,
    resolve_precision_backend,
)


def test_build_fog_model_config_and_forward():
    config = build_model_config(
        arch_family="fog",
        depth=2,
        aspect_ratio=32,
        head_dim=16,
        max_seq_len=32,
        vocab_size=128,
        window_pattern="L",
        fog_variant="flash",
    )
    model = instantiate_model(config, runtime_backend="native")
    model.init_weights()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(idx, targets)
    assert loss.ndim == 0
    assert config.arch_family == "fog"
    assert config.fog_variant == "flash"


def test_patch_model_config_kwargs_defaults_legacy_nanochat():
    patched = patch_model_config_kwargs({
        "sequence_len": 32,
        "vocab_size": 128,
        "n_layer": 2,
        "n_head": 2,
        "n_kv_head": 2,
        "n_embd": 32,
    })
    assert patched["arch_family"] == "nanochat"
    assert patched["window_pattern"] == "L"


def test_strip_backend_extra_state_removes_te_metadata():
    state = {
        "transformer.h.0.attn.c_q.weight": torch.ones(2, 2),
        "transformer.h.0.attn.c_q._extra_state": {"amax_history": torch.ones(4)},
    }
    stripped = strip_backend_extra_state(state)
    assert "transformer.h.0.attn.c_q.weight" in stripped
    assert "transformer.h.0.attn.c_q._extra_state" not in stripped


def test_fp4_blackwell_rejects_non_blackwell_runtime():
    with pytest.raises(RuntimeError, match="requires Blackwell hardware"):
        resolve_precision_backend("fp4_blackwell", device_type="cuda", gpu_name="NVIDIA H100 PCIe")


def test_fp8_full_rejects_non_blackwell_runtime():
    with pytest.raises(RuntimeError, match="reserved for Blackwell MXFP8"):
        resolve_precision_backend("fp8_full", device_type="cuda", gpu_name="NVIDIA H100 PCIe")


def test_fp8_full_rejects_manual_stochastic_rounding_override():
    with pytest.raises(RuntimeError, match="does not expose a public stochastic-rounding control"):
        resolve_precision_backend(
            "fp8_full",
            device_type="cuda",
            gpu_name="NVIDIA B300 SXM",
            stochastic_rounding="on",
        )


def test_full_context_window_pattern_helper():
    assert is_full_context_window_pattern("L")
    assert is_full_context_window_pattern("ll")
    assert not is_full_context_window_pattern("LS")
    assert not is_full_context_window_pattern("")
    assert precision_recipe_requires_full_context_window("fp8_full")
    assert precision_recipe_requires_full_context_window("fp4_blackwell")
    assert not precision_recipe_requires_full_context_window("bf16")


def test_fp4_blackwell_stochastic_rounding_toggle(monkeypatch):
    class FakeNVFP4BlockScaling:
        def __init__(self, *, disable_rht=False, disable_stochastic_rounding=False, disable_2d_quantization=False):
            self.disable_rht = disable_rht
            self.disable_stochastic_rounding = disable_stochastic_rounding
            self.disable_2d_quantization = disable_2d_quantization

    class FakeRecipeModule:
        NVFP4BlockScaling = FakeNVFP4BlockScaling

    class FakeTEModule:
        @staticmethod
        def is_nvfp4_available(return_reason=False):
            return (True, "") if return_reason else True

    monkeypatch.setattr(precision_mod, "_import_transformer_engine", lambda: (FakeTEModule, FakeRecipeModule))

    enabled = resolve_precision_backend(
        "fp4_blackwell",
        device_type="cuda",
        gpu_name="NVIDIA B300 SXM",
        stochastic_rounding="on",
    )
    assert enabled.stochastic_rounding == "enabled"
    assert enabled.te_recipe.disable_stochastic_rounding is False

    disabled = resolve_precision_backend(
        "fp4_blackwell",
        device_type="cuda",
        gpu_name="NVIDIA B300 SXM",
        stochastic_rounding="off",
    )
    assert disabled.stochastic_rounding == "disabled"
    assert disabled.te_recipe.disable_stochastic_rounding is True


def test_te_attention_training_path_rejects_sliding_window():
    config = FOGConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, window_pattern="S")
    attn = FogSelfAttention(config, layer_idx=0, runtime_backend="native")
    attn.te_attention = object()
    x = torch.randn(2, 8, config.n_embd)
    head_dim = config.n_embd // config.n_head
    freqs = torch.randn(1, x.size(1), 1, head_dim // 2)
    with pytest.raises(RuntimeError, match="requires full-context attention"):
        attn(x, (freqs, freqs), (4, 0), None)
