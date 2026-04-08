import pytest
import torch

from nanochat.model_factory import build_model_config, instantiate_model, patch_model_config_kwargs, strip_backend_extra_state
from nanochat.precision import resolve_precision_backend


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
