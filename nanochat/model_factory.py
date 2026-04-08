"""Model-family construction helpers."""

from __future__ import annotations

from dataclasses import asdict

import torch

from nanochat.fog import FOG, FOGConfig
from nanochat.gpt import GPT, GPTConfig


def infer_model_dims(depth: int, aspect_ratio: int, head_dim: int) -> tuple[int, int]:
    """Infer model width/head count from the repo's depth-driven scaling rule."""
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return model_dim, num_heads


def patch_model_config_kwargs(model_config_kwargs: dict) -> dict:
    """Patch missing config keys from older checkpoints and normalize families."""
    arch_family = model_config_kwargs.get("arch_family", "nanochat")
    patched = dict(model_config_kwargs)
    patched["arch_family"] = arch_family
    if "window_pattern" not in patched:
        patched["window_pattern"] = "L"
    if arch_family == "fog":
        patched.setdefault("fog_variant", "flash")
    return patched


def build_model_config(
    *,
    arch_family: str,
    depth: int,
    aspect_ratio: int,
    head_dim: int,
    max_seq_len: int,
    vocab_size: int,
    window_pattern: str,
    fog_variant: str,
):
    model_dim, num_heads = infer_model_dims(depth, aspect_ratio, head_dim)
    common_kwargs = dict(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    if arch_family == "nanochat":
        return GPTConfig(**common_kwargs, arch_family="nanochat")
    if arch_family == "fog":
        return FOGConfig(**common_kwargs, arch_family="fog", fog_variant=fog_variant)
    raise ValueError(f"Unsupported arch_family: {arch_family}")


def instantiate_model(model_config, *, runtime_backend: str = "native"):
    """Build a model instance from a config dataclass."""
    arch_family = getattr(model_config, "arch_family", "nanochat")
    if arch_family == "nanochat":
        if runtime_backend != "native":
            raise ValueError("nanochat GPT only supports the native runtime backend")
        return GPT(model_config)
    if arch_family == "fog":
        return FOG(model_config, runtime_backend=runtime_backend)
    raise ValueError(f"Unsupported arch_family: {arch_family}")


def build_model_from_config_kwargs(model_config_kwargs: dict, *, runtime_backend: str = "native"):
    patched = patch_model_config_kwargs(model_config_kwargs)
    arch_family = patched["arch_family"]
    if arch_family == "nanochat":
        config = GPTConfig(**patched)
    elif arch_family == "fog":
        config = FOGConfig(**patched)
    else:
        raise ValueError(f"Unsupported arch_family: {arch_family}")
    model = instantiate_model(config, runtime_backend=runtime_backend)
    return model, config


def model_config_to_dict(model_config) -> dict:
    return asdict(model_config)


def strip_backend_extra_state(model_state_dict: dict) -> dict:
    """Drop backend-only state that native eval models do not need."""
    return {
        k: v
        for k, v in model_state_dict.items()
        if not k.endswith("._extra_state")
    }


def patch_missing_model_state(model_data: dict, model_config) -> dict:
    """Patch old checkpoint parameter sets for backwards compatibility."""
    patched = dict(model_data)
    arch_family = getattr(model_config, "arch_family", "nanochat")
    if arch_family == "nanochat":
        n_layer = model_config.n_layer
        if "resid_lambdas" not in patched:
            patched["resid_lambdas"] = torch.ones(n_layer)
        if "x0_lambdas" not in patched:
            patched["x0_lambdas"] = torch.zeros(n_layer)
    return patched
