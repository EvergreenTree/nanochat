#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log_step() {
    echo
    echo "==> $*"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_expected_shards() {
    local num_train_shards="$1"
    local shard_dir="$BASE_DIR/base_data_climbmix"
    local idx shard_path
    for ((idx = 0; idx < num_train_shards; idx++)); do
        shard_path="$(printf '%s/shard_%05d.parquet' "$shard_dir" "$idx")"
        [ -f "$shard_path" ] || die "Expected train shard is missing after download: $shard_path"
    done
    shard_path="$(printf '%s/shard_%05d.parquet' "$shard_dir" 6542)"
    [ -f "$shard_path" ] || die "Expected validation shard is missing after download: $shard_path"
}

activate_env() {
    # shellcheck disable=SC1090
    source "$ENV_DIR/bin/activate"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv-b300}"
BASE_DIR="${BASE_DIR:-$REPO_ROOT/.b300-setup-cache}"

TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

VALIDATE_SETUP="${VALIDATE_SETUP:-1}"
VALIDATION_DATASET_SHARDS="${VALIDATION_DATASET_SHARDS:-1}"
VALIDATION_DATASET_WORKERS="${VALIDATION_DATASET_WORKERS:-2}"
VALIDATION_TOKENIZER_MAX_CHARS="${VALIDATION_TOKENIZER_MAX_CHARS:-2000000}"
VALIDATION_VOCAB_SIZE="${VALIDATION_VOCAB_SIZE:-4096}"
VALIDATION_MODEL_TAG="${VALIDATION_MODEL_TAG:-b300_setup_smoke}"
VALIDATION_PRECISION_RECIPE="${VALIDATION_PRECISION_RECIPE:-fp8_full}"
VALIDATION_RUN_NAME="${VALIDATION_RUN_NAME:-dummy}"
VALIDATION_NUM_ITERATIONS="${VALIDATION_NUM_ITERATIONS:-2}"
VALIDATION_DEPTH="${VALIDATION_DEPTH:-4}"
VALIDATION_HEAD_DIM="${VALIDATION_HEAD_DIM:-64}"
VALIDATION_MAX_SEQ_LEN="${VALIDATION_MAX_SEQ_LEN:-256}"
VALIDATION_DEVICE_BATCH_SIZE="${VALIDATION_DEVICE_BATCH_SIZE:-4}"
VALIDATION_TOTAL_BATCH_SIZE="${VALIDATION_TOTAL_BATCH_SIZE:-4096}"
VALIDATION_EVAL_TOKENS="${VALIDATION_EVAL_TOKENS:-4096}"
VALIDATION_MASTER_PORT="${VALIDATION_MASTER_PORT:-29600}"

require_cmd "$PYTHON_BIN"

log_step "Creating or updating virtualenv at $ENV_DIR"
"$PYTHON_BIN" -m venv "$ENV_DIR"
activate_env

log_step "Upgrading pip/build tooling"
python -m pip install --upgrade pip setuptools wheel packaging ninja

log_step "Installing PyTorch $TORCH_VERSION for CUDA 12.8"
python -m pip install --upgrade --index-url "$PYTORCH_INDEX_URL" "torch==$TORCH_VERSION"

log_step "Installing runtime dependencies needed by data, tokenizer, train, and eval paths"
python -m pip install --upgrade \
    "datasets>=4.0.0" \
    "psutil>=7.1.0" \
    "rustbpe>=0.1.0" \
    "tiktoken>=0.11.0" \
    "tokenizers>=0.22.0" \
    "wandb>=0.21.3" \
    "requests>=2.32.0" \
    "pyarrow>=21.0.0" \
    "filelock>=3.19.0" \
    "PyYAML>=6.0.2"

log_step "Installing FlashAttention-4 and Transformer Engine"
python -m pip install --upgrade "flash-attn-4"
NVTE_FRAMEWORK=pytorch python -m pip install --upgrade --no-build-isolation "transformer_engine[pytorch]"

log_step "Verifying Python package consistency"
python -m pip check

log_step "Running Blackwell preflight"
python - <<'PY'
import importlib
import json

import torch

required_modules = [
    "requests",
    "pyarrow",
    "datasets",
    "rustbpe",
    "tiktoken",
    "tokenizers",
    "filelock",
    "yaml",
    "wandb",
]
for module_name in required_modules:
    importlib.import_module(module_name)

from flash_attn.cute import flash_attn_func  # noqa: F401
import transformer_engine.pytorch as te  # noqa: F401
from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling  # noqa: F401

from nanochat.flash_attention import ATTENTION_BACKEND, ATTENTION_BACKEND_REASON, HAS_FA4
from nanochat.precision import resolve_precision_backend

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required on the target B300 machine.")

device_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
if major < 10:
    raise SystemExit(f"Expected a Blackwell-class GPU (SM >= 10.0), found {device_name} (SM {major}.{minor}).")

fp8_backend = resolve_precision_backend("fp8_full", device_type="cuda", gpu_name=device_name)
fp4_backend = resolve_precision_backend("fp4_blackwell", device_type="cuda", gpu_name=device_name)

print(json.dumps({
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "device_name": device_name,
    "compute_capability": f"{major}.{minor}",
    "attention_backend": ATTENTION_BACKEND,
    "attention_reason": ATTENTION_BACKEND_REASON,
    "has_fa4": HAS_FA4,
    "fp8_backend": fp8_backend.reason,
    "fp4_backend": fp4_backend.reason,
}, indent=2))
PY

if [ "$VALIDATE_SETUP" = "1" ]; then
    export NANOCHAT_BASE_DIR="$BASE_DIR"
    mkdir -p "$BASE_DIR"

    log_step "Resetting the report cache in $BASE_DIR"
    python -m nanochat.report reset

    log_step "Downloading a tiny validation dataset slice"
    python -m nanochat.dataset -n "$VALIDATION_DATASET_SHARDS" -w "$VALIDATION_DATASET_WORKERS"
    require_expected_shards "$VALIDATION_DATASET_SHARDS"

    log_step "Training a small validation tokenizer"
    python -m scripts.tok_train \
        --max-chars="$VALIDATION_TOKENIZER_MAX_CHARS" \
        --vocab-size="$VALIDATION_VOCAB_SIZE"

    log_step "Running tokenizer evaluation"
    python -m scripts.tok_eval

    log_step "Running a tiny single-GPU FOG smoke train"
    torchrun \
        --nnodes=1 \
        --nproc_per_node=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --master_port="$VALIDATION_MASTER_PORT" \
        -m scripts.base_train -- \
        --device-type=cuda \
        --run="$VALIDATION_RUN_NAME" \
        --model-tag="$VALIDATION_MODEL_TAG" \
        --arch-family=fog \
        --fog-variant=flash \
        --precision-recipe="$VALIDATION_PRECISION_RECIPE" \
        --seed=42 \
        --depth="$VALIDATION_DEPTH" \
        --head-dim="$VALIDATION_HEAD_DIM" \
        --max-seq-len="$VALIDATION_MAX_SEQ_LEN" \
        --window-pattern=L \
        --device-batch-size="$VALIDATION_DEVICE_BATCH_SIZE" \
        --total-batch-size="$VALIDATION_TOTAL_BATCH_SIZE" \
        --num-iterations="$VALIDATION_NUM_ITERATIONS" \
        --eval-every=1 \
        --eval-tokens="$VALIDATION_EVAL_TOKENS" \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=1 \
        --quant-monitor-every=1

    log_step "Running a tiny post-train eval"
    python -m scripts.base_eval \
        --device-type=cuda \
        --model-tag="$VALIDATION_MODEL_TAG" \
        --eval=bpb,sample \
        --device-batch-size=1 \
        --split-tokens="$VALIDATION_EVAL_TOKENS"
fi

echo
echo "Standalone B300 setup complete."
echo "Environment: $ENV_DIR"
echo "Repo root: $REPO_ROOT"
echo "Validation cache: $BASE_DIR"
echo
echo "Next command examples:"
echo "  ENV_DIR=$ENV_DIR BASE_DIR=/shared/nanochat bash runs/b300_remote_run.sh preflight"
echo "  ENV_DIR=$ENV_DIR BASE_DIR=/shared/nanochat DATASET_SHARDS=170 bash runs/b300_remote_run.sh prestage"
echo "  ENV_DIR=$ENV_DIR BASE_DIR=/shared/nanochat RUN_NAME=track1_fp8 NUM_ITERATIONS=120 MODEL_TAG=track1_fp8 bash runs/b300_remote_run.sh timed-arm"
