#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage:
  bash runs/b300_remote_run.sh <mode>

Modes:
  preflight   Verify the remote environment, FA4, and TE precision backends
  prestage    Download dataset shards and train the full tokenizer into BASE_DIR
  smoke       Run a short 1-GPU FOG smoke train on the target machine
  timed-arm   Run one scored FOG training arm, tuned for a 10-minute 4-node window
  compare     Run the matched bf16 / fp8_full / fp4_blackwell comparison harness
  eval        Run scripts.base_eval on a saved checkpoint

Common variables:
  ENV_DIR     Virtualenv created by runs/b300_remote_setup.sh
  BASE_DIR    Shared artifact directory used for dataset, tokenizer, and checkpoints
EOF
}

log_step() {
    echo
    echo "==> $*"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        die "$name must be set."
    fi
}

activate_env() {
    ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv-b300}"
    [ -x "$ENV_DIR/bin/python" ] || die "Expected virtualenv at $ENV_DIR. Run runs/b300_remote_setup.sh first."
    # shellcheck disable=SC1090
    source "$ENV_DIR/bin/activate"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    if [ -n "${BASE_DIR:-}" ]; then
        export NANOCHAT_BASE_DIR="$BASE_DIR"
    fi
}

run_preflight() {
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
}

require_prestaged_artifacts() {
    require_env BASE_DIR
    [ -f "$BASE_DIR/tokenizer/token_bytes.pt" ] || die "Tokenizer artifacts are missing under $BASE_DIR/tokenizer. Run prestage first."
    ls "$BASE_DIR"/base_data_climbmix/*.parquet >/dev/null 2>&1 || die "Dataset shards are missing under $BASE_DIR/base_data_climbmix. Run prestage first."
}

resolve_num_iterations_for_timed_run() {
    if [ -n "${NUM_ITERATIONS:-}" ]; then
        return
    fi
    TARGET_MINUTES="${TARGET_MINUTES:-10}"
    SAFETY_SECONDS="${SAFETY_SECONDS:-45}"
    STEP_TIME_S="${STEP_TIME_S:-}"
    [ -n "$STEP_TIME_S" ] || die "Set NUM_ITERATIONS directly, or provide STEP_TIME_S together with TARGET_MINUTES/SAFETY_SECONDS."
    export TARGET_MINUTES SAFETY_SECONDS STEP_TIME_S
    NUM_ITERATIONS="$(python - <<'PY'
import math
import os

target_seconds = float(os.environ["TARGET_MINUTES"]) * 60.0
safety_seconds = float(os.environ["SAFETY_SECONDS"])
step_time_s = float(os.environ["STEP_TIME_S"])
effective_seconds = target_seconds - safety_seconds
if effective_seconds <= 0:
    raise SystemExit("SAFETY_SECONDS must be smaller than TARGET_MINUTES * 60.")
if step_time_s <= 0:
    raise SystemExit("STEP_TIME_S must be positive.")
print(max(1, math.floor(effective_seconds / step_time_s)))
PY
)"
    export NUM_ITERATIONS
    echo "Resolved NUM_ITERATIONS=$NUM_ITERATIONS from STEP_TIME_S=$STEP_TIME_S, TARGET_MINUTES=$TARGET_MINUTES, SAFETY_SECONDS=$SAFETY_SECONDS"
}

run_fog_arm() {
    require_env BASE_DIR
    require_env RUN_NAME
    require_env MODEL_TAG
    require_env NUM_ITERATIONS
    require_env PRECISION_RECIPE
    require_prestaged_artifacts

    export NANOCHAT_BASE_DIR="$BASE_DIR"

    NNODES="${NNODES:-1}"
    NODE_RANK="${NODE_RANK:-0}"
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-29500}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

    FOG_VARIANT="${FOG_VARIANT:-flash}"
    SEED="${SEED:-42}"
    DEPTH="${DEPTH:-24}"
    HEAD_DIM="${HEAD_DIM:-64}"
    MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
    WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
    DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
    TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
    EVAL_EVERY="${EVAL_EVERY:--1}"
    EVAL_TOKENS="${EVAL_TOKENS:-262144}"
    SAVE_EVERY="${SAVE_EVERY:--1}"
    QUANT_MONITOR_EVERY="${QUANT_MONITOR_EVERY:--1}"
    STOCHASTIC_ROUNDING="${STOCHASTIC_ROUNDING:-auto}"
    SPLIT_ACCUMULATOR="${SPLIT_ACCUMULATOR:-auto}"
    EVAL_AFTER_RUN="${EVAL_AFTER_RUN:-0}"
    EVAL_MODES="${EVAL_MODES:-bpb,sample}"
    EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-4}"
    EVAL_SPLIT_TOKENS="${EVAL_SPLIT_TOKENS:-262144}"

    local train_args=(
        --device-type=cuda
        --run="$RUN_NAME"
        --model-tag="$MODEL_TAG"
        --arch-family=fog
        --fog-variant="$FOG_VARIANT"
        --precision-recipe="$PRECISION_RECIPE"
        --seed="$SEED"
        --depth="$DEPTH"
        --head-dim="$HEAD_DIM"
        --max-seq-len="$MAX_SEQ_LEN"
        --window-pattern="$WINDOW_PATTERN"
        --device-batch-size="$DEVICE_BATCH_SIZE"
        --total-batch-size="$TOTAL_BATCH_SIZE"
        --num-iterations="$NUM_ITERATIONS"
        --eval-every="$EVAL_EVERY"
        --eval-tokens="$EVAL_TOKENS"
        --core-metric-every=-1
        --sample-every=-1
        --save-every="$SAVE_EVERY"
        --quant-monitor-every="$QUANT_MONITOR_EVERY"
    )
    if [ "$STOCHASTIC_ROUNDING" != "auto" ]; then
        train_args+=(--stochastic-rounding="$STOCHASTIC_ROUNDING")
    fi
    if [ "$SPLIT_ACCUMULATOR" != "auto" ]; then
        train_args+=(--split-accumulator="$SPLIT_ACCUMULATOR")
    fi

    log_step "Launching FOG arm: recipe=$PRECISION_RECIPE run=$RUN_NAME model_tag=$MODEL_TAG"
    torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.base_train -- \
        "${train_args[@]}"

    if [ "$EVAL_AFTER_RUN" = "1" ] && [ "$NODE_RANK" = "0" ]; then
        log_step "Running post-train eval"
        python -m scripts.base_eval \
            --device-type=cuda \
            --model-tag="$MODEL_TAG" \
            --eval="$EVAL_MODES" \
            --device-batch-size="$EVAL_DEVICE_BATCH_SIZE" \
            --split-tokens="$EVAL_SPLIT_TOKENS"
    fi
}

MODE="${1:-help}"

case "$MODE" in
    help|-h|--help)
        usage
        ;;
    preflight)
        activate_env
        run_preflight
        ;;
    prestage)
        activate_env
        require_env BASE_DIR
        export BASE_DIR
        export DATASET_SHARDS="${DATASET_SHARDS:-170}"
        export TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
        export VOCAB_SIZE="${VOCAB_SIZE:-32768}"
        log_step "Running full prestage into $BASE_DIR"
        bash "$REPO_ROOT/runs/b300_prestage.sh"
        ;;
    smoke)
        activate_env
        require_env BASE_DIR
        export RUN_NAME="${RUN_NAME:-b300_smoke_fp8}"
        export MODEL_TAG="${MODEL_TAG:-b300_smoke_fp8}"
        export PRECISION_RECIPE="${PRECISION_RECIPE:-fp8_full}"
        export NNODES="${NNODES:-1}"
        export NODE_RANK="${NODE_RANK:-0}"
        export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        export MASTER_PORT="${MASTER_PORT:-29610}"
        export NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
        export DEPTH="${DEPTH:-6}"
        export HEAD_DIM="${HEAD_DIM:-64}"
        export MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
        export WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
        export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
        export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-8192}"
        export EVAL_EVERY="${EVAL_EVERY:-5}"
        export EVAL_TOKENS="${EVAL_TOKENS:-8192}"
        export SAVE_EVERY="${SAVE_EVERY:-5}"
        export QUANT_MONITOR_EVERY="${QUANT_MONITOR_EVERY:-1}"
        export EVAL_AFTER_RUN="${EVAL_AFTER_RUN:-1}"
        export EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-1}"
        export EVAL_SPLIT_TOKENS="${EVAL_SPLIT_TOKENS:-8192}"
        run_fog_arm
        ;;
    timed-arm|submit)
        activate_env
        require_env BASE_DIR
        export RUN_NAME="${RUN_NAME:-track1_fp8_10min}"
        export MODEL_TAG="${MODEL_TAG:-track1_fp8_10min}"
        export PRECISION_RECIPE="${PRECISION_RECIPE:-fp8_full}"
        export NNODES="${NNODES:-4}"
        export NODE_RANK="${NODE_RANK:-0}"
        export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        export MASTER_PORT="${MASTER_PORT:-29500}"
        export DEPTH="${DEPTH:-24}"
        export HEAD_DIM="${HEAD_DIM:-64}"
        export MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
        export WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
        export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
        export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
        export EVAL_EVERY="${EVAL_EVERY:--1}"
        export SAVE_EVERY="${SAVE_EVERY:--1}"
        export QUANT_MONITOR_EVERY="${QUANT_MONITOR_EVERY:--1}"
        export EVAL_AFTER_RUN="${EVAL_AFTER_RUN:-0}"
        resolve_num_iterations_for_timed_run
        run_fog_arm
        ;;
    compare)
        activate_env
        require_env BASE_DIR
        require_env RUN_NAME
        require_env NUM_ITERATIONS
        export BASE_DIR RUN_NAME NUM_ITERATIONS
        export NNODES="${NNODES:-1}"
        export NODE_RANK="${NODE_RANK:-0}"
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        export MASTER_PORT="${MASTER_PORT:-29500}"
        export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
        log_step "Launching matched bf16/fp8_full/fp4_blackwell comparison"
        bash "$REPO_ROOT/runs/fog_compare_b300.sh"
        ;;
    eval)
        activate_env
        require_env BASE_DIR
        require_env MODEL_TAG
        export NANOCHAT_BASE_DIR="$BASE_DIR"
        python -m scripts.base_eval \
            --device-type=cuda \
            --model-tag="$MODEL_TAG" \
            --eval="${EVAL_MODES:-bpb,sample}" \
            --device-batch-size="${EVAL_DEVICE_BATCH_SIZE:-4}" \
            --split-tokens="${EVAL_SPLIT_TOKENS:-262144}"
        ;;
    *)
        usage
        exit 1
        ;;
esac
