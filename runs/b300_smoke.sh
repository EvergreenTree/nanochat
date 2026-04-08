#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Error: $name must be set." >&2
        exit 1
    fi
}

run_step() {
    echo
    echo "==> $*"
    "$@"
}

require_env BASE_DIR

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="$BASE_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RUN_NAME="${RUN_NAME:-b300_smoke}"
MODEL_TAG="${MODEL_TAG:-b300_smoke}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
DEPTH="${DEPTH:-24}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
FP8="${FP8:-1}"
FP8_RECIPE="${FP8_RECIPE:-tensorwise}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_TOKENS="${EVAL_TOKENS:-1048576}"
SAVE_EVERY="${SAVE_EVERY:-10}"
WINDOW_PATTERN="${WINDOW_PATTERN:-auto}"
PATTERNED_WINDOW_PATTERN="${PATTERNED_WINDOW_PATTERN:-SSSL}"

if [ ! -f "$BASE_DIR/tokenizer/token_bytes.pt" ]; then
    echo "Error: tokenizer artifacts are missing under $BASE_DIR/tokenizer. Run b300_prestage.sh first." >&2
    exit 1
fi

if [ ! -d "$BASE_DIR/base_data_climbmix" ]; then
    echo "Error: dataset shards are missing under $BASE_DIR/base_data_climbmix. Run b300_prestage.sh first." >&2
    exit 1
fi

if ! ls "$BASE_DIR"/base_data_climbmix/*.parquet >/dev/null 2>&1; then
    echo "Error: no parquet shards were found under $BASE_DIR/base_data_climbmix. Run b300_prestage.sh first." >&2
    exit 1
fi

backend_info="$(python - <<'PY'
import torch
from nanochat.flash_attention import ATTENTION_BACKEND, ATTENTION_BACKEND_REASON, HAS_FA4

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the B300 smoke path.")

major, minor = torch.cuda.get_device_capability()
if major < 10:
    raise SystemExit(f"Expected a Blackwell-class GPU for B300 smoke, found SM {major}.{minor}.")

print(f"{ATTENTION_BACKEND}\t{int(HAS_FA4)}\t{torch.cuda.get_device_name(0)}\t{major}.{minor}\t{ATTENTION_BACKEND_REASON}")
PY
)"
IFS=$'\t' read -r ATTENTION_BACKEND HAS_FA4 GPU_NAME GPU_CAPABILITY ATTENTION_REASON <<<"$backend_info"

if [ "$WINDOW_PATTERN" = "auto" ]; then
    case "$ATTENTION_BACKEND" in
        fa3|fa4) RESOLVED_WINDOW_PATTERN="$PATTERNED_WINDOW_PATTERN" ;;
        *) RESOLVED_WINDOW_PATTERN="L" ;;
    esac
else
    RESOLVED_WINDOW_PATTERN="$WINDOW_PATTERN"
fi

echo "Shared BASE_DIR: $BASE_DIR"
echo "GPU: $GPU_NAME (SM $GPU_CAPABILITY)"
echo "Attention backend: $ATTENTION_BACKEND ($ATTENTION_REASON)"
echo "Window pattern: $WINDOW_PATTERN -> $RESOLVED_WINDOW_PATTERN"

if [ "$HAS_FA4" != "1" ] || [ "$ATTENTION_BACKEND" != "fa4" ]; then
    echo "Error: Flash Attention 4 is required for the B300 smoke run." >&2
    exit 1
fi

TRAIN_ARGS=(
    --device-type=cuda
    --run="$RUN_NAME"
    --model-tag="$MODEL_TAG"
    --depth="$DEPTH"
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --window-pattern="$RESOLVED_WINDOW_PATTERN"
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO"
    --num-iterations="$NUM_ITERATIONS"
    --eval-every="$EVAL_EVERY"
    --eval-tokens="$EVAL_TOKENS"
    --core-metric-every=-1
    --sample-every=-1
    --save-every="$SAVE_EVERY"
)

if [ "$FP8" = "1" ]; then
    TRAIN_ARGS+=(--fp8 --fp8-recipe="$FP8_RECIPE")
fi

run_step torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m scripts.base_train -- "${TRAIN_ARGS[@]}"

echo
echo "B300 smoke complete."
echo "Checkpoint dir: $BASE_DIR/base_checkpoints/$MODEL_TAG"
