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

for required_var in BASE_DIR NNODES NODE_RANK MASTER_ADDR MASTER_PORT RUN_NAME MODEL_TAG NUM_ITERATIONS; do
    require_env "$required_var"
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="$BASE_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEPTH="${DEPTH:-24}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
FP8="${FP8:-1}"
FP8_RECIPE="${FP8_RECIPE:-tensorwise}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_TOKENS="${EVAL_TOKENS:-1048576}"
EVAL_AFTER_RUN="${EVAL_AFTER_RUN:-1}"
WINDOW_PATTERN="${WINDOW_PATTERN:-auto}"
PATTERNED_WINDOW_PATTERN="${PATTERNED_WINDOW_PATTERN:-SSSL}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-}"

mkdir -p "$BASE_DIR"

if [ ! -f "$BASE_DIR/tokenizer/token_bytes.pt" ]; then
    echo "Error: tokenizer artifacts are missing under $BASE_DIR/tokenizer. Run b300_prestage.sh first." >&2
    exit 1
fi

if ! ls "$BASE_DIR"/base_data_climbmix/*.parquet >/dev/null 2>&1; then
    echo "Error: dataset shards are missing under $BASE_DIR/base_data_climbmix. Run b300_prestage.sh first." >&2
    exit 1
fi

backend_info="$(python - <<'PY'
import torch
from nanochat.flash_attention import ATTENTION_BACKEND, ATTENTION_BACKEND_REASON

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the B300 train path.")

major, minor = torch.cuda.get_device_capability()
print(f"{ATTENTION_BACKEND}\t{torch.cuda.get_device_name(0)}\t{major}.{minor}\t{ATTENTION_BACKEND_REASON}")
PY
)"
IFS=$'\t' read -r ATTENTION_BACKEND GPU_NAME GPU_CAPABILITY ATTENTION_REASON <<<"$backend_info"

if [ "$WINDOW_PATTERN" = "auto" ]; then
    case "$ATTENTION_BACKEND" in
        fa3|fa4) RESOLVED_WINDOW_PATTERN="$PATTERNED_WINDOW_PATTERN" ;;
        *) RESOLVED_WINDOW_PATTERN="L" ;;
    esac
else
    RESOLVED_WINDOW_PATTERN="$WINDOW_PATTERN"
fi

echo "Shared BASE_DIR: $BASE_DIR"
echo "Node rank: $NODE_RANK / $NNODES"
echo "GPU: $GPU_NAME (SM $GPU_CAPABILITY)"
echo "Attention backend: $ATTENTION_BACKEND ($ATTENTION_REASON)"
echo "Window pattern: $WINDOW_PATTERN -> $RESOLVED_WINDOW_PATTERN"

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

if [ -n "$TOTAL_BATCH_SIZE" ]; then
    TRAIN_ARGS+=(--total-batch-size="$TOTAL_BATCH_SIZE")
fi

if [ "$FP8" = "1" ]; then
    TRAIN_ARGS+=(--fp8 --fp8-recipe="$FP8_RECIPE")
fi

if [ -n "$RESUME_FROM_STEP" ]; then
    TRAIN_ARGS+=(--resume-from-step="$RESUME_FROM_STEP")
fi

run_step torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m scripts.base_train -- "${TRAIN_ARGS[@]}"

if [ "$EVAL_AFTER_RUN" = "1" ] && [ "$NODE_RANK" = "0" ]; then
    run_step bash "$REPO_ROOT/runs/b300_eval.sh"
fi
