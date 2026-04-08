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
require_env MODEL_TAG

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="$BASE_DIR"

DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
SPLIT_TOKENS="${SPLIT_TOKENS:-1048576}"
STEP="${STEP:-}"

if [ ! -d "$BASE_DIR/base_checkpoints/$MODEL_TAG" ]; then
    echo "Error: checkpoint directory $BASE_DIR/base_checkpoints/$MODEL_TAG does not exist." >&2
    exit 1
fi

EVAL_ARGS=(
    --device-type=cuda
    --model-tag="$MODEL_TAG"
    --eval=bpb,sample
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --split-tokens="$SPLIT_TOKENS"
)

if [ -n "$STEP" ]; then
    EVAL_ARGS+=(--step="$STEP")
fi

run_step python -m scripts.base_eval "${EVAL_ARGS[@]}"
