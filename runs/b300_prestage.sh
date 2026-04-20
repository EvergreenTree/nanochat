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

require_expected_shards() {
    local num_train_shards="$1"
    local shard_dir="$BASE_DIR/base_data_climbmix"
    local idx shard_path
    for ((idx = 0; idx < num_train_shards; idx++)); do
        shard_path="$(printf '%s/shard_%05d.parquet' "$shard_dir" "$idx")"
        [ -f "$shard_path" ] || {
            echo "Error: expected train shard is missing after download: $shard_path" >&2
            exit 1
        }
    done
    shard_path="$(printf '%s/shard_%05d.parquet' "$shard_dir" 6542)"
    [ -f "$shard_path" ] || {
        echo "Error: expected validation shard is missing after download: $shard_path" >&2
        exit 1
    }
}

require_env BASE_DIR

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="$BASE_DIR"

DATASET_SHARDS="${DATASET_SHARDS:-170}"
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"

mkdir -p "$BASE_DIR"

backend_info="$(python - <<'PY'
import torch
from nanochat.flash_attention import ATTENTION_BACKEND, ATTENTION_BACKEND_REASON, HAS_FA4

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the B300 prestage path.")

major, minor = torch.cuda.get_device_capability()
if major < 10:
    raise SystemExit(f"Expected a Blackwell-class GPU for B300 prestage, found SM {major}.{minor}.")

print(f"{ATTENTION_BACKEND}\t{int(HAS_FA4)}\t{torch.cuda.get_device_name(0)}\t{major}.{minor}\t{ATTENTION_BACKEND_REASON}")
PY
)"
IFS=$'\t' read -r ATTENTION_BACKEND HAS_FA4 GPU_NAME GPU_CAPABILITY ATTENTION_REASON <<<"$backend_info"

echo "Shared BASE_DIR: $BASE_DIR"
echo "GPU: $GPU_NAME (SM $GPU_CAPABILITY)"
echo "Attention backend: $ATTENTION_BACKEND ($ATTENTION_REASON)"

if [ "$HAS_FA4" != "1" ]; then
    echo "Error: Flash Attention 4 is not importable on this Blackwell environment." >&2
    exit 1
fi

run_step python -m nanochat.dataset -n "$DATASET_SHARDS"
require_expected_shards "$DATASET_SHARDS"
run_step python -m scripts.tok_train --max-chars="$TOKENIZER_MAX_CHARS" --vocab-size="$VOCAB_SIZE"
run_step python -m scripts.tok_eval

echo
echo "B300 prestage complete."
echo "Artifacts:"
echo "  dataset: $BASE_DIR/base_data_climbmix"
echo "  tokenizer: $BASE_DIR/tokenizer"
