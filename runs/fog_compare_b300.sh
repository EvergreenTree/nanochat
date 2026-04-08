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
require_env RUN_NAME
require_env NUM_ITERATIONS

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="$BASE_DIR"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

MODEL_TAG_PREFIX="${MODEL_TAG_PREFIX:-${RUN_NAME}_fog}"
FOG_VARIANT="${FOG_VARIANT:-flash}"
DEPTH="${DEPTH:-24}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_TOKENS="${EVAL_TOKENS:-1048576}"
SAVE_EVERY="${SAVE_EVERY:-500}"
QUANT_MONITOR_EVERY="${QUANT_MONITOR_EVERY:-100}"
SEED="${SEED:-42}"

if [ ! -f "$BASE_DIR/tokenizer/token_bytes.pt" ]; then
    echo "Error: tokenizer artifacts are missing under $BASE_DIR/tokenizer. Run b300_prestage.sh first." >&2
    exit 1
fi

if ! ls "$BASE_DIR"/base_data_climbmix/*.parquet >/dev/null 2>&1; then
    echo "Error: dataset shards are missing under $BASE_DIR/base_data_climbmix. Run b300_prestage.sh first." >&2
    exit 1
fi

COMMON_ARGS=(
    --device-type=cuda
    --arch-family=fog
    --fog-variant="$FOG_VARIANT"
    --seed="$SEED"
    --depth="$DEPTH"
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

run_arm() {
    local recipe="$1"
    local run_name="${RUN_NAME}-${recipe}"
    local model_tag="${MODEL_TAG_PREFIX}_${recipe}"
    run_step torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.base_train -- \
        "${COMMON_ARGS[@]}" \
        --run="$run_name" \
        --model-tag="$model_tag" \
        --precision-recipe="$recipe"
}

for recipe in bf16 fp8_full fp4_blackwell; do
    run_arm "$recipe"
done

if [ "$NODE_RANK" = "0" ]; then
    export MODEL_TAG_PREFIX RUN_NAME BASE_DIR
    python - <<'PY'
import json
import os
from pathlib import Path

base_dir = Path(os.environ["BASE_DIR"])
run_name = os.environ["RUN_NAME"]
tag_prefix = os.environ["MODEL_TAG_PREFIX"]
recipes = ["bf16", "fp8_full", "fp4_blackwell"]
comparison_dir = base_dir / "comparisons" / run_name
comparison_dir.mkdir(parents=True, exist_ok=True)

rows = {}
for recipe in recipes:
    summary_path = base_dir / "base_checkpoints" / f"{tag_prefix}_{recipe}" / "training_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        rows[recipe] = json.load(f)

bf16_val = rows["bf16"].get("val_bpb")
table_rows = []
for recipe, data in rows.items():
    row = {
        "recipe": recipe,
        "model_tag": data.get("model_tag"),
        "precision_backend": data.get("precision_backend"),
        "avg_tok_per_sec": data.get("avg_tok_per_sec"),
        "avg_step_time_s": data.get("avg_step_time_s"),
        "total_training_time_s": data.get("total_training_time_s"),
        "val_bpb": data.get("val_bpb"),
        "min_val_bpb": data.get("min_val_bpb"),
        "core_metric": data.get("core_metric"),
        "quality_delta_vs_bf16": (data.get("val_bpb") - bf16_val) if (bf16_val is not None and data.get("val_bpb") is not None) else None,
    }
    table_rows.append(row)

comparison = {
    "run_name": run_name,
    "recipes": table_rows,
}
with (comparison_dir / "comparison.json").open("w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)

headers = ["recipe", "avg_tok_per_sec", "avg_step_time_s", "total_training_time_s", "val_bpb", "min_val_bpb", "quality_delta_vs_bf16", "core_metric"]
with (comparison_dir / "comparison.md").open("w", encoding="utf-8") as f:
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for row in table_rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append("" if value is None else str(value))
        f.write("| " + " | ".join(values) + " |\n")

print(f"Wrote comparison artifacts to {comparison_dir}")
PY
fi
