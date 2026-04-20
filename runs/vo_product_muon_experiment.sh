#!/bin/bash

set -euo pipefail

# Staged VO Product Muon experiment runner.
# Usage:
#   bash runs/vo_product_muon_experiment.sh
#   RUN_PHASES=smoke,sweep,compare bash runs/vo_product_muon_experiment.sh

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
RESULTS_DIR="${VO_RESULTS_DIR:-$NANOCHAT_BASE_DIR/vo_product_muon_results}"
LOG_DIR="$RESULTS_DIR/logs"
MANIFEST="$RESULTS_DIR/manifest.csv"
RUN_PHASES="${RUN_PHASES:-smoke,sweep,compare}"
WANDB_PREFIX="${WANDB_PREFIX:-vo_product_muon}"

mkdir -p "$LOG_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

if [ ! -f "$MANIFEST" ]; then
    echo "tag,phase,optimizer_kind,seed,matrix_lr,matrix_adamw_beta1,matrix_adamw_beta2,matrix_adamw_wd_mult,log_file" > "$MANIFEST"
fi

phase_enabled() {
    local phase="$1"
    [[ ",$RUN_PHASES," == *",$phase,"* ]]
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_train() {
    local tag="$1"
    local phase="$2"
    local optimizer_kind="$3"
    local seed="$4"
    local matrix_lr="$5"
    local beta1="$6"
    local beta2="$7"
    local wd_mult="$8"
    shift 8

    local log_file="$LOG_DIR/${tag}.log"
    log "Running $tag"
    echo "$tag,$phase,$optimizer_kind,$seed,$matrix_lr,$beta1,$beta2,$wd_mult,$log_file" >> "$MANIFEST"

    python -m scripts.base_train \
        --run="${WANDB_PREFIX}_${tag}" \
        --model-tag="$tag" \
        --optimizer-kind="$optimizer_kind" \
        --seed="$seed" \
        --matrix-lr="$matrix_lr" \
        --matrix-adamw-beta1="$beta1" \
        --matrix-adamw-beta2="$beta2" \
        --matrix-adamw-wd-mult="$wd_mult" \
        "$@" \
        2>&1 | tee "$log_file"
}

SMOKE_DEPTH="${SMOKE_DEPTH:-4}"
SMOKE_SEQ="${SMOKE_SEQ:-512}"
SMOKE_DEVICE_BATCH="${SMOKE_DEVICE_BATCH:-1}"
SMOKE_TOTAL_BATCH="${SMOKE_TOTAL_BATCH:-512}"
SMOKE_ITERATIONS="${SMOKE_ITERATIONS:-50}"
SMOKE_EVAL_TOKENS="${SMOKE_EVAL_TOKENS:-512}"

VO_DEPTH="${VO_DEPTH:-8}"
VO_SEQ="${VO_SEQ:-512}"
VO_DEVICE_BATCH="${VO_DEVICE_BATCH:-4}"
VO_TOTAL_BATCH="${VO_TOTAL_BATCH:-16384}"
VO_NUM_ITERATIONS="${VO_NUM_ITERATIONS:-2000}"
VO_EVAL_EVERY="${VO_EVAL_EVERY:-250}"
VO_EVAL_TOKENS="${VO_EVAL_TOKENS:-524288}"
VO_CORE_METRIC_EVERY="${VO_CORE_METRIC_EVERY:--1}"
VO_CORE_METRIC_MAX_PER_TASK="${VO_CORE_METRIC_MAX_PER_TASK:-500}"
VO_SEEDS="${VO_SEEDS:-42,43,44}"

DEFAULT_MATRIX_LR="${DEFAULT_MATRIX_LR:-0.004}"
DEFAULT_MATRIX_BETA1="${DEFAULT_MATRIX_BETA1:-0.9}"
DEFAULT_MATRIX_BETA2="${DEFAULT_MATRIX_BETA2:-0.95}"
DEFAULT_MATRIX_WD_MULT="${DEFAULT_MATRIX_WD_MULT:-1.0}"

IFS=',' read -ra SEEDS <<< "$VO_SEEDS"
SWEEP_SEED="${SEEDS[0]}"

COMMON_SMOKE_ARGS=(
    --depth="$SMOKE_DEPTH"
    --max-seq-len="$SMOKE_SEQ"
    --device-batch-size="$SMOKE_DEVICE_BATCH"
    --total-batch-size="$SMOKE_TOTAL_BATCH"
    --eval-tokens="$SMOKE_EVAL_TOKENS"
    --core-metric-every=-1
    --sample-every=-1
    --save-every=-1
    --num-iterations="$SMOKE_ITERATIONS"
)

COMMON_BENCH_ARGS=(
    --depth="$VO_DEPTH"
    --max-seq-len="$VO_SEQ"
    --device-batch-size="$VO_DEVICE_BATCH"
    --total-batch-size="$VO_TOTAL_BATCH"
    --eval-every="$VO_EVAL_EVERY"
    --eval-tokens="$VO_EVAL_TOKENS"
    --core-metric-every="$VO_CORE_METRIC_EVERY"
    --core-metric-max-per-task="$VO_CORE_METRIC_MAX_PER_TASK"
    --sample-every=-1
    --save-every=-1
    --num-iterations="$VO_NUM_ITERATIONS"
)

if phase_enabled smoke; then
    run_train "smoke_muon" "smoke" "muon" "$SWEEP_SEED" "$DEFAULT_MATRIX_LR" "$DEFAULT_MATRIX_BETA1" "$DEFAULT_MATRIX_BETA2" "$DEFAULT_MATRIX_WD_MULT" "${COMMON_SMOKE_ARGS[@]}"
    run_train "smoke_adamw_all" "smoke" "adamw_all" "$SWEEP_SEED" "$DEFAULT_MATRIX_LR" "$DEFAULT_MATRIX_BETA1" "$DEFAULT_MATRIX_BETA2" "$DEFAULT_MATRIX_WD_MULT" "${COMMON_SMOKE_ARGS[@]}"
    run_train "smoke_vo_ls_muon" "smoke" "vo_product_muon" "$SWEEP_SEED" "$DEFAULT_MATRIX_LR" "$DEFAULT_MATRIX_BETA1" "$DEFAULT_MATRIX_BETA2" "$DEFAULT_MATRIX_WD_MULT" "${COMMON_SMOKE_ARGS[@]}"
fi

if phase_enabled sweep; then
    MATRIX_LRS=(0.002 0.004 0.008 0.016)
    MATRIX_BETAS=("0.9,0.95" "0.9,0.98")
    MATRIX_WD_MULTS=(0.5 1.0)
    for lr in "${MATRIX_LRS[@]}"; do
        for betas in "${MATRIX_BETAS[@]}"; do
                IFS=',' read -r beta1 beta2 <<< "$betas"
            for wd_mult in "${MATRIX_WD_MULTS[@]}"; do
                tag="sweep_adamw_lr${lr}_b${beta1}_${beta2}_wd${wd_mult}"
                run_train "$tag" "sweep" "adamw_all" "$SWEEP_SEED" "$lr" "$beta1" "$beta2" "$wd_mult" "${COMMON_BENCH_ARGS[@]}"
            done
        done
    done
fi

python -m scripts.vo_product_muon_analyze --results-dir "$RESULTS_DIR"

BEST_ADAMW=$(
python - "$RESULTS_DIR/summary.csv" "$DEFAULT_MATRIX_LR" "$DEFAULT_MATRIX_BETA1" "$DEFAULT_MATRIX_BETA2" "$DEFAULT_MATRIX_WD_MULT" <<'PY'
import csv
import sys

summary, default_lr, default_b1, default_b2, default_wd = sys.argv[1:6]
best = None
try:
    with open(summary, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "sweep" or row.get("optimizer_kind") != "adamw_all":
                continue
            try:
                val = float(row["final_val_bpb"])
            except (TypeError, ValueError):
                continue
            if best is None or val < best[0]:
                best = (val, row)
except FileNotFoundError:
    best = None

if best is None:
    print(default_lr, default_b1, default_b2, default_wd)
else:
    row = best[1]
    print(row["matrix_lr"], row["matrix_adamw_beta1"], row["matrix_adamw_beta2"], row["matrix_adamw_wd_mult"])
PY
)
read -r BEST_MATRIX_LR BEST_MATRIX_BETA1 BEST_MATRIX_BETA2 BEST_MATRIX_WD_MULT <<< "$BEST_ADAMW"
log "Best AdamW matrix config: lr=$BEST_MATRIX_LR beta=($BEST_MATRIX_BETA1,$BEST_MATRIX_BETA2) wd_mult=$BEST_MATRIX_WD_MULT"

if phase_enabled compare; then
    for seed in "${SEEDS[@]}"; do
        run_train "compare_muon_seed${seed}" "compare" "muon" "$seed" "$DEFAULT_MATRIX_LR" "$DEFAULT_MATRIX_BETA1" "$DEFAULT_MATRIX_BETA2" "$DEFAULT_MATRIX_WD_MULT" "${COMMON_BENCH_ARGS[@]}"
        run_train "compare_adamw_all_seed${seed}" "compare" "adamw_all" "$seed" "$BEST_MATRIX_LR" "$BEST_MATRIX_BETA1" "$BEST_MATRIX_BETA2" "$BEST_MATRIX_WD_MULT" "${COMMON_BENCH_ARGS[@]}"
        run_train "compare_vo_ls_muon_seed${seed}" "compare" "vo_product_muon" "$seed" "$DEFAULT_MATRIX_LR" "$DEFAULT_MATRIX_BETA1" "$DEFAULT_MATRIX_BETA2" "$DEFAULT_MATRIX_WD_MULT" "${COMMON_BENCH_ARGS[@]}"
    done
fi

python -m scripts.vo_product_muon_analyze --results-dir "$RESULTS_DIR"
log "Results saved to $RESULTS_DIR"
