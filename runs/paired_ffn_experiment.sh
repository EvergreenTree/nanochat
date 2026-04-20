#!/usr/bin/env bash
set -euo pipefail

# Tiny local ablation runner for:
#   1. all transformer matrices on Muon
#   2. Muon except FFN pairs on lossless stacked joint Muon
#
# Override any setting with environment variables, for example:
#   STEPS=500 DEVICE_TYPE=mps bash runs/paired_ffn_experiment.sh

DEVICE_TYPE="${DEVICE_TYPE:-}"
STEPS="${STEPS:-50}"
DEPTH="${DEPTH:-4}"
SEQ_LEN="${SEQ_LEN:-512}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-512}"
SEED="${SEED:-42}"
SAVE_EVERY="${SAVE_EVERY:--1}"
RUN_ID="${RUN_ID:-paired_ffn_$(date +%s)}"
LOG_DIR="${LOG_DIR:-${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/paired_ffn_results/$RUN_ID}"

mkdir -p "$LOG_DIR"

COMMON=(
  --depth="$DEPTH"
  --max-seq-len="$SEQ_LEN"
  --device-batch-size="$DEVICE_BATCH_SIZE"
  --total-batch-size="$TOTAL_BATCH_SIZE"
  --num-iterations="$STEPS"
  --core-metric-every=-1
  --sample-every=-1
  --save-every="$SAVE_EVERY"
  --run=dummy
  --seed="$SEED"
  --mlp-c-proj-init=gaussian
  --mlp-activation=relu
)

if [[ -n "$DEVICE_TYPE" ]]; then
  COMMON+=(--device-type="$DEVICE_TYPE")
fi

run_recipe() {
  local name="$1"
  shift
  local tag="${RUN_ID}_${name}"
  local log="${LOG_DIR}/${name}.log"
  echo "Running ${name}; log=${log}; model-tag=${tag}"
  python -m scripts.base_train "${COMMON[@]}" --model-tag="$tag" "$@" 2>&1 | tee "$log"
}

run_recipe muon --optimizer-kind=muon
run_recipe paired_ffn --optimizer-kind=paired_ffn

python - <<'PY' "$LOG_DIR"
import re
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
pattern = re.compile(r"^step\s+(\d+)/(\d+).*?loss:\s+([0-9.]+).*?tok/sec:\s+([0-9,]+)")
print("recipe,logged_steps,lowest_loss,lowest_step,final_step,final_loss,mean_tok_per_sec")
for name in ["muon", "paired_ffn"]:
    rows = []
    for line in (log_dir / f"{name}.log").read_text().splitlines():
        match = pattern.search(line)
        if match:
            rows.append((int(match.group(1)), float(match.group(3)), int(match.group(4).replace(",", ""))))
    if not rows:
        print(f"{name},0,,,,")
        continue
    low = min(rows, key=lambda row: row[1])
    final = rows[-1]
    tps = [row[2] for row in rows if row[0] > 10] or [row[2] for row in rows]
    print(f"{name},{len(rows)},{low[1]:.6f},{low[0]},{final[0]},{final[1]:.6f},{sum(tps)/len(tps):.1f}")
PY
