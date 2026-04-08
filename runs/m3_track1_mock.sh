#!/bin/bash

set -euo pipefail

EXPECTED_PYTHON_PREFIX="/Users/evergreen/miniforge3/envs/py3_12"
MODEL_TAG="m3_track1_mock"
CACHE_DIR="$HOME/.cache/nanochat"

run_step() {
    echo
    echo "==> $*"
    "$@"
}

if [ -n "${NANOCHAT_BASE_DIR:-}" ]; then
    echo "Error: NANOCHAT_BASE_DIR is set to '$NANOCHAT_BASE_DIR'." >&2
    echo "Unset it before running this script so nanochat uses the default cache at $CACHE_DIR." >&2
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    echo "Error: 'python' is not on PATH." >&2
    exit 1
fi

echo "Running nanochat M3 track-1 mock with python: $(command -v python)"
echo "Expected active Conda env prefix: $EXPECTED_PYTHON_PREFIX"
echo "Default cache directory: $CACHE_DIR"

python - <<'PY'
import pathlib
import sys

expected_prefix = pathlib.Path("/Users/evergreen/miniforge3/envs/py3_12").resolve()
python_exe = pathlib.Path(sys.executable).resolve()
python_prefix = pathlib.Path(sys.prefix).resolve()

if not python_exe.is_relative_to(expected_prefix):
    raise SystemExit(
        f"Active python is {python_exe}, expected it to live under {expected_prefix}. "
        "Activate the local py3_12 Conda env first."
    )

if python_prefix != expected_prefix:
    raise SystemExit(
        f"Active sys.prefix is {python_prefix}, expected {expected_prefix}. "
        "Activate the local py3_12 Conda env first."
    )

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is missing from the active local env. "
        "Use the existing py3_12 Conda env for this script."
    ) from exc

if not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available in the active local env.")

try:
    import rustbpe  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "rustbpe is missing from the active local env. "
        "Install it once with `python -m pip install rustbpe` and rerun."
    ) from exc

print(f"Preflight OK: {python_exe}")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
PY

mkdir -p "$CACHE_DIR"

run_step python -m nanochat.report reset
run_step python -m nanochat.dataset -n 2
run_step python -m scripts.tok_train --max-chars=500000000 --vocab-size=32768
run_step python -m scripts.tok_eval
run_step python -m scripts.base_train \
    --device-type=mps \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=500 \
    --eval-tokens=65536 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --num-iterations=2000 \
    --run=dummy \
    --model-tag="$MODEL_TAG"
run_step python -m scripts.base_eval \
    --device-type=mps \
    --model-tag="$MODEL_TAG" \
    --eval=bpb,sample \
    --device-batch-size=1 \
    --split-tokens=16384
run_step python -m nanochat.report generate

echo
echo "M3 track-1 mock complete."
echo "Artifacts:"
echo "  tokenizer: $CACHE_DIR/tokenizer"
echo "  checkpoints: $CACHE_DIR/base_checkpoints/$MODEL_TAG"
echo
echo "If MPS OOMs, rerun just the training step with:"
echo "  --device-batch-size=16 --total-batch-size=8192"
