# B300 Remote Agent Playbook

This file is for the remote execution agent, not for a human reader doing open-ended exploration.

## Objective

The scoring objective is not "run a benchmark." It is:

1. Use the dev B300 node before the scored window to make the environment deterministic.
2. Use the 4-node B300 cluster for one 10-minute training run on the specified data.
3. Spend the 10-minute window on training, not on dependency fixes, dataset downloads, tokenizer work, or debugging.

The safest default submission recipe in this fork is `FOG + fp8_full + full-context attention`.

## Files To Use

- `runs/b300_remote_setup.sh`
- `runs/b300_remote_run.sh`

Do not start from `README.md` during the event. Use this document and those two scripts.

## Required Inputs

- A clone of this branch on the remote machine
- A Blackwell/B300 CUDA machine
- A writable shared artifact directory visible from every cluster node
- The event-provided dataset scope

Use one shared directory for everything:

```bash
export BASE_DIR=/shared/nanochat
```

If the organizers specify a different shard count or a pre-mounted dataset location, obey that. Do not silently change dataset code.

## Phase 1: Setup Once On A Dev B300 Node

Run:

```bash
bash runs/b300_remote_setup.sh
```

Recommended explicit version:

```bash
ENV_DIR=$PWD/.venv-b300 \
BASE_DIR=/shared/nanochat_setup_smoke \
VALIDATE_SETUP=1 \
bash runs/b300_remote_setup.sh
```

Expected success criteria:

- PyTorch imports and sees CUDA
- GPU capability is SM `10.x` or newer
- `flash_attn.cute.flash_attn_func` imports
- Transformer Engine imports
- `fp8_full` resolves to Blackwell MXFP8
- `fp4_blackwell` resolves to NVFP4
- A tiny data download works
- A tiny tokenizer train works
- A tiny FOG training smoke works
- A tiny `base_eval` works

If setup fails, stop there. Do not spend cluster time trying random pip changes.

## Phase 2: Full Prestage Before The Scored Window

Do this on the dev node, not during the 10-minute cluster window.

Run:

```bash
ENV_DIR=$PWD/.venv-b300 \
BASE_DIR=/shared/nanochat \
DATASET_SHARDS=170 \
TOKENIZER_MAX_CHARS=2000000000 \
VOCAB_SIZE=32768 \
bash runs/b300_remote_run.sh prestage
```

Adjust `DATASET_SHARDS` only if the organizers specify a different data budget.

Expected success criteria:

- `BASE_DIR/base_data_climbmix/*.parquet` exists
- `BASE_DIR/tokenizer/tokenizer.pkl` exists
- `BASE_DIR/tokenizer/token_bytes.pt` exists
- `scripts.tok_eval` completes

The prestage path does not trust the downloader exit code by itself. It verifies that every expected train shard plus the validation shard actually exists on disk before moving on.

If prestage fails, do not proceed to multi-node training.

## Phase 3: Smoke The Real Training Stack

Do one short smoke run on a B300 node before the scored window.

Run:

```bash
ENV_DIR=$PWD/.venv-b300 \
BASE_DIR=/shared/nanochat \
PRECISION_RECIPE=fp8_full \
RUN_NAME=track1_smoke_fp8 \
MODEL_TAG=track1_smoke_fp8 \
bash runs/b300_remote_run.sh smoke
```

Expected success criteria:

- `Attention backend` reports `fa4`
- Training starts without OOM
- `tok/sec` is nonzero
- Checkpoint directory appears under `BASE_DIR/base_checkpoints/track1_smoke_fp8`
- Post-train eval runs

If `fp8_full` fails here, use the same smoke path with `PRECISION_RECIPE=bf16` as the fallback recipe. Do not discover that during the scored window.

## Phase 4: Use The 10-Minute Four-Node Window Properly

The scored run should be one training arm, not a three-way comparison. Comparison is for tuning before the event.

Default scored recipe:

- `arch-family=fog`
- `fog-variant=flash`
- `precision-recipe=fp8_full`
- `window-pattern=L`
- `NNODES=4`
- `NPROC_PER_NODE=8`
- `EVAL_EVERY=-1`
- `SAVE_EVERY=-1`
- `QUANT_MONITOR_EVERY=-1`

The remote script mode for this is `timed-arm`.

If you already know the correct iteration count, set it directly:

```bash
export ENV_DIR=$PWD/.venv-b300
export BASE_DIR=/shared/nanochat
export NNODES=4
export NODE_RANK=0
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export RUN_NAME=track1_fp8_submit
export MODEL_TAG=track1_fp8_submit
export PRECISION_RECIPE=fp8_full
export NUM_ITERATIONS=120
bash runs/b300_remote_run.sh timed-arm
```

If you only have measured average step time, let the script derive `NUM_ITERATIONS`:

```bash
export ENV_DIR=$PWD/.venv-b300
export BASE_DIR=/shared/nanochat
export NNODES=4
export NODE_RANK=0
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export RUN_NAME=track1_fp8_submit
export MODEL_TAG=track1_fp8_submit
export PRECISION_RECIPE=fp8_full
export STEP_TIME_S=4.35
export TARGET_MINUTES=10
export SAFETY_SECONDS=45
bash runs/b300_remote_run.sh timed-arm
```

Use the same command on each node, changing only `NODE_RANK`.

## Choosing The Recipe Before The Event

If you still need the controlled BF16 vs FP8 vs FP4 comparison, run it before the scored window:

```bash
ENV_DIR=$PWD/.venv-b300 \
BASE_DIR=/shared/nanochat \
RUN_NAME=track1_compare \
NUM_ITERATIONS=40 \
bash runs/b300_remote_run.sh compare
```

Do not run `compare` during the scored 10-minute cluster window.

## After The Timed Run

Run eval only after the scored window or only if the event rules allow it.

```bash
ENV_DIR=$PWD/.venv-b300 \
BASE_DIR=/shared/nanochat \
MODEL_TAG=track1_fp8_submit \
bash runs/b300_remote_run.sh eval
```

Useful output locations:

- Dataset: `BASE_DIR/base_data_climbmix`
- Tokenizer: `BASE_DIR/tokenizer`
- Checkpoints: `BASE_DIR/base_checkpoints/<MODEL_TAG>`
- Training summary: `BASE_DIR/base_checkpoints/<MODEL_TAG>/training_summary.json`
- Comparison artifacts: `BASE_DIR/comparisons/<RUN_NAME>`

## Non-Negotiable Rules

- Do not use the 10-minute 4-node window for package installation.
- Do not use the 10-minute 4-node window for dataset download or tokenizer training.
- Do not change the dataset source under time pressure.
- Do not run the three-arm comparison during the scored window.
- Do not switch away from `window-pattern=L` for `fp8_full` or `fp4_blackwell`.
- If FA4 or Transformer Engine is missing, stop and fix setup on the dev node first.

## Fast Triage

If the failure is:

- Hugging Face `503` or other transient dataset download errors: rerun `prestage`; the scripts now hard-fail if the expected shard files are still missing
- Import error for `flash_attn.cute`: rerun setup, confirm `flash-attn-4` installed into `ENV_DIR`
- Import error for Transformer Engine: rerun setup, confirm `transformer_engine[pytorch]` installed into `ENV_DIR`
- `Expected a Blackwell-class GPU`: you are on the wrong machine
- Missing tokenizer or parquet shards: rerun `prestage`
- OOM during smoke: lower `DEVICE_BATCH_SIZE` first, then rerun smoke before touching other knobs
- Slow or unstable timed run: prefer switching the scored recipe from `fp8_full` to `bf16` only if the smoke run already proved FP8 is unsafe
