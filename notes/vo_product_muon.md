# VO Product Muon Experiment

This experiment adds opt-in optimizer variants for base pretraining:

- `muon`: current nanochat behavior.
- `adamw_all`: transformer block matrices use AdamW instead of Muon.
- `vo_product_muon`: attention `c_v.weight` and `c_proj.weight` pairs use the paired VO optimizer; remaining transformer matrices use Muon.

The default path is unchanged when no optimizer flags are passed.

## Implemented Formulas

For each active square same-shaped VO pair, the attention forward pass records
batch feature covariances:

```text
Wv = attn.c_v.weight
Wo = attn.c_proj.weight
Gv = grad(Wv)
Go = grad(Wo)
X = residual input before c_v
H = mixed value activations before c_proj
Cx = X.T @ X / (B * T)
Ch[head] = H_head.T @ H_head / (B * T)
```

The optimizer builds a per-head proxy instead of one whole concatenated
parameter product. With PyTorch Linear weight orientation, the dimensionally
correct value-side term is `Wo_head @ Gv_head @ Cx`:

```text
G_pair[head] =
  0.5 * (Go_head @ Ch[head] @ Wv_head + Wo_head @ Gv_head @ Cx)
```

`ls_muon` is now the only VO split. It runs the same style of Muon processing
used by the fused matrix path on the stacked per-head proxy: momentum, Polar
Express, and variance reduction. The processed proxy is `T[head]`.

```text
d = model_dim
dir_o_head = 0.5 * (T_head @ Wv_head.T) / (||Wv_head||_F^2 / d + eps)
dir_v_head = 0.5 * (Wo_head.T @ T_head) / (||Wo_head||_F^2 / d + eps)
```

The per-head directions are stacked back into `c_v.weight` and
`c_proj.weight`, then descent is applied as:

```text
W <- W - lr * dir - lr * wd * W * mask
mask = (dir * W) >= 0
```

The mask intentionally follows current nanochat Muon semantics. It is computed from the gradient-like direction before applying the negative learning-rate sign.

## Optimizer State

VO groups store a flat `params` list plus `pair_offsets`, for example:

```python
params=[Wv0, Wo0, Wv1, Wo1]
pair_offsets=[(0, 1), (2, 3)]
```

This keeps `optimizer.state_dict()` compatible with PyTorch's normal parameter-group serialization. VO state is anchored on each pair's `Wv` parameter and contains `step`, `momentum_buffer`, and `second_momentum_buffer`.

## Shape Fallback

Pairing is enabled only when both weights are 2D, square, and same-shaped. If a future GQA config makes `c_v.weight` and `c_proj.weight` incompatible, both parameters are routed back to standard Muon and the skipped pair count is logged.

`attn.ve_gate.weight` remains a transformer matrix and is routed with the rest of the transformer matrices according to `--optimizer-kind`.

## DDP Limitation

`vo_product_muon` is single-GPU only in v1. DDP setup raises `NotImplementedError` before constructing the optimizer. `adamw_all` works with the existing distributed AdamW path.

## Smoke Runs

These runs are intended to catch crashes, NaNs, and checkpoint/state issues before longer experiments:

```bash
python -m scripts.base_train \
  --depth=4 --max-seq-len=512 --device-batch-size=1 --total-batch-size=512 \
  --eval-tokens=512 --core-metric-every=-1 --sample-every=-1 \
  --num-iterations=50 --run=dummy --optimizer-kind=muon

python -m scripts.base_train \
  --depth=4 --max-seq-len=512 --device-batch-size=1 --total-batch-size=512 \
  --eval-tokens=512 --core-metric-every=-1 --sample-every=-1 \
  --num-iterations=50 --run=dummy --optimizer-kind=adamw_all

python -m scripts.base_train \
  --depth=4 --max-seq-len=512 --device-batch-size=1 --total-batch-size=512 \
  --eval-tokens=512 --core-metric-every=-1 --sample-every=-1 \
  --num-iterations=50 --run=dummy --optimizer-kind=vo_product_muon
```

## Experiment Helper

Run the staged benchmark helper:

```bash
bash runs/vo_product_muon_experiment.sh
```

Useful environment overrides:

```bash
RUN_PHASES=smoke,sweep,compare bash runs/vo_product_muon_experiment.sh
VO_DEPTH=8 VO_NUM_ITERATIONS=2000 VO_SEEDS=42,43,44 bash runs/vo_product_muon_experiment.sh
```

Artifacts are written under:

```text
$NANOCHAT_BASE_DIR/vo_product_muon_results
```

The analyzer can be rerun directly:

```bash
python -m scripts.vo_product_muon_analyze --results-dir "$NANOCHAT_BASE_DIR/vo_product_muon_results"
```

It writes:

```text
summary.csv
val_history.csv
val_bpb_vs_step.png
val_bpb_vs_wall_clock.png
benchmark_summary.md
```
