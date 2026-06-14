# Experiment: sparse_obs_cross_attn — TC Intensity Classifier

**Goal:** Given sparse in-situ land surface observations within a fixed radius of a query position at time *t*, predict the Saffir-Simpson Hurricane Wind Scale (SSHS) intensity class of any tropical cyclone present, or classify the sample as background (no storm).

This is a binary + ordinal classification problem over 11 classes:

| Class | Meaning | SSHS |
|------:|---------|------|
| 0 | No storm (background) | — |
| 1–3 | Sub-tropical / pre-tropical disturbances | −4 to −2 |
| 4 | Tropical Depression | −1 |
| 5 | Tropical Storm | 0 |
| 6–10 | Category 1–5 Hurricane | +1 to +5 |

---

## Data sources

**IBTrACS** (`ibtracs_full.npz`) — storm centre position, timestamp, and SSHS class for every 6-hourly TC observation in the training domain. 10,191 rows across all cyclone types.

**InsituLand** (`insitu_land_clean.npz` + `insitu_land_station_meta.npz`) — land surface hourly observations from Copernicus C3S for 552 stations in the Caribbean / Gulf domain (LAT 0–30°N, LON 100–45°W). 74.7M observation rows.

**Observed variables** (per station per timestamp):
- `air_pressure_at_sea_level` (Pa)
- `air_temperature` (K)
- `dew_point_temperature` (K)
- `wind_east`, `wind_north` (m/s) — *derived*: `TCDataset` decomposes the raw
  `wind_speed` + `wind_from_direction` columns into (u, v) velocity components
  (meteorological FROM convention, `utils/geoscience/met_conversions.py`).
  This removes the 0°/360° direction seam and shrinks low-speed direction noise
  by magnitude. Calm reports (speed 0, direction missing) become an exact
  (0, 0) vector rather than a missing value; non-calm speed without direction
  stays missing. `obs_bounds` for the components are signed symmetric
  (±115 m/s) so 0 m/s normalises to exactly 0 under `minmax_11`.

**Splits:**
- Train: IBTrACS seasons 2005–2020
- Val: 2021–2022
- Test: 2023–2025
- Hard test: multi-storm timestamps (870 times when ≥2 storms were active simultaneously) — held out entirely

**Batching:** each batch is 1:1 balanced — half TC samples (storm centre as query), half background samples (domain point during non-TC periods). Train backgrounds are fresh uniform draws each step; val/test loaders use ONE frozen background set (Latin-Hypercube positions + fixed-seed synoptic timestamps) reused every epoch, plus deterministic nearest-N station selection, so eval differences are purely model change. Sequential (eval) epochs yield every valid TC sample — the final partial batch is flushed with proportionally fewer backgrounds.

---

## Architecture

`TCClassifier` is a single `TransformerEncoder` over N+1 tokens: N station tokens followed by one query/CLS token. An asymmetric attention mask separates the two roles — stations contextualise each other without seeing the query; the query then reads the fully-contextualised station network and drives the classification head.

### Token construction

**Senseiver-style single projection (2023).** A station token is one linear map (`token_proj`) over the concatenation of observation values, the missingness mask, and the Fourier-embedded position — observations and position are projected *jointly, once*, rather than through separate obs/position layers that are summed:

```
station_input  = concat([obs_zeroed, obs_mask, Fourier(station_coords)])   # (B, N, 2F+K)
station_tokens = token_proj(station_input)                                 # (B, N, D)
```

Missing observations are filled with 0; the concatenated mask channel — not the fill value — disambiguates "observed 0" from "absent" (the mask's projection weights carry the distinction). No learned or constant sentinel is needed.

**Query token** — the *same* `token_proj`, applied to a learned stand-in `query_obs_slots` (ξ) occupying the obs/mask slots, with `Fourier(query_coords)` supplying the position slots:

```
query_input = concat([query_obs_slots, Fourier(query_coords)])             # (B, 2F+K)
query_token = token_proj(query_input)                                      # (B, 1, D)
```

There is no separate query projection. For `unit_circle` the query sits at `(0,0)`, so its position features are constant and the learned ξ carries the query identity. The model is **coordinate-agnostic** — it projects whatever `station_coords`/`query_coords` the datamodule supplies, so the location-encoding choice lives only in the data config.

### Asymmetric attention mask

```
                  attend to →
              station_0 … station_N-1   query
              ─────────────────────────────────
station_0  │      ✓              ✓        ✗
    ⋮      │      ✓              ✓        ✗    ← stations can't see query
station_N-1│      ✓              ✓        ✗
query      │      ✓              ✓        ✓    ← query reads everything
```

Padding stations (where `station_mask = False`) have their entire column blocked — no token can attend to a padding position.

Set `model.full_self_attention: true` to open the `stations → query` block (the `✗` column above), giving complete self-attention over all N+1 tokens — the standard unrestricted Transformer pattern. Default `false` keeps the asymmetric mask. Both paths go through `build_attention_mask` (the single source of truth, also used by `plot_attention_mask`).

### Location encoding modes

Set in `data.location_encoding` only — the model is coordinate-agnostic and projects whatever coords it is handed:

| Mode | `station_coords` | `query_coords` |
|------|-----------------|----------------|
| `unit_circle` | `[x, y]` storm-centred local map (`norm_dist·sin(bearing), norm_dist·cos(bearing)`), each in [-1, 1] | `[0, 0]` = storm position; Fourier features constant, so ξ carries query identity |
| `domain` | `[norm_lat_rad, norm_lon_rad]` absolute | same encoding; `Fourier(query_coords)` varies and feeds the shared `token_proj` |

`unit_circle` is rotation-equivariant (the storm is always "at the origin"). `domain` retains absolute geographic position.

### Missing observations

Stations within the radius may have missing values for some variables. Missing entries are flagged by `obs_mask` and filled with 0 inside the model. The disambiguation between "observed 0" and "absent" is carried by an explicit mask channel:

`model.missingness_indicator: true` (default) concatenates `obs_mask` (as 0./1.) as its own channel in the `token_proj` input. The mask's learned projection weights separate a missing feature from a real observation equal to 0 — the fill value itself is irrelevant. Set `false` to drop the mask channel and reproduce the aliased behaviour where missing == observed-zero (ablation only).

### Full forward pass

```
1.  obs_zeroed = where(obs_mask, station_obs, 0)                            (B, N, F)
2.  station_input  = concat([obs_zeroed, obs_mask, Fourier(station_coords)])  (B, N, 2F+K)
    station_tokens = token_proj(station_input)                             (B, N, D)
3.  query_input = concat([query_obs_slots, Fourier(query_coords)])         (B, 2F+K)
    query_token = token_proj(query_input)                                  (B, 1, D)
4.  tokens  = concat([station_tokens, query_token], axis=1)                 (B, N+1, D)
5.  encoded = TransformerEncoder(tokens, mask=asymmetric_mask)              (B, N+1, D)
6.  logits  = head(LayerNorm(encoded[:, N, :]))                             (B, 11)
```

---

## File layout

```
sparse_obs_cross_attn/
├── configs/
│   ├── tc_classifier.yaml   Full training config
│   ├── tc_tune.yaml         Short-epoch config for HP search
│   └── schema.json          JSON schema for config validation
├── runs/                Training run artifacts (checkpoints, logs, figures)
├── baselines/           Baseline models for comparison
├── data/
│   ├── dataset.py       TCDataset — joins IBTrACS + InsituLand per sample,
│   │                       coordinate encoding, obs normalisation
│   ├── datamodule.py    TCDataModule + TCLoader — balanced TC/background
│   │                       batches, station_selection (nearest/random),
│   │                       frozen LHS eval backgrounds, partial-batch flush
│   ├── encoding.py      encode/decode pairs (unit_circle local x-y,
│   │                       domain FOV-normalised) — exact inverses shared
│   │                       by dataset + plotting
│   ├── splits.py        resolve_splits — data.split config → per-split
│   │                       datasets + run manifest ('season' and 'sid'
│   │                       strategies)
│   └── sources/
│       ├── ibtracs.py       IBTrACSDataset — filter primitives (seasons,
│       │                       SIDs, single/multi-storm), sid-meta
│       │                       validation, SSHS label mapping
│       └── insitu_land.py   InsituLandDataset — haversine spatial filter,
│                                reliability/year filtering, binary-search time queries
├── plotting/
│   └── plotting.py      Confusion matrix and class metrics (thin wrappers
│                           over jrt/utils/plotting fields.plot_heatmap and
│                           curves.plot_grouped_bars), geographic
│                           attention plots (unit_circle: storm-centred
│                           local x-y scatter with km rings via
│                           _value_scatter; domain: thin wrapper over
│                           fields.plot_scatter_overlay), layers×heads
│                           attention-matrix grid, and the static
│                           asymmetric-mask figure
└── train/
    ├── model.py         TCClassifier — unified Transformer encoder
    ├── metrics.py       build_metrics_fns glue (metrics live in
    │                       jrt/training/metrics.py)
    ├── train.py         CLI entry point + observability callbacks
    │                       (attention entropy/map/grid, gradient flow)
    ├── evaluate.py      Evaluation pipeline (metrics + predictions)
    └── tune.py          Optuna hyperparameter search entry point
```

---

## Setup

### 1. Environment

```bash
conda env create -f environment.yaml   # creates the 'jrt' environment
conda activate jrt
```

For GPU support:
```bash
pip install --upgrade "jax[cuda12]"    # CUDA 12.x — see jax.readthedocs.io
```

Register the environment as a Jupyter kernel (one-time):
```bash
python -m ipykernel install --user --name jrt --display-name "JAX Research Template"
```

### 2. Data

The five data files are not in the repository. Place them on the target machine and update the `data:` paths in the config. Recommended layout:

```
/data/sparse_obs/
    ibtracs/
        ibtracs_full.npz
        ibtracs_multi_storm_times.npz
        ibtracs_sid_meta.npz
    insitu-land/
        insitu_land_clean.npz
        insitu_land_station_meta.npz
```

`ibtracs_sid_meta.npz` is the per-storm metadata table (SID, SEASON, peak SSHS, track start/end, ...). It is validated against `ibtracs_full.npz` on load — SID set and row counts must match exactly — so regenerate both together.

To keep machine-specific paths out of version control, create a `configs/tc_classifier_local.yaml` (gitignored) and load it instead, or patch paths programmatically in a notebook (see below).

```
# .gitignore
jrt/experiments/sparse_obs_cross_attn/configs/*_local.yaml
```

### 3. Config

The only required edits before a first run are the five `data:` paths. Key flags to understand:

- `data.split` is required — resolved by `data/splits.py` into filtered datasets plus a run manifest written next to the checkpoints. Two strategies: `season` (per-split IBTrACS season lists, disjoint, validated — the default config reproduces the original hardcoded split: train 2005–2020, val 2021–2022, test 2023–2025, `hard_test: multi_storm`) and `sid` (hybrid: `test.seasons` lists edge years, with membership decided by track calendar years from the sid-meta table; remaining interior storms are assigned train/val by SID at `val.fraction`, seeded, stratified by `val.stratify_by`, train being the implicit remainder; train and val deliberately share the interior-year insitu stream — only test is time-separated)
- `data.location_encoding` — `unit_circle` (default) or `domain`; the model is coordinate-agnostic so this lives in the data block only
- `model.missingness_indicator: true` (default) — concatenate the obs mask as its own channel (missing obs filled with 0); `false` drops it (ablation only)
- `trainer.run_dir` — change per run to avoid overwriting checkpoints

---

## Usage

### CLI quick reference

All commands run from the **repository root** with **`jrt/` on the Python
path** — the packages are rooted at `jrt/` (no `jrt.` import prefix), so a bare
`python -m experiments...` from the repo root fails with
`No module named 'experiments'`. Set `PYTHONPATH` once per shell:

```bash
# bash / zsh
export PYTHONPATH=jrt
```
```powershell
# PowerShell
$env:PYTHONPATH = "jrt"
```

`CFG` below is the config path (positional argument for every entry point):

```bash
CFG=jrt/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml

# Train                                  # Resume from latest checkpoint
python -m experiments.sparse_obs_cross_attn.train.train $CFG
python -m experiments.sparse_obs_cross_attn.train.train $CFG --resume

# Evaluate (best checkpoint)
python -m experiments.sparse_obs_cross_attn.train.evaluate $CFG \
    --checkpoint_dir runs/tc_classifier/run_01/checkpoints \
    --output_dir    runs/tc_classifier/run_01/eval \
    --split test --n_attn_samples 4 --geo --no_show

# Hyperparameter search
python -m experiments.sparse_obs_cross_attn.train.tune \
    jrt/experiments/sparse_obs_cross_attn/configs/tc_tune.yaml \
    --n_trials 50 \
    --storage sqlite:///runs/tc_classifier/hp_search/study.db \
    --study_name tc_classifier_v1
```

| Entry point | Argument | Default | Meaning |
|---|---|---|---|
| `train.train` | `config` (positional) | — | YAML config path |
| | `--resume` | off | continue from `checkpoints/latest/` |
| `train.evaluate` | `config` (positional) | — | YAML config path |
| | `--checkpoint_dir` | from config | override `trainer.checkpoint_dir`/`run_dir` |
| | `--output_dir` | None | save figures as PNGs here |
| | `--split` | `test` | `test` or `val` |
| | `--n_attn_samples` | 4 | attention figures from the first batch; 0 = skip |
| | `--geo` | off | cartopy map canvases (azimuthal storm-centred for unit_circle, PlateCarree for domain; needs cartopy) |
| | `--no_show` | off | don't open figure windows |
| `train.tune` | `config` (positional) | — | tune YAML config path |
| | `--n_trials` | 25 | Optuna trials |
| | `--storage` | in-memory | e.g. `sqlite:///path/study.db` (re-run same command to resume) |
| | `--study_name` | `tc_classifier` | study identifier inside the storage |
| | `--direction` | `minimize` | optimization direction |
| | `--n_startup_trials` | 5 | trials before pruning activates |
| | `--n_warmup_steps` | 10 | epochs per trial before pruning is checked |

Profiling is config-driven, not a CLI flag: set `trainer.profile: true`
(+ optional `trainer.profile_steps`) and the first training steps are traced
to `<run_dir>/logs/profile` — view with `tensorboard --logdir <that dir>`.

### Training (CLI)

Run from the **repository root** with `PYTHONPATH=jrt` set (see the CLI quick
reference above):

```bash
# First run
python -m experiments.sparse_obs_cross_attn.train.train \
    jrt/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml

# Resume an interrupted run
python -m experiments.sparse_obs_cross_attn.train.train \
    jrt/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
    --resume
```

Artifacts are written under `trainer.run_dir` (default `runs/tc_classifier/run_01/`):

```
runs/tc_classifier/run_01/
├── checkpoints/
│   ├── best/      Best validation checkpoint (orbax pytree)
│   └── latest/    Resume checkpoint
└── logs/
    ├── hparams.json
    └── figures/   Attention maps etc. (NullLogger mode)
```

### Evaluation (CLI)

The config path is a **positional** argument:

```bash
# Evaluate on test split (default)
python -m experiments.sparse_obs_cross_attn.train.evaluate \
    jrt/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
    --checkpoint_dir runs/tc_classifier/run_01/checkpoints \
    --output_dir runs/tc_classifier/run_01/eval

# Validation split
python -m experiments.sparse_obs_cross_attn.train.evaluate \
    jrt/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
    --split val

# Save plots to disk without displaying
python -m experiments.sparse_obs_cross_attn.train.evaluate \
    jrt/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
    --output_dir runs/tc_classifier/run_01/eval \
    --no_show
```

Outputs:
- Confusion matrix — row-normalised 11×11 and raw counts
- Per-class precision / recall / F1 bar chart
- Geographic attention maps — storm-centred local x-y scatter with km distance rings (unit_circle) or lat/lon scatter (domain), coloured by mean attention weight of the last layer's query row. Figure titles carry storm attribution: `"<SID> <NAME> — true: Cat 3, pred: TS"` (or `background`). With `--geo` (or `geo=True`) the scatter is drawn on a cartopy map with coastlines/borders: PlateCarree for domain, and for unit_circle an **AzimuthalEquidistant projection centred on the storm** — that projection's native coordinates are metres east/north of the centre, i.e. exactly the local x-y encoding × radius, so stations, km rings, and coastlines align by construction. Requires cartopy (optional dependency); default stays cartopy-free
- Attention-matrix grids — layers × heads panels of the full (N+1)×(N+1) matrices per sample (`plot_attention_matrix_grid`; plain imshow, no per-token labels, dashed lines mark the query row/column)
- Attention-mask figure — the exact asymmetric boolean mask the model builds (`plot_attention_mask`, single source of truth: `model.build_attention_mask`)

### Hyperparameter search (CLI)

```bash
# In-memory search (quick experiments, results lost on exit)
python -m experiments.sparse_obs_cross_attn.train.tune \
    jrt/experiments/sparse_obs_cross_attn/configs/tc_tune.yaml \
    --n_trials 25

# Persistent search — resume by running the same command again
python -m experiments.sparse_obs_cross_attn.train.tune \
    jrt/experiments/sparse_obs_cross_attn/configs/tc_tune.yaml \
    --n_trials 50 \
    --storage sqlite:///runs/tc_classifier/hp_search/study.db \
    --study_name tc_classifier_v1
```

After the study finishes, best parameters are printed and written to `runs/tc_classifier/hp_search/best_params.json`. Copy those values into `tc_classifier.yaml` and re-train at full length.

### Jupyter notebook

Select the `jrt` kernel (see Setup → Environment above), then:

```python
# Cell 1 — path and env setup
import sys, os
sys.path.insert(0, 'jrt')        # resolves `from experiments.* import ...`
os.environ['JAX_LOG_COMPILES'] = '0'
```

```python
# Cell 2 — load config, override paths for this machine
import yaml

with open('jrt/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml') as f:
    config = yaml.safe_load(f)

DATA_ROOT = '/data/sparse_obs'   # ← set to your data location
config['data'].update({
    'ibtracs_path':    f'{DATA_ROOT}/ibtracs/ibtracs_full.npz',
    'multi_storm_path':f'{DATA_ROOT}/ibtracs/ibtracs_multi_storm_times.npz',
    'insitu_obs_path': f'{DATA_ROOT}/insitu-land/insitu_land_clean.npz',
    'insitu_meta_path':f'{DATA_ROOT}/insitu-land/insitu_land_station_meta.npz',
})
config['trainer']['run_dir']     = 'runs/notebook/run_01'
config['trainer']['log_backend'] = 'null'
config['trainer']['num_epochs']  = 10     # short run to check everything works
seed = int(config.get('seed', 42))
config['trainer'].setdefault('seed', seed)
```

```python
# Cell 3 — build components
from experiments.sparse_obs_cross_attn.data.datamodule import TCDataModule
from experiments.sparse_obs_cross_attn.train.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.train.model import TCClassifier
from training.trainer import Trainer

dm           = TCDataModule.from_config(config['data'])
model        = TCClassifier(**config['model'])
metrics_fns  = build_metrics_fns()
trainer      = Trainer(model, metrics_fns, config['trainer'])
train_loader = dm.train_loader(seed=seed, shuffle=True)
val_loader   = dm.val_loader()

# Sanity-check one batch
batch = next(iter(train_loader))
print('station_obs: ', batch['X']['station_obs'].shape)   # (B, N, F)
print('station_mask:', batch['X']['station_mask'].shape)  # (B, N)
print('y classes:   ', set(batch['y'].tolist()))
```

```python
# Cell 4 — verify one forward pass before committing to training
import jax.numpy as jnp

state  = trainer._init_state(batch)
logits = model.apply({'params': state.params}, batch['X'], train=False)
print('logits:', logits.shape)                       # (B, 11)
print('finite:', bool(jnp.all(jnp.isfinite(logits))))
```

```python
# Cell 5 — train with live attention entropy logging
import numpy as np

def attn_callback(state, epoch, global_step):
    _, w = model.apply({'params': state.params}, batch['X'],
                       train=False, return_weights=True)
    w       = np.asarray(w)[-1][:, :, -1, :]   # last layer, query row → (B, H, N+1)
    entropy = float(-np.sum(w * np.log(w + 1e-12), axis=-1).mean())
    print(f'  epoch {epoch:3d}  val/attn_entropy = {entropy:.4f}')

best_state = trainer.fit(train_loader, val_loader,
                         epoch_callbacks=[attn_callback])
```

```python
# Cell 6 — evaluate on test split
from experiments.sparse_obs_cross_attn.train.evaluate import (
    collect_predictions, confusion_matrix, per_class_metrics,
    per_storm_metrics, CLASS_NAMES,
)
from experiments.sparse_obs_cross_attn.plotting.plotting import (
    plot_confusion_matrix, plot_class_metrics,
)
import matplotlib.pyplot as plt

variables = {'params': best_state.params}
preds, labels, logits, meta = collect_predictions(model, variables, dm.test_loader())

# Every prediction is attributable to a named storm (background → sid None)
storm_metrics = per_storm_metrics(preds, labels, meta['sid'])

cm  = confusion_matrix(preds, labels)
pcm = per_class_metrics(cm)
plot_confusion_matrix(cm, CLASS_NAMES, normalize=True, title='Test — row-normalised')
plot_class_metrics(pcm, CLASS_NAMES)
plt.show()
```

```python
# Cell 7 — attention figures
from experiments.sparse_obs_cross_attn.plotting.plotting import (
    extract_attention_weights, plot_attention_geographic,
    plot_attention_matrix_grid, plot_attention_mask,
)

attn_batch   = next(iter(val_loader))
attn_weights = extract_attention_weights(model, variables, attn_batch)
print('weights:', attn_weights.shape)   # (num_layers, B, H, N+1, N+1)

# Geographic map — query row of the last layer
fig = plot_attention_geographic(
    attn_weights[-1][:, :, -1, :], attn_batch,
    location_encoding = config['data']['location_encoding'],
    fov_lat           = config['data'].get('fov_lat'),
    fov_lon           = config['data'].get('fov_lon'),
    radius_km         = config['data'].get('radius_km', 500.0),
    sample_idx        = 0,
)

# Layers × heads grid of full (N+1)×(N+1) matrices
fig_grid = plot_attention_matrix_grid(attn_weights, sample_idx=0)

# Static asymmetric-mask figure (stations blocked from query, padding blocked)
fig_mask = plot_attention_mask(np.asarray(attn_batch['X']['station_mask'][0]))
plt.show()
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Training objective from `trainer.loss` (e.g. `squared_emd`) |
| `val/loss` | Same objective evaluated on val |
| `val/cross_entropy` | Validation CE — always reported for cross-run comparability; patience metric for early stopping |
| `val/accuracy` | Top-1 accuracy over all 11 classes |
| `val/binary_accuracy` | TC vs no-TC (class 0 vs class > 0); random chance = 0.5 |
| `val/mae_class` | Mean \|predicted class − true class\| in class units |
| `val/qwk` | Quadratic-weighted kappa (full val set, every `eval_plots_every_n_epochs`) — ordinal agreement; 1 = perfect, 0 = chance, negative = worse than chance |
| `val/ece` | Expected calibration error (full val set, every `eval_plots_every_n_epochs`) — gap between confidence and accuracy; 0 = perfectly calibrated |
| `val/attn_entropy` | Entropy of the LAST layer's query-row attention weights over N+1 positions. A falling curve means the model is concentrating attention on specific stations. |

**Interpretation:**
- A model that always predicts class 0 achieves `binary_accuracy = 0.5` but `mae_class ≈ 3`. Use `mae_class` as the primary signal for ordinal quality.
- `val/qwk` and `val/ece` are FULL-SET metrics (computed over the accumulated val predictions in `evaluate.py`/the eval-plots callback), not per-batch — they're too noisy/ill-defined on a `batch_size`-8 step to live in `metrics_fns`. `val/qwk` tells you whether an ordinal loss (`squared_emd`) is actually improving ordinal agreement over flat CE; `val/ece` is the calibration measurement.
- **Temperature scaling** (Guo et al. 2017): `evaluate.py` fits a single temperature `T` on the **val** split (`fit_temperature`, an exact ternary search since NLL is convex in `1/T`) and `print_report` prints both `<split>/ece` and `<split>/ece_tempscaled` with the fitted `T`. `T` divides the logits, so it recalibrates confidence without changing the argmax — accuracy, QWK and the per-class table are identical. For `--split test` this is the proper val→test transfer; for `--split val` it is an in-sample check.
- `val/attn_entropy` includes the query's self-attention weight (the last of N+1 positions). A high self-attention weight early in training is expected — the model is relying on its learned query prior (`query_obs_slots`). Expect it to decrease as the model learns to trust station data.

---

## Config reference

Key fields in `tc_classifier.yaml`:

| Key | Default | Notes |
|-----|---------|-------|
| `seed` | 3678 | Single seed for model init, dropout, and data shuffle |
| `data.radius_km` | 500 | Stations outside this radius are excluded |
| `data.time_window_hours` | 0.1 | Temporal tolerance (±) for matching obs to the query time; each station contributes only its report nearest in time |
| `data.max_stations` | 64 | Padding / truncation limit |
| `data.min_stations` | 1 | Samples with fewer stations are dropped |
| `data.station_selection` | `random` | TRAIN-loader station subsampling above `max_stations`: `random` (epoch-varying augmentation) or `nearest`. Val/test loaders always default to `nearest` (deterministic) |
| `data.location_encoding` | `unit_circle` | `unit_circle` or `domain`; model is coordinate-agnostic, so this lives in the data block only |
| `data.obs_normalisation` | `minmax_11` | `minmax_01` / `minmax_11` / `standardise` |
| `model.full_self_attention` | `false` | `false` = asymmetric mask (stations never attend to the query); `true` = complete self-attention over all N+1 tokens |
| `model.missingness_indicator` | `true` | `true` = concatenate `obs_mask` as its own channel in `token_proj` (missing obs filled 0), disambiguating "missing" from a real obs equal to 0; `false` = aliased behaviour (ablation) |
| `model.embed_dim` | 128 | Token dimensionality |
| `model.num_heads` | 4 | Attention heads (`embed_dim` must be divisible) |
| `model.num_layers` | 4 | Total encoder depth (unified self-attention over all N+1 tokens) |
| `model.fourier_dim` | 64 | `GaussianFourierEmbedding` output dim (must be even) |
| `model.fourier_scale` | 1.0 | Std dev of frequency matrix; log-uniformly tuned in HP search [0.1, 10.0] |
| `trainer.loss` | `squared_emd` | Training objective from `training/losses.py` LOSSES registry (`cross_entropy` or `squared_emd`); `val/cross_entropy` is always reported separately for cross-run comparability |
| `trainer.loss_kwargs` | `{n_classes: 11}` | Forwarded to the loss factory (e.g. `n_classes` for `squared_emd`) |
| `trainer.steps_per_epoch` | 500 | Random TC-sampling mode: gradient steps per epoch. Omit/`null` = sequential mode (one pass over TC data) |
| `trainer.profile` | `false` | Trace the first `profile_steps` training steps (JAX profiler) → `<run_dir>/logs/profile`; WandB uploads it as an artifact, TensorBoard/Null leave it on disk |
| `trainer.attn_fig_every_n_epochs` | 5 | Epoch cadence for `val/attn_map` + `val/attn_grid` figures (VAL probe batch); 0 = disabled |
| `trainer.grad_hist_every_n_epochs` | 5 | Epoch cadence for `grad_flow/*` gradient histograms (TRAIN probe batch, also at init); 0 = disabled |
| `trainer.patience_metric` | `val/cross_entropy` | |
| `trainer.run_dir` | `runs/tc_classifier/run_01` | Change per run to avoid overwriting |
| `trainer.log_backend` | `wandb` | `wandb` / `tensorboard` / `null` |

For WandB: set `log_backend: wandb`, add `project` / `name` / `tags` under `log_kwargs`, and export `WANDB_API_KEY` as an environment variable (never in the config file).

---

## Implementation notes

**Single `token_proj` (Senseiver-style):** one `Dense((2F+K) → embed_dim)` projects every token from the concatenation `[obs; mask; Fourier(position)]`. Observations, missingness, and position therefore share one learned map and live in the same coordinate space — the right inductive bias for relative geometry, and one fewer projection than the old separate obs/position layers (which were provably equivalent up to a redundant bias). The query token uses the same layer.

**`query_obs_slots` param (ξ):** a learned `(2F,)` vector (or `(F,)` when `missingness_indicator=False`) occupying the obs/mask slots of the query's projection input; `Fourier(query_coords)` supplies the position slots. For unit_circle the position is constant at `(0,0)`, so the query token reduces to a fixed learned vector — the standard ViT/BERT special-token role. For domain the query position varies and flows through the same `token_proj`.

**Attention observability** (`train.py`): a fixed validation probe batch is held in memory for the duration of training. `return_weights=True` returns the full attention matrices from EVERY encoder layer, shape `(num_layers, B, H, N+1, N+1)`. The entropy callback computes mean entropy over the last layer's query row and logs it as `val/attn_entropy` (step cadence). Every `attn_fig_every_n_epochs` epochs two figures are logged from the same probe: `val/attn_map` (geographic query-row scatter — the query self-attention weight is dropped before the station mask is applied) and `val/attn_grid` (layers × heads grid of full matrices). The static `val/attn_mask` figure is logged once at step 0. Attention figures are VAL/TEST diagnostics — `evaluate.py` produces the same map + grid + mask figures for the test split; nothing attention-related runs on training batches.

**Gradient-flow callback** (`train.py`): a fixed TRAINING probe batch; `jax.grad` of the cross-entropy loss, one histogram per parameter leaf named by its tree path (`grad_flow/encoder/blocks_0/...`), pushed via `logger.log_histogram` at init (step 0) and every `grad_hist_every_n_epochs` epochs. Train-only — vanishing/exploding layers show up as histograms collapsing or blowing up across depth.

**Multi-storm exclusion:** IBTrACS timestamps with ≥2 active storms are not used during training or validation — the model sees only unambiguous single-storm or background samples. These timestamps form the `hard_test` split for post-training analysis.

**Synoptic background pool:** background timestamps are restricted to the 3-hourly synoptic grid (00/03/…/21 UTC, exact) that best-track rows sit on, so time-of-day can never become a class shortcut. A handful of off-grid best-track special rows (landfall/peak inserts) remain on the TC side — known, accepted asymmetry.

**Sample metadata:** every batch carries a `meta` entry alongside `X`/`y` — SID (None for background), ISO time, raw query lat/lon, and station counts (`n_available` post-dedup candidates, `n_used` after the `max_stations` cap). It is never part of the model inputs (the `Trainer` drops it before its jitted steps); `evaluate.py` uses it for per-storm attribution and `TCDataModule.summary()` reports station-count diagnostics from it.
