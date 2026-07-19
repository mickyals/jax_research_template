# Experiment: tc_perceiver_io — TC Intensity Classifier

**Goal:** Given sparse in-situ land surface observations within a fixed radius of a query position at time *t*, predict the degree of tropical-cyclone organisation of any system present, or classify the sample as background (no storm).

This is a binary + ordinal classification problem over **9 classes**, ordered by degree of organisation. The class is STATUS-driven (the agency `USA_STATUS` code sets it); `USA_SSHS` supplies only the hurricane category number. Off-axis systems (extratropical, post-tropical, dissipating, etc.) are excluded from training. The canonical label space lives in `data/sources/ibtracs.py` (`CLASS_NAMES`, `status_sshs_to_class`).

| Class | Name | USA_STATUS | Wind |
|------:|------|------------|------|
| 0 | Background | — | no coherent system |
| 1 | Disturbance | DB / LO / WV / MD | — |
| 2 | Depression | TD / SD | < 35 kt |
| 3 | Storm | TS / SS | 35–64 kt |
| 4–8 | Category 1–5 | HU / TY / ST / … | Saffir-Simpson (from SSHS) |

---

## Data sources

**IBTrACS** (`ibtracs_full.npz`) — storm centre position, timestamp, `USA_STATUS`, and `USA_SSHS` for every 6-hourly TC observation in the training domain. 10,191 rows across all cyclone types.

**InsituLand** (`insitu_land_clean.npz` + `insitu_land_station_meta.npz`) — land surface hourly observations from Copernicus C3S for 552 stations in the Caribbean / Gulf domain (LAT 0–30°N, LON 100–45°W). 74.7M observation rows.

> **Faster startup (automatic).** Loading the 19.5 GB `insitu_land_clean.npz` and sorting its 74.7M rows by time is an ~8-minute cost (it's what sits between the `run_dir` line and the data summary). With `data.cache_sorted_obs: true` (default), the **first** run takes that hit once and writes a sibling `<stem>_sorted/` directory of pre-sorted, memory-mappable columns; **every subsequent run auto-detects and `mmap`s it, starting in seconds** — no config change needed. Set `cache_sorted_obs: false` to disable (e.g. tight disk). You can also build the cache ahead of time:
> ```bash
> PYTHONPATH=jrt python -m experiments.tc_perceiver_io.data.sources.insitu_land \
>     E:/sparse_obs/insitu-land/insitu_land_clean.npz \
>     E:/sparse_obs/insitu-land/insitu_land_clean_sorted
> ```
> `InsituLandDataset` accepts either the `.npz` or a converted directory and falls back to the slow load+sort path when no cache exists.

> **GPU selection.** On a multi-GPU box, JAX otherwise claims *every* visible device and preallocates memory on each. Pin training to one GPU with the top-level `gpu:` config field (or `--gpu N`, or a shell `CUDA_VISIBLE_DEVICES`, which wins) — it sets `CUDA_VISIBLE_DEVICES` before JAX initialises.

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

**Splits** (by ISO_TIME calendar year; default `year` strategy):
- Train: years 2005–2022
- Val: 2023–2024
- Test: 2025
- Hard test: multi-storm timestamps (870 times when ≥2 storms were active simultaneously) — held out entirely

**Batching:** the TC share of each batch is `data.tc_fraction` — a single number for all splits, or a `{train, val, test}` mapping (the default config uses a low train fraction with balanced `0.5` val/test for honest, comparable eval metrics). The rest of the batch is background samples (a domain point clear of any storm). All stations within `radius_km` are candidates; when a sample has more than `max_stations`, `station_selection` decides which are kept — `nearest` (deterministic; always used for val/test) or `random` (a train-only view augmentation that only bites when the cap binds). Train backgrounds are fresh draws each step; val/test loaders use ONE frozen background set (Latin-Hypercube positions + fixed-seed synoptic timestamps) reused every epoch, so eval differences are purely model change. Sequential (eval) epochs yield every valid TC sample — the final partial batch is flushed with proportionally fewer backgrounds. Background cleanliness is governed by `data.background_sampling` (see the Data-pipeline notes below).

---

## Architecture

`TCPerceiverIO` is a Perceiver IO (Jaegle et al. 2021/2022) over sparse station observations, built as three explicit stages composed by one top module:

- **Read** — a learned latent array `(N, D)` cross-attends the `M` station tokens (latents = queries, stations = keys/values), compressing the variable, padded station set into `N` fixed latents.
- **Processor** — `L` self-attention blocks over the `N` latents (depth decoupled from `M`; no re-read — that is the original Perceiver, pointless at small `M`).
- **Decoder** — maps latents → logits. Two tracks (Fig. 6): `attention` (a single learned output query cross-attends the latents, then value-proj + MLP — the Perceiver IO default, more expressive) or `avgproj` (mean-pool the latents → one linear).

Each stage wraps the GPT-2-style pre-LN residual block (Perceiver IO eqs 4–6: `LayerNorm → attention → +residual → LayerNorm → MLP → +residual`) around a self/cross attention primitive pulled from the vendored `train/legacy_attention.py` registry (`get_attention`; moved out of `core.attention` 2026-07-17 when core went to the single v3 primitive). A trailing `LayerNorm` after the Processor produces the normalised latent representation — the encoder asset. There is **no CLS or query token**; the learned latent array is the encode query.

### Token construction

A station token is one linear projection (`token_proj`) over the concatenation of observation values, the missingness mask, and the Fourier-embedded position — projected jointly, once:

```
station_input  = concat([obs_zeroed, obs_mask, Fourier(station_coords)])   # (B, M, 2F+K)
station_tokens = token_proj(station_input)                                 # (B, M, D)
```

Missing observations are filled with 0; the concatenated mask channel — not the fill value — disambiguates "observed 0" from "absent". No sentinel needed. `model.missingness_indicator: false` drops the mask channel (missing == observed-zero — ablation only).

### Latent array and the encode mask

The latent array is a learned parameter `(N, D)` (truncated-normal init, mean 0 / std 0.02), broadcast across the batch as the Read query. Only **Read** needs a mask — its keys/values are the variable-count, padded stations, so padded columns (`station_mask = False`) are blocked. The Processor and Decoder operate over the fixed `N` latents and need no mask.

### Location encoding modes

Set in `data.location_encoding` only — the model is coordinate-agnostic and projects whatever coords it is handed:

| Mode | `station_coords` |
|------|-----------------|
| `unit_circle` | `[x, y]` storm-centred local map (`norm_dist·sin(bearing), norm_dist·cos(bearing)`), each in [-1, 1] |
| `domain` | `[norm_lat_rad, norm_lon_rad]` absolute |

`unit_circle` is rotation-equivariant (the storm is always "at the origin"); `domain` retains absolute geographic position. The model no longer uses `query_coords` — the learned latent array replaces any positioned query token.

### Missing observations

Stations within the radius may have missing values for some variables. Missing entries are flagged by `obs_mask` and filled with 0 inside the model. `model.missingness_indicator: true` (default) concatenates `obs_mask` (as 0./1.) as its own channel in the `token_proj` input, so the mask's learned weights separate a missing feature from a real observation equal to 0 — the fill value is irrelevant. Set `false` to drop the channel (ablation only).

### Full forward pass

```
1.  obs_zeroed     = where(obs_mask, station_obs, 0)                          (B, M, F)
2.  station_input  = concat([obs_zeroed, obs_mask, Fourier(station_coords)])  (B, M, 2F+K)
    station_tokens = token_proj(station_input)                               (B, M, D)
3.  latents        = broadcast(latent_array, B)                              (B, N, D)
4.  z = Read(latents, station_tokens, mask=station padding)                  (B, N, D)   cross-attention
5.  z = Processor(z)        # L latent self-attention blocks                 (B, N, D)
6.  z = LayerNorm(z)        # encoder asset (headless → mean-pool → returned) (B, N, D)
7.  logits = Decoder(z)     # optional head (n_classes set)                  (B, C)
```

Headless (`n_classes=None`): no Decoder is built and `__call__` returns the mean-pooled normalised latents `z` (B, D) — the representation a linear probe reads.

`return_weights=True` additionally returns a dict of **pre-softmax** attention maps for val/test diagnostics: `{'read': (B, H, N, M), 'processor': (L, B, H, N, N), 'decoder': (B, H, 1, N)}`.

### Encoder / head split (frozen-encoder probing)

The transferable encoder is everything that produces the normalised latents — the latent array, tokenization, Read, Processor, and the trailing LayerNorm. The **Decoder** is the only swappable head, and the seam is the **`decoder` key** in the param pytree:

| Param leaves | Role |
|--------------|------|
| `latents`, `token_proj`, `read`, `processor`, `norm` | the trained, transferable encoder asset (every non-`decoder` leaf; `coord_embedding` is a fixed Fourier projection with no params) |
| `decoder` | target-specific head (swappable; omitted when headless) |

A headless `TCPerceiverIO`'s param tree is exactly the encoder leaves. Three helpers in `model.py` make transfer a few operations:

- `split_encoder_head(params)` → `(encoder_params, head_params)` — slices on the `decoder` key
- `attach_encoder(fresh_params, encoder_params)` → load a trained encoder into a freshly-initialised model that carries a new decoder
- `encoder_freeze_labels(params)` → an optax `multi_transform` label tree (`'frozen'` for every encoder leaf / `'trainable'` for `decoder`) to freeze the encoder during probe training

The Decoder reads the normalised latents (the final norm lives in the encoder), so a probe measures linear separability of the frozen representation. *(The train-time wiring to load an encoder checkpoint and freeze it is added alongside the first probe run — it needs a trained encoder to exist first.)*

---

## File layout

```
tc_perceiver_io/
├── configs/
│   ├── train.yaml   Full training config
│   ├── tune.yaml         Short-epoch config for HP search
│   └── schema.json          JSON schema for config validation
├── runs/                Training run artifacts (checkpoints, logs, figures)
├── baselines/           Baseline models for comparison
├── data/
│   ├── dataset.py       TCDataset — joins IBTrACS + InsituLand per sample,
│   │                       orchestrates assembly (delegates the swappable
│   │                       input transforms to the InputSpec)
│   ├── datamodule.py    TCDataModule + TCLoader — balanced TC/background
│   │                       batches, nearest-N station cap,
│   │                       frozen LHS eval backgrounds, partial-batch flush;
│   │                       exposes .input_spec / .target_spec
│   ├── inputs.py        InputSpec + resolve_input — declarative input config
│   │                       (obs_vars, normalisation, coordinate encoding, FOV);
│   │                       config-built, mirrors targets.py
│   ├── targets.py       TargetSpec + resolve_target — declarative prediction
│   │                       target (label builder, head size, loss, class names)
│   ├── transforms/      swappable input transforms selected by InputSpec
│   │   ├── normalise.py    obs normalisers (minmax_01/minmax_11/standardise)
│   │   ├── encoding.py     coordinate encoders/decoders (unit_circle local
│   │   │                      x-y, domain FOV-normalised) — exact inverses
│   │   │                      shared by dataset + plotting
│   │   └── derived.py      derived obs variables (wind components)
│   ├── splits.py        resolve_splits — data.split config → per-split
│   │                       datasets + run manifest ('year' and 'year_random'
│   │                       strategies)
│   └── sources/
│       ├── ibtracs.py       IBTrACSDataset — filter primitives (years,
│       │                       SIDs, single/multi-storm), sid-meta validation,
│       │                       ordinal organisation labels (status_sshs_to_class)
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
    ├── model.py         TCPerceiverIO — Read / Processor / Decoder Perceiver IO
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
jrt/experiments/tc_perceiver_io/configs/*_local.yaml
```

### 3. Config

The only required edits before a first run are the five `data:` paths. Key flags to understand:

- `data.split` is required — resolved by `data/splits.py` into filtered datasets plus a run manifest written next to the checkpoints. Splits are by **ISO_TIME calendar year** (not the IBTrACS SEASON column), so the TC and insitu/background streams stay year-aligned. Two strategies: `year` (explicit disjoint year lists per split, validated — e.g. train 2005–2022, val 2023–2024, test 2025, `hard_test: multi_storm`) and `year_random` (`train_val.years` pooled + a disjoint `test.years`; the pooled rows are split into train/val by `val.fraction`/`val.seed` at the **row/timestep** level — adjacent points of one storm may fall in both train and val, a mild deliberate train↔val leakage; test stays the clean held-out years, and train/val share the train+val-year insitu stream; no `train` block — train is the random remainder)
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
CFG=jrt/experiments/tc_perceiver_io/configs/train.yaml

# Train                                  # Resume from latest checkpoint
python -m experiments.tc_perceiver_io.train.train $CFG
python -m experiments.tc_perceiver_io.train.train $CFG --resume

# Evaluate (best checkpoint)
python -m experiments.tc_perceiver_io.train.evaluate $CFG \
    --checkpoint_dir runs/tc_classifier/run_01/checkpoints \
    --output_dir    runs/tc_classifier/run_01/eval \
    --split test --n_attn_samples 4 --geo --no_show

# Hyperparameter search
python -m experiments.tc_perceiver_io.train.tune \
    jrt/experiments/tc_perceiver_io/configs/tune.yaml \
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
python -m experiments.tc_perceiver_io.train.train \
    jrt/experiments/tc_perceiver_io/configs/train.yaml

# Resume an interrupted run
python -m experiments.tc_perceiver_io.train.train \
    jrt/experiments/tc_perceiver_io/configs/train.yaml \
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
python -m experiments.tc_perceiver_io.train.evaluate \
    jrt/experiments/tc_perceiver_io/configs/train.yaml \
    --checkpoint_dir runs/tc_classifier/run_01/checkpoints \
    --output_dir runs/tc_classifier/run_01/eval

# Validation split
python -m experiments.tc_perceiver_io.train.evaluate \
    jrt/experiments/tc_perceiver_io/configs/train.yaml \
    --split val

# Save plots to disk without displaying
python -m experiments.tc_perceiver_io.train.evaluate \
    jrt/experiments/tc_perceiver_io/configs/train.yaml \
    --output_dir runs/tc_classifier/run_01/eval \
    --no_show
```

Outputs:
- Confusion matrix — row-normalised 11×11 and raw counts
- Per-class precision / recall / F1 bar chart
- Precision-recall curves — binary TC-vs-background (`plot_pr_curve`, with the no-skill base-rate line and the AP that matches `pr_auc`) and a per-class one-vs-rest overlay (`plot_pr_curves_per_class`, each curve's AP being its `mAP` term)
- Per-class exemplars — one test sample of each true class (Background → Cat 5) on the Read attention map, with a printed `true → pred ✓/✗` table (`--no_class_examples` to skip)
- Per-component attention maps (one set per sample, titled with storm attribution `"<SID> <NAME> — true: Cat 3, pred: TS"`, or `background`):
  - **Read map** (`plot_attention_geographic`) — *which stations the model attends to*. The Read cross-attention `softmax(attn['read'])` `(B, H, N, M)` is averaged over the N latents and H heads to a per-station weight `(M,)`, then scattered on the station geometry: storm-centred local x-y with km distance rings (unit_circle) or lat/lon (domain). With `--geo` (or `geo=True`) the scatter is drawn on a cartopy map with coastlines/borders: PlateCarree for domain, and for unit_circle an **AzimuthalEquidistant projection centred on the storm** — that projection's native coordinates are metres east/north of the centre, i.e. exactly the local x-y encoding × radius, so stations, km rings, and coastlines align by construction. Requires cartopy (optional dependency); default stays cartopy-free
  - **Processor grid** (`plot_attention_matrix_grid`) — layers × heads panels of the N×N latent self-attention matrices `softmax(attn['processor'])` `(L, B, H, N, N)` per sample (plain imshow, no per-latent labels)
  - **Decoder-query map** (`plot_decoder_query`) — heads × latents heatmap of the single output query's attention over the N latents `softmax(attn['decoder'])` `(B, H, 1, N)` (decode_mode `attention` only; absent for `avgproj`)

### Hyperparameter search (CLI)

```bash
# In-memory search (quick experiments, results lost on exit)
python -m experiments.tc_perceiver_io.train.tune \
    jrt/experiments/tc_perceiver_io/configs/tune.yaml \
    --n_trials 25

# Persistent search — resume by running the same command again
python -m experiments.tc_perceiver_io.train.tune \
    jrt/experiments/tc_perceiver_io/configs/tune.yaml \
    --n_trials 50 \
    --storage sqlite:///runs/tc_classifier/hp_search/study.db \
    --study_name tc_classifier_v1
```

After the study finishes, best parameters are printed and written to `runs/tc_classifier/hp_search/best_params.json`. Copy those values into `train.yaml` and re-train at full length.

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

with open('jrt/experiments/tc_perceiver_io/configs/train.yaml') as f:
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
from experiments.tc_perceiver_io.data.datamodule import TCDataModule
from experiments.tc_perceiver_io.train.metrics import build_metrics_fns
from experiments.tc_perceiver_io.train.model import TCPerceiverIO
from training.trainer import Trainer

dm = TCDataModule.from_config(config['data'])
model = TCPerceiverIO(**config['model'])
metrics_fns = build_metrics_fns()
trainer = Trainer(model, metrics_fns, config['trainer'])
train_loader = dm.train_loader(seed=seed, shuffle=True)
val_loader = dm.val_loader()

# Sanity-check one batch
batch = next(iter(train_loader))
print('station_obs: ', batch['X']['station_obs'].shape)  # (B, N, F)
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
    import jax.nn as jnn
    _, attn = model.apply({'params': state.params}, batch['X'],
                          train=False, return_weights=True)
    # Read entropy: each latent's attention over the M stations (softmax of the
    # pre-softmax Read scores), averaged over batch/heads/latents.
    p       = np.asarray(jnn.softmax(attn['read'], axis=-1))   # (B, H, N, M)
    entropy = float(-np.sum(p * np.log(p + 1e-12), axis=-1).mean())
    print(f'  epoch {epoch:3d}  val/attn_entropy = {entropy:.4f}')

best_state = trainer.fit(train_loader, val_loader,
                         epoch_callbacks=[attn_callback])
```

```python
# Cell 6 — evaluate on test split
from experiments.tc_perceiver_io.train.evaluate import (
  collect_predictions, confusion_matrix, per_class_metrics,
  per_storm_metrics, CLASS_NAMES,
)
from experiments.tc_perceiver_io.plotting.plotting import (
  plot_confusion_matrix, plot_class_metrics,
)
import matplotlib.pyplot as plt

variables = {'params': best_state.params}
preds, labels, logits, meta = collect_predictions(model, variables, dm.test_loader())

# Every prediction is attributable to a named storm (background → sid None)
storm_metrics = per_storm_metrics(preds, labels, meta['sid'])

cm = confusion_matrix(preds, labels)
pcm = per_class_metrics(cm)
plot_confusion_matrix(cm, CLASS_NAMES, normalize=True, title='Test — row-normalised')
plot_class_metrics(pcm, CLASS_NAMES)
plt.show()
```

```python
# Cell 7 — per-component attention figures
import jax.nn as jnn
from experiments.tc_perceiver_io.plotting.plotting import (
  plot_attention_geographic, plot_attention_matrix_grid, plot_decoder_query,
)
from experiments.tc_perceiver_io.train.evaluate import domain_latlon_for_sample

attn_batch = next(iter(val_loader))
_, attn = model.apply(variables, attn_batch['X'], train=False, return_weights=True)
# attn holds PRE-softmax scores; softmax each over its LAST axis to get
# distributions. Shapes: read (B,H,N,M), processor (L,B,H,N,N), decoder (B,H,1,N).
read = np.asarray(jnn.softmax(attn['read'],      axis=-1))
proc = np.asarray(jnn.softmax(attn['processor'], axis=-1))
dec  = (np.asarray(jnn.softmax(attn['decoder'], axis=-1))
        if attn.get('decoder') is not None else None)   # None for decode_mode='avgproj'

# Read map — which stations the latents attend to (mean over latents + heads).
# unit_circle needs nothing extra; domain mode requires the caller to decode
# coords→lat/lon (the plotter doesn't import the coordinate encoding) — use
# domain_latlon_for_sample(attn_batch, 0, fov_lat, fov_lon) and pass
# station_latlon=/query_latlon=.
fig_read = plot_attention_geographic(
  read, attn_batch,
  location_encoding=config['data']['location_encoding'],
  fov_lat=config['data'].get('fov_lat'),
  fov_lon=config['data'].get('fov_lon'),
  radius_km=config['data'].get('radius_km', 500.0),
  sample_idx=0,
)

# Processor — layers × heads grid of the N×N latent self-attention matrices
fig_grid = plot_attention_matrix_grid(proc, sample_idx=0)

# Decoder — heads × latents output-query attention (skip when None / avgproj)
if dec is not None:
  fig_dec = plot_decoder_query(dec, sample_idx=0)
plt.show()
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Training objective from `trainer.loss` (e.g. `cross_entropy`, optionally focal / class-weighted) |
| `val/loss` | Same objective evaluated on val |
| `val/cross_entropy` | Validation CE — always reported for cross-run comparability; patience metric for early stopping |
| `val/accuracy` | Top-1 accuracy over all 9 classes |
| `val/binary_accuracy` | TC vs no-TC (class 0 vs class > 0). **Read against the eval background base rate.** With the default balanced val (`tc_fraction.val = 0.5`) the base rate is ~0.5, so an "always background" model scores ≈0.5; set an imbalanced eval fraction and the base rate rises to ~(1 − tc_fraction) (≈0.90 at 0.1). Use `pr_auc` / the confusion matrix to judge real detection. |
| `val/mae_class` | Mean \|predicted class − true class\| in class units |
| `val/mAP` | Macro one-vs-rest mean average precision (full val set, every `eval_plots_every_n_epochs`) — imbalance-robust; surfaces rare classes (Cat 4/5) that accuracy hides |
| `val/pr_auc` | Binary TC-vs-background detection average precision / PR-AUC (full val set) — the right detection summary under heavy imbalance (ROC/AUC flatters when negatives dominate). The PR **curve** behind this scalar is logged as the `val/pr_curve` figure (and `val/pr_curves_per_class` for the one-vs-rest curves behind `mAP`) |
| `val/attn_entropy` | Mean entropy of the Read cross-attention — each latent's distribution over the M stations (`softmax(attn['read'])`, averaged over batch/heads/latents). A falling curve means latents are concentrating on fewer stations rather than attending uniformly. |

**Interpretation:**
- **`binary_accuracy` is base-rate-dependent.** A model that always predicts class 0 scores `binary_accuracy ≈ (1 − eval tc_fraction)` — ≈0.5 with the default balanced val/test (`tc_fraction 0.5`, which is exactly why eval is balanced), but ≈0.90 if you eval at `tc_fraction = 0.1` — while detecting nothing. So check it against the eval background fraction, and lean on `pr_auc`, the confusion matrix (is the TC row bleeding into class 0?), and `mae_class` for real signal.
- `val/mAP` and `val/pr_auc` are FULL-SET metrics (computed over the accumulated val predictions in `evaluate.py` / the eval-plots callback, via `train/full_set_metrics.py` (moved from jrt 2026-07-05)), not per-batch — they integrate a precision-recall curve over the whole split, so they cannot live in the per-batch `metrics_fns`. They live in the separate `FULL_SET_METRICS` registry. `mAP` is the imbalance-robust multiclass headline; `pr_auc` is the TC-vs-background detection scalar.
- `val/attn_entropy` starts high (latents attend near-uniformly over the M stations) and is expected to fall as the model learns which stations matter; padded station columns are masked out of the Read attention so they do not contribute.

---

## Config reference

Key fields in `train.yaml`:

| Key | Default | Notes |
|-----|---------|-------|
| `seed` | 3678 | Single seed for model init, dropout, and data shuffle |
| `data.radius_km` | 500 | Stations outside this radius are excluded |
| `data.time_window_hours` | 0.1 | Temporal tolerance (±) for matching obs to the query time; each station contributes only its report nearest in time |
| `data.max_stations` | 64 | Cap on station tokens; all within radius are used, more are nearest-trimmed, fewer are zero-padded |
| `data.min_stations` | 1 | Samples with fewer stations are dropped |
| `data.location_encoding` | `unit_circle` | `unit_circle` or `domain`; model is coordinate-agnostic, so this lives in the data block only |
| `data.obs_normalisation` | `minmax_11` | `minmax_01` / `minmax_11` / `standardise` |
| `data.tc_fraction` | `0.5` | TC share of each batch. A single number (all splits) or a `{train, val, test}` mapping — typically a low train fraction with balanced `0.5` val/test for honest eval metrics; a split absent from the mapping defaults to `0.5` |
| `data.station_selection` | `nearest` | When a sample exceeds `max_stations`: `nearest` (deterministic; always val/test) or `random` (train-only view augmentation; only bites when the cap binds) |
| `data.class_weight_scheme` | `none` | `none` / `inverse_freq` / `sqrt_inverse_freq` / `effective_number` / `median_freq` — computes `class_weights` at setup from train-split counts incl. **background (label 0, count = `n_background`)**, stored in manifest; overridden by an explicit `trainer.loss_kwargs.class_weights`. When active, the **train** loader drops `tc_fraction` oversampling and samples at natural prevalence so the two correctors don't stack (val/test keep `tc_fraction`); `effective_number` recommended so the ~millions:10K ratio doesn't blow up rare-class weights |
| `data.class_weight_beta` | `0.999` | effective-number β (that scheme only); higher = wider common/rare spread |
| `data.class_weight_normalize` | `true` | Scale present-class weights to mean 1.0. Cosmetic for training (`cross_entropy_loss` reduces by the weight sum); `false` records the raw scheme values in the manifest |
| `data.n_background` | `null` | Effective class-0 (background) **population** size folded into the weights so they span class 0..8 over the whole train set. Background is synthesised, so this is a hyperparameter; `null` → realized per-epoch count (`steps_per_epoch × bg_half`) |
| `data.bg_refresh_every` | `1` | Random-mode train loader: steps between background-buffer refreshes. `1` = assemble fresh backgrounds every step. Larger values reuse pre-assembled backgrounds to cut per-step assembly cost when batches are background-heavy (e.g. natural prevalence) — assembly drops to ~`bg_buffer_size / bg_refresh_every` draws/step |
| `data.bg_buffer_size` | `null` | Size of the reusable background buffer each step samples its backgrounds from. `null` = background count per batch (floored there). Pair a larger buffer with `bg_refresh_every` to retain diversity while reusing draws |
| `data.background_sampling` | `time` | `time` = the pool excludes any synoptic timestamp within `background_buffer_hours` of ANY active TC basin-wide (shrinks in peak season). `spatial` = keep every grid timestamp and accept a draw iff no storm is within `background_exclusion_radius_km` at ±`background_buffer_hours` (decouples space; gates train draws AND the frozen eval set) |
| `data.background_exclusion_radius_km` | `null` | `spatial` mode: storm-free radius required for a background draw. `null` → `radius_km` (strictest — no storm-affected station can enter the sample); smaller keeps more background but admits storm-adjacent stations |
| `data.background_buffer_hours` | `6.0` | `time` mode: exclusion half-window around any TC obs. `spatial` mode: time tolerance for the storm-proximity check |
| `model.missingness_indicator` | `true` | `true` = concatenate `obs_mask` as its own channel in `token_proj` (missing obs filled 0), disambiguating "missing" from a real obs equal to 0; `false` = aliased behaviour (ablation) |
| `model.embed_dim` | 128 | Latent + token dimensionality D |
| `model.num_heads` | 4 | Attention heads (`embed_dim` must be divisible) |
| `model.num_latents` | 16 | N — number of learned latent vectors (the index dim the model processes) |
| `model.num_process_layers` | 2 | L — latent self-attention blocks in the Processor |
| `model.processor_weight_sharing` | `false` | `true` = one Processor block applied L times (recurrent / cross-layer weight tying, Senseiver-style) — same depth, one block's params |
| `model.decode_mode` | `attention` | Decoder track: `attention` (single learned query) or `avgproj` (mean + linear) |
| `model.fourier_dim` | 64 | `GaussianFourierEmbedding` output dim (must be even) |
| `model.fourier_scale` | 1.0 | Std dev of frequency matrix; log-uniformly tuned in HP search [0.1, 10.0] |
| `trainer.loss` | `cross_entropy` | Training objective from `training/losses.py` LOSSES registry; `val/cross_entropy` is always reported separately for cross-run comparability |
| `trainer.loss_kwargs` | `{}` | Composable kwargs for `cross_entropy`: `focal_gamma` (focal loss), `emd_lambda`/`emd_omega`/`emd_mu` (squared-EMD regulariser), and/or explicit `class_weights` (length-9, index 0 = background; overrides `data.class_weight_scheme`) |
| `trainer.steps_per_epoch` | 500 | Random TC-sampling mode: gradient steps per epoch. Omit/`null` = sequential mode (one pass over TC data) |
| `trainer.profile` | `false` | Trace the first `profile_steps` training steps (JAX profiler) → `<run_dir>/logs/profile`; WandB uploads it as an artifact, TensorBoard/Null leave it on disk |
| `trainer.attn_fig_every_n_epochs` | 5 | Epoch cadence for the per-component attention figures `val/attn_read_map` + `val/attn_processor_grid` + `val/attn_decoder_query` (VAL probe batch); 0 = disabled |
| `trainer.grad_hist_every_n_epochs` | 5 | Epoch cadence for `grad_flow/*` gradient histograms (TRAIN probe batch, also at init); 0 = disabled |
| `trainer.patience_metric` | `val/cross_entropy` | |
| `trainer.run_group` | `runs/tc_classifier` | Parent dir; `train.py` auto-creates the next `run_NN` under it (no clobbering). `--name <slug>` appends the run's purpose → `run_NN-<slug>` and sets the WandB run name |
| `trainer.run_dir` | _(unset)_ | Pin a fixed run directory instead of auto-incrementing (takes precedence over `run_group`) |
| `trainer.log_backend` | `wandb` | `wandb` / `tensorboard` / `null` |

For WandB: set `log_backend: wandb`, add `project` / `name` / `tags` under `log_kwargs`, and export `WANDB_API_KEY` as an environment variable (never in the config file).

---

## Implementation notes

**Single `token_proj`:** one `Dense((2F+K) → embed_dim)` projects every station token from the concatenation `[obs; mask; Fourier(coords)]`. Observations, missingness, and position therefore share one learned map and live in the same coordinate space — the right inductive bias for relative geometry. There is no query token: the model is coordinate-agnostic and the learned latent array is the encode query (`location_encoding` configures the datamodule only).

**Learned latent array (ξ):** a `(num_latents, embed_dim)` parameter (truncated-normal init), broadcast over the batch and used as the Read cross-attention query. It replaces any explicit query/CLS token — the index dimension the Processor and Decoder operate over is the N latents, decoupled from the station count M.

**Attention observability** (`train.py`): a fixed validation probe batch is held in memory for the duration of training. `return_weights=True` returns a dict of PRE-softmax scores, one per component: `read (B, H, N, M)`, `processor (L, B, H, N, N)`, `decoder (B, H, 1, N)` (None for `decode_mode='avgproj'`). Softmax over the last axis turns any of these into distributions. The entropy callback logs the Read entropy as `val/attn_entropy` (step cadence). Every `attn_fig_every_n_epochs` epochs three figures are logged from the same probe: `val/attn_read_map` (geographic Read scatter, per-station mean over latents+heads), `val/attn_processor_grid` (layers × heads grid of the N×N latent self-attention), and `val/attn_decoder_query` (heads × latents output-query heatmap, skipped for `avgproj`). Attention figures are VAL/TEST diagnostics — `evaluate.py` produces the same three figures per sample for the test split; nothing attention-related runs on training batches.

**Gradient-flow callback** (`train.py`): a fixed TRAINING probe batch; `jax.grad` of the cross-entropy loss, gradient histograms pushed via `logger.log_histogram` at init (step 0) and every `grad_hist_every_n_epochs` epochs. By default (`final_layers_only=True`) only each stage's OUTPUT layer is logged — Read's FFN output (`grad_flow/read/mlp/output_layer/...`), the last Processor block's FFN output (`grad_flow/processor/blocks_<L-1>/mlp/output_layer/...`), and the Decoder head (`grad_flow/decoder/head/...`) — enough to read flow across Read→Process→Decode without a ~70-leaf dump; set `final_layers_only=False` for every leaf. Train-only — vanishing/exploding stages show up as histograms collapsing or blowing up.

**Diagnostics distribution figures** (`utils/jax_core/diagnostics._plot_dists`): the weight / gradient / activation histograms wrap into a grid of at most 4 panels per row (`max_cols=4`) rather than one long horizontal strip, so a deep model stays reviewable.

**Multi-storm exclusion:** IBTrACS timestamps with ≥2 active storms are not used during training or validation — the model sees only unambiguous single-storm or background samples. These timestamps form the `hard_test` split for post-training analysis.

**Synoptic background pool:** background timestamps are restricted to the 3-hourly synoptic grid (00/03/…/21 UTC, exact) that best-track rows sit on, so time-of-day can never become a class shortcut. A handful of off-grid best-track special rows (landfall/peak inserts) remain on the TC side — known, accepted asymmetry.

**Background cleanliness (`data.background_sampling`):** how a `(point, time)` draw is judged storm-free. `time` (default) bakes a basin-wide exclusion into the pool — any synoptic timestamp within `background_buffer_hours` of *any* active TC is dropped; simple, but in peak season (a storm almost always active *somewhere*) the pool shrinks sharply. `spatial` keeps every grid timestamp and instead validates each draw geographically: a draw is background iff no storm is within `background_exclusion_radius_km` (null → `radius_km`) at ±`background_buffer_hours`, via a time-sorted IBTrACS proximity index on `TCDataset` (binary-search window + haversine — cheap, no 19 GB load). This decouples space from time so a point far from every active storm stays valid mid-season. Spatial validation gates both the random train draws and the frozen eval set (which over-draws to absorb rejections).

**Sample metadata:** every batch carries a `meta` entry alongside `X`/`y` — SID (None for background), ISO time, raw query lat/lon, and station counts (`n_available` post-dedup candidates, `n_used` after the `max_stations` cap). It is never part of the model inputs (the `Trainer` drops it before its jitted steps); `evaluate.py` uses it for per-storm attribution and `TCDataModule.summary()` reports station-count diagnostics from it.
