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
- `wind_speed` (m/s)
- `wind_from_direction` (°)

**Splits:**
- Train: IBTrACS seasons 2005–2020
- Val: 2021–2022
- Test: 2023–2025
- Hard test: multi-storm timestamps (870 times when ≥2 storms were active simultaneously) — held out entirely

**Batching:** each batch is 1:1 balanced — half TC samples (storm centre as query), half background samples (random domain point during non-TC periods).

---

## Architecture

`TCClassifier` is a single `TransformerEncoder` over N+1 tokens: N station tokens followed by one query/CLS token. An asymmetric attention mask separates the two roles — stations contextualise each other without seeing the query; the query then reads the fully-contextualised station network and drives the classification head.

### Token construction

**Station tokens** — observation content and spatial position are encoded additively:

```
station_tokens = station_proj(obs_fixed)            # (B, N, D)  content
               + pos_proj(Fourier(station_coords))  # (B, N, D)  position
```

`obs_fixed` substitutes a learned mask token (or constant sentinel) for any missing observations. `pos_proj` is a shared Dense layer used for both station and query position encoding.

**Query token** — a single learned content vector present in both modes; domain mode adds a positional component via the same shared `pos_proj`:

| Mode | Content | Position |
|------|---------|---------|
| `unit_circle` | `learned_query` param | none (Fourier([0,0]) is degenerate) |
| `domain` | `learned_query` param | `pos_proj(Fourier(query_coords))` — shared `pos_proj` |

The `learned_query` vector captures *what it means to be the query*; position is layered on top for domain mode only. There is no separate projection layer for the query.

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

### Location encoding modes

Set in both `data.location_encoding` and `model.location_encoding` (they must match):

| Mode | `station_coords` | `query_coords` |
|------|-----------------|----------------|
| `unit_circle` | `[norm_distance, bearing_rad]` relative to storm | `[0, 0]` sentinel; query uses `learned_query` only |
| `domain` | `[norm_lat_rad, norm_lon_rad]` absolute | same encoding; adds `pos_proj(Fourier(...))` to query |

`unit_circle` is rotation-equivariant (the storm is always "at the origin"). `domain` retains absolute geographic position.

### Missing observations

Stations within the radius may have missing values for some variables. Missing entries are flagged by `obs_mask`; inside the model they are replaced by one of two strategies:

| `use_learned_mask` | Behaviour |
|--------------------|-----------|
| `true` (default) | A trainable `(F,)` parameter — one value per obs feature — initialised with `normal(0.02)`. The optimizer learns what "absent" should look like for each variable. `missing_value` is ignored. |
| `false` | Missing obs are replaced with the fixed scalar `missing_value` (default −10.0) at every forward pass. The value never changes. |

−10.0 is chosen to be clearly outside the normalised obs range ([-1, 1] for minmax_11) without being extreme enough to cause gradient issues.

### Full forward pass

```
1.  obs_fixed  = where(obs_mask, station_obs, mask_token)                   (B, N, F)
2.  station_tokens = station_proj(obs_fixed)
                   + pos_proj(Fourier(station_coords))                       (B, N, D)
3.  query_token = learned_query [+ pos_proj(Fourier(query_coords)) if domain]  (B, 1, D)
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
│   │                       batches, re-iterable with per-epoch seed
│   └── sources/
│       ├── ibtracs.py       IBTrACSDataset — season splits, multi-storm
│       │                       filtering, SSHS label mapping
│       └── insitu_land.py   InsituLandDataset — haversine spatial filter,
│                                reliability filtering, binary-search time queries
├── plotting/
│   └── plotting.py      Confusion matrix and class metrics (thin wrappers
│                           over jrt/utils/plotting fields.plot_heatmap and
│                           curves.plot_grouped_bars), and geographic
│                           attention plots (unit_circle: bespoke polar
│                           scatter via _value_scatter; domain: thin wrapper
│                           over fields.plot_scatter_overlay)
└── train/
    ├── model.py         TCClassifier — unified Transformer encoder
    ├── metrics.py       cross_entropy, accuracy, binary_accuracy, mae_class
    ├── train.py         CLI entry point + attention entropy epoch callback
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

The four data files are not in the repository. Place them on the target machine and update the `data:` paths in the config. Recommended layout:

```
/data/sparse_obs/
    ibtracs/
        ibtracs_full.npz
        ibtracs_multi_storm_times.npz
    insitu-land/
        insitu_land_clean.npz
        insitu_land_station_meta.npz
```

To keep machine-specific paths out of version control, create a `configs/tc_classifier_local.yaml` (gitignored) and load it instead, or patch paths programmatically in a notebook (see below).

```
# .gitignore
jrt/experiments/sparse_obs_cross_attn/configs/*_local.yaml
```

### 3. Config

The only required edits before a first run are the four `data:` paths. Key flags to understand:

- `data.location_encoding` and `model.location_encoding` must match — `unit_circle` (default) or `domain`
- `model.use_learned_mask: true` (default) — learned mask token for missing obs; `false` for constant sentinel
- `trainer.run_dir` — change per run to avoid overwriting checkpoints

---

## Usage

### Training (CLI)

Run from the **repository root** so Python module imports resolve:

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
- Geographic attention maps — polar scatter (unit_circle) or lat/lon scatter (domain), coloured by mean attention weight

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
    w       = np.asarray(w)                          # (B, H, N+1)
    entropy = float(-np.sum(w * np.log(w + 1e-12), axis=-1).mean())
    print(f'  epoch {epoch:3d}  val/attn_entropy = {entropy:.4f}')

best_state = trainer.fit(train_loader, val_loader,
                         epoch_callbacks=[attn_callback])
```

```python
# Cell 6 — evaluate on test split
from experiments.sparse_obs_cross_attn.train.evaluate import (
    collect_predictions, confusion_matrix, per_class_metrics, CLASS_NAMES,
)
from experiments.sparse_obs_cross_attn.plotting.plotting import (
    plot_confusion_matrix, plot_class_metrics,
)
import matplotlib.pyplot as plt

variables = {'params': best_state.params}
preds, labels, logits = collect_predictions(model, variables, dm.test_loader())

cm  = confusion_matrix(preds, labels)
pcm = per_class_metrics(cm)
plot_confusion_matrix(cm, CLASS_NAMES, normalize=True, title='Test — row-normalised')
plot_class_metrics(pcm, CLASS_NAMES)
plt.show()
```

```python
# Cell 7 — geographic attention map
from experiments.sparse_obs_cross_attn.plotting.plotting import (
    extract_attention_weights, plot_attention_geographic,
)

attn_batch   = next(iter(val_loader))
attn_weights = extract_attention_weights(model, variables, attn_batch)
print('weights:', attn_weights.shape)   # (B, H, N+1)

fig = plot_attention_geographic(
    attn_weights, attn_batch,
    location_encoding = config['data']['location_encoding'],
    fov_lat           = config['data'].get('fov_lat'),
    fov_lon           = config['data'].get('fov_lon'),
    radius_km         = config['data'].get('radius_km', 500.0),
    sample_idx        = 0,
)
plt.show()
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| `train/cross_entropy` | Softmax CE over 11 classes — training loss |
| `val/cross_entropy` | Validation CE — patience metric for early stopping |
| `val/accuracy` | Top-1 accuracy over all 11 classes |
| `val/binary_accuracy` | TC vs no-TC (class 0 vs class > 0); random chance = 0.5 |
| `val/mae_class` | Mean \|predicted class − true class\| in class units |
| `val/attn_entropy` | Entropy of query-row attention weights over N+1 positions — logged per epoch. A falling curve means the model is concentrating attention on specific stations. |

**Interpretation:**
- A model that always predicts class 0 achieves `binary_accuracy = 0.5` but `mae_class ≈ 3`. Use `mae_class` as the primary signal for ordinal quality.
- `val/attn_entropy` includes the query's self-attention weight (the last of N+1 positions). A high self-attention weight early in training is expected — the model is relying on its `learned_query` prior. Expect it to decrease as the model learns to trust station data.

---

## Config reference

Key fields in `tc_classifier.yaml`:

| Key | Default | Notes |
|-----|---------|-------|
| `seed` | 3678 | Single seed for model init, dropout, and data shuffle |
| `data.radius_km` | 500 | Stations outside this radius are excluded |
| `data.time_window_hours` | 0.1 | Temporal window for matching obs to TC timestamp |
| `data.max_stations` | 64 | Padding / truncation limit |
| `data.min_stations` | 1 | Samples with fewer stations are dropped |
| `data.location_encoding` | `unit_circle` | `unit_circle` or `domain`; must match `model.location_encoding` |
| `data.obs_normalisation` | `minmax_11` | `minmax_01` / `minmax_11` / `standardise` |
| `model.location_encoding` | `unit_circle` | Must match `data.location_encoding` |
| `model.use_learned_mask` | `true` | `true` = trainable mask token, normal(0.02) init; `false` = fixed constant sentinel |
| `model.missing_value` | −10.0 | Only used when `use_learned_mask: false`. Fixed sentinel value for missing obs. |
| `model.embed_dim` | 128 | Token dimensionality |
| `model.num_heads` | 4 | Attention heads (`embed_dim` must be divisible) |
| `model.num_layers` | 4 | Total encoder depth (unified self-attention over all N+1 tokens) |
| `model.fourier_dim` | 64 | `GaussianFourierEmbedding` output dim (must be even) |
| `model.fourier_scale` | 1.0 | Std dev of frequency matrix; log-uniformly tuned in HP search [0.1, 10.0] |
| `trainer.patience_metric` | `val/cross_entropy` | |
| `trainer.run_dir` | `runs/tc_classifier/run_01` | Change per run to avoid overwriting |
| `trainer.log_backend` | `wandb` | `wandb` / `tensorboard` / `null` |

For WandB: set `log_backend: wandb`, add `project` / `name` / `tags` under `log_kwargs`, and export `WANDB_API_KEY` as an environment variable (never in the config file).

---

## Implementation notes

**Shared `pos_proj`:** the same `Dense(fourier_dim → embed_dim)` layer encodes position for station tokens and (domain mode) the query token. Station and query positions therefore live in the same learned coordinate space, which is the right inductive bias for interpreting relative geometry. There is no separate query projection layer.

**`learned_query` param:** a single `(embed_dim,)` vector initialised with `normal(0.02)` — the standard ViT/BERT special-token init. In unit_circle mode it is the entire query token (absolute position is degenerate at [0,0]). In domain mode it contributes content while `pos_proj` contributes location.

**Attention entropy callback** (`train.py`): a fixed validation probe batch is held in memory for the duration of training. After each validation epoch a JIT-compiled forward pass with `return_weights=True` computes mean entropy over all heads and all N+1 attention weights and logs it as `val/attn_entropy`. Every 5 epochs a geographic attention figure is also logged — the query self-attention weight is excluded from the map by slicing `weights[:, :, :N]` before applying the station mask.

**Multi-storm exclusion:** IBTrACS timestamps with ≥2 active storms are not used during training or validation — the model sees only unambiguous single-storm or background samples. These timestamps form the `hard_test` split for post-training analysis.
