# Experiment: Sparse Observation Cross-Attention TC Classifier

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

`TCClassifier` supports two attention paths controlled by a single config flag.

### Path A — Self-attention then cross-attention (`use_self_attention: true`)

```
Station obs (B, N, F) + Station coords (B, N, 2)
    → concat → Dense(embed_dim) → station tokens (B, N, E)
    → TransformerEncoder (num_layers blocks of Pre-LN self-attention + FFN)
    → contextualised station tokens (B, N, E)

Query token (B, 1, E)  [learned CLS-style token for unit_circle encoding;
                        Fourier-encoded position for domain encoding]
    → CrossAttentionBlock × num_cross_layers
        Q = query token, K = V = contextualised stations
        mask = station_mask (B, 1, N)
    → context vector (B, E)

LayerNorm → Dense(n_classes) → logits (B, 11)
```

### Path B — Direct cross-attention (`use_self_attention: false`)

```
Station coords (B, N, 2) → GaussianFourierEmbedding → K (B, N, fourier_dim)
Station obs    (B, N, F) → sentinel-replace missing  → V (B, N, F)

Query token (B, 1, E)  [same as Path A]
    → SeparateKVCrossAttentionBlock × num_cross_layers
        Q = query token
        K projected from coords (w_k: fourier_dim → embed_dim, per layer)
        V projected from obs    (w_v: F → embed_dim, per layer)
        mask = station_mask (B, N)
    → context vector (B, E)

LayerNorm → Dense(n_classes) → logits (B, 11)
```

Path B separates geometric context (K) from observation content (V), which is the key inductive bias of this experiment — can the model learn to attend to the right *locations* independently of what those locations are measuring?

### Location encoding modes

Two coordinate encoding modes, set in both `data.location_encoding` and `model.location_encoding` (must match):

| Mode | `station_coords` | `query_coords` |
|------|-----------------|----------------|
| `unit_circle` | `[norm_distance, bearing_rad]` relative to query | `[0, 0]` sentinel → model uses a learned token |
| `domain` | `[norm_lat_rad, norm_lon_rad]` absolute | `[norm_lat_rad, norm_lon_rad]` encoded same way |

`unit_circle` is rotation-equivariant (the storm is always "at the origin"); `domain` retains absolute geographic position.

### Missing observations

Stations within the radius may have missing values for some variables. Missing entries are zeroed in the datamodule (`obs_mask` tracks which are valid), then replaced with a large negative sentinel (`missing_value: -1e9`) inside the model so the network can distinguish "missing" from a genuine near-zero measurement. LayerNorm within each block normalises away the extreme magnitude.

---

## File layout

```
sparse_obs_cross_attn/
├── ibtracs.py          IBTrACSDataset — NpzDataset subclass, season splits,
│                           multi-storm filtering, SSHS label mapping
├── insitu_land.py      InsituLandDataset — two-file (obs + station meta),
│                           haversine spatial filter, reliability filtering,
│                           binary-search O(log N) time queries
├── dataset.py          TCDataset — joins IBTrACS + InsituLand per sample,
│                           coordinate encoding, obs normalisation
├── datamodule.py       TCDataModule + TCLoader — balanced TC/background batches,
│                           re-iterable with per-epoch seed
├── model.py            TCClassifier + SeparateKVCrossAttentionBlock
├── metrics.py          cross_entropy, accuracy, binary_accuracy, mae_class
│                           + build_metrics_fns() factory
├── train.py            CLI entry point
├── evaluate.py         Evaluation pipeline + geographic attention plots
├── tune.py             Optuna hyperparameter search entry point
├── baselines/          Baseline implementations for comparison
└── configs/
    ├── tc_classifier.yaml   Full training config
    ├── tc_tune.yaml         Short-epoch config for HP search
    └── schema.json          JSON schema
```

---

## Quickstart

### 1. Set data paths

Edit the four `data:` paths in the config:

```
src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml
```

### 2. Train

```bash
python -m experiments.sparse_obs_cross_attn.train \
    src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml

# Resume an interrupted run
python -m experiments.sparse_obs_cross_attn.train \
    src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
    --resume
```

### 3. Evaluate

```bash
python -m experiments.sparse_obs_cross_attn.evaluate \
    --checkpoint_dir runs/tc_classifier/run_01/checkpoints \
    --config src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
    --split val \
    --output_dir runs/tc_classifier/run_01/eval_figures
```

Outputs:
- **Confusion matrix** — row-normalised 11×11 with raw counts
- **Per-class P/R/F1 bar chart**
- **Geographic attention maps** — polar scatter (unit_circle mode) or lat/lon scatter (domain mode), coloured by mean attention weight over the last cross-attention layer

### 4. Hyperparameter search

```bash
python -m experiments.sparse_obs_cross_attn.tune \
    src/experiments/sparse_obs_cross_attn/configs/tc_tune.yaml \
    --n_trials 25 \
    --storage sqlite:///runs/tc_classifier/hp_search/study.db \
    --study_name tc_classifier_v1
```

After the study finishes, the best parameters are printed and written to `runs/tc_classifier/hp_search/best_params.json`. Copy those values into `tc_classifier.yaml` and re-train at full length.

---

## Metrics

| Metric | Description |
|--------|-------------|
| `train/cross_entropy` | Softmax CE over 11 classes — training loss |
| `val/cross_entropy` | Validation CE — patience metric for early stopping |
| `val/accuracy` | Top-1 accuracy over all 11 classes |
| `val/binary_accuracy` | TC vs no-TC (class 0 vs class > 0); random chance = 0.5 |
| `val/mae_class` | Mean \|predicted class − true class\| in class units |
| `val/attn_entropy` | Entropy of cross-attention weights — logged per epoch via callback |

**Interpretation:**
- A model that always predicts class 0 achieves `binary_accuracy = 0.5` and `accuracy ≈ 0.5` but `mae_class ≈ 3`. Use `mae_class` as the primary signal for ordinal quality.
- Falling `val/attn_entropy` means the model is learning to concentrate on specific stations rather than attending uniformly — a sign the attention is doing useful work.

---

## Config reference

Key knobs in `tc_classifier.yaml`:

| Key | Default | Notes |
|-----|---------|-------|
| `data.radius_km` | 500 | Stations outside this radius are excluded |
| `data.time_window_hours` | 0.1 | Temporal window for matching obs to TC timestamp |
| `data.max_stations` | 64 | Padding/truncation limit; larger = richer context but slower |
| `data.min_stations` | 1 | Samples with fewer stations are dropped |
| `data.location_encoding` | `unit_circle` | `unit_circle` or `domain` |
| `data.obs_normalisation` | `minmax_11` | `minmax_01` / `minmax_11` / `standardise` |
| `model.use_self_attention` | `true` | Path A (true) or Path B (false) |
| `model.embed_dim` | 128 | Token dimensionality |
| `model.num_heads` | 4 | Attention heads (embed_dim must be divisible) |
| `model.num_layers` | 2 | Self-attention depth (Path A only) |
| `model.num_cross_layers` | 1 | Cross-attention depth (both paths) |
| `model.fourier_dim` | 64 | GaussianFourierEmbedding output dim (must be even) |
| `trainer.patience_metric` | `val/cross_entropy` | |
| `trainer.run_dir` | `runs/tc_classifier/run_01` | Change per run to avoid overwriting |
| `trainer.log_backend` | `null` | Switch to `wandb` for remote tracking |

For WandB logging, set `log_backend: wandb` and `WANDB_API_KEY` as an environment variable, then add `project`, `name`, and optionally `tags` under `log_kwargs`.

---

## Implementation notes

**`SeparateKVCrossAttentionBlock`** is defined in `model.py` rather than `core/nets/transformers.py` because it is experiment-specific: it projects K and V from different input tensors (coordinates vs observations) rather than a single context sequence. If this pattern proves reusable it can be promoted to `core/`.

**Attention entropy callback** (`train.py`): a fixed validation probe batch is held in memory for the duration of training. After each validation epoch a JIT-compiled forward pass with `return_weights=True` computes the mean attention entropy over all heads and logs it as `val/attn_entropy`. Every 5 epochs an attention geographic figure is also logged.

**Multi-storm exclusion:** IBTrACS timestamps with ≥2 active storms are not used during training or validation — the model sees only unambiguous single-storm or background samples. These timestamps form the `hard_test` split for post-training analysis.
