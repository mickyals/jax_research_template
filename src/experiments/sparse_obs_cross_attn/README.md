# Experiment: Sparse Observation Cross-Attention TC Classifier

**Goal** — given sparse in-situ land surface observations within a radius of a query position at time *t*, predict the SSHS intensity class of any storm present.

This is the pretraining stage of a two-phase pipeline. After convergence, the cross-attention encoder is frozen and a regression head is attached to estimate physical TC parameters (wind speed, central pressure, RMW).

---

## Architecture

```
Query position (storm centre or background point)
    → unit-sphere [x, y, z]
    → GaussianFourierEmbedding(input_dim=3, mapping_dim=64)
    → Dense(embed_dim)
    → query token  (B, 1, embed_dim)

Station observations (≤ max_stations per sample)
    [pressure, temperature, dew point, wind speed,          ← physical obs (NaN where missing)
     bearing_sin, bearing_cos, log_dist_norm]               ← geometric features (always valid)
    concat with obs validity mask (same shape, float)       ← tells model which obs are absent
    → Dense(embed_dim)
    → station tokens  (B, N, embed_dim)

N × CrossAttentionBlock
    Q = query token, K/V = station tokens
    mask = station_mask  (True → real station, False → padding)
    → context vector  (B, embed_dim)

OrdinalHead  →  9 logits  →  CORAL ordinal loss
```

**Output** — 10 ordinal classes:

| Class | Label          | USA_SSHS |
|------:|----------------|----------|
|     0 | No storm       | —        |
|     1 | Disturbance    | −3       |
|     2 | Subtropical    | −2       |
|     3 | Tropical Depression | −1  |
|     4 | Tropical Storm | 0        |
|     5 | Category 1     | 1        |
|     6 | Category 2     | 2        |
|     7 | Category 3     | 3        |
|     8 | Category 4     | 4        |
|     9 | Category 5     | 5        |

---

## Data

Two sources are joined per sample:

- **IBTrACS** (`ibtracs_tc_clean.npz`) — storm centre position, timestamp, SSHS class
- **InsituLand** (`insitu_land_clean.npz` + `insitu_land_station_meta.npz`) — land surface observations from Copernicus C3S, Caribbean / Gulf domain (LAT 0–30 N, LON 100–45 W)

Temporal split follows IBTrACS seasons: train 2005–2020, val 2021–2022, test 2023–2025. Multi-storm timesteps are held out as a hard test set.

Each training batch is **1:1 balanced** — half TC samples, half background samples drawn from non-TC periods at random domain positions.

---

## Quick start

### 1. Set WandB credentials

```bash
export WANDB_API_KEY=your_key   # preferred
# or: wandb login                # interactive, persists to ~/.netrc
```

### 2. Edit paths in the config

```
src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml
```

Update the four data paths under the `data:` block.

### 3. Train

```bash
python src/experiments/sparse_obs_cross_attn/train.py

# Resume interrupted run
python src/experiments/sparse_obs_cross_attn/train.py --resume

# Custom config
python src/experiments/sparse_obs_cross_attn/train.py \
    --config path/to/my_config.yaml
```

### 4. Evaluate

```bash
# Attention maps + confusion matrix (val split, 8 examples)
python src/experiments/sparse_obs_cross_attn/evaluate.py \
    --config  src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
    --ckpt    checkpoints/tc_classifier/best \
    --split   val \
    --n       8 \
    --out_dir figures/

# Confusion matrix only
python src/experiments/sparse_obs_cross_attn/evaluate.py \
    --config ... --ckpt ... --mode confusion
```

---

## Metrics

Logged to WandB automatically by the Trainer:

| Metric | When | Description |
|--------|------|-------------|
| `train/ordinal_loss` | every `log_every_n_steps` steps | CORAL binary cross-entropy (training loss) |
| `val/ordinal_loss` | every epoch | Same loss on validation set |
| `val/accuracy` | every epoch | Exact SSHS class match fraction |
| `val/mae_class` | every epoch | Mean \|predicted class − true class\| — primary ordinal metric |
| `val/within_1_class` | every epoch | Fraction within 1 class step (e.g. Cat-1 vs Cat-2 counts as correct) |
| `val/within_2_class` | every epoch | Fraction within 2 class steps |

**Interpretation guide:**
- Early training: `val/ordinal_loss` decreases, `val/accuracy` climbs slowly (class imbalance).
- `val/mae_class` below 1.5 is a reasonable early target.
- A model that always predicts No-Storm achieves accuracy ≈ 0.5 but `mae_class` ≈ 2.5 — use `mae_class` to detect this collapse.

---

## Evaluation outputs

`evaluate.py` produces two figure types:

**Attention maps** (`figures/attn_val_NNN.png`) — geographic scatter plot centred on the storm, stations coloured by mean cross-attention weight from the final layer. Shows which land stations the model uses to infer TC intensity.

**Confusion matrix** (`figures/confusion_val.png`) — 10×10 ordinal confusion matrix. Errors should cluster near the diagonal; large off-diagonal mass indicates calibration problems.

---

## Config reference

Full field documentation: [`configs/schema.py`](configs/schema.py)

Key knobs:

| Key | Default | Notes |
|-----|---------|-------|
| `data.radius_km` | 500 | Increase to pull in more distant stations |
| `data.time_window_hours` | 3.0 | Set to 0.0 for same-minute observations only |
| `data.max_stations` | 64 | Larger values slow training; smaller values lose context |
| `model.n_layers` | 2 | More layers → richer query–context interaction |
| `model.embed_dim` | 128 | Bottleneck of the attention representation |
| `trainer.scheduler_kwargs.decay_steps` | 50000 | Set to ~`n_tc_train / 32 × num_epochs` |

---

## File layout

```
sparse_obs_cross_attn/
├── model.py        TCClassifier Flax module + forward_with_weights
├── datamodule.py   JointDataModule → Trainer-compatible loaders
├── metrics.py      Ordinal metrics (ordinal_loss, mae_class, within_k)
├── train.py        Entry point — calls Trainer.fit()
├── evaluate.py     Attention maps, confusion matrix
├── configs/
│   ├── tc_classifier.yaml   Default hyperparameter config
│   └── schema.py            Typed schema + validate_config()
└── README.md       This file
```

Reusable components imported from the template:

```
core/attention.py           CrossAttention, CrossAttentionBlock
core/embeddings.py          GaussianFourierEmbedding
core/nets/heads.py          OrdinalHead
datasets/joint/dataset.py   JointTCDataset
training/ordinal_loss.py    ordinal_loss, ordinal_predict, ordinal_probs
training/trainer.py         Trainer
training/logger.py          WandbLogger / TensorBoardLogger / NullLogger
```
