# jax_research_template

A research template for building JAX/Flax deep learning experiments in geoscience. It provides reusable building blocks for models, data pipelines, and training so you can focus on the experiment rather than the scaffolding.

**Stack:** Python 3.12 · JAX 0.10+ · Flax 0.12+ · Optax 0.2+ · Orbax · Optuna

---

## What this template provides

| Layer | What it gives you |
|-------|-------------------|
| `src/core/` | Registry-based attention, embeddings, norms, activations, initializers, pooling, and assembled nets (MLP, CNN, Transformer, ViT, Swin) |
| `src/datasets/` | Generic `.npz` dataset loader, batching utilities, and a `DataModule` ABC that any experiment can subclass |
| `src/training/` | Single-device `Trainer` with early stopping, checkpointing, and logging; loss library; optimizer/scheduler registry; Optuna `Tuner` |
| `src/utils/` | Geoscience helpers (Haversine, Vincenty, met conversions), JAX utilities, coordinate sampling, and plotting |
| `src/experiments/` | Self-contained experiment directories that wire together the above components |

---

## Repository layout

```
jax_research_template/
├── src/
│   ├── core/               Model building blocks and registered nets
│   ├── datasets/           Generic data loading and batching
│   ├── training/           Trainer, losses, optimizers, logger, tuner
│   ├── utils/              Geoscience, JAX helpers, plotting, sampling
│   └── experiments/        One directory per experiment
│       └── sparse_obs_cross_attn/   Tropical cyclone classifier (reference impl)
├── src_test/               Pytest suite mirroring src/ structure
├── environment.yaml        Conda environment (name: jrt)
└── pytest.ini
```

---

## Quickstart

### 1. Create the environment

```bash
conda env create -f environment.yaml   # creates the 'jrt' environment
conda activate jrt
```

For GPU/TPU support, follow the [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html) after activating the environment:

```bash
pip install --upgrade "jax[cuda12]"   # CUDA 12.x
```

### 2. Run the reference experiment

```bash
# Edit data paths in the config
src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml

# Train
python -m experiments.sparse_obs_cross_attn.train \
    src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml

# Resume interrupted training
python -m experiments.sparse_obs_cross_attn.train \
    src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
    --resume

# Evaluate
python -m experiments.sparse_obs_cross_attn.evaluate \
    --checkpoint_dir runs/tc_classifier/run_01/checkpoints \
    --config src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml

# Hyperparameter search
python -m experiments.sparse_obs_cross_attn.tune \
    src/experiments/sparse_obs_cross_attn/configs/tc_tune.yaml \
    --n_trials 25 \
    --storage sqlite:///runs/tc_classifier/hp_search/study.db
```

### 3. Run tests

```bash
conda activate jrt
pytest src_test/
```

---

## Starting a new experiment

Each experiment lives in its own directory under `src/experiments/`. The reference experiment `sparse_obs_cross_attn/` is the canonical example of how to structure one.

**Minimum files:**

```
src/experiments/my_experiment/
├── dataset.py       Subclass NpzDataset or write a custom loader
├── datamodule.py    Subclass BaseDataModule; expose train/val/test loaders
├── model.py         Flax nn.Module; __call__(X, train: bool) -> predictions
├── metrics.py       Dict of scalar loss/metric functions
├── train.py         Load config → build model + datamodule → Trainer.fit()
├── evaluate.py      Load checkpoint → run predictions → figures
└── configs/
    └── my_config.yaml
```

**Wiring the Trainer:**

```python
from training.trainer import Trainer

metrics_fns = {"mse": mse, "mae": mae}   # first key is the training loss
trainer     = Trainer(model, metrics_fns, config["trainer"])
best_state  = trainer.fit(dm.train_loader(), dm.val_loader())
test_metrics = trainer.test(dm.test_loader())
```

The Trainer expects loaders that yield `{"X": array_or_dict, "y": array}` dicts. For models with structured inputs (like attention-based models that take dicts), `batch["X"]` can itself be a dict — the Trainer handles both transparently.

**Run directory layout** (set `trainer.run_dir` in config):

```
runs/my_experiment/run_01/
├── checkpoints/
│   ├── best/       Best validation checkpoint (orbax pytree)
│   └── latest/     End-of-epoch checkpoint for resume
└── logs/
    ├── hparams.json
    ├── figures/    Saved figures (NullLogger) or WandB local cache
    └── wandb/      (when log_backend: wandb)
```

---

## Using core components

All registries follow the same `get_X` / `list_Xs` pattern and are string-addressable from YAML configs.

```python
from core import get_activation, get_initializer
from core.norms import get_norm
from core.pooling import get_pooling
from core.embeddings import get_embedding
from core.nets.transformers import get_transformer

# Retrieve by name (case-insensitive)
act   = get_activation("gelu")
norm  = get_norm("layernorm")
model = get_transformer("VIT", patch_size=16, embed_dim=256, num_heads=8, num_layers=6)
```

See [`src/core/README.md`](src/core/README.md) for the full registry listing.

---

## Logging backends

Set `trainer.log_backend` in the YAML config:

| Backend | Config value | Notes |
|---------|-------------|-------|
| No-op (default) | `null` | Saves figures as PNG to `logs/figures/` |
| Weights & Biases | `wandb` | Set `WANDB_API_KEY` env var; pass `project`/`name` in `log_kwargs` |
| TensorBoard | `tensorboard` | Cluster-friendly; no account needed |

All three backends expose the same interface: `log_metrics`, `log_hyperparams`, `log_figure`, `log_image`, `log_histogram`.

---

## Hyperparameter tuning

The `Tuner` in `src/training/tuner.py` wraps Optuna around the existing `Trainer`. Experiments provide a `tune.py` entry point with a `SEARCH_SPACE` dict and a `suggest_fn` that maps trial samples into the config.

```bash
python -m experiments.my_experiment.tune configs/my_tune.yaml \
    --n_trials 50 \
    --storage sqlite:///runs/hp_search.db \
    --study_name my_experiment_v1
```

See [`src/training/README.md`](src/training/README.md) for the full Tuner API.

---

## Acknowledgements

Much of the modeling and experimentation code in this package is adapted from the
[UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/)
by Phillip Lippe (University of Amsterdam). The JAX+Flax tutorial notebooks were
especially helpful as a foundation.
