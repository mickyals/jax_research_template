# jax_research_template

A research template for building JAX/Flax deep learning experiments in geoscience. It provides reusable building blocks for models, data pipelines, and training so you can focus on the experiment rather than the scaffolding.

**Stack:** Python 3.12 · JAX 0.10+ · Flax 0.12+ · Optax 0.2+ · Orbax · Optuna

---

## What this template provides

| Layer | What it gives you |
|-------|-------------------|
| `jrt/core/` | Registry-based attention, embeddings, norms, activations, initializers, pooling, and assembled nets (MLP, CNN, Transformer, ViT, Swin) |
| `jrt/datasets/` | Generic `.npz` dataset loader, batching utilities, and a `DataModule` ABC that any experiment can subclass |
| `jrt/training/` | Single-device `Trainer` with early stopping, checkpointing, and logging; loss library; optimizer/scheduler registry; Optuna `Tuner` |
| `jrt/utils/` | Geoscience helpers (Haversine, Vincenty, met conversions), JAX utilities, coordinate sampling, and plotting |
| `jrt/experiments/` | Self-contained experiment directories that wire together the above components |

---

## Repository layout

```
jax_research_template/
├── jrt/
│   ├── core/               Model building blocks and registered nets
│   ├── datasets/           Generic data loading and batching
│   ├── training/           Trainer, losses, optimizers, logger, tuner
│   ├── utils/              Geoscience, JAX helpers, plotting, sampling
│   └── experiments/        One directory per experiment
│       └── tc_perceiver_io/   Tropical cyclone classifier (reference impl)
├── tests/                  Pytest suite mirroring jrt/ structure
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
jrt/experiments/tc_perceiver_io/configs/tc_classifier.yaml

# Put jrt/ on the Python path once (packages are rooted there):
#   export PYTHONPATH=jrt        (bash)   |   $env:PYTHONPATH="jrt"   (PowerShell)

# Train
python -m experiments.tc_perceiver_io.train.train \
    jrt/experiments/tc_perceiver_io/configs/tc_classifier.yaml

# Resume interrupted training
python -m experiments.tc_perceiver_io.train.train \
    jrt/experiments/tc_perceiver_io/configs/tc_classifier.yaml \
    --resume

# Evaluate (config path is positional)
python -m experiments.tc_perceiver_io.train.evaluate \
    jrt/experiments/tc_perceiver_io/configs/tc_classifier.yaml \
    --checkpoint_dir runs/tc_classifier/run_01/checkpoints \
    --output_dir runs/tc_classifier/run_01/eval

# Hyperparameter search
python -m experiments.tc_perceiver_io.train.tune \
    jrt/experiments/tc_perceiver_io/configs/tc_tune.yaml \
    --n_trials 25 \
    --storage sqlite:///runs/tc_classifier/hp_search/study.db
```

### 3. Run tests

```bash
conda activate jrt
pytest tests/
```

---

## Starting a new experiment

Each experiment lives in its own directory under `jrt/experiments/`. The reference experiment `tc_perceiver_io/` is the canonical example of how to structure one.

**Minimum files:**

```
jrt/experiments/my_experiment/
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

See [`jrt/core/README.md`](jrt/core/README.md) for the full registry listing.

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

The `Tuner` in `jrt/training/tuner.py` wraps Optuna around the existing `Trainer`. Experiments provide a `tune.py` entry point with a `SEARCH_SPACE` dict and a `suggest_fn` that maps trial samples into the config.

```bash
python -m experiments.my_experiment.tune configs/my_tune.yaml \
    --n_trials 50 \
    --storage sqlite:///runs/hp_search.db \
    --study_name my_experiment_v1
```

See [`jrt/training/README.md`](jrt/training/README.md) for the full Tuner API.

---

## Acknowledgements

Much of the modeling and experimentation code in this package is adapted from the
[UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/)
by Phillip Lippe (University of Amsterdam). The JAX+Flax tutorial notebooks were
especially helpful as a foundation.
