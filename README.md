# jax_research_template

A research template for building JAX/Flax deep learning experiments in geoscience. It provides reusable building blocks for models, data pipelines, and training so you can focus on the experiment rather than the scaffolding.

**Stack:** Python 3.12 · JAX 0.10+ · Flax 0.12+ · Optax 0.2+ · Orbax · Optuna

---

## What this template provides

| Layer | What it gives you |
|-------|-------------------|
| `jrt/core/` | Registry-based attention, embeddings, norms, activations, initializers, pooling, and assembled nets (MLP, CNN, Transformer, ViT, Swin) |
| `jrt/datasets/` | Generic `.npz` dataset loader, batching utilities, and a `DataModule` ABC that any experiment can subclass |
| `jrt/training/` | Single-device `Trainer` (early stopping, checkpointing, val cadence, step callbacks); loss stack with prediction + model terms; metrics atoms (exact confusion-matrix accumulation); optimizer/scheduler registry; wandb/TensorBoard/null logger; multiprocess `ProcessPrefetcher`; Optuna `Tuner` |
| `jrt/utils/` | Geoscience helpers (haversine, met/unit conversions, geodesic areas), JAX utilities, numpy normalisers, coordinate sampling, and plotting mechanics (geo axes, heatmaps, curves) |
| `jrt/experiments/` | Self-contained experiment directories that wire together the above components |

The layering rule: experiments import `jrt/*` freely; `jrt` never imports
experiments. Reusable mechanics get promoted UP into `jrt`; policy (which
variables, which losses, which figures) stays in the experiment.

---

## Repository layout

```
jax_research_template/
├── jrt/
│   ├── core/               Model building blocks and registered nets
│   ├── datasets/           Generic data loading and batching
│   ├── training/           Trainer, losses, metrics, prefetch, logger, tuner
│   ├── utils/              Geoscience, JAX helpers, plotting, normalisers
│   └── experiments/        One directory per experiment
│       ├── cyclone_jax/       CURRENT: TC intensity from sparse in-situ obs
│       │                      (the canonical example — start here)
│       └── tc_perceiver_io/   frozen v1 line (kept for reference)
├── tests/                  Pytest suite mirroring jrt/ structure
├── environment.yaml        Conda environment (name: jrt)
└── pytest.ini
```

Documentation trail: this README → an experiment's `README.md` (layout +
CLI) → per-package `usage_doc.md` files (`data/`, `models/`) for the
deep-dive on each surface. Every knob a yaml accepts is key-set validated
(a typo is an error, not a silent default).

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

### 2. Run the current experiment (cyclone_jax)

```bash
# Put jrt/ on the Python path once (packages are rooted there):
#   export PYTHONPATH=jrt        (bash)   |   $env:PYTHONPATH="jrt"   (PowerShell)

# Edit the data root + year lists in the scenario yaml, then:
export WANDB_API_KEY=...           # when trainer.logger: wandb
python -m experiments.cyclone_jax.train.train \
    jrt/experiments/cyclone_jax/configs/train/train.yaml --gpu 0
```

The entry yaml points at one data scenario and one model config
(`data: overfit`, `model: mlp | siren | finer`) — swapping either is a
one-line edit; `config.load_config` resolves the pointers and validates
every key. See `jrt/experiments/cyclone_jax/README.md` for the run
records (norm stats, data manifest, figures), GPU pinning, and the
`num_workers` knob for multi-CPU boxes.

### 3. Run tests

```bash
conda activate jrt
pytest tests/
```

---

## Starting a new experiment

Each experiment lives in its own directory under `jrt/experiments/`.
`cyclone_jax/` is the canonical example of how to structure one; the
shape that has worked:

```
jrt/experiments/my_experiment/
├── config.py        load_config: resolve config pointers, validate key sets
├── configs/
│   ├── data/        one yaml per DATA SCENARIO (splits, normalisation, tags)
│   ├── models/      one yaml per model (registry name + hyperparams + tags)
│   └── train/       entry points: point at {data, model}, inline trainer block
├── data/            spec objects (Input/Target/NormSpec) → Loader → streams;
│                    one build_data(cfg, seed) entry point returning a bundle
├── models/          MODELS registry + build_model(cfg, targets)
├── train/           train.py orchestrator + thin losses/metrics/log builders
│                    (mechanics live in jrt.training; these files are glue)
└── visualise/       figure mechanics (log.py decides when/what to log)
```

Principles that keep it tractable: ONE seed drives model init and data
order; data modules stay jax-free (multiprocess-worker purity) with a
fixed `pad_to` so jit compiles once; batches are named dicts
`{'X', 'y', 'meta'}` (meta = eval/plot identity, never model input);
thin experiment files — anything reusable gets promoted into `jrt`.

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

Set `trainer.log_backend` in the YAML config (cyclone_jax builds its
logger itself — `trainer.logger` + `logger_kwargs` — so model/data tags
and the `{model}-{data}-s{seed}` run name reach wandb):

| Backend | Config value | Notes |
|---------|-------------|-------|
| No-op (default) | `null` | Saves figures as PNG to `logs/figures/` |
| Weights & Biases | `wandb` | Set `WANDB_API_KEY` env var; pass `project`/`name` in `log_kwargs` |
| TensorBoard | `tensorboard` | Cluster-friendly; no account needed |

All three backends expose the same interface: `log_metrics`, `log_hyperparams`, `log_figure`, `log_image`, `log_histogram`.

---

## Hyperparameter tuning

The `Tuner` in `jrt/training/tuner.py` wraps Optuna around the existing `Trainer`. Experiments provide a `tune.py` entry point with a `SEARCH_SPACE` dict and a `suggest_fn` that maps trial samples into the config (tc_perceiver_io pattern; cyclone_jax's tune.py is the next build step).

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
