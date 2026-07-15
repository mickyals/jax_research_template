# jrt/experiments

Each subdirectory is a self-contained experiment. Experiments import from `core/`, `datasets/`, `training/`, and `utils/` but are otherwise independent of each other.

---

## Structure of an experiment

`cyclone_jax/` is the canonical (v2) shape — start there. The older
dataset.py/datamodule.py single-file shape (tc_perceiver_io) still works
with the same Trainer, but new experiments should follow:

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
└── visualise/       figure mechanics (train/log.py decides when/what to log)
```

---

## How experiments connect to the template

```
experiment/train(.py | /train.py)
    │
    ├── loads YAML config (cyclone_jax: load_config resolves data/model
    │   pointers and key-set-validates every block)
    ├── builds data (build_data(cfg, seed) → streams; or a BaseDataModule)
    ├── builds model (Flax nn.Module using core/ building blocks)
    ├── builds metrics_fns dict (scalar JAX fns; loss may be a LossStack)
    │
    └── training.trainer.Trainer(model, metrics_fns, trainer_cfg, logger=...)
            │
            ├── training.optimizers  →  optimizer + scheduler
            ├── training.logger      →  WandB / TensorBoard / Null
            └── .fit(train_stream, val_stream,
                     epoch_callbacks=[...], step_callbacks=[(fn, every), ...])
                    │
                    └── orbax checkpoints → run_dir/checkpoints/best|latest
```

**The Trainer does not know anything about the model or data** — it only sees `train_step(state, batch)` and `eval_step(state, batch)` with JAX arrays. Any model that accepts `(X, train: bool)` and any loader that yields `{"X": ..., "y": ...}` dicts will work.

---

## Running an experiment

```bash
# Run from repo root with jrt/ on the path (packages are rooted at jrt/):
#   export PYTHONPATH=jrt        (bash)   |   $env:PYTHONPATH="jrt"   (PowerShell)

# Current experiment (see its README for site env vars + GPU pinning):
python -m experiments.cyclone_jax.train.train \
    jrt/experiments/cyclone_jax/configs/train/train.yaml --gpu 0
```

---

## Existing experiments

| Directory | Description |
|-----------|-------------|
| [`cyclone_jax/`](cyclone_jax/README.md) | **CURRENT.** TC intensity classification from sparse in-situ obs (arcana volume/bookshelf data layer; MLP/SIREN/FINER baselines, Perceiver-IO planned) — the canonical v2 experiment shape |
| [`tc_perceiver_io/`](tc_perceiver_io/README.md) | Frozen v1 line, kept for reference: TC classifier from sparse land surface obs with a unified Transformer / asymmetric attention mask |

---

## Adding a new experiment

1. Copy the cyclone_jax structure above into a new directory.
2. Build the data side around one `build_data(cfg, seed) -> bundle` entry
   point (spec objects → Loader → streams yielding `{'X','y','meta'}`).
3. Implement models against `core/` blocks, registered + built via
   `build_model(cfg, targets)`.
4. Keep `train/` files thin glue over `jrt.training` (losses/metrics/log
   builders); figures go in `visualise/`, cadence decisions in
   `train/log.py`.
5. Wire it in `train/train.py`:

```python
data  = build_data(cfg['data'], seed=seed)
model, tags = build_model(cfg['model'], data.targets)
trainer = Trainer(model, build_metrics_fns(cfg['trainer']),
                  build_trainer_config(cfg), logger=logger)
best_state   = trainer.fit(data.streams['train'], data.streams['val'],
                           step_callbacks=build_callbacks(cfg, data, logger))
test_metrics = trainer.test(data.streams['test'])
```

Checkpointing, logging, early stopping, val cadence, and resume are all
handled by the Trainer.
