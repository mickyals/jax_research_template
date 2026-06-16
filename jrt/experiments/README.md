# jrt/experiments

Each subdirectory is a self-contained experiment. Experiments import from `core/`, `datasets/`, `training/`, and `utils/` but are otherwise independent of each other.

---

## Structure of an experiment

```
jrt/experiments/my_experiment/
├── dataset.py       Domain-specific dataset (subclasses NpzDataset or custom)
├── datamodule.py    Subclass of BaseDataModule; exposes train/val/test loaders
├── model.py         Flax nn.Module; __call__(X, train: bool) -> predictions
├── metrics.py       Dict of scalar metric functions + build_metrics_fns() factory
├── train.py         CLI entry point: config → datamodule + model + Trainer.fit()
├── evaluate.py      Load checkpoint → predictions → confusion matrix / figures
├── tune.py          Optuna HP search entry point (optional)
└── configs/
    ├── my_config.yaml     Full training config
    ├── my_tune.yaml       Short-epoch tuning config (optional)
    └── schema.json        JSON schema for config validation (optional)
```

---

## How experiments connect to the template

```
experiment/train.py
    │
    ├── loads YAML config
    ├── builds datamodule (subclasses datasets/datamodule.BaseDataModule)
    ├── builds model (Flax nn.Module using core/ building blocks)
    ├── builds metrics_fns dict (scalar JAX functions)
    │
    └── training.trainer.Trainer(model, metrics_fns, config["trainer"])
            │
            ├── training.optimizers  →  optimizer + scheduler
            ├── training.logger      →  WandB / TensorBoard / Null
            └── .fit(train_loader, val_loader, epoch_callbacks=[...])
                    │
                    └── orbax checkpoints → run_dir/checkpoints/best|latest
```

**The Trainer does not know anything about the model or data** — it only sees `train_step(state, batch)` and `eval_step(state, batch)` with JAX arrays. Any model that accepts `(X, train: bool)` and any loader that yields `{"X": ..., "y": ...}` dicts will work.

---

## Running an experiment

```bash
# Run from repo root with jrt/ on the path (packages are rooted at jrt/):
#   export PYTHONPATH=jrt        (bash)   |   $env:PYTHONPATH="jrt"   (PowerShell)
# Train
python -m experiments.my_experiment.train configs/my_config.yaml

# Resume
python -m experiments.my_experiment.train configs/my_config.yaml --resume

# Evaluate
python -m experiments.my_experiment.evaluate \
    --checkpoint_dir runs/my_experiment/run_01/checkpoints \
    --config configs/my_config.yaml

# Hyperparameter search
python -m experiments.my_experiment.tune configs/my_tune.yaml \
    --n_trials 25 \
    --storage sqlite:///runs/hp_search.db
```

---

## Existing experiments

| Directory | Description |
|-----------|-------------|
| [`sparse_obs_cross_attn/`](sparse_obs_encoder/README.md) | Tropical cyclone intensity classifier from sparse land surface observations using a unified Transformer with asymmetric attention mask |

---

## Adding a new experiment

1. Copy the structure above into a new directory.
2. Implement `dataset.py` and `datamodule.py` for your data source.
3. Implement `model.py` using building blocks from `core/`.
4. Implement `metrics.py` — a `build_metrics_fns()` factory that returns a `{name: fn}` dict where each `fn(pred, target) -> scalar`.
5. Wire it together in `train.py`:

```python
dm          = MyDataModule.from_config(config["data"])
model       = MyModel(**config["model"])
metrics_fns = build_metrics_fns()
trainer     = Trainer(model, metrics_fns, config["trainer"])
trainer.log_hyperparams(config)
best_state  = trainer.fit(dm.train_loader(), dm.val_loader())
test_metrics = trainer.test(dm.test_loader())
```

That's the complete training loop — checkpointing, logging, early stopping, and resume are all handled by the Trainer.
