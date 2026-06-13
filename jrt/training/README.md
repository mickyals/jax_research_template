# jrt/training

Orchestration layer: training loop, logging, losses, optimizers, and hyperparameter tuning. Nothing here is experiment-specific — it wires together a model, a datamodule, and a config dict.

---

## `trainer.py` — `Trainer`

Single-device JAX/Flax training loop.

```python
from training.trainer import Trainer

metrics_fns = {"cross_entropy": ce_fn, "accuracy": acc_fn}
trainer     = Trainer(model, metrics_fns, config["trainer"])

# Train — returns the best TrainState (by patience_metric)
best_state = trainer.fit(dm.train_loader(), dm.val_loader())

# Resume interrupted training
best_state = trainer.fit(dm.train_loader(), dm.val_loader(), resume=True)

# Evaluate on test split using the saved best checkpoint
test_metrics = trainer.test(dm.test_loader())
```

**What it manages:**
- JIT-compiled `train_step` and `eval_step` (built once on first call to `fit`)
- Per-step logging (every `log_every_n_steps` gradient updates)
- Epoch-level validation across all registered metrics
- Early stopping on `patience_metric` with configurable direction
- Orbax checkpointing of both `best/` and `latest/` states
- `num_steps` budget as an alternative to `num_epochs`
- Epoch callbacks: `fit(..., epoch_callbacks=[fn])` where each `fn(state, epoch, global_step)` is called after validation
- Run manifests: `trainer.write_manifest(dm.manifest())` writes `manifest.json` under `checkpoint_dir` (the durable record of what the run trained on — e.g. resolved split membership) and pushes a copy to the logger via `log_hyperparams`

**Config keys** (YAML `trainer:` block):

| Key | Default | Notes |
|-----|---------|-------|
| `batch_size` | — | required |
| `optimizer` | — | required; name from optimizer registry |
| `scheduler` | — | required; name from scheduler registry |
| `optimizer_kwargs` | `{}` | passed to optimizer |
| `scheduler_kwargs` | `{}` | passed to scheduler |
| `loss_key` | first metric | which metric is differentiated |
| `num_epochs` | 1000 | |
| `num_steps` | inf | stop whichever limit hits first |
| `patience` | 10 | |
| `patience_metric` | `val/<loss_key>` | metric to watch for early stopping |
| `patience_direction` | `lower_is_better` | or `higher_is_better` |
| `max_grad_norm` | None | gradient clipping; None = disabled |
| `run_dir` | — | co-locates `checkpoints/` and `logs/` under one root |
| `checkpoint_dir` | `checkpoints` | ignored when `run_dir` is set |
| `log_every_n_steps` | 50 | |
| `log_backend` | `null` | `null` / `wandb` / `tensorboard` |
| `log_kwargs` | `{}` | forwarded to the logger constructor |
| `seed` | 42 | model init and dropout RNG |
| `use_tqdm` | true | progress bars |
| `profile` | false | JAX-profiler trace of the first `profile_steps` training steps → `<log_dir>/profile`, then handed to the logger via `log_artifact`: WandB uploads it to the run's artifact store; TensorBoard/Null leave it on disk and print the path (TensorBoard backend prints the `--logdir` viewer hint). Viewing always uses TensorBoard's Profile plugin — WandB stores but cannot render XLA traces. First traced step includes jit compilation; read later steps for steady-state timing |
| `profile_steps` | 5 | steps to trace when `profile: true` |

**Run directory layout** (when `run_dir` is set):

```
<run_dir>/
├── checkpoints/
│   ├── best/              Best validation state (orbax pytree)
│   ├── latest/            End-of-epoch state for --resume
│   ├── latest_metadata.json
│   └── manifest.json      Run manifest (resolved data split etc.), via write_manifest()
└── logs/
    ├── hparams.json
    ├── figures/           Saved by NullLogger
    └── wandb/             Local WandB cache
```

**Loader contract:** any iterable that yields `{"X": array_or_dict, "y": array}` dicts. `batch["X"]` may itself be a dict for models with structured inputs — the Trainer handles both. The top-level key `"meta"` is reserved for non-model sample metadata (attribution strings, diagnostics): the Trainer drops it before its jitted train/eval steps, so loaders may attach it freely.

---

## `losses.py`

Scalar loss functions for JAX/Flax training. All functions have the signature `(pred, target) -> scalar`.

| Function | Description |
|----------|-------------|
| `mse(pred, target)` | Mean squared error |
| `rmse(pred, target)` | Root mean squared error |
| `mae(pred, target)` | Mean absolute error |
| `huber(pred, target, delta)` | Huber loss |
| `log_cosh(pred, target)` | Log-cosh loss |
| `masked_mse`, `masked_rmse`, ... | NaN-safe variants; `mask=True` means valid |
| `ordinal_loss(logits, labels)` | CORAL ordinal cross-entropy (Cao et al. 2020) |
| `ordinal_predict(logits)` | Predicted class from ordinal logits |
| `ordinal_probs(logits)` | Per-class probabilities from ordinal logits |

Masked variants derive the mask from `jnp.isfinite(target)` when no mask is passed — NaN targets are excluded automatically. They return `0.0` when no valid positions exist.

---

## `optimizers.py`

String-addressable optimizer and scheduler registries built on Optax.

```python
from training.optimizers import get_optimizer, get_scheduler, list_optimizers, list_schedulers

schedule  = get_scheduler("warmup_cosine",
                          init_value=0.0, peak_value=1e-3,
                          warmup_steps=500, decay_steps=10_000)
optimizer = get_optimizer("adamw", learning_rate=schedule, weight_decay=1e-4)
```

**Optimizers:**

| Name | Description |
|------|-------------|
| `adam` | Adam (Kingma & Ba 2015) |
| `adamw` | Adam with decoupled weight decay (Loshchilov & Hutter 2019) |
| `sgd` | SGD with optional momentum and Nesterov |
| `rmsprop` | RMSProp |
| `lbfgs` | L-BFGS (requires custom train_step; see module docstring) |

**Schedulers:**

| Name | Description |
|------|-------------|
| `constant` | Fixed value |
| `cosine_decay` | Cosine annealing to near-zero |
| `cosine_onecycle` | One-cycle cosine (Smith 2019) |
| `exponential_decay` | Continuous or staircase exponential |
| `polynomial` | Polynomial interpolation from init to end value |
| `warmup_cosine` | Linear warmup then cosine decay (default for transformers) |
| `warmup_constant` | Linear warmup then constant |

---

## `logger.py`

Unified logging interface with three backends.

```python
from training.logger import create_logger

logger = create_logger("wandb", log_dir="runs/exp01/logs",
                       project="my_project", name="run_01")
logger = create_logger("tensorboard", log_dir="runs/exp01/logs")
logger = create_logger("null", log_dir="runs/exp01/logs")  # default
```

All three share the same interface:

| Method | Description |
|--------|-------------|
| `log_metrics(metrics_dict, step)` | Scalar metrics |
| `log_hyperparams(params_dict)` | Hyperparameters (written to `hparams.json`) |
| `log_figure(key, fig, step)` | Matplotlib figure |
| `log_image(key, image, step)` | NumPy image array |
| `log_histogram(key, values, step)` | Scalar distribution |
| `log_artifact(name, path, artifact_type)` | File/dir attached to the run — WandB uploads to its artifact store; TensorBoard/Null leave it on disk and print the path |
| `finalize(status)` | Called at end of training |

`NullLogger` saves figures to `log_dir/figures/` as PNG files. `WandbLogger` streams everything remotely. Set `WANDB_API_KEY` as an environment variable — never put it in config files.

Access the logger from anywhere in a training script via `trainer.logger` (read-only property).

---

## `metrics.py`

Metric functions used in experiment `metrics.py` files. Contains accuracy, classification metrics, and regression metrics built on JAX/NumPy. These are scalar functions with signature `(pred, target) -> float` suitable for passing to `Trainer`.

---

## `tuner.py` — `Tuner`

Optuna hyperparameter search wrapped around `Trainer`.

```python
from training.tuner import Tuner, apply_search_space
import copy

SEARCH_SPACE = {
    "lr":           {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
    "dropout_rate": {"type": "float", "low": 0.0,  "high": 0.4},
    "num_layers":   {"type": "int",   "low": 1,    "high": 6},
    "activation":   {"type": "categorical", "choices": ["relu", "gelu", "sine"]},
}

def suggest_fn(trial, base_config):
    hp  = apply_search_space(trial, SEARCH_SPACE)
    cfg = copy.deepcopy(base_config)
    cfg["trainer"]["scheduler_kwargs"]["peak_value"] = hp["lr"]
    cfg["trainer"]["optimizer_kwargs"]["weight_decay"] = hp["weight_decay"]
    cfg["model"]["dropout_rate"] = hp["dropout_rate"]
    return cfg  # return full config dict

def model_fn(config):
    return MyModel(**config["model"])

tuner = Tuner(
    suggest_fn      = suggest_fn,   # (trial, base_config) -> full config dict
    base_config     = full_config,  # entire YAML config dict
    model_fn        = model_fn,     # (config) -> nn.Module
    metrics_fns     = metrics_fns,
    train_loader_fn = lambda: dm.train_loader(),
    val_loader_fn   = lambda: dm.val_loader(),
    study_name      = "my_experiment",
    direction       = "minimize",
    storage         = "sqlite:///runs/hp_search.db",  # None = in-memory
)

tuner.run(n_trials=50)
tuner.summary()
print(tuner.best_params)
```

**How it works:**
- Each trial calls `suggest_fn` to sample a config, then runs `Trainer.fit()` with that config
- The Trainer reports the `patience_metric` to Optuna after each epoch
- `MedianPruner` cuts unpromising trials early (`n_startup_trials` / `n_warmup_steps` control when pruning activates)
- Each trial gets an isolated `run_dir/trial_N/` subdirectory
- The study persists to SQLite so a search can be resumed by running the same command again

**`suggest_fn` contract:** must return either:
- A full config dict with a `"trainer"` key (for experiments that tune architecture HPs alongside training HPs), or
- A bare trainer config dict (for training-only searches)

`apply_search_space(trial, space)` handles the Optuna suggestion boilerplate and returns `{name: sampled_value}`. Supported types: `float`, `int`, `categorical`.
