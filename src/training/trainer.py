"""
training/trainer.py

Single-device JAX / Flax trainer.

Wires together a Flax model, a metrics dict, an optax optimiser, array dicts
from the DataModule, and a logger into a training loop with:
  - per-step logging (every log_every_n_steps gradient updates)
  - epoch-level validation across all registered metrics
  - early stopping on a configurable patience metric
  - orbax checkpointing of both the best and latest states
  - num_steps budget as an alternative stopping criterion
  - resume-from-checkpoint via fit(..., resume=True)
  - tqdm progress bars for epoch and batch loops

Usage
-----
    from training.losses import masked_mse, masked_rmse

    metrics_fns = {"masked_mse": masked_mse, "masked_rmse": masked_rmse}
    trainer     = Trainer(model, metrics_fns, config["trainer"])
    best_state  = trainer.fit(dm.train_arrays(), dm.val_arrays())

    # Resume interrupted training
    best_state = trainer.fit(dm.train_arrays(), dm.val_arrays(), resume=True)

    # Evaluate on test split using the saved the best checkpoint
    test_metrics = trainer.test(dm.test_arrays())

Config schema  (YAML  trainer:  block)
--------------------------------------
Required
    batch_size        int
    optimizer         str    name in the optimizer registry
    scheduler         str    name in the scheduler registry

Optional  (defaults shown)
    loss_key          str    first key in metrics_fns  — which metric is
                             differentiated during training
    optimizer_kwargs  dict   {}
    scheduler_kwargs  dict   {}
    num_epochs        int    1000
    num_steps         int    inf    stop at whichever limit is hit first
    patience          int    10
    patience_metric   str    'val/<first metrics_fns key>'
    patience_direction str   'lower_is_better'  or  'higher_is_better'
    max_grad_norm     float  None   gradient clipping; None = disabled
    checkpoint_dir    str    'checkpoints'
    log_every_n_steps int    50     gradient steps between step-level logs
    log_backend       str    'null'
    log_kwargs        dict   {}
    seed              int    42
    use_tqdm          bool   True

Checkpoint layout
-----------------
    checkpoint_dir/
      best/                orbax pytree — best state seen during training
      latest/              orbax pytree — state at end of most recent epoch
      latest_metadata.json training-loop scalars for resume
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state


from training.logger import create_logger
from training.optimizers import get_optimizer, get_scheduler
from utils.jax_core.helpers import create_rng, create_rng_dict

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    _tqdm = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _to_jax_batch(batch: dict) -> dict:
    """Convert array values in a loader batch to jax arrays.

    Handles batch['X'] being either a plain array or a dict of arrays,
    supporting models whose __call__ accepts a structured dict input.
    Non-array non-dict values are passed through unchanged.
    """
    result = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            result[k] = {k2: jnp.asarray(v2) for k2, v2 in v.items()}
        else:
            result[k] = jnp.asarray(v)
    return result


def _batch_head(batch: dict, n: int = 4) -> dict:
    """Return the first n rows from each value in a batch dict.

    Used to build a small example batch for model initialization.
    Handles dict-valued X and list-valued metadata fields.
    """
    result = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            result[k] = {k2: jnp.asarray(v2[:n]) for k2, v2 in v.items()}
        elif isinstance(v, list):
            result[k] = v[:n]
        else:
            result[k] = jnp.asarray(v[:min(n, v.shape[0])])
    return result


def _leading_dim(x) -> int:
    """Return the leading dimension of an array or a dict of arrays."""
    if isinstance(x, dict):
        return next(iter(x.values())).shape[0]
    return x.shape[0]


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    """Flax TrainState extended with BatchNorm state and dropout RNG.

    batch_stats : FrozenDict of running statistics for BatchNorm layers,
                  or None for models that do not use BatchNorm.
    rng         : PRNGKey that splits each training step so dropout
                  receives a fresh key without external threading.
    """
    batch_stats: Any
    rng: Any


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Single-device JAX / Flax trainer.

    Parameters
    ----------
    model : flax.linen.Module
        Any model whose __call__ accepts (x, train: bool, rngs=...).
        All registered models in this template satisfy this contract.
    metrics_fns : dict[str, Callable]
        Mapping of metric name to scalar loss/metric function
        (pred, target) -> scalar.  The key specified by config['loss_key']
        (default: first key) is differentiated during training.  All keys
        are evaluated during validation and test.
    config : dict
        The YAML  trainer:  block loaded into a plain dict.
    """

    def __init__(
        self,
        model:       Any,
        metrics_fns: dict[str, Callable],
        config:      dict,
    ) -> None:
        self.model       = model
        self.metrics_fns = metrics_fns

        # Which key in metrics_fns is differentiated during training
        self._loss_key = config.get("loss_key", next(iter(metrics_fns)))
        if self._loss_key not in metrics_fns:
            raise KeyError(
                f"loss_key '{self._loss_key}' not found in metrics_fns. "
                f"Available keys: {list(metrics_fns.keys())}"
            )

        # --- config ---
        self._batch_size      = config["batch_size"]
        self._num_epochs      = config.get("num_epochs", 1_000)
        self._num_steps       = config.get("num_steps",  float("inf"))
        self._patience        = config.get("patience",   10)
        self._patience_metric = config.get(
            "patience_metric", f"val/{self._loss_key}"
        )
        self._lower_is_better = (
            config.get("patience_direction", "lower_is_better")
            == "lower_is_better"
        )
        # run_dir co-locates checkpoints and logs under one root.
        # If set it takes precedence over checkpoint_dir.
        run_dir = config.get("run_dir")
        if run_dir is not None:
            _run = Path(run_dir)
            self._checkpoint_dir = _run / "checkpoints"
            _log_dir             = str(_run / "logs")
        else:
            self._checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
            _log_dir             = str(self._checkpoint_dir / "logs")

        self._log_every       = config.get("log_every_n_steps", 50)
        self._seed            = config.get("seed", 42)
        self._use_tqdm        = config.get("use_tqdm", True)

        # --- optimizer + scheduler ---
        schedule = get_scheduler(
            config["scheduler"],
            **config.get("scheduler_kwargs", {}),
        )
        self._optimizer = get_optimizer(
            config["optimizer"],
            learning_rate=schedule,
            **config.get("optimizer_kwargs", {}),
        )

        max_grad_norm = config.get("max_grad_norm")
        if max_grad_norm is not None:
            self._optimizer = optax.chain(
                optax.clip_by_global_norm(float(max_grad_norm)),
                self._optimizer,
            )

        # --- logger ---
        self._logger = create_logger(
            config.get("log_backend", "null"),
            log_dir=_log_dir,
            **config.get("log_kwargs", {}),
        )

        self._train_step:       Optional[Callable] = None
        self._eval_step:        Optional[Callable] = None
        self._global_step:      int   = 0
        self._last_state:       Optional[TrainState] = None
        self._best_metric_value: float = float("nan")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_state(self, exmp_batch: dict) -> TrainState:
        """Initialize model parameters, optimizer state, and RNGs."""
        rngs        = create_rng_dict(self._seed, keys=["params", "dropout"])
        variables   = self.model.init(rngs, exmp_batch["X"], train=False)
        params      = variables["params"]
        batch_stats = variables.get("batch_stats")

        self._build_steps(has_batch_stats=batch_stats is not None)

        state = TrainState.create(
            apply_fn    = self.model.apply,
            params      = params,
            tx          = self._optimizer,
            batch_stats = batch_stats,
            rng         = create_rng(self._seed + 1),
        )
        self._last_state = state
        return state

    # ------------------------------------------------------------------
    # Step construction
    # ------------------------------------------------------------------

    def _build_steps(self, has_batch_stats: bool) -> None:
        """Create and JIT-compile train_step and eval_step.

        metrics_fns is captured at build time as a compile-time constant.
        has_batch_stats is resolved once so JAX specialises the correct branch.
        The training loss is metrics_fns[loss_key]; all metrics are evaluated
        in eval_step.
        """
        model       = self.model
        metrics_fns = self.metrics_fns
        loss_key    = self._loss_key

        def train_step(state: TrainState, batch: dict):
            rng, dropout_rng = jax.random.split(state.rng)

            def compute_loss(params):
                rngs = {"dropout": dropout_rng}
                if has_batch_stats:
                    pred, updates = model.apply(
                        {"params": params, "batch_stats": state.batch_stats},
                        batch["X"], train=True, rngs=rngs,
                        mutable=["batch_stats"],
                    )
                    new_batch_stats = updates["batch_stats"]
                else:
                    pred = model.apply(
                        {"params": params},
                        batch["X"], train=True, rngs=rngs,
                    )
                    new_batch_stats = None
                loss = metrics_fns[loss_key](pred, batch["y"])
                return loss, new_batch_stats

            (loss, new_batch_stats), grads = jax.value_and_grad(
                compute_loss, has_aux=True
            )(state.params)

            new_state = state.apply_gradients(grads=grads)
            new_state = new_state.replace(
                batch_stats = new_batch_stats,
                rng         = rng,
            )
            return new_state, {loss_key: loss}

        def eval_step(state: TrainState, batch: dict):
            variables = {"params": state.params}
            if has_batch_stats:
                variables["batch_stats"] = state.batch_stats
            pred = model.apply(variables, batch["X"], train=False)
            return {k: fn(pred, batch["y"]) for k, fn in metrics_fns.items()}

        self._train_step = jax.jit(train_step)
        self._eval_step  = jax.jit(eval_step)

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        state:        TrainState,
        train_loader: Any,
        epoch:        int,
    ) -> tuple[TrainState, dict]:
        """One full pass over train_loader with optional tqdm inner bar.

        Accepts any iterable that yields {'X': array, 'y': array} dicts —
        an ArrayLoader, a PyTorch DataLoader with a JAX wrapper, etc.
        Batches are converted to JAX arrays per-step so the loader itself
        may yield numpy or PyTorch tensors.

        Stops early (mid-epoch) if the num_steps budget is reached.
        """
        losses   = []
        use_tqdm = self._use_tqdm and _TQDM_AVAILABLE
        n_steps  = len(train_loader) if hasattr(train_loader, "__len__") else None
        iterator = iter(train_loader)

        if use_tqdm:
            pbar     = _tqdm(iterator, total=n_steps,
                             desc=f"  epoch {epoch:4d}", unit="step", leave=False)
            iterator = pbar

        for batch in iterator:
            batch = _to_jax_batch(batch)
            state, step_metrics = self._train_step(state, batch)
            self._global_step  += 1
            loss_val            = float(step_metrics[self._loss_key])
            losses.append(loss_val)

            if use_tqdm:
                pbar.set_postfix({self._loss_key: f"{loss_val:.4f}"})

            if self._global_step % self._log_every == 0:
                self._logger.log_metrics(
                    {f"train/{self._loss_key}": loss_val},
                    step=self._global_step,
                )

            if self._global_step >= self._num_steps:
                break

        return state, {f"train/{self._loss_key}": float(np.mean(losses))}

    def _eval_model(
        self,
        state:      TrainState,
        val_loader: Any,
        prefix:     str = "val",
    ) -> dict:
        """Evaluate all metrics by iterating val_loader.

        Uses weighted averaging so an incomplete last batch (drop_last=False)
        contributes proportionally — the result equals computing each metric
        over all samples as if they were in one batch.

        Accepts the same loader types as _train_epoch.
        """
        totals  = {k: 0.0 for k in self.metrics_fns}
        n_total = 0
        for batch in val_loader:
            batch   = _to_jax_batch(batch)
            n       = _leading_dim(batch["X"])
            metrics = self._eval_step(state, batch)
            for k in metrics:
                totals[k] += float(metrics[k]) * n
            n_total += n
        if n_total == 0:
            return {f"{prefix}/{k}": float("nan") for k in totals}
        return {f"{prefix}/{k}": totals[k] / n_total for k in totals}

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def is_better(self, current: float, best: float) -> bool:
        """True when current is an improvement over best.

        NaN comparisons always return False — a NaN metric increments
        patience rather than crashing training.
        """
        if self._lower_is_better:
            return float(current) < float(best)
        return float(current) > float(best)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: TrainState) -> None:
        """Overwrite checkpoint_dir/best with the current state."""
        self._save_state(state, "best")

    def _save_state(self, state: TrainState, tag: str) -> None:
        path = self._checkpoint_dir / tag
        if path.exists():
            shutil.rmtree(path)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ocp.PyTreeCheckpointer().save(str(path), state)

    def _save_latest(
        self,
        state:         TrainState,
        epoch:         int,
        global_step:   int,
        best_metric:   float,
        patience_count: int,
    ) -> None:
        """Save the latest state + training-loop metadata every epoch.

        Used by resume-from-checkpoint to restart exactly where training
        was interrupted, not from the best epoch.
        """
        self._save_state(state, "latest")
        meta = {
            "epoch":         epoch,
            "global_step":   global_step,
            "best_metric":   str(best_metric),   # str handles ±inf
            "patience_count": patience_count,
        }
        meta_path = self._checkpoint_dir / "latest_metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

    def load_checkpoint(self, abstract_state: TrainState) -> TrainState:
        """Restore the best checkpoint.

        Parameters
        ----------
        abstract_state : TrainState
            An already-initialised TrainState providing the target structure.

        Raises
        ------
        FileNotFoundError
            If no best checkpoint has been saved.
        """
        path = self._checkpoint_dir / "best"
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at '{path}'. "
                "Run fit() first or check checkpoint_dir in config."
            )
        return ocp.PyTreeCheckpointer().restore(str(path), item=abstract_state)

    def _load_latest(
        self, abstract_state: TrainState
    ) -> tuple[TrainState, dict]:
        """Restore the latest state and training-loop metadata."""
        state_path = self._checkpoint_dir / "latest"
        meta_path  = self._checkpoint_dir / "latest_metadata.json"
        if not state_path.exists():
            raise FileNotFoundError(
                f"No latest checkpoint at '{state_path}'. "
                "Run fit() at least one epoch before resuming."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No training metadata at '{meta_path}'. "
                "The latest checkpoint may be from an older run."
            )
        state = ocp.PyTreeCheckpointer().restore(
            str(state_path), item=abstract_state
        )
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta["best_metric"]   = float(meta["best_metric"])   # "inf" → inf
        meta["global_step"]   = int(meta["global_step"])
        meta["epoch"]         = int(meta["epoch"])
        meta["patience_count"] = int(meta["patience_count"])
        return state, meta

    # ------------------------------------------------------------------
    # Public logging helpers
    # ------------------------------------------------------------------

    @property
    def logger(self):
        """The experiment logger (read-only)."""
        return self._logger

    def log_hyperparams(self, params: dict) -> None:
        """Log a hyperparameter dict to the experiment logger."""
        self._logger.log_hyperparams(params)

    def init_state(self, exmp_batch: dict) -> TrainState:
        """Initialise model and optimizer state from one example batch.

        Returns an abstract TrainState with the correct pytree structure,
        suitable for passing to load_checkpoint() without running fit().
        """
        return self._init_state(exmp_batch)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader:    Any,
        val_loader:      Any,
        resume:          bool = False,
        trial:           Any  = None,
        epoch_callbacks: Optional[list] = None,
    ) -> TrainState:
        """Train and return the best TrainState.

        Parameters
        ----------
        train_loader : iterable
            Yields {'X': array, 'y': array} dicts for training.
            Typically, dm.train_loader(batch_size).  Any re-iterable object
            works — ArrayLoader, a wrapped PyTorch DataLoader, etc.
        val_loader : iterable
            Yields batches for validation.  Typically, dm.val_loader(batch_size).
        resume : bool
            If True, load checkpoint_dir/latest/ and continue training from
            the last completed epoch.  The best checkpoint is preserved and
            only updated if a new improvement is found.
        epoch_callbacks : list of callables, optional
            Each callable is invoked after validation with signature
            ``callback(state: TrainState, epoch: int, global_step: int) -> None``.
            Useful for logging custom per-epoch diagnostics (e.g. attention
            entropy) without modifying the training loop.

        Returns
        -------
        TrainState
            State from the epoch with the best patience_metric value.
        """
        self._logger.log_hyperparams({
            "jax_backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "n_devices":   len(jax.devices()),
        })

        # Peek one batch to get shapes for model initialization.
        # The loader is re-iterable so the epoch loop restarts from scratch.
        exmp_batch = next(iter(train_loader))
        exmp_small = _batch_head(exmp_batch, n=4)
        state = self._init_state(exmp_small)

        # Training-loop state
        start_epoch    = 0
        best_metric    = float("inf") if self._lower_is_better else float("-inf")
        patience_count = 0
        best_state     = state
        self._global_step = 0

        if resume:
            state, meta    = self._load_latest(state)
            start_epoch    = meta["epoch"] + 1
            self._global_step = meta["global_step"]
            best_metric    = meta["best_metric"]
            patience_count = meta["patience_count"]
            best_state     = state
            print(
                f"Resuming from epoch {start_epoch}, "
                f"step {self._global_step}, "
                f"best {self._patience_metric}={best_metric:.5f}"
            )

        use_tqdm   = self._use_tqdm and _TQDM_AVAILABLE
        epoch_range = range(start_epoch, self._num_epochs)

        if use_tqdm:
            epoch_bar  = _tqdm(
                epoch_range, desc="Training", unit="epoch",
                initial=start_epoch, total=self._num_epochs,
            )
            epoch_iter = epoch_bar
        else:
            epoch_bar  = None
            epoch_iter = epoch_range

        for epoch in epoch_iter:
            state, train_metrics = self._train_epoch(state, train_loader, epoch)
            val_metrics = self._eval_model(state, val_loader, prefix="val")

            # Epoch-level log (step = epoch for clean separation from
            # within-epoch step-level logs)
            self._logger.log_metrics(
                {**train_metrics, **val_metrics}, step=epoch
            )

            if epoch_callbacks:
                for cb in epoch_callbacks:
                    cb(state, epoch, self._global_step)

            if self._patience_metric not in val_metrics:
                raise KeyError(
                    f"patience_metric '{self._patience_metric}' not found in "
                    f"val_metrics. Available: {list(val_metrics.keys())}. "
                    "Check patience_metric in config."
                )
            current  = val_metrics[self._patience_metric]
            improved = self.is_better(current, best_metric)

            # optuna pruning: report intermediate value and stop early if
            # the trial looks unpromising.  Lazy import keeps optuna optional.
            if trial is not None:
                import optuna as _optuna
                trial.report(float(current), epoch)
                if trial.should_prune():
                    self._best_metric_value = best_metric
                    raise _optuna.TrialPruned()

            if improved:
                best_metric    = current
                best_state     = state
                patience_count = 0
                self.save_checkpoint(state)
            else:
                patience_count += 1

            # Save latest state + metadata every epoch (enables resume)
            self._save_latest(state, epoch, self._global_step,
                              best_metric, patience_count)

            train_key = f"train/{self._loss_key}"
            summary = (
                f"epoch {epoch:4d} | "
                f"{train_key} {train_metrics[train_key]:.5f} | "
                f"{self._patience_metric} {current:.5f} | "
                f"patience {patience_count}/{self._patience}"
                + (" [best]" if improved else "")
            )

            if use_tqdm and epoch_bar is not None:
                epoch_bar.set_postfix({
                    f"tr/{self._loss_key}": f"{train_metrics[train_key]:.4f}",
                    "val":       f"{current:.4f}",
                    "patience":  f"{patience_count}/{self._patience}",
                })
            else:
                print(summary)

            if patience_count >= self._patience:
                msg = f"Early stopping at epoch {epoch}."
                if use_tqdm and epoch_bar is not None:
                    epoch_bar.write(msg)
                else:
                    print(msg)
                break

            if self._global_step >= self._num_steps:
                msg = f"Step budget reached ({self._num_steps})."
                if use_tqdm and epoch_bar is not None:
                    epoch_bar.write(msg)
                else:
                    print(msg)
                break

        self._logger.finalize("completed")
        self._best_metric_value = best_metric
        return best_state

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------

    def test(self, test_loader: Any) -> dict:
        """Evaluate all metrics on the test split using the best checkpoint.

        Parameters
        ----------
        test_loader : iterable
            Yields {'X': array, 'y': array} dicts.
            Typically, dm.test_loader(batch_size).

        Returns
        -------
        dict
            {'test/{metric_name}': float, ...} for every key in metrics_fns.

        Raises
        ------
        RuntimeError  if fit() has not been called.
        FileNotFoundError  if no best checkpoint exists.
        """
        if self._last_state is None:
            raise RuntimeError(
                "Model not initialised. Call fit() before test()."
            )
        best_state = self.load_checkpoint(self._last_state)
        metrics    = self._eval_model(best_state, test_loader, prefix="test")
        self._logger.log_metrics(metrics, step=0)
        return metrics
