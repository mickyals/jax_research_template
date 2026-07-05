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
    from training.losses import mse

    metrics_fns = {"mse": mse, "rmse": lambda p, t: mse(p, t) ** 0.5}
    trainer     = Trainer(model, metrics_fns, config["trainer"])
    best_state  = trainer.fit(dm.train_loader(batch_size), dm.val_loader(batch_size))

    # Resume interrupted training
    best_state = trainer.fit(dm.train_loader(batch_size), dm.val_loader(batch_size), resume=True)

    # Evaluate on test split using the saved best checkpoint
    test_metrics = trainer.test(dm.test_loader(batch_size))

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
                             any 'val/<metric>' works, as does
                             'train/<loss_key>' — early stopping on the
                             training loss (memorisation/overfit gates)
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

    The top-level key 'meta' is reserved for non-model sample metadata
    (attribution strings, diagnostics) and is dropped here — it must never
    reach the jitted train/eval steps, which trace every batch leaf.
    """
    result = {}
    for k, v in batch.items():
        if k == 'meta':
            continue
        if isinstance(v, dict):
            result[k] = {k2: jnp.asarray(v2) for k2, v2 in v.items()}
        else:
            result[k] = jnp.asarray(v)
    return result


def _batch_head(batch: dict, n: int = 4) -> dict:
    """Return the first n rows from each value in a batch dict.

    Used to build a small example batch for model initialization.
    Handles dict-valued X and list-valued metadata fields. The reserved
    'meta' key is dropped (see _to_jax_batch).
    """
    result = {}
    for k, v in batch.items():
        if k == 'meta':
            continue
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
        The loss entry may also be a training.losses.LossStack (or any
        callable with ``needs_model=True``): it is then called with
        ``params/apply_fn/batch`` keywords so model terms (weight-space
        penalties, physics residuals) can participate; multi-term stacks
        get their per-term values logged under ``<loss_key>/<term>``.
    config : dict
        The YAML  trainer:  block loaded into a plain dict.
    """

    def __init__(
        self,
        model:       Any,
        metrics_fns: dict[str, Callable],
        config:      dict,
        logger:      Any = None,
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
            _run = Path(run_dir).resolve()        # resolve to absolute — orbax requires it
            self._checkpoint_dir = _run / "checkpoints"
            _log_dir             = str(_run / "logs")
        else:
            self._checkpoint_dir = Path(
                config.get("checkpoint_dir", "checkpoints")
            ).resolve()
            _log_dir             = str(self._checkpoint_dir / "logs")

        # Store for startup summary print
        self._optimizer_name   = config["optimizer"]
        self._optimizer_kwargs = config.get("optimizer_kwargs", {})
        self._scheduler_name   = config["scheduler"]
        self._scheduler_kwargs = config.get("scheduler_kwargs", {})
        self._max_grad_norm_val = config.get("max_grad_norm")

        self._log_every       = config.get("log_every_n_steps", 50)
        self._seed            = config.get("seed", 42)
        self._use_tqdm        = config.get("use_tqdm", True)

        # --- profiling ---
        # profile: true traces the first profile_steps training steps of the
        # run with the JAX profiler, written to <log_dir>/profile. View with
        # TensorBoard's Profile plugin (WandB cannot render XLA traces).
        # The first traced step includes jit compilation — read the later
        # steps in the trace for steady-state timings.
        self._profile          = bool(config.get("profile", False))
        self._profile_steps    = int(config.get("profile_steps", 5))
        self._profile_dir      = str(Path(_log_dir) / "profile")
        self._profile_active   = False
        self._profile_done     = False

        # --- optimizer + scheduler ---
        schedule = get_scheduler(
            config["scheduler"],
            **config.get("scheduler_kwargs", {}),
        )
        self._schedule  = schedule   # stored for per-step LR logging
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
        # A pre-built logger may be passed in (e.g. so the caller can start the
        # run — and its stdout capture — BEFORE the Trainer is constructed);
        # otherwise build one from the config here.
        self._logger = logger if logger is not None else create_logger(
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

        Loss functions that carry ``needs_model=True`` (e.g. a
        training.losses.LossStack with model terms) are called with
        ``params/apply_fn/batch`` keywords in both steps, so terms can
        penalise parameters or re-differentiate through the model; plain
        ``(pred, y)`` losses keep the unchanged path. A multi-term stack's
        ``detailed`` method additionally surfaces per-term values in the
        step metrics under ``<loss_key>/<term>`` (single-term stacks emit
        no redundant extra curve). Both are resolved here, at build time,
        as compile-time constants.
        """
        model       = self.model
        metrics_fns = self.metrics_fns
        loss_key    = self._loss_key

        loss_fn     = metrics_fns[loss_key]
        needs_model = bool(getattr(loss_fn, "needs_model", False))
        detailed    = getattr(loss_fn, "detailed", None)
        log_terms   = (detailed is not None
                       and len(getattr(loss_fn, "term_names", ())) > 1)

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
                kw = ({"params": params, "apply_fn": model.apply,
                       "batch": batch} if needs_model else {})
                if log_terms:
                    loss, term_vals = detailed(pred, batch["y"], **kw)
                else:
                    loss, term_vals = loss_fn(pred, batch["y"], **kw), {}
                return loss, (new_batch_stats, term_vals)

            (loss, (new_batch_stats, term_vals)), grads = jax.value_and_grad(
                compute_loss, has_aux=True
            )(state.params)

            new_state = state.apply_gradients(grads=grads)
            new_state = new_state.replace(
                batch_stats = new_batch_stats,
                rng         = rng,
            )
            step_metrics = {loss_key: loss}
            step_metrics.update(
                {f"{loss_key}/{k}": v for k, v in term_vals.items()})
            return new_state, step_metrics

        def eval_step(state: TrainState, batch: dict):
            variables = {"params": state.params}
            if has_batch_stats:
                variables["batch_stats"] = state.batch_stats
            pred = model.apply(variables, batch["X"], train=False)
            out = {}
            for k, fn in metrics_fns.items():
                if k == loss_key and needs_model:
                    out[k] = fn(pred, batch["y"], params=state.params,
                                apply_fn=model.apply, batch=batch)
                else:
                    out[k] = fn(pred, batch["y"])
            return out

        self._train_step = jax.jit(train_step)
        self._eval_step  = jax.jit(eval_step)

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        state:          TrainState,
        train_loader:   Any,
        epoch:          int,
        step_callbacks: Optional[list] = None,
    ) -> tuple[TrainState, dict]:
        """One full pass over train_loader with optional tqdm inner bar.

        Accepts any iterable that yields {'X': array, 'y': array} dicts —
        an ArrayLoader, a PyTorch DataLoader with a JAX wrapper, etc.
        Batches are converted to JAX arrays per-step so the loader itself
        may yield numpy or PyTorch tensors.

        Stops early (mid-epoch) if the num_steps budget is reached.

        Parameters
        ----------
        step_callbacks : list of (callable, int), optional
            Each entry is ``(fn, every_n_steps)`` where ``fn`` has the
            signature ``fn(state, epoch, global_step) -> None``.
            Called inside the step loop whenever
            ``global_step % every_n_steps == 0``.
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

            if self._profile and not self._profile_done and not self._profile_active:
                Path(self._profile_dir).mkdir(parents=True, exist_ok=True)
                jax.profiler.start_trace(self._profile_dir)
                self._profile_active = True
                self._profile_start  = self._global_step

            if self._profile_active:
                with jax.profiler.StepTraceAnnotation(
                    "train_step", step_num=self._global_step
                ):
                    state, step_metrics = self._train_step(state, batch)
            else:
                state, step_metrics = self._train_step(state, batch)

            self._global_step  += 1
            loss_val            = float(step_metrics[self._loss_key])
            losses.append(loss_val)

            if (self._profile_active
                    and self._global_step - self._profile_start >= self._profile_steps):
                self._stop_profile()

            if use_tqdm:
                pbar.set_postfix({self._loss_key: f"{loss_val:.4f}"})

            if self._global_step % self._log_every == 0:
                # Compute all metrics on the current training batch.
                # eval_step is a pure forward pass — no gradients computed.
                # We override the loss key with the exact value from the backward
                # pass to avoid floating-point differences from a second forward.
                all_metrics = {
                    k: float(v)
                    for k, v in self._eval_step(state, batch).items()
                }
                all_metrics[self._loss_key] = loss_val
                # Per-term loss values (multi-term LossStack) ride along from
                # the backward pass under '<loss_key>/<term>'.
                for k, v in step_metrics.items():
                    if k != self._loss_key:
                        all_metrics[k] = float(v)
                log_dict = {f"train/{k}": v for k, v in all_metrics.items()}
                log_dict['train/lr'] = float(self._schedule(self._global_step))
                self._logger.log_metrics(log_dict, step=self._global_step)

            # Step-level callbacks — each fires at its own configurable frequency.
            if step_callbacks:
                for cb_fn, cb_every in step_callbacks:
                    if self._global_step % cb_every == 0:
                        cb_fn(state, epoch, self._global_step)

            if self._global_step >= self._num_steps:
                break

        # Tiny loaders: the epoch may end before profile_steps completes.
        if self._profile_active:
            self._stop_profile()

        return state, {f"train/{self._loss_key}": float(np.mean(losses))}

    def _stop_profile(self) -> None:
        """Finish the JAX profiler trace (loss floats above already forced
        device sync, so the traced steps are fully captured).

        The trace is handed to the logger as an artifact — WandB uploads
        it to the run's artifact store; TensorBoard/Null leave it on disk
        and print where it lives (with the viewer hint where it applies).
        """
        jax.profiler.stop_trace()
        self._profile_active = False
        self._profile_done   = True
        print(
            f"[profiler] traced training steps written to {self._profile_dir}"
        )
        self._logger.log_artifact(
            "profile-trace", self._profile_dir, artifact_type="profile",
        )

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

        # Progress bar for the eval pass (val during training, test at the end)
        # so a finished train bar isn't followed by a silent wait.
        iterator = val_loader
        if self._use_tqdm and _TQDM_AVAILABLE:
            n_batches = len(val_loader) if hasattr(val_loader, "__len__") else None
            iterator  = _tqdm(val_loader, total=n_batches,
                              desc=f"  {prefix:<5}", unit="batch", leave=False)

        for batch in iterator:
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

    def write_manifest(self, manifest: dict, filename: str = "manifest.json") -> None:
        """Write a run manifest next to checkpoints and push it to the logger.

        Three destinations: the file under checkpoint_dir is the source of
        truth; ``log_hyperparams`` puts a browsable copy in the run config; and
        ``log_artifact`` uploads the file itself as a per-run artifact (wandb:
        the run's Artifacts tab, downloadable; TensorBoard/null: the path is
        printed). The last one is what makes the manifest visible *per run* in
        wandb rather than buried in the config.

        Parameters
        ----------
        manifest : dict
            JSON-serialisable summary of what this run trained on, e.g.
            from DataModule.manifest().
        filename : str
            Name of the JSON file written under checkpoint_dir. Default
            'manifest.json'.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_dir / filename
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        self._logger.log_hyperparams({"manifest": manifest})
        self._logger.log_artifact("manifest", path, "manifest")

    def init_state(self, exmp_batch: dict) -> TrainState:
        """Initialise model and optimizer state from one example batch.

        Returns an abstract TrainState with the correct pytree structure,
        suitable for passing to load_checkpoint() without running fit().
        """
        return self._init_state(exmp_batch)

    # ------------------------------------------------------------------
    # Startup summary
    # ------------------------------------------------------------------

    def _print_startup_summary(self, steps_per_epoch: Optional[int]) -> None:
        """Print a one-time trainer configuration block to stdout.

        Called once at the start of fit() so the operator can confirm
        optimizer, schedule, and checkpoint settings before waiting for
        JIT compilation.
        """
        direction  = "lower_is_better" if self._lower_is_better else "higher_is_better"
        arrow      = "↓" if self._lower_is_better else "↑"
        clip_str   = str(self._max_grad_norm_val) if self._max_grad_norm_val else "none"
        steps_str  = f"~{steps_per_epoch:,}" if steps_per_epoch else "unknown"
        backend    = jax.default_backend()
        n_dev      = len(jax.devices())

        # Format scheduler kwargs as key=value pairs on one line
        sched_kw = "  ".join(f"{k}={v}" for k, v in self._scheduler_kwargs.items())
        opt_kw   = "  ".join(f"{k}={v}" for k, v in self._optimizer_kwargs.items())

        print()
        print("─" * 58)
        print("Trainer")
        print(f"  backend    : {backend}  ·  {n_dev} device{'s' if n_dev != 1 else ''}")
        print(f"  optimizer  : {self._optimizer_name}"
              + (f"   ({opt_kw})" if opt_kw else ""))
        print(f"  scheduler  : {self._scheduler_name}"
              + (f"   ({sched_kw})" if sched_kw else ""))
        print(f"  batch size : {self._batch_size}")
        print(f"  epochs     : {self._num_epochs}   "
              f"patience={self._patience} on {self._patience_metric} ({arrow} {direction})")
        print(f"  steps/ep   : {steps_str}")
        print(f"  grad clip  : {clip_str}")
        print(f"  checkpoints: {self._checkpoint_dir}")
        print("─" * 58)
        print()

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
        step_callbacks:  Optional[list] = None,
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
            Useful for per-epoch diagnostics (e.g. geographic attention maps).
        step_callbacks : list of (callable, int), optional
            Each entry is ``(fn, every_n_steps)`` where ``fn`` has signature
            ``fn(state: TrainState, epoch: int, global_step: int) -> None``.
            Called inside the step loop at the specified frequency.
            Useful for intra-epoch diagnostics (e.g. attention entropy curves).

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

        # Print trainer config once so the operator can sanity-check before
        # waiting through the first JIT compilation.
        steps_per_epoch = (
            len(train_loader) if hasattr(train_loader, "__len__") else None
        )
        self._print_startup_summary(steps_per_epoch)

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
            state, train_metrics = self._train_epoch(
                state, train_loader, epoch, step_callbacks=step_callbacks
            )
            val_metrics = self._eval_model(state, val_loader, prefix="val")

            # Epoch-level log — use global_step so all metrics share one x-axis
            # with the step-level train/loss and attention entropy curves.
            self._logger.log_metrics(
                {**train_metrics, **val_metrics}, step=self._global_step
            )

            if epoch_callbacks:
                for cb in epoch_callbacks:
                    cb(state, epoch, self._global_step)

            epoch_metrics = {**train_metrics, **val_metrics}
            if self._patience_metric not in epoch_metrics:
                raise KeyError(
                    f"patience_metric '{self._patience_metric}' not found in "
                    f"epoch metrics. Available: {list(epoch_metrics.keys())}. "
                    "Check patience_metric in config."
                )
            current  = epoch_metrics[self._patience_metric]
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

        # Do NOT finalize here — the caller (train.py) owns the logger
        # lifecycle and will call trainer.logger.finalize() after test().
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
        # Log at the final training step so test metrics appear on the same
        # x-axis as training metrics rather than at the arbitrary step=0.
        self._logger.log_metrics(metrics, step=self._global_step)
        return metrics
