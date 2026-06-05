"""
training/tuner.py

Hyperparameter search over Trainer configs via Optuna.

Design
------
The Tuner wraps Optuna's study/trial machinery around our existing Trainer.
The user provides three callables that describe the problem:

  suggest_fn(trial, base_config) -> config dict
      Samples hyperparameters from the trial and returns a full trainer
      config dict.  Use apply_search_space() to handle the sampling
      boilerplate, then merge sampled values into the config manually:

          def suggest_fn(trial, base_config):
              hp = apply_search_space(trial, SEARCH_SPACE)
              cfg = copy.deepcopy(base_config)
              cfg['scheduler_kwargs']['peak_value'] = hp['lr']
              cfg['optimizer_kwargs']['weight_decay'] = hp['weight_decay']
              return cfg

  model_fn(config) -> flax.linen.Module
      Constructs the model from the sampled config.  Architecture HPs
      (hidden_features, n_layers, dropout_rate) are typically sampled in
      suggest_fn and stored in config['model'] for model_fn to read.

  train_loader_fn() / val_loader_fn() -> iterable
      Called fresh each trial so batch-size HPs can also be tuned.

Pruning
-------
The Trainer reports the patience metric to Optuna after every epoch.
If Optuna's pruner decides the trial is unpromising it raises
optuna.TrialPruned, which is caught by study.optimize() and the next
trial starts immediately.  Use MedianPruner (default) for typical
regression/classification tasks.

Persistence
-----------
Pass a SQLite URL as storage to persist the study across Python sessions:

    storage='sqlite:///runs/hp_search.db'

The study can be reloaded from the same URL to continue an interrupted
search or to analyse results later.

Usage
-----
    from training.tuner import Tuner, apply_search_space
    import copy

    SEARCH_SPACE = {
        'lr':           {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
        'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True},
        'dropout_rate': {'type': 'float', 'low': 0.0,  'high': 0.5},
    }

    def suggest_fn(trial, base_config):
        hp  = apply_search_space(trial, SEARCH_SPACE)
        cfg = copy.deepcopy(base_config)
        cfg['scheduler_kwargs']['peak_value'] = hp['lr']
        cfg['optimizer_kwargs']['weight_decay'] = hp['weight_decay']
        return cfg

    def model_fn(config):
        return get_mlp('SIREN', out_features=6, hidden_features=256, n_layers=4)

    tuner = Tuner(
        suggest_fn       = suggest_fn,
        base_config      = config['trainer'],
        model_fn         = model_fn,
        metrics_fns      = {'masked_mse': masked_mse},
        train_loader_fn  = lambda: dm.train_loader(batch_size=256),
        val_loader_fn    = lambda: dm.val_loader(batch_size=256),
        study_name       = 'ibtracs_siren',
        direction        = 'minimize',
        storage          = 'sqlite:///runs/hp_search.db',
    )
    tuner.run(n_trials=25)
    print(tuner.best_params)
"""

from __future__ import annotations

import copy
import gc
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Search-space helper
# ---------------------------------------------------------------------------

def apply_search_space(trial: Any, space: dict) -> dict:
    """Sample from a structured search-space description.

    Each key in ``space`` defines one hyperparameter.  The value is a dict
    with at least a ``type`` key.

    Supported types and their extra keys
    -------------------------------------
    float       low, high, step (optional), log (bool, optional)
    int         low, high, step (optional), log (bool, optional)
    categorical choices (list)

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial.
    space : dict
        Mapping of HP name → spec dict.

    Returns
    -------
    dict
        {hp_name: sampled_value} for every entry in space.

    Example
    -------
    >>> hp = apply_search_space(trial, {
    ...     'lr':       {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
    ...     'n_layers': {'type': 'int',   'low': 2,    'high': 8},
    ...     'act':      {'type': 'categorical', 'choices': ['relu', 'sine']},
    ... })
    """
    sampled = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            sampled[name] = trial.suggest_float(
                name, spec["low"], spec["high"],
                step=spec.get("step"), log=spec.get("log", False),
            )
        elif kind == "int":
            sampled[name] = trial.suggest_int(
                name, spec["low"], spec["high"],
                step=spec.get("step", 1), log=spec.get("log", False),
            )
        elif kind == "categorical":
            sampled[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(
                f"apply_search_space: unknown type '{kind}' for HP '{name}'. "
                "Use 'float', 'int', or 'categorical'."
            )
    return sampled


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------

class Tuner:
    """Optuna hyperparameter search over Trainer configs.

    Parameters
    ----------
    suggest_fn : callable(trial, base_config) -> config dict
        Samples hyperparameters for one trial and returns the full
        trainer config dict to pass to Trainer().
    base_config : dict
        Base trainer config with all non-tuned hyperparameters.
        Copied deeply inside each trial — never mutated.
    model_fn : callable(config) -> flax.linen.Module
        Builds the model from the trial's config dict.
    metrics_fns : dict[str, Callable]
        Metric functions passed to Trainer unchanged.
    train_loader_fn : callable() -> iterable
        Called fresh each trial.  Must return a re-iterable loader.
    val_loader_fn : callable() -> iterable
        Called fresh each trial.
    study_name : str
        Optuna study identifier.
    direction : str
        'minimize' for loss, 'maximize' for accuracy / R².
    storage : str or None
        Optuna storage URL.  'sqlite:///path/to/file.db' for persistence
        across sessions.  None = in-memory (results lost on exit).
    n_startup_trials : int
        Trials to run before the pruner becomes active.
    n_warmup_steps : int
        Epochs per trial before pruning is checked.
    """

    def __init__(
        self,
        suggest_fn:      Callable,
        base_config:     dict,
        model_fn:        Callable,
        metrics_fns:     dict,
        train_loader_fn: Callable,
        val_loader_fn:   Callable,
        study_name:      str   = "hp_search",
        direction:       str   = "minimize",
        storage:         Optional[str] = None,
        n_startup_trials: int  = 5,
        n_warmup_steps:   int  = 50,
    ) -> None:
        self._suggest_fn        = suggest_fn
        self._base_config       = base_config
        self._model_fn          = model_fn
        self._metrics_fns       = metrics_fns
        self._train_loader_fn   = train_loader_fn
        self._val_loader_fn     = val_loader_fn
        self._study_name        = study_name
        self._direction         = direction
        self._storage           = storage
        self._n_startup_trials  = n_startup_trials
        self._n_warmup_steps    = n_warmup_steps
        self._study             = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, n_trials: int = 25) -> "optuna.Study":
        """Run the hyperparameter search for up to n_trials trials.

        If a persistent study already exists at ``storage`` it is resumed
        automatically.  Only the remaining (n_trials - completed) are run.

        Parameters
        ----------
        n_trials : int
            Total number of trials to run (including any already completed
            when resuming).

        Returns
        -------
        optuna.Study
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self._study is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials = self._n_startup_trials,
                n_warmup_steps   = self._n_warmup_steps,
            )
            self._study = optuna.create_study(
                study_name     = self._study_name,
                storage        = self._storage,
                direction      = self._direction,
                pruner         = pruner,
                load_if_exists = True,
            )

        remaining = n_trials - len(self._study.trials)
        if remaining <= 0:
            print(f"Study '{self._study_name}' already has {len(self._study.trials)} "
                  f"trials — requested {n_trials}.  Nothing to run.")
            return self._study

        self._study.optimize(
            self._make_objective(),
            n_trials = remaining,
            n_jobs   = 1,
        )
        return self._study

    @property
    def study(self) -> "optuna.Study":
        """The underlying optuna.Study (available after run())."""
        if self._study is None:
            raise RuntimeError("No study yet.  Call run() first.")
        return self._study

    @property
    def best_params(self) -> dict:
        """Hyperparameters of the best trial."""
        return self.study.best_trial.params

    @property
    def best_value(self) -> float:
        """Objective value of the best trial."""
        return self.study.best_value

    def summary(self) -> None:
        """Print a table of all completed trials sorted by objective value."""
        import optuna
        completed = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]
        print(f"\nStudy: {self._study_name}  |  direction: {self._direction}")
        print(f"  completed: {len(completed)}   pruned: {len(pruned)}")
        print(f"  best value: {self.best_value:.6f}")
        print(f"  best params:")
        for k, v in self.best_params.items():
            print(f"    {k}: {v}")

        print(f"\n  {'trial':>6}  {'value':>12}  params")
        print(f"  {'─'*6}  {'─'*12}  {'─'*40}")
        reverse = (self._direction == "maximize")
        for t in sorted(completed, key=lambda t: t.value, reverse=reverse):
            param_str = "  ".join(f"{k}={v:.4g}" if isinstance(v, float)
                                  else f"{k}={v}" for k, v in t.params.items())
            print(f"  {t.number:>6}  {t.value:>12.6f}  {param_str}")

    # ------------------------------------------------------------------
    # Objective construction
    # ------------------------------------------------------------------

    def _make_objective(self) -> Callable:
        """Return the Optuna objective function for this study."""
        suggest_fn      = self._suggest_fn
        base_config     = self._base_config
        model_fn        = self._model_fn
        metrics_fns     = self._metrics_fns
        train_loader_fn = self._train_loader_fn
        val_loader_fn   = self._val_loader_fn

        def objective(trial):
            import optuna
            from training.trainer import Trainer

            # Deep-copy so suggest_fn cannot mutate shared base_config
            config = suggest_fn(trial, copy.deepcopy(base_config))

            # Each trial gets its own checkpoint subdirectory
            base_ckpt = Path(config.get("checkpoint_dir", "checkpoints"))
            config["checkpoint_dir"] = str(base_ckpt / f"trial_{trial.number}")

            # Suppress inner tqdm bars during search
            config["use_tqdm"] = False

            model   = model_fn(config)
            trainer = Trainer(model, metrics_fns, config)

            try:
                trainer.fit(
                    train_loader_fn(),
                    val_loader_fn(),
                    trial=trial,
                )
                result = trainer._best_metric_value
            except optuna.TrialPruned:
                result = trainer._best_metric_value
                raise
            finally:
                del trainer
                gc.collect()

            return result

        return objective
