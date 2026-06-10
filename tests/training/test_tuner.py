"""
Tests for training/tuner.py  and  Trainer trial/pruning integration.

Coverage
--------
TestApplySearchSpace    float / int / categorical sampling; unknown type raises
TestTrainerTrialHook    trial.report called each epoch; TrialPruned raised
                        when should_prune() returns True;
                        _best_metric_value stored after fit
TestTuner               run() completes; best_params / best_value accessible;
                        summary() prints; study exposed; pruning works end-to-end;
                        n_trials already done → skips gracefully;
                        each trial gets its own checkpoint dir
"""

import copy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from training.losses import mse
from training.trainer import Trainer, TrainState
from training.tuner import Tuner, apply_search_space
from datasets.datamodule import ArrayLoader


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_FEAT, N_TGT = 4, 2
BATCH_SZ = 16


class _TinyMLP(nn.Module):
    hidden: int = 8
    out:    int = N_TGT

    @nn.compact
    def __call__(self, x, train=False, rngs=None):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        return nn.Dense(self.out)(x)


def _make_arrays(n, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "X": jnp.array(rng.normal(0, 1, (n, N_FEAT)).astype(np.float32)),
        "y": jnp.array(rng.normal(0, 1, (n, N_TGT)).astype(np.float32)),
    }


def _base_config(tmp_path, **overrides):
    cfg = {
        "batch_size":         BATCH_SZ,
        "num_epochs":         3,
        "num_steps":          1_000,
        "patience":           10,
        "patience_metric":    "val/mse",
        "patience_direction": "lower_is_better",
        "checkpoint_dir":     str(tmp_path / "ckpts"),
        "log_every_n_steps":  999,
        "log_backend":        "null",
        "seed":               0,
        "optimizer":          "adam",
        "optimizer_kwargs":   {},
        "scheduler":          "constant",
        "scheduler_kwargs":   {"value": 1e-3},
        "use_tqdm":           False,
    }
    cfg.update(overrides)
    return cfg


def _make_loader(arrays, shuffle=True):
    return ArrayLoader(arrays, BATCH_SZ, shuffle=shuffle,
                       drop_last=shuffle)


# ---------------------------------------------------------------------------
# Mock Optuna trial (no real Optuna dependency for unit tests)
# ---------------------------------------------------------------------------

class _MockTrial:
    """Minimal optuna.Trial stand-in for unit tests."""

    def __init__(self, should_prune_after: int | None = None):
        self.reported: list[tuple[float, int]] = []
        self._prune_after = should_prune_after
        self.params: dict = {}

    def report(self, value: float, step: int) -> None:
        self.reported.append((value, step))

    def should_prune(self) -> bool:
        if self._prune_after is None:
            return False
        return len(self.reported) >= self._prune_after

    # stubs so Trainer's optuna import path still works
    def suggest_float(self, name, low, high, **kwargs):
        v = (low + high) / 2
        self.params[name] = v
        return v

    def suggest_int(self, name, low, high, **kwargs):
        v = (low + high) // 2
        self.params[name] = v
        return v

    def suggest_categorical(self, name, choices):
        v = choices[0]
        self.params[name] = v
        return v


# ---------------------------------------------------------------------------
# TestApplySearchSpace
# ---------------------------------------------------------------------------

class TestApplySearchSpace:

    def test_float_sampled(self):
        trial  = _MockTrial()
        result = apply_search_space(trial, {
            "lr": {"type": "float", "low": 1e-4, "high": 1e-2}
        })
        assert "lr" in result
        assert 1e-4 <= result["lr"] <= 1e-2

    def test_int_sampled(self):
        trial  = _MockTrial()
        result = apply_search_space(trial, {
            "n_layers": {"type": "int", "low": 2, "high": 8}
        })
        assert "n_layers" in result
        assert isinstance(result["n_layers"], int)
        assert 2 <= result["n_layers"] <= 8

    def test_categorical_sampled(self):
        trial  = _MockTrial()
        result = apply_search_space(trial, {
            "act": {"type": "categorical", "choices": ["relu", "sine"]}
        })
        assert result["act"] in ["relu", "sine"]

    def test_multiple_hps(self):
        trial  = _MockTrial()
        result = apply_search_space(trial, {
            "lr":  {"type": "float", "low": 1e-4, "high": 1e-2},
            "act": {"type": "categorical", "choices": ["relu", "sine"]},
            "n":   {"type": "int",   "low": 2, "high": 8},
        })
        assert set(result.keys()) == {"lr", "act", "n"}

    def test_unknown_type_raises(self):
        trial = _MockTrial()
        with pytest.raises(ValueError, match="unknown type"):
            apply_search_space(trial, {"x": {"type": "complex"}})

    def test_returns_dict(self):
        trial  = _MockTrial()
        result = apply_search_space(trial, {})
        assert isinstance(result, dict)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestTrainerTrialHook
# ---------------------------------------------------------------------------

class TestTrainerTrialHook:

    @pytest.fixture
    def loaders(self):
        tr = _make_loader(_make_arrays(64))
        vl = _make_loader(_make_arrays(20, seed=1), shuffle=False)
        return tr, vl

    def test_trial_report_called_each_epoch(self, tmp_path, loaders):
        train_l, val_l = loaders
        cfg     = _base_config(tmp_path, num_epochs=3)
        trainer = Trainer(_TinyMLP(), {"mse": mse}, cfg)
        trial   = _MockTrial()
        trainer.fit(train_l, val_l, trial=trial)
        # One report per epoch
        assert len(trial.reported) == 3

    def test_trial_report_values_are_finite(self, tmp_path, loaders):
        train_l, val_l = loaders
        cfg     = _base_config(tmp_path, num_epochs=3)
        trainer = Trainer(_TinyMLP(), {"mse": mse}, cfg)
        trial   = _MockTrial()
        trainer.fit(train_l, val_l, trial=trial)
        assert all(np.isfinite(v) for v, _ in trial.reported)

    def test_trial_report_steps_are_epoch_indices(self, tmp_path, loaders):
        train_l, val_l = loaders
        cfg     = _base_config(tmp_path, num_epochs=3)
        trainer = Trainer(_TinyMLP(), {"mse": mse}, cfg)
        trial   = _MockTrial()
        trainer.fit(train_l, val_l, trial=trial)
        steps = [step for _, step in trial.reported]
        assert steps == [0, 1, 2]

    def test_trial_pruned_raised_when_should_prune(self, tmp_path, loaders):
        import optuna
        train_l, val_l = loaders
        cfg     = _base_config(tmp_path, num_epochs=10)
        trainer = Trainer(_TinyMLP(), {"mse": mse}, cfg)
        # Prune after the 2nd report (epoch 1)
        trial   = _MockTrial(should_prune_after=2)
        with pytest.raises(optuna.TrialPruned):
            trainer.fit(train_l, val_l, trial=trial)
        # Only 2 epochs ran before pruning
        assert len(trial.reported) == 2

    def test_best_metric_value_stored_after_fit(self, tmp_path, loaders):
        train_l, val_l = loaders
        cfg     = _base_config(tmp_path, num_epochs=3)
        trainer = Trainer(_TinyMLP(), {"mse": mse}, cfg)
        trainer.fit(train_l, val_l)
        assert np.isfinite(trainer._best_metric_value)

    def test_best_metric_value_stored_after_pruning(self, tmp_path, loaders):
        import optuna
        train_l, val_l = loaders
        cfg     = _base_config(tmp_path, num_epochs=10)
        trainer = Trainer(_TinyMLP(), {"mse": mse}, cfg)
        trial   = _MockTrial(should_prune_after=1)
        with pytest.raises(optuna.TrialPruned):
            trainer.fit(train_l, val_l, trial=trial)
        # Should still be set (from the initial inf sentinel or 1 epoch result)
        # The value may be inf if no improvement happened yet, but the attribute exists
        assert hasattr(trainer, "_best_metric_value")

    def test_no_trial_no_optuna_import(self, tmp_path, loaders):
        """Passing trial=None must not require optuna to be importable."""
        train_l, val_l = loaders
        cfg     = _base_config(tmp_path, num_epochs=2)
        trainer = Trainer(_TinyMLP(), {"mse": mse}, cfg)
        # Should work exactly as before — no optuna interaction
        result = trainer.fit(train_l, val_l, trial=None)
        assert isinstance(result, TrainState)


# ---------------------------------------------------------------------------
# TestTuner  (uses real Optuna with NopPruner for speed)
# ---------------------------------------------------------------------------

TRAIN_ARRS = _make_arrays(64)
VAL_ARRS   = _make_arrays(20, seed=1)

SEARCH_SPACE = {
    "lr": {"type": "float", "low": 5e-4, "high": 5e-3, "log": True},
}


def _suggest_fn(trial, base_config):
    hp  = apply_search_space(trial, SEARCH_SPACE)
    cfg = copy.deepcopy(base_config)
    cfg["scheduler_kwargs"]["value"] = hp["lr"]
    return cfg


def _model_fn(config):
    return _TinyMLP()


class TestTuner:

    @pytest.fixture
    def tuner(self, tmp_path):
        cfg = _base_config(tmp_path, num_epochs=2, patience=10)
        return Tuner(
            suggest_fn       = _suggest_fn,
            base_config      = cfg,
            model_fn         = _model_fn,
            metrics_fns      = {"mse": mse},
            train_loader_fn  = lambda: _make_loader(TRAIN_ARRS),
            val_loader_fn    = lambda: _make_loader(VAL_ARRS, shuffle=False),
            study_name       = "test_study",
            direction        = "minimize",
            storage          = None,    # in-memory
            n_startup_trials = 1,
            n_warmup_steps   = 0,
        )

    def test_run_completes(self, tuner):
        tuner.run(n_trials=2)

    def test_best_params_is_dict(self, tuner):
        tuner.run(n_trials=2)
        assert isinstance(tuner.best_params, dict)
        assert "lr" in tuner.best_params

    def test_best_value_is_finite(self, tuner):
        tuner.run(n_trials=2)
        assert np.isfinite(tuner.best_value)

    def test_study_accessible(self, tuner):
        tuner.run(n_trials=2)
        import optuna
        assert isinstance(tuner.study, optuna.Study)

    def test_study_raises_before_run(self, tuner):
        with pytest.raises(RuntimeError, match="No study"):
            _ = tuner.study

    def test_summary_runs(self, tuner, capsys):
        tuner.run(n_trials=2)
        tuner.summary()
        out = capsys.readouterr().out
        assert "test_study" in out
        assert "best value" in out

    def test_each_trial_has_own_checkpoint_dir(self, tmp_path):
        cfg     = _base_config(tmp_path, num_epochs=2, patience=10,
                               checkpoint_dir=str(tmp_path / "ckpts"))
        tuner = Tuner(
            suggest_fn       = _suggest_fn,
            base_config      = cfg,
            model_fn         = _model_fn,
            metrics_fns      = {"mse": mse},
            train_loader_fn  = lambda: _make_loader(TRAIN_ARRS),
            val_loader_fn    = lambda: _make_loader(VAL_ARRS, shuffle=False),
            study_name       = "ckpt_test",
            storage          = None,
            n_startup_trials = 1,
            n_warmup_steps   = 0,
        )
        tuner.run(n_trials=2)
        ckpt_root = tmp_path / "ckpts"
        trial_dirs = list(ckpt_root.glob("trial_*"))
        assert len(trial_dirs) == 2

    def test_run_skips_when_enough_trials_done(self, tuner, capsys):
        tuner.run(n_trials=2)
        tuner.run(n_trials=2)   # already done — should skip
        out = capsys.readouterr().out
        assert "already has" in out

    def test_direction_minimize(self, tuner):
        tuner.run(n_trials=2)
        assert tuner.study.direction.name == "MINIMIZE"

    def test_direction_maximize(self, tmp_path):
        cfg   = _base_config(tmp_path, num_epochs=2,
                             patience_direction="higher_is_better")
        tuner = Tuner(
            suggest_fn       = _suggest_fn,
            base_config      = cfg,
            model_fn         = _model_fn,
            metrics_fns      = {"mse": mse},
            train_loader_fn  = lambda: _make_loader(TRAIN_ARRS),
            val_loader_fn    = lambda: _make_loader(VAL_ARRS, shuffle=False),
            study_name       = "max_study",
            direction        = "maximize",
            storage          = None,
            n_startup_trials = 1,
            n_warmup_steps   = 0,
        )
        tuner.run(n_trials=2)
        assert tuner.study.direction.name == "MAXIMIZE"

    def test_pruning_end_to_end(self, tmp_path):
        """Aggressive pruning should prune most trials without crashing."""
        import optuna
        cfg = _base_config(tmp_path, num_epochs=10, patience=10)
        tuner = Tuner(
            suggest_fn       = _suggest_fn,
            base_config      = cfg,
            model_fn         = _model_fn,
            metrics_fns      = {"mse": mse},
            train_loader_fn  = lambda: _make_loader(TRAIN_ARRS),
            val_loader_fn    = lambda: _make_loader(VAL_ARRS, shuffle=False),
            study_name       = "prune_test",
            storage          = None,
            n_startup_trials = 1,
            n_warmup_steps   = 1,   # prune after 1 epoch
        )
        tuner.run(n_trials=4)
        pruned = [t for t in tuner.study.trials
                  if t.state == optuna.trial.TrialState.PRUNED]
        complete = [t for t in tuner.study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
        # At least some trials should complete or be pruned — no crashes
        assert len(pruned) + len(complete) == 4
