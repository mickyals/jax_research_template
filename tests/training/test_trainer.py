"""
Tests for training/trainer.py.

The Trainer now takes DataLoader objects (any iterable yielding
{'X': array, 'y': array} dicts) rather than raw in-memory arrays.
ArrayLoader from datasets.datamodule is used throughout for convenience,
but any iterable works — this is tested explicitly.

Coverage
--------
TestTrainState          custom fields; batch_stats populated for BatchNorm
TestTrainerInit         metrics_fns / loss_key / optimizer / clipping
TestInitState           params; opt_state; last_state; seed differences
TestTrainStep           params change; loss finite; rng advances;
                        dropout and BatchNorm variants
TestEvalStep            finite; deterministic; params unchanged;
                        BatchNorm eval exercised; all metrics_fns keys returned
TestTrainEpoch          state advances; metrics returned; tqdm inner bar;
                        any-iterable accepted
TestEvalModel           all metric keys returned; numpy/JAX loader;
                        weighted average; multi-target shape
TestIsBetter            lower/higher direction; NaN behaviour documented
TestCheckpointing       best save/load round-trip (plain MLP and BatchNorm);
                        latest save/load round-trip with metadata
TestFit                 loss decreases; early stopping; step budget (sub-epoch);
                        numpy loader; gradient clipping; bad patience_metric;
                        tqdm flag accepted; any-iterable loader accepted
TestResume              resume=True loads latest state + metadata and continues
TestTest                raises before fit; all metric keys present; any loader
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from datasets.datamodule import ArrayLoader
from training.losses import mae, mse
from training.trainer import Trainer, TrainState, _TQDM_AVAILABLE
from utils.jax_core.helpers import create_rng


# ---------------------------------------------------------------------------
# Model stubs
# ---------------------------------------------------------------------------

N_FEAT, N_TGT = 4, 2
BATCH_SZ = 16


class _TinyMLP(nn.Module):
    out: int = N_TGT

    @nn.compact
    def __call__(self, x, train: bool = False, rngs=None):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        return nn.Dense(self.out)(x)


class _TinyMLPDropout(nn.Module):
    out: int = N_TGT

    @nn.compact
    def __call__(self, x, train: bool = False, rngs=None):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.1, deterministic=not train)(x)
        return nn.Dense(self.out)(x)


class _TinyMLPBatchNorm(nn.Module):
    out: int = N_TGT

    @nn.compact
    def __call__(self, x, train: bool = False, rngs=None):
        x = nn.Dense(16)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return nn.Dense(self.out)(x)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_arrays(n: int, seed: int = 0, n_tgt: int = N_TGT) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "X": jnp.array(rng.normal(0, 1, (n, N_FEAT)).astype(np.float32)),
        "y": jnp.array(rng.normal(0, 1, (n, n_tgt)).astype(np.float32)),
    }


def _make_loader(arrays, batch_size=BATCH_SZ, shuffle=True, seed=0):
    return ArrayLoader(arrays, batch_size, shuffle=shuffle, seed=seed,
                       drop_last=shuffle)


def _base_config(tmp_path: Path, **overrides) -> dict:
    cfg = {
        "batch_size":         BATCH_SZ,
        "num_epochs":         5,
        "num_steps":          1_000,
        "patience":           5,
        "patience_metric":    "val/mse",
        "patience_direction": "lower_is_better",
        "checkpoint_dir":     str(tmp_path / "ckpts"),
        "log_every_n_steps":  2,
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


_METRICS       = {"mse": mse}
_MULTI_METRICS = {"mse": mse, "mae": mae}


@pytest.fixture
def train_arrs():
    return _make_arrays(64)


@pytest.fixture
def val_arrs():
    return _make_arrays(20, seed=1)


@pytest.fixture
def train_loader(train_arrs):
    return _make_loader(train_arrs, shuffle=True)


@pytest.fixture
def val_loader(val_arrs):
    return _make_loader(val_arrs, shuffle=False)


@pytest.fixture
def model():
    return _TinyMLP()


@pytest.fixture
def trainer(tmp_path, model):
    return Trainer(model, _METRICS, _base_config(tmp_path))


@pytest.fixture
def state(trainer, train_arrs):
    exmp = {k: v[:4] for k, v in train_arrs.items()}
    return trainer._init_state({k: jnp.asarray(v) for k, v in exmp.items()})


# ---------------------------------------------------------------------------
# TestTrainState
# ---------------------------------------------------------------------------

class TestTrainState:

    def test_has_batch_stats_field(self, state):
        assert hasattr(state, "batch_stats")

    def test_has_rng_field(self, state):
        assert hasattr(state, "rng")

    def test_batch_stats_none_for_plain_mlp(self, state):
        assert state.batch_stats is None

    def test_rng_is_jax_array(self, state):
        assert isinstance(state.rng, jax.Array)

    def test_step_starts_at_zero(self, state):
        assert int(state.step) == 0

    def test_batch_stats_populated_for_batchnorm_model(self, tmp_path):
        t = Trainer(_TinyMLPBatchNorm(), _METRICS, _base_config(tmp_path))
        s = t._init_state(_make_arrays(4))
        assert s.batch_stats is not None

    def test_params_present(self, state):
        assert jax.tree_util.tree_leaves(state.params)


# ---------------------------------------------------------------------------
# TestTrainerInit
# ---------------------------------------------------------------------------

class TestTrainerInit:

    def test_metrics_fns_stored(self, trainer):
        assert trainer.metrics_fns is _METRICS

    def test_loss_key_defaults_to_first_key(self, trainer):
        assert trainer._loss_key == "mse"

    def test_loss_key_from_config(self, tmp_path):
        cfg = _base_config(tmp_path, loss_key="mae")
        t   = Trainer(_TinyMLP(), _MULTI_METRICS, cfg)
        assert t._loss_key == "mae"

    def test_bad_loss_key_raises(self, tmp_path):
        with pytest.raises(KeyError, match="nonexistent"):
            Trainer(_TinyMLP(), _METRICS, _base_config(tmp_path, loss_key="nonexistent"))

    def test_patience_metric_defaults_to_val_loss_key(self, tmp_path):
        assert Trainer(_TinyMLP(), _METRICS, _base_config(tmp_path))._patience_metric == "val/mse"

    def test_batch_size_parsed(self, trainer):
        assert trainer._batch_size == BATCH_SZ

    def test_lower_is_better_default(self, trainer):
        assert trainer._lower_is_better is True

    def test_higher_is_better_parsed(self, tmp_path):
        cfg = _base_config(tmp_path, patience_direction="higher_is_better")
        assert Trainer(_TinyMLP(), _METRICS, cfg)._lower_is_better is False

    def test_optimizer_built(self, trainer):
        import optax
        assert isinstance(trainer._optimizer, optax.GradientTransformation)

    def test_grad_clipping_chain_built(self, tmp_path):
        import optax
        cfg = _base_config(tmp_path, max_grad_norm=1.0)
        assert isinstance(Trainer(_TinyMLP(), _METRICS, cfg)._optimizer,
                          optax.GradientTransformation)

    def test_step_functions_none_before_init(self, tmp_path):
        t = Trainer(_TinyMLP(), _METRICS, _base_config(tmp_path))
        assert t._train_step is None
        assert t._eval_step  is None


# ---------------------------------------------------------------------------
# TestInitState
# ---------------------------------------------------------------------------

class TestInitState:

    def test_step_functions_built_after_init(self, state, trainer):
        assert trainer._train_step is not None
        assert trainer._eval_step  is not None

    def test_params_have_leaves(self, state):
        assert jax.tree_util.tree_leaves(state.params)

    def test_opt_state_created(self, state):
        assert state.opt_state is not None

    def test_last_state_stored(self, trainer, train_arrs):
        trainer._init_state({k: v[:4] for k, v in train_arrs.items()})
        assert trainer._last_state is not None

    def test_different_seeds_give_different_params(self, tmp_path, train_arrs):
        exmp = {k: v[:4] for k, v in train_arrs.items()}
        t1 = Trainer(_TinyMLP(), _METRICS, _base_config(tmp_path, seed=0))
        t2 = Trainer(_TinyMLP(), _METRICS, _base_config(tmp_path, seed=99))
        l1 = jax.tree_util.tree_leaves(t1._init_state(exmp).params)
        l2 = jax.tree_util.tree_leaves(t2._init_state(exmp).params)
        assert any(not jnp.allclose(a, b) for a, b in zip(l1, l2))


# ---------------------------------------------------------------------------
# TestTrainStep
# ---------------------------------------------------------------------------

class TestTrainStep:

    def test_params_change(self, trainer, state, train_arrs):
        batch     = {k: v[:BATCH_SZ] for k, v in train_arrs.items()}
        new_state, _ = trainer._train_step(state, batch)
        old = jax.tree_util.tree_leaves(state.params)
        new = jax.tree_util.tree_leaves(new_state.params)
        assert any(not jnp.allclose(o, n) for o, n in zip(old, new))

    def test_loss_finite(self, trainer, state, train_arrs):
        batch     = {k: v[:BATCH_SZ] for k, v in train_arrs.items()}
        _, metrics = trainer._train_step(state, batch)
        assert jnp.isfinite(metrics[trainer._loss_key])

    def test_step_counter_increments(self, trainer, state, train_arrs):
        batch     = {k: v[:BATCH_SZ] for k, v in train_arrs.items()}
        new_state, _ = trainer._train_step(state, batch)
        assert int(new_state.step) == int(state.step) + 1

    def test_rng_advances(self, trainer, state, train_arrs):
        batch     = {k: v[:BATCH_SZ] for k, v in train_arrs.items()}
        new_state, _ = trainer._train_step(state, batch)
        assert not jnp.array_equal(state.rng, new_state.rng)

    def test_dropout_model(self, tmp_path, train_arrs):
        t     = Trainer(_TinyMLPDropout(), _METRICS, _base_config(tmp_path))
        state = t._init_state({k: v[:4] for k, v in train_arrs.items()})
        batch = {k: v[:BATCH_SZ] for k, v in train_arrs.items()}
        new_state, metrics = t._train_step(state, batch)
        assert jnp.isfinite(metrics[t._loss_key])
        old = jax.tree_util.tree_leaves(state.params)
        new = jax.tree_util.tree_leaves(new_state.params)
        assert any(not jnp.allclose(o, n) for o, n in zip(old, new))

    def test_batchnorm_model_updates_batch_stats(self, tmp_path, train_arrs):
        t     = Trainer(_TinyMLPBatchNorm(), _METRICS, _base_config(tmp_path))
        state = t._init_state({k: v[:4] for k, v in train_arrs.items()})
        batch = {k: v[:BATCH_SZ] for k, v in train_arrs.items()}
        new_state, metrics = t._train_step(state, batch)
        assert jnp.isfinite(metrics[t._loss_key])
        assert new_state.batch_stats is not None


# ---------------------------------------------------------------------------
# TestEvalStep
# ---------------------------------------------------------------------------

class TestEvalStep:

    def test_all_metric_keys_returned(self, tmp_path, train_arrs, val_arrs):
        t     = Trainer(_TinyMLP(), _MULTI_METRICS, _base_config(tmp_path))
        state = t._init_state({k: v[:4] for k, v in train_arrs.items()})
        batch = {k: jnp.array(v) for k, v in val_arrs.items()}
        assert set(t._eval_step(state, batch).keys()) == set(_MULTI_METRICS.keys())

    def test_loss_finite(self, trainer, state, val_arrs):
        batch   = {k: jnp.array(v) for k, v in val_arrs.items()}
        metrics = trainer._eval_step(state, batch)
        assert all(jnp.isfinite(v) for v in metrics.values())

    def test_deterministic(self, trainer, state, val_arrs):
        batch = {k: jnp.array(v) for k, v in val_arrs.items()}
        m1    = trainer._eval_step(state, batch)
        m2    = trainer._eval_step(state, batch)
        for k in m1:
            assert float(m1[k]) == float(m2[k])

    def test_params_unchanged_after_eval(self, trainer, state, val_arrs):
        batch  = {k: jnp.array(v) for k, v in val_arrs.items()}
        before = jax.tree_util.tree_leaves(state.params)
        trainer._eval_step(state, batch)
        after  = jax.tree_util.tree_leaves(state.params)
        for b, a in zip(before, after):
            assert jnp.allclose(b, a)

    def test_batchnorm_eval_is_finite_and_deterministic(self, tmp_path, train_arrs, val_arrs):
        t     = Trainer(_TinyMLPBatchNorm(), _METRICS, _base_config(tmp_path))
        state = t._init_state({k: v[:4] for k, v in train_arrs.items()})
        batch = {k: v[:BATCH_SZ] for k, v in train_arrs.items()}
        for _ in range(3):
            state, _ = t._train_step(state, batch)
        jax_batch = {k: jnp.array(v) for k, v in val_arrs.items()}
        m1 = t._eval_step(state, jax_batch)
        m2 = t._eval_step(state, jax_batch)
        for k in m1:
            assert jnp.isfinite(m1[k])
            assert float(m1[k]) == float(m2[k])


# ---------------------------------------------------------------------------
# TestTrainEpoch
# ---------------------------------------------------------------------------

class TestTrainEpoch:

    def test_state_step_advances(self, trainer, state, train_arrs):
        loader    = _make_loader(train_arrs)
        new_state, _ = trainer._train_epoch(state, loader, epoch=0)
        assert int(new_state.step) > int(state.step)

    def test_metrics_returned(self, trainer, state, train_arrs):
        loader = _make_loader(train_arrs)
        _, metrics = trainer._train_epoch(state, loader, epoch=0)
        assert f"train/{trainer._loss_key}" in metrics
        assert np.isfinite(list(metrics.values())[0])

    def test_params_change_over_epoch(self, trainer, state, train_arrs):
        loader    = _make_loader(train_arrs)
        new_state, _ = trainer._train_epoch(state, loader, epoch=0)
        old = jax.tree_util.tree_leaves(state.params)
        new = jax.tree_util.tree_leaves(new_state.params)
        assert any(not jnp.allclose(o, n) for o, n in zip(old, new))

    def test_accepts_any_iterable(self, trainer, state, train_arrs):
        # A plain list of dicts is a valid "loader"
        batch = {k: v[:BATCH_SZ] for k, v in train_arrs.items()}
        any_loader = [batch, batch, batch]
        new_state, _ = trainer._train_epoch(state, any_loader, epoch=0)
        assert int(new_state.step) == 3

    def test_tqdm_inner_bar_accepted(self, tmp_path, train_arrs, state):
        if not _TQDM_AVAILABLE:
            pytest.skip("tqdm not installed")
        cfg     = _base_config(tmp_path, use_tqdm=True)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        trainer._init_state({k: v[:4] for k, v in train_arrs.items()})
        loader  = _make_loader(train_arrs)
        new_state, _ = trainer._train_epoch(state, loader, epoch=0)
        assert new_state is not None


# ---------------------------------------------------------------------------
# TestEvalModel
# ---------------------------------------------------------------------------

class TestEvalModel:

    def test_all_metric_keys_with_prefix(self, tmp_path, train_arrs, val_arrs):
        t     = Trainer(_TinyMLP(), _MULTI_METRICS, _base_config(tmp_path))
        state = t._init_state({k: v[:4] for k, v in train_arrs.items()})
        loader  = _make_loader(val_arrs, shuffle=False)
        metrics = t._eval_model(state, loader, prefix="val")
        assert "val/mse" in metrics
        assert "val/mae" in metrics

    def test_accepts_numpy_loader(self, trainer, state, val_arrs):
        np_arrs = {k: np.array(v) for k, v in val_arrs.items()}
        loader  = _make_loader(np_arrs, shuffle=False)
        metrics = trainer._eval_model(state, loader)
        assert all(np.isfinite(v) for v in metrics.values())

    def test_accepts_jax_loader(self, trainer, state, val_arrs):
        loader  = _make_loader(val_arrs, shuffle=False)
        metrics = trainer._eval_model(state, loader)
        assert all(np.isfinite(v) for v in metrics.values())

    def test_custom_prefix(self, trainer, state, val_arrs):
        loader  = _make_loader(val_arrs, shuffle=False)
        metrics = trainer._eval_model(state, loader, prefix="test")
        assert all(k.startswith("test/") for k in metrics)

    def test_deterministic(self, trainer, state, val_arrs):
        loader  = _make_loader(val_arrs, shuffle=False)
        m1 = trainer._eval_model(state, loader)
        m2 = trainer._eval_model(state, loader)
        for k in m1:
            assert m1[k] == m2[k]

    def test_weighted_average_over_batches(self, trainer, state, val_arrs):
        # val has 20 samples; batch_size=16 gives batches of 16 + 4
        # weighted average should equal eval on all 20 at once
        loader  = _make_loader(val_arrs, batch_size=16, shuffle=False)
        batched = trainer._eval_model(state, loader)
        single  = trainer._eval_model(state, [val_arrs])
        assert abs(batched["val/mse"] - single["val/mse"]) < 1e-4

    def test_accepts_any_iterable(self, trainer, state, val_arrs):
        single_batch = [val_arrs]   # plain list works
        metrics = trainer._eval_model(state, single_batch)
        assert all(np.isfinite(v) for v in metrics.values())

    def test_multi_target_shape(self, tmp_path):
        n_tgt   = 4
        t       = Trainer(_TinyMLP(out=n_tgt), _METRICS, _base_config(tmp_path))
        arrays  = _make_arrays(32, n_tgt=n_tgt)
        state   = t._init_state({k: v[:4] for k, v in arrays.items()})
        loader  = _make_loader(arrays, shuffle=False)
        metrics = t._eval_model(state, loader)
        assert all(np.isfinite(v) for v in metrics.values())


# ---------------------------------------------------------------------------
# TestIsBetter
# ---------------------------------------------------------------------------

class TestIsBetter:

    def test_lower_is_better_improvement(self, trainer):
        assert trainer.is_better(0.1, 0.5) is True

    def test_lower_is_better_regression(self, trainer):
        assert trainer.is_better(0.9, 0.5) is False

    def test_lower_is_better_equal(self, trainer):
        assert trainer.is_better(0.5, 0.5) is False

    def test_higher_is_better_improvement(self, tmp_path):
        cfg = _base_config(tmp_path, patience_direction="higher_is_better")
        assert Trainer(_TinyMLP(), _METRICS, cfg).is_better(0.9, 0.5) is True

    def test_higher_is_better_regression(self, tmp_path):
        cfg = _base_config(tmp_path, patience_direction="higher_is_better")
        assert Trainer(_TinyMLP(), _METRICS, cfg).is_better(0.1, 0.5) is False

    def test_nan_current_returns_false_lower(self, trainer):
        assert trainer.is_better(float("nan"), 0.5) is False

    def test_nan_current_returns_false_higher(self, tmp_path):
        cfg = _base_config(tmp_path, patience_direction="higher_is_better")
        assert Trainer(_TinyMLP(), _METRICS, cfg).is_better(float("nan"), 0.5) is False

    def test_nan_best_returns_false(self, trainer):
        assert trainer.is_better(0.3, float("nan")) is False


# ---------------------------------------------------------------------------
# TestCheckpointing
# ---------------------------------------------------------------------------

class TestCheckpointing:

    def test_save_best_creates_dir(self, trainer, state):
        trainer.save_checkpoint(state)
        assert (trainer._checkpoint_dir / "best").exists()

    def test_load_best_restores_params(self, trainer, state):
        trainer.save_checkpoint(state)
        restored = trainer.load_checkpoint(state)
        for o, r in zip(
            jax.tree_util.tree_leaves(state.params),
            jax.tree_util.tree_leaves(restored.params),
        ):
            assert jnp.allclose(o, r)

    def test_load_raises_when_no_checkpoint(self, trainer, state):
        with pytest.raises(FileNotFoundError, match="No checkpoint"):
            trainer.load_checkpoint(state)

    def test_save_latest_creates_dir_and_metadata(self, trainer, state):
        trainer._save_latest(state, epoch=2, global_step=100,
                             best_metric=0.5, patience_count=1)
        assert (trainer._checkpoint_dir / "latest").exists()
        assert (trainer._checkpoint_dir / "latest_metadata.json").exists()

    def test_load_latest_restores_state_and_metadata(self, trainer, state):
        trainer._save_latest(state, epoch=2, global_step=100,
                             best_metric=0.5, patience_count=1)
        restored, meta = trainer._load_latest(state)
        assert meta["epoch"]          == 2
        assert meta["global_step"]    == 100
        assert meta["patience_count"] == 1
        assert meta["best_metric"]    == pytest.approx(0.5)
        for o, r in zip(
            jax.tree_util.tree_leaves(state.params),
            jax.tree_util.tree_leaves(restored.params),
        ):
            assert jnp.allclose(o, r)

    def test_load_latest_raises_when_missing(self, trainer, state):
        with pytest.raises(FileNotFoundError):
            trainer._load_latest(state)

    def test_latest_metadata_survives_inf(self, trainer, state):
        trainer._save_latest(state, epoch=0, global_step=0,
                             best_metric=float("inf"), patience_count=0)
        _, meta = trainer._load_latest(state)
        assert meta["best_metric"] == float("inf")

    def test_batchnorm_best_round_trip(self, tmp_path, train_arrs):
        t     = Trainer(_TinyMLPBatchNorm(), _METRICS, _base_config(tmp_path))
        state = t._init_state({k: v[:4] for k, v in train_arrs.items()})
        batch = {k: v[:BATCH_SZ] for k, v in train_arrs.items()}
        for _ in range(3):
            state, _ = t._train_step(state, batch)
        t.save_checkpoint(state)
        restored = t.load_checkpoint(state)
        assert restored.batch_stats is not None
        for o, r in zip(
            jax.tree_util.tree_leaves(state.batch_stats),
            jax.tree_util.tree_leaves(restored.batch_stats),
        ):
            assert jnp.allclose(o, r)


# ---------------------------------------------------------------------------
# TestManifest
# ---------------------------------------------------------------------------

class TestManifest:

    def test_writes_json_next_to_checkpoints(self, trainer):
        manifest = {"train": {"seasons": [2019], "n_rows": 5}}
        trainer.write_manifest(manifest)
        path = trainer._checkpoint_dir / "manifest.json"
        assert path.exists()
        with open(path) as fh:
            assert json.load(fh) == manifest

    def test_custom_filename(self, trainer):
        trainer.write_manifest({"a": 1}, filename="split_manifest.json")
        assert (trainer._checkpoint_dir / "split_manifest.json").exists()

    def test_pushes_to_logger_hparams(self, trainer):
        trainer.write_manifest({"train": {"n_rows": 5}})
        with open(trainer.logger.log_dir / "hparams.json") as fh:
            hparams = json.load(fh)
        assert hparams["manifest"] == {"train": {"n_rows": 5}}

    def test_creates_checkpoint_dir_if_missing(self, tmp_path, model):
        cfg = _base_config(tmp_path, checkpoint_dir=str(tmp_path / "fresh_ckpts"))
        t = Trainer(model, _METRICS, cfg)
        t.write_manifest({"x": 1})
        assert (t._checkpoint_dir / "manifest.json").exists()


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_returns_train_state(self, tmp_path, train_loader, val_loader):
        cfg    = _base_config(tmp_path, num_epochs=3, patience=10)
        result = Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)
        assert isinstance(result, TrainState)

    def test_all_val_metrics_logged(self, tmp_path, train_arrs, val_arrs, capsys):
        cfg     = _base_config(tmp_path, num_epochs=2, patience=10)
        trainer = Trainer(_TinyMLP(), _MULTI_METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        trainer.fit(tl, vl)
        assert "val/mse" in capsys.readouterr().out

    def test_early_stopping_triggers(self, tmp_path, train_loader, val_loader, capsys):
        cfg = _base_config(tmp_path, num_epochs=50, patience=1)
        Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)
        assert "Early stopping" in capsys.readouterr().out

    def test_num_steps_budget(self, tmp_path, train_loader, val_loader, capsys):
        cfg = _base_config(tmp_path, num_epochs=1000, num_steps=5, patience=100)
        Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)
        assert "budget" in capsys.readouterr().out

    def test_num_steps_sub_epoch(self, tmp_path, train_arrs, val_arrs, capsys):
        # 64 / 16 = 4 steps/epoch; budget=2 → stops mid-epoch-0
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        cfg = _base_config(tmp_path, num_epochs=100, num_steps=2, patience=100)
        Trainer(_TinyMLP(), _METRICS, cfg).fit(tl, vl)
        out   = capsys.readouterr().out
        assert "budget" in out
        epoch_lines = [l for l in out.splitlines()
                       if l.startswith("epoch") and "val/mse" in l]
        assert len(epoch_lines) == 1

    def test_numpy_loader_accepted(self, tmp_path, val_arrs):
        np_arrs = {k: np.array(v) for k, v in _make_arrays(64).items()}
        tl = _make_loader(np_arrs); vl = _make_loader(val_arrs, shuffle=False)
        cfg    = _base_config(tmp_path, num_epochs=2, patience=10)
        result = Trainer(_TinyMLP(), _METRICS, cfg).fit(tl, vl)
        assert isinstance(result, TrainState)

    def test_any_iterable_accepted(self, tmp_path, train_arrs, val_arrs):
        # Plain list of batch dicts works as a loader
        batch   = {k: v[:BATCH_SZ] for k, v in train_arrs.items()}
        any_tl  = [batch] * 3
        any_vl  = [val_arrs]
        cfg     = _base_config(tmp_path, num_epochs=2, patience=10)
        result  = Trainer(_TinyMLP(), _METRICS, cfg).fit(any_tl, any_vl)
        assert isinstance(result, TrainState)

    def test_grad_clipping_runs(self, tmp_path, train_loader, val_loader):
        cfg    = _base_config(tmp_path, num_epochs=2, patience=10, max_grad_norm=1.0)
        result = Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)
        assert isinstance(result, TrainState)

    def test_bad_patience_metric_raises(self, tmp_path, train_loader, val_loader):
        cfg = _base_config(tmp_path, patience_metric="val/nonexistent", num_epochs=2)
        with pytest.raises(KeyError, match="nonexistent"):
            Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)

    def test_latest_checkpoint_saved_each_epoch(self, tmp_path, train_loader, val_loader):
        cfg     = _base_config(tmp_path, num_epochs=3, patience=10)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        trainer.fit(train_loader, val_loader)
        assert (trainer._checkpoint_dir / "latest").exists()
        assert (trainer._checkpoint_dir / "latest_metadata.json").exists()

    def test_use_tqdm_flag_accepted(self, tmp_path, train_arrs, val_arrs):
        if not _TQDM_AVAILABLE:
            pytest.skip("tqdm not installed")
        tl  = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        cfg = _base_config(tmp_path, num_epochs=2, patience=10, use_tqdm=True)
        assert isinstance(Trainer(_TinyMLP(), _METRICS, cfg).fit(tl, vl), TrainState)

    def test_loss_decreases_over_epochs(self, tmp_path):
        import contextlib, io, re
        big_train = _make_arrays(512)
        big_val   = _make_arrays(64, seed=1)
        tl = _make_loader(big_train); vl = _make_loader(big_val, shuffle=False)
        cfg = _base_config(tmp_path, num_epochs=20, patience=20,
                           scheduler_kwargs={"value": 1e-2})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Trainer(_TinyMLP(), _METRICS, cfg).fit(tl, vl)
        lines = [l for l in buf.getvalue().splitlines() if "val/mse" in l]

        def _ex(line):
            m = re.search(r"val/mse[\s=]+([0-9.]+)", line)
            return float(m.group(1)) if m else None

        vals = [v for v in (_ex(l) for l in lines) if v is not None]
        assert vals[-1] < vals[0]


# ---------------------------------------------------------------------------
# TestResume
# ---------------------------------------------------------------------------

class TestResume:

    def _run(self, tmp_path, n, train_arrs, val_arrs):
        cfg     = _base_config(tmp_path, num_epochs=n, patience=n + 5)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        trainer.fit(tl, vl)
        return trainer

    def test_resume_without_error(self, tmp_path, train_arrs, val_arrs):
        self._run(tmp_path, 3, train_arrs, val_arrs)
        cfg     = _base_config(tmp_path, num_epochs=6, patience=10)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        result  = trainer.fit(tl, vl, resume=True)
        assert isinstance(result, TrainState)

    def test_resume_starts_past_epoch_zero(self, tmp_path, train_arrs, val_arrs, capsys):
        self._run(tmp_path, 3, train_arrs, val_arrs)
        cfg = _base_config(tmp_path, num_epochs=6, patience=10)
        tl  = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        Trainer(_TinyMLP(), _METRICS, cfg).fit(tl, vl, resume=True)
        assert "Resuming from epoch" in capsys.readouterr().out

    def test_resume_restores_global_step(self, tmp_path, train_arrs, val_arrs):
        t1 = self._run(tmp_path, 3, train_arrs, val_arrs)
        with open(t1._checkpoint_dir / "latest_metadata.json") as fh:
            first_steps = int(json.load(fh)["global_step"])

        cfg = _base_config(tmp_path, num_epochs=6, patience=10)
        t2  = Trainer(_TinyMLP(), _METRICS, cfg)
        tl  = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        t2.fit(tl, vl, resume=True)
        assert t2._global_step > first_steps

    def test_resume_raises_when_no_latest(self, tmp_path, train_arrs, val_arrs):
        cfg     = _base_config(tmp_path, num_epochs=5, patience=10)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        trainer._init_state({k: v[:4] for k, v in train_arrs.items()})
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        with pytest.raises(FileNotFoundError, match="No latest checkpoint"):
            trainer.fit(tl, vl, resume=True)

    def test_best_checkpoint_preserved_after_resume(self, tmp_path, train_arrs, val_arrs):
        t1 = self._run(tmp_path, 3, train_arrs, val_arrs)
        assert (t1._checkpoint_dir / "best").exists()
        cfg = _base_config(tmp_path, num_epochs=6, patience=10)
        t2  = Trainer(_TinyMLP(), _METRICS, cfg)
        tl  = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        t2.fit(tl, vl, resume=True)
        assert (t2._checkpoint_dir / "best").exists()


# ---------------------------------------------------------------------------
# TestTest
# ---------------------------------------------------------------------------

class TestTest:

    def test_raises_before_fit(self, tmp_path):
        trainer = Trainer(_TinyMLP(), _METRICS, _base_config(tmp_path))
        with pytest.raises(RuntimeError, match="not initialised"):
            trainer.test(_make_loader(_make_arrays(20), shuffle=False))

    def test_returns_all_metric_keys(self, tmp_path, train_arrs, val_arrs):
        cfg     = _base_config(tmp_path, num_epochs=2, patience=10)
        trainer = Trainer(_TinyMLP(), _MULTI_METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        trainer.fit(tl, vl)
        test_l  = _make_loader(_make_arrays(20, seed=2), shuffle=False)
        metrics = trainer.test(test_l)
        assert "test/mse" in metrics
        assert "test/mae" in metrics

    def test_all_values_finite(self, tmp_path, train_arrs, val_arrs):
        cfg     = _base_config(tmp_path, num_epochs=2, patience=10)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        trainer.fit(tl, vl)
        test_l  = _make_loader(_make_arrays(20, seed=2), shuffle=False)
        metrics = trainer.test(test_l)
        assert all(np.isfinite(v) for v in metrics.values())

    def test_accepts_any_iterable(self, tmp_path, train_arrs, val_arrs):
        cfg     = _base_config(tmp_path, num_epochs=2, patience=10)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        trainer.fit(tl, vl)
        # plain list works
        test_arrs = _make_arrays(20, seed=3)
        metrics   = trainer.test([test_arrs])
        assert all(np.isfinite(v) for v in metrics.values())


# ---------------------------------------------------------------------------
# TestStartupSummary
# ---------------------------------------------------------------------------

class TestStartupSummary:
    """_print_startup_summary and checkpoint path resolution."""

    def test_summary_printed_during_fit(self, tmp_path, train_loader, val_loader, capsys):
        cfg = _base_config(tmp_path, num_epochs=1, patience=10)
        Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)
        out = capsys.readouterr().out
        assert "Trainer" in out

    def test_summary_contains_optimizer_name(self, tmp_path, train_loader, val_loader, capsys):
        cfg = _base_config(tmp_path, num_epochs=1, patience=10)
        Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)
        out = capsys.readouterr().out
        assert "adam" in out          # optimizer name from _base_config

    def test_summary_contains_scheduler_name(self, tmp_path, train_loader, val_loader, capsys):
        cfg = _base_config(tmp_path, num_epochs=1, patience=10)
        Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)
        out = capsys.readouterr().out
        assert "constant" in out      # scheduler name from _base_config

    def test_summary_contains_backend(self, tmp_path, train_loader, val_loader, capsys):
        cfg = _base_config(tmp_path, num_epochs=1, patience=10)
        Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)
        out = capsys.readouterr().out
        assert "backend" in out

    def test_summary_contains_checkpoint_path(self, tmp_path, train_loader, val_loader, capsys):
        cfg = _base_config(tmp_path, num_epochs=1, patience=10)
        Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)
        out = capsys.readouterr().out
        assert "checkpoints" in out.lower()

    def test_summary_contains_steps_estimate(self, tmp_path, train_loader, val_loader, capsys):
        cfg = _base_config(tmp_path, num_epochs=1, patience=10)
        Trainer(_TinyMLP(), _METRICS, cfg).fit(train_loader, val_loader)
        out = capsys.readouterr().out
        assert "steps" in out.lower()

    def test_summary_direct_call_with_known_steps(self, tmp_path, capsys):
        t = Trainer(_TinyMLP(), _METRICS, _base_config(tmp_path))
        t._print_startup_summary(steps_per_epoch=42)
        out = capsys.readouterr().out
        assert "42" in out

    def test_summary_direct_call_unknown_steps(self, tmp_path, capsys):
        t = Trainer(_TinyMLP(), _METRICS, _base_config(tmp_path))
        t._print_startup_summary(steps_per_epoch=None)
        out = capsys.readouterr().out
        assert "unknown" in out

    def test_summary_lower_arrow_shown(self, tmp_path, capsys):
        cfg = _base_config(tmp_path, patience_direction="lower_is_better")
        Trainer(_TinyMLP(), _METRICS, cfg)._print_startup_summary(10)
        out = capsys.readouterr().out
        assert "↓" in out

    def test_summary_higher_arrow_shown(self, tmp_path, capsys):
        cfg = _base_config(tmp_path, patience_direction="higher_is_better",
                           patience_metric="val/mse")
        Trainer(_TinyMLP(), _METRICS, cfg)._print_startup_summary(10)
        out = capsys.readouterr().out
        assert "↑" in out

    def test_summary_grad_clip_shown_when_set(self, tmp_path, capsys):
        cfg = _base_config(tmp_path, max_grad_norm=1.5)
        Trainer(_TinyMLP(), _METRICS, cfg)._print_startup_summary(10)
        out = capsys.readouterr().out
        assert "1.5" in out

    def test_summary_grad_clip_none_when_unset(self, tmp_path, capsys):
        cfg = _base_config(tmp_path)   # no max_grad_norm
        Trainer(_TinyMLP(), _METRICS, cfg)._print_startup_summary(10)
        out = capsys.readouterr().out
        assert "none" in out.lower()

    # --- Orbax absolute path resolution ---

    def test_checkpoint_dir_absolute_from_relative_checkpoint_dir(
            self, tmp_path, monkeypatch):
        # A relative checkpoint_dir must be resolved to an absolute path
        # so Orbax never receives a relative path. chdir into tmp_path:
        # resolution is CWD-relative and the logger mkdirs under it.
        monkeypatch.chdir(tmp_path)
        cfg = _base_config(tmp_path, checkpoint_dir="relative/ckpts")
        t   = Trainer(_TinyMLP(), _METRICS, cfg)
        assert t._checkpoint_dir.is_absolute()

    def test_checkpoint_dir_absolute_from_relative_run_dir(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = _base_config(tmp_path)
        cfg.pop("checkpoint_dir", None)
        cfg["run_dir"] = "relative/run_01"
        t = Trainer(_TinyMLP(), _METRICS, cfg)
        assert t._checkpoint_dir.is_absolute()

    def test_checkpoint_dir_unchanged_when_already_absolute(self, tmp_path):
        # An already-absolute tmp_path should survive resolve() unchanged.
        cfg = _base_config(tmp_path)   # uses absolute tmp_path / "ckpts"
        t   = Trainer(_TinyMLP(), _METRICS, cfg)
        assert t._checkpoint_dir.is_absolute()
        assert str(tmp_path) in str(t._checkpoint_dir)


# ---------------------------------------------------------------------------
# TestStepCallbacks
# ---------------------------------------------------------------------------

class TestStepCallbacks:
    """step_callbacks fired inside _train_epoch at configurable frequency."""

    def test_step_callback_called_at_correct_frequency(self, tmp_path, train_arrs, val_arrs):
        """Callback fires exactly every every_n_steps steps."""
        calls = []

        def cb(state, epoch, global_step):
            calls.append(global_step)

        cfg     = _base_config(tmp_path, num_epochs=1, patience=10)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        trainer.fit(tl, vl, step_callbacks=[(cb, 2)])

        # Every recorded step must be divisible by 2
        assert all(s % 2 == 0 for s in calls)
        assert len(calls) > 0

    def test_step_callback_receives_correct_global_step(self, tmp_path, train_arrs, val_arrs):
        """global_step passed to the callback matches the trainer's counter."""
        recorded = []

        def cb(state, epoch, global_step):
            recorded.append(global_step)

        cfg     = _base_config(tmp_path, num_epochs=1, patience=10)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        trainer.fit(tl, vl, step_callbacks=[(cb, 1)])

        # Steps must be strictly increasing starting from 1
        assert recorded == list(range(1, len(recorded) + 1))

    def test_step_callback_receives_correct_epoch(self, tmp_path, train_arrs, val_arrs):
        """Epoch index passed to step callback matches the outer loop epoch."""
        epochs_seen = set()

        def cb(state, epoch, global_step):
            epochs_seen.add(epoch)

        cfg     = _base_config(tmp_path, num_epochs=3, patience=10)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        trainer.fit(tl, vl, step_callbacks=[(cb, 1)])

        assert epochs_seen == {0, 1, 2}

    def test_multiple_step_callbacks_with_different_frequencies(
        self, tmp_path, train_arrs, val_arrs
    ):
        calls_2 = []
        calls_3 = []

        def cb2(state, epoch, gs): calls_2.append(gs)
        def cb3(state, epoch, gs): calls_3.append(gs)

        cfg     = _base_config(tmp_path, num_epochs=1, patience=10)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        trainer.fit(tl, vl, step_callbacks=[(cb2, 2), (cb3, 3)])

        assert all(s % 2 == 0 for s in calls_2)
        assert all(s % 3 == 0 for s in calls_3)

    def test_step_callback_none_does_not_crash(self, tmp_path, train_loader, val_loader):
        """step_callbacks=None is the default and must not raise."""
        cfg    = _base_config(tmp_path, num_epochs=1, patience=10)
        result = Trainer(_TinyMLP(), _METRICS, cfg).fit(
            train_loader, val_loader, step_callbacks=None
        )
        assert isinstance(result, TrainState)

    def test_step_and_epoch_callbacks_coexist(self, tmp_path, train_arrs, val_arrs):
        """Both step and epoch callbacks fire independently in the same run."""
        step_calls  = []
        epoch_calls = []

        def scb(state, epoch, gs):  step_calls.append(gs)
        def ecb(state, epoch, gs):  epoch_calls.append(epoch)

        cfg     = _base_config(tmp_path, num_epochs=2, patience=10)
        trainer = Trainer(_TinyMLP(), _METRICS, cfg)
        tl = _make_loader(train_arrs); vl = _make_loader(val_arrs, shuffle=False)
        trainer.fit(tl, vl, step_callbacks=[(scb, 1)], epoch_callbacks=[ecb])

        assert len(step_calls)  > 0
        assert epoch_calls == [0, 1]
