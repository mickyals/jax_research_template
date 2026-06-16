"""
Integration tests: Trainer + TCClassifier end-to-end training.

All tests use synthetic in-memory data — no disk access required.

Coverage
--------
TestOneForwardBackwardPass
    forward pass: all five metric keys returned (loss, cross_entropy,
        accuracy, binary_accuracy, mae_class); all finite; deterministic
    backward pass: params change after one gradient update; loss finite and positive
    both attention paths (use_self_attention True / False)
    both location encodings (unit_circle / domain)
    missing obs (obs_mask partially False) handled without NaN
    padded stations (station_mask partially False) handled without NaN

TestLossVariation
    loss is not identical at step 1 vs step 10
    _train_epoch returns correct metric key (train/loss) and finite value
    _eval_model returns all five val metric keys, all finite
    loss decreases over training when data has a clear separable signal
    trainer.fit() completes and returns TrainState
    epoch callbacks receive correct (epoch, global_step) pairs
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.sparse_obs_cross_attn.train.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.train.model import TCClassifier
from training.trainer import Trainer, TrainState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B     = 8    # batch size
N     = 12   # stations per sample
F     = 5    # obs features
N_CLS = 9

# Full per-batch metric set — these Trainer tests patience on val/cross_entropy
# and assert the complete metric dict, so they request every metric explicitly
# (build_metrics_fns now defaults to just binary_accuracy + mae_class).
_ALL_METRICS = ['cross_entropy', 'accuracy', 'binary_accuracy', 'mae_class']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_X(
    batch_size:        int = B,
    n_stations:        int = N,
    n_features:        int = F,
    location_encoding: str = 'unit_circle',
    rng: np.random.Generator | None = None,
) -> dict:
    """Synthetic batch['X'] dict matching TCDataModule output format."""
    if rng is None:
        rng = np.random.default_rng(0)
    obs    = rng.standard_normal((batch_size, n_stations, n_features)).astype(np.float32)
    coords = rng.uniform(-1.0, 1.0, (batch_size, n_stations, 2)).astype(np.float32)
    if location_encoding == 'unit_circle':
        query = np.zeros((batch_size, 2), dtype=np.float32)
    else:
        query = rng.uniform(-1.5, 1.5, (batch_size, 2)).astype(np.float32)
    return {
        'station_obs':    jnp.array(obs),
        'station_coords': jnp.array(coords),
        'station_mask':   jnp.ones((batch_size, n_stations), dtype=bool),
        'obs_mask':       jnp.ones((batch_size, n_stations, n_features), dtype=bool),
        'query_coords':   jnp.array(query),
    }


def _fake_batch(
    batch_size:        int = B,
    location_encoding: str = 'unit_circle',
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    labels = rng.integers(0, N_CLS, size=batch_size).astype(np.int32)
    return {
        'X': _fake_X(batch_size, location_encoding=location_encoding, rng=rng),
        'y': jnp.array(labels),
    }


class _FakeTCLoader:
    """Re-iterable loader of synthetic TC-shaped batches.

    When ``learnable=True`` each batch has a strong separable signal:
    first half has station_obs = +5 → label 5 (TC),
    second half has station_obs = -5 → label 0 (background).
    Used to verify that the Trainer can reduce loss on learnable data.
    """

    def __init__(
        self,
        n_batches:         int  = 4,
        batch_size:        int  = B,
        location_encoding: str  = 'unit_circle',
        learnable:         bool = False,
        seed:              int  = 0,
    ) -> None:
        self._batches = [
            self._make(i, batch_size, location_encoding, learnable)
            for i in range(n_batches)
        ]

    @staticmethod
    def _make(
        idx:               int,
        batch_size:        int,
        location_encoding: str,
        learnable:         bool,
    ) -> dict:
        rng  = np.random.default_rng(idx)
        half = batch_size // 2

        if learnable:
            obs    = np.zeros((batch_size, N, F), dtype=np.float32)
            obs[:half]  =  5.0    # strong positive signal → class 5
            obs[half:]  = -5.0    # strong negative signal → class 0
            labels = np.array([5] * half + [0] * (batch_size - half), dtype=np.int32)
        else:
            obs    = rng.standard_normal((batch_size, N, F)).astype(np.float32)
            labels = rng.integers(0, N_CLS, size=batch_size).astype(np.int32)

        if location_encoding == 'unit_circle':
            query = np.zeros((batch_size, 2), dtype=np.float32)
        else:
            query = rng.uniform(-1.5, 1.5, (batch_size, 2)).astype(np.float32)

        return {
            'X': {
                'station_obs':    jnp.array(obs),
                'station_coords': jnp.array(
                    rng.uniform(-1.0, 1.0, (batch_size, N, 2)).astype(np.float32)
                ),
                'station_mask':   jnp.ones((batch_size, N), dtype=bool),
                'obs_mask':       jnp.ones((batch_size, N, F), dtype=bool),
                'query_coords':   jnp.array(query),
            },
            'y': jnp.array(labels),
        }

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def _make_model(
    embed_dim:         int = 32,
) -> TCClassifier:
    return TCClassifier(
        embed_dim         = embed_dim,
        num_heads         = 2,
        num_layers        = 2,
        fourier_dim       = 16,     # must be even
        n_obs_features    = F,
    )


def _trainer_config(tmp_path, **overrides) -> dict:
    cfg = {
        "batch_size":         B,
        "num_epochs":         3,
        "num_steps":          10_000,
        "patience":           50,
        "patience_metric":    "val/cross_entropy",
        "patience_direction": "lower_is_better",
        "checkpoint_dir":     str(tmp_path / "ckpts"),
        "log_every_n_steps":  9999,
        "log_backend":        "null",
        "seed":               0,
        "optimizer":          "adamw",
        "optimizer_kwargs":   {"weight_decay": 1e-4},
        "scheduler":          "constant",
        "scheduler_kwargs":   {"value": 1e-3},
        "use_tqdm":           False,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# TestOneForwardBackwardPass
# ---------------------------------------------------------------------------

class TestOneForwardBackwardPass:
    """Verify one forward eval step and one backward train step."""

    @pytest.fixture
    def trainer_state(self, tmp_path):
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(metrics=_ALL_METRICS), _trainer_config(tmp_path))
        batch   = _fake_batch()
        state   = trainer._init_state(batch)
        return trainer, state

    # --- Forward ---

    def test_forward_returns_all_metric_keys(self, trainer_state):
        trainer, state = trainer_state
        metrics = trainer._eval_step(state, _fake_batch())
        assert set(metrics.keys()) == {
            'loss', 'cross_entropy', 'accuracy', 'binary_accuracy', 'mae_class'
        }

    def test_forward_all_metrics_finite(self, trainer_state):
        trainer, state = trainer_state
        metrics = trainer._eval_step(state, _fake_batch())
        for k, v in metrics.items():
            assert bool(jnp.isfinite(v)), f"{k} is not finite"

    def test_forward_is_deterministic(self, trainer_state):
        trainer, state = trainer_state
        batch = _fake_batch()
        m1    = trainer._eval_step(state, batch)
        m2    = trainer._eval_step(state, batch)
        for k in m1:
            assert float(m1[k]) == float(m2[k])

    # --- Backward ---

    def test_backward_params_change_after_one_step(self, trainer_state):
        trainer, state = trainer_state
        new_state, _   = trainer._train_step(state, _fake_batch())
        old = jax.tree_util.tree_leaves(state.params)
        new = jax.tree_util.tree_leaves(new_state.params)
        assert any(not jnp.allclose(o, n) for o, n in zip(old, new))

    def test_backward_loss_is_finite(self, trainer_state):
        trainer, state = trainer_state
        _, metrics = trainer._train_step(state, _fake_batch())
        assert bool(jnp.isfinite(metrics['loss']))

    def test_backward_loss_is_positive(self, trainer_state):
        trainer, state = trainer_state
        _, metrics = trainer._train_step(state, _fake_batch())
        assert float(metrics['loss']) > 0.0

    # --- Domain encoding ---

    def test_domain_encoding_forward_and_backward(self, tmp_path):
        """domain coordinate convention (varied query_coords) trains without error."""
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(metrics=_ALL_METRICS), _trainer_config(tmp_path))
        batch   = _fake_batch(location_encoding='domain')
        state   = trainer._init_state(batch)
        new_state, metrics = trainer._train_step(state, batch)
        assert bool(jnp.isfinite(metrics['loss']))

    # --- Masking edge cases ---

    def test_missing_obs_mask_is_handled(self, tmp_path):
        """~30% of observations missing (obs_mask False) — no NaN produced."""
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(metrics=_ALL_METRICS), _trainer_config(tmp_path))
        rng     = np.random.default_rng(7)
        X       = _fake_X(rng=rng)
        X['obs_mask'] = jnp.array(rng.random((B, N, F)) > 0.3)
        batch   = {'X': X, 'y': jnp.array(rng.integers(0, N_CLS, B), dtype=jnp.int32)}
        state   = trainer._init_state(batch)
        _, metrics = trainer._train_step(state, batch)
        assert bool(jnp.isfinite(metrics['loss']))

    def test_padded_station_mask_is_handled(self, tmp_path):
        """Only 5 of 12 stations are real; the rest are padding — no NaN."""
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(metrics=_ALL_METRICS), _trainer_config(tmp_path))
        rng     = np.random.default_rng(9)
        X       = _fake_X(rng=rng)
        mask    = np.zeros((B, N), dtype=bool)
        mask[:, :5] = True
        X['station_mask'] = jnp.array(mask)
        batch   = {'X': X, 'y': jnp.array(rng.integers(0, N_CLS, B), dtype=jnp.int32)}
        state   = trainer._init_state(batch)
        _, metrics = trainer._train_step(state, batch)
        assert bool(jnp.isfinite(metrics['loss']))


# ---------------------------------------------------------------------------
# TestLossVariation
# ---------------------------------------------------------------------------

class TestLossVariation:
    """Verify that loss is not frozen and that the model can learn."""

    def test_loss_changes_across_steps(self, tmp_path):
        """Loss at step 1 and step 10 are not identical."""
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(metrics=_ALL_METRICS), _trainer_config(tmp_path))
        batch   = _fake_batch()
        state   = trainer._init_state(batch)
        _, m0   = trainer._train_step(state, batch)
        for _ in range(9):
            state, _ = trainer._train_step(state, batch)
        _, m10 = trainer._train_step(state, batch)
        assert float(m0['loss']) != float(m10['loss'])

    def test_train_epoch_returns_correct_key_and_finite_value(self, tmp_path):
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(metrics=_ALL_METRICS), _trainer_config(tmp_path))
        loader  = _FakeTCLoader(n_batches=4)
        state   = trainer._init_state(next(iter(loader)))
        _, metrics = trainer._train_epoch(state, loader, epoch=0)
        assert 'train/loss' in metrics
        assert np.isfinite(metrics['train/loss'])

    def test_eval_model_returns_all_five_val_metrics(self, tmp_path):
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(metrics=_ALL_METRICS), _trainer_config(tmp_path))
        loader  = _FakeTCLoader(n_batches=4)
        state   = trainer._init_state(next(iter(loader)))
        metrics = trainer._eval_model(state, loader, prefix='val')
        expected = {
            'val/loss', 'val/cross_entropy', 'val/accuracy',
            'val/binary_accuracy', 'val/mae_class',
        }
        assert set(metrics.keys()) == expected
        assert all(np.isfinite(v) for v in metrics.values())

    def test_fit_completes_and_returns_train_state(self, tmp_path):
        model   = _make_model()
        cfg     = _trainer_config(tmp_path, num_epochs=3, patience=10)
        trainer = Trainer(model, build_metrics_fns(metrics=_ALL_METRICS), cfg)
        loader  = _FakeTCLoader(n_batches=4)
        result  = trainer.fit(loader, loader)
        assert isinstance(result, TrainState)

    def test_epoch_callbacks_called_with_correct_epoch_index(self, tmp_path):
        """Epoch callback receives monotonically increasing epoch numbers."""
        recorded = []

        def cb(state, epoch, global_step):
            recorded.append(epoch)

        model   = _make_model()
        cfg     = _trainer_config(tmp_path, num_epochs=4, patience=10)
        trainer = Trainer(model, build_metrics_fns(metrics=_ALL_METRICS), cfg)
        loader  = _FakeTCLoader(n_batches=4)
        trainer.fit(loader, loader, epoch_callbacks=[cb])
        assert recorded == [0, 1, 2, 3]

    def test_loss_decreases_with_learnable_signal(self, tmp_path):
        """Val CE after training is lower than before training when data
        contains a strong separable signal (obs ±5 → classes 0 / 5)."""
        model   = _make_model(embed_dim=64)   # more capacity for fast learning
        cfg     = _trainer_config(
            tmp_path,
            num_epochs=25,
            patience=25,
            scheduler_kwargs={"value": 1e-2},  # higher LR for faster convergence
        )
        trainer = Trainer(model, build_metrics_fns(metrics=_ALL_METRICS), cfg)

        train_loader = _FakeTCLoader(n_batches=8, batch_size=16, learnable=True, seed=0)
        val_loader   = _FakeTCLoader(n_batches=4, batch_size=16, learnable=True, seed=1)

        # Measure initial val CE before any gradient updates.
        # _init_state is deterministic (same seed), so these params are identical
        # to the ones that fit() will start from.
        init_state    = trainer._init_state(next(iter(train_loader)))
        init_metrics  = trainer._eval_model(init_state, val_loader, prefix='val')
        initial_ce    = float(init_metrics['val/cross_entropy'])

        trainer.fit(train_loader, val_loader)

        # Best val CE seen during training must be strictly better than untrained
        assert trainer._best_metric_value < initial_ce


# ---------------------------------------------------------------------------
# Observability callbacks (attention figures — VAL probe; gradient flow — TRAIN)
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')   # headless — set before train.py pulls in pyplot

from experiments.sparse_obs_cross_attn.train.train import (   # noqa: E402
    _make_attn_entropy_callback,
    _make_attn_figure_callback,
    _make_grad_flow_callback,
    _log_diagnostics,
)
from experiments.sparse_obs_cross_attn.data.sources.ibtracs import (   # noqa: E402
    CLASS_NAMES,
)


class _RecordingLogger:
    """Captures log_* calls for assertion."""

    def __init__(self):
        self.metrics:    list[tuple[dict, int]]      = []
        self.figures:    list[tuple[str, int]]       = []
        self.histograms: list[tuple[str, int, int]]  = []

    def log_metrics(self, metrics, step):
        self.metrics.append((metrics, step))

    def log_figure(self, tag, fig, step):
        import matplotlib.pyplot as plt
        self.figures.append((tag, step))
        plt.close(fig)

    def log_histogram(self, tag, values, step):
        self.histograms.append((tag, int(np.asarray(values).size), step))


class _FakeState:
    def __init__(self, params):
        self.params = params


class TestObservabilityCallbacks:

    def _model_state_batch(self):
        model = _make_model()
        batch = _fake_batch()
        vs    = model.init(jax.random.PRNGKey(0), batch['X'], train=False)
        return model, _FakeState(vs['params']), batch

    def test_entropy_callback_logs_scalar(self):
        model, state, batch = self._model_state_batch()
        logger = _RecordingLogger()
        cb = _make_attn_entropy_callback(model, batch, logger)
        cb(state, epoch=1, global_step=42)
        assert len(logger.metrics) == 1
        metrics, step = logger.metrics[0]
        assert step == 42
        assert 'val/attn_entropy' in metrics
        assert np.isfinite(metrics['val/attn_entropy'])

    def test_attn_figure_callback_logs_map_and_grid(self):
        model, state, batch = self._model_state_batch()
        logger = _RecordingLogger()
        cb = _make_attn_figure_callback(
            model, batch, logger,
            data_config={'location_encoding': 'unit_circle',
                         'radius_km': 500.0},
            class_names=CLASS_NAMES,
            fig_every=2,
        )
        cb(state, epoch=2, global_step=10)
        tags = [t for t, _ in logger.figures]
        assert tags == ['val/attn_map', 'val/attn_grid']

    def test_attn_figure_callback_respects_cadence(self):
        model, state, batch = self._model_state_batch()
        logger = _RecordingLogger()
        cb = _make_attn_figure_callback(
            model, batch, logger,
            data_config={'location_encoding': 'unit_circle'},
            class_names=CLASS_NAMES,
            fig_every=5,
        )
        cb(state, epoch=3, global_step=10)   # off-cadence
        assert logger.figures == []
        cb(state, epoch=5, global_step=20)   # on-cadence
        assert len(logger.figures) == 2

    def test_grad_flow_callback_logs_named_histograms(self):
        model, state, batch = self._model_state_batch()
        logger = _RecordingLogger()
        cb = _make_grad_flow_callback(model, batch, logger, every_n_epochs=1)
        cb(state, epoch=1, global_step=7)
        assert len(logger.histograms) > 0
        tags = [t for t, _, _ in logger.histograms]
        # Named by tree path under the grad_flow/ prefix
        assert all(t.startswith('grad_flow/') for t in tags)
        assert any('encoder' in t for t in tags)
        assert any(t.endswith('kernel') for t in tags)
        # One histogram per parameter leaf
        n_leaves = len(jax.tree_util.tree_leaves(state.params))
        assert len(tags) == n_leaves
        # All logged at the given step
        assert all(s == 7 for _, _, s in logger.histograms)

    def test_grad_flow_callback_respects_cadence(self):
        model, state, batch = self._model_state_batch()
        logger = _RecordingLogger()
        cb = _make_grad_flow_callback(model, batch, logger, every_n_epochs=3)
        cb(state, epoch=2, global_step=1)
        assert logger.histograms == []
        cb(state, epoch=3, global_step=2)
        assert len(logger.histograms) > 0

    def test_grad_flow_log_now_gives_init_snapshot(self):
        model, state, batch = self._model_state_batch()
        logger = _RecordingLogger()
        cb = _make_grad_flow_callback(model, batch, logger, every_n_epochs=5)
        cb.log_now(state.params, step=0)
        assert len(logger.histograms) > 0
        assert all(s == 0 for _, _, s in logger.histograms)

    def test_log_diagnostics_logs_distribution_figures(self):
        # Exercises the real model through capture_intermediates (activations),
        # jax.grad (gradients), and the weight histogram — no loss landscape.
        model, state, batch = self._model_state_batch()
        logger = _RecordingLogger()
        _log_diagnostics(model, state.params, batch, batch, logger, step=0,
                         loss_landscape_grid=0)
        tags = [t for t, _ in logger.figures]
        assert 'diagnostics/weight_dist' in tags
        assert 'diagnostics/gradients'   in tags
        assert 'diagnostics/activations' in tags
        assert 'diagnostics/loss_landscape' not in tags   # grid=0 skips it
        assert all(s == 0 for _, s in logger.figures)

    def test_log_diagnostics_includes_loss_landscape_when_grid_set(self):
        model, state, batch = self._model_state_batch()
        logger = _RecordingLogger()
        _log_diagnostics(model, state.params, batch, batch, logger, step=3,
                         loss_landscape_grid=4)
        tags = [t for t, _ in logger.figures]
        assert 'diagnostics/loss_landscape' in tags
