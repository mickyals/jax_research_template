"""
Integration tests: Trainer + TCClassifier end-to-end training.

All tests use synthetic in-memory data — no disk access required.

Coverage
--------
TestOneForwardBackwardPass
    forward pass: all four metric keys returned; all finite; deterministic
    backward pass: params change after one gradient update; loss finite and positive
    both attention paths (use_self_attention True / False)
    both location encodings (unit_circle / domain)
    missing obs (obs_mask partially False) handled without NaN
    padded stations (station_mask partially False) handled without NaN

TestLossVariation
    loss is not identical at step 1 vs step 10
    _train_epoch returns correct metric key and finite value
    _eval_model returns all four val metric keys, all finite
    loss decreases over training when data has a clear separable signal
    trainer.fit() completes and returns TrainState
    epoch callbacks receive correct (epoch, global_step) pairs
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.sparse_obs_cross_attn.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.model import TCClassifier
from training.trainer import Trainer, TrainState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B     = 8    # batch size
N     = 12   # stations per sample
F     = 5    # obs features
N_CLS = 11


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
    use_self_attention: bool = True,
    location_encoding:  str  = 'unit_circle',
    embed_dim:          int  = 32,
) -> TCClassifier:
    return TCClassifier(
        embed_dim          = embed_dim,
        num_heads          = 2,
        num_layers         = 2,
        num_cross_layers   = 1,
        fourier_dim        = 16,     # must be even
        n_obs_features     = F,
        use_self_attention = use_self_attention,
        location_encoding  = location_encoding,
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
        trainer = Trainer(model, build_metrics_fns(), _trainer_config(tmp_path))
        batch   = _fake_batch()
        state   = trainer._init_state(batch)
        return trainer, state

    # --- Forward ---

    def test_forward_returns_all_metric_keys(self, trainer_state):
        trainer, state = trainer_state
        metrics = trainer._eval_step(state, _fake_batch())
        assert set(metrics.keys()) == {
            'cross_entropy', 'accuracy', 'binary_accuracy', 'mae_class'
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
        assert bool(jnp.isfinite(metrics['cross_entropy']))

    def test_backward_loss_is_positive(self, trainer_state):
        trainer, state = trainer_state
        _, metrics = trainer._train_step(state, _fake_batch())
        assert float(metrics['cross_entropy']) > 0.0

    # --- Path B ---

    def test_path_b_forward_and_backward(self, tmp_path):
        """Path B (use_self_attention=False) trains without error."""
        model   = _make_model(use_self_attention=False)
        trainer = Trainer(model, build_metrics_fns(), _trainer_config(tmp_path))
        batch   = _fake_batch()
        state   = trainer._init_state(batch)
        new_state, metrics = trainer._train_step(state, batch)
        assert bool(jnp.isfinite(metrics['cross_entropy']))
        old = jax.tree_util.tree_leaves(state.params)
        new = jax.tree_util.tree_leaves(new_state.params)
        assert any(not jnp.allclose(o, n) for o, n in zip(old, new))

    # --- Domain encoding ---

    def test_domain_encoding_forward_and_backward(self, tmp_path):
        """domain location encoding trains without error."""
        model   = _make_model(location_encoding='domain')
        trainer = Trainer(model, build_metrics_fns(), _trainer_config(tmp_path))
        batch   = _fake_batch(location_encoding='domain')
        state   = trainer._init_state(batch)
        new_state, metrics = trainer._train_step(state, batch)
        assert bool(jnp.isfinite(metrics['cross_entropy']))

    # --- Masking edge cases ---

    def test_missing_obs_mask_is_handled(self, tmp_path):
        """~30% of observations missing (obs_mask False) — no NaN produced."""
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(), _trainer_config(tmp_path))
        rng     = np.random.default_rng(7)
        X       = _fake_X(rng=rng)
        X['obs_mask'] = jnp.array(rng.random((B, N, F)) > 0.3)
        batch   = {'X': X, 'y': jnp.array(rng.integers(0, N_CLS, B), dtype=jnp.int32)}
        state   = trainer._init_state(batch)
        _, metrics = trainer._train_step(state, batch)
        assert bool(jnp.isfinite(metrics['cross_entropy']))

    def test_padded_station_mask_is_handled(self, tmp_path):
        """Only 5 of 12 stations are real; the rest are padding — no NaN."""
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(), _trainer_config(tmp_path))
        rng     = np.random.default_rng(9)
        X       = _fake_X(rng=rng)
        mask    = np.zeros((B, N), dtype=bool)
        mask[:, :5] = True
        X['station_mask'] = jnp.array(mask)
        batch   = {'X': X, 'y': jnp.array(rng.integers(0, N_CLS, B), dtype=jnp.int32)}
        state   = trainer._init_state(batch)
        _, metrics = trainer._train_step(state, batch)
        assert bool(jnp.isfinite(metrics['cross_entropy']))


# ---------------------------------------------------------------------------
# TestLossVariation
# ---------------------------------------------------------------------------

class TestLossVariation:
    """Verify that loss is not frozen and that the model can learn."""

    def test_loss_changes_across_steps(self, tmp_path):
        """Loss at step 1 and step 10 are not identical."""
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(), _trainer_config(tmp_path))
        batch   = _fake_batch()
        state   = trainer._init_state(batch)
        _, m0   = trainer._train_step(state, batch)
        for _ in range(9):
            state, _ = trainer._train_step(state, batch)
        _, m10 = trainer._train_step(state, batch)
        assert float(m0['cross_entropy']) != float(m10['cross_entropy'])

    def test_train_epoch_returns_correct_key_and_finite_value(self, tmp_path):
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(), _trainer_config(tmp_path))
        loader  = _FakeTCLoader(n_batches=4)
        state   = trainer._init_state(next(iter(loader)))
        _, metrics = trainer._train_epoch(state, loader, epoch=0)
        assert 'train/cross_entropy' in metrics
        assert np.isfinite(metrics['train/cross_entropy'])

    def test_eval_model_returns_all_four_val_metrics(self, tmp_path):
        model   = _make_model()
        trainer = Trainer(model, build_metrics_fns(), _trainer_config(tmp_path))
        loader  = _FakeTCLoader(n_batches=4)
        state   = trainer._init_state(next(iter(loader)))
        metrics = trainer._eval_model(state, loader, prefix='val')
        expected = {
            'val/cross_entropy', 'val/accuracy',
            'val/binary_accuracy', 'val/mae_class',
        }
        assert set(metrics.keys()) == expected
        assert all(np.isfinite(v) for v in metrics.values())

    def test_fit_completes_and_returns_train_state(self, tmp_path):
        model   = _make_model()
        cfg     = _trainer_config(tmp_path, num_epochs=3, patience=10)
        trainer = Trainer(model, build_metrics_fns(), cfg)
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
        trainer = Trainer(model, build_metrics_fns(), cfg)
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
        trainer = Trainer(model, build_metrics_fns(), cfg)

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
