"""
Tests for experiments/sparse_obs_cross_attn/train/metrics.py.

Generic metric implementations (cross_entropy, accuracy, binary_accuracy,
mae_class, quadratic_weighted_kappa, expected_calibration_error) are tested
in tests/training/test_metrics.py. This file covers only the experiment's
build_metrics_fns wiring.

Coverage
--------
TestBuildMetricsFns   expected keys; first key is loss; default loss matches
                      cross_entropy; focal+class-weighted selectable; unknown loss
                      raises; all callable; all produce finite scalars
"""

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.sparse_obs_cross_attn.train.metrics import build_metrics_fns

B     = 8
N_CLS = 11


def _rand_logits(seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.array(rng.standard_normal((B, N_CLS)).astype(np.float32))


def _rand_labels(seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.array(rng.integers(0, N_CLS, size=B), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# TestBuildMetricsFns
# ---------------------------------------------------------------------------

class TestBuildMetricsFns:

    def test_expected_keys(self):
        fns = build_metrics_fns()
        assert set(fns.keys()) == {
            'loss', 'cross_entropy', 'accuracy', 'binary_accuracy', 'mae_class'
        }

    def test_first_key_is_loss(self):
        assert list(build_metrics_fns().keys())[0] == 'loss'

    def test_default_loss_matches_cross_entropy(self):
        logits = _rand_logits()
        labels = _rand_labels()
        fns = build_metrics_fns()
        assert float(fns['loss'](logits, labels)) == pytest.approx(
            float(fns['cross_entropy'](logits, labels)), rel=1e-5
        )

    def test_focal_class_weighted_loss_selectable(self):
        logits = _rand_logits()
        labels = _rand_labels()
        fns = build_metrics_fns(
            loss='cross_entropy',
            loss_kwargs={'focal_gamma': 2.0, 'class_weights': [1.0] * N_CLS},
        )
        out = fns['loss'](logits, labels)
        assert out.shape == ()
        assert bool(jnp.isfinite(out))
        # focal modulation differs from flat cross_entropy in general
        assert float(out) != pytest.approx(float(fns['cross_entropy'](logits, labels)))

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            build_metrics_fns(loss='not_a_real_loss')

    def test_all_values_are_callable(self):
        for fn in build_metrics_fns().values():
            assert callable(fn)

    def test_all_produce_finite_scalars(self):
        logits = _rand_logits()
        labels = _rand_labels()
        for name, fn in build_metrics_fns().items():
            out = fn(logits, labels)
            assert out.shape == (), f"{name} is not scalar"
            assert bool(jnp.isfinite(out)), f"{name} is not finite"
