"""
Tests for experiments/tc_perceiver_io/train/metrics.py.

Generic metric implementations + the METRICS registry are tested in
tests/training/test_metrics.py. This file covers only the experiment's
build_metrics_fns wiring (r14: only 'loss' hardcoded, rest registry-selected).

Coverage
--------
TestBuildMetricsFns   default keys (loss + binary_accuracy + mae_class); loss is
                      first key; explicit metric selection (incl. cross_entropy);
                      focal+class-weighted loss selectable; unknown loss/metric
                      raises; all callable; all produce finite scalars
"""

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.tc_perceiver_io.train.metrics import (
    build_metrics_fns, DEFAULT_METRICS,
)

B     = 8
N_CLS = 9


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

    def test_default_keys(self):
        # Only 'loss' is hardcoded; the default reported set is
        # binary_accuracy + mae_class (top-1 accuracy demoted).
        fns = build_metrics_fns()
        assert set(fns.keys()) == {'loss', 'binary_accuracy', 'mae_class'}
        assert DEFAULT_METRICS == ('binary_accuracy', 'mae_class')

    def test_first_key_is_loss(self):
        assert list(build_metrics_fns().keys())[0] == 'loss'

    def test_explicit_metric_selection(self):
        fns = build_metrics_fns(metrics=['cross_entropy', 'accuracy'])
        assert set(fns.keys()) == {'loss', 'cross_entropy', 'accuracy'}

    def test_cross_entropy_selectable_matches_loss(self):
        # With the default loss (cross_entropy) and cross_entropy listed, the
        # 'loss' and 'cross_entropy' entries agree numerically.
        logits = _rand_logits()
        labels = _rand_labels()
        fns = build_metrics_fns(metrics=['cross_entropy'])
        assert float(fns['loss'](logits, labels)) == pytest.approx(
            float(fns['cross_entropy'](logits, labels)), rel=1e-5
        )

    def test_focal_class_weighted_loss_selectable(self):
        logits = _rand_logits()
        labels = _rand_labels()
        fns = build_metrics_fns(
            loss='cross_entropy',
            loss_kwargs={'focal_gamma': 2.0, 'class_weights': [1.0] * N_CLS},
            metrics=['cross_entropy'],
        )
        out = fns['loss'](logits, labels)
        assert out.shape == ()
        assert bool(jnp.isfinite(out))
        # focal modulation differs from flat cross_entropy in general
        assert float(out) != pytest.approx(float(fns['cross_entropy'](logits, labels)))

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            build_metrics_fns(loss='not_a_real_loss')

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            build_metrics_fns(metrics=['not_a_real_metric'])

    def test_all_values_are_callable(self):
        for fn in build_metrics_fns().values():
            assert callable(fn)

    def test_all_produce_finite_scalars(self):
        logits = _rand_logits()
        labels = _rand_labels()
        fns = build_metrics_fns(
            metrics=['cross_entropy', 'accuracy', 'binary_accuracy', 'mae_class'])
        for name, fn in fns.items():
            out = fn(logits, labels)
            assert out.shape == (), f"{name} is not scalar"
            assert bool(jnp.isfinite(out)), f"{name} is not finite"
