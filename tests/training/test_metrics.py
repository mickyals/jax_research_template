"""
Tests for training/metrics.py.

Coverage
--------
TestCrossEntropy      scalar output; finite; perfect predictions give low loss;
                      uniform logits ≈ log(n_classes); worse predictions give higher loss
TestAccuracy          all correct = 1.0; none correct = 0.0; half correct = 0.5;
                      output in [0,1]; scalar shape
TestBinaryAccuracy    perfect positive detection; perfect negative detection; all wrong;
                      output in [0,1]; scalar shape
TestMaeClass          exact match = 0.0; off-by-one = 1.0; larger offset;
                      non-negative; scalar shape
TestMetricsRegistry   registered names; get returns callable; case-insensitive;
                      threshold forwarded; unknown raises
"""

import jax.numpy as jnp
import numpy as np
import pytest

from training.metrics import (
    METRICS,
    FULL_SET_METRICS,
    accuracy,
    average_precision,
    binary_accuracy,
    binary_pr_auc,
    binary_pr_curve,
    compute_full_set_metrics,
    cross_entropy,
    get_metric,
    list_full_set_metrics,
    list_metrics,
    mae_class,
    per_class_pr_curves,
    precision_recall_curve,
)

B     = 8
N_CLS = 11


def _rand_logits(seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.array(rng.standard_normal((B, N_CLS)).astype(np.float32))


def _rand_labels(seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.array(rng.integers(0, N_CLS, size=B), dtype=jnp.int32)


def _confident_logits(correct_class: int) -> jnp.ndarray:
    """Logits with a large value on correct_class and -100 elsewhere."""
    return jnp.full((B, N_CLS), -100.0).at[:, correct_class].set(100.0)


# ---------------------------------------------------------------------------
# TestCrossEntropy
# ---------------------------------------------------------------------------

class TestCrossEntropy:

    def test_output_is_scalar(self):
        assert cross_entropy(_rand_logits(), _rand_labels()).shape == ()

    def test_output_is_finite(self):
        assert bool(jnp.isfinite(cross_entropy(_rand_logits(), _rand_labels())))

    def test_output_is_nonnegative(self):
        assert float(cross_entropy(_rand_logits(), _rand_labels())) >= 0.0

    def test_perfect_predictions_give_near_zero_loss(self):
        labels = jnp.array([3] * B, dtype=jnp.int32)
        assert float(cross_entropy(_confident_logits(3), labels)) < 0.01

    def test_uniform_logits_give_log_n_classes(self):
        labels   = jnp.zeros(B, dtype=jnp.int32)
        uniform  = jnp.zeros((B, N_CLS))
        expected = float(jnp.log(jnp.array(N_CLS)))
        assert abs(float(cross_entropy(uniform, labels)) - expected) < 0.01

    def test_wrong_logits_give_higher_loss_than_correct(self):
        labels  = jnp.array([0] * B, dtype=jnp.int32)
        correct = _confident_logits(0)
        wrong   = _confident_logits(5)
        assert float(cross_entropy(correct, labels)) < float(cross_entropy(wrong, labels))


# ---------------------------------------------------------------------------
# TestAccuracy
# ---------------------------------------------------------------------------

class TestAccuracy:

    def test_all_correct_gives_one(self):
        labels = jnp.arange(B, dtype=jnp.int32)
        logits = jnp.full((B, N_CLS), -100.0)
        for i in range(B):
            logits = logits.at[i, i].set(100.0)
        assert float(accuracy(logits, labels)) == pytest.approx(1.0)

    def test_all_wrong_gives_zero(self):
        labels = jnp.zeros(B, dtype=jnp.int32)    # all class 0
        logits = _confident_logits(5)              # all predict class 5
        assert float(accuracy(logits, labels)) == pytest.approx(0.0)

    def test_half_correct_gives_half(self):
        # first 4 are class 0, last 4 are class 5
        labels = jnp.array([0, 0, 0, 0, 5, 5, 5, 5], dtype=jnp.int32)
        logits = _confident_logits(0)              # all predict class 0
        assert float(accuracy(logits, labels)) == pytest.approx(0.5)

    def test_output_in_unit_interval(self):
        val = float(accuracy(_rand_logits(), _rand_labels()))
        assert 0.0 <= val <= 1.0

    def test_scalar_shape(self):
        assert accuracy(_rand_logits(), _rand_labels()).shape == ()


# ---------------------------------------------------------------------------
# TestBinaryAccuracy
# ---------------------------------------------------------------------------

class TestBinaryAccuracy:

    def test_perfect_positive_detection(self):
        labels = jnp.ones(B, dtype=jnp.int32) * 5   # all positive (class 5)
        logits = _confident_logits(5)                # correctly predict positive
        assert float(binary_accuracy(logits, labels)) == pytest.approx(1.0)

    def test_perfect_negative_detection(self):
        labels = jnp.zeros(B, dtype=jnp.int32)      # all negative (class 0)
        logits = _confident_logits(0)                # correctly predict class 0
        assert float(binary_accuracy(logits, labels)) == pytest.approx(1.0)

    def test_all_wrong_binary_gives_zero(self):
        labels = jnp.zeros(B, dtype=jnp.int32)      # all negative
        logits = _confident_logits(5)                # predict positive for everything
        assert float(binary_accuracy(logits, labels)) == pytest.approx(0.0)

    def test_custom_threshold(self):
        # threshold=3: class >= 3 is "positive"
        labels = jnp.array([2] * B, dtype=jnp.int32)  # below threshold -> negative
        logits = _confident_logits(2)
        assert float(binary_accuracy(logits, labels, threshold=3)) == pytest.approx(1.0)

    def test_output_in_unit_interval(self):
        val = float(binary_accuracy(_rand_logits(), _rand_labels()))
        assert 0.0 <= val <= 1.0

    def test_scalar_shape(self):
        assert binary_accuracy(_rand_logits(), _rand_labels()).shape == ()


# ---------------------------------------------------------------------------
# TestMaeClass
# ---------------------------------------------------------------------------

class TestMaeClass:

    def test_exact_match_gives_zero(self):
        labels = jnp.array([3] * B, dtype=jnp.int32)
        logits = _confident_logits(3)
        assert float(mae_class(logits, labels)) == pytest.approx(0.0)

    def test_off_by_one_gives_one(self):
        labels = jnp.array([3] * B, dtype=jnp.int32)
        logits = _confident_logits(4)              # predict class 4, truth is 3
        assert float(mae_class(logits, labels)) == pytest.approx(1.0)

    def test_off_by_three_gives_three(self):
        labels = jnp.array([0] * B, dtype=jnp.int32)
        logits = _confident_logits(3)              # predict class 3, truth is 0
        assert float(mae_class(logits, labels)) == pytest.approx(3.0)

    def test_output_is_nonnegative(self):
        assert float(mae_class(_rand_logits(), _rand_labels())) >= 0.0

    def test_scalar_shape(self):
        assert mae_class(_rand_logits(), _rand_labels()).shape == ()


# ---------------------------------------------------------------------------
# TestMetricsRegistry
# ---------------------------------------------------------------------------

class TestMetricsRegistry:

    def test_registered_names(self):
        assert set(list_metrics()) == {
            'CROSS_ENTROPY', 'ACCURACY', 'BINARY_ACCURACY', 'MAE_CLASS',
        }

    def test_get_returns_callable_metric(self):
        logits = _rand_logits()
        labels = _rand_labels()
        for name in ('cross_entropy', 'accuracy', 'binary_accuracy', 'mae_class'):
            fn = get_metric(name)
            assert callable(fn)
            out = fn(logits, labels)
            assert out.shape == () and bool(jnp.isfinite(out))

    def test_get_is_case_insensitive(self):
        assert get_metric('Cross_Entropy') is cross_entropy

    def test_binary_accuracy_threshold_forwarded(self):
        logits = _rand_logits()
        labels = _rand_labels()
        fn = get_metric('binary_accuracy', threshold=3)
        assert float(fn(logits, labels)) == pytest.approx(
            float(binary_accuracy(logits, labels, threshold=3))
        )

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            get_metric('not_a_metric')


# ---------------------------------------------------------------------------
# TestFullSetMetrics — mAP / pr_auc (full-set, NumPy)
# ---------------------------------------------------------------------------

class TestFullSetMetrics:

    def _separable(self):
        # 2-class, perfectly score-separable (class 0 vs class 1).
        logits = np.array([[6., -6.], [-6., 6.], [6., -6.], [-6., 6.]], np.float32)
        labels = np.array([0, 1, 0, 1], np.int32)
        return logits, labels

    def test_perfect_separation_gives_one(self):
        logits, labels = self._separable()
        assert average_precision(logits, labels) == pytest.approx(1.0, abs=1e-6)
        assert binary_pr_auc(logits, labels)     == pytest.approx(1.0, abs=1e-6)

    def test_metrics_in_unit_interval(self):
        rng    = np.random.default_rng(0)
        logits = rng.standard_normal((50, N_CLS)).astype(np.float32)
        labels = rng.integers(0, N_CLS, size=50).astype(np.int32)
        assert 0.0 <= average_precision(logits, labels) <= 1.0
        assert 0.0 <= binary_pr_auc(logits, labels)     <= 1.0

    def test_average_precision_skips_absent_classes(self):
        # Only classes 0 and 1 occur; the metric must still be well-defined.
        logits, labels = self._separable()
        assert 0.0 <= average_precision(logits, labels) <= 1.0

    def test_no_positives_pr_auc_is_zero(self):
        # All background (class 0) → no positives for the thr=1 detection AP.
        logits = np.zeros((5, N_CLS), np.float32)
        labels = np.zeros(5, np.int32)
        assert binary_pr_auc(logits, labels) == 0.0

    def test_compute_full_set_metrics_dict(self):
        logits, labels = self._separable()
        m = compute_full_set_metrics(logits, labels)
        assert set(m) == {'mAP', 'pr_auc'}
        assert all(0.0 <= v <= 1.0 for v in m.values())

    def test_registry_lists_names(self):
        assert set(list_full_set_metrics()) == {'MAP', 'PR_AUC'}

    def test_registry_get_is_case_insensitive(self):
        assert FULL_SET_METRICS.get('mAP') is average_precision


class TestPRCurves:

    def _rand(self, seed=1, n=60):
        rng    = np.random.default_rng(seed)
        logits = rng.standard_normal((n, N_CLS)).astype(np.float32)
        labels = rng.integers(0, N_CLS, n).astype(np.int32)
        return logits, labels

    def test_curve_ap_equals_scalar(self):
        # The figure's AP must match the pr_auc scalar (shared code path).
        logits, labels = self._rand()
        cv = binary_pr_curve(logits, labels)
        assert cv['ap'] == pytest.approx(binary_pr_auc(logits, labels), abs=1e-9)

    def test_curve_shape_and_bounds(self):
        logits, labels = self._rand()
        cv = binary_pr_curve(logits, labels)
        assert len(cv['precision']) == len(cv['recall'])
        assert cv['recall'][0] == 0.0
        assert cv['recall'][-1] == pytest.approx(1.0)
        assert np.all((cv['precision'] >= 0) & (cv['precision'] <= 1.0 + 1e-9))
        assert 0.0 <= cv['base_rate'] <= 1.0

    def test_recall_is_monotonic(self):
        logits, labels = self._rand(seed=2)
        cv = binary_pr_curve(logits, labels)
        assert np.all(np.diff(cv['recall']) >= -1e-9)

    def test_per_class_curves_match_map(self):
        # Mean of present-class APs equals mAP exactly.
        logits, labels = self._rand(seed=3)
        curves = per_class_pr_curves(logits, labels)
        assert set(curves) == set(np.unique(labels).tolist())
        mean_ap = float(np.mean([cv['ap'] for cv in curves.values()]))
        assert mean_ap == pytest.approx(average_precision(logits, labels), abs=1e-9)

    def test_no_positives_curve_is_degenerate(self):
        cv = precision_recall_curve(np.zeros(5), np.zeros(5, dtype=bool))
        assert cv['ap'] == 0.0 and cv['base_rate'] == 0.0
