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
    accuracy,
    binary_accuracy,
    cross_entropy,
    get_metric,
    list_metrics,
    macro_precision,
    macro_recall,
    mae_class,
    per_class_counts,
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
            'MACRO_PRECISION', 'MACRO_RECALL',
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
# TestPerClassCounts + macro precision/recall (jrt-v2 additions)
# ---------------------------------------------------------------------------

def _logits_for_preds(preds, n_cls=N_CLS):
    """Logits whose argmax is exactly `preds`."""
    out = np.full((len(preds), n_cls), -100.0, np.float32)
    out[np.arange(len(preds)), preds] = 100.0
    return jnp.array(out)


class TestPerClassCounts:

    def test_known_counts(self):
        from training.metrics import per_class_counts
        # truth [0,0,1], preds [0,1,1]
        tp, fp, fn = per_class_counts(_logits_for_preds([0, 1, 1]),
                                      jnp.array([0, 0, 1]))
        assert (float(tp[0]), float(fp[0]), float(fn[0])) == (1., 0., 1.)
        assert (float(tp[1]), float(fp[1]), float(fn[1])) == (1., 1., 0.)
        assert float(tp[2:].sum() + fp[2:].sum() + fn[2:].sum()) == 0.0

    def test_counts_sum_across_batches(self):
        from training.metrics import per_class_counts
        rng = np.random.default_rng(0)
        preds  = rng.integers(0, N_CLS, 32)
        labels = jnp.array(rng.integers(0, N_CLS, 32), dtype=jnp.int32)
        logits = _logits_for_preds(preds)
        whole  = per_class_counts(logits, labels)
        halves = [per_class_counts(logits[:16], labels[:16]),
                  per_class_counts(logits[16:], labels[16:])]
        for w, a, b in zip(whole, halves[0], halves[1]):
            assert jnp.allclose(w, a + b)   # counts accumulate EXACTLY


class TestMacroPrecisionRecall:

    def test_perfect_predictions_give_one(self):
        labels = _rand_labels()
        logits = _logits_for_preds(np.asarray(labels))
        assert float(macro_precision(logits, labels)) == pytest.approx(1.0)
        assert float(macro_recall(logits, labels))    == pytest.approx(1.0)

    def test_known_values(self):
        # truth [0,0,1], preds [0,1,1]:
        # precision: c0 1/1, c1 1/2 -> 0.75 ; recall: c0 1/2, c1 1/1 -> 0.75
        logits = _logits_for_preds([0, 1, 1])
        labels = jnp.array([0, 0, 1])
        assert float(macro_precision(logits, labels)) == pytest.approx(0.75)
        assert float(macro_recall(logits, labels))    == pytest.approx(0.75)

    def test_absent_classes_skipped(self):
        # Only class 3 present + predicted; other classes must not dilute.
        logits = _logits_for_preds([3, 3])
        labels = jnp.array([3, 3])
        assert float(macro_precision(logits, labels)) == pytest.approx(1.0)
        assert float(macro_recall(logits, labels))    == pytest.approx(1.0)

    def test_all_wrong_gives_zero(self):
        logits = _logits_for_preds([1, 1])
        labels = jnp.array([0, 0])
        assert float(macro_precision(logits, labels)) == 0.0
        assert float(macro_recall(logits, labels))    == 0.0

    def test_scalar_in_unit_interval(self):
        logits, labels = _rand_logits(), _rand_labels(seed=1)
        for fn in (macro_precision, macro_recall):
            v = fn(logits, labels)
            assert v.shape == () and 0.0 <= float(v) <= 1.0

    def test_registered(self):
        assert callable(get_metric('macro_precision'))
        assert callable(get_metric('macro_recall'))
