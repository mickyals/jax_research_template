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
TestConfusionCounts   known matrix; exact summation across batches
TestUpdateCm          accumulates confusion_counts into the state array;
                      dtype preserved; matches whole-split matrix
TestComputeFinalMetrics
                      tp/tn/fp/fn from a known cm; exact macro P/R; overall +
                      per-class (recall) accuracy; OVA accuracy; pairwise
                      accuracy matrix; zero-support classes excluded
"""

import jax.numpy as jnp
import numpy as np
import pytest

from training.metrics import (
    METRICS,
    accuracy,
    binary_accuracy,
    compute_final_metrics,
    confusion_counts,
    cross_entropy,
    get_metric,
    list_metrics,
    mae_class,
    update_cm,
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
        # macro precision/recall deliberately absent: ratios don't average
        # across batches — exact values come from update_cm +
        # compute_final_metrics instead (PR #5 ruling).
        assert set(list_metrics()) == {
            'CROSS_ENTROPY', 'ACCURACY', 'BINARY_ACCURACY', 'MAE_CLASS',
        }

    def test_macro_metrics_no_longer_registered(self):
        for name in ('macro_precision', 'macro_recall'):
            with pytest.raises(ValueError):
                get_metric(name)

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
# Confusion-matrix accumulator (jrt-v2, reshaped per PR #5 review)
# ---------------------------------------------------------------------------

def _logits_for_preds(preds, n_cls=N_CLS):
    """Logits whose argmax is exactly `preds`."""
    out = np.full((len(preds), n_cls), -100.0, np.float32)
    out[np.arange(len(preds)), preds] = 100.0
    return jnp.array(out)


class TestConfusionCounts:

    def test_known_matrix(self):
        # truth [0,0,1], preds [0,1,1]
        cm = confusion_counts(_logits_for_preds([0, 1, 1]),
                              jnp.array([0, 0, 1]))
        assert cm.shape == (N_CLS, N_CLS)
        assert float(cm[0, 0]) == 1. and float(cm[0, 1]) == 1.
        assert float(cm[1, 1]) == 1. and float(cm.sum()) == 3.

    def test_sums_exactly_across_batches(self):
        rng = np.random.default_rng(3)
        preds  = rng.integers(0, N_CLS, 32)
        labels = jnp.array(rng.integers(0, N_CLS, 32), dtype=jnp.int32)
        logits = _logits_for_preds(preds)
        whole = confusion_counts(logits, labels)
        assert jnp.allclose(whole,
                            confusion_counts(logits[:16], labels[:16])
                            + confusion_counts(logits[16:], labels[16:]))


class TestUpdateCm:

    def test_single_update_equals_confusion_counts(self):
        logits, labels = _logits_for_preds([0, 1, 1]), jnp.array([0, 0, 1])
        cm = update_cm(jnp.zeros((N_CLS, N_CLS), jnp.float32), logits, labels)
        assert jnp.allclose(cm, confusion_counts(logits, labels))

    def test_accumulates_to_whole_split_matrix(self):
        rng = np.random.default_rng(7)
        preds  = rng.integers(0, N_CLS, 48)
        labels = jnp.array(rng.integers(0, N_CLS, 48), dtype=jnp.int32)
        logits = _logits_for_preds(preds)
        cm = jnp.zeros((N_CLS, N_CLS), jnp.float32)
        for lo in range(0, 48, 16):                       # 3 "batches"
            cm = update_cm(cm, logits[lo:lo + 16], labels[lo:lo + 16])
        assert jnp.allclose(cm, confusion_counts(logits, labels))
        assert float(cm.sum()) == 48.0

    def test_state_dtype_preserved(self):
        cm = update_cm(jnp.zeros((N_CLS, N_CLS), jnp.int32),
                       _logits_for_preds([2]), jnp.array([2]))
        assert cm.dtype == jnp.int32 and int(cm[2, 2]) == 1


class TestComputeFinalMetrics:

    # truth [0,0,1], preds [0,1,1] over 3 classes:
    #   cm = [[1,1,0],[0,1,0],[0,0,0]]
    @staticmethod
    def _known_cm():
        return jnp.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 0.]])

    def test_count_primitives(self):
        m = compute_final_metrics(self._known_cm())
        assert jnp.allclose(m['tp'], jnp.array([1., 1., 0.]))
        assert jnp.allclose(m['fp'], jnp.array([0., 1., 0.]))
        assert jnp.allclose(m['fn'], jnp.array([1., 0., 0.]))
        assert jnp.allclose(m['tn'], jnp.array([1., 1., 3.]))
        assert jnp.allclose(m['support'], jnp.array([2., 1., 0.]))

    def test_exact_macro_precision_recall(self):
        m = compute_final_metrics(self._known_cm())
        # precision: c0 1/1, c1 1/2 (c2 never predicted -> excluded) -> 0.75
        # recall:    c0 1/2, c1 1/1 (c2 no support -> excluded)     -> 0.75
        assert float(m['macro_precision']) == pytest.approx(0.75)
        assert float(m['macro_recall'])    == pytest.approx(0.75)
        assert jnp.allclose(m['precision'], jnp.array([1.0, 0.5, 0.0]))
        assert jnp.allclose(m['recall'],    jnp.array([0.5, 1.0, 0.0]))

    def test_overall_and_ova_accuracy(self):
        m = compute_final_metrics(self._known_cm())
        assert float(m['accuracy']) == pytest.approx(2.0 / 3.0)
        # OVA k vs rest: (tp_k + tn_k) / n
        assert jnp.allclose(m['ova_accuracy'],
                            jnp.array([2 / 3, 2 / 3, 1.0]))

    def test_pairwise_accuracy_matrix(self):
        m = compute_final_metrics(self._known_cm())
        pw = m['pairwise_accuracy']
        assert pw.shape == (3, 3)
        # (0,1): (1+1)/(1+1+1+0) = 2/3 ; symmetric
        assert float(pw[0, 1]) == pytest.approx(2 / 3)
        assert float(pw[1, 0]) == pytest.approx(2 / 3)
        # (0,2) and (1,2): no cross-confusion -> 1.0
        assert float(pw[0, 2]) == pytest.approx(1.0)
        assert float(pw[1, 2]) == pytest.approx(1.0)
        # diagonal: trivial self-pair -> 1.0 where the class has correct hits
        assert float(pw[0, 0]) == pytest.approx(1.0)

    def test_perfect_cm_gives_ones(self):
        cm = jnp.diag(jnp.array([5., 3., 2.]))
        m = compute_final_metrics(cm)
        assert float(m['accuracy'])        == pytest.approx(1.0)
        assert float(m['macro_precision']) == pytest.approx(1.0)
        assert float(m['macro_recall'])    == pytest.approx(1.0)
        assert jnp.allclose(m['ova_accuracy'], jnp.ones(3))

    def test_matches_accumulated_stream(self):
        rng = np.random.default_rng(11)
        preds  = rng.integers(0, N_CLS, 64)
        labels_np = rng.integers(0, N_CLS, 64)
        labels = jnp.array(labels_np, dtype=jnp.int32)
        logits = _logits_for_preds(preds)
        cm = jnp.zeros((N_CLS, N_CLS), jnp.float32)
        for lo in range(0, 64, 16):
            cm = update_cm(cm, logits[lo:lo + 16], labels[lo:lo + 16])
        m = compute_final_metrics(cm)
        assert float(m['accuracy']) == pytest.approx(
            float(np.mean(preds == labels_np)))
