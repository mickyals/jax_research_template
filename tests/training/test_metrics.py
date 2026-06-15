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
TestQuadraticWeightedKappa
                      perfect agreement = 1.0; anti-ordinal cm is negative;
                      near-miss scores higher than far-miss at equal accuracy;
                      degenerate single-class cm -> 0.0, no crash
TestExpectedCalibrationError
                      perfectly-calibrated probs ~ 0; confident-but-wrong is
                      high; empty-bin and single-sample don't crash; in [0,1]
TestMaximumCalibrationError
                      empty -> 0; MCE >= ECE; confident-but-wrong high; in [0,1]
TestTemperatureScaling
                      apply divides + preserves argmax; empty -> T=1;
                      overconfident -> T>1 and lower ECE; calibrated -> T~1;
                      T always positive
"""

import jax.numpy as jnp
import numpy as np
import pytest

from training.metrics import (
    accuracy,
    apply_temperature,
    binary_accuracy,
    cross_entropy,
    expected_calibration_error,
    fit_temperature,
    mae_class,
    maximum_calibration_error,
    quadratic_weighted_kappa,
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
# TestQuadraticWeightedKappa
# ---------------------------------------------------------------------------

class TestQuadraticWeightedKappa:

    def test_perfect_agreement_gives_one(self):
        cm = np.eye(N_CLS, dtype=np.int64) * 5
        assert quadratic_weighted_kappa(cm) == pytest.approx(1.0)

    def test_anti_ordinal_is_negative(self):
        # All true class 0 predicted as class 10, and vice versa — the
        # worst-possible ordinal confusion.
        cm = np.zeros((N_CLS, N_CLS), dtype=np.int64)
        cm[0, N_CLS - 1] = 10
        cm[N_CLS - 1, 0] = 10
        assert quadratic_weighted_kappa(cm) < 0.0

    def test_near_miss_scores_higher_than_far_miss_at_equal_accuracy(self):
        cm_near = np.eye(N_CLS, dtype=np.int64) * 8
        cm_near[0, 1] = 2  # off by one

        cm_far = np.eye(N_CLS, dtype=np.int64) * 8
        cm_far[0, N_CLS - 1] = 2  # off by ten

        kappa_near = quadratic_weighted_kappa(cm_near)
        kappa_far  = quadratic_weighted_kappa(cm_far)
        assert kappa_near > kappa_far

    def test_degenerate_single_class_gives_zero(self):
        cm = np.zeros((N_CLS, N_CLS), dtype=np.int64)
        cm[0, 0] = 10
        assert quadratic_weighted_kappa(cm) == pytest.approx(0.0)

    def test_empty_confusion_matrix_gives_zero(self):
        cm = np.zeros((N_CLS, N_CLS), dtype=np.int64)
        assert quadratic_weighted_kappa(cm) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestExpectedCalibrationError
# ---------------------------------------------------------------------------

class TestExpectedCalibrationError:

    def test_perfectly_calibrated_is_near_zero(self):
        # confidence == accuracy within each bin: 80% of the probability
        # mass on the correct class, and the model is correct 80% of the time.
        rng    = np.random.default_rng(0)
        n      = 1000
        labels = rng.integers(0, N_CLS, size=n)
        correct_mask = rng.random(n) < 0.8

        probs = np.full((n, N_CLS), 0.02 / (N_CLS - 1))
        for i in range(n):
            target = labels[i] if correct_mask[i] else (labels[i] + 1) % N_CLS
            probs[i] = (1.0 - 0.8) / (N_CLS - 1)
            probs[i, target] = 0.8

        ece = expected_calibration_error(probs, labels)
        assert ece < 0.05

    def test_confident_but_wrong_is_high(self):
        n      = 200
        labels = np.zeros(n, dtype=np.int64)
        wrong  = np.full(n, 1, dtype=np.int64)

        probs = np.full((n, N_CLS), (1.0 - 0.95) / (N_CLS - 1))
        probs[np.arange(n), wrong] = 0.95

        ece = expected_calibration_error(probs, labels)
        assert ece > 0.5

    def test_empty_input_gives_zero(self):
        probs  = np.zeros((0, N_CLS))
        labels = np.zeros((0,), dtype=np.int64)
        assert expected_calibration_error(probs, labels) == pytest.approx(0.0)

    def test_single_sample_does_not_crash(self):
        probs  = np.full((1, N_CLS), 1.0 / N_CLS)
        labels = np.array([0], dtype=np.int64)
        ece = expected_calibration_error(probs, labels)
        assert np.isfinite(ece)

    def test_output_in_unit_interval(self):
        logits = np.asarray(_rand_logits())
        labels = np.asarray(_rand_labels())
        probs  = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        ece    = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# TestMaximumCalibrationError
# ---------------------------------------------------------------------------

class TestMaximumCalibrationError:

    def test_empty_input_gives_zero(self):
        assert maximum_calibration_error(np.zeros((0, N_CLS)),
                                         np.zeros((0,), dtype=np.int64)) == pytest.approx(0.0)

    def test_mce_at_least_ece(self):
        # worst bin >= occupancy-weighted average, always.
        rng    = np.random.default_rng(0)
        n      = 300
        labels = rng.integers(0, N_CLS, size=n)
        probs  = rng.random((n, N_CLS)); probs /= probs.sum(-1, keepdims=True)
        assert (maximum_calibration_error(probs, labels)
                >= expected_calibration_error(probs, labels) - 1e-9)

    def test_confident_but_wrong_is_high(self):
        n      = 200
        labels = np.zeros(n, dtype=np.int64)
        probs  = np.full((n, N_CLS), (1.0 - 0.95) / (N_CLS - 1))
        probs[:, 1] = 0.95            # confident on the wrong class
        assert maximum_calibration_error(probs, labels) > 0.5

    def test_output_in_unit_interval(self):
        logits = np.asarray(_rand_logits())
        labels = np.asarray(_rand_labels())
        probs  = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        assert 0.0 <= maximum_calibration_error(probs, labels) <= 1.0


# ---------------------------------------------------------------------------
# TestTemperatureScaling
# ---------------------------------------------------------------------------

class TestTemperatureScaling:

    def test_apply_temperature_divides(self):
        logits = _rand_logits()
        out = apply_temperature(np.asarray(logits), 2.0)
        assert np.allclose(out, np.asarray(logits) / 2.0)

    def test_apply_temperature_preserves_argmax(self):
        logits = np.asarray(_rand_logits())
        for T in (0.5, 2.0, 5.0):
            scaled = apply_temperature(logits, T)
            assert np.array_equal(scaled.argmax(-1), logits.argmax(-1))

    def test_empty_input_returns_unit_temperature(self):
        assert fit_temperature(np.zeros((0, N_CLS)), np.zeros((0,), dtype=np.int64)) == 1.0

    def test_overconfident_logits_fit_temperature_above_one(self):
        # Confident but only ~half correct -> needs softening (T > 1).
        rng = np.random.default_rng(0)
        n = 400
        labels = rng.integers(0, N_CLS, size=n)
        # Large logits on a class that is the true one only half the time.
        target = np.where(rng.random(n) < 0.5, labels, (labels + 1) % N_CLS)
        logits = np.full((n, N_CLS), -5.0)
        logits[np.arange(n), target] = 5.0
        T = fit_temperature(logits, labels)
        assert T > 1.0

    def test_temperature_reduces_ece_when_overconfident(self):
        rng = np.random.default_rng(1)
        n = 400
        labels = rng.integers(0, N_CLS, size=n)
        target = np.where(rng.random(n) < 0.5, labels, (labels + 1) % N_CLS)
        logits = np.full((n, N_CLS), -5.0)
        logits[np.arange(n), target] = 5.0

        T = fit_temperature(logits, labels)
        probs_before = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        scaled = apply_temperature(logits, T)
        probs_after = np.exp(scaled) / np.exp(scaled).sum(-1, keepdims=True)

        ece_before = expected_calibration_error(probs_before, labels)
        ece_after  = expected_calibration_error(probs_after, labels)
        assert ece_after < ece_before

    def test_well_calibrated_temperature_near_one(self):
        # Draw labels from the model's OWN softmax -> perfectly calibrated by
        # construction, so the optimal temperature is ~1.
        rng = np.random.default_rng(2)
        n = 3000
        logits = rng.standard_normal((n, N_CLS))
        probs  = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        labels = np.array([rng.choice(N_CLS, p=probs[i]) for i in range(n)])
        T = fit_temperature(logits, labels)
        assert 0.7 < T < 1.4   # near 1, not pinned to a bound

    def test_positive_temperature(self):
        logits = np.asarray(_rand_logits())
        labels = np.asarray(_rand_labels())
        assert fit_temperature(logits, labels) > 0.0
