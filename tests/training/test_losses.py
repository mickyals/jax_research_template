"""
Tests for training/losses.py.

Coverage
--------
TestOptaxAgreement      our scalar reductions agree with optax element-wise functions
TestMse                 correctness, shapes, edge cases
TestRmse                correctness
TestMae                 correctness, shapes
TestHuber               quadratic / linear regimes, boundary
TestLogCosh             zero residual, small-error approximation, symmetry
TestMaskedMse           NaN exclusion, explicit mask, all-NaN, shape mismatch
TestMaskedRmse          NaN exclusion
TestMaskedMae           NaN exclusion, explicit mask, all-NaN
TestMaskedHuber         NaN exclusion, explicit mask, all-NaN
TestMaskedLogCosh       NaN exclusion, all-NaN
TestGradients           grad shapes, finite grads, zero grad at NaN positions
TestJit                 JIT compatibility for all functions
TestReExports           optax losses accessible from training.losses
"""

import jax
import jax.numpy as jnp
import optax.losses as optax_losses
import pytest

from training.losses import (
    # scalar reductions
    mse, rmse, mae, huber, log_cosh,
    # masked
    masked_mse, masked_rmse, masked_mae, masked_huber, masked_log_cosh,
    # optax re-exports
    l2_loss, squared_error, huber_loss, log_cosh_loss,
    sigmoid_binary_cross_entropy,
    softmax_cross_entropy,
    softmax_cross_entropy_with_integer_labels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean():
    """Pred and target with no NaN; error = 0.5 everywhere."""
    pred   = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, 2.5, 3.5])
    return pred, target


@pytest.fixture
def with_nan():
    """Middle value NaN — positions 0 and 2 are valid."""
    pred   = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, jnp.nan, 3.5])
    return pred, target


@pytest.fixture
def all_nan():
    pred   = jnp.array([1.0, 2.0, 3.0])
    target = jnp.full(3, jnp.nan)
    return pred, target


# ---------------------------------------------------------------------------
# TestOptaxAgreement — our reductions must match optax element-wise functions
# ---------------------------------------------------------------------------

class TestOptaxAgreement:

    def test_mse_matches_squared_error(self, clean):
        pred, target = clean
        expected = float(jnp.mean(optax_losses.squared_error(pred, target)))
        assert float(mse(pred, target)) == pytest.approx(expected, rel=1e-5)

    def test_mse_is_twice_l2_loss(self, clean):
        pred, target = clean
        expected = float(2.0 * jnp.mean(optax_losses.l2_loss(pred, target)))
        assert float(mse(pred, target)) == pytest.approx(expected, rel=1e-5)

    def test_huber_matches_optax(self, clean):
        pred, target = clean
        expected = float(jnp.mean(optax_losses.huber_loss(pred, target, delta=1.0)))
        assert float(huber(pred, target, delta=1.0)) == pytest.approx(expected, rel=1e-5)

    def test_log_cosh_matches_optax(self, clean):
        pred, target = clean
        expected = float(jnp.mean(optax_losses.log_cosh(pred, target)))
        assert float(log_cosh(pred, target)) == pytest.approx(expected, rel=1e-5)

    def test_masked_mse_agrees_on_clean_data(self, clean):
        pred, target = clean
        assert float(masked_mse(pred, target)) == pytest.approx(
            float(mse(pred, target)), rel=1e-5
        )

    def test_masked_huber_agrees_on_clean_data(self, clean):
        pred, target = clean
        assert float(masked_huber(pred, target)) == pytest.approx(
            float(huber(pred, target)), rel=1e-5
        )

    def test_masked_log_cosh_agrees_on_clean_data(self, clean):
        pred, target = clean
        assert float(masked_log_cosh(pred, target)) == pytest.approx(
            float(log_cosh(pred, target)), rel=1e-5
        )


# ---------------------------------------------------------------------------
# TestMse
# ---------------------------------------------------------------------------

class TestMse:

    def test_known_value(self, clean):
        pred, target = clean
        assert float(mse(pred, target)) == pytest.approx(0.25, rel=1e-5)

    def test_perfect_prediction(self, clean):
        pred, _ = clean
        assert float(mse(pred, pred)) == pytest.approx(0.0, abs=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            mse(jnp.ones(3), jnp.ones(4))

    def test_2d_input(self):
        pred   = jnp.ones((4, 3))
        target = jnp.zeros((4, 3))
        assert float(mse(pred, target)) == pytest.approx(1.0)

    def test_scalar_output(self, clean):
        pred, target = clean
        assert mse(pred, target).ndim == 0


# ---------------------------------------------------------------------------
# TestRmse
# ---------------------------------------------------------------------------

class TestRmse:

    def test_known_value(self, clean):
        pred, target = clean
        assert float(rmse(pred, target)) == pytest.approx(0.5, rel=1e-5)

    def test_equals_sqrt_mse(self, clean):
        pred, target = clean
        assert float(rmse(pred, target)) == pytest.approx(
            float(jnp.sqrt(mse(pred, target))), rel=1e-5
        )


# ---------------------------------------------------------------------------
# TestMae
# ---------------------------------------------------------------------------

class TestMae:

    def test_known_value(self, clean):
        pred, target = clean
        assert float(mae(pred, target)) == pytest.approx(0.5, rel=1e-5)

    def test_perfect_prediction(self, clean):
        pred, _ = clean
        assert float(mae(pred, pred)) == pytest.approx(0.0, abs=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            mae(jnp.ones(3), jnp.ones(4))


# ---------------------------------------------------------------------------
# TestHuber
# ---------------------------------------------------------------------------

class TestHuber:

    def test_quadratic_regime(self, clean):
        # all errors = 0.5, delta=1.0 -> 0.5 * 0.5^2 = 0.125
        pred, target = clean
        assert float(huber(pred, target, delta=1.0)) == pytest.approx(0.125, rel=1e-5)

    def test_linear_regime(self):
        # error = 2.0, delta=1.0 -> 1.0 * 2.0 - 0.5 * 1.0^2 = 1.5
        pred   = jnp.array([0.0])
        target = jnp.array([2.0])
        assert float(huber(pred, target, delta=1.0)) == pytest.approx(1.5, rel=1e-5)

    def test_boundary_quadratic_linear_agree(self):
        # at error == delta both regimes must give the same value
        pred   = jnp.array([0.0])
        target = jnp.array([1.0])
        assert float(huber(pred, target, delta=1.0)) == pytest.approx(0.5, rel=1e-5)

    def test_perfect_prediction(self, clean):
        pred, _ = clean
        assert float(huber(pred, pred)) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestLogCosh
# ---------------------------------------------------------------------------

class TestLogCosh:

    def test_zero_residual(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert float(log_cosh(x, x)) == pytest.approx(0.0, abs=1e-6)

    def test_approximately_mse_for_small_errors(self):
        # log(cosh(x)) ~ x^2/2 for small |x|
        pred   = jnp.array([0.01])
        target = jnp.array([0.0])
        assert float(log_cosh(pred, target)) == pytest.approx(0.5 * 0.01**2, abs=1e-4)

    def test_symmetric(self):
        pred   = jnp.array([1.0, 2.0])
        target = jnp.array([0.0, 3.0])
        assert float(log_cosh(pred, target)) == pytest.approx(
            float(log_cosh(target, pred)), rel=1e-5
        )

    def test_nonnegative(self):
        pred   = jnp.array([1.0, -1.0, 0.5])
        target = jnp.array([0.0,  2.0, 0.5])
        assert float(log_cosh(pred, target)) >= 0.0


# ---------------------------------------------------------------------------
# TestMaskedMse
# ---------------------------------------------------------------------------

class TestMaskedMse:

    def test_nan_excluded(self, with_nan):
        # valid errors: 0.5 and 0.5 -> MSE = 0.25
        pred, target = with_nan
        assert float(masked_mse(pred, target)) == pytest.approx(0.25, rel=1e-5)

    def test_agrees_with_plain_mse_no_nan(self, clean):
        pred, target = clean
        assert float(masked_mse(pred, target)) == pytest.approx(
            float(mse(pred, target)), rel=1e-5
        )

    def test_all_nan_returns_zero(self, all_nan):
        pred, target = all_nan
        assert float(masked_mse(pred, target)) == 0.0

    def test_explicit_mask(self):
        pred   = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 99.0, 3.5])
        mask   = jnp.array([True, False, True])
        assert float(masked_mse(pred, target, mask=mask)) == pytest.approx(0.25, rel=1e-5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            masked_mse(jnp.ones(3), jnp.ones(4))


# ---------------------------------------------------------------------------
# TestMaskedRmse
# ---------------------------------------------------------------------------

class TestMaskedRmse:

    def test_known_value(self, with_nan):
        pred, target = with_nan
        assert float(masked_rmse(pred, target)) == pytest.approx(0.5, rel=1e-5)

    def test_equals_sqrt_masked_mse(self, with_nan):
        pred, target = with_nan
        assert float(masked_rmse(pred, target)) == pytest.approx(
            float(jnp.sqrt(masked_mse(pred, target))), rel=1e-5
        )


# ---------------------------------------------------------------------------
# TestMaskedMae
# ---------------------------------------------------------------------------

class TestMaskedMae:

    def test_nan_excluded(self, with_nan):
        pred, target = with_nan
        assert float(masked_mae(pred, target)) == pytest.approx(0.5, rel=1e-5)

    def test_all_nan_returns_zero(self, all_nan):
        pred, target = all_nan
        assert float(masked_mae(pred, target)) == 0.0

    def test_explicit_mask(self):
        pred   = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 99.0, 3.5])
        mask   = jnp.array([True, False, True])
        assert float(masked_mae(pred, target, mask=mask)) == pytest.approx(0.5, rel=1e-5)


# ---------------------------------------------------------------------------
# TestMaskedHuber
# ---------------------------------------------------------------------------

class TestMaskedHuber:

    def test_nan_excluded(self, with_nan):
        # valid errors = 0.5, delta=1.0 -> quadratic: 0.125
        pred, target = with_nan
        assert float(masked_huber(pred, target, delta=1.0)) == pytest.approx(0.125, rel=1e-5)

    def test_all_nan_returns_zero(self, all_nan):
        pred, target = all_nan
        assert float(masked_huber(pred, target)) == 0.0

    def test_explicit_mask_linear_regime(self):
        # error = 2.0, delta=1.0 -> 1.5; position 1 masked out
        pred   = jnp.array([0.0, 0.0])
        target = jnp.array([2.0, 99.0])
        mask   = jnp.array([True, False])
        assert float(masked_huber(pred, target, delta=1.0, mask=mask)) == pytest.approx(1.5, rel=1e-5)


# ---------------------------------------------------------------------------
# TestMaskedLogCosh
# ---------------------------------------------------------------------------

class TestMaskedLogCosh:

    def test_nan_excluded(self, with_nan):
        pred, target = with_nan
        valid_pred   = jnp.array([1.0, 3.0])
        valid_target = jnp.array([1.5, 3.5])
        expected = float(log_cosh(valid_pred, valid_target))
        assert float(masked_log_cosh(pred, target)) == pytest.approx(expected, rel=1e-4)

    def test_all_nan_returns_zero(self, all_nan):
        pred, target = all_nan
        assert float(masked_log_cosh(pred, target)) == 0.0


# ---------------------------------------------------------------------------
# TestGradients
# ---------------------------------------------------------------------------

class TestGradients:

    @pytest.mark.parametrize("loss_fn", [mse, rmse, mae, huber, log_cosh])
    def test_pointwise_grad_shape(self, loss_fn, clean):
        pred, target = clean
        grad = jax.grad(lambda p: loss_fn(p, target))(pred)
        assert grad.shape == pred.shape
        assert jnp.all(jnp.isfinite(grad))

    @pytest.mark.parametrize("loss_fn", [
        masked_mse, masked_rmse, masked_mae, masked_huber, masked_log_cosh
    ])
    def test_masked_grad_shape(self, loss_fn, with_nan):
        pred, target = with_nan
        grad = jax.grad(lambda p: loss_fn(p, target))(pred)
        assert grad.shape == pred.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_zero_at_nan_position(self, with_nan):
        """NaN target forces gradient to zero at that position."""
        pred, target = with_nan  # position 1 is NaN
        grad = jax.grad(lambda p: masked_mse(p, target))(pred)
        assert float(grad[1]) == 0.0

    def test_grad_nonzero_at_valid_positions(self, with_nan):
        pred, target = with_nan
        grad = jax.grad(lambda p: masked_mse(p, target))(pred)
        assert float(grad[0]) != 0.0
        assert float(grad[2]) != 0.0


# ---------------------------------------------------------------------------
# TestJit
# ---------------------------------------------------------------------------

class TestJit:

    @pytest.mark.parametrize("loss_fn", [mse, rmse, mae, huber, log_cosh])
    def test_pointwise_jit(self, loss_fn, clean):
        pred, target = clean
        out = jax.jit(loss_fn)(pred, target)
        assert jnp.isfinite(out)

    @pytest.mark.parametrize("loss_fn", [
        masked_mse, masked_rmse, masked_mae, masked_huber, masked_log_cosh
    ])
    def test_masked_jit(self, loss_fn, with_nan):
        pred, target = with_nan
        out = jax.jit(loss_fn)(pred, target)
        assert jnp.isfinite(out)


# ---------------------------------------------------------------------------
# TestReExports — optax losses must be importable from training.losses
# ---------------------------------------------------------------------------

class TestReExports:

    def test_l2_loss_importable(self):
        assert callable(l2_loss)

    def test_squared_error_importable(self):
        assert callable(squared_error)

    def test_huber_loss_importable(self):
        assert callable(huber_loss)

    def test_log_cosh_loss_importable(self):
        assert callable(log_cosh_loss)

    def test_sigmoid_binary_cross_entropy_importable(self):
        assert callable(sigmoid_binary_cross_entropy)

    def test_softmax_cross_entropy_importable(self):
        assert callable(softmax_cross_entropy)

    def test_softmax_cross_entropy_with_integer_labels_importable(self):
        assert callable(softmax_cross_entropy_with_integer_labels)

    def test_l2_loss_is_half_squared_error(self):
        pred   = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 2.5, 3.5])
        assert jnp.allclose(
            l2_loss(pred, target),
            0.5 * squared_error(pred, target),
            atol=1e-6,
        )
