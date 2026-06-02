"""
Tests for utils/losses/__init__.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils.losses import (
    mse, rmse, mae, huber, log_cosh,
    masked_mse, masked_rmse, masked_mae, masked_huber,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean():
    """Pred and target with no NaN, error = 0.5 everywhere."""
    pred   = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, 2.5, 3.5])
    return pred, target


@pytest.fixture
def with_nan():
    """Middle value is NaN — only positions 0 and 2 are valid."""
    pred   = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, jnp.nan, 3.5])
    return pred, target


@pytest.fixture
def all_nan():
    pred   = jnp.array([1.0, 2.0, 3.0])
    target = jnp.full(3, jnp.nan)
    return pred, target


# ---------------------------------------------------------------------------
# Pointwise losses — correctness
# ---------------------------------------------------------------------------

class TestMse:
    def test_known_value(self, clean):
        pred, target = clean
        assert abs(float(mse(pred, target)) - 0.25) < 1e-5

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


class TestRmse:
    def test_known_value(self, clean):
        pred, target = clean
        assert abs(float(rmse(pred, target)) - 0.5) < 1e-5

    def test_equals_sqrt_mse(self, clean):
        pred, target = clean
        assert float(rmse(pred, target)) == pytest.approx(
            float(jnp.sqrt(mse(pred, target))), rel=1e-5
        )


class TestMae:
    def test_known_value(self, clean):
        pred, target = clean
        assert abs(float(mae(pred, target)) - 0.5) < 1e-5

    def test_perfect_prediction(self, clean):
        pred, _ = clean
        assert float(mae(pred, pred)) == pytest.approx(0.0, abs=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            mae(jnp.ones(3), jnp.ones(4))


class TestHuber:
    def test_quadratic_regime(self, clean):
        # all errors = 0.5, delta=1.0 -> quadratic: 0.5 * 0.5^2 = 0.125
        pred, target = clean
        assert abs(float(huber(pred, target, delta=1.0)) - 0.125) < 1e-5

    def test_linear_regime(self):
        # error = 2.0, delta = 1.0 -> linear: 1.0 * (2.0 - 0.5) = 1.5
        pred   = jnp.array([0.0])
        target = jnp.array([2.0])
        assert abs(float(huber(pred, target, delta=1.0)) - 1.5) < 1e-5

    def test_boundary(self):
        # error == delta -> quadratic and linear should agree at boundary
        pred   = jnp.array([0.0])
        target = jnp.array([1.0])
        q = 0.5 * 1.0 ** 2        # 0.5
        li = 1.0 * (1.0 - 0.5)   # 0.5
        assert abs(float(huber(pred, target, delta=1.0)) - q) < 1e-5
        assert abs(float(huber(pred, target, delta=1.0)) - li) < 1e-5

    def test_perfect_prediction(self, clean):
        pred, _ = clean
        assert float(huber(pred, pred)) == pytest.approx(0.0, abs=1e-6)


class TestLogCosh:
    def test_zero_residual(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert float(log_cosh(x, x)) == pytest.approx(0.0, abs=1e-6)

    def test_approximately_mse_for_small_errors(self):
        # log_cosh(x) ~ x^2/2 for small x
        pred   = jnp.array([0.01])
        target = jnp.array([0.0])
        expected_approx = 0.5 * 0.01 ** 2
        assert abs(float(log_cosh(pred, target)) - expected_approx) < 1e-4

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
# Masked losses — NaN auto-masking
# ---------------------------------------------------------------------------

class TestMaskedMse:
    def test_nan_excluded(self, with_nan):
        # errors at positions 0 and 2: 0.5^2 = 0.25 each -> mean = 0.25
        pred, target = with_nan
        assert abs(float(masked_mse(pred, target)) - 0.25) < 1e-5

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
        target = jnp.array([1.5, 99.0, 3.5])  # position 1 is noise
        mask   = jnp.array([True, False, True])
        assert abs(float(masked_mse(pred, target, mask=mask)) - 0.25) < 1e-5

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            masked_mse(jnp.ones(3), jnp.ones(4))


class TestMaskedRmse:
    def test_known_value(self, with_nan):
        pred, target = with_nan
        assert abs(float(masked_rmse(pred, target)) - 0.5) < 1e-5

    def test_equals_sqrt_masked_mse(self, with_nan):
        pred, target = with_nan
        assert float(masked_rmse(pred, target)) == pytest.approx(
            float(jnp.sqrt(masked_mse(pred, target))), rel=1e-5
        )


class TestMaskedMae:
    def test_nan_excluded(self, with_nan):
        pred, target = with_nan
        assert abs(float(masked_mae(pred, target)) - 0.5) < 1e-5

    def test_all_nan_returns_zero(self, all_nan):
        pred, target = all_nan
        assert float(masked_mae(pred, target)) == 0.0

    def test_explicit_mask(self):
        pred   = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 99.0, 3.5])
        mask   = jnp.array([True, False, True])
        assert abs(float(masked_mae(pred, target, mask=mask)) - 0.5) < 1e-5


class TestMaskedHuber:
    def test_nan_excluded(self, with_nan):
        # errors = 0.5 at valid positions, delta=1.0 -> quadratic: 0.125
        pred, target = with_nan
        assert abs(float(masked_huber(pred, target, delta=1.0)) - 0.125) < 1e-5

    def test_all_nan_returns_zero(self, all_nan):
        pred, target = all_nan
        assert float(masked_huber(pred, target)) == 0.0

    def test_explicit_mask_linear_regime(self):
        # error = 2.0, delta = 1.0, linear: 1.0 * (2.0 - 0.5) = 1.5
        pred   = jnp.array([0.0, 0.0])
        target = jnp.array([2.0, 99.0])
        mask   = jnp.array([True, False])
        assert abs(float(masked_huber(pred, target, delta=1.0, mask=mask)) - 1.5) < 1e-5


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

class TestGradients:

    @pytest.mark.parametrize("loss_fn", [mse, rmse, mae, huber, log_cosh])
    def test_pointwise_grad_shape(self, loss_fn, clean):
        pred, target = clean
        grad = jax.grad(lambda p: loss_fn(p, target))(pred)
        assert grad.shape == pred.shape
        assert jnp.all(jnp.isfinite(grad))

    @pytest.mark.parametrize("loss_fn", [masked_mse, masked_rmse, masked_mae, masked_huber])
    def test_masked_grad_shape(self, loss_fn, with_nan):
        pred, target = with_nan
        grad = jax.grad(lambda p: loss_fn(p, target))(pred)
        assert grad.shape == pred.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_masked_grad_zero_at_nan_position(self, with_nan):
        """NaN target -> gradient must be exactly zero at that position."""
        pred, target = with_nan  # position 1 is NaN
        grad = jax.grad(lambda p: masked_mse(p, target))(pred)
        assert float(grad[1]) == 0.0

    def test_masked_grad_nonzero_at_valid_positions(self, with_nan):
        pred, target = with_nan
        grad = jax.grad(lambda p: masked_mse(p, target))(pred)
        assert float(grad[0]) != 0.0
        assert float(grad[2]) != 0.0


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJit:

    @pytest.mark.parametrize("loss_fn", [mse, rmse, mae, huber, log_cosh])
    def test_pointwise_jit(self, loss_fn, clean):
        pred, target = clean
        out = jax.jit(loss_fn)(pred, target)
        assert jnp.isfinite(out)

    @pytest.mark.parametrize("loss_fn", [masked_mse, masked_rmse, masked_mae, masked_huber])
    def test_masked_jit(self, loss_fn, with_nan):
        pred, target = with_nan
        out = jax.jit(loss_fn)(pred, target)
        assert jnp.isfinite(out)


# ---------------------------------------------------------------------------
# Removed functions should not be importable
# ---------------------------------------------------------------------------

class TestRemovedFunctions:

    def test_relative_mse_not_exported(self):
        import utils.losses as losses_module
        assert not hasattr(losses_module, "relative_mse")

    def test_multitask_mse_not_exported(self):
        import utils.losses as losses_module
        assert not hasattr(losses_module, "multitask_mse")
