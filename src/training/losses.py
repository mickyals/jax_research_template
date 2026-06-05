"""
training/losses.py

Loss functions for JAX/Flax training.

Design
------
Optax's element-wise losses (squared_error, huber_loss, log_cosh, etc.) are
used as the computational backend wherever they exist. This avoids maintaining
parallel implementations of the same math. Our contribution is:

  1. Scalar reductions (mse, rmse, mae, huber, log_cosh) — jnp.mean over
     the corresponding optax element-wise function.
  2. Masked variants — NaN-safe reductions over valid positions only.
     Critical for IBTrACS secondary targets where many observations are absent.
  3. Re-exports of optax.losses for direct use from a single import path.

Masked variants
---------------
Mask convention: True = valid, False = ignore.
If mask is not supplied it is derived from jnp.isfinite(target), so NaN
targets are automatically excluded without any caller-side handling.
Returns 0.0 when no valid positions exist (avoids NaN from 0/0).

Normalisation note
------------------
Losses assume inputs are already normalised. Scale-invariant variants are
not included — normalise upstream and use plain MSE/MAE.

optax.losses note
-----------------
optax.losses.l2_loss     = 0.5 * (pred - target)^2   (gradient = pred - target)
optax.losses.squared_error = (pred - target)^2        (gradient = 2*(pred - target))

mse() here uses squared_error so the loss value matches standard MSE
convention. For training the choice does not affect convergence — only the
reported loss magnitude differs.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import optax.losses as _optax

# ---------------------------------------------------------------------------
# Re-exports from optax.losses
# Import these directly when you want element-wise losses or classification.
# ---------------------------------------------------------------------------

# Regression — element-wise (apply jnp.mean yourself for a scalar)
l2_loss        = _optax.l2_loss         # 0.5*(pred-target)^2
squared_error  = _optax.squared_error   # (pred-target)^2
huber_loss     = _optax.huber_loss      # element-wise huber
log_cosh_loss  = _optax.log_cosh        # element-wise log-cosh

# Classification — element-wise
sigmoid_binary_cross_entropy          = _optax.sigmoid_binary_cross_entropy
softmax_cross_entropy                 = _optax.softmax_cross_entropy
softmax_cross_entropy_with_integer_labels = _optax.softmax_cross_entropy_with_integer_labels
sigmoid_focal_loss                    = _optax.sigmoid_focal_loss
hinge_loss                            = _optax.hinge_loss

# Similarity / ranking
cosine_distance    = _optax.cosine_distance
cosine_similarity  = _optax.cosine_similarity
ntxent             = _optax.ntxent
triplet_margin_loss = _optax.triplet_margin_loss


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_shapes(pred: jax.Array, target: jax.Array) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, "
            f"got {pred.shape} and {target.shape}."
        )


def _nan_to_zero(x: jax.Array) -> jax.Array:
    """Replace NaN with 0.0 so masked reductions are numerically safe."""
    return jnp.where(jnp.isnan(x), 0.0, x)


def _mask_from_target(target: jax.Array) -> jax.Array:
    """True where target is finite (not NaN, not Inf)."""
    return jnp.isfinite(target)


def _apply_mask(elementwise_loss: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean over valid positions; returns 0.0 when no valid positions exist."""
    n_valid = jnp.sum(mask)
    total   = jnp.sum(jnp.where(mask, elementwise_loss, 0.0))
    return jnp.where(n_valid > 0, total / n_valid, 0.0)


# ---------------------------------------------------------------------------
# Scalar regression losses (backed by optax element-wise functions)
# ---------------------------------------------------------------------------

def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean squared error.

    Backed by optax.losses.squared_error.

    Parameters
    ----------
    pred : jax.Array
    target : jax.Array  same shape as pred

    Returns
    -------
    jax.Array
        Scalar.

    Example
    -------
    >>> mse(jnp.array([1.0, 2.0]), jnp.array([1.5, 2.5]))
    Array(0.25, dtype=float32)
    """
    _check_shapes(pred, target)
    return jnp.mean(_optax.squared_error(pred, target))


def rmse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Root mean squared error.

    Example
    -------
    >>> rmse(jnp.array([1.0, 2.0]), jnp.array([1.5, 2.5]))
    Array(0.5, dtype=float32)
    """
    return jnp.sqrt(mse(pred, target))


def mae(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean absolute error.

    optax does not expose a scalar l1 loss, so this uses jnp.abs directly.

    Example
    -------
    >>> mae(jnp.array([1.0, 2.0]), jnp.array([1.5, 2.5]))
    Array(0.5, dtype=float32)
    """
    _check_shapes(pred, target)
    return jnp.mean(jnp.abs(pred - target))


def huber(
    pred:   jax.Array,
    target: jax.Array,
    delta:  float = 1.0,
) -> jax.Array:
    """Huber loss (smooth L1), backed by optax.losses.huber_loss.

    Quadratic for |error| <= delta, linear beyond. Less sensitive to
    outliers than MSE while remaining differentiable everywhere.

    Example
    -------
    >>> huber(jnp.array([0.0, 2.0]), jnp.array([0.5, 0.0]), delta=1.0)
    Array(0.875, dtype=float32)
    """
    _check_shapes(pred, target)
    return jnp.mean(_optax.huber_loss(pred, target, delta=delta))


def log_cosh(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Log-cosh loss, backed by optax.losses.log_cosh.

    Approximately MSE for small errors, MAE for large. Continuous second
    derivatives everywhere — useful when differentiating through the loss.

    Example
    -------
    >>> log_cosh(jnp.array([0.0, 1.0]), jnp.array([0.5, 0.0]))
    Array(0.2434, dtype=float32)
    """
    _check_shapes(pred, target)
    return jnp.mean(_optax.log_cosh(pred, target))


# ---------------------------------------------------------------------------
# Masked (NaN-safe) losses
# ---------------------------------------------------------------------------

def masked_mse(
    pred:   jax.Array,
    target: jax.Array,
    mask:   Optional[jax.Array] = None,
) -> jax.Array:
    """MSE over valid (non-NaN) positions only.

    Example
    -------
    >>> masked_mse(jnp.array([1., 2., 3.]), jnp.array([1.5, jnp.nan, 3.5]))
    Array(0.25, dtype=float32)
    """
    _check_shapes(pred, target)
    if mask is None:
        mask = _mask_from_target(target)
    target_safe = _nan_to_zero(target)
    return _apply_mask(_optax.squared_error(pred, target_safe), mask)


def masked_rmse(
    pred:   jax.Array,
    target: jax.Array,
    mask:   Optional[jax.Array] = None,
) -> jax.Array:
    """RMSE over valid (non-NaN) positions only."""
    return jnp.sqrt(masked_mse(pred, target, mask))


def masked_mae(
    pred:   jax.Array,
    target: jax.Array,
    mask:   Optional[jax.Array] = None,
) -> jax.Array:
    """MAE over valid (non-NaN) positions only.

    Example
    -------
    >>> masked_mae(jnp.array([1., 2., 3.]), jnp.array([1.5, jnp.nan, 3.5]))
    Array(0.5, dtype=float32)
    """
    _check_shapes(pred, target)
    if mask is None:
        mask = _mask_from_target(target)
    target_safe = _nan_to_zero(target)
    return _apply_mask(jnp.abs(pred - target_safe), mask)


def masked_huber(
    pred:   jax.Array,
    target: jax.Array,
    delta:  float = 1.0,
    mask:   Optional[jax.Array] = None,
) -> jax.Array:
    """Huber loss over valid (non-NaN) positions only.

    Example
    -------
    >>> masked_huber(jnp.array([0., 2., 3.]), jnp.array([0.5, jnp.nan, 3.5]))
    Array(0.1875, dtype=float32)
    """
    _check_shapes(pred, target)
    if mask is None:
        mask = _mask_from_target(target)
    target_safe = _nan_to_zero(target)
    return _apply_mask(_optax.huber_loss(pred, target_safe, delta=delta), mask)


def masked_log_cosh(
    pred:   jax.Array,
    target: jax.Array,
    mask:   Optional[jax.Array] = None,
) -> jax.Array:
    """Log-cosh loss over valid (non-NaN) positions only."""
    _check_shapes(pred, target)
    if mask is None:
        mask = _mask_from_target(target)
    target_safe = _nan_to_zero(target)
    return _apply_mask(_optax.log_cosh(pred, target_safe), mask)
