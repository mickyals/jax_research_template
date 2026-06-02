"""
utils/losses/__init__.py

Supervised loss functions for JAX/Flax research.

All functions are:
  - Pure JAX (jnp only) — JIT-compatible and differentiable
  - Shape-preserving in intermediate steps, returning a scalar
  - Accepting optional masks for missing / invalid targets

Masked variants are important when targets contain NaN (e.g. IBTrACS
secondary wind radii where observations are absent). The mask convention
is: True = valid observation, False = ignore. NaN inputs are never
propagated — masked positions are zeroed before any reduction.

Normalisation note
------------------
These losses assume targets and predictions are already normalised before
the loss is computed. Scale-invariant variants (e.g. relative MSE) are
therefore not included — normalise upstream and use plain MSE/MAE.

Loss functions
--------------
Pointwise:
    mse         mean squared error
    rmse        root mean squared error
    mae         mean absolute error
    huber       Huber (smooth L1), threshold delta
    log_cosh    log-cosh (smooth everywhere, bounded 2nd derivative)

Masked (NaN-safe):
    masked_mse      MSE over valid positions only
    masked_rmse     RMSE over valid positions only
    masked_mae      MAE over valid positions only
    masked_huber    Huber over valid positions only
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp


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
    """Return True where target is finite (not NaN, not Inf)."""
    return jnp.isfinite(target)


def _apply_mask(elementwise_loss: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean over valid positions only.

    Zeros masked positions before summing to avoid NaN propagation.
    Returns 0.0 if no valid positions exist (avoids NaN from 0/0).
    """
    n_valid = jnp.sum(mask)
    total   = jnp.sum(jnp.where(mask, elementwise_loss, 0.0))
    return jnp.where(n_valid > 0, total / n_valid, 0.0)


# ---------------------------------------------------------------------------
# Pointwise losses
# ---------------------------------------------------------------------------

def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean squared error.

    Parameters
    ----------
    pred : jax.Array
        Model predictions, any shape.
    target : jax.Array
        Ground truth, same shape as pred.

    Returns
    -------
    jax.Array
        Scalar loss.

    Example
    -------
    >>> mse(jnp.array([1.0, 2.0]), jnp.array([1.5, 2.5]))
    Array(0.25, dtype=float32)
    """
    _check_shapes(pred, target)
    return jnp.mean((pred - target) ** 2)


def rmse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Root mean squared error.

    Parameters
    ----------
    pred : jax.Array
        Model predictions, any shape.
    target : jax.Array
        Ground truth, same shape as pred.

    Returns
    -------
    jax.Array
        Scalar loss.

    Example
    -------
    >>> rmse(jnp.array([1.0, 2.0]), jnp.array([1.5, 2.5]))
    Array(0.5, dtype=float32)
    """
    return jnp.sqrt(mse(pred, target))


def mae(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean absolute error.

    Parameters
    ----------
    pred : jax.Array
        Model predictions, any shape.
    target : jax.Array
        Ground truth, same shape as pred.

    Returns
    -------
    jax.Array
        Scalar loss.

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
    """Huber loss (smooth L1).

    Behaves as MSE for |error| <= delta and as MAE scaled by delta for
    larger errors. Less sensitive to outliers than MSE while remaining
    differentiable everywhere.

    Parameters
    ----------
    pred : jax.Array
        Model predictions, any shape.
    target : jax.Array
        Ground truth, same shape as pred.
    delta : float
        Threshold at which the loss transitions from quadratic to linear.
        Default 1.0.

    Returns
    -------
    jax.Array
        Scalar loss.

    Example
    -------
    >>> huber(jnp.array([0.0, 2.0]), jnp.array([0.5, 0.0]), delta=1.0)
    Array(0.875, dtype=float32)
    """
    _check_shapes(pred, target)
    err       = jnp.abs(pred - target)
    quadratic = 0.5 * err ** 2
    linear    = delta * (err - 0.5 * delta)
    return jnp.mean(jnp.where(err <= delta, quadratic, linear))


def log_cosh(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Log-cosh loss.

    Approximately MSE for small errors and MAE for large errors, with
    continuous second derivatives everywhere. Useful when you need to
    differentiate through the loss itself (e.g. future PINN work).

    log_cosh(x) = log(cosh(x)) ~ x²/2  for small |x|
                               ~ |x| - log(2)  for large |x|

    Parameters
    ----------
    pred : jax.Array
        Model predictions, any shape.
    target : jax.Array
        Ground truth, same shape as pred.

    Returns
    -------
    jax.Array
        Scalar loss.

    Example
    -------
    >>> log_cosh(jnp.array([0.0, 1.0]), jnp.array([0.5, 0.0]))
    Array(0.2434, dtype=float32)
    """
    _check_shapes(pred, target)
    err = pred - target
    # Numerically stable: log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
    return jnp.mean(
        jnp.abs(err) + jnp.log1p(jnp.exp(-2.0 * jnp.abs(err))) - jnp.log(2.0)
    )


# ---------------------------------------------------------------------------
# Masked (NaN-safe) losses
#
# Convention: mask=True  -> include this position
#             mask=False -> ignore this position
#
# If mask is not supplied it is derived from jnp.isfinite(target), so
# NaN targets are automatically excluded without any caller-side handling.
# ---------------------------------------------------------------------------

def masked_mse(
    pred:   jax.Array,
    target: jax.Array,
    mask:   Optional[jax.Array] = None,
) -> jax.Array:
    """MSE over valid (non-NaN) positions only.

    Parameters
    ----------
    pred : jax.Array
        Model predictions, any shape.
    target : jax.Array
        Ground truth, same shape as pred. NaN values are automatically
        excluded when mask is not supplied.
    mask : jax.Array of bool, optional
        True = include, False = exclude. Defaults to isfinite(target).

    Returns
    -------
    jax.Array
        Scalar. Returns 0.0 if no valid positions exist.

    Example
    -------
    >>> pred   = jnp.array([1.0, 2.0, 3.0])
    >>> target = jnp.array([1.5, jnp.nan, 3.5])
    >>> masked_mse(pred, target)
    Array(0.25, dtype=float32)
    """
    _check_shapes(pred, target)
    if mask is None:
        mask = _mask_from_target(target)
    target_safe = _nan_to_zero(target)
    return _apply_mask((pred - target_safe) ** 2, mask)


def masked_rmse(
    pred:   jax.Array,
    target: jax.Array,
    mask:   Optional[jax.Array] = None,
) -> jax.Array:
    """RMSE over valid (non-NaN) positions only.

    Parameters
    ----------
    pred : jax.Array
    target : jax.Array
    mask : jax.Array of bool, optional

    Returns
    -------
    jax.Array
        Scalar.

    Example
    -------
    >>> pred   = jnp.array([1.0, 2.0, 3.0])
    >>> target = jnp.array([1.5, jnp.nan, 3.5])
    >>> masked_rmse(pred, target)
    Array(0.5, dtype=float32)
    """
    return jnp.sqrt(masked_mse(pred, target, mask))


def masked_mae(
    pred:   jax.Array,
    target: jax.Array,
    mask:   Optional[jax.Array] = None,
) -> jax.Array:
    """MAE over valid (non-NaN) positions only.

    Parameters
    ----------
    pred : jax.Array
    target : jax.Array
    mask : jax.Array of bool, optional

    Returns
    -------
    jax.Array
        Scalar.

    Example
    -------
    >>> pred   = jnp.array([1.0, 2.0, 3.0])
    >>> target = jnp.array([1.5, jnp.nan, 3.5])
    >>> masked_mae(pred, target)
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

    Parameters
    ----------
    pred : jax.Array
    target : jax.Array
    delta : float
        Huber threshold. Default 1.0.
    mask : jax.Array of bool, optional

    Returns
    -------
    jax.Array
        Scalar.

    Example
    -------
    >>> pred   = jnp.array([0.0, 2.0, 3.0])
    >>> target = jnp.array([0.5, jnp.nan, 3.5])
    >>> masked_huber(pred, target, delta=1.0)
    Array(0.1875, dtype=float32)
    """
    _check_shapes(pred, target)
    if mask is None:
        mask = _mask_from_target(target)
    target_safe = _nan_to_zero(target)
    err         = jnp.abs(pred - target_safe)
    quadratic   = 0.5 * err ** 2
    linear      = delta * (err - 0.5 * delta)
    elementwise = jnp.where(err <= delta, quadratic, linear)
    return _apply_mask(elementwise, mask)
