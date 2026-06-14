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

import warnings
import inspect
from typing import Callable, Optional

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


# ---------------------------------------------------------------------------
# Ordinal classification (CORAL)
# ---------------------------------------------------------------------------

def ordinal_loss(
    logits:    jax.Array,
    labels:    jax.Array,
    n_classes: int,
) -> jax.Array:
    """CORAL ordinal loss over K-1 cumulative thresholds.

    For K ordinal classes labelled {0, ..., K-1}, the model produces K-1
    logits representing P(Y > k). Loss is mean sigmoid BCE across all
    K-1 thresholds, built on sigmoid_binary_cross_entropy.

    Parameters
    ----------
    logits : jax.Array  shape (B, n_classes - 1)
    labels : jax.Array  shape (B,)  integer class labels in {0, ..., K-1}
    n_classes : int

    Returns
    -------
    jax.Array  scalar

    Example
    -------
    >>> ordinal_loss(jnp.zeros((4, 9)), jnp.array([0, 3, 7, 9]), n_classes=10).shape
    ()
    """
    thresholds = jnp.arange(n_classes - 1)
    targets    = (labels[:, None] > thresholds[None, :]).astype(jnp.float32)
    return jnp.mean(_optax.sigmoid_binary_cross_entropy(logits, targets))


def ordinal_predict(logits: jax.Array) -> jax.Array:
    """Convert ordinal logits to predicted class indices.

    Returns k = sum(sigmoid(logit_j) > 0.5 for j in 0..K-2).

    Parameters
    ----------
    logits : jax.Array  shape (B, K-1)

    Returns
    -------
    jax.Array  shape (B,)  int32

    Example
    -------
    >>> ordinal_predict(jnp.array([[10., 10., -10., -10., -10., -10., -10., -10., -10.]]))
    Array([2], dtype=int32)
    """
    return jnp.sum(jax.nn.sigmoid(logits) > 0.5, axis=-1).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Ordinal classification (squared EMD over class CDFs)
# ---------------------------------------------------------------------------

def squared_emd_loss(
    logits:    jax.Array,
    labels:    jax.Array,
    n_classes: int = 11,
) -> jax.Array:
    """Squared Earth Mover's Distance loss for ordinal classes.

    Compares the cumulative distribution of the predicted class
    probabilities against the cumulative distribution of the (one-hot)
    target along the ordinal class axis. Unlike flat cross-entropy,
    predicting a class far from the true one (e.g. Cat5 -> TD) accumulates
    a larger penalty than a near miss (Cat5 -> Cat4), without any model
    change — it operates on the existing K-way softmax output.

    Parameters
    ----------
    logits : jax.Array  shape (B, n_classes)
    labels : jax.Array  shape (B,)  integer class indices in {0, ..., n_classes-1}
    n_classes : int

    Returns
    -------
    jax.Array  scalar

    Example
    -------
    >>> logits = jnp.zeros((4, 11)).at[:, 0].set(100.0)  # confident class 0
    >>> labels = jnp.zeros(4, dtype=jnp.int32)
    >>> float(squared_emd_loss(logits, labels)) < 1e-6
    True
    """
    probs    = jax.nn.softmax(logits, axis=-1)
    target   = jax.nn.one_hot(labels, n_classes)
    cdf_diff = jnp.cumsum(probs, axis=-1) - jnp.cumsum(target, axis=-1)
    return jnp.mean(jnp.sum(cdf_diff ** 2, axis=-1))


def ordinal_probs(logits: jax.Array) -> jax.Array:
    """Convert ordinal logits to a class probability distribution.

    Derives P(Y = k) from cumulative probabilities P(Y > k) = sigmoid(logit_k).
    Differences are clamped to [0, 1] to guard against non-monotone outputs.

    Parameters
    ----------
    logits : jax.Array  shape (B, K-1)

    Returns
    -------
    jax.Array  shape (B, K)

    Example
    -------
    >>> p = ordinal_probs(jnp.zeros((2, 9))); p.shape
    (2, 10)
    """
    cum       = jax.nn.sigmoid(logits)
    ones      = jnp.ones((*logits.shape[:-1], 1))
    zeros     = jnp.zeros((*logits.shape[:-1], 1))
    augmented = jnp.concatenate([ones, cum, zeros], axis=-1)
    return jnp.clip(augmented[..., :-1] - augmented[..., 1:], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Classification loss registry
# ---------------------------------------------------------------------------
#
# Mirrors the optimizer/scheduler registries in training/optimizers.py:
# registered entries are FACTORY functions that take config kwargs and
# return a (logits, labels) -> scalar callable. This lets trainer.loss +
# trainer.loss_kwargs select the training objective by name, on the same
# footing as trainer.optimizer/scheduler.

LOSSES: dict[str, dict] = {}


def register_loss(name: str, description: str = ""):
    """Register a classification loss factory by name.

    Parameters
    ----------
    name : str
        Registry key (case-insensitive).
    description : str, optional
        Short description shown by list_losses().

    Returns
    -------
    callable
        Function decorator. The decorated function must return a
        ``(logits, labels) -> scalar`` callable.

    Raises
    ------
    ValueError
        If a loss with the same name is already registered.

    Example
    -------
    >>> @register_loss("MY_LOSS", description="Custom loss")
    ... def _my_loss(weight: float = 1.0):
    ...     def loss_fn(logits, labels):
    ...         return weight * cross_entropy_loss()(logits, labels)
    ...     return loss_fn
    """
    name = name.upper()

    def decorator(fn):
        if name in LOSSES:
            raise ValueError(f"Loss '{name}' is already registered.")
        LOSSES[name] = {"fn": fn, "description": description}
        return fn

    return decorator


def get_loss(name: str, **kwargs) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Instantiate a registered classification loss.

    Parameters
    ----------
    name : str
        Registry key (case-insensitive).
    **kwargs
        Forwarded to the loss factory. Unknown kwargs trigger a
        UserWarning and are dropped rather than causing a TypeError.

    Returns
    -------
    Callable[[jax.Array, jax.Array], jax.Array]
        ``(logits, labels) -> scalar``.

    Raises
    ------
    ValueError
        If the name is not registered.

    Example
    -------
    >>> loss_fn = get_loss("cross_entropy")
    >>> loss_fn = get_loss("squared_emd", n_classes=11)
    """
    name = name.upper()
    if name not in LOSSES:
        available = ", ".join(sorted(LOSSES.keys()))
        raise ValueError(
            f"Loss '{name}' is not registered. Available: {available}"
        )

    fn = LOSSES[name]["fn"]

    if kwargs:
        sig = inspect.signature(fn)
        valid = {
            k for k, p in sig.parameters.items()
            if p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        unknown = set(kwargs.keys()) - valid
        if unknown:
            warnings.warn(
                f"get_loss('{name}'): unknown kwargs {unknown} will be "
                f"ignored. Valid kwargs: {valid or 'none'}.",
                UserWarning,
                stacklevel=2,
            )
        kwargs = {k: v for k, v in kwargs.items() if k in valid}

    return fn(**kwargs)


def list_losses() -> dict[str, str]:
    """Return all registered loss names and their descriptions.

    Returns
    -------
    dict[str, str]

    Example
    -------
    >>> list_losses()
    {'CROSS_ENTROPY': '...', 'SQUARED_EMD': '...'}
    """
    return {name: info["description"] for name, info in LOSSES.items()}


@register_loss(
    "cross_entropy",
    description="Softmax cross-entropy with integer labels.",
)
def _cross_entropy_loss() -> Callable[[jax.Array, jax.Array], jax.Array]:
    def loss_fn(logits: jax.Array, labels: jax.Array) -> jax.Array:
        return jnp.mean(
            _optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        )
    return loss_fn


@register_loss(
    "squared_emd",
    description=(
        "Squared Earth Mover's Distance over ordinal class CDFs "
        "(ordinal-aware alternative to flat cross-entropy)."
    ),
)
def _squared_emd_loss(n_classes: int = 11) -> Callable[[jax.Array, jax.Array], jax.Array]:
    def loss_fn(logits: jax.Array, labels: jax.Array) -> jax.Array:
        return squared_emd_loss(logits, labels, n_classes=n_classes)
    return loss_fn
