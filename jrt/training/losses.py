"""
training/losses.py

Loss functions for JAX/Flax training.

Design
------
Losses are string-addressable through a shared registry (utils.registry.Registry,
declared at the top of this module) so an experiment selects its objective by
name via trainer.loss + trainer.loss_kwargs. Registered entries:

  * ``mse``           — mean squared error; ``masked`` kwarg gives the NaN-safe
                        variant (mask derived from finite targets).
  * ``cross_entropy`` — softmax CE; composes ``focal_gamma`` (Lin et al. 2017),
                        ``class_weights`` (imbalance), and a squared-EMD
                        regulariser (``emd_lambda``/``emd_omega``/``emd_mu``,
                        Hou et al. 2016).

(A CORAL ordinal loss + K-1-logit head is a planned Tier-3 addition; not yet
present.)

Optax's element-wise losses (squared_error, etc.) are the computational backend
and are re-exported below for direct use. MSE is the canonical regression base;
other reductions (RMSE/MAE/Huber/log-cosh) live in optax and can be wrapped and
registered when a regression experiment needs them.

Mask convention (mse): ``mask=True`` -> valid positions are the finite targets;
returns 0.0 when no valid positions exist (avoids NaN from 0/0). Inputs are
assumed already normalised.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import optax.losses as _optax

from utils.registry import Registry

# ---------------------------------------------------------------------------
# Re-exports from optax.losses
# The element-wise optax functions used by the registered losses, re-exported
# so callers building custom losses have a single import path. optax exposes
# many more (huber_loss, log_cosh, cosine_distance, ntxent, ...) — import them
# from optax.losses directly when needed.
# ---------------------------------------------------------------------------

squared_error                             = _optax.squared_error              # (pred-target)^2  — used by mse
softmax_cross_entropy_with_integer_labels = _optax.softmax_cross_entropy_with_integer_labels  # used by cross_entropy_loss


# ---------------------------------------------------------------------------
# Classification loss registry
# ---------------------------------------------------------------------------
# String-addressable, built on the shared utils.registry.Registry (same machinery
# as the optimizer/scheduler registries). Registered entries are FACTORY
# functions taking config kwargs and returning a (logits, labels) -> scalar
# callable, so trainer.loss + trainer.loss_kwargs select the objective by name.
# The registered factories are defined further down (search @register_loss).

LOSSES = Registry("Loss")
register_loss = LOSSES.register


def get_loss(name: str, **kwargs) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Instantiate a registered classification loss factory by name.

    Case-insensitive; unknown kwargs are dropped with a UserWarning (see
    utils.registry.Registry.get). Raises ValueError for an unknown name.

    Example
    -------
    >>> loss_fn = get_loss("cross_entropy")
    >>> loss_fn = get_loss("cross_entropy", focal_gamma=2.0, class_weights=[1.0]*11)
    """
    return LOSSES.get(name, **kwargs)


def list_losses() -> dict[str, str]:
    """Map of registered loss name -> description."""
    return LOSSES.describe()


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
# Regression loss (mse) with optional masking
# ---------------------------------------------------------------------------

def mse(
    pred:   jax.Array,
    target: jax.Array,
    mask:   Optional[jax.Array | bool] = None,
) -> jax.Array:
    """Mean squared error, optionally masked (NaN-safe).

    MSE is the canonical regression base. Other reductions (RMSE = sqrt(mse),
    MAE, Huber, log-cosh) are available element-wise via the optax re-exports
    at the top of this module and can be wrapped/registered when a regression
    experiment needs them.

    Parameters
    ----------
    pred : jax.Array
    target : jax.Array  same shape as pred
    mask : None | True | jax.Array
        * ``None`` (default): plain mean over all elements.
        * ``True``: NaN-safe — the mask is derived from ``jnp.isfinite(target)``
          so NaN targets are excluded automatically (the only mode expressible
          from a YAML config). Returns 0.0 when no valid positions exist.
        * a boolean array (same shape, True = valid): explicit mask, for direct
          Python calls (an array cannot be passed through a config).

    Returns
    -------
    jax.Array  scalar

    Example
    -------
    >>> mse(jnp.array([1.0, 2.0]), jnp.array([1.5, 2.5]))
    Array(0.25, dtype=float32)
    >>> mse(jnp.array([1., 2., 3.]), jnp.array([1.5, jnp.nan, 3.5]), mask=True)
    Array(0.25, dtype=float32)
    """
    _check_shapes(pred, target)
    if mask is None:
        return jnp.mean(squared_error(pred, target))
    if mask is True:
        mask = _mask_from_target(target)
    target_safe = _nan_to_zero(target)
    return _apply_mask(squared_error(pred, target_safe), mask)


@register_loss(
    "mse",
    description=(
        "Mean squared error. kwarg: masked (bool) — when true, NaN targets "
        "are excluded (mask derived from finite targets)."
    ),
)
def _mse(masked: bool = False) -> Callable[[jax.Array, jax.Array], jax.Array]:
    def loss_fn(pred: jax.Array, target: jax.Array) -> jax.Array:
        return mse(pred, target, mask=True if masked else None)
    return loss_fn


# -------------------------------------------------------------------------
# Cross-entropy (softmax) — composable: focal + class weights + EMD regulariser
# -------------------------------------------------------------------------

def cross_entropy_loss(
    logits:        jax.Array,
    labels:        jax.Array,
    class_weights: Optional[jax.Array] = None,
    focal_gamma:   Optional[float]     = None,
    emd_lambda:    Optional[float]     = None,
    emd_omega:     float               = 1.0,
    emd_mu:        float               = 0.0,
) -> jax.Array:
    """Softmax cross-entropy with composable focal modulation, class weights,
    and a self-guided squared-EMD regulariser (Hou et al. 2016).

    Composable pieces over the per-sample CE term
    ``ce_i = -log softmax(logits_i)[y_i]``:

    * **basic** (defaults): plain mean CE.
    * **focal** (``focal_gamma`` set): multiply each CE term by
      ``(1 - pt_i)**γ`` where ``pt_i = exp(-ce_i)`` — down-weights easy,
      confident-correct samples (Lin et al. 2017).
    * **squared-EMD regulariser** (``emd_lambda`` set): add
      ``λ · Σ_j p_{i,j}^2 (|j - y_i|^ω + μ)`` to each sample, where
      ``p_i = softmax(logits_i)`` and ``|j - y_i|`` is the ordinal ground
      distance to the true class. ω sets distance sensitivity; a negative μ
      makes near-class mass a *reward* (Hou et al. use the EMD term as a
      regulariser on top of CE — the standalone EMD loss collapses to uniform,
      so it is not offered on its own).
    * **class-weighted** (``class_weights`` set): weight each per-sample loss
      (CE + EMD term) by ``class_weights[y_i]`` and take a weighted mean (sum
      of weighted terms / sum of weights), so the loss scale stays comparable
      regardless of weight magnitudes. The weighting method is the caller's
      choice (inverse-freq, effective-number (Cui et al. 2019), median-freq
      (Eigen & Fergus 2015), ...) — this function only consumes the vector.

    Any subset of the kwargs may be combined.

    Parameters
    ----------
    logits : jax.Array  shape (B, n_classes)
    labels : jax.Array  shape (B, )  integer class indices
    class_weights : jax.Array, optional  shape (n_classes,)
        Per-class weight, indexed by class label. None = uniform.
    focal_gamma : float, optional
        Focal focusing parameter γ ≥ 0. None (or 0) = no focal modulation.
    emd_lambda : float, optional
        Weight of the squared-EMD regulariser. None (or 0) = no regulariser.
    emd_omega : float
        Ground-distance power ω (default 1.0). Higher = penalise only far misses.
    emd_mu : float
        Ground-distance bias μ (default 0.0). Negative rewards near-class mass.

    Returns
    -------
    jax.Array  scalar
    """
    ce = softmax_cross_entropy_with_integer_labels(logits, labels)
    if focal_gamma:
        pt = jnp.exp(-ce)
        ce = ((1.0 - pt) ** focal_gamma) * ce

    loss = ce
    if emd_lambda:
        probs = jax.nn.softmax(logits, axis=-1)
        idx   = jnp.arange(logits.shape[-1])
        dist  = jnp.abs(idx[None, :] - labels[:, None]).astype(probs.dtype)  # (B, C)
        emd2  = jnp.sum(probs ** 2 * (dist ** emd_omega + emd_mu), axis=-1)  # (B,)
        loss  = loss + emd_lambda * emd2

    if class_weights is not None:
        w = jnp.asarray(class_weights)[labels]
        return jnp.sum(w * loss) / jnp.sum(w)
    return jnp.mean(loss)


@register_loss(
    "cross_entropy",
    description=(
        "Softmax cross-entropy with integer labels. Optional kwargs compose: "
        "focal_gamma (Lin et al. 2017 focal modulation), class_weights "
        "(length-n_classes per-class weights for imbalance), and a squared-EMD "
        "regulariser (emd_lambda/emd_omega/emd_mu, Hou et al. 2016). Basic / "
        "focal / class-balanced / EMD-regularised and any combination are all "
        "this one loss."
    ),
)
def _cross_entropy_loss(
    class_weights: Optional[list]  = None,
    focal_gamma:   Optional[float] = None,
    emd_lambda:    Optional[float] = None,
    emd_omega:     float           = 1.0,
    emd_mu:        float           = 0.0,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    cw = jnp.asarray(class_weights) if class_weights is not None else None

    def loss_fn(logits: jax.Array, labels: jax.Array) -> jax.Array:
        return cross_entropy_loss(
            logits, labels, class_weights=cw, focal_gamma=focal_gamma,
            emd_lambda=emd_lambda, emd_omega=emd_omega, emd_mu=emd_mu,
        )
    return loss_fn

