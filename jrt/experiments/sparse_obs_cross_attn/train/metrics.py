"""
experiments/sparse_obs_cross_attn/train/metrics.py

Metric functions for the TC classification experiment.

Per-batch metrics share the signature (logits, labels) -> scalar so they
slot directly into the Trainer's metrics_fns dict.

    logits : jax.Array  shape (B, n_classes)   raw model output
    labels : jax.Array  shape (B,)             int32 class indices

Metrics
-------
cross_entropy    Softmax CE — training loss and patience metric
accuracy         Overall 11-class top-1 accuracy
binary_accuracy  TC vs. no-TC (class 0 vs. class > 0) — primary detection signal
mae_class        Mean absolute error in class units — ordinal distance

Full-set metrics (NOT in metrics_fns — too noisy on a single batch; computed
in evaluate.py over accumulated val/test predictions)
-----------------------------------------------------
quadratic_weighted_kappa  Cohen's kappa with quadratic class-distance
                          weights, from a confusion matrix — ordinal
                          agreement, penalises far misses more than near ones
expected_calibration_error  ECE from softmax probabilities — confidence vs.
                          accuracy calibration gap
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np
import optax.losses as _optax

from training.losses import get_loss


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Mean softmax cross-entropy over the batch.

    Parameters
    ----------
    logits : jax.Array  (B, n_classes)
    labels : jax.Array  (B,) int32

    Returns
    -------
    jax.Array  scalar
    """
    return jnp.mean(
        _optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    )


def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Top-1 accuracy across all 11 classes.

    Parameters
    ----------
    logits : jax.Array  (B, n_classes)
    labels : jax.Array  (B, ) int32

    Returns
    -------
    jax.Array  scalar in [0, 1]
    """
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)


def binary_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Binary TC-detection accuracy: class 0 (no storm) vs. class > 0 (any storm).

    The primary detection signal early in training. Batches are 50/50
    TC/background so random chance gives 0.5.

    Parameters
    ----------
    logits : jax.Array  (B, n_classes)
    labels : jax.Array  (B, ) int32

    Returns
    -------
    jax.Array  scalar in [0, 1]
    """
    preds   = jnp.argmax(logits, axis=-1)
    pred_tc = preds   > 0
    true_tc = labels  > 0
    return jnp.mean(pred_tc == true_tc)


def mae_class(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error in class units.

    Captures ordinal distance: predicting class 5 when truth is class 6
    costs 1, not equal to predicting class 0. Useful for tracking intensity
    discrimination beyond binary detection.

    Parameters
    ----------
    logits : jax.Array  (B, n_classes)
    labels : jax.Array  (B,) int32

    Returns
    -------
    jax.Array  scalar >= 0
    """
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(
        jnp.abs(preds.astype(jnp.float32) - labels.astype(jnp.float32))
    )


# ---------------------------------------------------------------------------
# Full-set metrics (computed in evaluate.py over accumulated predictions —
# too noisy/ill-defined on a single training batch to live in metrics_fns)
# ---------------------------------------------------------------------------

def quadratic_weighted_kappa(cm: np.ndarray) -> float:
    """Cohen's kappa with quadratic class-distance weights.

    Operates on a confusion matrix (no recompute from raw preds/labels).
    Rewards predictions that are ordinally close to the truth and penalises
    far misses more heavily than a flat-accuracy metric would.

    Parameters
    ----------
    cm : np.ndarray (n_classes, n_classes)
        ``cm[i, j]`` = count of true class i predicted as j.

    Returns
    -------
    float
        ~[-1, 1]. 1 = perfect agreement, 0 = chance-level agreement
        (given the observed marginals), negative = worse than chance.
        Returns 0.0 for degenerate confusion matrices (empty, or all mass
        in a single class so the expected-agreement denominator is zero).
    """
    cm = np.asarray(cm, dtype=np.float64)
    n  = cm.shape[0]
    total = cm.sum()
    if total == 0 or n <= 1:
        return 0.0

    O = cm / total
    row_marginal = O.sum(axis=1)
    col_marginal = O.sum(axis=0)
    E = np.outer(row_marginal, col_marginal)

    idx = np.arange(n)
    w   = (idx[:, None] - idx[None, :]) ** 2 / (n - 1) ** 2

    num = np.sum(w * O)
    den = np.sum(w * E)
    if den == 0.0:
        return 0.0
    return float(1.0 - num / den)


def expected_calibration_error(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) from softmax probabilities.

    Bins samples by prediction confidence (max softmax prob) into
    ``n_bins`` equal-width bins over [0, 1] and compares each bin's
    accuracy to its mean confidence, weighted by bin occupancy. A
    well-calibrated model has confidence ≈ accuracy in every bin, so
    ECE ≈ 0; a confident-but-wrong model gives a high ECE.

    Parameters
    ----------
    probs : np.ndarray (N, n_classes)
        Softmax class probabilities.
    labels : np.ndarray (N,) int
        True class indices.
    n_bins : int
        Number of equal-width confidence bins (default 15).

    Returns
    -------
    float
        In [0, 1]. Empty bins contribute 0. Returns 0.0 for N == 0.
    """
    probs  = np.asarray(probs)
    labels = np.asarray(labels)
    n = labels.shape[0]
    if n == 0:
        return 0.0

    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    correct     = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        n_b = int(in_bin.sum())
        if n_b == 0:
            continue
        acc_b  = correct[in_bin].mean()
        conf_b = confidences[in_bin].mean()
        ece   += (n_b / n) * abs(acc_b - conf_b)

    return float(ece)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_metrics_fns(
    loss:        str            = 'cross_entropy',
    loss_kwargs: Optional[dict] = None,
) -> dict:
    """Return the metrics_fns dict for the Trainer.

    The first key ('loss') is used as the training loss and the default
    patience metric (val/loss) — resolved from the loss registry in
    training/losses.py via trainer.loss + trainer.loss_kwargs. 'cross_entropy'
    is always reported separately so runs with a different training loss
    (e.g. squared_emd) remain comparable on a common scale.

    Parameters
    ----------
    loss : str
        Name of a registered loss (training/losses.py LOSSES registry),
        e.g. 'cross_entropy' (default) or 'squared_emd'.
    loss_kwargs : dict, optional
        Forwarded to the loss factory (e.g. {'n_classes': 11} for
        'squared_emd').

    Returns
    -------
    dict[str, Callable]
        Keys: 'loss', 'cross_entropy', 'accuracy', 'binary_accuracy', 'mae_class'.
    """
    loss_fn = get_loss(loss, **(loss_kwargs or {}))
    return {
        'loss':            loss_fn,
        'cross_entropy':   cross_entropy,
        'accuracy':        accuracy,
        'binary_accuracy': binary_accuracy,
        'mae_class':       mae_class,
    }
