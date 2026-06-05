"""
experiments/sparse_obs_cross_attn/metrics.py

Metric functions for the TC classification experiment.

All functions share the signature (logits, labels) -> scalar so they
slot directly into the Trainer's metrics_fns dict.

    logits : jax.Array  shape (B, n_classes)   raw model output
    labels : jax.Array  shape (B,)             int32 class indices

Metrics
-------
cross_entropy    Softmax CE — training loss and patience metric
accuracy         Overall 11-class top-1 accuracy
binary_accuracy  TC vs. no-TC (class 0 vs. class > 0) — primary detection signal
mae_class        Mean absolute error in class units — ordinal distance
"""

from __future__ import annotations

import jax.numpy as jnp
import optax.losses as _optax


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
# Factory
# ---------------------------------------------------------------------------

def build_metrics_fns() -> dict:
    """Return the metrics_fns dict for the Trainer.

    The first key (cross_entropy) is used as the training loss and the
    default patience metric (val/cross_entropy).

    Returns
    -------
    dict[str, Callable]
        Keys: 'cross_entropy', 'accuracy', 'binary_accuracy', 'mae_class'.
    """
    return {
        'cross_entropy':   cross_entropy,
        'accuracy':        accuracy,
        'binary_accuracy': binary_accuracy,
        'mae_class':       mae_class,
    }
