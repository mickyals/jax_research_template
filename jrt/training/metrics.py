"""
training/metrics.py

Generic evaluation metrics for JAX/Flax training, reusable across
experiments. Experiment `metrics.py` files hold only experiment-specific
glue (label-name maps, `build_metrics_fns` wiring) and should import from
here rather than re-implementing these.

Per-batch metrics share the signature (logits, labels) -> scalar so they
slot directly into the Trainer's metrics_fns dict.

    logits : jax.Array  shape (B, n_classes)   raw model output
    labels : jax.Array  shape (B,)             int32 class indices

Metrics
-------
cross_entropy    Softmax CE — training loss and patience metric
accuracy         Overall top-1 accuracy
binary_accuracy  Thresholded binary accuracy (e.g. class 0 vs. class > 0)
mae_class        Mean absolute error in class units — ordinal distance
"""

from __future__ import annotations

import jax.numpy as jnp
import optax.losses as _optax

from utils.registry import Registry


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
    """Top-1 accuracy across all classes.

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


def binary_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    threshold: int = 1,
) -> jnp.ndarray:
    """Binary accuracy from a thresholded ordinal class index.

    Collapses an ordinal class label into two groups — class < threshold
    vs. class >= threshold — and measures whether the prediction falls in
    the same group as the truth. E.g. with the default threshold=1, this is
    "class 0 vs. class > 0" (presence/absence detection).

    Parameters
    ----------
    logits : jax.Array  (B, n_classes)
    labels : jax.Array  (B, ) int32
    threshold : int
        Class index at/above which a sample is considered "positive"
        (default 1).

    Returns
    -------
    jax.Array  scalar in [0, 1]
    """
    preds = jnp.argmax(logits, axis=-1)
    pred_positive = preds  >= threshold
    true_positive = labels >= threshold
    return jnp.mean(pred_positive == true_positive)


def mae_class(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error in class units.

    Captures ordinal distance: predicting class 5 when truth is class 6
    costs 1, not equal to predicting class 0. Useful for tracking ordinal
    discrimination beyond top-1 accuracy.

    Parameters
    ----------
    logits : jax.Array  (B, n_classes)
    labels : jax.Array  (B, ) int32

    Returns
    -------
    jax.Array  scalar >= 0
    """
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(
        jnp.abs(preds.astype(jnp.float32) - labels.astype(jnp.float32))
    )


# ---------------------------------------------------------------------------
# Per-batch metric registry
# ---------------------------------------------------------------------------
# Maps a name to a *factory* returning a (logits, labels) -> scalar callable,
# matching the shared Registry contract (as used by losses/optimizers). The
# experiment's build_metrics_fns selects which of these to report from the
# trainer.metrics config list; only the training 'loss' itself is hardcoded.

METRICS = Registry("metric")


@METRICS.register("cross_entropy", description="Mean softmax cross-entropy (unweighted comparability anchor)")
def _cross_entropy_metric():
    return cross_entropy


@METRICS.register("accuracy", description="Top-1 accuracy over all classes")
def _accuracy_metric():
    return accuracy


@METRICS.register("binary_accuracy", description="Thresholded detection accuracy (class < thr vs >= thr)")
def _binary_accuracy_metric(threshold: int = 1):
    if threshold == 1:
        return binary_accuracy
    return lambda logits, labels: binary_accuracy(logits, labels, threshold)


@METRICS.register("mae_class", description="Mean absolute class-index error (ordinal distance)")
def _mae_class_metric():
    return mae_class


def get_metric(name: str, **kwargs):
    """Return a configured per-batch metric callable from the METRICS registry."""
    return METRICS.get(name, **kwargs)


def list_metrics() -> list[str]:
    """Sorted names of the registered per-batch metrics."""
    return METRICS.names()
