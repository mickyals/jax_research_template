"""
training/metrics.py

Generic evaluation metrics for JAX/Flax training, reusable across
experiments. Experiment `metrics.py` files hold only experiment-specific
glue (label-name maps, `build_metrics_fns` wiring) and should import from
here rather than re-implementing these.

Scaffolding, not an encyclopedia (jrt-v2 ruling, 2026-07-05): this module
holds the METRICS registry, the atoms most classification metrics derive
from (``per_class_counts``, ``confusion_counts``), and the universal metrics. Experiment-specific
metrics register INTO the registry from the experiment's own metrics module
(see experiments/tc_perceiver_io/train/metrics.py for the pattern); the
full-set PR-curve machinery (mAP, pr_auc) moved to its only consumer,
experiments/tc_perceiver_io/train/full_set_metrics.py.

PER-BATCH contract: every registered metric is (logits, labels) -> scalar
so it slots directly into the Trainer's metrics_fns dict and is averaged
across batches:

    logits : jax.Array  shape (B, n_classes)   raw model output
    labels : jax.Array  shape (B,)             int32 class indices

Registered metrics
------------------
cross_entropy    Softmax CE — training loss and patience metric
accuracy         Overall top-1 accuracy (batch-averaging is EXACT)
macro_precision  Macro precision over classes predicted in the batch —
                 batch-averaged curves are an APPROXIMATION (ratios of
                 counts don't average); exact split values come from
                 summing per_class_counts over all batches
macro_recall     Macro recall over classes present in the batch (same caveat)
binary_accuracy  Thresholded binary accuracy (e.g. class 0 vs. class > 0)
mae_class        Mean absolute error in class units — ordinal distance
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
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


def per_class_counts(
    logits: jnp.ndarray, labels: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-class TP / FP / FN counts from argmax predictions.

    THE atom most classification metrics derive from (accuracy, precision,
    recall, F1, per-class anything). Counts SUM exactly across batches —
    unlike the ratio metrics built on them — so exact split-level
    precision/recall come from accumulating these and dividing once
    (evaluate.py territory).

    Parameters
    ----------
    logits : jax.Array  (B, n_classes)
    labels : jax.Array  (B,) int32

    Returns
    -------
    (tp, fp, fn) : each jax.Array (n_classes,) float32
    """
    n_classes = logits.shape[-1]
    preds   = jnp.argmax(logits, axis=-1)
    classes = jnp.arange(n_classes)
    pred_1h = (preds[:, None]  == classes).astype(jnp.float32)   # (B, C)
    true_1h = (labels[:, None] == classes).astype(jnp.float32)
    tp = jnp.sum(pred_1h * true_1h, axis=0)
    fp = jnp.sum(pred_1h * (1.0 - true_1h), axis=0)
    fn = jnp.sum((1.0 - pred_1h) * true_1h, axis=0)
    return tp, fp, fn


def confusion_counts(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Full (C, C) confusion-count matrix, rows = true class, cols = predicted.

    Like per_class_counts, counts SUM exactly across batches, so a split
    confusion matrix is the sum of per-batch matrices (how the
    confusion-matrix logging callback accumulates it). per_class_counts is
    recoverable from it (tp = diagonal, fp = col sums - tp, fn = row
    sums - tp).

    Returns jax.Array (n_classes, n_classes) float32.
    """
    n_classes = logits.shape[-1]
    preds   = jnp.argmax(logits, axis=-1)
    classes = jnp.arange(n_classes)
    pred_1h = (preds[:, None]  == classes).astype(jnp.float32)
    true_1h = (labels[:, None] == classes).astype(jnp.float32)
    return true_1h.T @ pred_1h


def _macro_over_valid(numer: jnp.ndarray, denom: jnp.ndarray) -> jnp.ndarray:
    """Mean of numer/denom over classes where denom > 0; 0.0 if none."""
    valid   = denom > 0
    ratios  = jnp.where(valid, numer / jnp.maximum(denom, 1.0), 0.0)
    n_valid = jnp.sum(valid)
    return jnp.where(n_valid > 0, jnp.sum(ratios) / n_valid, 0.0)


def macro_precision(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Macro precision: mean of TP/(TP+FP) over classes PREDICTED in the
    batch (classes never predicted have an undefined precision and are
    skipped). Per-batch values averaged over an epoch are a noisy
    APPROXIMATION of split-level macro precision — fine as a live training
    curve; use accumulated per_class_counts for the exact number.

    Returns jax.Array scalar in [0, 1].
    """
    tp, fp, _ = per_class_counts(logits, labels)
    return _macro_over_valid(tp, tp + fp)


def macro_recall(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Macro recall: mean of TP/(TP+FN) over classes PRESENT in the batch
    truth (absent classes are skipped). Same batch-averaging caveat as
    macro_precision.

    Returns jax.Array scalar in [0, 1].
    """
    tp, _, fn = per_class_counts(logits, labels)
    return _macro_over_valid(tp, tp + fn)


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


@METRICS.register("macro_precision", description="Macro precision over predicted-in-batch classes (per-batch approximation; exact via per_class_counts)")
def _macro_precision_metric():
    return macro_precision


@METRICS.register("macro_recall", description="Macro recall over present-in-batch classes (per-batch approximation; exact via per_class_counts)")
def _macro_recall_metric():
    return macro_recall


@METRICS.register("mae_class", description="Mean absolute class-index error (ordinal distance)")
def _mae_class_metric():
    return mae_class


def get_metric(name: str, **kwargs):
    """Return a configured per-batch metric callable from the METRICS registry."""
    return METRICS.get(name, **kwargs)


def list_metrics() -> list[str]:
    """Sorted names of the registered per-batch metrics."""
    return METRICS.names()
