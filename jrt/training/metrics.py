"""
training/metrics.py

Generic evaluation metrics for JAX/Flax training, reusable across
experiments. Experiment `metrics.py` files hold only experiment-specific
glue (label-name maps, `build_metrics_fns` wiring) and should import from
here rather than re-implementing these.

Scaffolding, not an encyclopedia (jrt-v2 ruling, 2026-07-05): this module
holds the METRICS registry, the confusion-matrix atom + accumulator most
classification metrics derive from (``confusion_counts``, ``update_cm``,
``compute_final_metrics``), and the universal metrics. Experiment-specific
metrics register INTO the registry from the experiment's own metrics module
(see experiments/tc_perceiver_io/train/metrics.py for the pattern); the
full-set PR-curve machinery (mAP, pr_auc) moved to its only consumer,
experiments/tc_perceiver_io/train/full_set_metrics.py.

PER-BATCH contract: every registered metric is (logits, labels) -> scalar
so it slots directly into the Trainer's metrics_fns dict and is averaged
across batches:

    logits : jax.Array  shape (B, n_classes)   raw model output
    labels : jax.Array  shape (B,)             int32 class indices

Only LINEAR metrics — those whose batch average IS the split value — are
registered (PR #5 ruling). Ratio-of-counts metrics (macro precision/recall,
per-class anything) do NOT average across batches; they are derived exactly
from an accumulated confusion matrix instead: a plain ``(C, C)`` array is
the state, ``update_cm(cm, logits, labels)`` folds each batch in, and
``compute_final_metrics(cm)`` derives the whole family (TP/TN/FP/FN, exact
macro precision/recall, per-class + OVA accuracy, the pairwise accuracy
matrix) once at the end. Consumers: the confusion-matrix logging callback
(experiment log.py) and evaluate.py.

Registered metrics
------------------
cross_entropy    Softmax CE — training loss and patience metric
accuracy         Overall top-1 accuracy (batch-averaging is EXACT)
binary_accuracy  Thresholded binary accuracy (e.g. class 0 vs. class > 0)
mae_class        Mean absolute error in class units — ordinal distance
"""

from __future__ import annotations

import jax
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


def confusion_counts(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Full (C, C) confusion-count matrix, rows = true class, cols = predicted.

    THE atom classification metrics derive from: counts SUM exactly across
    batches, so a split confusion matrix is the sum of per-batch matrices
    (``update_cm`` is that fold), and per-class TP/FP/FN are its slices
    (tp = diagonal, fp = col sums - tp, fn = row sums - tp — see
    ``compute_final_metrics``).

    Returns jax.Array (n_classes, n_classes) float32.
    """
    n_classes = logits.shape[-1]
    preds = jnp.argmax(logits, axis=-1)
    idx   = labels.astype(jnp.int32) * n_classes + preds.astype(jnp.int32)
    return (jnp.bincount(idx, length=n_classes * n_classes)
               .reshape(n_classes, n_classes)
               .astype(jnp.float32))


@jax.jit
def update_cm(
    cm: jnp.ndarray, logits: jnp.ndarray, labels: jnp.ndarray,
) -> jnp.ndarray:
    """Fold one batch into an accumulated confusion matrix.

    The plain ``(C, C)`` array IS the accumulator state — initialise with
    ``jnp.zeros((n_classes, n_classes))``, thread through the batch stream,
    then hand the result to ``compute_final_metrics`` (and/or the confusion
    figure). Accumulation is exact, unlike averaging ratio metrics per batch.

    Parameters
    ----------
    cm : jax.Array  (C, C)   running counts (any numeric dtype; preserved)
    logits : jax.Array  (B, C)
    labels : jax.Array  (B,) int32

    Returns
    -------
    jax.Array  (C, C)  updated counts, same dtype as ``cm``.
    """
    return cm + confusion_counts(logits, labels).astype(cm.dtype)


def _macro_over_valid(numer: jnp.ndarray, denom: jnp.ndarray) -> jnp.ndarray:
    """Mean of numer/denom over classes where denom > 0; 0.0 if none."""
    valid   = denom > 0
    ratios  = jnp.where(valid, numer / jnp.maximum(denom, 1.0), 0.0)
    n_valid = jnp.sum(valid)
    return jnp.where(n_valid > 0, jnp.sum(ratios) / n_valid, 0.0)


def compute_final_metrics(cm: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Derive the exact metric family from an accumulated confusion matrix.

    End-of-stream counterpart of ``update_cm``: one call at callback /
    evaluation time replaces every per-batch ratio approximation. Classes
    with a zero denominator (never predicted for precision, no support for
    recall) hold 0.0 in the per-class arrays and are EXCLUDED from the
    macro means.

    Parameters
    ----------
    cm : jax.Array  (C, C)  accumulated counts, rows = true, cols = predicted.

    Returns
    -------
    dict[str, jax.Array]
        tp, tn, fp, fn        (C,)  one-vs-all count primitives
        support               (C,)  true samples per class (row sums)
        precision, recall     (C,)  exact per-class ratios; ``recall`` IS the
                                    per-class accuracy (diagonal / row sum)
        macro_precision, macro_recall   scalars over valid classes
        accuracy              scalar  trace / total
        ova_accuracy          (C,)  one-vs-all binary accuracy
                                    (tp_k + tn_k) / total
        pairwise_accuracy     (C, C) restricted two-class accuracy: of the
                                    samples of classes i and j predicted as
                                    i or j, the fraction predicted right —
                                    (cm_ii + cm_jj) / (cm_ii + cm_jj +
                                    cm_ij + cm_ji). Off-diagonals are the
                                    informative entries (how separable the
                                    pair is); the diagonal is trivially 1.
                                    Pairs with no such samples give 0.0.
    """
    cm    = jnp.asarray(cm, jnp.float32)
    tp    = jnp.diag(cm)
    fn    = jnp.sum(cm, axis=1) - tp
    fp    = jnp.sum(cm, axis=0) - tp
    total = jnp.sum(cm)
    tn    = total - tp - fp - fn

    precision = jnp.where(tp + fp > 0, tp / jnp.maximum(tp + fp, 1.0), 0.0)
    recall    = jnp.where(tp + fn > 0, tp / jnp.maximum(tp + fn, 1.0), 0.0)

    pair_correct  = tp[:, None] + tp[None, :]
    pair_confused = cm + cm.T - jnp.diag(2.0 * tp)   # zero self-confusion
    pair_total    = pair_correct + pair_confused
    pairwise = jnp.where(pair_total > 0,
                         pair_correct / jnp.maximum(pair_total, 1.0), 0.0)

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'support': tp + fn,
        'precision': precision,
        'recall': recall,
        'macro_precision': _macro_over_valid(tp, tp + fp),
        'macro_recall':    _macro_over_valid(tp, tp + fn),
        'accuracy': jnp.where(total > 0, jnp.sum(tp) / jnp.maximum(total, 1.0), 0.0),
        'ova_accuracy': jnp.where(total > 0,
                                  (tp + tn) / jnp.maximum(total, 1.0), 0.0),
        'pairwise_accuracy': pairwise,
    }


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
