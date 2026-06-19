"""
training/metrics.py

Generic evaluation metrics for JAX/Flax training, reusable across
experiments. Experiment `metrics.py` files hold only experiment-specific
glue (label-name maps, `build_metrics_fns` wiring) and should import from
here rather than re-implementing these.

Two registries, two contracts:

PER-BATCH metrics (``METRICS``) share the signature (logits, labels) -> scalar
so they slot directly into the Trainer's metrics_fns dict and are averaged
across batches:

    logits : jax.Array  shape (B, n_classes)   raw model output
    labels : jax.Array  shape (B,)             int32 class indices

FULL-SET metrics (``FULL_SET_METRICS``) take the WHOLE split's accumulated
``(logits, labels)`` as NumPy arrays and return a scalar. They cannot be
averaged per batch — they integrate a ranking/curve over all samples — so they
are computed once over the accumulated predictions (in evaluate.py / the
eval-plots callback), NOT inside the Trainer's per-step metrics_fns.

Per-batch metrics
-----------------
cross_entropy    Softmax CE — training loss and patience metric
accuracy         Overall top-1 accuracy
binary_accuracy  Thresholded binary accuracy (e.g. class 0 vs. class > 0)
mae_class        Mean absolute error in class units — ordinal distance

Full-set metrics
----------------
mAP              Macro one-vs-rest average precision over classes — an
                 imbalance-robust headline that surfaces rare classes (Cat 4/5)
                 that accuracy/QWK hide
pr_auc           Binary detection average precision (TC vs. background) — the
                 PR-curve area, the right detection scalar under heavy imbalance
                 where ROC/AUC flatters
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


# ---------------------------------------------------------------------------
# Full-set metrics (computed once over the accumulated split, NumPy)
# ---------------------------------------------------------------------------
# Unlike the per-batch metrics above, these integrate a precision-recall curve
# over the whole evaluation set and cannot be averaged across batches. They are
# called by evaluate.py / the eval-plots callback over the accumulated
# (logits, labels). Signature: (logits (N, C) float, labels (N,) int) -> float.


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _pr_points(scores: np.ndarray, y_true: np.ndarray):
    """Cumulative (precision, recall) over samples sorted by descending score.

    The single source of truth for BOTH the average-precision scalar and the
    plotted PR curve — so the figure can never disagree with the number.

    Returns ``(precision, recall, n_pos)`` where precision/recall are length-N
    arrays at successive thresholds and n_pos is the positive count.
    """
    y_true = np.asarray(y_true).astype(bool)
    n_pos  = int(y_true.sum())
    order  = np.argsort(-np.asarray(scores), kind='stable')
    y      = y_true[order]
    tp     = np.cumsum(y)
    fp     = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall    = tp / n_pos if n_pos else np.zeros_like(tp, dtype=float)
    return precision, recall, n_pos


def _average_precision(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Binary average precision = area under the precision-recall curve.

    AP = Σ_k (R_k − R_{k−1}) · P_k over samples sorted by descending score —
    the interpolation-free PR-AUC used by ``sklearn.average_precision_score``.

    Parameters
    ----------
    scores : np.ndarray (N,)  higher = more positive
    y_true : np.ndarray (N,)  bool / {0,1}
    """
    precision, recall, n_pos = _pr_points(scores, y_true)
    if n_pos == 0:
        return 0.0                     # AP undefined with no positives
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def precision_recall_curve(scores: np.ndarray, y_true: np.ndarray) -> dict:
    """Plottable PR curve for a binary problem, with its AP and base rate.

    Same sorted-by-score points whose area ``_average_precision`` integrates,
    plus a leading (precision=1, recall=0) anchor for a clean curve start.

    Returns ``{'precision', 'recall', 'ap', 'base_rate'}`` — base_rate is the
    positive prevalence (the PR no-skill baseline).
    """
    precision, recall, n_pos = _pr_points(scores, y_true)
    n = len(np.asarray(y_true))
    if n_pos == 0:
        return {'precision': np.array([1.0, 0.0]), 'recall': np.array([0.0, 1.0]),
                'ap': 0.0, 'base_rate': 0.0}
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - recall_prev) * precision))
    return {
        'precision': np.concatenate(([1.0], precision)),
        'recall':    np.concatenate(([0.0], recall)),
        'ap':        ap,
        'base_rate': n_pos / n,
    }


def average_precision(logits: np.ndarray, labels: np.ndarray) -> float:
    """Macro one-vs-rest mean average precision (mAP) over the present classes.

    Softmaxes the logits, computes per-class AP (class c probability vs. the
    binary "is class c" target), and averages over classes that actually occur
    in ``labels`` (absent classes are skipped — AP is undefined for them).
    """
    probs  = _softmax_np(np.asarray(logits, dtype=np.float64))
    labels = np.asarray(labels)
    aps = [
        _average_precision(probs[:, c], labels == c)
        for c in range(probs.shape[1]) if np.any(labels == c)
    ]
    return float(np.mean(aps)) if aps else 0.0


def binary_pr_auc(logits: np.ndarray, labels: np.ndarray,
                  threshold: int = 1) -> float:
    """Binary detection average precision (PR-AUC), class < thr vs. >= thr.

    The positive score is the total softmax mass on classes ``>= threshold``
    (default 1 → "any storm" vs. background). PR-AUC is the imbalance-robust
    detection summary; ROC-AUC is misleading when negatives dominate.
    """
    probs = _softmax_np(np.asarray(logits, dtype=np.float64))
    p_pos = probs[:, threshold:].sum(axis=1)
    return _average_precision(p_pos, np.asarray(labels) >= threshold)


def binary_pr_curve(logits: np.ndarray, labels: np.ndarray,
                    threshold: int = 1) -> dict:
    """PR curve for binary TC-vs-background detection (class < thr vs. >= thr).

    Returns the dict from ``precision_recall_curve`` (precision/recall/ap/
    base_rate); ``ap`` equals ``binary_pr_auc`` exactly (shared code path).
    """
    probs = _softmax_np(np.asarray(logits, dtype=np.float64))
    p_pos = probs[:, threshold:].sum(axis=1)
    return precision_recall_curve(p_pos, np.asarray(labels) >= threshold)


def per_class_pr_curves(logits: np.ndarray, labels: np.ndarray) -> dict:
    """One-vs-rest PR curve per PRESENT class.

    Returns ``{class_index: precision_recall_curve(...)}`` for each class that
    occurs in ``labels``; the per-class ``ap`` values are exactly the terms
    averaged into ``mAP`` (average_precision).
    """
    probs  = _softmax_np(np.asarray(logits, dtype=np.float64))
    labels = np.asarray(labels)
    return {
        c: precision_recall_curve(probs[:, c], labels == c)
        for c in range(probs.shape[1]) if np.any(labels == c)
    }


FULL_SET_METRICS = Registry("full_set_metric")


@FULL_SET_METRICS.register(
    "mAP",
    description="Macro one-vs-rest average precision over classes (full-set, "
                "imbalance-robust; surfaces rare classes)")
def _map_metric():
    return average_precision


@FULL_SET_METRICS.register(
    "pr_auc",
    description="Binary TC-vs-background detection average precision / PR-AUC "
                "(full-set)")
def _pr_auc_metric(threshold: int = 1):
    if threshold == 1:
        return binary_pr_auc
    return lambda logits, labels: binary_pr_auc(logits, labels, threshold)


# Default full-set metrics reported by evaluate.py / the eval-plots callback.
DEFAULT_FULL_SET_METRICS: tuple[str, ...] = ('mAP', 'pr_auc')


def compute_full_set_metrics(
    logits, labels, names: tuple[str, ...] = DEFAULT_FULL_SET_METRICS,
) -> dict[str, float]:
    """Evaluate the named full-set metrics over an accumulated split.

    Parameters
    ----------
    logits : array (N, n_classes)
    labels : array (N,) int
    names : tuple of registered full-set metric names.

    Returns
    -------
    dict[str, float]
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    return {n: float(FULL_SET_METRICS.get(n)(logits, labels)) for n in names}


def list_full_set_metrics() -> list[str]:
    """Sorted names of the registered full-set metrics."""
    return FULL_SET_METRICS.names()
