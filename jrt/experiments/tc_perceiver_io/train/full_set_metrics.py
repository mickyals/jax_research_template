"""
experiments/tc_perceiver_io/train/full_set_metrics.py

FULL-SET metrics for the TC classification task: mAP, detection PR-AUC and
the PR-curve machinery the eval plots are built on. Moved here from jrt
training/metrics.py (2026-07-05 jrt-v2 slim-down: jrt keeps the registry
scaffolding + universal per-batch metrics; this experiment is the only
consumer of the curve machinery).

Contract: full-set metrics take the WHOLE split's accumulated
``(logits, labels)`` as NumPy arrays and return a scalar. They integrate a
ranking/curve over all samples, so they CANNOT be averaged per batch and
never enter the Trainer's per-step metrics_fns — evaluate.py / the
eval-plots callback call them once over the accumulated predictions.

mAP        Macro one-vs-rest average precision over classes — an
           imbalance-robust headline that surfaces rare classes (Cat 4/5)
           that accuracy/QWK hide
pr_auc     Binary detection average precision (TC vs. background) — the
           PR-curve area, the right detection scalar under heavy imbalance
           where ROC/AUC flatters
"""

from __future__ import annotations

import numpy as np

from utils.registry import Registry


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
