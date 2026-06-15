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

Full-set metrics (NOT in metrics_fns — too noisy on a single batch; computed
over accumulated val/test predictions)
-----------------------------------------------------
quadratic_weighted_kappa  Cohen's kappa with quadratic class-distance
                          weights, from a confusion matrix — ordinal
                          agreement, penalises far misses more than near ones
expected_calibration_error  ECE from softmax probabilities — occupancy-weighted
                          confidence-vs-accuracy gap
maximum_calibration_error  MCE — the worst single bin's gap (high-stakes,
                          noisier than ECE)

Post-hoc calibration
--------------------
fit_temperature / apply_temperature  temperature scaling (Guo et al. 2017) —
                          fit a single T>0 on a held-out split, apply to the
                          eval split; recalibrates confidence without changing
                          the argmax
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
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
# Full-set metrics (computed over accumulated predictions — too noisy/
# ill-defined on a single training batch to live in metrics_fns)
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
    labels : np.ndarray (N, ) int
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


def maximum_calibration_error(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error (MCE) from softmax probabilities.

    Same equal-width confidence binning as
    :func:`expected_calibration_error`, but reports the **worst** bin's
    |accuracy − confidence| gap rather than the occupancy-weighted average
    (Guo et al. 2017). Useful for high-stakes settings where any single
    badly-calibrated confidence band matters; noisier than ECE since a
    sparsely-populated bin can dominate.

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
        In [0, 1]. The max gap over non-empty bins. Returns 0.0 for N == 0.
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
    mce = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        if not in_bin.any():
            continue
        gap = abs(correct[in_bin].mean() - confidences[in_bin].mean())
        mce = max(mce, float(gap))

    return mce


# ---------------------------------------------------------------------------
# Post-hoc calibration — temperature scaling (Guo et al. 2017)
# ---------------------------------------------------------------------------

def fit_temperature(
    logits:   np.ndarray,
    labels:   np.ndarray,
    u_bounds: tuple[float, float] = (1e-2, 20.0),
    n_iter:   int = 60,
) -> float:
    """Fit a single temperature T>0 by minimising NLL of softmax(logits / T).

    Temperature scaling (Guo et al. 2017, "On Calibration of Modern Neural
    Networks"): a one-parameter post-hoc calibrator. Dividing logits by T
    softens (T>1) or sharpens (T<1) the softmax without changing the argmax,
    so accuracy and any argmax-derived metric are unaffected — only the
    confidence calibration (e.g. ECE) changes. Fit on a held-out split (e.g.
    validation), then apply to the evaluation split.

    The mean NLL is convex in the inverse temperature u = 1/T (log-sum-exp
    composed with a linear map in u, minus a linear term), so this minimises
    over u with an exact ternary search — no learning-rate tuning, fully
    deterministic.

    Parameters
    ----------
    logits : np.ndarray (N, n_classes)
        Raw (pre-softmax) model outputs on the fit split.
    labels : np.ndarray (N, ) int
        True class indices.
    u_bounds : (float, float)
        Search bracket for the inverse temperature u = 1/T. The default
        (0.01, 20.0) covers T in [0.05, 100].
    n_iter : int
        Ternary-search iterations (each shrinks the bracket by 2/3).

    Returns
    -------
    float
        Fitted temperature T. Returns 1.0 for empty input (no-op).
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    n = labels.shape[0]
    if n == 0:
        return 1.0

    rows = np.arange(n)

    def nll(u: float) -> float:
        z = logits * u
        z = z - z.max(axis=-1, keepdims=True)          # stabilise
        logsumexp = np.log(np.exp(z).sum(axis=-1))
        true_logit = z[rows, labels]
        return float(np.mean(logsumexp - true_logit))

    lo, hi = u_bounds
    for _ in range(n_iter):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    u = 0.5 * (lo + hi)
    return float(1.0 / u)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by a fitted temperature: ``logits / temperature``.

    Pairs with :func:`fit_temperature`. Softmax of the result is the
    calibrated probability distribution; the argmax is unchanged.

    Parameters
    ----------
    logits : np.ndarray (N, n_classes)
    temperature : float
        Positive scalar from :func:`fit_temperature`.

    Returns
    -------
    np.ndarray
        Temperature-scaled logits, same shape as the input.
    """
    return np.asarray(logits) / temperature
