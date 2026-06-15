"""
datasets/class_weights.py

Derive per-class loss weights from class counts — a small data-distribution
helper. Class imbalance is a property of the data, so this lives with the
datasets layer (counts come from the resolve_splits manifest); the result
feeds the `class_weights` kwarg of `training.losses.cross_entropy_loss`.

The math is the same family across schemes (rarer class -> larger weight); they
differ only in how aggressively they up-weight:

    none               all 1.0 (no reweighting)
    inverse_freq       1 / count_c
    sqrt_inverse_freq  1 / sqrt(count_c)        (gentler, ~halves the spread)
    effective_number   (1 - beta) / (1 - beta**count_c)   (Cui et al. 2019)
    median_freq        median(count) / count_c            (Eigen & Fergus 2015)
                       — the segmentation "freq in images where c is present"
                       collapses to raw counts for one-label-per-sample data.

Compute once from a split's class counts (e.g. the resolve_splits manifest) and
store the realized vector; do not recompute per batch.

Conventions
-----------
* Classes with **zero count** keep weight 1.0 (neutral) and are excluded from
  normalization — so e.g. a background class absent from a TC-only count table
  is left at 1.0 while the present classes are rebalanced among themselves.
* With ``normalize=True`` (default) the present-class weights are scaled to mean
  1.0, keeping the overall loss scale comparable across schemes.
"""

from __future__ import annotations

import numpy as np

SCHEMES = (
    "none",
    "inverse_freq",
    "sqrt_inverse_freq",
    "effective_number",
    "median_freq",
)


def class_weights_from_counts(
    counts:    np.ndarray,
    scheme:    str   = "none",
    beta:      float = 0.999,
    normalize: bool  = True,
) -> np.ndarray:
    """Per-class weights from class counts.

    Parameters
    ----------
    counts : array-like, shape (n_classes,)
        Per-class sample counts (index = class label). Zero-count classes are
        left at weight 1.0.
    scheme : str
        One of SCHEMES.
    beta : float
        Effective-number hyperparameter (only used by ``effective_number``);
        near 1, e.g. 0.99 / 0.999 / 0.9999 — higher = more aggressive.
    normalize : bool
        Scale present-class weights to mean 1.0 (default True).

    Returns
    -------
    np.ndarray, shape (n_classes,), dtype float64
        Weights; zero-count classes are 1.0.

    Raises
    ------
    ValueError
        If ``scheme`` is unknown.
    """
    counts = np.asarray(counts, dtype=np.float64)
    w = np.ones(counts.shape[0], dtype=np.float64)
    if scheme == "none":
        return w

    present = counts > 0
    cp = counts[present]
    if cp.size == 0:
        return w

    if scheme == "inverse_freq":
        wp = 1.0 / cp
    elif scheme == "sqrt_inverse_freq":
        wp = 1.0 / np.sqrt(cp)
    elif scheme == "effective_number":
        wp = (1.0 - beta) / (1.0 - beta ** cp)
    elif scheme == "median_freq":
        wp = np.median(cp) / cp
    else:
        raise ValueError(
            f"Unknown class_weight scheme '{scheme}'. Options: {SCHEMES}"
        )

    if normalize:
        wp = wp / wp.mean()

    w[present] = wp
    return w
