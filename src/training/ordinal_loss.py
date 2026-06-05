"""
training/ordinal_loss.py

CORAL-style ordinal classification loss and utilities.

For K ordinal classes labelled {0, ..., K-1}, the model produces K-1
logits representing P(Y > k) at each cumulative threshold k.  The loss
is the mean sigmoid binary cross-entropy across all K-1 thresholds.

Reference: Cao et al. "Rank-Consistent Ordinal Regression for Neural
Networks with Application to Age Estimation" (2020).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def ordinal_loss(
    logits:    jax.Array,
    labels:    jax.Array,
    n_classes: int,
) -> jax.Array:
    """CORAL ordinal loss over K-1 cumulative thresholds.

    Parameters
    ----------
    logits : jax.Array  shape (B, n_classes - 1)
        Raw logits from OrdinalHead (no sigmoid applied).
    labels : jax.Array  shape (B,)
        Integer class labels in {0, ..., n_classes - 1}.
    n_classes : int
        Total number of ordinal classes K.

    Returns
    -------
    jax.Array  scalar
        Mean BCE over all K-1 thresholds and all samples in the batch.

    Notes
    -----
    For label y, the binary target at threshold k is 1 if y > k else 0.
    BCE is computed in numerically stable log-sum-exp form:
        bce(x, t) = max(x, 0) - x*t + log(1 + exp(-|x|))

    Example
    -------
    >>> logits = jnp.zeros((4, 9))
    >>> labels = jnp.array([0, 3, 7, 9])
    >>> loss = ordinal_loss(logits, labels, n_classes=10)
    >>> loss.shape
    ()
    """
    thresholds = jnp.arange(n_classes - 1)                              # (K-1,)
    targets    = (labels[:, None] > thresholds[None, :]).astype(jnp.float32)  # (B, K-1)
    # Numerically stable sigmoid BCE
    bce = (
        jnp.maximum(logits, 0.0)
        - logits * targets
        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )
    return jnp.mean(bce)


def ordinal_predict(logits: jax.Array) -> jax.Array:
    """Convert ordinal logits to predicted class indices.

    A sample is assigned to class k = sum(sigmoid(logit_j) > 0.5 for j in 0..K-2).
    This counts how many cumulative thresholds the model believes Y exceeds.

    Parameters
    ----------
    logits : jax.Array  shape (B, K-1)

    Returns
    -------
    jax.Array  shape (B,)  int32 class predictions in {0, ..., K-1}.

    Example
    -------
    >>> logits = jnp.array([[10., 10., -10., -10., -10., -10., -10., -10., -10.]])
    >>> ordinal_predict(logits)
    Array([2], dtype=int32)
    """
    return jnp.sum(jax.nn.sigmoid(logits) > 0.5, axis=-1).astype(jnp.int32)


def ordinal_probs(logits: jax.Array) -> jax.Array:
    """Convert ordinal logits to a class probability distribution.

    Derives P(Y = k) from the cumulative probabilities P(Y > k) = sigmoid(logit_k).

    Parameters
    ----------
    logits : jax.Array  shape (B, K-1)

    Returns
    -------
    jax.Array  shape (B, K)
        Non-negative probabilities summing to 1 per sample.

    Notes
    -----
    Construction (K classes, K-1 thresholds):
        P(Y = 0)   = 1 - P(Y > 0)
        P(Y = k)   = P(Y > k-1) - P(Y > k)   for 1 <= k <= K-2
        P(Y = K-1) = P(Y > K-2)

    Due to independent sigmoid outputs the monotonicity P(Y>k) >= P(Y>k+1) is
    not guaranteed at inference; clamp differences to [0, 1] to ensure
    non-negative probabilities.

    Example
    -------
    >>> logits = jnp.zeros((2, 9))
    >>> p = ordinal_probs(logits)
    >>> p.shape
    (2, 10)
    >>> jnp.allclose(p.sum(axis=-1), jnp.ones(2))
    Array(True, dtype=bool)
    """
    cum = jax.nn.sigmoid(logits)                              # (B, K-1)  P(Y>k)
    ones  = jnp.ones((*logits.shape[:-1], 1))
    zeros = jnp.zeros((*logits.shape[:-1], 1))
    augmented = jnp.concatenate([ones, cum, zeros], axis=-1) # (B, K+1)
    probs = augmented[..., :-1] - augmented[..., 1:]         # (B, K)
    return jnp.clip(probs, 0.0, 1.0)
