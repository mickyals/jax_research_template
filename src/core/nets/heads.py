"""
core/nets/heads.py

Output heads for classification and ordinal regression.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


class OrdinalHead(nn.Module):
    """CORAL-style ordinal classification head.

    Produces K-1 logits for cumulative threshold binary decisions.
    No sigmoid is applied here — use with ordinal_loss, ordinal_predict,
    and ordinal_probs from training.ordinal_loss.

    Parameters
    ----------
    n_classes : int
        Total number of ordinal classes K.  Output dimension is K-1.

    Notes
    -----
    Input:  (B, embed_dim)  — arbitrary embed_dim, inferred by Flax.
    Output: (B, n_classes - 1)  raw logits.

    Example
    -------
    >>> head = OrdinalHead(n_classes=10)
    >>> variables = head.init(jax.random.PRNGKey(0), jnp.ones((4, 128)))
    >>> logits = head.apply(variables, jnp.ones((4, 128)))
    >>> logits.shape
    (4, 9)
    """

    n_classes: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array  shape (B, embed_dim)

        Returns
        -------
        jax.Array  shape (B, n_classes - 1)  raw logits (no sigmoid).
        """
        return nn.Dense(self.n_classes - 1)(x)
