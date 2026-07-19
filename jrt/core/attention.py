# core/attention.py
"""
Scaled dot-product attention — ONE primitive.

Cross- and self-attention differ solely by where the query comes from
(Perceiver IO, Jaegle et al. 2021/22: encode and decode are cross-attention,
the processor is self-attention — three uses of one QKV operation). This
module provides that one operation:

    Attention        q from ``x_q``, k/v from ``x_kv``; self-attention is the
                     call ``attn(x, x)``. Owns its q/k/v/out projections
                     (DenseGeneral) so channel counts decouple:

                         x_q  (B, Tq, num_latent_channels) --q_proj--+
                         x_kv (B, Tk, data channels) --k_proj,v_proj-+-> arithmetic at
                                                                     |   num_attn_channels,
                         out  (B, Tq, num_out_channels) <--out_proj--+   split across heads

                     Returns ``(out, probs)`` where ``probs`` are the
                     POST-softmax attention weights (B, num_heads, Tq, Tk) —
                     the ruled observability output. Always returned; callers
                     that don't want them ignore them.

Naming (v3 convention: the feature axis is "channels", ``num_*`` is a count):
    num_attn_channels   channels q/k/v are projected INTO; the attention
                        arithmetic runs here. None -> q input channels.
                        Must be divisible by num_heads.
    num_out_channels    output channels. None -> q input channels (the
                        residual stream width, so blocks add without reshaping).

Masks: build with ``flax.linen.make_attention_mask(q_valid, kv_valid,
dtype=bool)`` / ``flax.linen.make_causal_mask``; True = attend. An additive
float ``bias`` is also accepted (rotary/ALiBi-style extensions).

The pre-LN residual block (LayerNorm -> attention -> +residual -> MLP ->
+residual) is assembled by ``core.nets.transformers``, not baked in here.

"""

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# The one primitive
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head scaled dot-product attention; q from ``x_q``, k/v from ``x_kv``.

    Parameters
    ----------
    num_heads : int
        Head count. head channels = num_attn_channels / num_heads (derived,
        never set).
    num_attn_channels : int, optional
        Channels q/k/v are projected into — where the attention arithmetic
        runs. None -> q input channels. Must be divisible by num_heads
        (checked at call, when the input width is known).
    num_out_channels : int, optional
        Output channels. None -> q input channels.
    dropout_rate : float
        Dropout on the post-softmax attention probabilities. Requires
        ``rngs={'dropout': key}`` when ``train=True`` and rate > 0.
    use_bias : bool
        Bias terms on the q/k/v/out projections. Default True.

    Notes
    -----
    Self-attention is ``attn(x, x)``. Returns ``(out, probs)``:
    ``out`` (B, Tq, num_out_channels); ``probs`` (B, num_heads, Tq, Tk)
    post-softmax attention weights (the observability output — apply nothing,
    they already sum to 1 over Tk on unmasked rows).

    Example
    -------
    >>> attn = Attention(num_heads=4)
    >>> out, probs = attn.apply(vs, latents, data_tokens, mask=mask)
    """
    num_heads: int
    num_attn_channels: Optional[int] = None
    num_out_channels: Optional[int] = None
    dropout_rate: float = 0.0
    use_bias: bool = True

    @nn.compact
    def __call__(
        self,
        x_q: jax.Array,
        x_kv: jax.Array,
        mask: Optional[jax.Array] = None,
        bias: Optional[jax.Array] = None,
        train: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Parameters
        ----------
        x_q : jax.Array
            Query source (B, Tq, q channels).
        x_kv : jax.Array
            Key/value source (B, Tk, kv channels). Pass ``x_q`` for
            self-attention.
        mask : jax.Array, optional
            Boolean, True = attend; broadcastable to
            (B, num_heads, Tq, Tk) — ``flax.linen.make_attention_mask``
            output shape (B, 1, Tq, Tk) broadcasts.
        bias : jax.Array, optional
            Additive float bias on the pre-softmax logits, same
            broadcastability.
        train : bool
            Enables attention-probability dropout.

        Returns
        -------
        (out, probs)
            out (B, Tq, num_out_channels); probs (B, num_heads, Tq, Tk).
        """
        attn_channels = self.num_attn_channels or x_q.shape[-1]
        out_channels = self.num_out_channels or x_q.shape[-1]
        if attn_channels % self.num_heads != 0:
            raise ValueError(
                f"Attention: num_attn_channels={attn_channels} must be "
                f"divisible by num_heads={self.num_heads}."
            )
        head_channels = attn_channels // self.num_heads

        init = nn.initializers.xavier_uniform()
        dense = lambda name: nn.DenseGeneral(
            features=(self.num_heads, head_channels),
            axis=-1, use_bias=self.use_bias, kernel_init=init, name=name,
        )
        q = dense("q_proj")(x_q)     # (B, Tq, H, head_channels)
        k = dense("k_proj")(x_kv)    # (B, Tk, H, head_channels)
        v = dense("v_proj")(x_kv)    # (B, Tk, H, head_channels)

        dropout_rng = (
            self.make_rng("dropout")
            if (train and self.dropout_rate > 0.0) else None
        )
        # Post-softmax attention weights (flax handles the head_channels**-0.5
        # scale, the mask -> -inf bias, and probability dropout).
        probs = nn.dot_product_attention_weights(
            q, k, bias=bias, mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate if train else 0.0,
            deterministic=not train,
        )                                                  # (B, H, Tq, Tk)

        out = jnp.einsum('...hqk,...khd->...qhd', probs, v)  # (B, Tq, H, hd)
        out = nn.DenseGeneral(
            features=out_channels, axis=(-2, -1),
            use_bias=self.use_bias, kernel_init=init, name="out_proj",
        )(out)                                             # (B, Tq, out_channels)

        return out, probs
