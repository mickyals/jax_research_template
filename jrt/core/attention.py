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

----------------------------------------------------------------------------
LEGACY (v2) SECTION at the bottom: MultiHeadAttention / CrossAttention / the
ATTENTION registry / hand-rolled mask helpers. Kept ONLY because the frozen
``experiments/tc_perceiver_io`` line imports ``get_attention``; delete the
whole section when that experiment is deleted. New code composes
``Attention`` directly — no registry.
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


# ===========================================================================
# LEGACY (v2) — kept ONLY for the frozen experiments/tc_perceiver_io line,
# which imports ``get_attention``. Delete this whole section together with
# that experiment. New code composes ``Attention`` above directly.
# ===========================================================================

from utils.registry import Registry


def make_causal_mask(seq_len: int) -> jax.Array:
    """LEGACY. Upper-triangular causal mask; True where attention is allowed.

    New code: ``flax.linen.make_causal_mask``.
    """
    i = jnp.arange(seq_len)
    return i[:, None] >= i[None, :]   # (T, T) -- True where allowed


def make_padding_mask(lengths: jax.Array, max_len: int) -> jax.Array:
    """LEGACY. Boolean padding mask for variable-length sequences.

    New code: ``station_valid = arange(pad_to) < n_stations`` +
    ``flax.linen.make_attention_mask``.
    """
    return jnp.arange(max_len)[None, :] < lengths[:, None]


ATTENTION = Registry("attention")
register_attention = ATTENTION.register
get_attention = ATTENTION.get


def list_attention() -> dict[str, str]:
    """LEGACY. Sorted ``{name: description}`` of registered attention types."""
    return dict(sorted(ATTENTION.describe().items()))


class MultiHeadAttention(nn.Module):
    """LEGACY (v2) multi-head attention: context=None -> self-attention.

    ``return_weights=True`` returns PRE-softmax logits (v2 contract). The v3
    primitive is ``Attention`` above (always returns post-softmax probs).
    """
    embed_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    use_bias: bool = True
    causal: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"MultiHeadAttention: embed_dim={self.embed_dim} must be "
                f"divisible by num_heads={self.num_heads}."
            )

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def setup(self):
        init = nn.initializers.xavier_uniform()
        self.q_proj = nn.DenseGeneral(
            features=(self.num_heads, self.head_dim),
            axis=-1,
            use_bias=self.use_bias,
            kernel_init=init,
        )
        self.k_proj = nn.DenseGeneral(
            features=(self.num_heads, self.head_dim),
            axis=-1,
            use_bias=self.use_bias,
            kernel_init=init,
        )
        self.v_proj = nn.DenseGeneral(
            features=(self.num_heads, self.head_dim),
            axis=-1,
            use_bias=self.use_bias,
            kernel_init=init,
        )
        # Merges (num_heads, head_dim) -> embed_dim in one step
        self.out_proj = nn.DenseGeneral(
            features=self.embed_dim,
            axis=(-2, -1),
            use_bias=self.use_bias,
            kernel_init=init,
        )

    def _build_bias(
        self,
        mask: Optional[jax.Array],
        q_len: int,
        kv_len: int,
    ) -> Optional[jax.Array]:
        """Convert mask to float additive bias broadcastable to
        (B, num_heads, T_q, T_kv)."""
        if mask is None and not self.causal:
            return None

        if mask is None:
            raw = make_causal_mask(q_len)
            bias = jnp.where(raw, 0.0, -1e9).astype(jnp.float32)
        elif mask.dtype == jnp.bool_:
            bias = jnp.where(mask, 0.0, -1e9).astype(jnp.float32)
        else:
            bias = mask.astype(jnp.float32)

        # Expand to (B, num_heads, T_q, T_kv)
        if bias.ndim == 2:
            bias = bias[None, None]    # (1, 1, T_q, T_kv)
        elif bias.ndim == 3:
            bias = bias[:, None]       # (B, 1, T_q, T_kv)
        # ndim == 4: already correct

        return bias

    def __call__(
        self,
        x: jax.Array,
        context: Optional[jax.Array] = None,
        mask: Optional[jax.Array] = None,
        train: bool = True,
        return_weights: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        kv_src = x if context is None else context
        q_len  = x.shape[1]
        kv_len = kv_src.shape[1]

        q = self.q_proj(x)        # (B, T_q,  num_heads, head_dim)
        k = self.k_proj(kv_src)   # (B, T_kv, num_heads, head_dim)
        v = self.v_proj(kv_src)   # (B, T_kv, num_heads, head_dim)

        bias = self._build_bias(mask, q_len, kv_len)

        weights = nn.dot_product_attention_weights(
            query=q,
            key=k,
            bias=bias,
            dropout_rng=self.make_rng('dropout') if (train and self.dropout_rate > 0) else None,
            dropout_rate=self.dropout_rate if train else 0.0,
            deterministic=not train,
        )

        out = jnp.einsum('bnij,bjnd->bind', weights, v)  # (B, T_q, num_heads, head_dim)
        out = self.out_proj(out)                          # (B, T_q, embed_dim)

        if return_weights:
            scale  = 1.0 / jnp.sqrt(jnp.asarray(self.head_dim, q.dtype))
            scores = jnp.einsum('bihd,bjhd->bhij', q, k) * scale  # (B, H, T_q, T_kv)
            if bias is not None:
                scores = scores + bias
            return out, scores

        return out


@register_attention(
    "self_attention",
    description="LEGACY v2 multi-head self-attention — kept for tc_perceiver_io",
)
def _self_attention(
    embed_dim: int,
    num_heads: int,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
) -> MultiHeadAttention:
    return MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
    )


class CrossAttention(nn.Module):
    """LEGACY (v2) explicit cross-attention wrapper over MultiHeadAttention."""
    embed_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    use_bias: bool = True

    def setup(self):
        self.attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            causal=False,
        )

    def __call__(
        self,
        x: jax.Array,
        context: jax.Array,
        mask: Optional[jax.Array] = None,
        train: bool = True,
        return_weights: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        return self.attn(
            x, context=context, mask=mask, train=train,
            return_weights=return_weights,
        )


@register_attention(
    "cross_attention",
    description="LEGACY v2 cross-attention — kept for tc_perceiver_io",
)
def _cross_attention(
    embed_dim: int,
    num_heads: int,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
) -> CrossAttention:
    return CrossAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
    )
