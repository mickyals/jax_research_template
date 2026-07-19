# experiments/tc_perceiver_io/train/legacy_attention.py
"""
VENDORED v2 attention — frozen with this experiment, deleted with it.

The v2 MultiHeadAttention / CrossAttention / ATTENTION registry / mask
helpers that used to live in ``core.attention``. Moved here 2026-07-17 when
jrt core was rewritten to the single v3 ``Attention`` primitive: this frozen
line is their only consumer, so they live and die with it. Do not import
from anywhere else.
"""

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn

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
