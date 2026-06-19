# core/attention.py
"""
Pure attention mechanisms for transformer / Perceiver architectures.

Provides two attention modules and a small registry. Self- and cross-attention
are the *only* attention primitives here (Perceiver IO, Jaegle et al. 2021/22:
encode and decode are cross-attention, the processor is self-attention — three
uses of one QKV operation). The full pre-LN residual block (eqs 4-6: LayerNorm
-> attention -> +residual -> MLP -> +residual) is assembled by the *model* that
composes these, not baked in here.

    MultiHeadAttention   scaled dot-product MHA. context=None -> self-attention;
                         context given -> cross-attention. Owns its QKV
                         projections for clean weight introspection.
    CrossAttention       Q-from-one-source, KV-from-another. A thin, explicit
                         wrapper over MultiHeadAttention for call-site clarity.

Registry (so models can select by name, mirroring activations/initializers):
    get_attention("self_attention",  embed_dim=..., num_heads=...)  -> module
    get_attention("cross_attention", embed_dim=..., num_heads=...)  -> module

Mask utilities (module-level functions):
    make_causal_mask     autoregressive upper-triangular mask
    make_padding_mask    variable-length sequence padding mask

All attention modules share a consistent __call__ signature:
    (x, context=None, mask=None, train=True, return_weights=False)
where context=None means self-attention.

return_weights=True returns (output, scores) where ``scores`` are the
**pre-softmax** attention logits (QKᵀ / √head_dim, plus the mask bias), shape
(B, num_heads, T_q, T_kv). Pre-softmax is the interpretable form for attention
maps — post-softmax weights are very sparse/peaky and hard to read (Perceiver,
Fig. 3). Masked entries appear as a large negative bias; a plotter that knows
the padding can NaN those columns. Apply softmax to recover the true attention
distribution (e.g. for entropy diagnostics). Diagnostics are intended for
val/test only.
"""

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn

from utils.registry import Registry


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len: int) -> jax.Array:
    """Upper-triangular causal mask for autoregressive attention.

    Parameters
    ----------
    seq_len : int
        Sequence length.

    Returns
    -------
    jax.Array
        Boolean mask of shape (seq_len, seq_len). True where attention is
        allowed (lower triangle + diagonal), False where blocked.

    Notes
    -----
    Pass as `mask` to MultiHeadAttention. Boolean masks are converted to
    additive bias (0.0 / -1e9) internally before softmax.

    Example
    -------
    >>> mask = make_causal_mask(4)
    >>> mask.shape
    (4, 4)
    """
    i = jnp.arange(seq_len)
    return i[:, None] >= i[None, :]   # (T, T) -- True where allowed


def make_padding_mask(lengths: jax.Array, max_len: int) -> jax.Array:
    """Boolean padding mask for variable-length sequences.

    Parameters
    ----------
    lengths : jax.Array
        Integer array of shape (B,) with valid token counts per sequence.
    max_len : int
        Padded sequence length.

    Returns
    -------
    jax.Array
        Boolean mask of shape (B, max_len). True for valid positions,
        False for padding.

    Notes
    -----
    To use as an attention mask, broadcast to (B, 1, max_len) before
    passing to MultiHeadAttention. _build_bias handles the (B, T_q, T_kv)
    -> (B, 1, T_q, T_kv) expansion, so passing (B, 1, max_len) with a
    broadcast T_q dimension is the intended usage pattern:

        pad_mask = make_padding_mask(lengths, max_len)   # (B, max_len)
        mask = pad_mask[:, None, :]                      # (B, 1, max_len)
        out = attn(x, mask=mask)

    Example
    -------
    >>> lengths = jnp.array([3, 5, 2])
    >>> mask = make_padding_mask(lengths, max_len=6)
    >>> mask.shape
    (3, 6)
    """
    return jnp.arange(max_len)[None, :] < lengths[:, None]


# ---------------------------------------------------------------------------
# Attention registry
# ---------------------------------------------------------------------------
# Mirrors the activation/initializer registries: a model selects an attention
# primitive by name from config. The two entries are the only attention types
# Perceiver IO needs — self (processor) and cross (encode/decode). Each factory
# returns a configured nn.Module instance.

ATTENTION = Registry("attention")
register_attention = ATTENTION.register
get_attention = ATTENTION.get


def list_attention() -> dict[str, str]:
    """Sorted ``{name: description}`` of all registered attention types."""
    return dict(sorted(ATTENTION.describe().items()))


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention (self or cross).

    Owns QKV projections explicitly, enabling clean attention weight
    return without Flax intermediates machinery. The forward V-weighting
    uses flax.linen.dot_product_attention_weights (softmax + optional
    dropout); the returned diagnostic weights are the pre-softmax logits.

    Parameters
    ----------
    embed_dim : int
        Output and input dimensionality. Must be divisible by num_heads.
    num_heads : int
        Number of attention heads.
    dropout_rate : float
        Attention weight dropout applied during training. Default 0.0.
    use_bias : bool
        Whether QKV and output projections include bias. Default True.
    causal : bool
        If True, automatically applies a causal mask when no explicit
        mask is provided. Default False.

    Notes
    -----
    Input shape: (B, T, embed_dim).

    Supported mask shapes (expanded to (B, num_heads, T_q, T_kv)):
        (T_q, T_kv)               -- shared across batch and heads
        (B, T_q, T_kv)            -- shared across heads
        (B, num_heads, T_q, T_kv) -- fully specified

    Boolean masks: True = attend, False = block.
    Float masks: added directly to logits as additive bias.

    If both causal=True and an explicit mask are provided, the explicit
    mask takes precedence and causal is ignored.

    For cross-attention, pass context as the second positional argument
    or via the `context` keyword. Q is projected from x, K and V from
    context.

    Example
    -------
    >>> attn = MultiHeadAttention(embed_dim=128, num_heads=4)
    >>> out = attn(x, train=False)                          # self-attention
    >>> out = attn(x, context=memory, train=False)          # cross-attention
    >>> out, s = attn(x, train=False, return_weights=True)  # pre-softmax logits
    >>> s.shape  # (B, num_heads, T_q, T_kv)
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
        (B, num_heads, T_q, T_kv).

        Supported input shapes:
            (T_q, T_kv)               -> (1, 1, T_q, T_kv)
            (B, T_q, T_kv)            -> (B, 1, T_q, T_kv)
            (B, num_heads, T_q, T_kv) -> unchanged

        If causal=True and mask is None, generates a causal mask.
        Boolean masks are converted to 0.0 / -1e9.
        Float masks are cast to float32 and used as-is.
        """
        if mask is None and not self.causal:
            return None

        if mask is None:
            # Causal: (T_q, T_kv) bool -> float
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
        """
        Parameters
        ----------
        x : jax.Array
            Query source (B, T_q, embed_dim).
        context : jax.Array, optional
            Key/value source (B, T_kv, embed_dim). None = self-attention.
        mask : jax.Array, optional
            Boolean or float mask. See class docstring for shape conventions.
        train : bool
            Enables attention dropout. Requires rngs={'dropout': key} when
            train=True and dropout_rate > 0.
        return_weights : bool
            If True returns (output, scores) where ``scores`` are the
            PRE-softmax attention logits (QKᵀ / √head_dim + mask bias), shape
            (B, num_heads, T_q, T_kv). Apply softmax to recover the attention
            distribution.

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
            Output (B, T_q, embed_dim), optionally with pre-softmax logits.
        """
        kv_src = x if context is None else context
        q_len  = x.shape[1]
        kv_len = kv_src.shape[1]

        q = self.q_proj(x)        # (B, T_q,  num_heads, head_dim)
        k = self.k_proj(kv_src)   # (B, T_kv, num_heads, head_dim)
        v = self.v_proj(kv_src)   # (B, T_kv, num_heads, head_dim)

        bias = self._build_bias(mask, q_len, kv_len)

        # Attention weights (post-softmax): (B, num_heads, T_q, T_kv)
        weights = nn.dot_product_attention_weights(
            query=q,
            key=k,
            bias=bias,
            dropout_rng=self.make_rng('dropout') if (train and self.dropout_rate > 0) else None,
            dropout_rate=self.dropout_rate if train else 0.0,
            deterministic=not train,
        )

        # Aggregate: einsum over T_kv dimension
        # weights: (B, num_heads, T_q, T_kv)
        # v:       (B, T_kv, num_heads, head_dim)
        out = jnp.einsum('bnij,bjnd->bind', weights, v)  # (B, T_q, num_heads, head_dim)
        out = self.out_proj(out)                          # (B, T_q, embed_dim)

        if return_weights:
            # Pre-softmax logits — the interpretable form for attention maps
            # (post-softmax is too peaky; Perceiver Fig. 3). Includes the mask
            # bias so softmax(scores) recovers the true attention distribution.
            scale  = 1.0 / jnp.sqrt(jnp.asarray(self.head_dim, q.dtype))
            scores = jnp.einsum('bihd,bjhd->bhij', q, k) * scale  # (B, H, T_q, T_kv)
            if bias is not None:
                scores = scores + bias
            return out, scores

        return out


@register_attention(
    "self_attention",
    description="Multi-head self-attention (Q/K/V from one source) — Perceiver processor",
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


# ---------------------------------------------------------------------------
# CrossAttention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Explicit cross-attention: Q from x, K and V from context.

    Functionally equivalent to MultiHeadAttention with context provided,
    but makes the asymmetric Q/KV split structurally explicit at the call
    site. Used for Perceiver encode (latents query the inputs) and decode
    (an output query reads the latents).

    Parameters
    ----------
    embed_dim : int
        Output dimensionality.
    num_heads : int
        Number of attention heads.
    dropout_rate : float
        Default 0.0.
    use_bias : bool
        Default True.

    Notes
    -----
    x and context may differ in sequence length but must share embed_dim.
    Output shape matches x: (B, T_q, embed_dim).

    Example
    -------
    >>> cross = CrossAttention(embed_dim=128, num_heads=4)
    >>> out = cross(x, context=memory, train=False)
    >>> out, s = cross(x, context=memory, train=False, return_weights=True)
    >>> s.shape  # (B, num_heads, T_q, T_kv) pre-softmax logits
    """
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
        """
        Parameters
        ----------
        x : jax.Array
            Query source (B, T_q, embed_dim).
        context : jax.Array
            Key/value source (B, T_kv, embed_dim).
        mask : jax.Array, optional
            Broadcastable to (B, num_heads, T_q, T_kv).
        train : bool
        return_weights : bool
            Returns (output, scores) with pre-softmax logits
            (B, num_heads, T_q, T_kv).
        """
        return self.attn(
            x, context=context, mask=mask, train=train,
            return_weights=return_weights,
        )


@register_attention(
    "cross_attention",
    description="Cross-attention (Q from one source, K/V from another) — Perceiver encode/decode",
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
