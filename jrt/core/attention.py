# core/nets/attention.py
"""
Pure attention mechanisms for transformer architectures.

Provides three attention modules and supporting utilities:

    MultiHeadAttention      -- scaled dot-product MHA (self or cross), owns
                               QKV projections for clean weight introspection
    SwinWindowAttention     -- local window attention with relative position bias
                               and optional cyclic shift for Swin Transformers
    CrossAttention          -- explicit Q-from-one-source, KV-from-another module
                               for encoder-decoder architectures

Mask utilities (module-level functions):
    make_causal_mask        -- autoregressive upper-triangular mask
    make_padding_mask       -- variable-length sequence padding mask
    make_swin_shift_mask    -- cross-region additive bias for shifted window attention

All attention modules share a consistent __call__ signature:
    (x, context=None, mask=None, train=True, return_weights=False)

where context=None means self-attention. SwinWindowAttention ignores context.

return_weights=True returns (output, weights) where weights are raw
per-head attention probabilities:
    MultiHeadAttention  -> (B, num_heads, T_q, T_kv)
    CrossAttention      -> (B, num_heads, T_q, T_kv)
    SwinWindowAttention -> (B*num_windows, num_heads, M^2, M^2)

Weights are computed without dropout regardless of train flag, so they
reflect the full attention distribution for diagnostic purposes.
"""

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


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


def make_swin_shift_mask(
    window_size: int,
    shift_size: int,
    H: int,
    W: int,
) -> jax.Array:
    """Additive attention bias mask for shifted window attention.

    After a cyclic shift by (shift_size, shift_size), windows may contain
    tokens from non-adjacent spatial regions. This mask blocks attention
    between those regions within each shifted window, restoring locality.

    Parameters
    ----------
    window_size : int
        Window size M. Assumes square windows.
    shift_size : int
        Cyclic shift amount. Typically window_size // 2. 0 returns a zero
        mask (no blocking needed).
    H : int
        Feature map height in tokens. Must be divisible by window_size.
    W : int
        Feature map width in tokens. Must be divisible by window_size.

    Returns
    -------
    jax.Array
        Float additive bias of shape (num_windows, M^2, M^2) with values
        0.0 (attend) or -1e9 (block). Added directly to attention logits.

    Notes
    -----
    num_windows = (H // window_size) * (W // window_size).

    Computed with numpy then converted to jax -- this is a static quantity
    for fixed H, W, window_size, shift_size. Call once and reuse.

    Example
    -------
    >>> mask = make_swin_shift_mask(window_size=4, shift_size=2, H=16, W=16)
    >>> mask.shape
    (16, 16, 16)
    """
    if shift_size == 0:
        num_windows = (H // window_size) * (W // window_size)
        return jnp.zeros((num_windows, window_size ** 2, window_size ** 2))

    # Build region label map (H, W)
    img_mask = np.zeros((H, W), dtype=np.int32)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[h, w] = cnt
            cnt += 1

    # Partition into windows: (num_windows, M^2)
    num_h = H // window_size
    num_w = W // window_size
    mask_windows = img_mask.reshape(num_h, window_size, num_w, window_size)
    mask_windows = mask_windows.transpose(0, 2, 1, 3)           # (nH, nW, M, M)
    mask_windows = mask_windows.reshape(-1, window_size ** 2)   # (num_windows, M^2)

    # Diff mask: 0 where same region, -1e9 elsewhere
    attn_mask = mask_windows[:, :, None] - mask_windows[:, None, :]
    attn_mask = np.where(attn_mask != 0, -1e9, 0.0).astype(np.float32)
    return jnp.array(attn_mask)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention (self or cross).

    Owns QKV projections explicitly, enabling clean attention weight
    return without Flax intermediates machinery. Uses
    flax.linen.dot_product_attention_weights for the core computation.

    Parameters
    ----------
    embed_dim : int
        Output and input dimensionality. Must be divisible by num_heads.
    num_heads : int
        Number of attention heads.
    dropout_rate : float
        Attention weight dropout applied during training. Default 0.0.
        Weights returned via return_weights=True are computed without
        dropout regardless of train flag.
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
    >>> out, w = attn(x, train=False, return_weights=True)  # with weights
    >>> w.shape  # (B, num_heads, T, T)
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
            If True returns (output, weights) where weights has shape
            (B, num_heads, T_q, T_kv). Weights are without dropout.

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
            Output (B, T_q, embed_dim), optionally with attention weights.
        """
        kv_src = x if context is None else context
        q_len  = x.shape[1]
        kv_len = kv_src.shape[1]

        q = self.q_proj(x)        # (B, T_q,  num_heads, head_dim)
        k = self.k_proj(kv_src)   # (B, T_kv, num_heads, head_dim)
        v = self.v_proj(kv_src)   # (B, T_kv, num_heads, head_dim)

        bias = self._build_bias(mask, q_len, kv_len)

        # Attention weights: (B, num_heads, T_q, T_kv)
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
            # Recompute without dropout so returned weights are the full
            # attention distribution regardless of train flag
            if train and self.dropout_rate > 0:
                clean_weights = nn.dot_product_attention_weights(
                    query=q,
                    key=k,
                    bias=bias,
                    dropout_rate=0.0,
                    deterministic=True,
                )
            else:
                clean_weights = weights
            return out, clean_weights

        return out


# ---------------------------------------------------------------------------
# CrossAttention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Explicit cross-attention: Q from x, K and V from context.

    Functionally equivalent to MultiHeadAttention with context provided,
    but makes the asymmetric Q/KV split structurally explicit at the call
    site. Preferred in encoder-decoder blocks.

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
    >>> out, w = cross(x, context=memory, train=False, return_weights=True)
    >>> w.shape  # (B, num_heads, T_q, T_kv)
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
            Returns (output, weights) with weights (B, num_heads, T_q, T_kv).
        """
        return self.attn(
            x, context=context, mask=mask, train=train,
            return_weights=return_weights,
        )


# ---------------------------------------------------------------------------
# SwinWindowAttention
# ---------------------------------------------------------------------------

class SwinWindowAttention(nn.Module):
    """Local window multi-head self-attention with relative position bias.

    Implements W-MSA (window attention) and SW-MSA (shifted window attention)
    from Swin Transformer (Liu et al. 2021, arxiv.org/abs/2103.14030).

    Input is a spatial feature map (B, H, W, C). It is partitioned into
    non-overlapping windows of size (window_size, window_size), attention is
    computed independently within each window, then windows are merged back.

    For shifted window attention (shift_size > 0), the feature map is first
    cyclically shifted by (shift_size, shift_size), and an additive mask is
    applied to block attention between non-adjacent spatial regions. After
    attention, the shift is reversed.

    Parameters
    ----------
    embed_dim : int
        Channel dimension C. Must equal input C in (B, H, W, C).
    num_heads : int
        Number of attention heads. embed_dim must be divisible by num_heads.
    window_size : int
        Side length M of each square attention window. H and W must be
        divisible by window_size. Default 7.
    shift_size : int
        Cyclic shift amount for SW-MSA. Typically window_size // 2.
        0 = standard W-MSA with no shift. Default 0.
    dropout_rate : float
        Attention weight dropout. Default 0.0.
        Requires rngs={'dropout': key} when train=True and dropout_rate > 0.
    use_bias : bool
        QKV and output projection bias. Default True.

    Notes
    -----
    Relative position bias table shape: (2M-1, 2M-1, num_heads).
    Position index matrix shape:        (M^2, M^2) -- static, computed in setup().

    Output shape matches input: (B, H, W, C).

    SwinWindowAttention is self-attention only. context is not supported.

    return_weights=True returns raw per-window weights of shape
    (B*num_windows, num_heads, M^2, M^2). Windows are not merged since
    aggregation across windows is spatially ambiguous.

    The mask argument accepts an additional float additive bias of shape
    (num_windows, M^2, M^2). It is not a boolean mask -- use 0.0 / -1e9
    values. Added on top of the internal shift mask and relative position
    bias.

    Example
    -------
    >>> # W-MSA (no shift)
    >>> attn = SwinWindowAttention(embed_dim=96, num_heads=3, window_size=7)
    >>> # SW-MSA (with shift)
    >>> attn = SwinWindowAttention(embed_dim=96, num_heads=3, window_size=7,
    ...                            shift_size=3)
    >>> out = attn(x, train=False)
    >>> out, w = attn(x, train=False, return_weights=True)
    >>> w.shape  # (B*num_windows, num_heads, M^2, M^2)
    """
    embed_dim:    int
    num_heads:    int
    window_size:  int   = 7
    shift_size:   int   = 0
    dropout_rate: float = 0.0
    use_bias:     bool  = True

    def __post_init__(self):
        super().__post_init__()
        # Validate hyperparameters at construction time -- these are always
        # concrete Python values and never require input shapes
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"SwinWindowAttention: embed_dim={self.embed_dim} must be "
                f"divisible by num_heads={self.num_heads}."
            )
        if self.shift_size >= self.window_size:
            raise ValueError(
                f"SwinWindowAttention: shift_size={self.shift_size} must be "
                f"less than window_size={self.window_size}."
            )

    def setup(self):
        M        = self.window_size
        head_dim = self.embed_dim // self.num_heads
        init     = nn.initializers.xavier_uniform()

        self.q_proj = nn.DenseGeneral(
            features=(self.num_heads, head_dim), axis=-1,
            use_bias=self.use_bias, kernel_init=init,
        )
        self.k_proj = nn.DenseGeneral(
            features=(self.num_heads, head_dim), axis=-1,
            use_bias=self.use_bias, kernel_init=init,
        )
        self.v_proj = nn.DenseGeneral(
            features=(self.num_heads, head_dim), axis=-1,
            use_bias=self.use_bias, kernel_init=init,
        )
        self.out_proj = nn.DenseGeneral(
            features=self.embed_dim, axis=(-2, -1),
            use_bias=self.use_bias, kernel_init=init,
        )

        # Learnable relative position bias table: (2M-1, 2M-1, num_heads)
        self.rel_pos_bias_table = self.param(
            'rel_pos_bias_table',
            nn.initializers.truncated_normal(stddev=0.02),
            (2 * M - 1, 2 * M - 1, self.num_heads),
        )

        # Static relative position index (M^2, M^2) -- computed once
        coords_h = np.arange(M)
        coords_w = np.arange(M)
        grid    = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, M, M)
        flat    = grid.reshape(2, -1)                                        # (2, M^2)
        rel     = flat[:, :, None] - flat[:, None, :]                       # (2, M^2, M^2)
        rel[0] += M - 1
        rel[1] += M - 1
        rel_idx = rel[0] * (2 * M - 1) + rel[1]                            # (M^2, M^2)
        self.rel_pos_index = jax.device_put(jnp.array(rel_idx))

    def _relative_position_bias(self) -> jax.Array:
        """Gather bias from table via precomputed flat indices.

        Returns
        -------
        jax.Array
            Shape (num_heads, M^2, M^2).
        """
        M2         = self.window_size ** 2
        table_flat = self.rel_pos_bias_table.reshape(-1, self.num_heads)
        bias       = table_flat[self.rel_pos_index.reshape(-1)]  # (M^4, num_heads)
        bias       = bias.reshape(M2, M2, self.num_heads)
        return bias.transpose(2, 0, 1)                           # (num_heads, M^2, M^2)

    def _partition_windows(self, x: jax.Array) -> jax.Array:
        """(B, H, W, C) -> (B*nW, M, M, C)."""
        B, H, W, C = x.shape
        M = self.window_size
        x = x.reshape(B, H // M, M, W // M, M, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(-1, M, M, C)

    def _merge_windows(self, x: jax.Array, B: int, H: int, W: int) -> jax.Array:
        """(B*nW, M, M, C) -> (B, H, W, C)."""
        M  = self.window_size
        C  = x.shape[-1]
        nH = H // M
        nW = W // M
        x  = x.reshape(B, nH, nW, M, M, C)
        x  = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(B, H, W, C)

    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        train: bool = True,
        return_weights: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """
        Parameters
        ----------
        x : jax.Array
            Spatial feature map (B, H, W, C). H and W must be divisible
            by window_size.
        mask : jax.Array, optional
            Additional float additive bias of shape (num_windows, M^2, M^2).
            Added on top of shift mask and relative position bias.
            Not a boolean mask -- use 0.0 / -1e9 values.
        train : bool
            Enables dropout. Requires rngs={'dropout': key} when
            train=True and dropout_rate > 0.
        return_weights : bool
            If True returns (output, weights) where weights has shape
            (B*num_windows, num_heads, M^2, M^2). Weights are without dropout.

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
            Output (B, H, W, C), optionally with per-window attention weights.
        """
        # Validate spatial dims before JIT traces anything.
        # jax.ensure_compile_time_eval() forces this block to execute eagerly
        # at trace/compile time so the ValueError is always a plain Python
        # exception -- this means it fires correctly during training, not just
        # at init time.
        with jax.ensure_compile_time_eval():
            H_int = int(x.shape[1])
            W_int = int(x.shape[2])
            M     = self.window_size
            if H_int % M != 0 or W_int % M != 0:
                raise ValueError(
                    f"SwinWindowAttention: H={H_int}, W={W_int} must both be "
                    f"divisible by window_size={M}."
                )

        B, H, W, C = x.shape
        M2 = M * M

        # --- Cyclic shift (SW-MSA) ---
        if self.shift_size > 0:
            x          = jnp.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
            shift_mask = make_swin_shift_mask(M, self.shift_size, H_int, W_int)
        else:
            shift_mask = None

        # --- Window partition ---
        x_win  = self._partition_windows(x)   # (B*nW, M, M, C)
        x_flat = x_win.reshape(-1, M2, C)     # (B*nW, M^2, C)

        # --- QKV projections ---
        q = self.q_proj(x_flat)   # (B*nW, M^2, num_heads, head_dim)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        # --- Combined additive bias ---
        # rel_bias: (num_heads, M^2, M^2) -> (1, num_heads, M^2, M^2)
        rel_bias = self._relative_position_bias()[None]

        if shift_mask is not None:
            # (nW, M^2, M^2) -> (nW, 1, M^2, M^2) tiled to (B*nW, 1, M^2, M^2)
            shift_bias    = jnp.tile(shift_mask[:, None, :, :], (B, 1, 1, 1))
            combined_bias = rel_bias + shift_bias
        else:
            combined_bias = rel_bias

        if mask is not None:
            assert mask.ndim == 3, (
                f"SwinWindowAttention mask must have shape (num_windows, M^2, M^2), "
                f"got {mask.shape}"
            )
            # tile over batch: (nW, M^2, M^2) -> (B*nW, 1, M^2, M^2)
            user_bias     = jnp.tile(mask[:, None, :, :], (B, 1, 1, 1))
            combined_bias = combined_bias + user_bias

        # --- Attention weights: (B*nW, num_heads, M^2, M^2) ---
        weights = nn.dot_product_attention_weights(
            query=q,
            key=k,
            bias=combined_bias,
            dropout_rng=self.make_rng('dropout') if (train and self.dropout_rate > 0) else None,
            dropout_rate=self.dropout_rate if train else 0.0,
            deterministic=not train,
        )

        # --- Aggregate values ---
        # weights: (B*nW, num_heads, M^2, M^2)
        # v:       (B*nW, M^2,       num_heads, head_dim)
        out = jnp.einsum('bnij,bjnd->bind', weights, v)  # (B*nW, M^2, num_heads, head_dim)
        out = self.out_proj(out)                          # (B*nW, M^2, C)
        out = out.reshape(-1, M, M, C)                   # (B*nW, M, M, C)

        # --- Merge windows and reverse shift ---
        out = self._merge_windows(out, B, H_int, W_int)
        if self.shift_size > 0:
            out = jnp.roll(out, shift=(self.shift_size, self.shift_size), axis=(1, 2))

        if return_weights:
            if train and self.dropout_rate > 0:
                clean_weights = nn.dot_product_attention_weights(
                    query=q, key=k, bias=combined_bias,
                    dropout_rate=0.0, deterministic=True,
                )
            else:
                clean_weights = weights
            return out, clean_weights

        return out