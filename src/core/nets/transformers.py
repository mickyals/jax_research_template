"""
Transformer blocks and registered transformer architectures.

Blocks (not registered -- composed, not retrieved):
    TransformerBlock        Pre-LN self-attention + FFN
    CrossAttentionBlock     Pre-LN cross-attention + FFN
    SwinBlock               Pre-LN Swin window attention + FFN
    SwinBlockPair           W-MSA + SW-MSA SwinBlock pair
    PatchMerging            Swin spatial downsampling

Registered nets:
    TRANSFORMER_ENCODER     Sinusoidal/no pos enc + TransformerBlock stack
    TRANSFORMER_DECODER     TransformerBlock + CrossAttentionBlock pairs
                            Supports shared or per-layer context
                            Optional causal self-attention
    VIT                     PatchEmbed + LearnedPosEncoding + CLS token +
                            TransformerBlock stack + optional head
    MASKED_VIT              ViT encoder with MAE-style random patch masking
                            Returns (visible_tokens, mask, ids_restore)
    MAE_DECODER             Lightweight transformer decoder for MAE
                            reconstruction
    CONV_MAE_DECODER        ConvDecoder-based MAE reconstruction
                            Reshapes tokens to spatial grid before decoding
    SWIN_ENCODER            Hierarchical Swin stages + PatchMerging
                            Optional classification head
"""

from typing import Optional, Union
import jax
import jax.numpy as jnp
import flax.linen as nn
import warnings

from core.attention import (
    MultiHeadAttention,
    CrossAttention,
    SwinWindowAttention,
)
from core.nets.conv import PatchEmbed, ConvDecoder
from core.nets.mlp import MLP
from core.embeddings import SinusoidalPosEncoding, LearnedPosEncoding


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRANSFORMER_NETS: dict[str, dict] = {}


def register_transformer(name: str, description: str = ""):
    """Register a transformer net class by name.

    Parameters
    ----------
    name : str
        Name used for lookup. Stored uppercase.
    description : str, optional

    Returns
    -------
    callable
        Class decorator.

    Raises
    ------
    ValueError
        If a net with the same name is already registered.

    Example
    -------
    >>> @register_transformer("MY_NET", description="Custom transformer")
    ... class MyNet(nn.Module):
    ...     pass
    """
    name_upper = name.upper()

    def decorator(cls):
        if name_upper in TRANSFORMER_NETS:
            raise ValueError(
                f"Transformer net '{name_upper}' already exists."
            )
        TRANSFORMER_NETS[name_upper] = {"cls": cls, "description": description}
        return cls

    return decorator


def get_transformer(name: str, **kwargs):
    """Retrieve and instantiate a registered transformer net by name.

    Parameters
    ----------
    name : str
        Case-insensitive.
    **kwargs
        Forwarded to constructor. Unknown kwargs trigger UserWarning and
        are dropped.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        If no net with the given name exists.

    Example
    -------
    >>> net = get_transformer("VIT", patch_size=16, embed_dim=768,
    ...                       num_heads=12, num_layers=12, mlp_ratio=4)
    """
    name = name.upper()
    if name not in TRANSFORMER_NETS:
        available = ", ".join(sorted(TRANSFORMER_NETS.keys()))
        raise ValueError(
            f"Transformer net '{name}' does not exist. Available: {available}"
        )
    cls = TRANSFORMER_NETS[name]["cls"]
    if kwargs:
        try:
            valid = set(cls.__dataclass_fields__.keys())
            unknown = set(kwargs.keys()) - valid
            if unknown:
                warnings.warn(
                    f"get_transformer('{name}'): unknown kwargs {unknown} "
                    f"will be ignored. Valid kwargs: {valid or 'none'}.",
                    UserWarning,
                    stacklevel=2,
                )
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        except AttributeError:
            pass
    return cls(**kwargs)


def list_transformers() -> dict[str, str]:
    """Return sorted dict of registered transformer net names and descriptions.

    Example
    -------
    >>> list_transformers()
    {'MAE_DECODER': '...', 'SWIN_ENCODER': '...', 'VIT': '...', ...}
    """
    return {
        name: info["description"]
        for name, info in sorted(TRANSFORMER_NETS.items())
    }


# ---------------------------------------------------------------------------
# FFN helper
# ---------------------------------------------------------------------------

def _make_ffn(embed_dim: int, mlp_ratio: float, dropout_rate: float) -> MLP:
    """Construct the feedforward network used inside transformer blocks.

    Two-layer MLP: Linear(embed_dim -> mlp_ratio*embed_dim) -> GELU ->
    Dropout -> Linear(mlp_ratio*embed_dim -> embed_dim).

    Uses MLP from mlp.py with n_layers=1 (one hidden layer + output layer).

    Parameters
    ----------
    embed_dim : int
    mlp_ratio : float
        Hidden dim multiplier. Typically 4.0.
    dropout_rate : float

    Returns
    -------
    MLP
    """
    return MLP(
        out_features=embed_dim,
        hidden_features=int(embed_dim * mlp_ratio),
        n_layers=1,
        activation='gelu',
        initializer='xavier_uniform',
        dropout_rate=dropout_rate,
    )


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block.

    Pre-LayerNorm ordering (Xiong et al. 2020):
        x = x + Dropout(MHA(LN(x)))
        x = x + Dropout(FFN(LN(x)))

    Parameters
    ----------
    embed_dim : int
        Token embedding dimensionality.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        FFN hidden dim = mlp_ratio * embed_dim. Default 4.0.
    dropout_rate : float
        Applied after attention and FFN. Default 0.0.
    attn_dropout_rate : float
        Applied to attention weights. Default 0.0.
    causal : bool
        If True, applies causal mask in self-attention. Default False.
    use_bias : bool
        QKV projection bias. Default True.

    Notes
    -----
    Input/output: (B, T, embed_dim).

    For set/permutation-invariant tasks do not add positional encoding
    before passing to this block -- MHA is permutation-equivariant.

    Example
    -------
    >>> block = TransformerBlock(embed_dim=256, num_heads=8)
    >>> variables = block.init(jax.random.PRNGKey(0),
    ...                        jnp.ones((2, 16, 256)), train=False)
    >>> out = block.apply(variables, jnp.ones((2, 16, 256)), train=False)
    >>> out.shape
    (2, 16, 256)
    """
    embed_dim:         int
    num_heads:         int
    mlp_ratio:         float = 4.0
    dropout_rate:      float = 0.0
    attn_dropout_rate: float = 0.0
    causal:            bool  = False
    use_bias:          bool  = True

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.attn  = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.attn_dropout_rate,
            use_bias=self.use_bias,
            causal=self.causal,
        )
        self.ffn  = _make_ffn(self.embed_dim, self.mlp_ratio,
                               self.dropout_rate)
        self.drop = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        train: bool = True,
        return_weights: bool = False,
    ) -> Union[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (B, T, embed_dim).
        mask : jax.Array, optional
            Attention mask. See MultiHeadAttention for shape conventions.
        train : bool
        return_weights : bool
            If True returns (output, attn_weights) where attn_weights has
            shape (B, num_heads, T, T).

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
        """
        # --- Self-attention path ---
        if return_weights:
            attn_out, weights = self.attn(
                self.norm1(x), mask=mask, train=train, return_weights=True
            )
        else:
            attn_out = self.attn(
                self.norm1(x), mask=mask, train=train
            )
        x = x + self.drop(attn_out, deterministic=not train)

        # --- FFN path ---
        x = x + self.drop(
            self.ffn(self.norm2(x), train=train),
            deterministic=not train,
        )

        if return_weights:
            return x, weights
        return x


# ---------------------------------------------------------------------------
# CrossAttentionBlock
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention block.

    Pre-LayerNorm ordering:
        x = x + Dropout(CrossAttn(LN_q(x), LN_kv(context)))
        x = x + Dropout(FFN(LN(x)))

    Separate LayerNorm for query (x) and key/value (context) sources,
    standard practice for encoder-decoder cross-attention.

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    mlp_ratio : float
        Default 4.0.
    dropout_rate : float
        Default 0.0.
    attn_dropout_rate : float
        Default 0.0.
    use_bias : bool
        Default True.

    Notes
    -----
    x and context must share embed_dim.
    Input/output x: (B, T_q, embed_dim).
    context:        (B, T_kv, embed_dim).

    Example
    -------
    >>> block = CrossAttentionBlock(embed_dim=256, num_heads=8)
    >>> variables = block.init(jax.random.PRNGKey(0),
    ...                        jnp.ones((2, 16, 256)),
    ...                        jnp.ones((2, 32, 256)),
    ...                        train=False)
    >>> out = block.apply(variables,
    ...                   jnp.ones((2, 16, 256)),
    ...                   jnp.ones((2, 32, 256)),
    ...                   train=False)
    >>> out.shape
    (2, 16, 256)
    """
    embed_dim:         int
    num_heads:         int
    mlp_ratio:         float = 4.0
    dropout_rate:      float = 0.0
    attn_dropout_rate: float = 0.0
    use_bias:          bool  = True

    def setup(self):
        self.norm_q  = nn.LayerNorm()
        self.norm_kv = nn.LayerNorm()
        self.norm_ff = nn.LayerNorm()
        self.attn    = CrossAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.attn_dropout_rate,
            use_bias=self.use_bias,
        )
        self.ffn  = _make_ffn(self.embed_dim, self.mlp_ratio,
                               self.dropout_rate)
        self.drop = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jax.Array,
        context: jax.Array,
        mask: Optional[jax.Array] = None,
        train: bool = True,
        return_weights: bool = False,
    ) -> Union[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        Parameters
        ----------
        x : jax.Array
            Query source (B, T_q, embed_dim).
        context : jax.Array
            Key/value source (B, T_kv, embed_dim).
        mask : jax.Array, optional
            Cross-attention mask broadcastable to
            (B, num_heads, T_q, T_kv).
        train : bool
        return_weights : bool
            If True returns (output, attn_weights) where attn_weights has
            shape (B, num_heads, T_q, T_kv).

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
        """
        if return_weights:
            attn_out, weights = self.attn(
                self.norm_q(x),
                context=self.norm_kv(context),
                mask=mask,
                train=train,
                return_weights=True,
            )
        else:
            attn_out = self.attn(
                self.norm_q(x),
                context=self.norm_kv(context),
                mask=mask,
                train=train,
            )
        x = x + self.drop(attn_out, deterministic=not train)
        x = x + self.drop(
            self.ffn(self.norm_ff(x), train=train),
            deterministic=not train,
        )

        if return_weights:
            return x, weights
        return x


# ---------------------------------------------------------------------------
# SwinBlock
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    """Pre-LN Swin window attention block.

    Pre-LayerNorm ordering:
        x = x + Dropout(SwinWindowAttn(LN(x)))
        x = x + Dropout(FFN(LN(x_flat)))

    FFN operates on flattened tokens (B, H*W, C) then reshapes back to
    (B, H, W, C) since MLP from mlp.py expects sequence input.

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    window_size : int
        Default 7.
    shift_size : int
        0 = W-MSA, window_size//2 = SW-MSA. Default 0.
    mlp_ratio : float
        Default 4.0.
    dropout_rate : float
        Default 0.0.
    attn_dropout_rate : float
        Default 0.0.
    use_bias : bool
        Default True.

    Notes
    -----
    Input/output: (B, H, W, embed_dim).
    H and W must be divisible by window_size.

    Example
    -------
    >>> block = SwinBlock(embed_dim=96, num_heads=3, window_size=7)
    >>> variables = block.init(jax.random.PRNGKey(0),
    ...                        jnp.ones((2, 56, 56, 96)), train=False)
    >>> out = block.apply(variables, jnp.ones((2, 56, 56, 96)), train=False)
    >>> out.shape
    (2, 56, 56, 96)
    """
    embed_dim:         int
    num_heads:         int
    window_size:       int   = 7
    shift_size:        int   = 0
    mlp_ratio:         float = 4.0
    dropout_rate:      float = 0.0
    attn_dropout_rate: float = 0.0
    use_bias:          bool  = True

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.attn  = SwinWindowAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=self.shift_size,
            dropout_rate=self.attn_dropout_rate,
            use_bias=self.use_bias,
        )
        self.ffn  = _make_ffn(self.embed_dim, self.mlp_ratio,
                               self.dropout_rate)
        self.drop = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jax.Array,
        train: bool = True,
        return_weights: bool = False,
    ) -> Union[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (B, H, W, embed_dim).
        train : bool
        return_weights : bool
            If True returns (output, attn_weights) where attn_weights has
            shape (B*num_windows, num_heads, M^2, M^2).

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
        """
        B, H, W, C = x.shape

        # --- Window attention path ---
        if return_weights:
            attn_out, weights = self.attn(
                self.norm1(x), train=train, return_weights=True
            )
        else:
            attn_out = self.attn(self.norm1(x), train=train)

        x = x + self.drop(attn_out, deterministic=not train)

        # --- FFN path -- flatten to sequence, apply FFN, reshape back ---
        x_flat = x.reshape(B, H * W, C)
        x_flat = x_flat + self.drop(
            self.ffn(self.norm2(x_flat), train=train),
            deterministic=not train,
        )
        x = x_flat.reshape(B, H, W, C)

        if return_weights:
            return x, weights
        return x


# ---------------------------------------------------------------------------
# SwinBlockPair
# ---------------------------------------------------------------------------

class SwinBlockPair(nn.Module):
    """W-MSA + SW-MSA SwinBlock pair -- the standard Swin building unit.

    The first block uses standard window attention (shift_size=0) and the
    second uses shifted window attention (shift_size=window_size//2),
    enabling cross-window communication every two blocks.

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    window_size : int
        Default 7.
    mlp_ratio : float
        Default 4.0.
    dropout_rate : float
        Default 0.0.
    attn_dropout_rate : float
        Default 0.0.
    use_bias : bool
        Default True.

    Notes
    -----
    Input/output: (B, H, W, embed_dim).
    H and W must be divisible by window_size.

    Example
    -------
    >>> pair = SwinBlockPair(embed_dim=96, num_heads=3, window_size=7)
    >>> variables = pair.init(jax.random.PRNGKey(0),
    ...                       jnp.ones((2, 56, 56, 96)), train=False)
    >>> out = pair.apply(variables, jnp.ones((2, 56, 56, 96)), train=False)
    >>> out.shape
    (2, 56, 56, 96)
    """
    embed_dim:         int
    num_heads:         int
    window_size:       int   = 7
    mlp_ratio:         float = 4.0
    dropout_rate:      float = 0.0
    attn_dropout_rate: float = 0.0
    use_bias:          bool  = True

    def setup(self):
        self.block_w  = SwinBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=0,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
            use_bias=self.use_bias,
        )
        self.block_sw = SwinBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=self.window_size // 2,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
            use_bias=self.use_bias,
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (B, H, W, embed_dim).
        train : bool

        Returns
        -------
        jax.Array
            Shape (B, H, W, embed_dim).
        """
        x = self.block_w(x,  train=train)
        x = self.block_sw(x, train=train)
        return x


# ---------------------------------------------------------------------------
# PatchMerging
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """Swin Transformer patch merging (spatial downsampling).

    Concatenates 2x2 neighbouring patches along the channel dimension
    then applies LayerNorm and a linear projection to produce 2*C channels.

    (B, H, W, C) -> (B, H//2, W//2, 2*C)

    Parameters
    ----------
    use_bias : bool
        Linear projection bias. Default False.

    Notes
    -----
    H and W must be even. Validated at call time before JAX tracing.

    Uses @nn.compact for the Dense projection since input channels are
    not known at construction time (they depend on the Swin stage).

    Example
    -------
    >>> merge = PatchMerging()
    >>> variables = merge.init(jax.random.PRNGKey(0),
    ...                        jnp.ones((2, 56, 56, 96)), train=False)
    >>> out = merge.apply(variables, jnp.ones((2, 56, 56, 96)), train=False)
    >>> out.shape
    (2, 28, 28, 192)
    """
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (B, H, W, C). H and W must be even.

        Returns
        -------
        jax.Array
            Shape (B, H//2, W//2, 2*C).
        """
        with jax.ensure_compile_time_eval():
            H_int, W_int = int(x.shape[1]), int(x.shape[2])
            if H_int % 2 != 0 or W_int % 2 != 0:
                raise ValueError(
                    f"PatchMerging: H={H_int}, W={W_int} must both be even."
                )

        B, H, W, C = x.shape

        x0 = x[:, 0::2, 0::2, :]   # (B, H//2, W//2, C) top-left
        x1 = x[:, 1::2, 0::2, :]   # bottom-left
        x2 = x[:, 0::2, 1::2, :]   # top-right
        x3 = x[:, 1::2, 1::2, :]   # bottom-right

        x = jnp.concatenate([x0, x1, x2, x3], axis=-1)  # (B, H//2, W//2, 4C)
        x = nn.LayerNorm()(x)
        x = nn.Dense(2 * C, use_bias=self.use_bias)(x)   # (B, H//2, W//2, 2C)
        return x


# ---------------------------------------------------------------------------
# Registered nets
# ---------------------------------------------------------------------------

@register_transformer(
    "TRANSFORMER_ENCODER",
    description="Transformer encoder with optional sinusoidal positional encoding",
)
class TransformerEncoder(nn.Module):
    """Transformer encoder: optional pos encoding + TransformerBlock stack.

    Parameters
    ----------
    num_layers : int
        Number of TransformerBlocks.
    embed_dim : int
    num_heads : int
    mlp_ratio : float
        Default 4.0.
    dropout_rate : float
        Default 0.0.
    attn_dropout_rate : float
        Default 0.0.
    causal : bool
        If True, causal masking in all blocks. Default False.
    use_bias : bool
        Default True.
    add_pos_encoding : bool
        If True, adds sinusoidal positional encoding to the input before
        the transformer blocks. Set False for set/permutation-invariant
        tasks or when positional encoding is handled externally.
        Default True.
    max_len : int
        Maximum sequence length for sinusoidal encoding. Default 5000.

    Notes
    -----
    Input:  (B, T, embed_dim) -- already-embedded tokens.
    Output: (B, T, embed_dim).

    Input projection (raw features -> embed_dim) is the caller's
    responsibility.

    Example
    -------
    >>> enc = TransformerEncoder(num_layers=6, embed_dim=256, num_heads=8)
    >>> variables = enc.init(jax.random.PRNGKey(0),
    ...                      jnp.ones((2, 16, 256)), train=False)
    >>> out = enc.apply(variables, jnp.ones((2, 16, 256)), train=False)
    >>> out.shape
    (2, 16, 256)
    """
    num_layers:        int
    embed_dim:         int
    num_heads:         int
    mlp_ratio:         float = 4.0
    dropout_rate:      float = 0.0
    attn_dropout_rate: float = 0.0
    causal:            bool  = False
    use_bias:          bool  = True
    add_pos_encoding:  bool  = True
    max_len:           int   = 5000

    def setup(self):
        self.pos_enc = (
            SinusoidalPosEncoding(d_model=self.embed_dim, max_len=self.max_len)
            if self.add_pos_encoding else None
        )
        self.blocks = [
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                attn_dropout_rate=self.attn_dropout_rate,
                causal=self.causal,
                use_bias=self.use_bias,
            )
            for _ in range(self.num_layers)
        ]
        self.norm = nn.LayerNorm()

    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (B, T, embed_dim).
        mask : jax.Array, optional
            Attention mask threaded to all blocks.
        train : bool

        Returns
        -------
        jax.Array
            Shape (B, T, embed_dim).
        """
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x, mask=mask, train=train)

        return self.norm(x)

    def get_attention_maps(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        train: bool = False,
    ) -> list[jax.Array]:
        """Return per-layer attention weight maps for visualisation.

        Parameters
        ----------
        x : jax.Array
            Shape (B, T, embed_dim).
        mask : jax.Array, optional
        train : bool

        Returns
        -------
        list of jax.Array
            Length num_layers, each (B, num_heads, T, T).
        """
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        maps = []
        for block in self.blocks:
            x, w = block(x, mask=mask, train=train, return_weights=True)
            maps.append(w)
        return maps


@register_transformer(
    "TRANSFORMER_DECODER",
    description="Transformer decoder with self-attention + cross-attention pairs",
)
class TransformerDecoder(nn.Module):
    """Transformer decoder: TransformerBlock + CrossAttentionBlock pairs.

    Supports both shared context (one tensor for all layers) and per-layer
    context (one tensor per layer, for multi-scale encoder outputs such as
    hierarchical Swin encoder stages).

    Parameters
    ----------
    num_layers : int
    embed_dim : int
    num_heads : int
    mlp_ratio : float
        Default 4.0.
    dropout_rate : float
        Default 0.0.
    attn_dropout_rate : float
        Default 0.0.
    causal : bool
        If True, self-attention blocks use causal masking. Useful for
        autoregressive decoding. Default False.
    use_bias : bool
        Default True.

    Notes
    -----
    Input/output x:   (B, T_q, embed_dim).
    context (shared): (B, T_kv, embed_dim).
    context (per-layer): list of num_layers tensors,
                         each (B, T_kv_i, embed_dim).

    Example
    -------
    >>> dec = TransformerDecoder(num_layers=6, embed_dim=256, num_heads=8)
    >>> variables = dec.init(jax.random.PRNGKey(0),
    ...                      jnp.ones((2, 16, 256)),
    ...                      jnp.ones((2, 32, 256)),
    ...                      train=False)
    >>> out = dec.apply(variables,
    ...                 jnp.ones((2, 16, 256)),
    ...                 jnp.ones((2, 32, 256)),
    ...                 train=False)
    >>> out.shape
    (2, 16, 256)
    """
    num_layers:        int
    embed_dim:         int
    num_heads:         int
    mlp_ratio:         float = 4.0
    dropout_rate:      float = 0.0
    attn_dropout_rate: float = 0.0
    causal:            bool  = False
    use_bias:          bool  = True

    def setup(self):
        self.self_attn_blocks = [
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                attn_dropout_rate=self.attn_dropout_rate,
                causal=self.causal,
                use_bias=self.use_bias,
            )
            for _ in range(self.num_layers)
        ]
        self.cross_attn_blocks = [
            CrossAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                attn_dropout_rate=self.attn_dropout_rate,
                use_bias=self.use_bias,
            )
            for _ in range(self.num_layers)
        ]
        self.norm = nn.LayerNorm()

    def __call__(
        self,
        x: jax.Array,
        context: Union[jax.Array, list[jax.Array]],
        self_mask: Optional[jax.Array] = None,
        cross_mask: Optional[jax.Array] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array
            Decoder input (B, T_q, embed_dim).
        context : jax.Array or list of jax.Array
            Encoder output(s). Single tensor for shared context (standard),
            list of num_layers tensors for per-layer context (e.g. from
            multi-scale Swin encoder stages).
        self_mask : jax.Array, optional
            Mask for self-attention blocks.
        cross_mask : jax.Array, optional
            Mask for cross-attention blocks.
        train : bool

        Returns
        -------
        jax.Array
            Shape (B, T_q, embed_dim).
        """
        if isinstance(context, jax.Array):
            contexts = [context] * self.num_layers
        else:
            if len(context) != self.num_layers:
                raise ValueError(
                    f"TransformerDecoder: per-layer context requires "
                    f"{self.num_layers} tensors, got {len(context)}."
                )
            contexts = context

        for sa_block, ca_block, ctx in zip(
            self.self_attn_blocks, self.cross_attn_blocks, contexts
        ):
            x = sa_block(x, mask=self_mask, train=train)
            x = ca_block(x, context=ctx, mask=cross_mask, train=train)

        return self.norm(x)


@register_transformer(
    "VIT",
    description="Vision Transformer (Dosovitskiy et al. 2020)",
)
class ViT(nn.Module):
    """Vision Transformer for image classification or feature extraction.

    PatchEmbed -> CLS token prepend -> LearnedPosEncoding ->
    Dropout -> TransformerEncoder -> LayerNorm -> head (optional).

    Parameters
    ----------
    patch_size : int
        Patch side length P. H and W must be divisible by patch_size.
    embed_dim : int
        Token embedding dimensionality.
    num_heads : int
    num_layers : int
    mlp_ratio : float
        Default 4.0.
    num_classes : int, optional
        If provided, adds a linear classification head on the CLS token.
        If None, returns CLS token features (B, embed_dim). Default None.
    dropout_rate : float
        Applied to the token sequence after positional encoding and after
        attention/FFN in each block. Default 0.0.
    attn_dropout_rate : float
        Default 0.0.
    use_bias : bool
        Default True.

    Notes
    -----
    Input:  (B, H, W, C) channels-last image.
    Output: (B, num_classes) if num_classes is set,
            (B, embed_dim) CLS token features otherwise.

    Positional encoding is learnable (LearnedPosEncoding), initialised
    with stddev=0.02. The CLS token is prepended before positional
    encoding, so the encoding covers T+1 positions (T patch tokens + 1
    CLS token).

    Example
    -------
    >>> vit = ViT(patch_size=16, embed_dim=768, num_heads=12,
    ...           num_layers=12, num_classes=1000)
    >>> variables = vit.init(jax.random.PRNGKey(0),
    ...                      jnp.ones((2, 224, 224, 3)), train=False)
    >>> out = vit.apply(variables, jnp.ones((2, 224, 224, 3)), train=False)
    >>> out.shape
    (2, 1000)
    """
    patch_size:        int
    embed_dim:         int
    num_heads:         int
    num_layers:        int
    mlp_ratio:         float         = 4.0
    num_classes:       Optional[int] = None
    dropout_rate:      float         = 0.0
    attn_dropout_rate: float         = 0.0
    use_bias:          bool          = True

    def setup(self):
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            flatten=True,
            use_bias=True,
        )
        self.cls_token = self.param(
            'cls_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim),
        )
        self.encoder = TransformerEncoder(
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
            use_bias=self.use_bias,
            add_pos_encoding=False,
        )
        self.norm = nn.LayerNorm()
        self.head = (
            nn.Dense(self.num_classes) if self.num_classes is not None else None
        )
        self.drop = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def _embed(self, x: jax.Array, train: bool = True) -> jax.Array:
        """Embed patches, prepend CLS token, add positional encoding.

        Extracted into a separate @nn.compact method so that
        LearnedPosEncoding is defined in a single compact scope and can
        be called from both __call__ and get_attention_maps without
        triggering AssignSubModuleError.

        Parameters
        ----------
        x : jax.Array
            Shape (B, H, W, C).
        train : bool

        Returns
        -------
        jax.Array
            Shape (B, T+1, embed_dim).
        """
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, T, embed_dim)
        T = x.shape[1]

        cls = jnp.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x   = jnp.concatenate([cls, x], axis=1)         # (B, T+1, embed_dim)

        x = LearnedPosEncoding(
            num_tokens=T + 1,
            embed_dim=self.embed_dim,
            name='pos_encoding',
        )(x)

        return self.drop(x, deterministic=not train)

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (B, H, W, C).
        train : bool

        Returns
        -------
        jax.Array
            Shape (B, num_classes) or (B, embed_dim).
        """
        x       = self._embed(x, train=train)
        x       = self.encoder(x, train=train)
        x       = self.norm(x)
        cls_out = x[:, 0]

        if self.head is not None:
            return self.head(cls_out)
        return cls_out

    def get_attention_maps(
        self,
        x: jax.Array,
        train: bool = False,
    ) -> list[jax.Array]:
        """Return per-layer attention maps.

        Parameters
        ----------
        x : jax.Array
            Shape (B, H, W, C).
        train : bool

        Returns
        -------
        list of jax.Array
            Length num_layers, each (B, num_heads, T+1, T+1).
        """
        x = self._embed(x, train=train)
        return self.encoder.get_attention_maps(x, train=train)


@register_transformer(
    "MASKED_VIT",
    description="Masked Vision Transformer encoder for MAE pretraining",
)
class MaskedViT(nn.Module):
    """ViT encoder with MAE-style random patch masking.

    Encodes only visible (unmasked) patches. Returns encoded visible tokens,
    the boolean mask, and the restore indices needed by the decoder to
    reconstruct the full sequence.

    At train time: randomly masks mask_ratio fraction of patches using the
    'mask' PRNG key. At eval time: encodes all patches (no masking).

    Parameters
    ----------
    patch_size : int
    embed_dim : int
    num_heads : int
    num_layers : int
    mask_ratio : float
        Fraction of patches to mask during training. Default 0.75.
    mlp_ratio : float
        Default 4.0.
    dropout_rate : float
        Default 0.0.
    attn_dropout_rate : float
        Default 0.0.
    use_bias : bool
        Default True.

    Notes
    -----
    Input:  (B, H, W, C).
    Output: (visible_tokens, mask, ids_restore) where:
        visible_tokens: (B, T_visible, embed_dim)
        mask:           (B, T) bool, True = masked (not seen by encoder)
        ids_restore:    (B, T) int, indices to unshuffle the full sequence

    T_visible = round(T * (1 - mask_ratio)) during training.
    T_visible = T during eval (no masking), mask is all-False,
    ids_restore is identity.

    Requires rngs={'mask': key} at train time.

    The masking procedure follows He et al. 2022 (MAE):
    1. Sample random noise per token per sample.
    2. Sort by noise to get a shuffle order.
    3. Keep the last T_visible tokens (those with highest noise rank,
       i.e. the ones NOT masked).
    4. ids_restore = argsort(shuffle order) -- used by decoder to place
       tokens back in original positions.

    Example
    -------
    >>> mvit = MaskedViT(patch_size=16, embed_dim=768, num_heads=12,
    ...                  num_layers=12, mask_ratio=0.75)
    >>> variables = mvit.init(
    ...     {'params': jax.random.PRNGKey(0), 'mask': jax.random.PRNGKey(1)},
    ...     jnp.ones((2, 224, 224, 3)), train=True)
    >>> visible, mask, ids_restore = mvit.apply(
    ...     variables, jnp.ones((2, 224, 224, 3)), train=True,
    ...     rngs={'mask': jax.random.PRNGKey(2)})
    >>> visible.shape      # (2, 49, 768) -- 25% of 196 patches
    >>> mask.shape         # (2, 196)
    >>> ids_restore.shape  # (2, 196)
    """
    patch_size:        int
    embed_dim:         int
    num_heads:         int
    num_layers:        int
    mask_ratio:        float = 0.75
    mlp_ratio:         float = 4.0
    dropout_rate:      float = 0.0
    attn_dropout_rate: float = 0.0
    use_bias:          bool  = True

    def setup(self):
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            flatten=True,
            use_bias=True,
        )
        self.cls_token = self.param(
            'cls_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim),
        )
        self.encoder = TransformerEncoder(
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
            use_bias=self.use_bias,
            add_pos_encoding=False,
        )
        self.norm = nn.LayerNorm()
        self.drop = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        train: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (B, H, W, C).
        train : bool
            If True applies random masking using rngs={'mask': key}.
            If False encodes all patches, returns all-False mask and
            identity ids_restore.

        Returns
        -------
        tuple[jax.Array, jax.Array, jax.Array]
            (visible_tokens, mask, ids_restore)
        """
        B = x.shape[0]

        # embed patches: (B, H, W, C) -> (B, T, embed_dim)
        x = self.patch_embed(x)
        T = x.shape[1]

        # positional encoding before masking so every visible token carries
        # its original spatial position information into the encoder
        x = LearnedPosEncoding(
            num_tokens=T,
            embed_dim=self.embed_dim,
            name='pos_encoding',
        )(x)

        if train:
            num_masked  = int(T * self.mask_ratio)
            num_visible = T - num_masked

            # per-sample random shuffle
            noise        = jax.random.uniform(
                self.make_rng('mask'), (B, T)
            )
            ids_shuffle  = jnp.argsort(noise, axis=1)       # (B, T)
            ids_restore  = jnp.argsort(ids_shuffle, axis=1) # (B, T)

            # keep the last num_visible tokens in the shuffle order
            # (those with the highest noise rank = not masked)
            ids_keep = ids_shuffle[:, num_masked:]           # (B, T_visible)

            # gather visible tokens -- works per sample via take_along_axis
            x_visible = jnp.take_along_axis(
                x,
                ids_keep[:, :, None] * jnp.ones(
                    (1, 1, self.embed_dim), dtype=jnp.int32
                ),
                axis=1,
            )                                                # (B, T_visible, D)

            # mask: True where token is masked
            # tokens with shuffle rank < num_masked are masked
            mask = ids_restore < num_masked                  # (B, T) bool

        else:
            x_visible   = x
            ids_restore = jnp.broadcast_to(
                jnp.arange(T)[None], (B, T)
            )
            mask = jnp.zeros((B, T), dtype=jnp.bool_)

        # prepend CLS token to visible sequence
        cls       = jnp.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x_visible = jnp.concatenate([cls, x_visible], axis=1)

        x_visible = self.drop(x_visible, deterministic=not train)
        x_visible = self.encoder(x_visible, train=train)
        x_visible = self.norm(x_visible)

        # strip CLS token before returning
        x_visible = x_visible[:, 1:]                        # (B, T_visible, D)

        return x_visible, mask, ids_restore


@register_transformer(
    "MAE_DECODER",
    description="Lightweight transformer decoder for MAE patch reconstruction",
)
class MAEDecoder(nn.Module):
    """Transformer decoder for MAE patch reconstruction.

    Takes visible encoded tokens, the boolean mask, and ids_restore from
    MaskedViT. Inserts learnable mask tokens at masked positions using
    ids_restore to unshuffle the full sequence, adds positional encoding,
    then runs a lightweight transformer to reconstruct all patches.

    Parameters
    ----------
    num_patches : int
        Total number of patches T (before masking).
    patch_dim : int
        Reconstruction target dimensionality = patch_size^2 * in_channels.
    embed_dim : int
        Decoder embedding dim. Typically smaller than encoder embed_dim.
    num_heads : int
    num_layers : int
        Typically 4-8 (lighter than encoder).
    mlp_ratio : float
        Default 4.0.
    dropout_rate : float
        Default 0.0.
    use_bias : bool
        Default True.

    Notes
    -----
    visible_tokens: (B, T_visible, encoder_embed_dim) -- from MaskedViT
    mask:           (B, T) bool, True = masked
    ids_restore:    (B, T) int -- from MaskedViT

    A linear projection maps encoder_embed_dim -> embed_dim before
    decoding, allowing encoder and decoder to have different widths.

    Output: (B, T, patch_dim) -- reconstructed pixel values for all patches.
    Loss should be computed only on masked patches (where mask=True).

    Example
    -------
    >>> dec = MAEDecoder(num_patches=196, patch_dim=768,
    ...                  embed_dim=512, num_heads=8, num_layers=4)
    """
    num_patches:  int
    patch_dim:    int
    embed_dim:    int
    num_heads:    int
    num_layers:   int
    mlp_ratio:    float = 4.0
    dropout_rate: float = 0.0
    use_bias:     bool  = True

    def setup(self):
        self.mask_token = self.param(
            'mask_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim),
        )
        self.proj_in  = nn.Dense(self.embed_dim, use_bias=True)
        self.proj_out = nn.Dense(self.patch_dim, use_bias=True)
        self.decoder_blocks = [
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
            )
            for _ in range(self.num_layers)
        ]
        self.norm = nn.LayerNorm()

    @nn.compact
    def __call__(
        self,
        visible_tokens: jax.Array,
        mask: jax.Array,
        ids_restore: jax.Array,
        train: bool = True,
    ) -> jax.Array:
        """
        Parameters
        ----------
        visible_tokens : jax.Array
            Shape (B, T_visible, encoder_embed_dim).
        mask : jax.Array
            Shape (B, T) bool. True = masked position.
        ids_restore : jax.Array
            Shape (B, T) int. From MaskedViT -- used to unshuffle the
            full sequence back to original token order.
        train : bool

        Returns
        -------
        jax.Array
            Shape (B, T, patch_dim). Reconstructed patches for all positions.
            Use mask to select only masked patches for loss computation.
        """
        B  = visible_tokens.shape[0]
        T  = self.num_patches
        T_vis = visible_tokens.shape[1]

        # project encoder dim -> decoder dim
        visible = self.proj_in(visible_tokens)  # (B, T_visible, embed_dim)

        # build full shuffled sequence:
        # [visible tokens | mask tokens] then unshuffle via ids_restore
        mask_tokens = jnp.broadcast_to(
            self.mask_token, (B, T - T_vis, self.embed_dim)
        )
        # concatenate: visible first, then mask tokens
        x_shuffled = jnp.concatenate([visible, mask_tokens], axis=1)  # (B, T, D)

        # unshuffle: ids_restore[b, i] = original position of shuffled token i
        x = jnp.take_along_axis(
            x_shuffled,
            ids_restore[:, :, None] * jnp.ones(
                (1, 1, self.embed_dim), dtype=jnp.int32
            ),
            axis=1,
        )  # (B, T, embed_dim) -- tokens in original spatial order

        # add positional encoding
        x = LearnedPosEncoding(
            num_tokens=T,
            embed_dim=self.embed_dim,
            name='pos_encoding',
        )(x)

        # decode
        for block in self.decoder_blocks:
            x = block(x, train=train)
        x = self.norm(x)

        return self.proj_out(x)   # (B, T, patch_dim)


@register_transformer(
    "CONV_MAE_DECODER",
    description="Conv-based MAE decoder for spatially structured patch reconstruction",
)
class ConvMAEDecoder(nn.Module):
    """ConvDecoder-based MAE reconstruction decoder.

    Reshapes the full reconstructed token sequence to a spatial grid then
    decodes with ConvDecoder. Better suited than a transformer decoder when
    the input has strong spatial structure (e.g. GOES satellite patches)
    because ConvDecoder explicitly models spatial locality via residual
    blocks and upsampling.

    Parameters
    ----------
    num_patches_h : int
        Number of patches along height (H // patch_size).
    num_patches_w : int
        Number of patches along width (W // patch_size).
    encoder_embed_dim : int
        Encoder output dimensionality (projected from).
    decoder_embed_dim : int
        Intermediate embedding dim before ConvDecoder.
    channels : tuple of int
        ConvDecoder channel schedule, e.g. (256, 128, 64).
    out_features : int
        Final output channels. Typically C_in (original image channels).
    num_res_blocks : int
        ConvDecoder residual blocks per level. Default 2.
    dropout_rate : float
        Default 0.0.

    Notes
    -----
    visible_tokens: (B, T_visible, encoder_embed_dim) -- from MaskedViT
    mask:           (B, T) bool, True = masked
    ids_restore:    (B, T) int -- from MaskedViT

    Output: (B, H_orig, W_orig, out_features) -- reconstructed image.

    The token infilling (mask tokens at masked positions) and unshuffling
    use the same ids_restore mechanism as MAEDecoder, ensuring correct
    per-sample spatial placement.

    Example
    -------
    >>> dec = ConvMAEDecoder(
    ...     num_patches_h=14, num_patches_w=14,
    ...     encoder_embed_dim=768, decoder_embed_dim=256,
    ...     channels=(256, 128, 64), out_features=3,
    ... )
    """
    num_patches_h:     int
    num_patches_w:     int
    encoder_embed_dim: int
    decoder_embed_dim: int
    channels:          tuple
    out_features:      int
    num_res_blocks:    int   = 2
    dropout_rate:      float = 0.0

    def setup(self):
        self.mask_token = self.param(
            'mask_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.encoder_embed_dim),
        )
        self.proj_in = nn.Dense(self.decoder_embed_dim, use_bias=True)
        self.decoder = ConvDecoder(
            channels=self.channels,
            out_features=self.out_features,
            num_res_blocks=self.num_res_blocks,
            dropout_rate=self.dropout_rate,
        )

    def __call__(
        self,
        visible_tokens: jax.Array,
        mask: jax.Array,
        ids_restore: jax.Array,
        train: bool = True,
    ) -> jax.Array:
        """
        Parameters
        ----------
        visible_tokens : jax.Array
            Shape (B, T_visible, encoder_embed_dim).
        mask : jax.Array
            Shape (B, T) bool. True = masked.
        ids_restore : jax.Array
            Shape (B, T) int. From MaskedViT.
        train : bool

        Returns
        -------
        jax.Array
            Shape (B, H_orig, W_orig, out_features).
        """
        B     = visible_tokens.shape[0]
        T     = self.num_patches_h * self.num_patches_w
        T_vis = visible_tokens.shape[1]
        nH    = self.num_patches_h
        nW    = self.num_patches_w

        # infill with mask tokens then unshuffle -- same as MAEDecoder
        mask_tokens = jnp.broadcast_to(
            self.mask_token, (B, T - T_vis, self.encoder_embed_dim)
        )
        x_shuffled = jnp.concatenate(
            [visible_tokens, mask_tokens], axis=1
        )  # (B, T, encoder_embed_dim)

        x = jnp.take_along_axis(
            x_shuffled,
            ids_restore[:, :, None] * jnp.ones(
                (1, 1, self.encoder_embed_dim), dtype=jnp.int32
            ),
            axis=1,
        )  # (B, T, encoder_embed_dim) -- original spatial order

        # project and reshape to spatial grid
        x = self.proj_in(x)                           # (B, T, decoder_embed_dim)
        x = x.reshape(B, nH, nW, self.decoder_embed_dim)  # (B, nH, nW, D)

        return self.decoder(x, train=train)           # (B, H_orig, W_orig, C)


@register_transformer(
    "SWIN_ENCODER",
    description="Hierarchical Swin Transformer encoder (Liu et al. 2021)",
)
class SwinEncoder(nn.Module):
    """Hierarchical Swin Transformer encoder.

    PatchEmbed (spatial grid) -> LayerNorm -> Dropout ->
    [SwinBlockPair x depths[i] -> PatchMerging] x stages ->
    LayerNorm -> global avg pool -> optional head.

    Parameters
    ----------
    patch_size : int
        Initial patch embedding stride. Default 4.
    embed_dim : int
        Base channel count after patch embedding. Doubles after each
        PatchMerging stage. Default 96.
    depths : tuple of int
        Number of SwinBlockPairs per stage. Default (2, 2, 6, 2).
    num_heads : tuple of int
        Attention heads per stage. Must match len(depths).
        Default (3, 6, 12, 24).
    window_size : int
        Swin attention window size. Default 7.
    mlp_ratio : float
        Default 4.0.
    num_classes : int, optional
        If set, adds a linear head after global avg pool.
        If None, returns pooled feature vector. Default None.
    dropout_rate : float
        Default 0.0.
    attn_dropout_rate : float
        Default 0.0.
    use_bias : bool
        Default True.

    Notes
    -----
    Input:  (B, H, W, C) channels-last.
    Output: (B, num_classes) if num_classes set,
            (B, embed_dim * 2^(num_stages-1)) otherwise.

    H must be divisible by patch_size * window_size.
    W must be divisible by patch_size * window_size.

    Channel schedule (doubles after each PatchMerging):
        stage 0: embed_dim
        stage 1: 2 * embed_dim
        stage 2: 4 * embed_dim
        stage 3: 8 * embed_dim  (for default 4-stage config)

    PatchEmbed uses flatten=False to preserve the spatial grid layout
    required by SwinWindowAttention.

    Example
    -------
    >>> enc = SwinEncoder(patch_size=4, embed_dim=96,
    ...                   depths=(2, 2, 6, 2),
    ...                   num_heads=(3, 6, 12, 24),
    ...                   num_classes=1000)
    >>> variables = enc.init(jax.random.PRNGKey(0),
    ...                      jnp.ones((2, 224, 224, 3)), train=False)
    >>> out = enc.apply(variables, jnp.ones((2, 224, 224, 3)), train=False)
    >>> out.shape
    (2, 1000)
    """
    patch_size:        int          = 4
    embed_dim:         int          = 96
    depths:            tuple        = (2, 2, 6, 2)
    num_heads:         tuple        = (3, 6, 12, 24)
    window_size:       int          = 7
    mlp_ratio:         float        = 4.0
    num_classes:       Optional[int] = None
    dropout_rate:      float        = 0.0
    attn_dropout_rate: float        = 0.0
    use_bias:          bool         = True

    def __post_init__(self):
        super().__post_init__()
        if len(self.depths) != len(self.num_heads):
            raise ValueError(
                f"SwinEncoder: depths and num_heads must have the same length, "
                f"got {len(self.depths)} and {len(self.num_heads)}."
            )

    def setup(self):
        # flatten=False: PatchEmbed returns (B, nH, nW, embed_dim) spatial grid
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            flatten=False,
            use_bias=True,
        )
        self.patch_norm = nn.LayerNorm()
        self.drop = nn.Dropout(rate=self.dropout_rate)

        num_stages = len(self.depths)
        stages = []
        merges = []

        for i, (depth, heads) in enumerate(zip(self.depths, self.num_heads)):
            stage_dim = self.embed_dim * (2 ** i)
            stages.append([
                SwinBlockPair(
                    embed_dim=stage_dim,
                    num_heads=heads,
                    window_size=self.window_size,
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    attn_dropout_rate=self.attn_dropout_rate,
                    use_bias=self.use_bias,
                )
                for _ in range(depth)
            ])
            merges.append(
                PatchMerging(use_bias=False) if i < num_stages - 1 else None
            )

        # assign once -- Flax tracks these as submodule collections
        self.stages = stages
        self.merges = merges

        self.norm = nn.LayerNorm()
        self.head = (
            nn.Dense(self.num_classes) if self.num_classes is not None else None
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (B, H, W, C).
        train : bool

        Returns
        -------
        jax.Array
            Shape (B, num_classes) or (B, final_channels).
        """
        # nH, nW computed from input before PatchEmbed flattens anything
        with jax.ensure_compile_time_eval():
            H_int = int(x.shape[1])
            W_int = int(x.shape[2])
            P     = self.patch_size
            if H_int % P != 0 or W_int % P != 0:
                raise ValueError(
                    f"SwinEncoder: H={H_int}, W={W_int} must both be "
                    f"divisible by patch_size={P}."
                )

        # PatchEmbed with flatten=False: (B, H, W, C) -> (B, nH, nW, embed_dim)
        x = self.patch_embed(x)
        x = self.patch_norm(x)
        x = self.drop(x, deterministic=not train)

        for stage_blocks, merge in zip(self.stages, self.merges):
            for block_pair in stage_blocks:
                x = block_pair(x, train=train)
            if merge is not None:
                x = merge(x, train=train)

        x = self.norm(x)

        # global average pooling: (B, H', W', C') -> (B, C')
        x = jnp.mean(x, axis=(1, 2))

        if self.head is not None:
            return self.head(x)
        return x