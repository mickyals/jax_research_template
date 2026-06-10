# core/nets/conv.py
import inspect
import warnings
from typing import Optional, Callable, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn

from core import get_activation
from core import get_initializer
from core import get_norm


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONV_NETS: dict[str, dict] = {}


def register_conv_net(name: str, description: str = ""):
    """Register a conv net class by name.

    Parameters
    ----------
    name : str
        Name used for lookup. Stored uppercase.
    description : str, optional
        Short description shown in ``list_conv_nets()``.

    Returns
    -------
    callable
        Class decorator.

    Raises
    ------
    ValueError
        If a conv net with the same name is already registered.

    Example
    -------
    >>> @register_conv_net("MY_NET", description="Custom conv net")
    ... class MyNet(nn.Module):
    ...     pass
    """
    name_upper = name.upper()

    def decorator(cls):
        if name_upper in CONV_NETS:
            raise ValueError(f"Conv net with name '{name_upper}' already exists.")
        CONV_NETS[name_upper] = {"cls": cls, "description": description}
        return cls

    return decorator


def get_conv_net(name: str, **kwargs):
    """Retrieve and instantiate a registered conv net by name.

    Uses ``__dataclass_fields__`` for reliable kwarg inspection since
    Flax modules are dataclasses.

    Parameters
    ----------
    name : str
        Name of the registered conv net (case-insensitive).
    **kwargs
        Arguments forwarded to the net constructor. Unknown kwargs
        trigger a UserWarning and are dropped.

    Returns
    -------
    nn.Module
        An instantiated Flax Linen conv net module.

    Raises
    ------
    ValueError
        If no conv net with the given name exists.

    Example
    -------
    >>> net = get_conv_net("RESNET", c_hidden=(32, 64, 128), num_blocks=(2, 2, 2))
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((2, 32, 32, 3)),
    ...                      train=True)
    """
    name = name.upper()
    if name not in CONV_NETS:
        available = ", ".join(sorted(CONV_NETS.keys()))
        raise ValueError(f"Conv net '{name}' does not exist. Available: {available}")

    cls = CONV_NETS[name]["cls"]

    if kwargs:
        try:
            valid = set(cls.__dataclass_fields__.keys())
            unknown = set(kwargs.keys()) - valid
            if unknown:
                warnings.warn(
                    f"get_conv_net('{name}'): unknown kwargs {unknown} will be "
                    f"ignored. Valid kwargs: {valid or 'none'}.",
                    UserWarning,
                    stacklevel=2,
                )
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        except AttributeError:
            pass

    return cls(**kwargs)


def list_conv_nets() -> dict[str, str]:
    """Return a sorted dictionary of all registered conv net names and descriptions.

    Returns
    -------
    dict[str, str]

    Example
    -------
    >>> list_conv_nets()
    {'CONV_ENCODER': '...', 'DENSENET': '...', 'RESNET': '...', ...}
    """
    return {name: info["description"] for name, info in sorted(CONV_NETS.items())}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_norm(norm: nn.Module, x: jax.Array, train: bool) -> jax.Array:
    """Call a norm module, forwarding train only if it accepts it.

    BatchNorm requires train to switch between batch and running statistics.
    LayerNorm, GroupNorm, InstanceNorm, RMSNorm ignore train but accept it
    for API consistency.

    Parameters
    ----------
    norm : nn.Module
        Norm module from core.norms.
    x : jax.Array
        Input array.
    train : bool
        Training flag.

    Returns
    -------
    jax.Array
        Normalised array.
    """
    sig = inspect.signature(norm.__call__)
    if 'train' in sig.parameters:
        return norm(x, train=train)
    return norm(x)


# ---------------------------------------------------------------------------
# 2D Blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Single conv layer with norm, activation, optional pooling and dropout.

    Supports both post-norm (conv -> norm -> act -> pool -> drop) and
    pre-norm (norm -> act -> conv -> pool -> drop) orderings.

    Parameters
    ----------
    features : int
        Number of output channels.
    kernel_size : tuple of int
        Convolution kernel size. Default (3, 3).
    strides : tuple of int
        Convolution stride. Default (1, 1).
    padding : str
        Padding mode. Default 'SAME'.
    use_bias : bool
        Whether to include a bias term. Default False.
    norm : str
        Registered norm name. Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
        Kwargs forwarded to get_norm.
    activation : str
        Registered activation name. Default 'silu'.
    activation_kwargs : dict, optional
        Kwargs forwarded to get_activation.
    initializer : str
        Kernel initializer name. Default 'lecun_normal'.
    pre_norm : bool
        If True, applies norm -> act -> conv. Default False.
    pooling : str, optional
        Registered pooling name applied after conv+norm+act.
        Use 'SPATIAL_MAX' or 'SPATIAL_AVG' for downsampling.
        Default None (no pooling).
    pooling_kwargs : dict, optional
        Kwargs forwarded to get_pooling.
    dropout_rate : float
        Spatial dropout rate applied after activation (and pooling if set).
        Zeros entire feature maps rather than individual elements.
        Default 0.0 (no dropout). Requires rngs={'dropout': key} at
        call time when rate > 0 and train=True.

    Notes
    -----
    BatchNorm requires mutable batch_stats in the variable collection
    during training. See core.norms.BatchNorm for details.

    When dropout_rate > 0, pass rngs={'dropout': key} to apply:
        block.apply(variables, x, train=True, rngs={'dropout': key},
                    mutable=['batch_stats'])

    Example
    -------
    >>> block = ConvBlock(features=64)
    >>> block = ConvBlock(features=64, pooling='SPATIAL_MAX',
    ...                   dropout_rate=0.1)
    """
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    padding: str = "SAME"
    use_bias: bool = False
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    pre_norm: bool = False
    pooling: Optional[str] = None
    pooling_kwargs: Optional[dict] = None
    dropout_rate: float = 0.0

    def setup(self):
        from core.pooling import get_pooling
        kernel_init = get_initializer(self.initializer)
        self.conv = nn.Conv(
            self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
        )
        self.norm_layer = get_norm(self.norm, **(self.norm_kwargs or {}))
        self.act = get_activation(self.activation, **(self.activation_kwargs or {}))
        self.pool = (
            get_pooling(self.pooling, **(self.pooling_kwargs or {}))
            if self.pooling is not None else None
        )
        self.drop = nn.Dropout(
            rate=self.dropout_rate,
            broadcast_dims=(1, 2),  # spatial dropout -- zero whole feature maps
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        if self.pre_norm:
            x = _call_norm(self.norm_layer, x, train)
            x = self.act(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = _call_norm(self.norm_layer, x, train)
            x = self.act(x)

        if self.pool is not None:
            x = self.pool(x)

        x = self.drop(x, deterministic=not train)
        return x


class ResidualBlock(nn.Module):
    """Residual block with two convolutions, skip connection, optional
    pooling and spatial dropout.

    Supports both post-activation and pre-activation orderings. Handles
    channel mismatch via a 1x1 projection on the skip path.

    Parameters
    ----------
    features : int
        Number of output channels.
    norm : str
        Registered norm name. Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.
    use_bias : bool
        Default False.
    pre_norm : bool
        If True uses pre-activation ordering (He et al. 2016 v2).
        Default False.
    pooling : str, optional
        Registered pooling applied after the residual add. Default None.
    pooling_kwargs : dict, optional
    dropout_rate : float
        Spatial dropout rate applied after the activation (post-norm) or
        after the second conv (pre-norm), before the skip add.
        Default 0.0.

    Notes
    -----
    Input channels are inferred from x at first call. Channel mismatch
    between input and output is handled by a 1x1 projection on the skip path.

    When dropout_rate > 0, pass rngs={'dropout': key} to apply.

    Example
    -------
    >>> block = ResidualBlock(features=128)
    >>> block = ResidualBlock(features=128, dropout_rate=0.1,
    ...                       pooling='SPATIAL_AVG')
    """
    features: int
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    use_bias: bool = False
    pre_norm: bool = False
    pooling: Optional[str] = None
    pooling_kwargs: Optional[dict] = None
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        from core.pooling import get_pooling
        kernel_init = get_initializer(self.initializer)
        norm_kw = self.norm_kwargs or {}
        act = get_activation(self.activation, **(self.activation_kwargs or {}))
        drop = nn.Dropout(rate=self.dropout_rate, broadcast_dims=(1, 2))
        in_features = x.shape[-1]

        residual = x

        if self.pre_norm:
            x = _call_norm(get_norm(self.norm, **norm_kw), x, train)
            x = act(x)
            x = nn.Conv(self.features, (3, 3), padding="SAME",
                        use_bias=self.use_bias, kernel_init=kernel_init)(x)
            x = _call_norm(get_norm(self.norm, **norm_kw), x, train)
            x = act(x)
            x = nn.Conv(self.features, (3, 3), padding="SAME",
                        use_bias=self.use_bias, kernel_init=kernel_init)(x)
            x = drop(x, deterministic=not train)
        else:
            x = nn.Conv(self.features, (3, 3), padding="SAME",
                        use_bias=self.use_bias, kernel_init=kernel_init)(x)
            x = _call_norm(get_norm(self.norm, **norm_kw), x, train)
            x = act(x)
            x = drop(x, deterministic=not train)
            x = nn.Conv(self.features, (3, 3), padding="SAME",
                        use_bias=self.use_bias, kernel_init=kernel_init)(x)
            x = _call_norm(get_norm(self.norm, **norm_kw), x, train)

        if in_features != self.features:
            residual = nn.Conv(self.features, (1, 1), padding="SAME",
                               use_bias=False, kernel_init=kernel_init)(residual)

        out = x + residual

        if not self.pre_norm:
            out = act(out)

        if self.pooling is not None:
            pool = get_pooling(self.pooling, **(self.pooling_kwargs or {}))
            out = pool(out)

        return out


class DownsampleBlock(nn.Module):
    """Spatial downsampling by factor 2.

    Supports strided conv or pooling-based downsampling.

    Parameters
    ----------
    features : int
        Number of output channels.
    padding_mode : str
        For strided conv only.
        'asymmetric' -- pads (0,1) on H and W before a stride-2 conv with
        padding='VALID'. Guarantees exact spatial alignment in encoder/decoder
        pairs (VQVAE, UNet). Recommended when a UpsampleBlock counterpart exists.
        'same' -- uses padding='SAME' on the stride-2 conv. Simpler, correct
        for pure encoders without reconstruction alignment requirements.
        Default 'asymmetric'.
    pool_type : str, optional
        If set, uses pooling for downsampling instead of strided conv.
        Use 'SPATIAL_MAX' or 'SPATIAL_AVG'. When set, padding_mode is
        ignored and a 1x1 conv adjusts channels after pooling.
        Default None (strided conv).
    use_bias : bool
        Default False.
    initializer : str
        Default 'lecun_normal'.

    Notes
    -----
    Channels-last input: (B, H, W, C) -> (B, H//2, W//2, features).

    For odd spatial dimensions, padding_mode='same' preserves more spatial
    information (ceil(H/2)) while 'asymmetric' floors (floor(H/2)).
    For even spatial dimensions both give H//2, but 'asymmetric' guarantees
    exact alignment with UpsampleBlock in encoder/decoder pairs.
    Use 'same' if your input spatial dimensions may be odd.

    Example
    -------
    >>> block = DownsampleBlock(features=128)                    # strided conv
    >>> block = DownsampleBlock(features=128, padding_mode='same')
    >>> block = DownsampleBlock(features=128, pool_type='SPATIAL_AVG')
    """
    features: int
    padding_mode: str = "asymmetric"
    pool_type: Optional[str] = None
    use_bias: bool = False
    initializer: str = "lecun_normal"

    def setup(self):
        from core.pooling import get_pooling
        kernel_init = get_initializer(self.initializer)
        if self.pool_type is not None:
            self.pool = get_pooling(self.pool_type, kernel_size=(2, 2),
                                    strides=(2, 2))
            self.proj = nn.Conv(self.features, (1, 1), use_bias=self.use_bias,
                                kernel_init=kernel_init)
        else:
            padding = "VALID" if self.padding_mode == "asymmetric" else "SAME"
            self.conv = nn.Conv(
                self.features,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=padding,
                use_bias=self.use_bias,
                kernel_init=kernel_init,
            )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        if self.pool_type is not None:
            x = self.pool(x)
            return self.proj(x)
        if self.padding_mode == "asymmetric":
            x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """Spatial upsampling by factor 2 via bilinear interpolation + conv.

    Parameters
    ----------
    features : int
        Number of output channels.
    use_bias : bool
        Default False.
    initializer : str
        Default 'lecun_normal'.

    Notes
    -----
    Channels-last input: (B, H, W, C) -> (B, H*2, W*2, features).

    Example
    -------
    >>> block = UpsampleBlock(features=64)
    """
    features: int
    use_bias: bool = False
    initializer: str = "lecun_normal"

    def setup(self):
        kernel_init = get_initializer(self.initializer)
        self.conv = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=self.use_bias,
            kernel_init=kernel_init,
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        B, H, W, C = x.shape
        x = jax.image.resize(x, shape=(B, H * 2, W * 2, C), method="bilinear")
        return self.conv(x)


class NonLocalBlock(nn.Module):
    """Spatial self-attention block (non-local means).

    Parameters
    ----------
    norm : str
        Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
    downsample_factor : int, optional
        Reduces spatial resolution of keys and values before attention.
        Reduces memory from O(H²W²) to O((H/f)²(W/f)²). Default None.
    dropout_rate : float
        Dropout applied to attention weights. Default 0.0.
    initializer : str
        Default 'lecun_normal'.

    Notes
    -----
    Channels-last: (B, H, W, C) -> (B, H, W, C).
    Full resolution attention is O(H²W²). For H*W > 1024 consider
    setting downsample_factor=2 or 4.

    Example
    -------
    >>> block = NonLocalBlock()
    >>> block = NonLocalBlock(downsample_factor=2, dropout_rate=0.1)
    """
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    downsample_factor: Optional[int] = None
    dropout_rate: float = 0.0
    initializer: str = "lecun_normal"

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        kernel_init = get_initializer(self.initializer)
        norm_kw = self.norm_kwargs or {}
        B, H, W, C = x.shape

        h = _call_norm(get_norm(self.norm, **norm_kw), x, train)

        if self.downsample_factor is not None:
            f = self.downsample_factor
            h_ds = jax.image.resize(
                h, shape=(B, H // f, W // f, C), method="bilinear"
            )
        else:
            h_ds = h

        q = nn.Conv(C, (1, 1), use_bias=False, kernel_init=kernel_init)(h)
        k = nn.Conv(C, (1, 1), use_bias=False, kernel_init=kernel_init)(h_ds)
        v = nn.Conv(C, (1, 1), use_bias=False, kernel_init=kernel_init)(h_ds)

        q = q.reshape(B, H * W, C)
        k = k.reshape(B, -1, C)
        v = v.reshape(B, -1, C)

        scale = C ** -0.5
        attn = jnp.einsum('bic,bjc->bij', q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.dropout_rate)(
            attn, deterministic=not train
        )

        out = jnp.einsum('bij,bjc->bic', attn, v)
        out = out.reshape(B, H, W, C)
        out = nn.Conv(C, (1, 1), use_bias=False, kernel_init=kernel_init)(out)

        return x + out


# ---------------------------------------------------------------------------
# Inception block
# ---------------------------------------------------------------------------

class InceptionBlock(nn.Module):
    """Inception block with four parallel branches.

    Parameters
    ----------
    c_red : dict
        Reduced channel sizes. Keys: '3x3', '5x5'.
    c_out : dict
        Output channel sizes. Keys: '1x1', '3x3', '5x5', 'max'.
    norm : str
        Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.
    dropout_rate : float
        Spatial dropout applied after concatenation. Default 0.0.

    Notes
    -----
    Output channels = sum(c_out.values()).
    Channels-last: (B, H, W, C) -> (B, H, W, sum(c_out.values())).

    Example
    -------
    >>> block = InceptionBlock(
    ...     c_red={'3x3': 32, '5x5': 16},
    ...     c_out={'1x1': 16, '3x3': 32, '5x5': 8, 'max': 8},
    ... )
    """
    c_red: dict
    c_out: dict
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        kernel_init = get_initializer(self.initializer)
        norm_kw = self.norm_kwargs or {}
        act = get_activation(self.activation, **(self.activation_kwargs or {}))

        def conv_norm_act(inp, features, kernel):
            h = nn.Conv(features, kernel, padding="SAME",
                        use_bias=False, kernel_init=kernel_init)(inp)
            h = _call_norm(get_norm(self.norm, **norm_kw), h, train)
            return act(h)

        x_1x1 = conv_norm_act(x, self.c_out["1x1"], (1, 1))

        x_3x3 = conv_norm_act(x, self.c_red["3x3"], (1, 1))
        x_3x3 = conv_norm_act(x_3x3, self.c_out["3x3"], (3, 3))

        x_5x5 = conv_norm_act(x, self.c_red["5x5"], (1, 1))
        x_5x5 = conv_norm_act(x_5x5, self.c_out["5x5"], (5, 5))

        x_max = nn.max_pool(x, (3, 3), strides=(1, 1), padding="SAME")
        x_max = conv_norm_act(x_max, self.c_out["max"], (1, 1))

        out = jnp.concatenate([x_1x1, x_3x3, x_5x5, x_max], axis=-1)
        out = nn.Dropout(rate=self.dropout_rate, broadcast_dims=(1, 2))(
            out, deterministic=not train
        )
        return out


# ---------------------------------------------------------------------------
# DenseNet blocks
# ---------------------------------------------------------------------------

class DenseLayer(nn.Module):
    """Single layer in a DenseBlock.

    BN -> act -> 1x1 conv -> BN -> act -> 3x3 conv -> concat.

    Parameters
    ----------
    growth_rate : int
        Channels added per layer.
    bn_size : int
        Bottleneck factor. Default 4.
    norm : str
        Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.
    dropout_rate : float
        Spatial dropout after 3x3 conv, before concat. Default 0.0.

    Notes
    -----
    (B, H, W, C) -> (B, H, W, C + growth_rate).

    Example
    -------
    >>> layer = DenseLayer(growth_rate=16)
    """
    growth_rate: int
    bn_size: int = 4
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        kernel_init = get_initializer(self.initializer)
        norm_kw = self.norm_kwargs or {}
        act = get_activation(self.activation, **(self.activation_kwargs or {}))

        z = _call_norm(get_norm(self.norm, **norm_kw), x, train)
        z = act(z)
        z = nn.Conv(self.bn_size * self.growth_rate, (1, 1),
                    use_bias=False, kernel_init=kernel_init)(z)
        z = _call_norm(get_norm(self.norm, **norm_kw), z, train)
        z = act(z)
        z = nn.Conv(self.growth_rate, (3, 3), padding="SAME",
                    use_bias=False, kernel_init=kernel_init)(z)
        z = nn.Dropout(rate=self.dropout_rate, broadcast_dims=(1, 2))(
            z, deterministic=not train
        )
        return jnp.concatenate([x, z], axis=-1)


class DenseBlock(nn.Module):
    """Stack of DenseLayers with growing channel concatenation.

    Parameters
    ----------
    num_layers : int
    growth_rate : int
    bn_size : int
        Default 4.
    norm : str
        Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.
    dropout_rate : float
        Forwarded to each DenseLayer. Default 0.0.

    Notes
    -----
    Output channels = input_channels + num_layers * growth_rate.

    When norm='GROUP_NORM', both input_channels and growth_rate must be
    divisible by num_groups (default 8). growth_rate is checked at
    construction time; input_channels are checked on the first forward pass.

    Example
    -------
    >>> block = DenseBlock(num_layers=6, growth_rate=16)
    """
    num_layers: int
    growth_rate: int
    bn_size: int = 4
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    dropout_rate: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.norm.upper() == "GROUP_NORM":
            num_groups = (self.norm_kwargs or {}).get("num_groups", 8)
            if self.growth_rate % num_groups != 0:
                raise ValueError(
                    f"DenseBlock: growth_rate={self.growth_rate} is not divisible by "
                    f"num_groups={num_groups}. All intermediate channel counts "
                    f"(C_in + k * growth_rate) require both C_in and growth_rate to be "
                    f"divisible by num_groups. Use a growth_rate divisible by {num_groups}, "
                    f"or switch to a norm without group constraints (e.g. LAYER_NORM)."
                )

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        if self.norm.upper() == "GROUP_NORM":
            num_groups = (self.norm_kwargs or {}).get("num_groups", 8)
            if x.shape[-1] % num_groups != 0:
                raise ValueError(
                    f"DenseBlock: input channels={x.shape[-1]} is not divisible by "
                    f"num_groups={num_groups}. Adjust the preceding layer's output "
                    f"channels or switch norm (e.g. LAYER_NORM)."
                )
        for _ in range(self.num_layers):
            x = DenseLayer(
                growth_rate=self.growth_rate,
                bn_size=self.bn_size,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                activation=self.activation,
                activation_kwargs=self.activation_kwargs,
                initializer=self.initializer,
                dropout_rate=self.dropout_rate,
            )(x, train=train)
        return x

class TransitionLayer(nn.Module):
    """Transition layer between DenseBlocks.

    Reduces channels via 1x1 conv and spatial resolution via pooling.

    Parameters
    ----------
    features : int
        Output channel count. Typically half the input channels.
    pool_type : str
        Pooling for spatial downsampling. 'SPATIAL_AVG' or 'SPATIAL_MAX'.
        Default 'SPATIAL_AVG'.
    norm : str
        Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.

    Notes
    -----
    (B, H, W, C) -> (B, H//2, W//2, features).

    Example
    -------
    >>> layer = TransitionLayer(features=64)
    >>> layer = TransitionLayer(features=64, pool_type='SPATIAL_MAX')
    """
    features: int
    pool_type: str = "SPATIAL_AVG"
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"

    def setup(self):
        from core.pooling import get_pooling
        kernel_init = get_initializer(self.initializer)
        self.conv = nn.Conv(self.features, (1, 1), use_bias=False,
                            kernel_init=kernel_init)
        self.pool = get_pooling(self.pool_type, kernel_size=(2, 2),
                                strides=(2, 2))
        self.norm_layer = get_norm(self.norm, **(self.norm_kwargs or {}))
        self.act = get_activation(self.activation, **(self.activation_kwargs or {}))

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        x = _call_norm(self.norm_layer, x, train)
        x = self.act(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class PatchEmbed(nn.Module):
    """Conv-based patch embedding for Vision Transformers.

    Splits a spatial image into non-overlapping patches using a strided
    convolution and projects each patch to embed_dim.

    Parameters
    ----------
    patch_size : int
        Side length P of each square patch. H and W must be divisible
        by patch_size.
    embed_dim : int
        Output embedding dimension per patch.
    flatten : bool
        If True, returns (B, H//P * W//P, embed_dim) sequence format.
        If False, returns (B, H//P, W//P, embed_dim) spatial grid format.
        Use True for ViT (sequence attention), False for Swin (window
        attention requires spatial layout). Default True.
    use_bias : bool
        Whether to include a bias in the projection. Default True.
    initializer : str
        Kernel initializer. Default 'lecun_normal'.

    Notes
    -----
    Input:  (B, H, W, C)
    Output: (B, H//P * W//P, embed_dim)  if flatten=True
            (B, H//P, W//P, embed_dim)   if flatten=False

    H and W must be divisible by patch_size. Validated at call time before
    JAX traces the function.

    No norm, activation, or positional encoding -- PatchEmbed is a pure
    linear projection. Norm and positional encoding are the responsibility
    of the calling transformer.

    Example
    -------
    >>> # ViT usage -- sequence format
    >>> embed = PatchEmbed(patch_size=16, embed_dim=768)
    >>> out = embed.apply(variables, jnp.ones((2, 224, 224, 3)))
    >>> out.shape
    (2, 196, 768)

    >>> # Swin usage -- spatial grid format
    >>> embed = PatchEmbed(patch_size=4, embed_dim=96, flatten=False)
    >>> out = embed.apply(variables, jnp.ones((2, 224, 224, 3)))
    >>> out.shape
    (2, 56, 56, 96)
    """
    patch_size:  int
    embed_dim:   int
    flatten:     bool = True
    use_bias:    bool = True
    initializer: str  = "lecun_normal"

    def setup(self):
        self.proj = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=self.use_bias,
            kernel_init=get_initializer(self.initializer),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (B, H, W, C). H and W must be divisible by patch_size.

        Returns
        -------
        jax.Array
            Shape (B, H//P * W//P, embed_dim) if flatten=True,
            or    (B, H//P, W//P, embed_dim)  if flatten=False.
        """
        assert x.ndim == 4, (
            f"PatchEmbed: expected 4D input (B, H, W, C), got shape {x.shape}."
        )
        with jax.ensure_compile_time_eval():
            H_int, W_int = int(x.shape[1]), int(x.shape[2])
            P = self.patch_size
            if H_int % P != 0 or W_int % P != 0:
                raise ValueError(
                    f"PatchEmbed: H={H_int}, W={W_int} must both be "
                    f"divisible by patch_size={P}."
                )
        B = x.shape[0]
        x = self.proj(x)   # (B, H//P, W//P, embed_dim)
        if self.flatten:
            x = x.reshape(B, -1, self.embed_dim)  # (B, T, embed_dim)
        return x

# ---------------------------------------------------------------------------
# 1D Blocks
# ---------------------------------------------------------------------------

class ConvBlock1d(nn.Module):
    """Single 1D conv layer with norm, activation, optional pooling and dropout.

    Parameters
    ----------
    features : int
    kernel_size : int
        Default 3.
    strides : int
        Default 1.
    padding : str
        Default 'SAME'.
    use_bias : bool
        Default False.
    norm : str
        Default 'LAYER_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.
    pre_norm : bool
        Default False.
    dropout_rate : float
        Standard (not spatial) dropout for 1D sequences. Default 0.0.

    Notes
    -----
    Input: (B, L, C) channels-last.

    Example
    -------
    >>> block = ConvBlock1d(features=64)
    >>> block = ConvBlock1d(features=64, dropout_rate=0.1)
    """
    features: int
    kernel_size: int = 3
    strides: int = 1
    padding: str = "SAME"
    use_bias: bool = False
    norm: str = "LAYER_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    pre_norm: bool = False
    dropout_rate: float = 0.0

    def setup(self):
        kernel_init = get_initializer(self.initializer)
        self.conv = nn.Conv(
            self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
        )
        self.norm_layer = get_norm(self.norm, **(self.norm_kwargs or {}))
        self.act = get_activation(self.activation, **(self.activation_kwargs or {}))
        self.drop = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        if self.pre_norm:
            x = _call_norm(self.norm_layer, x, train)
            x = self.act(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = _call_norm(self.norm_layer, x, train)
            x = self.act(x)
        x = self.drop(x, deterministic=not train)
        return x


class ResidualBlock1d(nn.Module):
    """1D residual block with optional dropout.

    Parameters
    ----------
    features : int
    norm : str
        Default 'LAYER_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.
    use_bias : bool
        Default False.
    pre_norm : bool
        Default False.
    dropout_rate : float
        Default 0.0.

    Notes
    -----
    Input: (B, L, C) -> (B, L, features).

    Example
    -------
    >>> block = ResidualBlock1d(features=64, dropout_rate=0.1)
    """
    features: int
    norm: str = "LAYER_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    use_bias: bool = False
    pre_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        kernel_init = get_initializer(self.initializer)
        norm_kw = self.norm_kwargs or {}
        act = get_activation(self.activation, **(self.activation_kwargs or {}))
        drop = nn.Dropout(rate=self.dropout_rate)
        in_features = x.shape[-1]
        residual = x

        if self.pre_norm:
            x = _call_norm(get_norm(self.norm, **norm_kw), x, train)
            x = act(x)
            x = nn.Conv(self.features, (3,), padding="SAME",
                        use_bias=self.use_bias, kernel_init=kernel_init)(x)
            x = _call_norm(get_norm(self.norm, **norm_kw), x, train)
            x = act(x)
            x = nn.Conv(self.features, (3,), padding="SAME",
                        use_bias=self.use_bias, kernel_init=kernel_init)(x)
            x = drop(x, deterministic=not train)
        else:
            x = nn.Conv(self.features, (3,), padding="SAME",
                        use_bias=self.use_bias, kernel_init=kernel_init)(x)
            x = _call_norm(get_norm(self.norm, **norm_kw), x, train)
            x = act(x)
            x = drop(x, deterministic=not train)
            x = nn.Conv(self.features, (3,), padding="SAME",
                        use_bias=self.use_bias, kernel_init=kernel_init)(x)
            x = _call_norm(get_norm(self.norm, **norm_kw), x, train)

        if in_features != self.features:
            residual = nn.Conv(self.features, (1,), padding="SAME",
                               use_bias=False, kernel_init=kernel_init)(residual)

        out = x + residual
        if not self.pre_norm:
            out = act(out)
        return out


class DownsampleBlock1d(nn.Module):
    """1D downsampling by factor 2 via stride-2 conv.

    Parameters
    ----------
    features : int
    use_bias : bool
        Default False.
    initializer : str
        Default 'lecun_normal'.

    Notes
    -----
    Input: (B, L, C) -> (B, L//2, features).

    Example
    -------
    >>> block = DownsampleBlock1d(features=128)
    """
    features: int
    use_bias: bool = False
    initializer: str = "lecun_normal"

    def setup(self):
        kernel_init = get_initializer(self.initializer)
        self.conv = nn.Conv(
            self.features,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            use_bias=self.use_bias,
            kernel_init=kernel_init,
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        return self.conv(x)


class UpsampleBlock1d(nn.Module):
    """1D upsampling by factor 2 via linear interpolation + conv.

    Parameters
    ----------
    features : int
    use_bias : bool
        Default False.
    initializer : str
        Default 'lecun_normal'.

    Notes
    -----
    Input: (B, L, C) -> (B, L*2, features).

    Example
    -------
    >>> block = UpsampleBlock1d(features=64)
    """
    features: int
    use_bias: bool = False
    initializer: str = "lecun_normal"

    def setup(self):
        kernel_init = get_initializer(self.initializer)
        self.conv = nn.Conv(
            self.features,
            kernel_size=(3,),
            strides=(1,),
            padding="SAME",
            use_bias=self.use_bias,
            kernel_init=kernel_init,
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        B, L, C = x.shape
        x = jax.image.resize(x, shape=(B, L * 2, C), method="linear")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Registered nets
# ---------------------------------------------------------------------------

@register_conv_net("CONV_ENCODER",
                   description="Conv encoder: ResidualBlocks + DownsampleBlocks")
class ConvEncoder(nn.Module):
    """Convolutional encoder using ResidualBlocks and DownsampleBlocks.

    Parameters
    ----------
    channels : tuple of int
        Output channels at each resolution level.
    num_res_blocks : int
        Residual blocks per level. Default 2.
    norm : str
        Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.
    downsample_padding : str
        'asymmetric' or 'same'. Default 'asymmetric'.
    downsample_pool_type : str, optional
        If set, uses pooling instead of strided conv for downsampling.
        Default None.
    use_non_local : bool
        Insert NonLocalBlock after last resolution level. Default False.
    non_local_downsample : int, optional
        downsample_factor for NonLocalBlock. Default None.
    pre_norm : bool
        Default False.
    dropout_rate : float
        Spatial dropout forwarded to each ResidualBlock. Default 0.0.

    Notes
    -----
    Input: (B, H, W, C) -> (B, H // 2^(n-1), W // 2^(n-1), channels[-1])
    where n = len(channels).

    Example
    -------
    >>> enc = ConvEncoder(channels=(64, 128, 256), num_res_blocks=2,
    ...                   dropout_rate=0.1)
    >>> variables = enc.init(jax.random.PRNGKey(0),
    ...                      jnp.ones((2, 64, 64, 3)), train=True)
    >>> out, updates = enc.apply(variables, jnp.ones((2, 64, 64, 3)),
    ...                          train=True, mutable=['batch_stats'],
    ...                          rngs={'dropout': jax.random.PRNGKey(1)})
    """
    channels: tuple
    num_res_blocks: int = 2
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    downsample_padding: str = "asymmetric"
    downsample_pool_type: Optional[str] = None
    use_non_local: bool = False
    non_local_downsample: Optional[int] = None
    pre_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        norm_kw = self.norm_kwargs or {}
        act_kw = self.activation_kwargs or {}

        for i, c in enumerate(self.channels):
            for _ in range(self.num_res_blocks):
                x = ResidualBlock(
                    features=c,
                    norm=self.norm, norm_kwargs=norm_kw,
                    activation=self.activation, activation_kwargs=act_kw,
                    initializer=self.initializer,
                    pre_norm=self.pre_norm,
                    dropout_rate=self.dropout_rate,
                )(x, train=train)

            if i < len(self.channels) - 1:
                x = DownsampleBlock(
                    features=c,
                    padding_mode=self.downsample_padding,
                    pool_type=self.downsample_pool_type,
                    initializer=self.initializer,
                )(x, train=train)

        if self.use_non_local:
            x = NonLocalBlock(
                norm=self.norm, norm_kwargs=norm_kw,
                downsample_factor=self.non_local_downsample,
                initializer=self.initializer,
            )(x, train=train)

        return x


@register_conv_net("CONV_DECODER",
                   description="Conv decoder: ResidualBlocks + UpsampleBlocks")
class ConvDecoder(nn.Module):
    """Convolutional decoder using ResidualBlocks and UpsampleBlocks.

    Parameters
    ----------
    channels : tuple of int
        Output channels at each resolution level.
    num_res_blocks : int
        Default 2.
    out_features : int, optional
        Final 1x1 conv to project to this channel count. Default None.
    norm : str
        Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.
    use_non_local : bool
        Insert NonLocalBlock before first upsample. Default False.
    non_local_downsample : int, optional
        Default None.
    pre_norm : bool
        Default False.
    dropout_rate : float
        Spatial dropout forwarded to each ResidualBlock. Default 0.0.

    Notes
    -----
    Input: (B, H, W, C) -> (B, H * 2^(n-1), W * 2^(n-1), channels[-1]).

    Example
    -------
    >>> dec = ConvDecoder(channels=(256, 128, 64), out_features=3,
    ...                   dropout_rate=0.1)
    >>> variables = dec.init(jax.random.PRNGKey(0),
    ...                      jnp.ones((2, 16, 16, 256)), train=True)
    >>> out, updates = dec.apply(variables, jnp.ones((2, 16, 16, 256)),
    ...                          train=True, mutable=['batch_stats'],
    ...                          rngs={'dropout': jax.random.PRNGKey(1)})
    """
    channels: tuple
    num_res_blocks: int = 2
    out_features: Optional[int] = None
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    use_non_local: bool = False
    non_local_downsample: Optional[int] = None
    pre_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        norm_kw = self.norm_kwargs or {}
        act_kw = self.activation_kwargs or {}
        kernel_init = get_initializer(self.initializer)

        if self.use_non_local:
            x = NonLocalBlock(
                norm=self.norm, norm_kwargs=norm_kw,
                downsample_factor=self.non_local_downsample,
                initializer=self.initializer,
            )(x, train=train)

        for i, c in enumerate(self.channels):
            for _ in range(self.num_res_blocks):
                x = ResidualBlock(
                    features=c,
                    norm=self.norm, norm_kwargs=norm_kw,
                    activation=self.activation, activation_kwargs=act_kw,
                    initializer=self.initializer,
                    pre_norm=self.pre_norm,
                    dropout_rate=self.dropout_rate,
                )(x, train=train)

            if i < len(self.channels) - 1:
                x = UpsampleBlock(
                    features=c,
                    initializer=self.initializer,
                )(x, train=train)

        if self.out_features is not None:
            x = nn.Conv(self.out_features, (1, 1), use_bias=True,
                        kernel_init=kernel_init)(x)

        return x


@register_conv_net("RESNET",
                   description="ResNet with configurable blocks and norm (He et al. 2016)")
class ResNet(nn.Module):
    """ResNet with configurable block type, norm, activation, and dropout.

    Parameters
    ----------
    num_classes : int
    c_hidden : tuple of int
        Default (64, 128, 256).
    num_blocks : tuple of int
        Default (3, 3, 3).
    pre_norm : bool
        Default False.
    norm : str
        Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.
    dropout_rate : float
        Spatial dropout forwarded to each ResidualBlock. Default 0.0.

    Notes
    -----
    Input: (B, H, W, C) channels-last.
    Downsampling at first block of each group except the first.

    Example
    -------
    >>> net = ResNet(num_classes=10, dropout_rate=0.1)
    >>> variables = net.init(jax.random.PRNGKey(0),
    ...                      jnp.ones((2, 32, 32, 3)), train=True)
    >>> out, updates = net.apply(variables, jnp.ones((2, 32, 32, 3)),
    ...                          train=True, mutable=['batch_stats'],
    ...                          rngs={'dropout': jax.random.PRNGKey(1)})
    """
    num_classes: int
    c_hidden: tuple = (64, 128, 256)
    num_blocks: tuple = (3, 3, 3)
    pre_norm: bool = False
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        norm_kw = self.norm_kwargs or {}
        act_kw = self.activation_kwargs or {}
        kernel_init = get_initializer(self.initializer)
        act = get_activation(self.activation, **act_kw)

        x = nn.Conv(self.c_hidden[0], (3, 3), padding="SAME",
                    use_bias=False, kernel_init=kernel_init)(x)
        if not self.pre_norm:
            x = _call_norm(get_norm(self.norm, **norm_kw), x, train)
            x = act(x)

        for group_idx, (c, n) in enumerate(zip(self.c_hidden, self.num_blocks)):
            for block_idx in range(n):
                subsample = (block_idx == 0 and group_idx > 0)
                if subsample:
                    x = DownsampleBlock(
                        features=c,
                        padding_mode="asymmetric",
                        initializer=self.initializer,
                    )(x, train=train)
                x = ResidualBlock(
                    features=c,
                    norm=self.norm, norm_kwargs=norm_kw,
                    activation=self.activation, activation_kwargs=act_kw,
                    initializer=self.initializer,
                    pre_norm=self.pre_norm,
                    dropout_rate=self.dropout_rate,
                )(x, train=train)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, kernel_init=kernel_init)(x)
        return x


@register_conv_net("DENSENET",
                   description="DenseNet with configurable blocks and norm (Huang et al. 2017)")
class DenseNet(nn.Module):
    """DenseNet with configurable norm, activation, and dropout.

    Parameters
    ----------
    num_classes : int
    num_layers : tuple of int
        Default (6, 6, 6, 6).
    growth_rate : int
        Default 16.
    bn_size : int
        Default 4.
    norm : str
        Default 'GROUP_NORM'.
    norm_kwargs : dict, optional
    activation : str
        Default 'silu'.
    activation_kwargs : dict, optional
    initializer : str
        Default 'lecun_normal'.
    transition_pool_type : str
        Pooling in TransitionLayer. Default 'SPATIAL_AVG'.
    dropout_rate : float
        Forwarded to each DenseLayer. Default 0.0.

    Notes
    -----
    Input: (B, H, W, C) channels-last.

    Example
    -------
    >>> net = DenseNet(num_classes=10, dropout_rate=0.1)
    >>> variables = net.init(jax.random.PRNGKey(0),
    ...                      jnp.ones((2, 32, 32, 3)), train=True)
    >>> out, updates = net.apply(variables, jnp.ones((2, 32, 32, 3)),
    ...                          train=True, mutable=['batch_stats'],
    ...                          rngs={'dropout': jax.random.PRNGKey(1)})
    """
    num_classes: int
    num_layers: tuple = (6, 6, 6, 6)
    growth_rate: int = 16
    bn_size: int = 4
    norm: str = "GROUP_NORM"
    norm_kwargs: Optional[dict] = None
    activation: str = "silu"
    activation_kwargs: Optional[dict] = None
    initializer: str = "lecun_normal"
    transition_pool_type: str = "SPATIAL_AVG"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        norm_kw = self.norm_kwargs or {}
        act_kw = self.activation_kwargs or {}
        kernel_init = get_initializer(self.initializer)
        act = get_activation(self.activation, **act_kw)

        c_hidden = self.growth_rate * self.bn_size
        x = nn.Conv(c_hidden, (3, 3), padding="SAME",
                    use_bias=False, kernel_init=kernel_init)(x)

        for block_idx, n in enumerate(self.num_layers):
            x = DenseBlock(
                num_layers=n,
                growth_rate=self.growth_rate,
                bn_size=self.bn_size,
                norm=self.norm, norm_kwargs=norm_kw,
                activation=self.activation, activation_kwargs=act_kw,
                initializer=self.initializer,
                dropout_rate=self.dropout_rate,
            )(x, train=train)
            c_hidden += n * self.growth_rate

            if block_idx < len(self.num_layers) - 1:
                x = TransitionLayer(
                    features=c_hidden // 2,
                    pool_type=self.transition_pool_type,
                    norm=self.norm, norm_kwargs=norm_kw,
                    activation=self.activation, activation_kwargs=act_kw,
                    initializer=self.initializer,
                )(x, train=train)
                c_hidden //= 2

        x = _call_norm(get_norm(self.norm, **norm_kw), x, train)
        x = act(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, kernel_init=kernel_init)(x)
        return x