# `nets/conv.py`

Convolutional building blocks and registered conv net architectures. All modules are `flax.linen.Module` subclasses. Inputs are channels-last: `(B, H, W, C)` for 2D and `(B, L, C)` for 1D. Norm, activation, and initializer names are all resolved via registries in `core`.

---

## Registry Functions

---

### `register_conv_net`

```
register_conv_net(name: str, description: str = "") -> callable
```

Class decorator. Names stored uppercase, must be unique.

---

### `get_conv_net`

```
get_conv_net(name: str, **kwargs) -> nn.Module
```

Retrieve and instantiate a registered conv net by name. Uses `__dataclass_fields__` for kwarg inspection. Unknown kwargs trigger a `UserWarning` and are dropped.

**Raises:** `ValueError` if no net with the given name exists.

---

### `list_conv_nets`

```
list_conv_nets() -> dict[str, str]
```

Return a sorted dictionary of all registered conv net names and their descriptions.

---

## 2D Building Blocks

---

### `ConvBlock`

```python
ConvBlock(
    features: int,
    kernel_size: tuple = (3, 3),
    strides: tuple = (1, 1),
    padding: str = "SAME",
    use_bias: bool = False,
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    activation: str = "silu",
    activation_kwargs: Optional[dict] = None,
    initializer: str = "lecun_normal",
    pre_norm: bool = False,
    pooling: Optional[str] = None,
    pooling_kwargs: Optional[dict] = None,
    dropout_rate: float = 0.0,
)
```

Single conv layer with norm, activation, optional spatial pooling, and spatial dropout (zeros entire feature maps).

**Orderings:**
- Post-norm (default): `conv → norm → act → pool → drop`
- Pre-norm (`pre_norm=True`): `norm → act → conv → pool → drop`

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `features` | `int` | Output channels | required |
| `kernel_size` | `tuple` | Convolution kernel size | `(3, 3)` |
| `strides` | `tuple` | Convolution stride | `(1, 1)` |
| `padding` | `str` | Padding mode | `"SAME"` |
| `use_bias` | `bool` | | `False` |
| `norm` | `str` | Registered norm name | `"GROUP_NORM"` |
| `norm_kwargs` | `dict`, optional | Forwarded to `get_norm` | `None` |
| `activation` | `str` | Registered activation name | `"silu"` |
| `activation_kwargs` | `dict`, optional | | `None` |
| `initializer` | `str` | Kernel initializer name | `"lecun_normal"` |
| `pre_norm` | `bool` | If `True`, use pre-norm ordering | `False` |
| `pooling` | `str`, optional | Registered pooling name applied after norm+act | `None` |
| `pooling_kwargs` | `dict`, optional | Forwarded to `get_pooling` | `None` |
| `dropout_rate` | `float` | Spatial dropout rate (broadcast over H, W). Requires `rngs={'dropout': key}` when > 0 | `0.0` |

---

### `ResidualBlock`

```python
ResidualBlock(
    features: int,
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    activation: str = "silu",
    activation_kwargs: Optional[dict] = None,
    initializer: str = "lecun_normal",
    use_bias: bool = False,
    pre_norm: bool = False,
    pooling: Optional[str] = None,
    pooling_kwargs: Optional[dict] = None,
    dropout_rate: float = 0.0,
)
```

Two-conv residual block with skip connection. Handles channel mismatch via a 1×1 projection on the skip path (inferred at call time from input channels).

**Orderings:**
- Post-activation (default `He et al. 2016 v1`): `conv → norm → act → drop → conv → norm → add → act`
- Pre-activation (`pre_norm=True`, `He et al. 2016 v2`): `norm → act → conv → norm → act → conv → drop → add`

Pooling is applied after the residual add.

---

### `DownsampleBlock`

```python
DownsampleBlock(
    features: int,
    padding_mode: str = "asymmetric",
    pool_type: Optional[str] = None,
    use_bias: bool = False,
    initializer: str = "lecun_normal",
)
```

Spatial downsampling by factor 2. `(B, H, W, C) → (B, H//2, W//2, features)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `features` | `int` | Output channels | required |
| `padding_mode` | `str` | `'asymmetric'` — pads `(0,1)` on H and W before stride-2 conv with `padding='VALID'`. Guarantees exact alignment in encoder/decoder pairs (VQVAE, UNet). `'same'` — stride-2 conv with `padding='SAME'` | `"asymmetric"` |
| `pool_type` | `str`, optional | Use `'SPATIAL_MAX'` or `'SPATIAL_AVG'` for pooling-based downsampling instead of strided conv. A 1×1 conv adjusts channels after pooling. When set, `padding_mode` is ignored | `None` |
| `use_bias` | `bool` | | `False` |
| `initializer` | `str` | | `"lecun_normal"` |

**Notes:** For odd spatial dimensions, `padding_mode='same'` preserves more spatial information `(ceil(H/2))` while `'asymmetric'` floors `(floor(H/2))`. Use `'same'` if input spatial dimensions may be odd.

---

### `UpsampleBlock`

```python
UpsampleBlock(
    features: int,
    use_bias: bool = False,
    initializer: str = "lecun_normal",
)
```

Spatial upsampling by factor 2 via bilinear interpolation + 3×3 conv. `(B, H, W, C) → (B, H*2, W*2, features)`.

---

### `NonLocalBlock`

```python
NonLocalBlock(
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    downsample_factor: Optional[int] = None,
    dropout_rate: float = 0.0,
    initializer: str = "lecun_normal",
)
```

Spatial self-attention block (non-local means). Full-resolution scaled dot-product attention over the spatial dimensions. Output is added to the input as a residual. `(B, H, W, C) → (B, H, W, C)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `norm` | `str` | Applied to input before QKV projections | `"GROUP_NORM"` |
| `downsample_factor` | `int`, optional | Reduces spatial resolution of K and V before attention by bilinear downsampling, reducing memory from O(H²W²) to O((H/f)²(W/f)²). Default `None` (full resolution) | `None` |
| `dropout_rate` | `float` | Applied to attention weights | `0.0` |

**Notes:** Full resolution attention is O(H²W²). For H×W > 1024 consider `downsample_factor=2` or `4`.

---

### `InceptionBlock`

```python
InceptionBlock(
    c_red: dict,
    c_out: dict,
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    activation: str = "silu",
    activation_kwargs: Optional[dict] = None,
    initializer: str = "lecun_normal",
    dropout_rate: float = 0.0,
)
```

Four-branch Inception block. Branches: 1×1 conv, 1×1→3×3, 1×1→5×5, max pool→1×1. Outputs are concatenated.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `c_red` | `dict` | Bottleneck channel sizes. Keys: `'3x3'`, `'5x5'` |
| `c_out` | `dict` | Output channel sizes per branch. Keys: `'1x1'`, `'3x3'`, `'5x5'`, `'max'` |

Output channels = `sum(c_out.values())`. Spatial dimensions are unchanged.

---

### `DenseLayer`

```python
DenseLayer(
    growth_rate: int,
    bn_size: int = 4,
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    activation: str = "silu",
    activation_kwargs: Optional[dict] = None,
    initializer: str = "lecun_normal",
    dropout_rate: float = 0.0,
)
```

Single layer in a `DenseBlock`. Pre-activation ordering: `norm → act → 1×1 conv → norm → act → 3×3 conv → concat`. `(B, H, W, C) → (B, H, W, C + growth_rate)`.

---

### `DenseBlock`

```python
DenseBlock(
    num_layers: int,
    growth_rate: int,
    bn_size: int = 4,
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    activation: str = "silu",
    activation_kwargs: Optional[dict] = None,
    initializer: str = "lecun_normal",
    dropout_rate: float = 0.0,
)
```

Stack of `DenseLayers` with growing channel concatenation. Output channels = `input_channels + num_layers * growth_rate`.

**Notes:** When `norm='GROUP_NORM'`, both `input_channels` and `growth_rate` must be divisible by `num_groups` (default 8). `growth_rate` is validated at construction; `input_channels` are validated on the first forward pass.

---

### `TransitionLayer`

```python
TransitionLayer(
    features: int,
    pool_type: str = "SPATIAL_AVG",
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    activation: str = "silu",
    activation_kwargs: Optional[dict] = None,
    initializer: str = "lecun_normal",
)
```

Transition layer between `DenseBlocks`. Reduces channels via 1×1 conv and spatial resolution via pooling. `(B, H, W, C) → (B, H//2, W//2, features)`. `features` is typically `C // 2`.

---

### `PatchEmbed`

```python
PatchEmbed(
    patch_size: int,
    embed_dim: int,
    flatten: bool = True,
    use_bias: bool = True,
    initializer: str = "lecun_normal",
)
```

Conv-based patch embedding for Vision Transformers. Splits a spatial image into non-overlapping patches using a strided convolution and projects each patch to `embed_dim`. H and W must be divisible by `patch_size`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `patch_size` | `int` | Side length P of each square patch | required |
| `embed_dim` | `int` | Output embedding dimension per patch | required |
| `flatten` | `bool` | If `True`, returns `(B, T, embed_dim)` sequence format. If `False`, returns `(B, H//P, W//P, embed_dim)` spatial grid format. Use `True` for ViT, `False` for Swin | `True` |
| `use_bias` | `bool` | Bias in the projection conv | `True` |

No norm, activation, or positional encoding — `PatchEmbed` is a pure linear projection.

---

## 1D Building Blocks

1D variants mirror the 2D blocks. Inputs are `(B, L, C)` channels-last. Default norm is `LAYER_NORM` (rather than `GROUP_NORM`) since 1D sequences typically use transformer-style norms. Dropout is standard (not spatial).

| Class | Description |
|-------|-------------|
| `ConvBlock1d` | Single 1D conv + norm + act + dropout |
| `ResidualBlock1d` | 1D residual block with two convs and skip connection |
| `DownsampleBlock1d` | Stride-2 conv, `(B, L, C) → (B, L//2, features)` |
| `UpsampleBlock1d` | Linear interpolation + conv, `(B, L, C) → (B, L*2, features)` |

All share the same parameters as their 2D counterparts with `int` instead of `tuple` for kernel/stride. See `ConvBlock` and `ResidualBlock` for parameter descriptions.

---

## Registered Conv Nets

---

### `CONV_ENCODER`

```python
ConvEncoder(
    channels: tuple,
    num_res_blocks: int = 2,
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    activation: str = "silu",
    activation_kwargs: Optional[dict] = None,
    initializer: str = "lecun_normal",
    downsample_padding: str = "asymmetric",
    downsample_pool_type: Optional[str] = None,
    use_non_local: bool = False,
    non_local_downsample: Optional[int] = None,
    pre_norm: bool = False,
    dropout_rate: float = 0.0,
)
```

Convolutional encoder: `ResidualBlock` × `num_res_blocks` + `DownsampleBlock` at each resolution level.

Input: `(B, H, W, C)` → `(B, H // 2^(n-1), W // 2^(n-1), channels[-1])` where `n = len(channels)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `channels` | `tuple[int, ...]` | Output channels at each resolution level | required |
| `num_res_blocks` | `int` | Residual blocks per level | `2` |
| `downsample_padding` | `str` | `'asymmetric'` or `'same'`. See `DownsampleBlock` | `"asymmetric"` |
| `downsample_pool_type` | `str`, optional | If set, uses pooling instead of strided conv | `None` |
| `use_non_local` | `bool` | Insert `NonLocalBlock` after the last resolution level | `False` |
| `non_local_downsample` | `int`, optional | `downsample_factor` for `NonLocalBlock` | `None` |
| `pre_norm` | `bool` | Use pre-activation ordering in residual blocks | `False` |
| `dropout_rate` | `float` | Spatial dropout forwarded to each `ResidualBlock` | `0.0` |

---

### `CONV_DECODER`

```python
ConvDecoder(
    channels: tuple,
    num_res_blocks: int = 2,
    out_features: Optional[int] = None,
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    activation: str = "silu",
    activation_kwargs: Optional[dict] = None,
    initializer: str = "lecun_normal",
    use_non_local: bool = False,
    non_local_downsample: Optional[int] = None,
    pre_norm: bool = False,
    dropout_rate: float = 0.0,
)
```

Convolutional decoder: `ResidualBlock` × `num_res_blocks` + `UpsampleBlock` at each resolution level.

Input: `(B, H, W, C)` → `(B, H * 2^(n-1), W * 2^(n-1), channels[-1])`.

Optional `out_features`: if set, a final 1×1 conv projects to that channel count.

---

### `RESNET`

```python
ResNet(
    num_classes: int,
    c_hidden: tuple = (64, 128, 256),
    num_blocks: tuple = (3, 3, 3),
    pre_norm: bool = False,
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    activation: str = "silu",
    activation_kwargs: Optional[dict] = None,
    initializer: str = "lecun_normal",
    dropout_rate: float = 0.0,
)
```

ResNet with configurable block type, norm, and activation (He et al. 2016). Downsampling at the first block of each group except the first. Global average pooling + linear head.

Input: `(B, H, W, C)` → `(B, num_classes)`.

---

### `DENSENET`

```python
DenseNet(
    num_classes: int,
    num_layers: tuple = (6, 6, 6, 6),
    growth_rate: int = 16,
    bn_size: int = 4,
    norm: str = "GROUP_NORM",
    norm_kwargs: Optional[dict] = None,
    activation: str = "silu",
    activation_kwargs: Optional[dict] = None,
    initializer: str = "lecun_normal",
    transition_pool_type: str = "SPATIAL_AVG",
    dropout_rate: float = 0.0,
)
```

DenseNet with configurable norm, activation, and dropout (Huang et al. 2017). `DenseBlock` stages separated by `TransitionLayer` downsampling. Global average pooling + linear head.

Input: `(B, H, W, C)` → `(B, num_classes)`.
