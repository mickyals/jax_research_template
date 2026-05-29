# `nets/transformers.py`

Transformer blocks and registered transformer architectures. Depends on `core.attention`, `core.nets.conv` (`PatchEmbed`, `ConvDecoder`), `core.nets.mlp` (`MLP`), and `core.embeddings` (`SinusoidalPosEncoding`, `LearnedPosEncoding`).

---

## Registry Functions

---

### `register_transformer`

```
register_transformer(name: str, description: str = "") -> callable
```

Class decorator. Names stored uppercase, must be unique.

---

### `get_transformer`

```
get_transformer(name: str, **kwargs) -> nn.Module
```

Retrieve and instantiate a registered transformer by name. Uses `__dataclass_fields__` for kwarg inspection. Unknown kwargs trigger a `UserWarning` and are dropped.

**Raises:** `ValueError` if no net with the given name exists.

---

### `list_transformers`

```
list_transformers() -> dict[str, str]
```

Return a sorted dictionary of all registered transformer names and their descriptions.

---

## Transformer Blocks

Blocks are not registered — they are composable units used by the registered nets. Import directly from `core.nets.transformers` if you need them.

---

### `TransformerBlock`

```python
TransformerBlock(
    embed_dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    causal: bool = False,
    use_bias: bool = True,
)
```

Pre-LN transformer encoder block (Xiong et al. 2020).

```
x = x + Dropout(MHA(LN(x)))
x = x + Dropout(FFN(LN(x)))
```

FFN is a two-layer MLP: `Linear → GELU → Dropout → Linear` with hidden dim `= mlp_ratio * embed_dim`.

Input/output: `(B, T, embed_dim)`.

**`__call__` signature:**

```
__call__(
    x: jax.Array,
    mask: Optional[jax.Array] = None,
    train: bool = True,
    return_weights: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]
```

`return_weights=True` returns `(output, attn_weights)` where `attn_weights` has shape `(B, num_heads, T, T)`.

---

### `CrossAttentionBlock`

```python
CrossAttentionBlock(
    embed_dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    use_bias: bool = True,
)
```

Pre-LN cross-attention block with separate LayerNorm for query and key/value sources.

```
x = x + Dropout(CrossAttn(LN_q(x), LN_kv(context)))
x = x + Dropout(FFN(LN(x)))
```

`x` and `context` must share `embed_dim`. Input/output `x`: `(B, T_q, embed_dim)`. `context`: `(B, T_kv, embed_dim)`.

**`__call__` signature:**

```
__call__(
    x: jax.Array,
    context: jax.Array,
    mask: Optional[jax.Array] = None,
    train: bool = True,
    return_weights: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]
```

`return_weights=True` returns `(output, attn_weights)` with weights shape `(B, num_heads, T_q, T_kv)`.

---

### `SwinBlock`

```python
SwinBlock(
    embed_dim: int,
    num_heads: int,
    window_size: int = 7,
    shift_size: int = 0,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    use_bias: bool = True,
)
```

Pre-LN Swin window attention block.

```
x = x + Dropout(SwinWindowAttn(LN(x)))
x = x + Dropout(FFN(LN(x_flat)))
```

FFN operates on flattened tokens `(B, H*W, C)` then reshapes back to `(B, H, W, C)`.

Input/output: `(B, H, W, embed_dim)`. H and W must be divisible by `window_size`.

`shift_size=0` → W-MSA (no shift). `shift_size=window_size//2` → SW-MSA.

`return_weights=True` returns `(output, attn_weights)` with weights shape `(B*num_windows, num_heads, M^2, M^2)`.

---

### `SwinBlockPair`

```python
SwinBlockPair(
    embed_dim: int,
    num_heads: int,
    window_size: int = 7,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    use_bias: bool = True,
)
```

W-MSA + SW-MSA `SwinBlock` pair — the standard Swin building unit. The first block uses `shift_size=0`, the second uses `shift_size=window_size//2`, enabling cross-window communication every two blocks.

Input/output: `(B, H, W, embed_dim)`.

---

### `PatchMerging`

```python
PatchMerging(use_bias: bool = False)
```

Swin Transformer spatial downsampling. Concatenates 2×2 neighbouring patches along the channel dimension then applies `LayerNorm` and a linear projection to produce `2*C` channels.

`(B, H, W, C) → (B, H//2, W//2, 2*C)`

H and W must be even. Validated at call time before JAX tracing.

---

## Registered Nets

---

### `TRANSFORMER_ENCODER`

```python
TransformerEncoder(
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    causal: bool = False,
    use_bias: bool = True,
    add_pos_encoding: bool = True,
    max_len: int = 5000,
)
```

Transformer encoder: optional sinusoidal positional encoding + `TransformerBlock` stack + final `LayerNorm`.

Input: `(B, T, embed_dim)` — already-embedded tokens. Input projection is the caller's responsibility.
Output: `(B, T, embed_dim)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `num_layers` | `int` | Number of `TransformerBlock`s | required |
| `embed_dim` | `int` | | required |
| `num_heads` | `int` | | required |
| `causal` | `bool` | Causal masking in all blocks | `False` |
| `add_pos_encoding` | `bool` | Add sinusoidal positional encoding before blocks. Set `False` for set/permutation-invariant tasks or when positional encoding is handled externally | `True` |
| `max_len` | `int` | Maximum sequence length for sinusoidal encoding | `5000` |

**Methods:**

- `get_attention_maps(x, mask=None, train=False) -> list[jax.Array]` — returns per-layer attention weight maps, each `(B, num_heads, T, T)`.

---

### `TRANSFORMER_DECODER`

```python
TransformerDecoder(
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    causal: bool = False,
    use_bias: bool = True,
)
```

Transformer decoder: `TransformerBlock` + `CrossAttentionBlock` pairs. Each layer first applies self-attention, then cross-attention to the encoder context.

Supports shared context (one tensor for all layers) or per-layer context (list of `num_layers` tensors, for multi-scale encoder outputs such as hierarchical Swin stages).

Input/output `x`: `(B, T_q, embed_dim)`.

**`__call__` signature:**

```
__call__(
    x: jax.Array,
    context: jax.Array | list[jax.Array],
    self_mask: Optional[jax.Array] = None,
    cross_mask: Optional[jax.Array] = None,
    train: bool = True,
) -> jax.Array
```

`context` as a list requires exactly `num_layers` tensors.

---

### `VIT`

```python
ViT(
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    mlp_ratio: float = 4.0,
    num_classes: Optional[int] = None,
    dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    use_bias: bool = True,
)
```

Vision Transformer for image classification or feature extraction (Dosovitskiy et al. 2020).

Pipeline: `PatchEmbed → CLS token → LearnedPosEncoding → Dropout → TransformerEncoder → LayerNorm → head`.

Input: `(B, H, W, C)` channels-last.
Output: `(B, num_classes)` if `num_classes` is set, `(B, embed_dim)` CLS token features otherwise.

Positional encoding is learnable (`LearnedPosEncoding`) covering `T+1` positions (T patch tokens + 1 CLS).

**Methods:**

- `get_attention_maps(x, train=False) -> list[jax.Array]` — returns per-layer attention maps, each `(B, num_heads, T+1, T+1)`.

---

### `MASKED_VIT`

```python
MaskedViT(
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    mask_ratio: float = 0.75,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    use_bias: bool = True,
)
```

ViT encoder with MAE-style random patch masking (He et al. 2022). Encodes only visible (unmasked) patches. Returns encoded visible tokens, the boolean mask, and restore indices needed by the decoder.

**At train time:** randomly masks `mask_ratio` fraction of patches using `rngs={'mask': key}`.
**At eval time:** encodes all patches (no masking), returns all-False mask and identity `ids_restore`.

**`__call__` returns:** `(visible_tokens, mask, ids_restore)` where:

| Output | Shape | Description |
|--------|-------|-------------|
| `visible_tokens` | `(B, T_visible, embed_dim)` | Encoded visible patch tokens (CLS stripped) |
| `mask` | `(B, T)` bool | `True` = masked (not seen by encoder) |
| `ids_restore` | `(B, T)` int | Indices to unshuffle the full sequence back to original order |

`T_visible = round(T * (1 - mask_ratio))` during training; `T` during eval.

Positional encoding is applied before masking so visible tokens carry their original spatial positions.

**Requires `rngs={'mask': key}` at train time.**

---

### `MAE_DECODER`

```python
MAEDecoder(
    num_patches: int,
    patch_dim: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
)
```

Lightweight transformer decoder for MAE patch reconstruction. Takes `(visible_tokens, mask, ids_restore)` from `MaskedViT`, inserts learnable mask tokens at masked positions, unshuffles via `ids_restore`, adds positional encoding, then runs a lightweight transformer to reconstruct all patches.

A linear projection maps `encoder_embed_dim → embed_dim` before decoding, allowing encoder and decoder to have different widths.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `num_patches` | `int` | Total number of patches T (before masking) |
| `patch_dim` | `int` | Reconstruction target dimensionality = `patch_size^2 * in_channels` |
| `embed_dim` | `int` | Decoder embedding dim (typically smaller than encoder) |
| `num_layers` | `int` | Typically 4–8 (lighter than encoder) |

**`__call__` signature:**

```
__call__(
    visible_tokens: jax.Array,   # (B, T_visible, encoder_embed_dim)
    mask: jax.Array,             # (B, T) bool
    ids_restore: jax.Array,      # (B, T) int
    train: bool = True,
) -> jax.Array                   # (B, T, patch_dim)
```

Output is reconstructed pixel values for **all** patches. Compute loss only on masked patches (`mask=True`).

---

### `CONV_MAE_DECODER`

```python
ConvMAEDecoder(
    num_patches_h: int,
    num_patches_w: int,
    encoder_embed_dim: int,
    decoder_embed_dim: int,
    channels: tuple,
    out_features: int,
    num_res_blocks: int = 2,
    dropout_rate: float = 0.0,
)
```

`ConvDecoder`-based MAE reconstruction decoder. Better suited than a transformer decoder when the input has strong spatial structure (e.g. satellite image patches), because `ConvDecoder` explicitly models spatial locality via residual blocks and upsampling.

Reshapes the full token sequence `(B, T, D) → (B, nH, nW, D)` then decodes with `ConvDecoder`.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `num_patches_h` | `int` | Patches along height (`H // patch_size`) |
| `num_patches_w` | `int` | Patches along width (`W // patch_size`) |
| `encoder_embed_dim` | `int` | Encoder output dimensionality |
| `decoder_embed_dim` | `int` | Intermediate dim before `ConvDecoder` |
| `channels` | `tuple` | `ConvDecoder` channel schedule, e.g. `(256, 128, 64)` |
| `out_features` | `int` | Final output channels (typically `C_in`) |

**`__call__` signature:** Same as `MAEDecoder`.
**Output:** `(B, H_orig, W_orig, out_features)` — reconstructed image.

---

### `SWIN_ENCODER`

```python
SwinEncoder(
    patch_size: int = 4,
    embed_dim: int = 96,
    depths: tuple = (2, 2, 6, 2),
    num_heads: tuple = (3, 6, 12, 24),
    window_size: int = 7,
    mlp_ratio: float = 4.0,
    num_classes: Optional[int] = None,
    dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    use_bias: bool = True,
)
```

Hierarchical Swin Transformer encoder (Liu et al. 2021).

Pipeline: `PatchEmbed (spatial) → LayerNorm → Dropout → [SwinBlockPair × depths[i] → PatchMerging] × stages → LayerNorm → global avg pool → optional head`.

Input: `(B, H, W, C)` channels-last.
Output: `(B, num_classes)` if `num_classes` set, `(B, final_channels)` otherwise.

H and W must be divisible by `patch_size * window_size`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `patch_size` | `int` | Initial patch embedding stride | `4` |
| `embed_dim` | `int` | Base channel count. Doubles after each `PatchMerging` | `96` |
| `depths` | `tuple[int, ...]` | Number of `SwinBlockPair`s per stage | `(2, 2, 6, 2)` |
| `num_heads` | `tuple[int, ...]` | Attention heads per stage. Must match `len(depths)` | `(3, 6, 12, 24)` |
| `window_size` | `int` | Window size for all stages | `7` |
| `num_classes` | `int`, optional | Linear head after pooling. If `None`, returns feature vector | `None` |

**Channel schedule** (doubles after each `PatchMerging`):

| Stage | Channels |
|-------|----------|
| 0 | `embed_dim` |
| 1 | `2 × embed_dim` |
| 2 | `4 × embed_dim` |
| 3 | `8 × embed_dim` (default 4-stage) |

**Raises:** `ValueError` if `len(depths) != len(num_heads)` or if H/W is not divisible by `patch_size`.
