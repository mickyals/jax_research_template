# `attention.py`

Pure attention mechanisms for transformer architectures. All modules use `flax.linen` and share a consistent `__call__` signature:

```
(x, context=None, mask=None, train=True, return_weights=False)
```

`context=None` means self-attention. `SwinWindowAttention` does not support a `context` argument.

`return_weights=True` returns `(output, weights)` where `weights` are raw per-head attention probabilities **without dropout** regardless of the `train` flag (diagnostic use).

---

## Mask Utilities

---

### `make_causal_mask`

```
make_causal_mask(seq_len: int) -> jax.Array
```

Upper-triangular causal mask for autoregressive attention.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `seq_len` | `int` | Sequence length |

**Returns:** Boolean array of shape `(seq_len, seq_len)`. `True` where attention is allowed (lower triangle + diagonal), `False` where blocked.

**Notes:** Pass as `mask` to `MultiHeadAttention`. Boolean masks are converted to additive bias `(0.0 / -1e9)` internally before softmax.

---

### `make_padding_mask`

```
make_padding_mask(lengths: jax.Array, max_len: int) -> jax.Array
```

Boolean padding mask for variable-length sequences.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `lengths` | `jax.Array` | Integer array of shape `(B,)` with valid token counts per sequence |
| `max_len` | `int` | Padded sequence length |

**Returns:** Boolean array of shape `(B, max_len)`. `True` for valid positions, `False` for padding.

**Notes:** To use as an attention mask, broadcast to `(B, 1, max_len)` before passing to `MultiHeadAttention`:

```python
pad_mask = make_padding_mask(lengths, max_len)   # (B, max_len)
mask = pad_mask[:, None, :]                       # (B, 1, max_len)
out = attn(x, mask=mask)
```

---

### `make_swin_shift_mask`

```
make_swin_shift_mask(window_size: int, shift_size: int, H: int, W: int) -> jax.Array
```

Additive attention bias mask for shifted window attention. After a cyclic shift by `(shift_size, shift_size)`, windows may contain tokens from non-adjacent spatial regions. This mask blocks attention between those regions, restoring locality.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `window_size` | `int` | Window size M. Assumes square windows |
| `shift_size` | `int` | Cyclic shift amount. Typically `window_size // 2`. `0` returns a zero mask |
| `H` | `int` | Feature map height in tokens. Must be divisible by `window_size` |
| `W` | `int` | Feature map width in tokens. Must be divisible by `window_size` |

**Returns:** Float additive bias of shape `(num_windows, M^2, M^2)` with values `0.0` (attend) or `-1e9` (block). Added directly to attention logits.

**Notes:** `num_windows = (H // window_size) * (W // window_size)`. Computed with NumPy then converted to JAX — this is a static quantity for fixed `H`, `W`, `window_size`, `shift_size`. Call once and reuse.

---

## `MultiHeadAttention`

```python
MultiHeadAttention(
    embed_dim: int,
    num_heads: int,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    causal: bool = False,
)
```

Multi-head scaled dot-product attention (self or cross). Owns QKV projections explicitly via `nn.DenseGeneral`, enabling clean weight return without Flax intermediates machinery. Uses `flax.linen.dot_product_attention_weights` for the core computation.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `embed_dim` | `int` | Output and input dimensionality. Must be divisible by `num_heads` | required |
| `num_heads` | `int` | Number of attention heads | required |
| `dropout_rate` | `float` | Attention weight dropout applied during training | `0.0` |
| `use_bias` | `bool` | Whether QKV and output projections include bias | `True` |
| `causal` | `bool` | If `True`, automatically applies a causal mask when no explicit mask is provided | `False` |

**Raises:** `ValueError` if `embed_dim % num_heads != 0`.

**Input shape:** `(B, T, embed_dim)`.

**Mask shapes** (all expanded to `(B, num_heads, T_q, T_kv)` internally):

| Shape | Meaning |
|-------|---------|
| `(T_q, T_kv)` | Shared across batch and heads |
| `(B, T_q, T_kv)` | Shared across heads |
| `(B, num_heads, T_q, T_kv)` | Fully specified |

Boolean masks: `True` = attend, `False` = block. Float masks: added directly to logits as additive bias.

If both `causal=True` and an explicit mask are provided, the explicit mask takes precedence.

**`__call__` signature:**

```
__call__(
    x: jax.Array,
    context: Optional[jax.Array] = None,
    mask: Optional[jax.Array] = None,
    train: bool = True,
    return_weights: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]
```

| Parameter | Description |
|-----------|-------------|
| `x` | Query source `(B, T_q, embed_dim)` |
| `context` | Key/value source `(B, T_kv, embed_dim)`. `None` = self-attention |
| `mask` | Boolean or float mask |
| `train` | Enables attention dropout. Requires `rngs={'dropout': key}` when `train=True` and `dropout_rate > 0` |
| `return_weights` | If `True` returns `(output, weights)` where `weights` has shape `(B, num_heads, T_q, T_kv)` |

---

## `CrossAttention`

```python
CrossAttention(
    embed_dim: int,
    num_heads: int,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
)
```

Explicit cross-attention: Q from `x`, K and V from `context`. Functionally equivalent to `MultiHeadAttention` with context provided, but makes the asymmetric Q/KV split structurally explicit at the call site. Preferred in encoder-decoder blocks.

`x` and `context` may differ in sequence length but must share `embed_dim`. Output shape matches `x`: `(B, T_q, embed_dim)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `embed_dim` | `int` | Output dimensionality | required |
| `num_heads` | `int` | Number of attention heads | required |
| `dropout_rate` | `float` | | `0.0` |
| `use_bias` | `bool` | | `True` |

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

`context` is required (non-optional). `return_weights=True` returns `(output, weights)` with weights shape `(B, num_heads, T_q, T_kv)`.

---

## `SwinWindowAttention`

```python
SwinWindowAttention(
    embed_dim: int,
    num_heads: int,
    window_size: int = 7,
    shift_size: int = 0,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
)
```

Local window multi-head self-attention with learnable relative position bias. Implements W-MSA (`shift_size=0`) and SW-MSA (`shift_size > 0`) from Swin Transformer (Liu et al. 2021).

Input is a spatial feature map `(B, H, W, C)`. It is partitioned into non-overlapping windows of size `(window_size, window_size)`, attention is computed independently within each window, then windows are merged back. For shifted window attention the feature map is first cyclically shifted and an additive mask blocks attention between non-adjacent regions.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `embed_dim` | `int` | Channel dimension C. Must equal input C | required |
| `num_heads` | `int` | Number of attention heads. `embed_dim` must be divisible by `num_heads` | required |
| `window_size` | `int` | Side length M of each square attention window. H and W must be divisible by this | `7` |
| `shift_size` | `int` | Cyclic shift amount for SW-MSA. Typically `window_size // 2`. `0` = W-MSA | `0` |
| `dropout_rate` | `float` | Attention weight dropout | `0.0` |
| `use_bias` | `bool` | QKV and output projection bias | `True` |

**Raises:**
- `ValueError` if `embed_dim % num_heads != 0`
- `ValueError` if `shift_size >= window_size`
- `ValueError` at call time if H or W is not divisible by `window_size`

**Relative position bias table shape:** `(2M-1, 2M-1, num_heads)`.

**`__call__` signature:**

```
__call__(
    x: jax.Array,
    mask: Optional[jax.Array] = None,
    train: bool = True,
    return_weights: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]
```

| Parameter | Description |
|-----------|-------------|
| `x` | Spatial feature map `(B, H, W, C)`. H and W must be divisible by `window_size` |
| `mask` | Additional float additive bias of shape `(num_windows, M^2, M^2)`. Not a boolean mask — use `0.0 / -1e9` values. Added on top of the internal shift mask and relative position bias |
| `train` | Enables dropout |
| `return_weights` | Returns `(output, weights)` where weights has shape `(B*num_windows, num_heads, M^2, M^2)` |

**Notes:**
- Output shape matches input: `(B, H, W, C)`.
- `context` is not supported — self-attention only.
- `return_weights=True` returns raw per-window weights. Windows are not merged since aggregation across windows is spatially ambiguous.
- The internal shift mask is computed and cached each forward pass. Use `make_swin_shift_mask` externally if you need to inspect or reuse it.
