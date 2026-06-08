# src/core

Reusable, registry-based building blocks for JAX/Flax models. Every component is string-addressable from YAML configs — nothing is hard-wired to a specific experiment.

---

## The registry pattern

Each module defines a `register_X` decorator and a corresponding `get_X` / `list_Xs` pair. Decorating a class at definition time adds it to a module-level dict so it can be looked up by name at runtime.

```python
# Define (in core/activations.py)
@register_activation("SINE", description="SIREN sinusoidal activation")
class Sine(nn.Module):
    w0: float = 1.0
    @nn.compact
    def __call__(self, x): return jnp.sin(self.w0 * x)

# Retrieve (anywhere)
from core import get_activation
act = get_activation("sine")          # case-insensitive
act = get_activation("SINE", w0=30.0) # kwargs forwarded to __init__
```

`list_Xs()` prints a table of all registered names and descriptions — useful when exploring what is available.

---

## Modules

### `activations.py`

Standard and INR (implicit neural representation) activations.

| Name | Description |
|------|-------------|
| `relu`, `leaky_relu`, `elu`, `selu` | Standard nonlinearities |
| `gelu`, `swish`, `mish`, `softplus` | Smooth activations |
| `tanh`, `sigmoid` | Bounded activations |
| `sine` | SIREN sinusoidal (Sitzmann et al. 2020) |
| `finer` | FINER variable-period sine (Liu et al. 2023) |
| `gaussian` | Gaussian activation for INRs |
| `wire` | WIRE complex Gabor wavelet (Saragadam et al. 2023) |
| `hosc` | Higher-order spectral (HOSC) activation |
| `sinc` | Sinc activation |

### `attention.py`

Core attention primitives used by `nets/transformers.py`.

| Class | Description |
|-------|-------------|
| `MultiHeadAttention` | Standard scaled dot-product MHA with optional causal mask |
| `CrossAttention` | MHA with separate query and key/value sequences |
| `SwinWindowAttention` | Shifted-window local attention (Liu et al. 2021) |

Mask convention: `True` = attend, `False` = ignore (padding). 3D masks `(B, T_q, T_kv)` are broadcast across heads; 2D masks `(B, T_kv)` are broadcast across both heads and query positions.

### `embeddings.py`

Position and coordinate encodings.

| Name | Description |
|------|-------------|
| `SINUSOIDAL` | Fixed sinusoidal positional encoding (Transformer-style) |
| `LEARNED_1D` | Learned position embedding for sequences |
| `LEARNED_2D` | Learned position embedding for 2D grids |
| `FOURIER` | Fourier feature mapping (Tancik et al. 2020) |
| `GAUSSIAN_FOURIER` | Random Gaussian Fourier features; `mapping_dim` must be even |
| `SPHERICAL_GRID` | Spherical harmonic encoding on a lat/lon grid |
| `SPHERICAL_C` | Continuous spherical harmonics |
| `SPHERICAL_M` | Mode-indexed spherical harmonics |
| `SPHERICAL_DFS` | Driscoll-Healy spherical encoding |

`GaussianFourierEmbedding` is particularly useful for encoding continuous coordinates (distances, angles, lat/lon) into a fixed-dim feature space.

### `initializations.py`

Weight initializers for standard and INR networks.

| Name | Description |
|------|-------------|
| `xavier_uniform`, `xavier_normal` | Glorot initialization |
| `lecun_uniform`, `lecun_normal` | LeCun initialization |
| `he_uniform`, `he_normal` | Kaiming initialization |
| `siren` | SIREN-specific sinusoidal initialization |
| `finer` | FINER initialization |
| `gabor` | Gabor/WIRE initialization |
| `zeros`, `ones` | Constant initializers |

### `norms.py`

Normalization layers.

| Name | Description |
|------|-------------|
| `layernorm` | Layer normalization (default in transformers) |
| `batchnorm` | Batch normalization (requires `batch_stats` in state) |
| `groupnorm` | Group normalization |
| `instancenorm` | Instance normalization |
| `rmsnorm` | RMS normalization (no mean subtraction) |

### `pooling.py`

Spatial and global pooling operations.

| Name | Description |
|------|-------------|
| `mean`, `max`, `min`, `sum`, `std` | Global reductions over token dim |
| `meanmax` | Concatenated mean + max pooling |
| `spatial_mean_2d`, `spatial_max_2d` | 2D spatial pooling over `(H, W)` |
| `global_mean`, `global_max` | Named aliases for global operations |

---

## `nets/` — assembled architectures

The `nets/` subdirectory contains complete architectures composed from the primitives above. Registered nets follow the same string-lookup pattern.

### `nets/mlp.py`

`MLP` — a plain fully-connected network.

```python
from core.nets.mlp import MLP

mlp = MLP(
    features      = [256, 256, 128],  # hidden then output dims
    activation    = "gelu",
    initializer   = "xavier_uniform",
    dropout_rate  = 0.1,
    use_bias      = True,
)
```

No registry — `MLP` is used by composing directly (e.g. as the FFN inside transformer blocks).

### `nets/conv.py`

| Class | Description |
|-------|-------------|
| `PatchEmbed` | Non-overlapping patch tokenizer (ViT / Swin input stem) |
| `ConvDecoder` | Transposed-conv upsampling decoder (used in CONV_MAE_DECODER) |

### `nets/transformers.py`

**Unregistered blocks** — compose these to build custom architectures:

| Class | Description |
|-------|-------------|
| `TransformerBlock` | Pre-LN self-attention + FFN |
| `CrossAttentionBlock` | Pre-LN cross-attention + FFN |
| `SwinBlock` | Pre-LN Swin window attention + FFN |
| `SwinBlockPair` | Paired W-MSA + SW-MSA Swin blocks |
| `PatchMerging` | Swin spatial downsampling |

All blocks accept `mlp_activation` and `mlp_initializer` string args that are resolved through the registries at build time.

**Registered nets** — retrieve with `get_transformer(name, **kwargs)`:

| Name | Description |
|------|-------------|
| `TRANSFORMER_ENCODER` | Sinusoidal/no positional encoding + TransformerBlock stack; `__call__` accepts `return_weights=True` to return `(output, last_layer_attn_weights)` |
| `TRANSFORMER_DECODER` | TransformerBlock + CrossAttentionBlock pairs; optional causal self-attention; shared or per-layer context |
| `VIT` | PatchEmbed + learned pos encoding + CLS token + TransformerBlock stack + optional head |
| `MASKED_VIT` | ViT encoder with MAE-style random patch masking; returns `(visible_tokens, mask, ids_restore)` |
| `MAE_DECODER` | Lightweight transformer decoder for MAE reconstruction |
| `CONV_MAE_DECODER` | ConvDecoder-based MAE reconstruction; reshapes tokens to spatial grid |
| `SWIN_ENCODER` | Hierarchical Swin stages + PatchMerging; optional classification head |

---

## Adding a new component

1. Write the class in the appropriate module.
2. Decorate it with the corresponding `@register_X` decorator.
3. Import the module in `core/__init__.py` if it isn't already (so the decorator runs on import).

That's it — the class is now available by name anywhere in the codebase.
