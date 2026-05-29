# `norms.py`

Registry-based normalisation modules for Flax/JAX models. All norms are `flax.linen.Module` subclasses and share a consistent `__call__(self, x, train=True)` signature for API compatibility with `BatchNorm`, which requires the `train` flag. Norms that do not depend on batch statistics (`LAYER_NORM`, `GROUP_NORM`, `INSTANCE_NORM`, `RMS_NORM`) accept but ignore `train`.

---

## Registry Functions

---

### `register_norm`

```
register_norm(name: str, description: str = "") -> callable
```

Class decorator that registers a normalisation module under a given name. Names are stored uppercase and must be unique.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name used for lookup. Stored uppercase | required |
| `description` | `str` | Short description shown in `list_norms()` | `""` |

**Returns:** Class decorator.

**Raises:** `ValueError` if a norm with the same name is already registered.

---

### `get_norm`

```
get_norm(name: str, **kwargs) -> nn.Module
```

Retrieve and instantiate a registered norm by name. Uses `__dataclass_fields__` for kwarg inspection (Flax modules are dataclasses). Unknown kwargs trigger a `UserWarning` and are dropped.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name of the registered norm (case-insensitive) | required |
| `**kwargs` | any | Forwarded to the norm constructor. Unknown kwargs are warned about and dropped | |

**Returns:** `nn.Module` — an instantiated Flax Linen normalisation module.

**Raises:** `ValueError` if no norm with the given name exists.

---

### `list_norms`

```
list_norms() -> dict[str, str]
```

Return a sorted dictionary of all registered norm names and their descriptions.

---

## Normalisation Modules

All modules have `__call__(self, x: jax.Array, train: bool = True) -> jax.Array`.

---

### `BATCH_NORM`

```python
BatchNorm(
    use_scale: bool = True,
    use_bias: bool = True,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
)
```

Batch normalisation (Ioffe & Szegedy 2015). Normalises over the batch dimension. Maintains running statistics during training for use at eval time.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `use_scale` | `bool` | Learn scale parameter (gamma) | `True` |
| `use_bias` | `bool` | Learn bias parameter (beta) | `True` |
| `momentum` | `float` | Momentum for running statistics update | `0.1` |
| `epsilon` | `float` | Small constant for numerical stability | `1e-5` |

**Notes:** `BatchNorm` requires mutable `batch_stats` in the variable collection during training:

```python
out, updates = model.apply(
    {'params': params, 'batch_stats': batch_stats},
    x, train=True,
    mutable=['batch_stats'],
)
```

Pass `train=False` at eval time to use running statistics.

---

### `LAYER_NORM`

```python
LayerNorm(
    use_scale: bool = True,
    use_bias: bool = True,
    epsilon: float = 1e-6,
)
```

Layer normalisation (Ba et al. 2016). Normalises over the last dimension (feature dimension). Does not depend on batch size — behaviour is identical at train and eval time.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `use_scale` | `bool` | Learn scale parameter (gamma) | `True` |
| `use_bias` | `bool` | Learn bias parameter (beta) | `True` |
| `epsilon` | `float` | Small constant for numerical stability | `1e-6` |

---

### `GROUP_NORM`

```python
GroupNorm(
    num_groups: int = 8,
    use_scale: bool = True,
    use_bias: bool = True,
    epsilon: float = 1e-6,
)
```

Group normalisation (Wu & He 2018). Divides channels into `num_groups` groups and normalises within each group. Does not depend on batch size — recommended over `BatchNorm` for small batches. `num_groups` must divide the channel dimension evenly.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `num_groups` | `int` | Number of groups. Must divide the channel dimension | `8` |
| `use_scale` | `bool` | Learn scale parameter (gamma) | `True` |
| `use_bias` | `bool` | Learn bias parameter (beta) | `True` |
| `epsilon` | `float` | Small constant for numerical stability | `1e-6` |

---

### `INSTANCE_NORM`

```python
InstanceNorm(
    use_scale: bool = True,
    use_bias: bool = True,
    epsilon: float = 1e-6,
)
```

Instance normalisation (Ulyanov et al. 2016). Normalises each sample and each channel independently. Equivalent to `GroupNorm` with `num_groups` equal to the number of channels. Does not depend on batch size.

Implemented via `nn.GroupNorm(num_groups=None, group_size=1)` — the idiomatic Flax way to achieve per-channel normalisation.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `use_scale` | `bool` | Learn scale parameter (gamma) | `True` |
| `use_bias` | `bool` | Learn bias parameter (beta) | `True` |
| `epsilon` | `float` | Small constant for numerical stability | `1e-6` |

---

### `RMS_NORM`

```python
RMSNorm(
    use_scale: bool = True,
    epsilon: float = 1e-6,
)
```

RMS normalisation. Normalises by the root mean square of the activations with no mean centering. Used in modern transformer variants (LLaMA, Gemma, etc.) as a cheaper alternative to `LayerNorm`. No bias term by design — the absence of mean centering makes a bias redundant.

Feature dimension is inferred at first call and fixed thereafter. Do not reuse this module with inputs of different feature dimensions.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `use_scale` | `bool` | Learn a scale parameter | `True` |
| `epsilon` | `float` | Small constant for numerical stability | `1e-6` |
