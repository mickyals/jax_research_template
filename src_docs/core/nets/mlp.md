# `nets/mlp.py`

MLP variants and an embedding composition layer for implicit neural representations (INR) and general regression. All nets are `flax.linen.Module` subclasses built on a shared `_BaseMLP` base class.

---

## Registry Functions

---

### `register_mlp`

```
register_mlp(name: str, description: str = "") -> callable
```

Class decorator. Names stored uppercase, must be unique.

**Raises:** `ValueError` if a net with the same name is already registered.

---

### `get_mlp`

```
get_mlp(name: str, **kwargs) -> nn.Module
```

Retrieve and instantiate a registered MLP by name. Uses `__dataclass_fields__` for kwarg inspection. Unknown kwargs trigger a `UserWarning` and are dropped.

**Raises:** `ValueError` if no net with the given name exists.

---

### `list_mlps`

```
list_mlps() -> dict[str, str]
```

Return a sorted dictionary of all registered MLP names and their descriptions.

---

## Embedding Helpers

These classes adapt embeddings from `core.embeddings` for use with the `embedding` field on `_BaseMLP`.

---

### `LatLonEmbeddingWrapper`

```python
LatLonEmbeddingWrapper(embedding: nn.Module)
```

Wraps a spherical embedding that takes `(lat, lon)` into a single `x` input. Adapts embeddings like `SphericalGridEmbedding`, `DFS`, or `SphericalHarmonicsEmbedding` (which expect separate lat and lon arrays) to accept a single concatenated input of shape `(N, 2)`.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `embedding` | `nn.Module` | Spherical embedding with signature `(lat, lon) -> jnp.ndarray` |

**Notes:** Expects `x` of shape `(N, 2)` where `x[:, 0]` is lat and `x[:, 1]` is lon, both in radians.

---

### `CombinedEmbedding`

```python
CombinedEmbedding(
    spatial_dim: int,
    spatial_embedding: Optional[nn.Module] = None,
    time_embedding: Optional[nn.Module] = None,
)
```

Splits input into spatial and temporal components, embeds each independently, and concatenates the results before the first dense layer.

`x[:, :spatial_dim]` → `spatial_embedding` (or raw if `None`)
`x[:, spatial_dim:]` → `time_embedding` (or raw if `None`)

Output is `jnp.concatenate([spatial_out, time_out], axis=-1)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `spatial_dim` | `int` | Number of spatial input columns. `x` must have at least this many columns | required |
| `spatial_embedding` | `nn.Module`, optional | Applied to `x[:, :spatial_dim]`. Use `LatLonEmbeddingWrapper` for spherical embeddings | `None` |
| `time_embedding` | `nn.Module`, optional | Applied to `x[:, spatial_dim:]`. If `None`, temporal coords are concatenated raw | `None` |

**Notes:** If a sub-embedding's `__call__` accepts a `train` argument it will receive it automatically (forward-compatible with future trainable embeddings).

---

## `_BaseMLP` — Base Class

All registered MLPs inherit from `_BaseMLP`. It provides:

- Optional `embedding` applied before the first dense layer
- Configurable kernel, bias, and output bias initializers (via registry names)
- Global dropout (`dropout_rate`) or per-layer dropout (`dropout_rates`)

**Shared fields (inherited by all subclasses):**

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `out_features` | `int` | Output dimensionality | required |
| `hidden_features` | `int` | Hidden layer width | required |
| `n_layers` | `int` | Number of hidden layers (>= 1). n_layers=1 → one hidden layer + output | required |
| `use_bias` | `bool` | Include bias in all dense layers | `True` |
| `bias_initializer` | `str` | Registry name for hidden layer bias init | `"zeros"` |
| `bias_initializer_kwargs` | `dict`, optional | Kwargs for bias init | `None` |
| `output_bias_initializer` | `str` | Registry name for output layer bias init | `"zeros"` |
| `output_bias_initializer_kwargs` | `dict`, optional | | `None` |
| `embedding` | `nn.Module`, optional | Applied to `x` before first dense layer | `None` |
| `dropout_rate` | `float` | Global dropout rate after every hidden activation | `0.0` |
| `dropout_rates` | `list`, optional | Per-layer dropout rates of length `n_layers`. Overrides `dropout_rate` | `None` |

**Notes:**
- When `dropout_rate > 0` or `dropout_rates` is set, pass `rngs={'dropout': key}` at call time with `train=True`. At eval time, pass `train=False` — no PRNG key needed.
- Subclasses implement `_make_act()`, `_make_kernel_init()`, and optionally `_make_first_kernel_init()`, `_make_first_act()`, `_make_bias_init()`, `_make_param_dtype()`.

---

## `_BaseComplexMLP`

Internal base for complex-valued networks. Hidden layers operate in `complex64` throughout. Output layer takes the real part of the final hidden state. Embedding is applied before the complex cast. Dropout zeros both real and imaginary parts of dropped units.

---

## Registered Nets

---

### `MLP`

```python
MLP(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    activation: str = "relu",
    initializer: str = "xavier_uniform",
    initializer_kwargs: Optional[dict] = None,
    activation_kwargs: Optional[dict] = None,
    # + all _BaseMLP fields
)
```

General MLP with fully configurable activation and initializer.

**Parameters (net-specific):**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `activation` | `str` | Registered activation name | `"relu"` |
| `initializer` | `str` | Kernel initializer for all layers | `"xavier_uniform"` |
| `initializer_kwargs` | `dict`, optional | Forwarded to `get_initializer` | `None` |
| `activation_kwargs` | `dict`, optional | Forwarded to `get_activation` | `None` |

---

### `SIREN`

```python
SIREN(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    first_omega: float = 30.0,
    hidden_omega: float = 30.0,
    # + all _BaseMLP fields
)
```

Sinusoidal representation network (Sitzmann et al. 2020). Uses `SINE` activation throughout with paper-specified SIREN weight initialisation (uniform bounds scaled by omega).

**Parameters (net-specific):**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `first_omega` | `float` | Frequency for the first layer | `30.0` |
| `hidden_omega` | `float` | Frequency for hidden layers | `30.0` |

---

### `FINER`

```python
FINERNet(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    first_omega: float = 30.0,
    hidden_omega: float = 30.0,
    bias_k: float = 1.0,
    bias_initializer: str = "finer_bias",
    # + all _BaseMLP fields
)
```

Adaptive frequency SIREN (Liu et al. 2024). Uses `FINER` activation with `alpha(x) = |x| + 1` adaptive frequency scaling. Bias initialised with `U(-k, k)` per the FINER scheme.

**Parameters (net-specific):**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `first_omega` | `float` | | `30.0` |
| `hidden_omega` | `float` | | `30.0` |
| `bias_k` | `float` | Half-range for FINER bias init `U(-k, k)`. Used when `bias_initializer='finer_bias'` | `1.0` |
| `bias_initializer` | `str` | Override to use a different bias scheme | `"finer_bias"` |

---

### `GAUSSIAN`

```python
GaussianNet(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    sigma: float = 10.0,
    initializer: str = "xavier_uniform",
    initializer_kwargs: Optional[dict] = None,
    # + all _BaseMLP fields
)
```

MLP with Gaussian activation `exp(-(sigma*x)^2)`.

---

### `GAUSSIAN_FINER`

```python
GaussianFINERNet(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    sigma: float = 10.0,
    omega: float = 30.0,
    initializer: str = "xavier_uniform",
    initializer_kwargs: Optional[dict] = None,
    # + all _BaseMLP fields
)
```

MLP with FINER Gaussian activation `exp(-((sigma/omega)*sin(omega*alpha(x)*x))^2)`.

---

### `WIRE`

```python
WireNet(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    omega_0: float = 20.0,
    sigma_0: float = 10.0,
    # + all _BaseMLP fields
)
```

MLP with real-valued WIRE activation (magnitude of the complex Gabor wavelet). Safe for use with real-valued dense layers.

---

### `WIRE_FINER`

```python
WireFINERNet(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    omega_0: float = 20.0,
    sigma_0: float = 10.0,
    omega_finer: float = 5.0,
    # + all _BaseMLP fields
)
```

MLP with real-valued FINER WIRE activation.

---

### `WIRE_COMPLEX`

```python
WireComplexNet(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    omega_0: float = 20.0,
    sigma_0: float = 10.0,
    # + all _BaseMLP fields
)
```

MLP with full complex Gabor wavelet activation throughout (Saragadam et al. 2023). Hidden representations are `complex64`. Output takes the real part. Embedding is applied before the complex cast.

**Notes:** WIRE_FINER complex variant is not supported — complex sin applied to complex hidden states causes `sinh` explosion via `cosh(im)` overflow.

---

### `HOSC`

```python
HOSCNet(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    beta: float = 10.0,
    initializer: str = "xavier_uniform",
    initializer_kwargs: Optional[dict] = None,
    # + all _BaseMLP fields
)
```

MLP with hyperbolic sine composition activation `tanh(beta*sin(x))`. Input should be scaled to approximately `[-pi, pi]`.

---

### `HOSC_FINER`

```python
HOSCFINERNet(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    beta: float = 10.0,
    omega: float = 30.0,
    initializer: str = "xavier_uniform",
    initializer_kwargs: Optional[dict] = None,
    # + all _BaseMLP fields
)
```

MLP with FINER HOSC activation `tanh((beta/omega)*sin(omega*alpha(x)*x))`.

---

### `SINC`

```python
SincNet(
    out_features: int,
    hidden_features: int,
    n_layers: int,
    omega: float = 30.0,
    initializer: str = "xavier_uniform",
    initializer_kwargs: Optional[dict] = None,
    # + all _BaseMLP fields
)
```

MLP with normalised sinc activation `sinc(omega*x) = sin(pi*omega*x) / (pi*omega*x)`.
