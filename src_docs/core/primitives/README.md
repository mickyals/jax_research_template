# Core / Primitives

Reusable registry-based primitives for JAX/Flax models.

- **`activations.py`**: Activation function registry with standard wrappers, SIREN, FINER, Gaussian, WIRE, HOSC, and Sinc variants
- **`embeddings.py`**: Positional and spherical embedding registry with Gaussian Fourier features, deterministic frequency banks, Sphere2Vec variants, DFS, and spherical harmonics
- **`initializations.py`**: Weight initializer registry with SIREN, FINER, Xavier, LeCun, WIRE, and Gabor variants

All three modules follow the same pattern: a `dict`-backed registry, a `@register_*` decorator, a `get_*` factory, and a `list_*` introspection function. The factory inspects constructor signatures and emits a `UserWarning` for unknown kwargs rather than raising, so hyperparameter sweeps that pass extra keys do not crash at instantiation.

---

## Module: `activations.py`

Activations are plain callable classes -- no Flax module machinery. They can be passed directly to `nn.Dense` as `kernel_init`, stored in a config dict, or called as `act(x)`.

---

### Registry Functions

---

#### `register_activation`

```
register_activation(name: str, description: str = "") -> callable
```

Class decorator that registers an activation under a given name. Names are stored uppercase and must be unique.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name used for lookup. Stored uppercase | required |
| `description` | `str` | Short description shown in `list_activations()` | `""` |

**Returns:**

| Type | Description |
|------|-------------|
| `callable` | Class decorator |

**Raises:**

| Type | Condition |
|------|-----------|
| `ValueError` | If an activation with the same name is already registered |

---

#### `get_activation`

```
get_activation(name: str, **kwargs) -> callable
```

Retrieve and instantiate a registered activation by name. Unknown kwargs are warned about and dropped rather than forwarded.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name of the registered activation (case-insensitive) | required |
| `**kwargs` | any | Arguments forwarded to the activation constructor. Unknown kwargs trigger a `UserWarning` and are dropped | |

**Returns:**

| Type | Description |
|------|-------------|
| `callable` | An instantiated activation with signature `(x: jax.Array) -> jax.Array` |

**Raises:**

| Type | Condition |
|------|-----------|
| `ValueError` | If no activation with the given name exists. Error message lists all available names |

---

#### `list_activations`

```
list_activations() -> dict[str, str]
```

Return a sorted dictionary of all registered activation names and their descriptions.

**Returns:**

| Type | Description |
|------|-------------|
| `dict[str, str]` | Mapping of uppercase name to description string |

---

### Built-in Activations

All activations are callable classes with signature `__call__(self, x: jax.Array) -> jax.Array`.

---

#### Standard Wrappers

Thin wrappers around `flax.nnx` and `jax.numpy` built-ins.

| Name | Formula | Parameters |
|------|---------|------------|
| `RELU` | `max(0, x)` | none |
| `LEAKY_RELU` | `max(negative_slope * x, x)` | `negative_slope: float = 0.01` |
| `SILU` | `x * sigmoid(x)` | none |
| `SIGMOID` | `1 / (1 + exp(-x))` | none |
| `TANH` | `tanh(x)` | none |
| `GELU` | Gaussian error linear unit | `approximate: bool = True` |
| `ELU` | `x if x > 0 else alpha * (exp(x) - 1)` | `alpha: float = 1.0` |
| `SELU` | Scaled ELU | none |
| `SOFTPLUS` | `log(1 + exp(x))` | none |
| `IDENTITY` | `x` | none |

---

#### `SINE`

```
SineActivation(omega: float = 30.0)
```

Applies `sin(omega * x)`. Standard SIREN activation for implicit neural representations.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `omega` | `float` | Frequency parameter | `30.0` |

---

#### `FINER`

```
SineFinerActivation(omega: float = 30.0)
```

Applies `sin(omega * alpha(x) * x)` where `alpha(x) = |x| + 1`. The adaptive factor increases effective frequency for larger-magnitude inputs, improving representational capacity over standard SIREN.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `omega` | `float` | Frequency parameter | `30.0` |

---

#### `GAUSSIAN`

```
GaussianActivation(sigma: float = 10.0)
```

Applies `exp(-(sigma * x)^2)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `sigma` | `float` | Width parameter | `10.0` |

---

#### `GAUSSIAN_FINER`

```
GaussianFinerActivation(sigma: float = 10.0, omega: float = 30.0)
```

Applies `exp(-((sigma/omega) * sin(omega * alpha(x) * x))^2)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `sigma` | `float` | Width parameter | `10.0` |
| `omega` | `float` | Frequency parameter | `30.0` |

---

#### `WIRE`

```
WireActivation(omega_0: float = 20.0, sigma_0: float = 10.0)
```

Applies `exp(j * omega_0 * x) * exp(-(sigma_0 * |x|)^2)`.

Returns a **complex-valued** array. Intended for networks explicitly designed for complex arithmetic. Passing the output to a standard real-valued `Dense` layer will raise a dtype error. Use `WIRE_REAL` for a safe real-valued alternative.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `omega_0` | `float` | Frequency parameter | `20.0` |
| `sigma_0` | `float` | Width parameter | `10.0` |

---

#### `WIRE_REAL`

```
WireRealActivation(omega_0: float = 20.0, sigma_0: float = 10.0)
```

Applies `|exp(j * omega_0 * x) * exp(-(sigma_0 * |x|)^2)|`. Real-valued magnitude of the complex Gabor wavelet. Safe to use with standard real-valued `Dense` layers.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `omega_0` | `float` | Frequency parameter | `20.0` |
| `sigma_0` | `float` | Width parameter | `10.0` |

---

#### `WIRE_FINER`

```
WireFinerActivation(omega_0: float = 20.0, sigma_0: float = 10.0, omega_finer: float = 5.0)
```

WIRE with FINER-style adaptive frequency scaling via `alpha(x) = |x| + 1`. Returns a **complex-valued** array. See `WIRE` for notes on complex output. Use `WIRE_FINER_REAL` for a safe real-valued alternative.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `omega_0` | `float` | Frequency parameter | `20.0` |
| `sigma_0` | `float` | Width parameter | `10.0` |
| `omega_finer` | `float` | FINER frequency parameter | `5.0` |

---

#### `WIRE_FINER_REAL`

```
WireFinerRealActivation(omega_0: float = 20.0, sigma_0: float = 10.0, omega_finer: float = 5.0)
```

Real-valued magnitude of `WIRE_FINER`. Safe to use with standard real-valued `Dense` layers.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `omega_0` | `float` | Frequency parameter | `20.0` |
| `sigma_0` | `float` | Width parameter | `10.0` |
| `omega_finer` | `float` | FINER frequency parameter | `5.0` |

---

#### `HOSC`

```
HoscActivation(beta: float = 10.0)
```

Applies `tanh(beta * sin(x))`. Input `x` should be scaled to approximately `[-pi, pi]`; gradients become highly oscillatory for large `x` due to the periodicity of `sin`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `beta` | `float` | Scaling parameter | `10.0` |

---

#### `HOSC_FINER`

```
HoscFinerActivation(beta: float = 10.0, omega: float = 30.0)
```

Applies `tanh((beta/omega) * sin(omega * alpha(x) * x))`. Input `x` should be scaled to approximately `[-pi, pi]`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `beta` | `float` | Scaling parameter | `10.0` |
| `omega` | `float` | Frequency parameter | `30.0` |

---

#### `SINC`

```
SincActivation(omega: float = 30.0)
```

Applies `sinc(omega * x)`. Uses `jnp.sinc`, which implements the normalised sinc: `sin(pi * t) / (pi * t)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `omega` | `float` | Frequency parameter | `30.0` |

---

### Private Helpers

#### `_generate_alpha`

```
_generate_alpha(x: jax.Array) -> jax.Array
```

Computes `|x| + 1`. Always `>= 1`. Shared adaptive scaling factor used by all FINER-style activations.

---

## Module: `embeddings.py`

A registry-based embedding system for positional and spherical encodings. All embeddings are Flax Linen `nn.Module` subclasses. Instantiating a module does not run computation -- call `module.init(key, *inputs)` to initialise and `module.apply(variables, *inputs)` to run a forward pass.

---

### Registry Functions

---

#### `register_embedding`

```
register_embedding(name: str, description: str = "") -> callable
```

Class decorator that registers an embedding under a given name. Names are stored uppercase and must be unique.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name used for lookup. Stored uppercase | required |
| `description` | `str` | Short description shown in `list_embeddings()` | `""` |

**Returns:**

| Type | Description |
|------|-------------|
| `callable` | Class decorator |

**Raises:**

| Type | Condition |
|------|-----------|
| `ValueError` | If an embedding with the same name is already registered |

---

#### `get_embedding`

```
get_embedding(name: str, **kwargs) -> nn.Module
```

Retrieve and instantiate a registered embedding by name. Unknown kwargs are warned about and dropped.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name of the registered embedding (case-insensitive) | required |
| `**kwargs` | any | Arguments forwarded to the embedding constructor. Unknown kwargs trigger a `UserWarning` and are dropped | |

**Returns:**

| Type | Description |
|------|-------------|
| `nn.Module` | An instantiated Flax Linen embedding module |

**Raises:**

| Type | Condition |
|------|-----------|
| `ValueError` | If no embedding with the given name exists. Error message lists all available names |

---

#### `list_embeddings`

```
list_embeddings() -> dict[str, str]
```

Return a sorted dictionary of all registered embedding names and their descriptions.

**Returns:**

| Type | Description |
|------|-------------|
| `dict[str, str]` | Mapping of uppercase name to description string |

---

### General Embeddings

---

#### `GAUSSIAN_POSITIONAL`

```
GaussianFourierEmbedding(input_dim: int, mapping_dim: int, scale: float, seed: int = 0)
```

Random Fourier features with a Gaussian frequency matrix. Samples a fixed projection matrix `B ~ N(0, scale^2)` at init time and computes `[cos(2*pi*x*B), sin(2*pi*x*B)]`.

`B` is stored as a plain Python attribute in `setup()`, making it invisible to the optimizer and never updated during training. This is equivalent to `eqx.static_field()`. Different seeds produce different projections.

Following Tancik et al. 2020 (https://arxiv.org/abs/2006.10739).

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `input_dim` | `int` | Dimensionality of the input coordinates | required |
| `mapping_dim` | `int` | Output dimensionality. Must be even | required |
| `scale` | `float` | Standard deviation of the Gaussian used to sample frequencies | required |
| `seed` | `int` | Seed used to sample the frequency matrix | `0` |

**Attributes:**

| Name | Description |
|------|-------------|
| `out_features` | Equal to `mapping_dim` |

**Raises:**

| Type | Condition |
|------|-----------|
| `ValueError` | If `mapping_dim` is not even |

---

#### `GENERAL_POSITIONAL`

```
PositionalEmbedding(input_dim: int, mapping_dim: int, scale: float)
```

Deterministic positional encoding with geometrically spaced frequencies. Frequencies are spaced as `scale^(j / mapping_dim)` for `j = 0 ... mapping_dim-1`, broadcast across all input dimensions. With `scale=1` this reduces to standard sinusoidal positional encoding. No parameters or constants -- `variables == {}`.

Following Tancik et al. 2020 (https://arxiv.org/abs/2006.10739).

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `input_dim` | `int` | Dimensionality of the input coordinates | required |
| `mapping_dim` | `int` | Number of frequencies per input dimension | required |
| `scale` | `float` | Frequency growth base. `scale=1` gives uniform frequencies | required |

**Attributes:**

| Name | Description |
|------|-------------|
| `out_features` | Equal to `2 * mapping_dim` |

---

### Spherical Embeddings

All spherical embeddings take `lat` and `lon` in radians as separate arguments. No parameters or constants -- `variables == {}` for all Sphere2Vec variants.

---

#### `SPHERE_GRID`

```
SphericalGridEmbedding(scale: int, r_min: float, r_max: float = 1.0)
```

Independent multi-scale sinusoidal encoding of lat and lon. Applies a geometric frequency bank independently to each and concatenates sin and cos. No cross-terms between lat and lon.

`output_dim = 4 * scale`

Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `scale` | `int` | Number of frequency bins per coordinate | required |
| `r_min` | `float` | Minimum frequency | required |
| `r_max` | `float` | Maximum frequency | `1.0` |

---

#### `SPHERE_C`

```
SphericalCartesianEmbedding(scale: int, r_min: float, r_max: float = 1.0)
```

Multi-scale encoding of the 3D unit Cartesian vector. Converts `(lat, lon)` to `(cos(lat)cos(lon), cos(lat)sin(lon), sin(lat))` and applies a geometric frequency bank to each Cartesian component via `sin`.

`output_dim = 3 * scale`

Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `scale` | `int` | Number of frequency bins per Cartesian component | required |
| `r_min` | `float` | Minimum frequency | required |
| `r_max` | `float` | Maximum frequency | `1.0` |

---

#### `SPHERE_M`

```
SphericalMultiScaleEmbedding(scale: int, r_min: float, r_max: float = 1.0)
```

Multi-scale encoding mixing transformed and raw spherical coordinates. Combines multi-scale transformed lat terms with raw lon and vice versa to capture cross-coordinate interactions at different scales.

`output_dim = 5 * scale`

Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `scale` | `int` | Number of frequency bins | required |
| `r_min` | `float` | Minimum frequency | required |
| `r_max` | `float` | Maximum frequency | `1.0` |

---

#### `DFS`

```
DoubleFourierSphericalEmbedding(scale: int, r_lat_min: float, r_lon_min: float, r_max: float = 1.0)
```

Double Fourier Sphere encoding with cross-frequency interaction terms. Computes base sin/cos terms for lat and lon independently, then all pairwise products (cos\*cos, cos\*sin, sin\*cos, sin\*sin) across the scale dimension.

`output_dim = 4 * scale + 4 * scale^2`

Output grows quadratically with scale. `scale <= 16` is recommended.

Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `scale` | `int` | Number of frequency bins per coordinate | required |
| `r_lat_min` | `float` | Minimum frequency for latitude | required |
| `r_lon_min` | `float` | Minimum frequency for longitude | required |
| `r_max` | `float` | Maximum frequency | `1.0` |

---

#### `SPHERE_C+`

```
SphericalCartesianPlusEmbedding(scale: int, r_min: float, r_max: float = 1.0)
```

Sphere-C augmented with independent lat/lon sinusoidal terms. Extends `SPHERE_C` by adding `sin/cos` of transformed lat and lon directly alongside the Cartesian component encoding.

`output_dim = 6 * scale`

Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `scale` | `int` | Number of frequency bins | required |
| `r_min` | `float` | Minimum frequency | required |
| `r_max` | `float` | Maximum frequency | `1.0` |

---

#### `SPHERE_M+`

```
SphericalMultiScalePlusEmbedding(scale: int, r_min: float, r_max: float = 1.0)
```

Sphere-M augmented with independent transformed lat/lon sin/cos terms. Extends `SPHERE_M` by adding `sin(lat_t)`, `sin(lon_t)`, `cos(lon_t)`.

`output_dim = 8 * scale`

Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `scale` | `int` | Number of frequency bins | required |
| `r_min` | `float` | Minimum frequency | required |
| `r_max` | `float` | Maximum frequency | `1.0` |

---

#### `SPHERICAL_HARMONICS`

```
SphericalHarmonicsEmbedding(legendre_polys: int = 10)
```

Real spherical harmonic basis functions as positional encoding. Evaluates `Y_l^m(lat, lon)` for degrees `0` to `legendre_polys - 1`. Unlike DFS and Sphere2Vec variants, spherical harmonics are natively defined on the sphere and produce no pole artifacts.

For each degree `l` there are `2l + 1` basis functions, giving `legendre_polys^2` total features.

Normalisation constants are precomputed in `setup()` using `math.factorial` and stored as a static array -- never traced by JAX. No parameters or constants in the Flax variable system -- `variables == {}`.

`output_dim = legendre_polys^2`

Following Russwurm et al. 2024 (https://arxiv.org/abs/2310.06743).

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `legendre_polys` | `int` | Number of degrees (0 to `legendre_polys - 1`). Recommended range 5-20. Values above 20 may accumulate numerical errors in the Legendre recursion | `10` |

**Attributes:**

| Name | Description |
|------|-------------|
| `out_features` | Equal to `legendre_polys^2` |

**Notes:**

`lat` must be in radians in `[-pi/2, pi/2]`. `lon` must be in radians in `[-pi, pi]`. Internally converts to colatitude and longitude in SH convention.

---

### Private Helpers

#### `_make_beta`

```
_make_beta(scale: int, r_min: float, r_max: float) -> jax.Array
```

Geometric frequency schedule from `r_min` to `r_max` over `scale` steps. Shared by all Sphere2Vec embeddings. Returns shape `(scale,)`.

---

## Module: `initializations.py`

Weight initializers are plain callable classes with signature `(key: jax.Array, shape: tuple, dtype) -> jax.Array`. They can be passed directly to `nn.Dense` as `kernel_init` or `bias_init`.

---

### Registry Functions

---

#### `register_initializer`

```
register_initializer(name: str, description: str = "") -> callable
```

Class decorator that registers an initializer under a given name. Names are stored uppercase and must be unique.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name used for lookup. Stored uppercase | required |
| `description` | `str` | Short description shown in `list_initializers()` | `""` |

**Returns:**

| Type | Description |
|------|-------------|
| `callable` | Class decorator |

**Raises:**

| Type | Condition |
|------|-----------|
| `ValueError` | If an initializer with the same name is already registered |

---

#### `get_initializer`

```
get_initializer(name: str, **kwargs) -> callable
```

Retrieve and instantiate a registered initializer by name. Unknown kwargs are warned about and dropped.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name of the registered initializer (case-insensitive) | required |
| `**kwargs` | any | Arguments forwarded to the initializer constructor. Unknown kwargs trigger a `UserWarning` and are dropped | |

**Returns:**

| Type | Description |
|------|-------------|
| `callable` | An instantiated initializer with signature `(key: jax.Array, shape: tuple, dtype) -> jax.Array` |

**Raises:**

| Type | Condition |
|------|-----------|
| `ValueError` | If no initializer with the given name exists. Error message lists all available names |

---

#### `list_initializers`

```
list_initializers() -> dict[str, str]
```

Return a sorted dictionary of all registered initializer names and their descriptions.

**Returns:**

| Type | Description |
|------|-------------|
| `dict[str, str]` | Mapping of uppercase name to description string |

---

### SIREN Initializers

---

#### `SIREN`

```
SirenInit(fan_in: int, is_first: bool = False, omega: float = 30.0)
```

SIREN weight initializer (Sitzmann et al. 2020).

First layer: `U(-1/fan_in, 1/fan_in)`
Hidden layers: `U(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)`

Note: the first-layer bound is only wider than the hidden-layer bound when `fan_in > omega^2 / 6` (150 for the default `omega=30`). For smaller `fan_in` the hidden-layer bound is wider.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `fan_in` | `int` | Number of input features to the layer | required |
| `is_first` | `bool` | If `True`, use first-layer bounds | `False` |
| `omega` | `float` | Frequency parameter | `30.0` |

---

### FINER Initializers

---

#### `FINER`

```
FinerInit(fan_in: int, is_first: bool = False, omega: float = 30.0)
```

FINER kernel initializer (Liu et al. 2024). Uses the same weight bounds as `SIREN`. Pair with `FINER_BIAS` for the full FINER init scheme.

First layer: `U(-1/fan_in, 1/fan_in)`
Hidden layers: `U(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)`

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `fan_in` | `int` | Number of input features to the layer | required |
| `is_first` | `bool` | If `True`, use first-layer bounds | `False` |
| `omega` | `float` | Frequency parameter | `30.0` |

---

#### `FINER_BIAS`

```
FinerBiasInit(k: float = 1.0)
```

FINER bias initializer. Draws from `U(-k, k)`. In Flax, kernel and bias inits are passed separately to `nn.Dense`; this provides the bias component of the FINER init scheme.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `k` | `float` | Half-range of the uniform distribution | `1.0` |

---

### Xavier Initializers

Both variants are implemented directly from the standard formulas rather than delegating to Flax, to avoid version-dependent behaviour.

---

#### `XAVIER_UNIFORM`

```
XavierUniformInit(gain: float = 1.0)
```

Draws from `U(-bound, bound)` where `bound = gain * sqrt(6 / (fan_in + fan_out))`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `gain` | `float` | Scaling factor applied to the standard bound | `1.0` |

---

#### `XAVIER_NORMAL`

```
XavierNormalInit(gain: float = 1.0)
```

Draws from `N(0, std^2)` where `std = gain * sqrt(2 / (fan_in + fan_out))`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `gain` | `float` | Scaling factor applied to the standard deviation | `1.0` |

---

### Standard Initializers

---

#### `LECUN_NORMAL`

```
LeCunNormalInit(scale: float = 1.0)
```

Draws from `N(0, std^2)` where `std = scale / sqrt(fan_in)`. Default for most JAX/Flax MLPs.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `scale` | `float` | Scaling factor applied to std | `1.0` |

---

#### `NORMAL`

```
NormalInit(mean: float = 0.0, std: float = 0.1)
```

Draws from `N(mean, std^2)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `mean` | `float` | Mean of the distribution | `0.0` |
| `std` | `float` | Standard deviation | `0.1` |

---

#### `UNIFORM`

```
UniformInit(a: float = -0.1, b: float = 0.1)
```

Draws from `U(a, b)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `a` | `float` | Lower bound | `-0.1` |
| `b` | `float` | Upper bound | `0.1` |

---

#### `IDENTITY`

```
IdentityInit()
```

Initialises a square weight matrix as the identity. Raises `ValueError` for non-square shapes.

---

#### `ORTHOGONAL`

```
OrthogonalInit(gain: float = 1.0)
```

Orthogonal matrix initialization via Flax. Output is scaled by `gain`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `gain` | `float` | Scaling factor applied to the orthogonal matrix | `1.0` |

---

### MFN / WIRE Initializers

---

#### `GABOR`

```
GaborInit(std_scale: float = 1.0)
```

Gabor filter weight initializer for Multiplicative Filter Networks (Fathony et al. 2021). Draws from `N(0, std^2)` where `std = std_scale / sqrt(fan_in)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `std_scale` | `float` | Scales the standard deviation relative to `1/sqrt(fan_in)` | `1.0` |

---

#### `WIRE`

```
WireInit(gain: float = 1.0)
```

Complex weight initializer for WIRE networks. Draws real and imaginary parts independently from `N(0, std^2)` where `std = gain * sqrt(2 / (fan_in + fan_out))`. The default output dtype is `complex64`. Passing `dtype=jnp.complex128` is supported but requires `jax_enable_x64=True`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `gain` | `float` | Scaling factor applied to std | `1.0` |

**Notes:**

Default `dtype` is `complex64`. The float backing dtype (`float32` or `float64`) is inferred automatically from the requested complex dtype.