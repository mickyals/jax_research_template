# Core

Reusable neural building blocks for JAX/Flax NNX models.

- **`activations.py`**: Activation function registry with standard, SIREN, FINER, Gaussian, WIRE, HOSC, and Sinc variants
- **`layers.py`**: Attention, cross-attention, set aggregation, feed-forward blocks, layer norm, type-specific projection layers, masked token padding *(in development)*
- **`embeddings.py`**: Cyclic time encoding, Fourier features, spatial positional encodings, delta-t encodings, learned vs fixed variants *(in development)*
- **`nets.py`**: Encoder, probabilistic field network, hypernetwork variant, full model *(in development)*

---

## Module: `activations.py`

A registry-based activation system. Activations are registered by name and retrieved via `get_activation`. The registry is extensible -- custom activations can be added via the `@register_activation` decorator without modifying the module.

---

### Registry Functions

---

#### `register_activation`

```
register_activation(name: str, description: str = "") -> callable
```

Class decorator that registers an activation under a given name. Names are stored uppercase and must be unique across the registry.

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

Retrieve and instantiate a registered activation by name. Inspects the constructor signature and emits a `UserWarning` for any kwargs not accepted by the activation class. Unknown kwargs are dropped rather than forwarded to prevent a `TypeError` at instantiation.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name of the registered activation (case-insensitive) | required |
| `**kwargs` | any | Arguments forwarded to the activation constructor. Unknown kwargs trigger a `UserWarning` and are dropped | |

**Returns:**

| Type | Description |
|------|-------------|
| `callable` | An instantiated activation function |

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

Thin wrappers around `flax.nnx` and `jax.numpy` built-ins. Requires `flax >= 0.8.0` for NNX support.

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

Applies `sin(omega * alpha(x) * x)` where `alpha(x) = |x| + 1`. The adaptive scaling factor increases effective frequency for inputs with larger magnitude, improving representational capacity over standard SIREN.

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

Returns a **complex-valued** array. Intended for networks explicitly designed for complex arithmetic. Passing the output to a standard real-valued Dense layer will raise a dtype error. Use `WIRE_REAL` for a safe real-valued alternative.

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

Applies `|exp(j * omega_0 * x) * exp(-(sigma_0 * |x|)^2)|`. Real-valued version of `WIRE` that returns the magnitude of the complex Gabor wavelet. Safe to use with standard real-valued Dense layers.

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

WIRE with FINER-style adaptive frequency scaling via `alpha(x) = |x| + 1`. Returns a **complex-valued** array. See `WIRE` for notes on complex output and downstream dtype compatibility. Use `WIRE_FINER_REAL` for a safe real-valued alternative.

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

Real-valued version of `WIRE_FINER` that returns the magnitude of the complex output. Safe to use with standard real-valued Dense layers.

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

Applies `tanh(beta * sin(x))`. Input `x` should be scaled to approximately `[-pi, pi]` since `sin(x)` is periodic and gradients become highly oscillatory for large `x`.

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

Applies `sinc(omega * x) = sin(pi * omega * x) / (pi * omega * x)`. Uses `jnp.sinc` which implements the normalised sinc function.

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

Computes `|x| + 1`. Always `>= 1`, providing a positive adaptive scaling factor used by all FINER-style activations to increase effective frequency for inputs with larger magnitude.