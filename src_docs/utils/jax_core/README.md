# JAX Core Utilities

This package provides core utility functions for JAX/Flax development, organized into two modules:

- **`helpers.py`**: General-purpose JAX helper functions for environment checking, RNG management, function inspection, coordinate conversions, and data normalization
- **`diagnostics.py`**: Visualization and diagnostic tools for neural network training analysis

---

## Module: `helpers.py`

---

### `check_environment`

```
check_environment() -> None
```

Check JAX environment and print details for GPUs, backend, JAX version, and available memory. Falls back gracefully when no GPU is available.

---

### `create_rng`

```
create_rng(seed: int = 42) -> jax.Array
```

Create a JAX PRNGKey from an integer seed for reproducible randomness.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `seed` | `int` | Random seed for reproducibility | `42` |

**Returns:**

| Type | Description |
|------|-------------|
| `jax.Array` | A JAX PRNGKey array |

---

### `create_rng_dict`

```
create_rng_dict(seed: int = 42, keys: list[str] | None = None) -> dict[str, jax.Array]
```

Split a root key into a dictionary of named subkeys for use with Flax model init and apply calls.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `seed` | `int` | Random seed for reproducibility | `42` |
| `keys` | `list[str] or None` | Key names for the dictionary | `["params", "dropout"]` |

**Returns:**

| Type | Description |
|------|-------------|
| `dict[str, jax.Array]` | Dictionary mapping key names to their respective PRNGKeys |

---

### `split_rng`

```
split_rng(rng: jax.Array) -> tuple[jax.Array, jax.Array]
```

Split a PRNGKey into a new root key and a subkey. Standard JAX pattern to avoid accidental key reuse.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `rng` | `jax.Array` | Current PRNGKey to split | required |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[jax.Array, jax.Array]` | A pair of `(new_rng, subkey)` |

---

### `key_to_seed`

```
key_to_seed(key: jax.Array) -> int
```

Derive a reproducible integer seed from a JAX PRNGKey. Used to bridge JAX's functional PRNG to stateful RNGs such as `numpy.random.Generator` or `scipy.stats.qmc` samplers.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `key` | `jax.Array` | JAX PRNGKey | required |

**Returns:**

| Type | Description |
|------|-------------|
| `int` | Integer seed in `[0, 2^31 - 1]` |

---

### `show_jaxpr`

```
show_jaxpr(fn: Callable, *sample_inputs, static_argnums: int | tuple[int, ...] | None = None) -> None
```

Print the JAX expression (jaxpr) representation of a function given example inputs. Useful for debugging tracing behaviour and verifying JIT compilation.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `fn` | `Callable` | Function to trace | required |
| `*sample_inputs` | any | Example inputs that define shapes and dtypes for tracing | required |
| `static_argnums` | `int or tuple[int, ...] or None` | Positional arguments to treat as static | `None` |

---

### `grad_fn`

```
grad_fn(fn: Callable, argnums: int | tuple[int, ...] = 0, has_aux: bool = False) -> Callable
```

Return a function that computes both the value and gradients of `fn`. Wraps `jax.value_and_grad`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `fn` | `Callable` | Function to differentiate | required |
| `argnums` | `int or tuple[int, ...]` | Which positional argument(s) to differentiate with respect to | `0` |
| `has_aux` | `bool` | If True, `fn` returns `(value, aux)` and gradients are computed with respect to `value` only | `False` |

**Returns:**

| Type | Description |
|------|-------------|
| `Callable` | A function that returns `(value, grads)` or `((value, aux), grads)` |

---

### `degrees_to_radians`

```
degrees_to_radians(x: jax.Array) -> jax.Array
```

Convert an array of values from degrees to radians element-wise.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `x` | `jax.Array` | Array of values in degrees | required |

**Returns:**

| Type | Description |
|------|-------------|
| `jax.Array` | Array of values in radians |

---

### `radians_to_degrees`

```
radians_to_degrees(x: jax.Array) -> jax.Array
```

Convert an array of values from radians to degrees element-wise.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `x` | `jax.Array` | Array of values in radians | required |

**Returns:**

| Type | Description |
|------|-------------|
| `jax.Array` | Array of values in degrees |

---

### `latlon_deg_to_rad`

```
latlon_deg_to_rad(lat_deg: jax.Array, lon_deg: jax.Array) -> tuple[jax.Array, jax.Array]
```

Convert lat/lon arrays from degrees to radians.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `lat_deg` | `jax.Array` | Latitudes in degrees, shape `(N,)` or broadcastable | required |
| `lon_deg` | `jax.Array` | Longitudes in degrees, shape `(N,)` or broadcastable | required |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[jax.Array, jax.Array]` | `(lat_rad, lon_rad)` in radians, same shapes as inputs |

---

### `latlon_rad_to_deg`

```
latlon_rad_to_deg(lat_rad: jax.Array, lon_rad: jax.Array) -> tuple[jax.Array, jax.Array]
```

Convert lat/lon arrays from radians to degrees.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `lat_rad` | `jax.Array` | Latitudes in radians | required |
| `lon_rad` | `jax.Array` | Longitudes in radians | required |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[jax.Array, jax.Array]` | `(lat_deg, lon_deg)` in degrees, same shapes as inputs |

---

### `spherical_to_cartesian`

```
spherical_to_cartesian(lat_rad: jax.Array, lon_rad: jax.Array) -> jax.Array
```

Convert spherical lat/lon in radians to unit Cartesian coordinates using the geographic convention: `x = cos(lat)cos(lon)`, `y = cos(lat)sin(lon)`, `z = sin(lat)`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `lat_rad` | `jax.Array` | Latitudes in radians, any shape broadcastable with `lon_rad` | required |
| `lon_rad` | `jax.Array` | Longitudes in radians, any shape broadcastable with `lat_rad` | required |

**Returns:**

| Type | Description |
|------|-------------|
| `jax.Array` | Unit Cartesian coordinates, shape `(*broadcast_shape, 3)` |

---

### `cartesian_to_spherical`

```
cartesian_to_spherical(xyz: jax.Array) -> tuple[jax.Array, jax.Array]
```

Convert Cartesian coordinates to spherical lat/lon in radians. Supports arbitrary leading dimensions via ellipsis indexing. Input need not be unit vectors -- only direction matters.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `xyz` | `jax.Array` | Cartesian coordinates, shape `(..., 3)` | required |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[jax.Array, jax.Array]` | `(lat_rad, lon_rad)` each shape `(...)`. lat in `[-pi/2, pi/2]`, lon in `[-pi, pi]` |

---

### `minmax_norm`

```
minmax_norm(
    x: jax.Array,
    x_min: float | jax.Array,
    x_max: float | jax.Array,
    mode: Literal["01", "-11"] = "01",
    eps: float = 1e-12,
) -> jax.Array
```

Min-max normalise an array to `[0, 1]` or `[-1, 1]`. Supports broadcasting for per-column bounds. A small `eps` is added to the denominator to guard against division by zero when `x_min == x_max`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `x` | `jax.Array` | Input array, any shape | required |
| `x_min` | `float or jax.Array` | Minimum value of the input range, supports broadcasting | required |
| `x_max` | `float or jax.Array` | Maximum value of the input range | required |
| `mode` | `"01" or "-11"` | Output range | `"01"` |
| `eps` | `float` | Small constant added to denominator | `1e-12` |

**Returns:**

| Type | Description |
|------|-------------|
| `jax.Array` | Normalised array, same shape as `x` |

**Raises:**

| Type | Condition |
|------|-----------|
| `ValueError` | If `mode` is not `"01"` or `"-11"` |

---

### `standardise`

```
standardise(
    x: jax.Array,
    mean: float | jax.Array | None = None,
    std: float | jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    eps: float = 1e-8,
) -> jax.Array
```

Standardise an array to zero mean and unit variance. If `mean` and `std` are not provided they are computed from `x` over `axis`. Pass pre-computed statistics to standardise a test set using training set statistics. Uses `keepdims=True` internally so broadcasting works for any axis choice.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `x` | `jax.Array` | Input array, any shape | required |
| `mean` | `float or jax.Array or None` | Mean to subtract. Computed from `x` if not provided | `None` |
| `std` | `float or jax.Array or None` | Standard deviation to divide by. Computed from `x` if not provided | `None` |
| `axis` | `int or tuple[int, ...] or None` | Axis or axes over which to compute statistics | `None` |
| `eps` | `float` | Small constant added to `std` to avoid division by zero | `1e-8` |

**Returns:**

| Type | Description |
|------|-------------|
| `jax.Array` | Standardised array, same shape as `x` |

---

## Module: `diagnostics.py`

---

### `get_grads`

```
get_grads(act_fn: Callable[[float], float], x: jax.Array) -> jax.Array
```

Compute per-element gradients of a scalar-to-scalar function using `vmap`. The function must map a single float to a single float.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `act_fn` | `Callable` | A scalar-to-scalar function | required |
| `x` | `jax.Array` | 1D input array of points to evaluate gradients at | required |

**Returns:**

| Type | Description |
|------|-------------|
| `jax.Array` | Array of same shape as `x` containing gradients of `act_fn` at each point |

---

### `vis_act_fn`

```
vis_act_fn(act_fn: Callable, ax: plt.Axes, x: jax.Array) -> None
```

Plot an activation function and its gradient on a given matplotlib axis.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `act_fn` | `Callable` | A scalar-to-scalar activation function | required |
| `ax` | `plt.Axes` | Matplotlib axis to plot on | required |
| `x` | `jax.Array` | 1D input array of points to evaluate | required |

---

### `visualize_weight_distribution`

```
visualize_weight_distribution(params: dict, color: str = "C0") -> None
```

Plot histograms of weight values per layer from a Flax parameter dict.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `params` | `dict` | Model parameters from `model.init(...)` | required |
| `color` | `str` | Histogram color | `"C0"` |

---

### `visualize_gradients`

```
visualize_gradients(
    params: dict,
    loss_fn: Callable[[dict], float],
    color: str = "C0",
    print_variance: bool = False,
) -> None
```

Plot histograms of per-layer gradient magnitudes. `loss_fn` must have signature `(params) -> scalar` with the model and batch captured in the closure.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `params` | `dict` | Model parameters | required |
| `loss_fn` | `Callable[[dict], float]` | Function returning a scalar loss | required |
| `color` | `str` | Histogram color | `"C0"` |
| `print_variance` | `bool` | If True, print the variance of each layer's gradients | `False` |

---

### `visualize_activations`

```
visualize_activations(
    net: nn.Module,
    params: dict,
    batch: jax.Array,
    color: str = "C0",
    print_variance: bool = False,
) -> None
```

Plot histograms of per-layer activation distributions using Flax's `capture_intermediates`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `net` | `nn.Module` | Flax module | required |
| `params` | `dict` | Model parameters | required |
| `batch` | `jax.Array` | Input batch to pass through the network | required |
| `color` | `str` | Histogram color | `"C0"` |
| `print_variance` | `bool` | If True, print the variance of each layer's activations | `False` |

---

### `get_layer_gradients`

```
get_layer_gradients(
    params: dict,
    loss_fn: Callable[[dict], float],
    include_bias: bool = False,
) -> list[jnp.ndarray]
```

Compute per-layer weight gradients for diagnostic visualisation.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `params` | `dict` | Model parameters | required |
| `loss_fn` | `Callable[[dict], float]` | Function returning a scalar loss | required |
| `include_bias` | `bool` | If True, include bias gradients (1D arrays) | `False` |

**Returns:**

| Type | Description |
|------|-------------|
| `list[jnp.ndarray]` | Flattened gradient arrays for each parameter |

---

### `get_layer_activations`

```
get_layer_activations(net: nn.Module, params: dict, batch: jax.Array) -> dict
```

Capture per-layer activations via Flax's `capture_intermediates`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `net` | `nn.Module` | Flax module | required |
| `params` | `dict` | Model parameters | required |
| `batch` | `jax.Array` | Input batch | required |

**Returns:**

| Type | Description |
|------|-------------|
| `dict` | Intermediate activations keyed by layer path |

---

### `count_inactive_neurons`

```
count_inactive_neurons(
    activations: dict,
    threshold: float = 1e-6,
    mode: str = "magnitude",
    aggregate_spatial: bool = False,
    spatial_reduction: str = "mean",
    channel_axis: int = -1,
) -> dict[str, dict[str, int | float]]
```

Count neurons that show minimal activation or variation across a batch.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `activations` | `dict` | Output of `get_layer_activations` | required |
| `threshold` | `float` | Cutoff for considering a neuron inactive | `1e-6` |
| `mode` | `str` | One of `"magnitude"`, `"variance"`, or `"both"` | `"magnitude"` |
| `aggregate_spatial` | `bool` | If True, reduce spatial dimensions before counting | `False` |
| `spatial_reduction` | `str` | One of `"mean"`, `"max"`, `"sum"` | `"mean"` |
| `channel_axis` | `int` | Axis holding channels/features | `-1` |

**Returns:**

| Type | Description |
|------|-------------|
| `dict[str, dict]` | Per-layer results with keys `"inactive"`, `"total"`, `"percent"` |

**Raises:**

| Type | Condition |
|------|-----------|
| `ValueError` | If `mode` is not one of `"magnitude"`, `"variance"`, `"both"` |
| `ValueError` | If `spatial_reduction` is not one of `"mean"`, `"max"`, `"sum"` |

**Notes:**

| mode | inactive definition |
|------|-------------------|
| `"magnitude"` | max absolute activation across batch never exceeds threshold |
| `"variance"` | variance across batch is below threshold |
| `"both"` | must fail both checks -- strictest |

---

### `plot_loss_landscape`

```
plot_loss_landscape(
    params: dict,
    loss_fn: Callable[[dict], float],
    grid_size: int = 50,
    range_scale: float = 1.0,
    seed: int = 0,
    plot_3d: bool = False,
    cmap: str = "viridis",
) -> None
```

Plot a 2D slice of the loss landscape around a set of parameters by projecting onto two random normalised directions in parameter space. Based on Li et al. (2018), "Visualizing the Loss Landscape of Neural Nets." Different seeds produce different cross-sections.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `params` | `dict` | Model parameters -- the centre point of the plot | required |
| `loss_fn` | `Callable[[dict], float]` | Function returning a scalar loss | required |
| `grid_size` | `int` | Number of points per axis; total evaluations = `grid_size^2` | `50` |
| `range_scale` | `float` | How far to sweep in each direction | `1.0` |
| `seed` | `int` | Random seed for generating direction vectors | `0` |
| `plot_3d` | `bool` | If True, render a 3D surface plot instead of a 2D heatmap | `False` |
| `cmap` | `str` | Matplotlib colormap | `"viridis"` |

---

### `model_tabulate`

```
model_tabulate(
    model: nn.Module,
    *init_inputs,
    mutable: Sequence[str] | None = None,
    seed: int = 0,
) -> None
```

Print a Flax tabulate summary including frozen constants. Wraps `nn.tabulate` with `mutable=["params", "constants"]` by default so frozen buffers such as positional encodings and fixed projection matrices appear alongside trainable parameters.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `model` | `nn.Module` | Any Flax module | required |
| `*init_inputs` | any | All positional arguments required by `model.init` for shape tracing | required |
| `mutable` | `Sequence[str] or None` | Variable collections to include in the table | `["params", "constants"]` |
| `seed` | `int` | Seed for the PRNGKey passed to `nn.tabulate` | `0` |

---

### `plot_output_at_init`

```
plot_output_at_init(
    model: nn.Module,
    init_inputs: tuple,
    grid_inputs: tuple,
    shape: tuple[int, int],
    seed: int = 0,
    cmap: str = "RdBu_r",
    title: str = "Model output at init",
    extent: list | None = None,
    view: str = "cartesian",
    lon_grid: np.ndarray | None = None,
    lat_grid: np.ndarray | None = None,
) -> None
```

Run a single forward pass at initialisation over a pre-built coordinate grid and plot the result. The caller is responsible for building `grid_inputs` in the correct coordinate system. For `view="mollweide"` the caller must supply `lon_grid` and `lat_grid` as 2D arrays of shape `shape` in radians.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `model` | `nn.Module` | Any Flax module | required |
| `init_inputs` | `tuple` | Small subset of inputs used for `model.init` shape tracing | required |
| `grid_inputs` | `tuple` | Full grid inputs passed to `model.apply` for the plot | required |
| `shape` | `tuple[int, int]` | `(rows, cols)` to reshape the flat model output into | required |
| `seed` | `int` | PRNGKey seed for init | `0` |
| `cmap` | `str` | Matplotlib colormap | `"RdBu_r"` |
| `title` | `str` | Plot title | `"Model output at init"` |
| `extent` | `list or None` | `[xmin, xmax, ymin, ymax]` for imshow. Only used when `view="cartesian"` | `None` |
| `view` | `str` | `"cartesian"` uses imshow; `"mollweide"` uses pcolormesh on a Mollweide projection | `"cartesian"` |
| `lon_grid` | `np.ndarray or None` | 2D array of longitudes in radians, shape `shape`. Required when `view="mollweide"` | `None` |
| `lat_grid` | `np.ndarray or None` | 2D array of latitudes in radians, shape `shape`. Required when `view="mollweide"` | `None` |

**Raises:**

| Type | Condition |
|------|-----------|
| `ValueError` | If `view="mollweide"` and `lon_grid` or `lat_grid` is not provided |
| `AssertionError` | If model output size does not match `np.prod(shape)` |