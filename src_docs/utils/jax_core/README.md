# JAX Core Utilities

This package provides core utility functions for JAX/Flax development, organized into two modules:

- **`helpers.py`**: General-purpose JAX helper functions for environment checking, RNG management, and function inspection
- **`diagnostics.py`**: Visualization and diagnostic tools for neural network training analysis

---

## Module: `helpers.py`

General-purpose JAX utilities for common operations.

### `check_environment()`

Check JAX environment and print details for GPUs, backend, JAX version, and available memory.

**Example:**
```python
>>> check_environment()
JAX version: 0.4.35
Backend: gpu
GPUs: 1 available
  [0] NVIDIA A100-SXM4-40GB - 40.0 GB
```

---

### `create_rng(seed: int = 42) -> jax.Array`

Create a JAX PRNGKey from a seed for reproducible randomness.

**Parameters:**
- `seed` (int): Random seed for reproducibility

**Returns:**
- `jax.Array`: A JAX PRNGKey array

**Example:**
```python
>>> create_rng(0)
Array([0, 0], dtype=uint32)
```

---

### `create_rng_dict(seed: int = 42, keys: list[str] | None = None) -> dict[str, jax.Array]`

Create a dictionary of PRNGKeys from a single seed. Splits a root key into named subkeys for use with Flax model init and apply calls.

**Parameters:**
- `seed` (int): Random seed for reproducibility
- `keys` (list[str], optional): Key names for the dictionary. Defaults to ["params", "dropout"]

**Returns:**
- `dict[str, jax.Array]`: Dictionary mapping key names to their respective PRNGKeys

**Example:**
```python
>>> rngs = create_rng_dict(0, keys=["params", "dropout"])
>>> list(rngs.keys())
['params', 'dropout']
>>> rngs["params"].shape
(2,)
```

---

### `split_rng(rng: jax.Array) -> tuple[jax.Array, jax.Array]`

Split a PRNGKey into a new root key and a subkey. This is the standard JAX pattern to avoid accidental key reuse.

**Parameters:**
- `rng` (jax.Array): Current PRNGKey to split

**Returns:**
- `tuple[jax.Array, jax.Array]`: A pair of (new_rng, subkey)

**Example:**
```python
>>> rng = create_rng(0)
>>> rng, key = split_rng(rng)
>>> jax.random.normal(key)
Array(-1.2515389, dtype=float32)
```

---

### `show_jaxpr(fn: Callable, *sample_inputs, static_argnums: int | tuple[int, ...] | None = None) -> None`

Print the jaxpr (JAX expression) representation of a function.

**Parameters:**
- `fn` (callable): Function to trace
- `*sample_inputs`: Example inputs that define the shapes and dtypes for tracing
- `static_argnums` (int or tuple[int, ...], optional): Positional arguments to treat as static (not traced)

**Example:**
```python
>>> show_jaxpr(lambda x: x ** 2 + 1, jnp.array(3.0))
{ lambda ; a:f32[]. let
    b:f32[] = integer_pow[y=2] a
    c:f32[] = add b 1.0
  in (c,) }
```

---

### `grad_fn(fn: Callable, argnums: int | tuple[int, ...] = 0, has_aux: bool = False) -> Callable`

Return a function that computes both value and gradients.

**Parameters:**
- `fn` (callable): Function to differentiate
- `argnums` (int or tuple[int]): Which positional argument(s) to differentiate with respect to
- `has_aux` (bool): If True, `fn` returns a pair `(value, aux)` and the gradient is computed with respect to `value` only

**Returns:**
- `callable`: A function that returns `(value, grads)` or `((value, aux), grads)`

**Example:**
```python
>>> f = lambda x: x ** 3
>>> val_and_grad = grad_fn(f)
>>> val_and_grad(2.0)
(Array(8., dtype=float32), Array(12., dtype=float32))
```

---

## Module: `diagnostics.py`

Visualization and diagnostic tools for neural network training analysis.

### Activation Function Visualization

#### `get_grads(act_fn: Callable[[float], float], x: jax.Array) -> jax.Array`

Compute the gradient of a scalar-to-scalar function at each point in x. Uses vmap to efficiently compute per-element gradients, which is useful for visualizing activation function derivatives.

**Parameters:**
- `act_fn` (callable): A scalar-to-scalar function (e.g., an activation function)
- `x` (jax.Array): 1D input array of points to evaluate gradients at

**Returns:**
- `jax.Array`: Array of same shape as x containing gradients of act_fn at each point

**Example:**
```python
>>> x = jnp.linspace(-3, 3, 5)
>>> get_grads(jax.nn.relu, x)
Array([0., 0., 0., 1., 1.], dtype=float32)
```

---

#### `vis_act_fn(act_fn: Callable, ax: plt.Axes, x: jax.Array) -> None`

Plot an activation function and its gradient on a given axis.

**Parameters:**
- `act_fn` (callable): A scalar-to-scalar activation function (e.g., `jax.nn.relu`)
- `ax` (matplotlib.axes.Axes): Matplotlib axis to plot on
- `x` (jax.Array): 1D input array of points to evaluate

**Example:**
```python
>>> import matplotlib.pyplot as plt
>>> fig, axes = plt.subplots(1, 3, figsize=(12, 4))
>>> x = jnp.linspace(-5, 5, 200)
>>> vis_act_fn(jax.nn.relu, axes[0], x)
>>> vis_act_fn(jax.nn.sigmoid, axes[1], x)
>>> vis_act_fn(jax.nn.gelu, axes[2], x)
>>> plt.tight_layout()
>>> plt.show()
```

---

### Weight, Gradient, and Activation Distribution Visualization

#### `visualize_weight_distribution(params: dict, color: str = "C0") -> None`

Plot histograms of weight values per layer.

**Parameters:**
- `params` (dict): Model parameters (e.g., from `model.init(...)`)
- `color` (str): Histogram color

**Example:**
```python
>>> model = nn.Dense(features=10)
>>> params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
>>> visualize_weight_distribution(params)
```

---

#### `visualize_gradients(params: dict, loss_fn: Callable[[dict], float], color: str = "C0", print_variance: bool = False) -> None`

Plot histograms of per-layer gradient magnitudes.

**Parameters:**
- `params` (dict): Model parameters
- `loss_fn` (callable): Function of signature `(params) -> scalar loss`. The model and batch should be captured in the closure
- `color` (str): Histogram color
- `print_variance` (bool): If True, print the variance of each layer's gradients

**Example:**
```python
>>> def loss_fn(p):
...     logits = model.apply(p, batch)
...     return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
>>> visualize_gradients(params, loss_fn)
```

---

#### `visualize_activations(net: "nn.Module", params: dict, batch: jax.Array, color: str = "C0", print_variance: bool = False) -> None`

Plot histograms of per-layer activation distributions. Uses Flax's `capture_intermediates` to collect activations without modifying the model.

**Parameters:**
- `net` (flax.linen.Module): Flax module
- `params` (dict): Model parameters
- `batch` (jax.Array): Input batch to pass through the network
- `color` (str): Histogram color
- `print_variance` (bool): If True, print the variance of each layer's activations

**Example:**
```python
>>> visualize_activations(model, params, batch, print_variance=True)
Layer 0 - Variance: 0.0822
Layer 1 - Variance: 0.0042
```

---

### Layer-level Data Extraction

#### `get_layer_gradients(params: dict, loss_fn: Callable[[dict], float], include_bias: bool = False) -> list[jnp.ndarray]`

Compute per-layer weight gradients for diagnostic visualization.

**Parameters:**
- `params` (dict): Model parameters
- `loss_fn` (callable): Function of signature (params) -> scalar loss
- `include_bias` (bool, optional): If True, include bias gradients (1-D arrays). Default False, which only includes weight matrices (ndim > 1)

**Returns:**
- `list[jnp.ndarray]`: Flattened gradient arrays for each parameter (weights and optionally biases)

---

#### `get_layer_activations(net: "nn.Module", params: dict, batch: jax.Array) -> dict`

Capture per-layer activations via Flax's capture_intermediates.

**Parameters:**
- `net` (flax.linen.Module): Flax module
- `params` (dict): Model parameters
- `batch` (jax.Array): Input batch

**Returns:**
- `dict`: Intermediate activations keyed by layer path

---

### Dead / Inactive Neuron Analysis

#### `count_inactive_neurons(activations: dict, threshold: float = 1e-6, mode: str = "magnitude", aggregate_spatial: bool = False, spatial_reduction: str = "mean", channel_axis: int = -1) -> dict[str, dict[str, int | float]]`

Count neurons that show minimal variation or activation across a batch. Works with any activation function.

The `mode` parameter controls what "inactive" means:
- **"magnitude"**: neuron's max absolute activation never exceeds threshold. Catches dead ReLU and near-zero outputs
- **"variance"**: neuron's variance across the batch is below threshold. Catches saturated tanh/sigmoid stuck at one value
- **"both"**: neuron must fail both checks (strictest)

**Parameters:**
- `activations` (dict): Output of get_layer_activations (nested dict of (value,) tuples)
- `threshold` (float): Cutoff for considering a neuron inactive
- `mode` (str): 'magnitude', 'variance', or 'both'
- `aggregate_spatial` (bool): If True and the activation has spatial dimensions (e.g., conv layers with shape `(batch, h, w, channels)`), each channel is treated as one neuron by reducing over spatial dimensions. Also works for transformer activations with shape `(batch, seq_len, features)` where each feature is treated as one neuron averaged over sequence positions. If False, every spatial position is counted independently
- `spatial_reduction` (str, optional): How to reduce spatial dimensions when aggregate_spatial is True. One of 'mean', 'max', or 'sum'. Ignored if aggregate_spatial is False
- `channel_axis` (int, optional): Axis that holds channels/features. Default -1 (last axis). Batch axis is assumed to be 0. All other axes are treated as spatial

**Returns:**
- `dict[str, dict]`: Per-layer results with keys: "inactive", "total", "percent"

---

### Loss Landscape Visualization

#### `plot_loss_landscape(params: dict, loss_fn: Callable[[dict], float], grid_size: int = 50, range_scale: float = 1.0, seed: int = 0, plot_3d: bool = False, cmap: str = "viridis") -> None`

Plot a 2D slice of the loss landscape around a set of parameters. Projects the high-dimensional loss surface onto two random directions in parameter space, following the approach from Li et al. (2018), "Visualizing the Loss Landscape of Neural Nets."

This is qualitative, not exhaustive. The slice is random so different seeds will show different cross-sections. Useful for comparing sharp vs flat minima across training runs or architectures.

**Parameters:**
- `params` (dict): Model parameters (the center point of the plot)
- `loss_fn` (callable): Function of signature `(params) -> scalar loss`. The model and batch should be captured in the closure
- `grid_size` (int): Number of points per axis. Total evaluations = grid_size^2. 50 is a reasonable default; reduce for large models
- `range_scale` (float): How far to sweep in each direction. Larger values show more of the landscape but may miss local structure
- `seed` (int): Random seed for generating the two direction vectors
- `plot_3d` (bool): If True, render a 3D surface plot. Otherwise a 2D heatmap
- `cmap` (str): Matplotlib colormap name

**Example:**
```python
>>> def loss_fn(p):
...     logits = model.apply(p, batch)
...     return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
>>> plot_loss_landscape(params, loss_fn, grid_size=30)
```

---

## Private Helper Functions

### `_plot_dists(val_dict: dict[str, jnp.ndarray], color: str = "C0", xlabel: str | None = None, stat: str = "count", use_kde: bool = True) -> plt.Figure`

Private helper function to plot histograms for a dict of named 1D arrays. Used internally by visualization functions.

**Parameters:**
- `val_dict` (dict[str, jnp.ndarray]): Mapping of layer names to 1D arrays
- `color` (str): Histogram color
- `xlabel` (str, optional): Label for the x-axis
- `stat` (str): Seaborn histplot stat parameter ('count', 'density', etc.)
- `use_kde` (bool): Whether to overlay a KDE curve (skipped when variance is near zero)

**Returns:**
- `matplotlib.figure.Figure`
