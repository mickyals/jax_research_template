import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Sequence, Optional


# ---------------------------------------------------------------------------
# Private plotting helper
# ---------------------------------------------------------------------------

def _plot_dists(val_dict: dict[str, jnp.ndarray], color: str = "C0",
                xlabel: str | None = None, stat: str = "count",
                use_kde: bool = True) -> plt.Figure:
    """Plot histograms for a dict of named 1D arrays.

    Parameters
    ----------
    val_dict : dict[str, jnp.ndarray]
        Mapping of layer names to 1D arrays.
    color : str
        Histogram color.
    xlabel : str, optional
        Label for the x-axis.
    stat : str
        Seaborn histplot stat parameter ('count', 'density', etc.).
    use_kde : bool
        Whether to overlay a KDE curve (skipped when variance is near zero).

    Returns
    -------
    matplotlib.figure.Figure
    """
    columns = len(val_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 3.5, 2.5))
    if columns == 1:
        ax = [ax]
    for idx, key in enumerate(sorted(val_dict.keys())):
        vals = val_dict[key]
        has_variance = float(vals.max() - vals.min()) > 1e-8
        sns.histplot(vals, ax=ax[idx], color=color, bins=50,
                     stat=stat, kde=use_kde and has_variance)
        ax[idx].set_title(key)
        if xlabel is not None:
            ax[idx].set_xlabel(xlabel)
    fig.subplots_adjust(wspace=0.4)
    return fig


# ---------------------------------------------------------------------------
# Activation function visualization
# ---------------------------------------------------------------------------

def get_grads(act_fn: Callable[[float], float], x: jax.Array) -> jax.Array:
    """Compute the gradient of a scalar-to-scalar function at each point in x.

    Uses vmap to efficiently compute per-element gradients, which is
    useful for visualizing activation function derivatives. The function
    must map a single float to a single float (e.g. ``jax.nn.relu``,
    ``jax.nn.sigmoid``). Batched or multi-output functions will fail.

    Parameters
    ----------
    act_fn : callable
        A scalar-to-scalar function (e.g. an activation function).
    x : jax.Array
        1D input array of points to evaluate gradients at.

    Returns
    -------
    jax.Array
        Array of same shape as x containing gradients of act_fn at each point.

    Example
    -------
    >>> x = jnp.linspace(-3, 3, 5)
    >>> get_grads(jax.nn.relu, x)
    Array([0., 0., 0., 1., 1.], dtype=float32)
    """
    return jax.vmap(jax.grad(act_fn))(x)


def vis_act_fn(act_fn: Callable, ax: plt.Axes, x: jax.Array) -> None:
    """Plot an activation function and its gradient on a given axis.

    Parameters
    ----------
    act_fn : callable
        A scalar-to-scalar activation function (e.g. ``jax.nn.relu``).
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on.
    x : jax.Array
        1D input array of points to evaluate.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    >>> x = jnp.linspace(-5, 5, 200)
    >>> vis_act_fn(jax.nn.relu, axes[0], x)
    >>> vis_act_fn(jax.nn.sigmoid, axes[1], x)
    >>> vis_act_fn(jax.nn.gelu, axes[2], x)
    >>> plt.tight_layout()
    >>> plt.show()
    """
    y = act_fn(x)
    y_grads = get_grads(act_fn, x)
    ax.plot(x, y, linewidth=2, label="ActFn")
    ax.plot(x, y_grads, linewidth=2, label="Gradient")
    name = getattr(act_fn, "__name__", str(act_fn))
    ax.set_title(name)
    ax.legend()
    y_min = float(min(y.min(), y_grads.min()))
    y_max = float(max(y.max(), y_grads.max()))
    margin = 0.2 * (y_max - y_min) if y_max != y_min else 0.2
    ax.set_ylim(y_min - margin, y_max + margin)


# ---------------------------------------------------------------------------
# Weight, gradient, and activation distribution visualization
# ---------------------------------------------------------------------------

def visualize_weight_distribution(params: dict, color: str = "C0") -> None:
    """Plot histograms of weight values per layer.

    Parameters
    ----------
    params : dict
        Model parameters (e.g. from ``model.init(...)``).
    color : str
        Histogram color.

    Example
    -------
    >>> model = nn.Dense(features=10)
    >>> params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
    >>> visualize_weight_distribution(params)
    """
    leaves = jax.tree_util.tree_leaves(params)
    weights = [jax.device_get(p).reshape(-1) for p in leaves if p.ndim > 1]
    weight_dict = {f"Layer {i}": w for i, w in enumerate(weights)}
    fig = _plot_dists(weight_dict, color=color, xlabel="Weight vals")
    fig.suptitle("Weight distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()


def visualize_gradients(params: dict, loss_fn: Callable[[dict], float],
                        color: str = "C0", print_variance: bool = False) -> None:
    """Plot histograms of per-layer gradient magnitudes.

    Parameters
    ----------
    params : dict
        Model parameters.
    loss_fn : callable
        Function of signature ``(params) -> scalar loss``. The model
        and batch should be captured in the closure.
    color : str
        Histogram color.
    print_variance : bool
        If True, print the variance of each layer's gradients.

    Example
    -------
    >>> def loss_fn(p):
    ...     logits = model.apply(p, batch)
    ...     return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    >>> visualize_gradients(params, loss_fn)
    """
    grads = jax.grad(loss_fn)(params)
    grads = jax.device_get(grads)
    leaves = jax.tree_util.tree_leaves(grads)
    grad_arrays = [g.reshape(-1) for g in leaves if g.ndim > 1]
    grad_dict = {f"Layer {i}": g for i, g in enumerate(grad_arrays)}

    fig = _plot_dists(grad_dict, color=color, xlabel="Grad magnitude")
    fig.suptitle("Gradient distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()

    if print_variance:
        for key in sorted(grad_dict.keys()):
            print(f"{key} - Variance: {grad_dict[key].var():.6f}")


def visualize_activations(net: "nn.Module", params: dict, batch: jax.Array,
                          color: str = "C0", print_variance: bool = False) -> None:
    """Plot histograms of per-layer activation distributions.

    Uses Flax's ``capture_intermediates`` to collect activations without
    modifying the model.

    Parameters
    ----------
    net : flax.linen.Module
        Flax module.
    params : dict
        Model parameters.
    batch : jax.Array
        Input batch to pass through the network.
    color : str
        Histogram color.
    print_variance : bool
        If True, print the variance of each layer's activations.

    Example
    -------
    >>> visualize_activations(model, params, batch, print_variance=True)
    Layer 0 - Variance: 0.0822
    Layer 1 - Variance: 0.0042
    """
    activations = get_layer_activations(net, params, batch)
    leaves = jax.tree_util.tree_leaves(activations)
    act_arrays = [jax.device_get(a).reshape(-1) for a in leaves if a.ndim >= 2]
    act_dict = {f"Layer {i}": a for i, a in enumerate(act_arrays)}

    fig = _plot_dists(act_dict, color=color, stat="density", xlabel="Activation vals")
    fig.suptitle("Activation distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()

    if print_variance:
        for key in sorted(act_dict.keys()):
            print(f"{key} - Variance: {act_dict[key].var():.6f}")


# ---------------------------------------------------------------------------
# Layer-level data extraction
# ---------------------------------------------------------------------------

def get_layer_gradients(
    params: dict, loss_fn: Callable[[dict], float], include_bias: bool = False
) -> list[jnp.ndarray]:
    """Compute per-layer weight gradients for diagnostic visualization.

    Parameters
    ----------
    params : dict
        Model parameters.
    loss_fn : callable
        Function of signature (params) -> scalar loss.
    include_bias : bool, optional
        If True, include bias gradients (1-D arrays). Default False, which
        only includes weight matrices (ndim > 1).

    Returns
    -------
    list[jnp.ndarray]
        Flattened gradient arrays for each parameter (weights and optionally biases).
    """
    grads = jax.grad(loss_fn)(params)
    grads = jax.device_get(grads)
    leaves = jax.tree_util.tree_leaves(grads)
    if include_bias:
        return [g.reshape(-1) for g in leaves]
    else:
        return [g.reshape(-1) for g in leaves if g.ndim > 1]


def get_layer_activations(net: "nn.Module", params: dict, batch: jax.Array) -> dict:
    """Capture per-layer activations via Flax's capture_intermediates.

    Parameters
    ----------
    net : flax.linen.Module
        Flax module.
    params : dict
        Model parameters.
    batch : jax.Array
        Input batch.

    Returns
    -------
    dict
        Intermediate activations keyed by layer path.
    """
    _, state = net.apply(params, batch, capture_intermediates=True)
    return state["intermediates"]


# ---------------------------------------------------------------------------
# Dead / inactive neuron analysis
# ---------------------------------------------------------------------------

def count_inactive_neurons(
    activations: dict,
    threshold: float = 1e-6,
    mode: str = "magnitude",
    aggregate_spatial: bool = False,
    spatial_reduction: str = "mean",
    channel_axis: int = -1,
) -> dict[str, dict[str, int | float]]:
    """Count neurons that show minimal variation or activation across a batch.

    Works with any activation function. The ``mode`` parameter controls
    what "inactive" means:

    - "magnitude": neuron's max absolute activation never exceeds threshold.
      Catches dead ReLU and near-zero outputs.
    - "variance": neuron's variance across the batch is below threshold.
      Catches saturated tanh/sigmoid stuck at one value.
    - "both": neuron must fail both checks (strictest).

    Parameters
    ----------
    activations : dict
        Output of get_layer_activations (nested dict of (value,) tuples).
    threshold : float
        Cutoff for considering a neuron inactive.
    mode : str
        'magnitude', 'variance', or 'both'.
    aggregate_spatial : bool
        If True and the activation has spatial dimensions (e.g., conv layers
        with shape ``(batch, h, w, channels)``), each channel is treated as
        one neuron by reducing over spatial dimensions. Also works for
        transformer activations with shape ``(batch, seq_len, features)``
        where each feature is treated as one neuron averaged over sequence
        positions. If False, every spatial position is counted independently.
    spatial_reduction : str, optional
        How to reduce spatial dimensions when aggregate_spatial is True.
        One of 'mean', 'max', or 'sum'. Ignored if aggregate_spatial is False.
    channel_axis : int, optional
        Axis that holds channels/features. Default -1 (last axis). Batch axis
        is assumed to be 0. All other axes are treated as spatial.

    Returns
    -------
    dict[str, dict]
        Per-layer results with keys: "inactive", "total", "percent".
    """
    results = {}

    def traverse(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                new_path = f"{path}/{k}" if path else k
                traverse(v, new_path)
        elif isinstance(node, (list, tuple)):
            for item in node:
                traverse(item, path)
        else:
            process_activation(node, path)

    def process_activation(arr: jax.Array, name: str):
        if arr.ndim < 2:
            return
        batch_size = arr.shape[0]

        if aggregate_spatial:
            spatial_axes = [i for i in range(1, arr.ndim) if i != channel_axis % arr.ndim]
            if spatial_axes:
                if spatial_reduction == "mean":
                    arr = arr.mean(axis=tuple(spatial_axes))
                elif spatial_reduction == "max":
                    arr = arr.max(axis=tuple(spatial_axes))
                elif spatial_reduction == "sum":
                    arr = arr.sum(axis=tuple(spatial_axes))
                else:
                    raise ValueError(
                        f"Unknown spatial_reduction: {spatial_reduction}. "
                        "Use 'mean', 'max', or 'sum'."
                    )

        flat_arr = arr.reshape(batch_size, -1)

        mag_dead = jnp.abs(flat_arr).max(axis=0) <= threshold
        var_dead = flat_arr.var(axis=0) <= threshold

        if mode == "magnitude":
            inactive = mag_dead
        elif mode == "variance":
            inactive = var_dead
        elif mode == "both":
            inactive = mag_dead & var_dead
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'magnitude', 'variance', or 'both'.")

        total = flat_arr.shape[1]
        dead_count = int(inactive.sum())
        results[name] = {
            "inactive": dead_count,
            "total": total,
            "percent": round(100 * dead_count / total, 2) if total > 0 else 0.0,
        }

    traverse(activations)
    return results


def plot_loss_landscape(
    params: dict,
    loss_fn: Callable[[dict], float],
    grid_size: int = 50,
    range_scale: float = 1.0,
    seed: int = 0,
    plot_3d: bool = False,
    cmap: str = "viridis",
) -> None:
    """Plot a 2D slice of the loss landscape around a set of parameters.

    Projects the high-dimensional loss surface onto two random directions
    in parameter space, following the approach from Li et al. (2018),
    "Visualizing the Loss Landscape of Neural Nets."

    This is qualitative, not exhaustive. The slice is random so different
    seeds will show different cross-sections. Useful for comparing sharp
    vs flat minima across training runs or architectures.

    Parameters
    ----------
    params : dict
        Model parameters (the center point of the plot).
    loss_fn : callable
        Function of signature ``(params) -> scalar loss``. The model
        and batch should be captured in the closure.
    grid_size : int
        Number of points per axis. Total evaluations = grid_size^2.
        50 is a reasonable default; reduce for large models.
    range_scale : float
        How far to sweep in each direction. Larger values show more
        of the landscape but may miss local structure.
    seed : int
        Random seed for generating the two direction vectors.
    plot_3d : bool
        If True, render a 3D surface plot. Otherwise a 2D heatmap.
    cmap : str
        Matplotlib colormap name.

    Example
    -------
    >>> def loss_fn(p):
    ...     logits = model.apply(p, batch)
    ...     return optax.softmax_cross_entropy_with_integer_labels(
    ...         logits, labels
    ...     ).mean()
    >>> plot_loss_landscape(params, loss_fn, grid_size=30)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    leaves, treedef = jax.tree_util.tree_flatten(params)

    rng = jax.random.PRNGKey(seed)
    rng, key1, key2 = jax.random.split(rng, 3)
    keys1 = jax.random.split(key1, len(leaves))
    keys2 = jax.random.split(key2, len(leaves))

    dir1 = [jax.random.normal(k, shape=p.shape) for k, p in zip(keys1, leaves)]
    dir2 = [jax.random.normal(k, shape=p.shape) for k, p in zip(keys2, leaves)]

    def _normalize(direction, reference):
        normalized = []
        for d, r in zip(direction, reference):
            r_norm = jnp.linalg.norm(r)
            d_norm = jnp.linalg.norm(d)
            if r_norm > 0 and d_norm > 0:
                normalized.append(d * (r_norm / d_norm))
            else:
                normalized.append(jnp.zeros_like(d))
        return normalized

    dir1 = _normalize(dir1, leaves)
    dir2 = _normalize(dir2, leaves)

    alphas = np.linspace(-range_scale, range_scale, grid_size)
    betas = np.linspace(-range_scale, range_scale, grid_size)

    # NOTE: a vmap approach would require stacking leaves/dir1/dir2 into
    # single arrays, which only works if all parameter tensors share the
    # same shape. Not practical for most networks, so we loop instead.
    losses = np.zeros((grid_size, grid_size))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            perturbed = [p + alpha * d1 + beta * d2
                         for p, d1, d2 in zip(leaves, dir1, dir2)]
            perturbed_params = jax.tree_util.tree_unflatten(treedef, perturbed)
            losses[j, i] = float(loss_fn(perturbed_params))

    fig = plt.figure(figsize=(8, 6))
    if plot_3d:
        ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(alphas, betas)
        ax.plot_surface(X, Y, losses, cmap=cmap, linewidth=0, antialiased=True)
        ax.set_zlabel("Loss")
    else:
        ax = fig.add_subplot(111)
        extent = [-range_scale, range_scale, -range_scale, range_scale]
        ax.imshow(losses[::-1], cmap=cmap, extent=extent, aspect="auto")

    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_title(f"Loss Landscape (seed={seed}, range={range_scale})")
    plt.tight_layout()
    plt.show()
    plt.close()



def model_tabulate(
    model: nn.Module,
    *init_inputs,
    mutable: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> None:
    """Print a Flax tabulate summary including frozen constants.

    A thin wrapper around ``nn.tabulate`` that includes the ``"constants"``
    collection by default, which ``nn.tabulate`` omits unless explicitly
    requested. Useful for modules with frozen buffers such as positional
    encodings or fixed projection matrices.

    Parameters
    ----------
    model : nn.Module
        Any Flax module.
    *init_inputs
        All positional arguments required by ``model.init`` for shape tracing.
        Must include any PRNGKeys the model needs (e.g. for dropout).
    mutable : sequence of str, optional
        Variable collections to include in the table.
        Defaults to ``["params", "constants"]``.
    seed : int
        Seed for the PRNGKey passed to ``nn.tabulate``.

    Example
    -------
    >>> model_tabulate(model, dummy_input)
    >>> model_tabulate(model, dummy_input, mutable=["params"])
    """
    if mutable is None:
        mutable = ["params", "constants"]
    tabulate_fn = nn.tabulate(
        model, jax.random.PRNGKey(seed), mutable=mutable
    )
    print(tabulate_fn(*init_inputs))


def plot_output_at_init(
    model: nn.Module,
    init_inputs: tuple,
    grid_inputs: tuple,
    shape: tuple[int, int],
    seed: int = 0,
    cmap: str = "RdBu_r",
    title: str = "Model output at init",
    extent: Optional[list] = None,
    view: str = "cartesian",
    lon_grid: Optional[np.ndarray] = None,
    lat_grid: Optional[np.ndarray] = None,
) -> None:
    """Forward pass at initialisation over a pre-built coordinate grid.

    The caller builds ``grid_inputs`` in whatever coordinate system the model
    expects and passes the flat arrays directly. This function handles init,
    apply, reshape, and plot only.

    For ``view="mollweide"`` the caller must also supply ``lon_grid`` and
    ``lat_grid`` as 2D arrays of shape ``shape`` (in radians), since
    pcolormesh needs explicit coordinates for the Mollweide projection.
    For ``view="cartesian"`` those are not needed -- ``extent`` is sufficient.

    Parameters
    ----------
    model : nn.Module
        Any Flax module.
    init_inputs : tuple
        Small subset of inputs used for ``model.init`` shape tracing.
    grid_inputs : tuple
        Full grid inputs passed to ``model.apply`` for the plot.
        The caller is responsible for coordinate system and ordering.
    shape : tuple[int, int]
        (rows, cols) to reshape the flat model output into.
    seed : int
        PRNGKey seed for init.
    cmap : str
        Matplotlib colormap.
    title : str
        Plot title.
    extent : list, optional
        [xmin, xmax, ymin, ymax] passed to imshow. Only used when
        ``view="cartesian"``.
    view : str
        ``"cartesian"`` uses imshow on a flat grid.
        ``"mollweide"`` uses pcolormesh on a Mollweide projection.
    lon_grid : np.ndarray, optional
        2D array of longitudes in radians, shape ``shape``.
        Required when ``view="mollweide"``.
    lat_grid : np.ndarray, optional
        2D array of latitudes in radians, shape ``shape``.
        Required when ``view="mollweide"``.

    Example
    -------
    >>> # Cartesian -- regional or volume slice, caller builds the grid
    >>> plot_output_at_init(
    ...     model,
    ...     init_inputs=(dummy[:5],),
    ...     grid_inputs=(grid_coords,),
    ...     shape=(200, 200),
    ...     extent=[-100, -40, 0, 30],
    ... )

    >>> # Mollweide -- caller builds lon/lat meshgrid in radians
    >>> plot_output_at_init(
    ...     sphere_model,
    ...     init_inputs=(dummy[:5],),
    ...     grid_inputs=(grid_coords,),
    ...     shape=(200, 200),
    ...     view="mollweide",
    ...     lon_grid=LON,
    ...     lat_grid=LAT,
    ... )
    """
    if view == "mollweide" and (lon_grid is None or lat_grid is None):
        raise ValueError(
            "view='mollweide' requires lon_grid and lat_grid as 2D arrays "
            "in radians of shape `shape`."
        )

    key = jax.random.PRNGKey(seed)
    variables = model.init(key, *init_inputs)
    pred = np.array(model.apply(variables, *grid_inputs)).squeeze()

    assert pred.size == np.prod(shape), (
        f"Model output size {pred.size} does not match shape {shape} "
        f"({np.prod(shape)} elements). Check grid_inputs and shape."
    )
    pred = pred.reshape(shape)
    vmax = abs(pred).max()

    fig = plt.figure(figsize=(10, 5))

    if view == "mollweide":
        ax = fig.add_subplot(111, projection="mollweide")
        im = ax.pcolormesh(lon_grid, lat_grid, pred, cmap=cmap,
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, orientation="horizontal",
                     pad=0.05, shrink=0.7)
    else:
        ax = fig.add_subplot(111)
        im = ax.imshow(pred, origin="lower", cmap=cmap,
                       vmin=-vmax, vmax=vmax, aspect="auto",
                       extent=extent)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()