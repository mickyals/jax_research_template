import jax
import jax.numpy as jnp
from typing import Callable, Literal


def check_environment() -> None:
    """Check JAX environment and print details for GPUs, backend, JAX version, and available memory.

    Example
    -------
    >>> check_environment()
    JAX version: 0.4.35
    Backend: gpu
    GPUs: 1 available
      [0] NVIDIA A100-SXM4-40GB - 40.0 GB
    """
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    try:
        gpus = jax.devices("gpu")
        print(f"GPUs: {len(gpus)} available")
        for i, gpu in enumerate(gpus):
            try:
                mem_stats = gpu.memory_stats()
                mem_gb = mem_stats.get("bytes_limit", 0) / 1e9
            except (AttributeError, KeyError):
                mem_gb = 0.0
            print(f"  [{i}] {gpu.device_kind} - {mem_gb:.1f} GB")
    except RuntimeError:
        print("GPUs: none, using CPU")


def create_rng(seed: int = 42) -> jax.Array:
    """Create a JAX PRNGKey from a seed for reproducible randomness.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    jax.Array
        A JAX PRNGKey array.

    Example
    -------
    >>> create_rng(0)
    Array([0, 0], dtype=uint32)
    """
    return jax.random.PRNGKey(seed)


def create_rng_dict(seed: int = 42, keys: list[str] | None = None) -> dict[str, jax.Array]:
    """Create a dictionary of PRNGKeys from a single seed.

    Splits a root key into named subkeys for use with Flax model
    init and apply calls.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    keys : list[str], optional
        Key names for the dictionary. Defaults to ["params", "dropout"].

    Returns
    -------
    dict[str, jax.Array]
        Dictionary mapping key names to their respective PRNGKeys.

    Example
    -------
    >>> rngs = create_rng_dict(0, keys=["params", "dropout"])
    >>> list(rngs.keys())
    ['params', 'dropout']
    >>> rngs["params"].shape
    (2,)
    """
    if keys is None:
        keys = ["params", "dropout"]
    root = jax.random.PRNGKey(seed)
    splits = jax.random.split(root, len(keys))
    return {k: v for k, v in zip(keys, splits)}


def split_rng(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split a PRNGKey into a new root key and a subkey.

    This is the standard JAX pattern to avoid accidental key reuse.

    Parameters
    ----------
    rng : jax.Array
        Current PRNGKey to split.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        A pair of (new_rng, subkey).

    Example
    -------
    >>> rng = create_rng(0)
    >>> rng, key = split_rng(rng)
    >>> jax.random.normal(key)
    Array(-1.2515389, dtype=float32)
    """
    return jax.random.split(rng)


def show_jaxpr(fn: Callable, *sample_inputs, static_argnums: int | tuple[int, ...] | None = None) -> None:
    """Print the jaxpr (JAX expression) representation of a function.

    Parameters
    ----------
    fn : callable
        Function to trace.
    *sample_inputs
        Example inputs that define the shapes and dtypes for tracing.
    static_argnums : int or tuple[int, ...], optional
        Positional arguments to treat as static (not traced).

    Example
    -------
    >>> show_jaxpr(lambda x: x ** 2 + 1, jnp.array(3.0))
    { lambda ; a:f32[]. let
        b:f32[] = integer_pow[y=2] a
        c:f32[] = add b 1.0
      in (c,) }
    """
    kwargs = {}
    if static_argnums is not None:
        kwargs["static_argnums"] = static_argnums
    print(jax.make_jaxpr(fn, **kwargs)(*sample_inputs))


def grad_fn(fn: Callable, argnums: int | tuple[int, ...] = 0, has_aux: bool = False) -> Callable:
    """Return a function that computes both value and gradients.

    Parameters
    ----------
    fn : callable
        Function to differentiate.
    argnums : int or tuple[int]
        Which positional argument(s) to differentiate with respect to.
    has_aux : bool
        If True, ``fn`` returns a pair ``(value, aux)`` and the gradient
        is computed with respect to ``value`` only.

    Returns
    -------
    callable
        A function that returns ``(value, grads)`` or ``((value, aux), grads)``.

    Example
    -------
    >>> f = lambda x: x ** 3
    >>> val_and_grad = grad_fn(f)
    >>> val_and_grad(2.0)
    (Array(8., dtype=float32), Array(12., dtype=float32))
    """
    return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)



# ---------------------------------------------------------------------------
# Generic angle conversion
# ---------------------------------------------------------------------------

def degrees_to_radians(x: jax.Array) -> jax.Array:
    """Convert degrees to radians element-wise.

    Parameters
    ----------
    x : jax.Array
        Array of values in degrees.

    Returns
    -------
    jax.Array
        Array of values in radians.

    Example
    -------
    >>> degrees_to_radians(jnp.array([0., 90., 180., 360.]))
    Array([0.       , 1.5707964, 3.1415927, 6.2831855], dtype=float32)
    """
    return jnp.radians(x)


def radians_to_degrees(x: jax.Array) -> jax.Array:
    """Convert radians to degrees element-wise.

    Parameters
    ----------
    x : jax.Array
        Array of values in radians.

    Returns
    -------
    jax.Array
        Array of values in degrees.

    Example
    -------
    >>> radians_to_degrees(jnp.array([0., jnp.pi / 2, jnp.pi]))
    Array([  0.,  90., 180.], dtype=float32)
    """
    return jnp.degrees(x)


# ---------------------------------------------------------------------------
# Lat/lon conversions
# ---------------------------------------------------------------------------

def latlon_deg_to_rad(
    lat_deg: jax.Array, lon_deg: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Convert lat/lon from degrees to radians.

    Parameters
    ----------
    lat_deg : jax.Array
        Latitudes in degrees. Shape (N,) or broadcastable.
    lon_deg : jax.Array
        Longitudes in degrees. Shape (N,) or broadcastable.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (lat_rad, lon_rad) in radians, same shapes as inputs.

    Example
    -------
    >>> latlon_deg_to_rad(jnp.array([0., 45.]), jnp.array([90., 180.]))
    (Array([0.       , 0.7853982], dtype=float32),
     Array([1.5707964, 3.1415927], dtype=float32))
    """
    return jnp.radians(lat_deg), jnp.radians(lon_deg)


def latlon_rad_to_deg(
    lat_rad: jax.Array, lon_rad: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Convert lat/lon from radians to degrees.

    Parameters
    ----------
    lat_rad : jax.Array
        Latitudes in radians.
    lon_rad : jax.Array
        Longitudes in radians.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (lat_deg, lon_deg) in degrees, same shapes as inputs.

    Example
    -------
    >>> latlon_rad_to_deg(jnp.array([0., jnp.pi / 4]), jnp.array([jnp.pi / 2, jnp.pi]))
    (Array([ 0., 45.], dtype=float32), Array([ 90., 180.], dtype=float32))
    """
    return jnp.degrees(lat_rad), jnp.degrees(lon_rad)


# ---------------------------------------------------------------------------
# Spherical <-> Cartesian
# ---------------------------------------------------------------------------

def spherical_to_cartesian(
    lat_rad: jax.Array, lon_rad: jax.Array
) -> jax.Array:
    """Convert spherical lat/lon (radians) to unit Cartesian (x, y, z).

    Uses the geographic convention:
        x = cos(lat) * cos(lon)
        y = cos(lat) * sin(lon)
        z = sin(lat)

    Parameters
    ----------
    lat_rad : jax.Array
        Latitudes in radians. Any shape broadcastable with lon_rad.
    lon_rad : jax.Array
        Longitudes in radians. Any shape broadcastable with lat_rad.

    Returns
    -------
    jax.Array
        Unit Cartesian coordinates. Shape (*broadcast_shape, 3).

    Example
    -------
    >>> spherical_to_cartesian(jnp.array([0.]), jnp.array([0.]))
    Array([[1., 0., 0.]], dtype=float32)
    """
    x = jnp.cos(lat_rad) * jnp.cos(lon_rad)
    y = jnp.cos(lat_rad) * jnp.sin(lon_rad)
    z = jnp.sin(lat_rad)
    return jnp.stack([x, y, z], axis=-1)


def cartesian_to_spherical(xyz: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Convert Cartesian (x, y, z) to spherical lat/lon in radians.

    Inverse of ``spherical_to_cartesian``. Supports arbitrary leading
    dimensions via ellipsis indexing.

    Parameters
    ----------
    xyz : jax.Array
        Cartesian coordinates with last dimension 3. Shape (..., 3).
        Need not be unit vectors -- only direction matters.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (lat_rad, lon_rad), each shape (...,).
        lat in [-pi/2, pi/2], lon in [-pi, pi].

    Example
    -------
    >>> cartesian_to_spherical(jnp.array([[1., 0., 0.]]))
    (Array([0.], dtype=float32), Array([0.], dtype=float32))
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lat = jnp.arcsin(jnp.clip(z, -1.0, 1.0))
    lon = jnp.arctan2(y, x)
    return lat, lon


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def minmax_norm(
    x: jax.Array,
    x_min: float | jax.Array,
    x_max: float | jax.Array,
    mode: Literal["01", "-11"] = "01",
    eps: float = 1e-12,
) -> jax.Array:
    """Min-max normalise an array to [0, 1] or [-1, 1].

    Parameters
    ----------
    x : jax.Array
        Input array, any shape.
    x_min : float or jax.Array
        Minimum value of the input range. Supports broadcasting
        (e.g. per-column bounds).
    x_max : float or jax.Array
        Maximum value of the input range.
    mode : {"01", "-11"}
        Output range. ``"01"`` maps to [0, 1]. ``"-11"`` maps to [-1, 1].
    eps : float
        Small constant added to the denominator to avoid division by zero
        when ``x_min == x_max``.

    Returns
    -------
    jax.Array
        Normalised array, same shape as ``x``.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"01"`` or ``"-11"``.

    Example
    -------
    >>> minmax_norm(jnp.array([0., 5., 10.]), 0., 10., mode="01")
    Array([0. , 0.5, 1. ], dtype=float32)
    >>> minmax_norm(jnp.array([0., 5., 10.]), 0., 10., mode="-11")
    Array([-1.,  0.,  1.], dtype=float32)
    """
    if mode not in ("01", "-11"):
        raise ValueError(f"mode must be '01' or '-11', got '{mode}'")
    x_norm = (x - x_min) / (x_max - x_min + eps)
    if mode == "-11":
        x_norm = x_norm * 2.0 - 1.0
    return x_norm


def standardise(
    x: jax.Array,
    mean: float | jax.Array | None = None,
    std: float | jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    eps: float = 1e-8,
) -> jax.Array:
    """Standardise an array to zero mean and unit variance.

    If ``mean`` and ``std`` are not provided they are computed from ``x``.
    Pass pre-computed statistics when standardising a test set using
    training set statistics.

    Parameters
    ----------
    x : jax.Array
        Input array, any shape.
    mean : float or jax.Array, optional
        Mean to subtract. Computed from ``x`` over ``axis`` if not provided.
    std : float or jax.Array, optional
        Standard deviation to divide by. Computed from ``x`` over ``axis``
        if not provided.
    axis : int or tuple of int, optional
        Axis or axes over which to compute statistics when ``mean`` or
        ``std`` are not provided. ``None`` computes over all elements.
        ``keepdims=True`` is used internally so broadcasting works for
        any axis choice.
    eps : float
        Small constant added to ``std`` to avoid division by zero.

    Returns
    -------
    jax.Array
        Standardised array, same shape as ``x``.

    Example
    -------
    >>> x = jnp.array([1., 2., 3., 4., 5.])
    >>> standardise(x)
    Array([-1.4142135, -0.7071068,  0.       ,  0.7071068,  1.4142135],
          dtype=float32)

    >>> # Per-feature standardisation on a (N, D) array
    >>> x = jnp.ones((10, 4))
    >>> standardise(x, axis=0).shape
    (10, 4)
    """
    if mean is None:
        mean = x.mean(axis=axis, keepdims=True)
    if std is None:
        std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + eps)