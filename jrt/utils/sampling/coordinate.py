import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats.qmc import LatinHypercube, scale
from typing import Optional, Literal

def _key_to_seed(key: jax.Array) -> int:
    """Derive a reproducible integer seed from a JAX PRNGKey.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey.

    Returns
    -------
    int
        Integer seed in [0, 2^31 - 1] suitable for scipy/numpy RNGs.
    """
    return int(jax.random.randint(key, (), minval=0, maxval=2**31 - 1))


# ---------------------------------------------------------------------------
# Uniform samplers
# ---------------------------------------------------------------------------

def sample_regional(
    key: jax.Array,
    n: int,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
) -> tuple[jax.Array, jax.Array]:
    """Uniform random sampling in a lon/lat box.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey.
    n : int
        Number of samples.
    lon_bounds : tuple[float, float]
        (lon_min, lon_max) in degrees.
    lat_bounds : tuple[float, float]
        (lat_min, lat_max) in degrees.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (lons, lats), each shape (n,).

    Example
    -------
    >>> key = jax.random.PRNGKey(0)
    >>> lons, lats = sample_regional(key, 100, (-100., -40.), (0., 30.))
    >>> lons.shape
    (100,)
    """
    k1, k2 = jax.random.split(key)
    lons = jax.random.uniform(k1, (n,), minval=lon_bounds[0], maxval=lon_bounds[1])
    lats = jax.random.uniform(k2, (n,), minval=lat_bounds[0], maxval=lat_bounds[1])
    return lons, lats


def sample_sphere_uniform_area(
    key: jax.Array,
    n: int,
) -> tuple[jax.Array, jax.Array]:
    """Uniform sampling on the sphere with respect to surface area.

    Uses the inverse CDF method: lat = arcsin(2u - 1), lon = 2*pi*v - pi.
    Avoids pole-clustering that occurs with uniform-in-angle sampling.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey.
    n : int
        Number of samples.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (lat, lon) in radians, each shape (n,).
        lat in [-pi/2, pi/2], lon in [-pi, pi].

    Example
    -------
    >>> key = jax.random.PRNGKey(0)
    >>> lat, lon = sample_sphere_uniform_area(key, 500)
    >>> lat.shape
    (500,)
    """
    k1, k2 = jax.random.split(key)
    lat = jnp.arcsin(2 * jax.random.uniform(k1, (n,)) - 1)
    lon = 2 * jnp.pi * jax.random.uniform(k2, (n,)) - jnp.pi
    return lat, lon


def sample_sphere_uniform_angle(
    key: jax.Array,
    n: int,
) -> tuple[jax.Array, jax.Array]:
    """Uniform-in-angle sampling on the sphere.

    Samples lat uniformly in [-pi/2, pi/2] and lon uniformly in [-pi, pi].
    This is NOT area-uniform -- it oversamples the poles relative to surface
    area. Useful when you want denser training coverage at high latitudes.
    Use ``sample_sphere_uniform_area`` for unbiased global coverage.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey.
    n : int
        Number of samples.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (lat, lon) in radians, each shape (n,).

    Example
    -------
    >>> key = jax.random.PRNGKey(0)
    >>> lat, lon = sample_sphere_uniform_angle(key, 500)
    >>> lat.shape
    (500,)
    """
    k1, k2 = jax.random.split(key)
    lat = (jax.random.uniform(k1, (n,)) - 0.5) * jnp.pi
    lon = 2 * jnp.pi * jax.random.uniform(k2, (n,)) - jnp.pi
    return lat, lon


def sample_volume(
    key: jax.Array,
    n: int,
    bounds: jax.Array,
) -> jax.Array:
    """Uniform random sampling in a 3D box.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey.
    n : int
        Number of samples.
    bounds : jax.Array
        Shape (3, 2). Each row is [min, max] for one dimension,
        ordered as [x, y, z] or [lon, lat, alt].

    Returns
    -------
    jax.Array
        Sampled coordinates of shape (n, 3).

    Example
    -------
    >>> key = jax.random.PRNGKey(0)
    >>> bounds = jnp.array([[-1., 1.], [-1., 1.], [0., 1.]])
    >>> coords = sample_volume(key, 100, bounds)
    >>> coords.shape
    (100, 3)
    """
    mins = bounds[:, 0]
    maxs = bounds[:, 1]
    return jax.random.uniform(key, (n, 3), minval=mins, maxval=maxs)


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling (via scipy.stats.qmc)
# ---------------------------------------------------------------------------

def lhs_sample(
    key: jax.Array,
    n: int,
    d: int,
    bounds: Optional[np.ndarray] = None,
    scramble: bool = True,
    optimization: Optional[Literal["random-cd", "lloyd"]] = None,
) -> jax.Array:
    """Latin Hypercube Sampling in d dimensions via scipy.stats.qmc.

    Generates n points with guaranteed one-sample-per-stratum coverage
    in each dimension. Substantially better space-filling than uniform
    random at the same n, especially for sparse observation experiments
    where uniform random can leave large empty regions.

    The JAX key is bridged to scipy's stateful RNG via ``_key_to_seed``
    for reproducibility. This function is not purely functional -- it
    calls scipy's stateful RNG internally -- but sequential calls with
    different keys produce independent, reproducible results.

    Requires scipy >= 1.8 for scipy.stats.qmc.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey used to seed scipy's RNG.
    n : int
        Number of samples.
    d : int
        Number of dimensions.
    bounds : np.ndarray, optional
        Shape (d, 2). Each row is [min, max] for one dimension.
        If None, points are returned in the unit hypercube [0, 1]^d
        with no scaling applied.
    scramble : bool
        If True (default), randomly place samples within strata.
        If False, center samples within strata (deterministic layout).
    optimization : {"random-cd", "lloyd"}, optional
        Post-processing optimisation to improve space-filling quality.
        "random-cd": minimise centered discrepancy via random permutations.
        "lloyd": move points toward a more uniform distribution.
        None: no optimisation (default, fastest).

    Returns
    -------
    jax.Array
        Sampled points of shape (n, d).

    Example
    -------
    >>> key = jax.random.PRNGKey(0)
    >>> pts = lhs_sample(key, 50, 2)
    >>> pts.shape
    (50, 2)
    >>> bounds = np.array([[-1., 1.], [-1., 1.], [0., 1.]])
    >>> pts = lhs_sample(key, 50, 3, bounds=bounds, optimization="random-cd")
    >>> pts.shape
    (50, 3)
    """
    sampler = LatinHypercube(d=d, scramble=scramble, seed=_key_to_seed(key),
                              optimization=optimization)
    pts = sampler.random(n)

    if bounds is not None:
        bounds = np.asarray(bounds)
        pts = scale(pts, l_bounds=bounds[:, 0], u_bounds=bounds[:, 1])

    return jnp.array(pts)


def lhs_sample_regional(
    key: jax.Array,
    n: int,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    scramble: bool = True,
    optimization: Optional[Literal["random-cd", "lloyd"]] = None,
) -> tuple[jax.Array, jax.Array]:
    """Latin Hypercube Sampling in a lon/lat box.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey.
    n : int
        Number of samples.
    lon_bounds : tuple[float, float]
        (lon_min, lon_max) in degrees.
    lat_bounds : tuple[float, float]
        (lat_min, lat_max) in degrees.
    scramble : bool
        Randomly place samples within strata. Default True.
    optimization : {"random-cd", "lloyd"}, optional
        Post-processing optimisation. See ``lhs_sample``.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (lons, lats), each shape (n,).

    Example
    -------
    >>> key = jax.random.PRNGKey(0)
    >>> lons, lats = lhs_sample_regional(key, 50, (-100., -40.), (0., 30.))
    >>> lons.shape
    (50,)
    """
    bounds = np.array([lon_bounds, lat_bounds])
    pts = lhs_sample(key, n, d=2, bounds=bounds, scramble=scramble,
                     optimization=optimization)
    return pts[:, 0], pts[:, 1]


def lhs_sample_volume(
    key: jax.Array,
    n: int,
    bounds: np.ndarray,
    scramble: bool = True,
    optimization: Optional[Literal["random-cd", "lloyd"]] = None,
) -> jax.Array:
    """Latin Hypercube Sampling in a 3D volume.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey.
    n : int
        Number of samples.
    bounds : np.ndarray
        Shape (3, 2). Each row is [min, max] for one dimension.
    scramble : bool
        Randomly place samples within strata. Default True.
    optimization : {"random-cd", "lloyd"}, optional
        Post-processing optimisation. See ``lhs_sample``.

    Returns
    -------
    jax.Array
        Sampled coordinates of shape (n, 3).

    Example
    -------
    >>> key = jax.random.PRNGKey(0)
    >>> bounds = np.array([[-1., 1.], [-1., 1.], [0., 1.]])
    >>> coords = lhs_sample_volume(key, 50, bounds)
    >>> coords.shape
    (50, 3)
    """
    return lhs_sample(key, n, d=3, bounds=bounds, scramble=scramble,
                      optimization=optimization)