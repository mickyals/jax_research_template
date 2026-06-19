"""
utils/geoscience/coordinates.py

Lat/lon coordinate encoders and angle/spherical conversions.

The NumPy encoders map geographic (lat, lon) in degrees to model-friendly
coordinate representations. They live in the geoscience utils — not in the
data-loading framework — so the generic DataModule stays domain-agnostic; an
experiment that wants to encode lat/lon feature columns calls these directly.
(Moved here from datasets/datamodule._apply_position_encoding, 2026-06-15,
plan r12.)

The JAX angle/spherical helpers at the bottom (degrees<->radians, lat/lon
deg<->rad, spherical<->Cartesian) were relocated from utils/jax_core/helpers.py
(2026-06-17, plan r16) — they are geographic, not JAX-core, utilities. JAX and
NumPy backends coexist here as they do in geodesic.py.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def lat_lon_to_unit_sphere(lat_deg, lon_deg) -> np.ndarray:
    """Map (lat, lon) degrees to 3-D unit-sphere Cartesian coordinates.

    Embeds the angles on the sphere so there are no lat/lon discontinuities
    (poles, the ±180° meridian seam).

    Parameters
    ----------
    lat_deg, lon_deg : array-like
        Latitude / longitude in decimal degrees.

    Returns
    -------
    np.ndarray
        Shape ``(..., 3)`` — ``(cos(lat)cos(lon), cos(lat)sin(lon), sin(lat))``.
    """
    lat_r = np.radians(np.asarray(lat_deg, dtype=np.float32))
    lon_r = np.radians(np.asarray(lon_deg, dtype=np.float32))
    return np.stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ], axis=-1)


def lat_lon_to_domain_normalised(
    lat_deg,
    lon_deg,
    fov_lat: tuple[float, float],
    fov_lon: tuple[float, float],
) -> np.ndarray:
    """Map (lat, lon) degrees to ``[-1, 1]^2`` over a field-of-view box.

    Parameters
    ----------
    lat_deg, lon_deg : array-like
        Latitude / longitude in decimal degrees.
    fov_lat, fov_lon : (min, max)
        Field-of-view bounds in degrees.

    Returns
    -------
    np.ndarray
        Shape ``(..., 2)`` — ``(norm_lat, norm_lon)``, each in [-1, 1] for
        in-FOV positions.
    """
    lat = np.asarray(lat_deg, dtype=np.float32)
    lon = np.asarray(lon_deg, dtype=np.float32)
    lat_min, lat_max = fov_lat
    lon_min, lon_max = fov_lon
    norm_lat = 2.0 * (lat - lat_min) / (lat_max - lat_min) - 1.0
    norm_lon = 2.0 * (lon - lon_min) / (lon_max - lon_min) - 1.0
    return np.stack([norm_lat, norm_lon], axis=-1)


# ---------------------------------------------------------------------------
# JAX angle conversion (relocated from utils/jax_core/helpers.py, r16)
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
# JAX lat/lon conversions
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
# JAX spherical <-> Cartesian
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
