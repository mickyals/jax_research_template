import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
from functools import partial

# ---------------------------------------------------------------------------
# Sources
#
# Vincenty inverse formula:
#   Maurycy Pietrzak
#   https://github.com/maurycyp/vincenty/blob/master/vincenty/__init__.py
#
# Haversine formula:
#   mapado/haversine
#   https://github.com/mapado/haversine/blob/main/haversine/haversine.py
#
# NumPy and JAX adaptations (vectorization, JIT, differentiability):
#   present file.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# float32 vs float64 precision note
#
# The _np variants operate in float64 (NumPy default). The _jax variants
# operate in float32 (JAX default on most hardware). This matters at very
# short distances: coincident or near-coincident points that return exactly
# 0.0 in NumPy may return a small residual (~1e-4 km, i.e. ~10 cm) in JAX
# due to float32 rounding through the trig and sqrt chain. For geophysical
# applications this is negligible, but if sub-metre precision is required
# cast inputs to float64 before calling:
#
#   haversine_jax(lat1.astype(jnp.float64), ...)
#   vincenty_jax(lat1.astype(jnp.float64), ...)
#
# Note: float64 requires JAX to be launched with x64 enabled:
#   jax.config.update("jax_enable_x64", True)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WGS-84 constants (Vincenty)
# ---------------------------------------------------------------------------

_A = 6_378_137.0        # semi-major axis, meters
_F = 1 / 298.257223563  # flattening
_B = 6_356_752.314245   # semi-minor axis, meters

_MAX_ITER = 200
_TOL      = 1e-12

# mean earth radius — https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
_AVG_EARTH_RADIUS_KM = 6371.0088

# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def haversine_np(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
    radius: float = _AVG_EARTH_RADIUS_KM,
) -> np.ndarray:
    """
    Haversine great-circle distance (NumPy, vectorized, non-differentiable).

    Adapted from mapado/haversine:
    https://github.com/mapado/haversine/blob/main/haversine/haversine.py

    Inputs are decimal degrees in [-90, 90] for latitude and [-180, 180]
    for longitude, broadcast-compatible shapes. No pre-conversion required.
    Returns distance in the same unit as `radius` (default km).

    Assumes spherical earth; error relative to Vincenty is under 0.3%
    for distances within ~1000 km. Prefer vincenty_np when azimuth is
    also needed or geodetic accuracy matters.

    Parameters
    ----------
    lat1_deg, lon1_deg : array-like
        Origin point(s) in decimal degrees.
    lat2_deg, lon2_deg : array-like
        Target point(s) in decimal degrees.
    radius : float
        Earth radius in the desired output unit. Default is mean radius in km.

    Returns
    -------
    np.ndarray
        Great-circle distance in the same unit as `radius`.

    Example
    -------
    >>> dist = haversine_np(43.65, -79.38, 40.71, -74.01)
    >>> round(float(dist), 1)
    549.2
    """
    lat1_deg, lon1_deg, lat2_deg, lon2_deg = map(
        np.asarray, (lat1_deg, lon1_deg, lat2_deg, lon2_deg)
    )

    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg)
    lon2 = np.radians(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat * 0.5) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a).clip(0.0, 1.0))


def initial_bearing_np(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    """Great-circle initial bearing (forward azimuth) from point 1 to point 2.

    Closed-form spherical formula — degrees in ``[0, 360)``, 0 = north, 90 =
    east. One-shot (no iteration), unlike ``vincenty_np``: for storm-centred
    station maps at ≤1000 km the ellipsoidal correction is sub-degree, and it
    matches the spherical ``haversine_np`` distance already used. Coincident
    points return 0.

    >>> round(float(initial_bearing_np(0.0, 0.0, 0.0, 1.0)), 1)
    90.0
    """
    lat1_deg, lon1_deg, lat2_deg, lon2_deg = map(
        np.asarray, (lat1_deg, lon1_deg, lat2_deg, lon2_deg)
    )
    lat1 = np.radians(lat1_deg)
    lat2 = np.radians(lat2_deg)
    dlon = np.radians(lon2_deg - lon1_deg)

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360.0


@partial(jax.jit, static_argnames=("radius",))
def haversine_jax(
    lat1_deg: jax.Array,
    lon1_deg: jax.Array,
    lat2_deg: jax.Array,
    lon2_deg: jax.Array,
    radius: float = _AVG_EARTH_RADIUS_KM,
) -> jax.Array:
    """
    Haversine great-circle distance (JAX, JIT-compiled, differentiable).

    Adapted from mapado/haversine:
    https://github.com/mapado/haversine/blob/main/haversine/haversine.py

    Inputs are decimal degrees in [-90, 90] for latitude and [-180, 180]
    for longitude. No pre-conversion required. Returns distance in the
    same unit as `radius` (default km). Fully differentiable w.r.t. all
    four coordinate inputs via jax.grad / jax.jacfwd.

    Parameters
    ----------
    lat1_deg, lon1_deg : jax.Array
        Origin point(s) in decimal degrees.
    lat2_deg, lon2_deg : jax.Array
        Target point(s) in decimal degrees.
    radius : float
        Earth radius in the desired output unit. Must be a Python float
        (static — traced once per unique value). Default is mean radius in km.

    Returns
    -------
    jax.Array
        Great-circle distance in the same unit as `radius`.

    Example
    -------
    >>> dist = haversine_jax(
    ...     jnp.array(43.65), jnp.array(-79.38),
    ...     jnp.array(40.71), jnp.array(-74.01),
    ... )
    >>> round(float(dist), 1)
    549.2
    """
    lat1 = jnp.radians(lat1_deg)
    lon1 = jnp.radians(lon1_deg)
    lat2 = jnp.radians(lat2_deg)
    lon2 = jnp.radians(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        jnp.sin(dlat * 0.5) ** 2
        + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon * 0.5) ** 2
    )

    return 2 * radius * jnp.arcsin(jnp.sqrt(a).clip(0.0, 1.0))


# ---------------------------------------------------------------------------
# Vincenty
# ---------------------------------------------------------------------------

def vincenty_np(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
    embed_bearing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    """
    Vincenty inverse formula (NumPy, vectorized, non-differentiable).

    Adapted from Maurycy Pietrzak's implementation:
    https://github.com/maurycyp/vincenty/blob/master/vincenty/__init__.py

    Inputs are decimal degrees in [-90, 90] for latitude and [-180, 180]
    for longitude, broadcast-compatible shapes. No pre-conversion required.

    Parameters
    ----------
    lat1_deg, lon1_deg : array-like
        Origin point (e.g. station position).
    lat2_deg, lon2_deg : array-like
        Target point (e.g. storm center from IBTrACS at time t).
    embed_bearing : bool, default False
        If True, also compute and return the (sin, cos) circular embedding
        of the forward azimuth. Cost is two elementwise ops on an existing
        array; negligible relative to the iterative solve.

    Returns
    -------
    distance_km : np.ndarray
        Geodesic distance in kilometers.
    forward_azimuth_deg : np.ndarray
        Bearing from point1 toward point2, clockwise from north, in [0, 360).
    back_azimuth_deg : np.ndarray
        Bearing from point2 back toward point1, clockwise from north, in [0, 360).
    bearing_embedding : tuple[np.ndarray, np.ndarray] or None
        (bearing_sin, bearing_cos) circular embedding of the forward azimuth,
        or None when embed_bearing=False.

    NaN is returned for any pair where iteration diverges (near-antipodal points).

    Example
    -------
    >>> dist, fwd, back, _ = vincenty_np(43.65, -79.38, 40.71, -74.01)
    >>> round(float(dist), 1)
    549.1
    """
    lat1_deg, lon1_deg, lat2_deg, lon2_deg = map(
        np.asarray, (lat1_deg, lon1_deg, lat2_deg, lon2_deg)
    )

    U1 = np.arctan((1 - _F) * np.tan(np.radians(lat1_deg)))
    U2 = np.arctan((1 - _F) * np.tan(np.radians(lat2_deg)))
    L  = np.radians(lon2_deg - lon1_deg)

    sinU1, cosU1 = np.sin(U1), np.cos(U1)
    sinU2, cosU2 = np.sin(U2), np.cos(U2)

    lam       = np.broadcast_to(L, np.broadcast_shapes(U1.shape, U2.shape, L.shape)).copy()
    converged = np.zeros(lam.shape, dtype=bool)

    for _ in range(_MAX_ITER):
        sinLam, cosLam = np.sin(lam), np.cos(lam)
        sin_sig = np.sqrt(
            (cosU2 * sinLam) ** 2
            + (cosU1 * sinU2 - sinU1 * cosU2 * cosLam) ** 2
        )
        cos_sig  = sinU1 * sinU2 + cosU1 * cosU2 * cosLam
        sigma    = np.arctan2(sin_sig, cos_sig)
        # Masked divides: only evaluate num/denom where denom != 0 (else 0). The
        # np.where form computed the division everywhere first, raising a benign
        # 0/0 RuntimeWarning at coincident/antipodal points before discarding it.
        sin_alp  = np.divide(cosU1 * cosU2 * sinLam, sin_sig,
                             out=np.zeros_like(lam), where=sin_sig != 0)
        c2a      = 1 - sin_alp ** 2
        _equ     = np.divide(2 * sinU1 * sinU2, c2a,
                             out=np.zeros_like(lam), where=c2a != 0)
        cos2m    = np.where(c2a == 0, 0.0, cos_sig - _equ)
        C        = _F / 16 * c2a * (4 + _F * (4 - 3 * c2a))
        lam_prev = lam
        lam      = L + (1 - C) * _F * sin_alp * (
            sigma + C * sin_sig * (cos2m + C * cos_sig * (-1 + 2 * cos2m ** 2))
        )
        converged |= np.abs(lam - lam_prev) < _TOL

    sinLam, cosLam = np.sin(lam), np.cos(lam)
    sin_sig = np.sqrt(
        (cosU2 * sinLam) ** 2
        + (cosU1 * sinU2 - sinU1 * cosU2 * cosLam) ** 2
    ).clip(1e-15)
    cos_sig = sinU1 * sinU2 + cosU1 * cosU2 * cosLam
    sigma   = np.arctan2(sin_sig, cos_sig)
    sin_alp = cosU1 * cosU2 * sinLam / sin_sig
    c2a     = 1 - sin_alp ** 2
    _equ    = np.divide(2 * sinU1 * sinU2, c2a,
                        out=np.zeros_like(lam), where=c2a != 0)
    cos2m   = np.where(c2a == 0, 0.0, cos_sig - _equ)

    forward_azimuth_deg: np.ndarray = np.degrees(np.arctan2(
        cosU2 * sinLam,
        cosU1 * sinU2 - sinU1 * cosU2 * cosLam,
    )) % 360

    back_azimuth_deg: np.ndarray = np.degrees(np.arctan2(
        cosU1 * sinLam,
        -sinU1 * cosU2 + cosU1 * sinU2 * cosLam,
    )) % 360

    uSq = c2a * (_A ** 2 - _B ** 2) / _B ** 2
    Av  = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    Bv  = uSq / 1024       * (256  + uSq * (-128 + uSq * ( 74 -  47 * uSq)))
    ds  = Bv * sin_sig * (
        cos2m + Bv / 4 * (
            cos_sig * (-1 + 2 * cos2m ** 2)
            - Bv / 6 * cos2m * (-3 + 4 * sin_sig ** 2) * (-3 + 4 * cos2m ** 2)
        )
    )

    distance_km: np.ndarray = _B * Av * (sigma - ds) / 1000.0

    coincident = sin_sig < 2e-15
    diverged   = ~converged & ~coincident
    distance_km         = np.where(coincident, 0.0, np.where(diverged, np.nan, distance_km))
    forward_azimuth_deg = np.where(coincident, 0.0, np.where(diverged, np.nan, forward_azimuth_deg))
    back_azimuth_deg    = np.where(coincident, 0.0, np.where(diverged, np.nan, back_azimuth_deg))

    if embed_bearing:
        az_rad = np.radians(forward_azimuth_deg)
        bearing_embedding: tuple[np.ndarray, np.ndarray] | None = (
            np.sin(az_rad), np.cos(az_rad)
        )
    else:
        bearing_embedding = None

    return distance_km, forward_azimuth_deg, back_azimuth_deg, bearing_embedding


@partial(jax.jit, static_argnames=("embed_bearing",))
def vincenty_jax(
    lat1_deg: jax.Array,
    lon1_deg: jax.Array,
    lat2_deg: jax.Array,
    lon2_deg: jax.Array,
    embed_bearing: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, jax.Array] | None]:
    """
    Vincenty inverse formula (JAX, JIT-compiled, differentiable).

    Adapted from Maurycy Pietrzak's implementation:
    https://github.com/maurycyp/vincenty/blob/master/vincenty/__init__.py

    Inputs are decimal degrees in [-90, 90] for latitude and [-180, 180]
    for longitude. No pre-conversion required.

    Parameters
    ----------
    lat1_deg, lon1_deg : jax.Array
        Origin point (e.g. station position).
    lat2_deg, lon2_deg : jax.Array
        Target point (e.g. storm center from IBTrACS at time t).
    embed_bearing : bool, default False
        If True, also compute and return the (sin, cos) circular embedding
        of the forward azimuth. Must be a static value so XLA traces a
        separate kernel per branch. Cost is two elementwise ops on an
        existing array; negligible relative to the iterative solve.

    Returns
    -------
    distance_km : jax.Array
        Geodesic distance in kilometers.
    forward_azimuth_deg : jax.Array
        Bearing from point1 toward point2, clockwise from north, in [0, 360).
    back_azimuth_deg : jax.Array
        Bearing from point2 back toward point1, clockwise from north, in [0, 360).
    bearing_embedding : tuple[jax.Array, jax.Array] or None
        (bearing_sin, bearing_cos) circular embedding of the forward azimuth,
        or None when embed_bearing=False.

    NaN is returned for any pair where iteration diverges (near-antipodal points).

    Differentiable via fixed-point re-evaluation: the iterative lambda solve
    runs under lax.stop_gradient; the closed-form distance and azimuth
    expressions are re-evaluated once at the converged lambda with autograd
    enabled. Gradients w.r.t. inputs are accurate to the tolerance of the
    fixed-point solve (~1e-12 rad).

    Example
    -------
    >>> dist, fwd, back, _ = vincenty_jax(
    ...     jnp.array(43.65), jnp.array(-79.38),
    ...     jnp.array(40.71), jnp.array(-74.01),
    ... )
    >>> round(float(dist), 1)
    549.1
    """
    U1 = jnp.arctan((1 - _F) * jnp.tan(jnp.radians(lat1_deg)))
    U2 = jnp.arctan((1 - _F) * jnp.tan(jnp.radians(lat2_deg)))
    L  = jnp.radians(lon2_deg - lon1_deg)

    sinU1, cosU1 = jnp.sin(U1), jnp.cos(U1)
    sinU2, cosU2 = jnp.sin(U2), jnp.cos(U2)

    def body(carry):
        lam, _, i = carry
        sinLam, cosLam = jnp.sin(lam), jnp.cos(lam)
        sin_sig = jnp.sqrt(
            (cosU2 * sinLam) ** 2
            + (cosU1 * sinU2 - sinU1 * cosU2 * cosLam) ** 2
        ).clip(1e-15)
        cos_sig  = sinU1 * sinU2 + cosU1 * cosU2 * cosLam
        sigma    = jnp.arctan2(sin_sig, cos_sig)
        sin_alp  = cosU1 * cosU2 * sinLam / sin_sig
        c2a      = 1 - sin_alp ** 2
        cos2m    = jnp.where(c2a == 0, 0.0, cos_sig - 2 * sinU1 * sinU2 / c2a)
        C        = _F / 16 * c2a * (4 + _F * (4 - 3 * c2a))
        lam_new  = L + (1 - C) * _F * sin_alp * (
            sigma + C * sin_sig * (cos2m + C * cos_sig * (-1 + 2 * cos2m ** 2))
        )
        delta = jnp.max(jnp.abs(lam_new - lam))
        return lam_new, delta, i + 1

    def cond(carry):
        _, delta, i = carry
        return (delta > _TOL) & (i < _MAX_ITER)

    init = (L, jnp.inf, jnp.zeros((), jnp.int32))
    lam, final_delta, _ = lax.while_loop(cond, body, init)

    # stop gradient so autograd does not attempt to diff through the loop
    lam = lax.stop_gradient(lam)

    sinLam, cosLam = jnp.sin(lam), jnp.cos(lam)
    sin_sig = jnp.sqrt(
        (cosU2 * sinLam) ** 2
        + (cosU1 * sinU2 - sinU1 * cosU2 * cosLam) ** 2
    ).clip(1e-15)
    cos_sig = sinU1 * sinU2 + cosU1 * cosU2 * cosLam
    sigma   = jnp.arctan2(sin_sig, cos_sig)
    sin_alp = cosU1 * cosU2 * sinLam / sin_sig
    c2a     = 1 - sin_alp ** 2
    cos2m   = jnp.where(c2a == 0, 0.0, cos_sig - 2 * sinU1 * sinU2 / c2a)

    forward_azimuth_deg: jax.Array = jnp.degrees(jnp.arctan2(
        cosU2 * sinLam,
        cosU1 * sinU2 - sinU1 * cosU2 * cosLam,
    )) % 360

    back_azimuth_deg: jax.Array = jnp.degrees(jnp.arctan2(
        cosU1 * sinLam,
        -sinU1 * cosU2 + cosU1 * sinU2 * cosLam,
    )) % 360

    uSq = c2a * (_A ** 2 - _B ** 2) / _B ** 2
    Av  = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    Bv  = uSq / 1024       * (256  + uSq * (-128 + uSq * ( 74 -  47 * uSq)))
    ds  = Bv * sin_sig * (
        cos2m + Bv / 4 * (
            cos_sig * (-1 + 2 * cos2m ** 2)
            - Bv / 6 * cos2m * (-3 + 4 * sin_sig ** 2) * (-3 + 4 * cos2m ** 2)
        )
    )

    distance_km: jax.Array = _B * Av * (sigma - ds) / 1000.0

    coincident = sin_sig < 2e-15
    diverged   = final_delta > _TOL
    distance_km         = jnp.where(coincident, 0.0, jnp.where(diverged, jnp.nan, distance_km))
    forward_azimuth_deg = jnp.where(coincident, 0.0, jnp.where(diverged, jnp.nan, forward_azimuth_deg))
    back_azimuth_deg    = jnp.where(coincident, 0.0, jnp.where(diverged, jnp.nan, back_azimuth_deg))

    if embed_bearing:
        az_rad = jnp.radians(forward_azimuth_deg)
        bearing_embedding: tuple[jax.Array, jax.Array] | None = (
            jnp.sin(az_rad), jnp.cos(az_rad)
        )
    else:
        bearing_embedding = None

    return distance_km, forward_azimuth_deg, back_azimuth_deg, bearing_embedding


# ---------------------------------------------------------------------------
# Spherical areas
# ---------------------------------------------------------------------------

def latlon_box_area(lon_min, lon_max, lat_min, lat_max,
                       radius: float = _AVG_EARTH_RADIUS_KM) -> float:
    """EXACT area of a lon/lat box on the sphere (like haversine is the
    exact great-circle distance, this is the exact spherical-surface area
    of the box — NOT the flat-map dlon*dlat rectangle, which overstates
    high-latitude area).

    area = R^2 * dlon_rad * (sin(lat_max) - sin(lat_min))

    Parameters
    ----------
    lon_min, lon_max, lat_min, lat_max : float, degrees
    radius : float
        Sphere radius; default mean earth radius in km -> area in km^2.

    Returns
    -------
    float  area in radius-units squared.
    """
    dlon = np.radians(lon_max - lon_min)
    return float(radius ** 2 * dlon
                 * (np.sin(np.radians(lat_max)) - np.sin(np.radians(lat_min))))
