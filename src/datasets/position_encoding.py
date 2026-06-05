"""
datasets/position_encoding.py

Pure positional encoding functions for geospatial coordinates.

Three modes, all producing arrays suitable as input to a Tancik-style
random Fourier feature encoder:

    storm_relative_polar  -- 3D, storm-relative polar (physically motivated)
    unit_sphere           -- 3D Cartesian on the unit sphere (global baseline)
    domain_normalised     -- 2D normalised within a bounding box (domain-locked)

All functions operate on NumPy float32 arrays.  The dispatcher
``encode_positions`` selects the right function from a mode string.
"""

from __future__ import annotations

import numpy as np

from utils.geoscience.geodesic import vincenty_np


def encode_storm_relative_polar(
    lat:       np.ndarray,
    lon:       np.ndarray,
    storm_lat: np.ndarray | float,
    storm_lon: np.ndarray | float,
    radius_km: float,
) -> np.ndarray:
    """Encode (lat, lon) in a storm-relative polar coordinate system.

    Origin is the storm centre.  Distance is normalised so r=1 at the edge
    of the observing window (``radius_km``).  Bearing is embedded as
    (sin θ, cos θ) to avoid periodicity discontinuities.

    Parameters
    ----------
    lat, lon : np.ndarray  shape (n,)
        Station positions in decimal degrees.
    storm_lat, storm_lon : array-like
        Storm centre position(s) in decimal degrees.
        Scalar or shape (n,) for per-row storm centres.
    radius_km : float
        Normalisation radius.  Distances beyond this are clipped to r=1.

    Returns
    -------
    np.ndarray  shape (n, 3)
        Columns: [r, sin(θ), cos(θ)]
        r is clipped to [0, 1].  θ is the forward azimuth from the storm
        centre to the station (clockwise from north, in radians).

    Example
    -------
    >>> lat = np.array([24.5, 26.0])
    >>> lon = np.array([-88.0, -85.0])
    >>> enc = encode_storm_relative_polar(lat, lon, 24.0, -87.0, 500.0)
    >>> enc.shape
    (2, 3)
    """
    dist_km, fwd_az_deg, _, _ = vincenty_np(
        np.asarray(storm_lat, dtype=np.float32),
        np.asarray(storm_lon, dtype=np.float32),
        np.asarray(lat,       dtype=np.float32),
        np.asarray(lon,       dtype=np.float32),
    )
    r     = (dist_km / radius_km).clip(0.0, 1.0).astype(np.float32)
    theta = np.radians(fwd_az_deg).astype(np.float32)
    return np.stack([r, np.sin(theta), np.cos(theta)], axis=1)


def encode_unit_sphere(
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Map (lat, lon) in degrees to 3D Cartesian coordinates on the unit sphere.

    Handles longitude periodicity and latitude non-periodicity correctly.
    No false connections at ±180° longitude or at the poles.

    Parameters
    ----------
    lat, lon : np.ndarray  shape (n,)
        In decimal degrees.

    Returns
    -------
    np.ndarray  shape (n, 3)
        Columns: [x, y, z]

    Example
    -------
    >>> lat = np.array([0.0, 90.0])
    >>> lon = np.array([0.0,  0.0])
    >>> enc = encode_unit_sphere(lat, lon)
    >>> enc.shape
    (2, 3)
    """
    lat_r = np.radians(np.asarray(lat, dtype=np.float32))
    lon_r = np.radians(np.asarray(lon, dtype=np.float32))
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.stack([x, y, z], axis=1)


def encode_domain_normalised(
    lat:     np.ndarray,
    lon:     np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    """Normalise (lat, lon) to [-1, 1] x [-1, 1] within a bounding box.

    Simple and fast.  Breaks for positions outside the bounding box —
    only use within a fixed training domain.

    Parameters
    ----------
    lat, lon : np.ndarray  shape (n,)
        In decimal degrees.
    lat_min, lat_max, lon_min, lon_max : float
        Bounding box in decimal degrees.

    Returns
    -------
    np.ndarray  shape (n, 2)
        Columns: [lat_norm, lon_norm], each in [-1, 1] within the bbox.

    Example
    -------
    >>> lat = np.array([15.0, 0.0, 30.0])
    >>> lon = np.array([-72.5, -100.0, -45.0])
    >>> enc = encode_domain_normalised(lat, lon, 0., 30., -100., -45.)
    >>> enc.shape
    (3, 2)
    """
    lat_norm = (2.0 * (np.asarray(lat, dtype=np.float32) - lat_min) / (lat_max - lat_min) - 1.0)
    lon_norm = (2.0 * (np.asarray(lon, dtype=np.float32) - lon_min) / (lon_max - lon_min) - 1.0)
    return np.stack([lat_norm, lon_norm], axis=1)


def encode_positions(
    lat: np.ndarray,
    lon: np.ndarray,
    mode: str,
    *,
    storm_lat: np.ndarray | float | None = None,
    storm_lon: np.ndarray | float | None = None,
    radius_km: float | None = None,
    lat_min:   float | None = None,
    lat_max:   float | None = None,
    lon_min:   float | None = None,
    lon_max:   float | None = None,
) -> np.ndarray:
    """Dispatch to the correct positional encoding function by mode name.

    Parameters
    ----------
    lat, lon : np.ndarray  shape (n,)
    mode : str
        'storm_relative_polar' | 'unit_sphere' | 'domain_normalised'
    storm_lat, storm_lon : required for storm_relative_polar
    radius_km : required for storm_relative_polar
    lat_min, lat_max, lon_min, lon_max : required for domain_normalised

    Returns
    -------
    np.ndarray
        shape (n, 3) for storm_relative_polar and unit_sphere;
        shape (n, 2) for domain_normalised.

    Raises
    ------
    ValueError
        For an unknown mode or missing required arguments.
    """
    if mode == "storm_relative_polar":
        if storm_lat is None or storm_lon is None or radius_km is None:
            raise ValueError(
                "storm_relative_polar requires storm_lat, storm_lon, and radius_km"
            )
        return encode_storm_relative_polar(lat, lon, storm_lat, storm_lon, radius_km)

    if mode == "unit_sphere":
        return encode_unit_sphere(lat, lon)

    if mode == "domain_normalised":
        if any(v is None for v in (lat_min, lat_max, lon_min, lon_max)):
            raise ValueError(
                "domain_normalised requires lat_min, lat_max, lon_min, lon_max"
            )
        return encode_domain_normalised(lat, lon, lat_min, lat_max, lon_min, lon_max)

    raise ValueError(
        f"Unknown position_encoding_mode '{mode}'. "
        "Choose from: storm_relative_polar, unit_sphere, domain_normalised."
    )
