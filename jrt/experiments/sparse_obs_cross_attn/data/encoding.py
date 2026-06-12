"""
experiments/sparse_obs_cross_attn/data/encoding.py

Coordinate encode/decode pairs — the experiment's model contract for how
positions enter the network. Each encode has its exact inverse beside it so
dataset assembly (encode) and plotting/diagnostics (decode) can never drift
apart (they were previously hand-maintained in two files).

Two encodings, selected by the ``location_encoding`` config key:

'unit_circle' — storm-centred local map ("tangent plane")
    (norm_dist, bearing_rad) → (x, y) = (d·sin θ, d·cos θ) ∈ [-1, 1]²
    North-up: +y = north, +x = east. The storm sits at the origin, so the
    query sentinel (0, 0) is literally the storm position. Compared with
    feeding (distance, bearing) directly this removes the 0/2π bearing seam,
    makes encoding distance proportional to physical distance everywhere,
    and gives both dimensions the same symmetric scale.

'domain' — absolute FOV-normalised position
    (lat, lon) degrees → each mapped over its FOV span to [-1, 1], scaled
    by π/2.

All functions are NumPy, vectorised, and accept scalars or arrays.
"""

from __future__ import annotations

import numpy as np

_HALF_PI = np.pi / 2.0


# ---------------------------------------------------------------------------
# unit_circle — storm-centred local x-y
# ---------------------------------------------------------------------------

def encode_unit_circle(norm_dist, bearing_rad):
    """Encode (normalised distance, bearing) as local x-y coordinates.

    Parameters
    ----------
    norm_dist : array-like
        Distance from the storm centre normalised by radius_km, in [0, 1].
    bearing_rad : array-like
        Bearing from storm to station, radians clockwise from north.

    Returns
    -------
    tuple
        (x, y) with x = d·sin(θ) (east) and y = d·cos(θ) (north),
        each in [-1, 1].
    """
    norm_dist   = np.asarray(norm_dist,   dtype=np.float32)
    bearing_rad = np.asarray(bearing_rad, dtype=np.float32)
    return norm_dist * np.sin(bearing_rad), norm_dist * np.cos(bearing_rad)


def decode_unit_circle(x, y):
    """Inverse of ``encode_unit_circle``.

    Returns
    -------
    tuple
        (norm_dist, bearing_rad) with norm_dist = hypot(x, y) and bearing
        in [0, 2π). At the origin (the storm position) the bearing is 0 by
        atan2 convention — distance 0 makes it meaningless anyway.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    norm_dist   = np.hypot(x, y)
    bearing_rad = np.arctan2(x, y) % (2.0 * np.pi)
    return norm_dist, bearing_rad


# ---------------------------------------------------------------------------
# domain — absolute FOV-normalised lat/lon
# ---------------------------------------------------------------------------

def encode_domain(lats, lons, fov_lat, fov_lon):
    """Encode lat/lon degrees as FOV-normalised coordinates scaled by π/2.

    Parameters
    ----------
    lats, lons : array-like
        Positions in decimal degrees.
    fov_lat, fov_lon : tuple[float, float]
        (min, max) field-of-view bounds in degrees.

    Returns
    -------
    tuple
        (norm_lat, norm_lon), each in [-π/2, π/2] for in-FOV positions.
    """
    lats = np.asarray(lats, dtype=np.float32)
    lons = np.asarray(lons, dtype=np.float32)
    lat_min, lat_max = fov_lat
    lon_min, lon_max = fov_lon
    norm_lat = ((lats - lat_min) / (lat_max - lat_min + 1e-12) * 2.0 - 1.0) * _HALF_PI
    norm_lon = ((lons - lon_min) / (lon_max - lon_min + 1e-12) * 2.0 - 1.0) * _HALF_PI
    return norm_lat.astype(np.float32), norm_lon.astype(np.float32)


def decode_domain(norm_lat, norm_lon, fov_lat, fov_lon):
    """Inverse of ``encode_domain``: normalised coords → lat/lon degrees."""
    norm_lat = np.asarray(norm_lat, dtype=np.float64)
    norm_lon = np.asarray(norm_lon, dtype=np.float64)
    lat_min, lat_max = fov_lat
    lon_min, lon_max = fov_lon
    lats = (norm_lat / _HALF_PI + 1.0) / 2.0 * (lat_max - lat_min + 1e-12) + lat_min
    lons = (norm_lon / _HALF_PI + 1.0) / 2.0 * (lon_max - lon_min + 1e-12) + lon_min
    return lats, lons
