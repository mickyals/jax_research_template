"""
experiments/tc_perceiver_io/data/transforms/encoding.py

Coordinate encode/decode pairs — the experiment's model contract for how
positions enter the network. Each encode has its exact inverse beside it so
dataset assembly (encode) and plotting/diagnostics (decode) can never drift
apart (they were previously hand-maintained in two files).

One of the swappable input transforms (see data/inputs.py InputSpec): the
``location_encoding`` config key selects a sample-level coordinate encoder
from ``COORD_ENCODERS`` (and the matching decoder from ``COORD_DECODERS``).

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

The low-level ``encode_*``/``decode_*`` pairs are NumPy, vectorised, and
accept scalars or arrays. The registered sample-level encoders
(``COORD_ENCODERS``) take the per-sample station frame plus the query
position and return the padded-ready ``(station_coords, query_coords)``.
"""

from __future__ import annotations

import numpy as np

from utils.geoscience.geodesic import vincenty_np
from utils.registry import Registry

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


# ---------------------------------------------------------------------------
# Sample-level coordinate encoders (the swappable input transform)
# ---------------------------------------------------------------------------
#
# Both encoders share one signature so the dataset can call them uniformly:
#
#     station_coords, query_coords = encoder(
#         df, query_lat, query_lon,
#         radius_km=..., fov_lat=..., fov_lon=...,
#     )
#
# `df` is the per-sample station frame (already trimmed to max_stations) from
# InsituLandDataset.get_obs_near — it carries 'latitude', 'longitude', and
# (for unit_circle) 'distance_km'. station_coords is (N, 2); query_coords is
# (2,). Each encoder ignores the kwargs it does not need (unit_circle ignores
# fov_*, domain ignores radius_km).

COORD_ENCODERS = Registry("coord_encoder")
COORD_DECODERS = Registry("coord_decoder")


@COORD_ENCODERS.register("unit_circle", "storm-centred local x-y (tangent plane)")
def _unit_circle_coords(df, query_lat, query_lon,
                        *, radius_km, fov_lat=None, fov_lon=None):
    """Encode station positions on the storm-centred unit map.

    Distance comes from the candidate frame's 'distance_km'; bearing is the
    forward azimuth from the query to each station (Vincenty). NaN bearings
    (near-coincident points) collapse to 0. The query sits at the origin.
    """
    n         = len(df)
    dist_km   = df['distance_km'].to_numpy(dtype=np.float32)
    norm_dist = np.clip(dist_km / radius_km, 0.0, 1.0)
    raw_lats  = df['latitude'].to_numpy(dtype=np.float32)
    raw_lons  = df['longitude'].to_numpy(dtype=np.float32)

    _, bearing_deg, _, _ = vincenty_np(
        np.full(n, query_lat),
        np.full(n, query_lon),
        raw_lats.astype(np.float64),
        raw_lons.astype(np.float64),
    )
    bearing_rad = np.radians(bearing_deg).astype(np.float32)
    bearing_rad = np.where(np.isfinite(bearing_rad), bearing_rad, 0.0)

    x, y = encode_unit_circle(norm_dist, bearing_rad)
    station_coords = np.stack([x, y], axis=-1).astype(np.float32)   # (N, 2)
    # (0, 0) = the storm position on the local map; the model adds a learned
    # content token for the query.
    query_coords   = np.zeros(2, dtype=np.float32)
    return station_coords, query_coords


@COORD_ENCODERS.register("domain", "absolute FOV-normalised lat/lon")
def _domain_coords(df, query_lat, query_lon,
                   *, radius_km=None, fov_lat, fov_lon):
    """Encode station + query positions as FOV-normalised lat/lon."""
    raw_lats = df['latitude'].to_numpy(dtype=np.float32)
    raw_lons = df['longitude'].to_numpy(dtype=np.float32)

    norm_lat, norm_lon = encode_domain(raw_lats, raw_lons, fov_lat, fov_lon)
    station_coords = np.stack([norm_lat, norm_lon], axis=-1).astype(np.float32)

    q_norm_lat, q_norm_lon = encode_domain(query_lat, query_lon, fov_lat, fov_lon)
    query_coords = np.array([q_norm_lat, q_norm_lon], dtype=np.float32)
    return station_coords, query_coords


# Decoders keyed by the same names (plotting/diagnostics recover geographic
# coordinates from encoded ones — see evaluate.domain_latlon_for_sample).
COORD_DECODERS.register("unit_circle", "→ (norm_dist, bearing_rad)")(decode_unit_circle)
COORD_DECODERS.register("domain", "→ (lat, lon) degrees")(decode_domain)


def get_coord_encoder(name: str):
    """Return the sample-level coordinate encoder registered under ``name``."""
    return COORD_ENCODERS[name]


def get_coord_decoder(name: str):
    """Return the coordinate decoder registered under ``name``."""
    return COORD_DECODERS[name]
