"""
utils/geoscience/coordinates.py

Pure lat/lon coordinate encoders. NumPy, vectorised, accept scalars or arrays.

These map geographic (lat, lon) in degrees to model-friendly coordinate
representations. They live in the geoscience utils — not in the data-loading
framework — so the generic DataModule stays domain-agnostic; an experiment that
wants to encode lat/lon feature columns calls these directly. (Moved here from
datasets/datamodule._apply_position_encoding, 2026-06-15, plan r12.)
"""

from __future__ import annotations

import numpy as np


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
