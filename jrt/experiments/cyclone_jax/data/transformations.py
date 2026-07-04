"""
experiments/cyclone_jax/data/transformations.py

MECHANICS of sample assembly — pure numpy functions. Policy (which vars,
which ID codes, norm bounds) lives with the sampler/config; these are the
array-level operations they resolve to.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Normalisation — fn(x, a, b) per-channel; a/b broadcast over the last axis
# ---------------------------------------------------------------------------

def normalise_01(x, lo, hi):
    """Min-max to [0, 1]."""
    return (x - lo) / (hi - lo + 1e-12)


def normalise_11(x, lo, hi):
    """Min-max to [-1, 1]."""
    return 2.0 * normalise_01(x, lo, hi) - 1.0


def standardise(x, mean, std):
    """Zero-mean unit-variance."""
    return (x - mean) / (std + 1e-8)


NORMALISERS = {'minmax_01': normalise_01, 'minmax_11': normalise_11,
               'standardise': standardise, 'none': None}


# ---------------------------------------------------------------------------
# Wind decomposition
# ---------------------------------------------------------------------------

def wind_to_uv(speed, direction_deg):
    """Meteorological (speed, FROM-direction) -> (u, v) components.

    u = -speed*sin(theta), v = -speed*cos(theta) (wind FROM north blows
    southward: u=0, v=-speed). Calm (speed == 0) -> (0, 0) even with NaN
    direction; NaN speed (or NaN direction with speed > 0) stays NaN.
    """
    speed = np.asarray(speed, np.float32)
    theta = np.deg2rad(np.asarray(direction_deg, np.float32))
    u = -speed * np.sin(theta)
    v = -speed * np.cos(theta)
    calm = speed == 0.0
    u = np.where(calm, 0.0, u).astype(np.float32)
    v = np.where(calm, 0.0, v).astype(np.float32)
    return u, v


# ---------------------------------------------------------------------------
# Missingness + source ID
# ---------------------------------------------------------------------------

def build_missingness(values):
    """(values NaN->0, mask) — mask True where a measurement was present."""
    values = np.asarray(values, np.float32)
    mask = np.isfinite(values)
    return np.where(mask, values, 0.0).astype(np.float32), mask


def stamp_source_id(n, code):
    """(n, 1) float32 column of the scalar source code (-1 land / +1 marine
    / 0 upper)."""
    return np.full((n, 1), float(code), np.float32)
