"""
experiments/cyclone_jax/data/transformations.py

MECHANICS of sample assembly — pure numpy functions. Policy (which vars,
which ID codes) lives in the specs (inputs.py / targets.py); these are the
array-level operations they resolve to.

Wind decomposition lives in utils/geoscience/met_conversions
(wind_to_components — meteorological FROM convention, calm -> (0, 0));
normalisation utilities live in utils/jax_core/helpers. Neither is
duplicated here.
"""

from __future__ import annotations

import numpy as np


def build_missingness(values):
    """(values NaN->0, mask) — mask True where a measurement was present."""
    values = np.asarray(values, np.float32)
    mask = np.isfinite(values)
    return np.where(mask, values, 0.0).astype(np.float32), mask
