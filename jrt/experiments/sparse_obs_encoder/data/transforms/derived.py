"""
experiments/sparse_obs_encoder/data/transforms/derived.py

Derived observation variables — one of the swappable input transforms (see
data/inputs.py InputSpec). A derived variable is a model-facing obs name that
is computed from one or more source columns rather than fetched directly from
InsituLandDataset.

Each entry declares its source columns (so the dataset knows what to fetch)
and a ``compute(df) -> np.ndarray`` that produces the derived column from the
per-sample station frame. The wind components are the (u, v) decomposition of
speed + FROM-direction (meteorological convention, see
utils.geoscience.met_conversions) — this kills the 0/360 direction seam and
shrinks low-speed direction noise by magnitude.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from utils.geoscience.met_conversions import wind_to_components
from utils.registry import Registry


class DerivedVar(NamedTuple):
    """A derived obs variable: its source columns and its compute function."""
    source_cols: tuple[str, ...]
    compute:     Callable        # (df) -> np.ndarray (n_real,)


DERIVED_VARS = Registry("derived_var")


@DERIVED_VARS.register("wind_east", "east (u) wind component from speed + dir")
def _wind_east() -> DerivedVar:
    def compute(df):
        u, _ = wind_to_components(
            df['wind_speed'].to_numpy(dtype=np.float32),
            df['wind_from_direction'].to_numpy(dtype=np.float32),
        )
        return u
    return DerivedVar(('wind_speed', 'wind_from_direction'), compute)


@DERIVED_VARS.register("wind_north", "north (v) wind component from speed + dir")
def _wind_north() -> DerivedVar:
    def compute(df):
        _, v = wind_to_components(
            df['wind_speed'].to_numpy(dtype=np.float32),
            df['wind_from_direction'].to_numpy(dtype=np.float32),
        )
        return v
    return DerivedVar(('wind_speed', 'wind_from_direction'), compute)


def resolve_fetch_vars(obs_vars) -> list[str]:
    """Source columns to fetch for ``obs_vars`` (derived names → their sources).

    Order-preserving and deduped — a derived variable's source columns replace
    it in place; plain variables pass through unchanged.
    """
    fetch: list[str] = []
    for v in obs_vars:
        cols = DERIVED_VARS.get(v).source_cols if v in DERIVED_VARS else (v,)
        for c in cols:
            if c not in fetch:
                fetch.append(c)
    return fetch


def compute_derived(df, obs_vars):
    """Return ``df`` with any derived columns in ``obs_vars`` assigned.

    No-op (returns df unchanged) when ``obs_vars`` contains no derived names.
    """
    new = {v: DERIVED_VARS.get(v).compute(df)
           for v in obs_vars if v in DERIVED_VARS}
    return df.assign(**new) if new else df
