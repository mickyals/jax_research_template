"""
experiments/sparse_obs_cross_attn/data/sources/insitu_land.py

InsituLandDataset: efficient spatial + temporal query over land-surface
station observations.

Design
------
The obs file (74 M rows) is sorted by report_timestamp once on load so
that time-window queries use binary search (O(log N)) rather than a full
scan.  After narrowing to the time window, haversine distance filtering
selects stations within radius_km.

Station IDs are converted to integers on load so that reliability
filtering uses fast integer np.isin instead of string comparisons.

Two files on disk:
    insitu_land_clean.npz         — 74 M flat obs rows
    insitu_land_station_meta.npz  — 552 station metadata rows
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.geoscience.geodesic import haversine_np


# ---------------------------------------------------------------------------
# Observable variable lists
# ---------------------------------------------------------------------------

ALL_OBS_VARS: list[str] = [
    'air_pressure',
    'air_pressure_at_sea_level',
    'air_temperature',
    'dew_point_temperature',
    'wind_speed',
    'wind_from_direction',
]

DEFAULT_OBS_VARS: list[str] = [
    'air_pressure_at_sea_level',
    'air_temperature',
    'dew_point_temperature',
    'wind_speed',
    'wind_from_direction',
]

RELIABILITY_LEVELS: list[str] = [
    'always_active',
    'mostly_active',
    'sparse',
    'sporadic',
    'unusable',
]


# ---------------------------------------------------------------------------
# InsituLandDataset
# ---------------------------------------------------------------------------

class InsituLandDataset:
    """Land-surface station observations with spatial + temporal queries.

    Parameters
    ----------
    obs_path : str or Path
        Path to insitu_land_clean.npz.
    meta_path : str or Path
        Path to insitu_land_station_meta.npz.
    """

    def __init__(self, obs_path: str | Path, meta_path: str | Path) -> None:
        self.obs_path  = Path(obs_path)
        self.meta_path = Path(meta_path)

        obs_raw  = np.load(self.obs_path,  allow_pickle=True)
        meta_raw = np.load(self.meta_path, allow_pickle=True)

        # Station metadata (552 rows) — keep as-is
        self._meta: dict[str, np.ndarray] = {k: meta_raw[k] for k in meta_raw.files}

        # Build station-ID integer index from obs before sorting.
        # np.unique(return_inverse=True) maps every string ID to an int in O(N log N).
        raw_ids = obs_raw['primary_station_id']
        self._unique_ids, inv = np.unique(raw_ids, return_inverse=True)
        # inv is int64 indices into _unique_ids — (74M,)

        # Sort all obs arrays by timestamp once.
        sort_order = np.argsort(obs_raw['report_timestamp'])
        self._obs: dict[str, np.ndarray] = {
            k: obs_raw[k][sort_order] for k in obs_raw.files
        }
        self._obs_station_int: np.ndarray = inv[sort_order].astype(np.int32)
        self._timestamps: np.ndarray = self._obs['report_timestamp']   # sorted int64

    # ------------------------------------------------------------------
    # Internal factory used by filter_reliability and split
    # ------------------------------------------------------------------

    @classmethod
    def _from_arrays(cls, obs: dict[str, np.ndarray], obs_station_int: np.ndarray, unique_ids: np.ndarray, meta: dict[str, np.ndarray], obs_path: Path, meta_path: Path,) -> InsituLandDataset:

        obj                  = cls.__new__(cls)
        obj.obs_path         = obs_path
        obj.meta_path        = meta_path
        obj._meta            = meta
        obj._obs             = obs
        obj._obs_station_int = obs_station_int
        obj._unique_ids      = unique_ids
        obj._timestamps      = obs['report_timestamp']
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_stations(self) -> int:
        """Number of unique stations represented in the current obs rows."""
        return int(np.unique(self._obs_station_int).size)

    @property
    def timestamps(self) -> np.ndarray:
        """All sorted int64 report_timestamps (may contain duplicates across stations)."""
        return self._timestamps

    # ------------------------------------------------------------------
    # Reliability filtering
    # ------------------------------------------------------------------

    def filter_reliability(self, levels: list[str]) -> InsituLandDataset:
        """Keep only obs from stations whose reliability is in levels.

        Parameters
        ----------
        levels : list[str]
            Subset of RELIABILITY_LEVELS, e.g. ['always_active', 'mostly_active'].

        Returns
        -------
        InsituLandDataset with obs and meta filtered to matching stations.
        """
        # Find matching station IDs from meta (fast — 552 rows)
        meta_mask  = np.isin(self._meta['station_reliability'], levels)
        valid_ids  = set(self._meta['primary_station_id'][meta_mask].tolist())

        # Map valid string IDs → their integer indices (fast — 552 unique IDs max)
        valid_ints = np.where(np.isin(self._unique_ids, list(valid_ids)))[0]

        # Filter 74 M-row obs array using integer comparisons (fast)
        obs_mask = np.isin(self._obs_station_int, valid_ints)

        filtered_obs = {k: v[obs_mask] for k, v in self._obs.items()}
        filtered_int = self._obs_station_int[obs_mask]
        filtered_meta = {k: v[meta_mask] for k, v in self._meta.items()}

        # Rebuild unique_ids for the filtered subset
        new_unique, new_inv = np.unique(filtered_obs['primary_station_id'],
                                        return_inverse=True)

        return self._from_arrays(
            obs             = filtered_obs,
            obs_station_int = new_inv.astype(np.int32),
            unique_ids      = new_unique,
            meta            = filtered_meta,
            obs_path        = self.obs_path,
            meta_path       = self.meta_path,
        )

    # ------------------------------------------------------------------
    # Spatial + temporal query
    # ------------------------------------------------------------------

    def get_obs_near(
        self,
        query_lat:    float,
        query_lon:    float,
        timestamp_ns: int,
        radius_km:    float,
        window_ns:    int,
        obs_vars:     list[str],
    ) -> pd.DataFrame:
        """Return one observation per station near a query point and time.

        The time window is a tolerance, not a collection interval: within
        ±window_ns each station contributes exactly ONE row — the report
        closest in time to timestamp_ns. Without this dedup an hourly
        station inside a wide window would appear as several near-duplicate
        rows (same coords, different values), inflating candidate counts
        and mixing report times within a sample.

        Parameters
        ----------
        query_lat, query_lon : float
            Query position in decimal degrees.
        timestamp_ns : int
            Centre timestamp in Unix nanoseconds.
        radius_km : float
            Spatial search radius in kilometres.
        window_ns : int
            Temporal tolerance in nanoseconds (±window_ns around
            timestamp_ns).
        obs_vars : list[str]
            Observation variables to include. Must be a subset of ALL_OBS_VARS.

        Returns
        -------
        pd.DataFrame
            Columns: latitude, longitude, primary_station_id, distance_km,
            + all obs_vars.  One row per station, sorted by distance_km
            ascending.  Empty DataFrame if no rows match.
        """
        # 1. Time window via binary search — O(log N)
        lo = int(np.searchsorted(self._timestamps, timestamp_ns - window_ns, side='left'))
        hi = int(np.searchsorted(self._timestamps, timestamp_ns + window_ns, side='right'))

        if lo >= hi:
            return _empty_df(obs_vars)

        # 2. Spatial filter on the time-windowed slice
        lats = self._obs['latitude'][lo:hi]
        lons = self._obs['longitude'][lo:hi]
        dist_km = haversine_np(
            np.float32(query_lat), np.float32(query_lon), lats, lons
        )
        spatial_mask = dist_km <= radius_km

        if not np.any(spatial_mask):
            return _empty_df(obs_vars)

        # 3. Per-station dedup — keep the single report nearest in time
        abs_idx = np.arange(lo, hi)[spatial_mask]
        dist    = dist_km[spatial_mask]
        tdiff   = np.abs(self._timestamps[abs_idx] - timestamp_ns)
        st_int  = self._obs_station_int[abs_idx]
        order   = np.lexsort((tdiff, st_int))   # by station, then |Δt|
        _, first = np.unique(st_int[order], return_index=True)
        keep    = order[first]
        abs_idx = abs_idx[keep]
        dist    = dist[keep]

        # 4. Build output DataFrame
        rows: dict[str, np.ndarray] = {
            'latitude':           self._obs['latitude'][abs_idx],
            'longitude':          self._obs['longitude'][abs_idx],
            'primary_station_id': self._obs['primary_station_id'][abs_idx],
            'distance_km':        dist,
        }
        for var in obs_vars:
            rows[var] = self._obs[var][abs_idx]

        return (
            pd.DataFrame(rows)
            .sort_values('distance_km')
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Temporal filtering
    # ------------------------------------------------------------------

    def filter_years(self, years: list[int]) -> InsituLandDataset:
        """Return obs rows whose calendar year (from report_timestamp) is in years.

        Generalized year-list filter primitive — policy (which years belong
        to which split) lives in the experiment's split resolver, not here.

        Parameters
        ----------
        years : list[int]

        Returns
        -------
        InsituLandDataset
        """
        obs_years = _timestamps_to_years(self._timestamps)
        mask = np.isin(obs_years, years)

        filtered_obs = {k: v[mask] for k, v in self._obs.items()}

        new_unique, new_inv = np.unique(filtered_obs['primary_station_id'],
                                        return_inverse=True)

        return self._from_arrays(
            obs             = filtered_obs,
            obs_station_int = new_inv.astype(np.int32),
            unique_ids      = new_unique,
            meta            = self._meta,
            obs_path        = self.obs_path,
            meta_path       = self.meta_path,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        years = _timestamps_to_years(self._timestamps)
        print(f"InsituLandDataset -- {self.obs_path.name}")
        print(f"  obs rows    : {len(self._timestamps):,}")
        print(f"  stations    : {self.n_stations}")
        print(f"  year range  : {int(years.min())}–{int(years.max())}")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _timestamps_to_years(timestamps_ns: np.ndarray) -> np.ndarray:
    """Convert Unix nanosecond timestamps to approximate calendar years (int32)."""
    return (timestamps_ns / 1e9 / 86400.0 / 365.25 + 1970.0).astype(np.int32) # a year is 365.242 for those who are unaware


def _empty_df(obs_vars: list[str]) -> pd.DataFrame:
    cols = ['latitude', 'longitude', 'primary_station_id', 'distance_km'] + obs_vars
    return pd.DataFrame(columns=cols)
