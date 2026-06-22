"""
experiments/tc_perceiver_io/data/sources/insitu_land.py

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

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.geoscience.geodesic import haversine_np

# Marker file + format tag for the pre-sorted, memory-mappable obs layout
# produced by ``InsituLandDataset.prepare_sorted`` (see __init__ fast path).
_SORTED_MANIFEST = 'manifest.json'
_SORTED_FORMAT   = 'insitu_sorted_v1'


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

    ``obs_path`` may be either:
      * the original ``insitu_land_clean.npz`` — loaded and argsort-by-time on
        construction (the 74 M-row sort is the slow startup cost), or
      * a directory produced by :meth:`prepare_sorted` — already sorted, with
        one ``.npy`` per column, **memory-mapped** on load (near-instant; pages
        in from disk on demand). Convert once, then point the config at the dir.

    Parameters
    ----------
    obs_path : str or Path
        Path to insitu_land_clean.npz OR a prepare_sorted() directory.
    meta_path : str or Path
        Path to insitu_land_station_meta.npz.
    """

    def __init__(self, obs_path: str | Path, meta_path: str | Path) -> None:
        self.obs_path  = Path(obs_path)
        self.meta_path = Path(meta_path)

        meta_raw = np.load(self.meta_path, allow_pickle=True)
        # Station metadata (552 rows) — keep as-is
        self._meta: dict[str, np.ndarray] = {k: meta_raw[k] for k in meta_raw.files}

        if self.obs_path.is_dir() and (self.obs_path / _SORTED_MANIFEST).exists():
            self._obs, self._obs_station_int, self._unique_ids = \
                self._load_sorted_dir(self.obs_path)
        else:
            self._obs, self._obs_station_int, self._unique_ids = \
                self._load_npz_and_sort(self.obs_path)

        self._timestamps: np.ndarray = self._obs['report_timestamp']   # sorted int64

    # ------------------------------------------------------------------
    # Loaders (slow npz / fast pre-sorted mmap) + offline converter
    # ------------------------------------------------------------------

    @staticmethod
    def _load_npz_and_sort(obs_path: Path):
        """Load the original .npz and sort every column by report_timestamp.

        The argsort over 74 M rows + full re-index is the slow startup path;
        prepare_sorted() pays it once offline so future loads can mmap.
        """
        obs_raw = np.load(obs_path, allow_pickle=True)
        # Station-ID integer index (string -> int) before sorting.
        unique_ids, inv = np.unique(obs_raw['primary_station_id'],
                                    return_inverse=True)
        sort_order = np.argsort(obs_raw['report_timestamp'])
        obs = {k: obs_raw[k][sort_order] for k in obs_raw.files}
        return obs, inv[sort_order].astype(np.int32), unique_ids

    @staticmethod
    def _load_sorted_dir(d: Path):
        """Memory-map a prepare_sorted() directory — no load/sort, O(1) startup."""
        manifest = json.loads((d / _SORTED_MANIFEST).read_text())
        if manifest.get('format') != _SORTED_FORMAT:
            raise ValueError(
                f"{d} is not a {_SORTED_FORMAT} obs directory "
                f"(found format={manifest.get('format')!r})."
            )
        obs = {k: np.load(d / f'{k}.npy', mmap_mode='r')
               for k in manifest['columns']}
        station_int = np.load(d / '_obs_station_int.npy', mmap_mode='r')
        unique_ids  = np.load(d / '_unique_ids.npy')         # small — keep in RAM
        return obs, station_int, unique_ids

    @classmethod
    def prepare_sorted(cls, obs_npz_path: str | Path,
                       out_dir: str | Path) -> Path:
        """Convert insitu_land_clean.npz → a sorted, memory-mappable directory.

        Run ONCE (it pays the load + argsort the slow path does on every
        construction), then set the config's insitu_obs_path to ``out_dir`` so
        every subsequent run mmaps the pre-sorted columns instead. Object/string
        columns are cast to fixed-width unicode so they too can be mmapped.

        Returns the output directory path.
        """
        obs_raw = np.load(Path(obs_npz_path), allow_pickle=True)
        cols    = list(obs_raw.files)
        unique_ids, inv = np.unique(obs_raw['primary_station_id'],
                                    return_inverse=True)
        order   = np.argsort(obs_raw['report_timestamp'])

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for k in cols:
            arr = np.asarray(obs_raw[k])[order]
            if arr.dtype == object:           # object strings aren't mmappable
                arr = arr.astype(np.str_)
            np.save(out / f'{k}.npy', arr)
        np.save(out / '_obs_station_int.npy', inv[order].astype(np.int32))
        np.save(out / '_unique_ids.npy', np.asarray(unique_ids).astype(np.str_))
        (out / _SORTED_MANIFEST).write_text(json.dumps(
            {'format': _SORTED_FORMAT, 'columns': cols, 'n_rows': int(order.size)}
        ))
        return out

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

        # All stations kept (e.g. reliability_levels lists every level): skip the
        # 74 M-row boolean index + copy entirely — return self. This keeps the
        # mmap fast path lazy when no reliability filtering actually applies.
        if valid_ints.size == self._unique_ids.size:
            return self

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


# ---------------------------------------------------------------------------
# CLI — one-time conversion to the fast pre-sorted/mmap layout
# ---------------------------------------------------------------------------
# Usage (run once; then set data.insitu_obs_path to OUT_DIR):
#   PYTHONPATH=jrt python -m experiments.tc_perceiver_io.data.sources.insitu_land \
#       E:/sparse_obs/insitu-land/insitu_land_clean.npz \
#       E:/sparse_obs/insitu-land/insitu_land_clean_sorted

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Convert insitu_land_clean.npz to the sorted, memory-mappable "
                    "directory layout (skips the per-run argsort; near-instant load).")
    parser.add_argument('obs_npz', help="Path to insitu_land_clean.npz")
    parser.add_argument('out_dir', help="Output directory for the sorted columns")
    args = parser.parse_args()

    t0 = time.time()
    print(f"Converting {args.obs_npz} → {args.out_dir} ...", flush=True)
    out = InsituLandDataset.prepare_sorted(args.obs_npz, args.out_dir)
    print(f"Done in {time.time() - t0:.0f}s. Point data.insitu_obs_path at:\n  {out}")
