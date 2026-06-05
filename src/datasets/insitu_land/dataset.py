"""
insitu_land/dataset.py

In-situ land surface observations for the Caribbean / Gulf of Mexico domain
(LAT 0–30 N, LON 100–45 W, 2005–2025), pre-processed from the Copernicus
C3S Global Land Surface Atmospheric Variables sub-daily dataset (r8.1).

Two npz files
-------------
obs_path   -- 74.7 M rows, one row per station per timestamp.
meta_path  -- 552 rows, one per unique station (always fully loaded).

Time index
----------
At init, observations are sorted by ``report_timestamp`` and a sorted index
array is kept alongside the sorted timestamps.  Time-range queries (the
primary access pattern at dataloader time) then use ``np.searchsorted``
for O(log n) lookup instead of a full array scan.

Column definitions
------------------
OBS_COLS          physical measurement columns (may contain NaN)
META_COLS         station identifier and metadata columns

Station reliability
-------------------
``station_reliability`` is one of the five levels in RELIABILITY_LEVELS.
Use ``filter_reliability`` to keep only stations meeting a minimum quality
threshold before constructing a training split.

Split convention
----------------
Splits follow the same year ranges as IBTrACS:
    train  2005–2020
    val    2021–2022
    test   2023–2025
Year is extracted from ``report_timestamp`` (Unix nanoseconds → UTC year).

Primary use at dataloader time
-------------------------------
``get_stations_at_time(timestamp_ns, window_ns, radius_km, storm_lat, storm_lon)``
returns a DataFrame of all station observations within a time window AND
spatial radius of a storm centre.  Each row is annotated with geodesic
distance and forward azimuth to the storm centre (via ``vincenty_np``).

Example
-------
>>> ds = InsituLandDataset('data/insitu_land_clean.npz',
...                         'data/insitu_land_station_meta.npz')
>>> ds = ds.filter_reliability(['always_active', 'mostly_active'])
>>> df = ds.get_stations_at_time(
...     timestamp_ns  = ibtracs_row['report_timestamp'],
...     window_ns     = 3 * 3600 * int(1e9),   # ±3 h
...     radius_km     = 500.0,
...     storm_lat     = ibtracs_row['LAT'],
...     storm_lon     = ibtracs_row['LON'],
... )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from datasets.base import NpzDataset
from utils.geoscience.geodesic import vincenty_np


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

OBS_COLS: list[str] = [
    "air_pressure_at_sea_level",
    "air_pressure",
    "air_temperature",
    "dew_point_temperature",
    "wind_speed",
    "wind_from_direction",
]

META_COLS: list[str] = [
    "primary_station_id",
    "station_name",
    "latitude",
    "longitude",
    "elevation",
    "report_timestamp",
    "slp_derived",
    "slp_unreliable",
    "station_reliability",
]

RELIABILITY_LEVELS: list[str] = [
    "always_active",
    "mostly_active",
    "sporadic",
    "sparse",
    "unusable",
]

# Year boundaries — same ranges as IBTRACS_*_SEASONS in schema.py
from datasets.schema import (
    IBTRACS_TRAIN_SEASONS as _TRAIN_YEARS,
    IBTRACS_VAL_SEASONS   as _VAL_YEARS,
    IBTRACS_TEST_SEASONS  as _TEST_YEARS,
)

_NS_PER_YEAR = int(365.25 * 24 * 3600 * 1e9)  # approximate, used for year extraction


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class InsituLandDataset(NpzDataset):
    """
    In-situ land surface observations pre-processed from Copernicus C3S
    Global Land Surface Atmospheric Variables (r8.1), Caribbean/Gulf domain.

    Parameters
    ----------
    obs_path : str or Path
        Path to ``insitu_land_clean.npz``  (74.7 M rows).
    meta_path : str or Path
        Path to ``insitu_land_station_meta.npz``  (552 rows).

    Notes
    -----
    The 74.7 M row obs file is loaded once into memory.  A sorted time
    index is built at init so that ``get_stations_at_time`` runs in
    O(log n) time rather than scanning the full array.

    Filtering (e.g. ``filter_reliability``) rebuilds the time index on the
    filtered subset.  For the typical workflow — filter once, then query
    many times — this is a one-off cost.
    """

    def __init__(
        self,
        obs_path:  str | Path,
        meta_path: str | Path,
    ) -> None:
        super().__init__(obs_path)

        meta_raw        = np.load(Path(meta_path), allow_pickle=True)
        self._meta_path = Path(meta_path)
        self._meta: dict[str, np.ndarray] = {k: meta_raw[k] for k in meta_raw.files}

        self._build_time_index()

    # ------------------------------------------------------------------
    # Internal: time index
    # ------------------------------------------------------------------

    def _build_time_index(self) -> None:
        """Sort observations by timestamp and keep the sort order for binary search."""
        ts = self._data["report_timestamp"]
        self._sorted_time_idx = np.argsort(ts, kind="stable")
        self._sorted_timestamps = ts[self._sorted_time_idx]

    # ------------------------------------------------------------------
    # Factory overrides to carry metadata through masking
    # ------------------------------------------------------------------

    @classmethod
    def _from_data(
        cls,
        data:      dict[str, np.ndarray],
        npz_path:  Path,
        **extra_attrs,
    ) -> "InsituLandDataset":
        obj            = cls.__new__(cls)
        obj.npz_path   = npz_path
        obj._data      = data
        obj._n         = len(next(iter(data.values()))) if data else 0
        obj._meta      = extra_attrs.get("_meta", {})
        obj._meta_path = extra_attrs.get("_meta_path", npz_path)
        obj._build_time_index()
        return obj

    def _mask_to_dataset(self, mask: np.ndarray) -> "InsituLandDataset":
        return self._from_data(
            data      = {k: v[mask] for k, v in self._data.items()},
            npz_path  = self.npz_path,
            _meta     = self._meta,
            _meta_path = self._meta_path,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InsituLandDataset("
            f"n={self._n:,}, "
            f"stations={self.n_stations}, "
            f"path='{self.npz_path.name}')"
        )

    @property
    def n_stations(self) -> int:
        """Number of unique stations in this subset."""
        return int(np.unique(self._data["primary_station_id"]).shape[0])

    @property
    def timestamps(self) -> np.ndarray:
        """``report_timestamp`` column as int64 Unix nanoseconds."""
        return self._data["report_timestamp"]

    @property
    def station_meta(self) -> pd.DataFrame:
        """Station metadata as a DataFrame (552 rows, always fully loaded)."""
        return pd.DataFrame(self._meta)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def filter_reliability(
        self,
        levels: list[str],
    ) -> "InsituLandDataset":
        """Keep only rows whose station_reliability is in levels.

        Parameters
        ----------
        levels : list[str]
            Subset of RELIABILITY_LEVELS, e.g. ['always_active', 'mostly_active'].

        Returns
        -------
        InsituLandDataset

        Example
        -------
        >>> ds = ds.filter_reliability(['always_active', 'mostly_active'])
        """
        return self._mask_to_dataset(
            np.isin(self._data["station_reliability"], levels)
        )

    def filter_time_range(
        self,
        start_ns: int,
        end_ns:   int,
    ) -> "InsituLandDataset":
        """Keep rows with report_timestamp in [start_ns, end_ns] (inclusive).

        Parameters
        ----------
        start_ns, end_ns : int
            Unix nanoseconds.  Use ``pd.Timestamp(...).value`` to convert.

        Returns
        -------
        InsituLandDataset
        """
        ts   = self._data["report_timestamp"]
        mask = (ts >= start_ns) & (ts <= end_ns)
        return self._mask_to_dataset(mask)

    def filter_bbox(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> "InsituLandDataset":
        """Keep rows within a lat/lon bounding box.

        Parameters
        ----------
        lat_min, lat_max : float  Latitude bounds in decimal degrees.
        lon_min, lon_max : float  Longitude bounds in decimal degrees.

        Returns
        -------
        InsituLandDataset
        """
        lat  = self._data["latitude"]
        lon  = self._data["longitude"]
        mask = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min) & (lon <= lon_max)
        )
        return self._mask_to_dataset(mask)

    def filter_radius(
        self,
        center_lat: float,
        center_lon: float,
        radius_km:  float,
    ) -> "InsituLandDataset":
        """Keep rows within ``radius_km`` of a given point (Vincenty distance).

        Parameters
        ----------
        center_lat, center_lon : float  Centre in decimal degrees.
        radius_km : float               Search radius in kilometres.

        Returns
        -------
        InsituLandDataset
        """
        dist_km, _, _, _ = vincenty_np(
            self._data["latitude"],
            self._data["longitude"],
            center_lat,
            center_lon,
        )
        return self._mask_to_dataset(dist_km <= radius_km)

    def filter_stations(self, station_ids: list[str]) -> "InsituLandDataset":
        """Keep only rows whose primary_station_id is in station_ids."""
        return self._mask_to_dataset(
            np.isin(self._data["primary_station_id"], station_ids)
        )

    # ------------------------------------------------------------------
    # Predefined train / val / test splits
    # ------------------------------------------------------------------

    def split(self, which: str) -> "InsituLandDataset":
        """Return a predefined temporal split.

        Parameters
        ----------
        which : str
            'train'  2005–2020
            'val'    2021–2022
            'test'   2023–2025

        Returns
        -------
        InsituLandDataset

        Raises
        ------
        ValueError
            If ``which`` is not one of the three accepted values.
        """
        year_map = {
            "train": _TRAIN_YEARS,
            "val":   _VAL_YEARS,
            "test":  _TEST_YEARS,
        }
        if which not in year_map:
            raise ValueError(
                f"Unknown split '{which}'. Choose from {list(year_map.keys())}."
            )
        years = self._timestamp_years()
        return self._mask_to_dataset(np.isin(years, year_map[which]))

    def _timestamp_years(self) -> np.ndarray:
        """Extract UTC year from int64 Unix nanoseconds."""
        return (
            pd.to_datetime(self._data["report_timestamp"])
            .year.to_numpy(dtype=np.int32)
        )

    # ------------------------------------------------------------------
    # Core query: stations near a storm at a given time
    # ------------------------------------------------------------------

    def get_stations_at_time(
        self,
        timestamp_ns: int,
        radius_km:    float,
        storm_lat:    float,
        storm_lon:    float,
        window_ns:    int = int(3 * 3600 * 1e9),
    ) -> pd.DataFrame:
        """Return all station observations near a storm centre at a given time.

        This is the primary method called at dataloader time.  For each
        storm observation in IBTrACS, call this with the storm's timestamp,
        position, and a search radius to retrieve the surrounding land
        surface observations.

        The time query uses a pre-built sorted index for O(log n) lookup.
        The spatial filter then applies Vincenty distance on the candidate
        rows returned by the time query.

        Parameters
        ----------
        timestamp_ns : int
            Storm observation time as Unix nanoseconds.
            ``pd.Timestamp("2010-09-05 06:00").value`` gives the right format.
        radius_km : float
            Spatial search radius in kilometres.
        storm_lat, storm_lon : float
            Storm centre in decimal degrees.
        window_ns : int
            Half-width of the time window in nanoseconds.
            Default 3 h = ``3 * 3600 * int(1e9)``.

        Returns
        -------
        pd.DataFrame
            One row per matching station observation.  Columns are all
            fields in META_COLS + OBS_COLS, plus:
              ``distance_km``         geodesic distance to storm centre
              ``forward_azimuth_deg`` bearing from station toward storm
              ``back_azimuth_deg``    bearing from storm back to station

            Returns an empty DataFrame when no observations match.

        Example
        -------
        >>> ts  = pd.Timestamp("2010-09-05 06:00").value
        >>> df  = ds.get_stations_at_time(ts, radius_km=500.0,
        ...                               storm_lat=24.5, storm_lon=-88.0)
        >>> df[['station_name', 'distance_km', 'air_pressure_at_sea_level']]
        """
        t_lo = timestamp_ns - window_ns
        t_hi = timestamp_ns + window_ns

        # O(log n) time-range lookup on sorted index
        lo = int(np.searchsorted(self._sorted_timestamps, t_lo))
        hi = int(np.searchsorted(self._sorted_timestamps, t_hi, side="right"))

        if lo >= hi:
            return pd.DataFrame(columns=META_COLS + OBS_COLS +
                                ["distance_km", "forward_azimuth_deg", "back_azimuth_deg"])

        row_idx = self._sorted_time_idx[lo:hi]

        # Spatial filter via Vincenty (handles NaN distances gracefully)
        lats = self._data["latitude"][row_idx]
        lons = self._data["longitude"][row_idx]
        dist_km, fwd_az, back_az, _ = vincenty_np(lats, lons, storm_lat, storm_lon)

        spatial_mask = dist_km <= radius_km
        if not np.any(spatial_mask):
            return pd.DataFrame(columns=META_COLS + OBS_COLS +
                                ["distance_km", "forward_azimuth_deg", "back_azimuth_deg"])

        row_idx  = row_idx[spatial_mask]
        dist_km  = dist_km[spatial_mask]
        fwd_az   = fwd_az[spatial_mask]
        back_az  = back_az[spatial_mask]

        # Build result DataFrame
        rows: dict[str, np.ndarray] = {}
        for col in META_COLS + OBS_COLS:
            if col in self._data:
                rows[col] = self._data[col][row_idx]
        rows["distance_km"]         = dist_km
        rows["forward_azimuth_deg"] = fwd_az
        rows["back_azimuth_deg"]    = back_az

        df = pd.DataFrame(rows)
        df = df.sort_values("distance_km").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self, cols: Optional[list[str]] = None) -> pd.DataFrame:
        """Convert to a pandas DataFrame.

        ``report_timestamp`` is cast to datetime64[ns] UTC.

        Parameters
        ----------
        cols : list[str], optional
            Subset of columns to include.  Defaults to all columns.

        Returns
        -------
        pd.DataFrame
        """
        cols = cols or list(self._data.keys())
        df   = pd.DataFrame({c: self._data[c] for c in cols if c in self._data})
        if "report_timestamp" in df.columns:
            df["report_timestamp"] = pd.to_datetime(df["report_timestamp"])
        return df

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a concise dataset summary."""
        ts    = self._data["report_timestamp"]
        t_min = pd.to_datetime(int(ts.min()))
        t_max = pd.to_datetime(int(ts.max()))

        print(f"InsituLandDataset — {self.npz_path.name}")
        print(f"  rows        : {self._n:,}")
        print(f"  stations    : {self.n_stations}")
        print(f"  time range  : {t_min.date()} → {t_max.date()}")

        # Reliability breakdown
        rel   = self._data["station_reliability"]
        print(f"\n  station_reliability:")
        for level in RELIABILITY_LEVELS:
            n = int((rel == level).sum())
            if n > 0:
                print(f"    {level:<20} {n:>10,} rows")

        # Variable sparsity
        print(f"\n  observation sparsity (NaN %):")
        for col in OBS_COLS:
            if col not in self._data:
                continue
            arr     = self._data[col].astype(np.float32)
            n_nan   = int(np.isnan(arr).sum())
            pct_nan = 100.0 * n_nan / self._n if self._n > 0 else 0.0
            print(f"    {col:<35} {pct_nan:5.1f}% missing")
