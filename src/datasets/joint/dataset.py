"""
datasets/joint/dataset.py

JointTCDataset: assembles per-sample dicts for the TC intensity
classification experiment.

Each sample pairs one storm-centre query position with the set of
land-surface station observations within radius_km at the storm's time.
Background samples substitute a random query position at a non-TC timestamp.

Class definitions (SSHS → ordinal class index):

    CLASS_NO_STORM    = 0   background — no active TC
    CLASS_DISTURBANCE = 1   USA_SSHS == -3  (DB, LO, WV, MD)
    CLASS_SUBTROPICAL = 2   USA_SSHS == -2  (SS, SD)
    CLASS_TD          = 3   USA_SSHS == -1
    CLASS_TS          = 4   USA_SSHS ==  0
    CLASS_CAT1        = 5   USA_SSHS ==  1
    CLASS_CAT2        = 6   USA_SSHS ==  2
    CLASS_CAT3        = 7   USA_SSHS ==  3
    CLASS_CAT4        = 8   USA_SSHS ==  4
    CLASS_CAT5        = 9   USA_SSHS ==  5

Per-sample output dict
----------------------
    query_coords : np.ndarray  (coord_dim,)   unit-sphere [x,y,z] of query lat/lon
    station_obs  : np.ndarray  (max_stations, OBS_DIM)  zero-padded, NaN→0
    station_mask : np.ndarray  (max_stations,) bool  True = real station
    obs_mask     : np.ndarray  (max_stations, OBS_DIM) bool  True = valid obs value
    label        : np.int32    ordinal class 0..9
    n_stations   : np.int32    number of real stations before padding
    sid          : str         IBTrACS SID or 'BACKGROUND'
    iso_time     : np.int64    Unix nanoseconds
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from datasets.insitu_land.dataset import InsituLandDataset
from datasets.ibtracs.dataset import IBTrACSDataset
from datasets.position_encoding import encode_unit_sphere

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Class constants
# ---------------------------------------------------------------------------

N_CLASSES = 10
CLASS_NO_STORM = 0

SSHS_TO_CLASS: dict[int, int] = {
    -3: 1, -2: 2, -1: 3, 0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9,
}

# Physical obs columns passed to the model (subset of InsituLandDataset.OBS_COLS)
_PHYSICAL_OBS_COLS: list[str] = [
    'air_pressure_at_sea_level',
    'air_temperature',
    'dew_point_temperature',
    'wind_speed',
]
_N_PHYSICAL = len(_PHYSICAL_OBS_COLS)  # 4

# Full station feature vector:
#   [physical_obs x 4,  bearing_sin,  bearing_cos,  log_dist_norm]
N_PHYSICAL_OBS = _N_PHYSICAL      # 4  — physical measurement columns
N_GEO_FEATURES = 3                # bearing_sin, bearing_cos, log_dist_norm
OBS_DIM = N_PHYSICAL_OBS + N_GEO_FEATURES  # = 7  — total station feature vector width

# Minimum time window enforced regardless of time_window_hours.
# Floors query timestamps to minute granularity before the window search
# so that two records nominally at "06:00" but differing by sub-second
# nanosecond noise are treated as simultaneous.
_MINUTE_NS: int = 60 * int(1e9)


# ---------------------------------------------------------------------------
# JointTCDataset
# ---------------------------------------------------------------------------

class JointTCDataset:
    """Assembles per-sample dicts for TC intensity classification.

    Not a NpzDataset subclass — samples are multi-modal dicts, not rows.

    Parameters
    ----------
    ibtracs : IBTrACSDataset
        Already-split, single-storm IBTrACS observations.
    insitu : InsituLandDataset
        Already-split in-situ land observations.
    radius_km : float
        Spatial search radius around the query position.
    time_window_hours : float
        Half-width of the observation time window (±).
    max_stations : int
        Pad or subsample context set to exactly this size for batching.
    min_stations : int
        Samples with fewer matching stations return None (skipped).
    background_timestamps : np.ndarray | None
        Pool of int64 Unix-ns timestamps for background sample draws.
        Required before calling get_background_sample.
    """

    def __init__(
        self,
        ibtracs:               IBTrACSDataset,
        insitu:                InsituLandDataset,
        radius_km:             float = 500.0,
        time_window_hours:     float = 3.0,
        max_stations:          int   = 64,
        min_stations:          int   = 1,
        background_timestamps: Optional[np.ndarray] = None,
    ) -> None:
        self.ibtracs    = ibtracs
        self.insitu     = insitu
        self.radius_km  = float(radius_km)
        # Enforce a minimum 1-minute window so that time_window_hours=0.0
        # still matches observations at the same hour:minute regardless of
        # sub-second nanosecond differences between data sources.
        self.window_ns  = max(int(time_window_hours * 3600 * 1e9), _MINUTE_NS)
        self.max_stations = max_stations
        self.min_stations = min_stations
        self.background_timestamps = background_timestamps

        # Cache frequently accessed arrays
        self._lat  = ibtracs['LAT'].astype(np.float32)
        self._lon  = ibtracs['LON'].astype(np.float32)
        self._time = ibtracs['ISO_TIME']      # int64 Unix ns
        self._sshs = ibtracs['USA_SSHS'].astype(np.float32)
        self._sid  = ibtracs['SID']

    def __len__(self) -> int:
        return len(self.ibtracs)

    def __repr__(self) -> str:
        return (
            f"JointTCDataset("
            f"n_tc={len(self)}, "
            f"n_insitu_stations={self.insitu.n_stations}, "
            f"radius_km={self.radius_km})"
        )

    # ------------------------------------------------------------------
    # TC sample
    # ------------------------------------------------------------------

    def get_tc_sample(
        self,
        idx: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[dict]:
        """Assemble one TC training sample.

        Parameters
        ----------
        idx : int
            Row index into the IBTrACS split.
        rng : np.random.Generator, optional
            When more than max_stations stations are available, used to
            subsample randomly (train augmentation).  If None, the
            closest max_stations by distance are used.

        Returns
        -------
        dict | None
            None when the number of matching stations < min_stations.
        """
        lat      = float(self._lat[idx])
        lon      = float(self._lon[idx])
        ts       = int(self._time[idx])
        sid      = str(self._sid[idx])
        sshs_raw = float(self._sshs[idx])

        # NaN SSHS is common in extratropical rows (ibtracs_full.npz).
        # int(nan) raises ValueError, so guard before rounding.
        if not np.isfinite(sshs_raw):
            return None
        sshs = int(round(sshs_raw))
        if sshs not in SSHS_TO_CLASS:
            return None
        label = SSHS_TO_CLASS[sshs]

        # Floor to minute granularity so sub-second differences between
        # IBTrACS and InsituLand timestamps don't cause missed matches.
        ts_query = (ts // _MINUTE_NS) * _MINUTE_NS

        df = self.insitu.get_stations_at_time(
            timestamp_ns=ts_query,
            radius_km=self.radius_km,
            storm_lat=lat,
            storm_lon=lon,
            window_ns=self.window_ns,
        )
        if len(df) < self.min_stations:
            return None

        return self._build_sample(df, lat, lon, label, sid, ts, rng)

    # ------------------------------------------------------------------
    # Background sample
    # ------------------------------------------------------------------

    def get_background_sample(
        self,
        rng:     np.random.Generator,
        fov_lat: tuple[float, float] = (0.0, 30.0),
        fov_lon: tuple[float, float] = (-100.0, -45.0),
    ) -> Optional[dict]:
        """Assemble one background (no-storm) training sample.

        Draws a random (lat, lon) uniformly within the field of view and
        a random timestamp from the pre-built background pool.

        Parameters
        ----------
        rng : np.random.Generator
        fov_lat : (lat_min, lat_max) in degrees
        fov_lon : (lon_min, lon_max) in degrees

        Returns
        -------
        dict | None
            None when < min_stations match or the pool is empty.
        """
        if self.background_timestamps is None or len(self.background_timestamps) == 0:
            raise RuntimeError(
                "background_timestamps pool is empty or not set. "
                "Build it via JointDataModule.setup()."
            )

        lat = float(rng.uniform(fov_lat[0], fov_lat[1]))
        lon = float(rng.uniform(fov_lon[0], fov_lon[1]))
        ts  = int(rng.choice(self.background_timestamps))
        ts_query = (ts // _MINUTE_NS) * _MINUTE_NS

        df = self.insitu.get_stations_at_time(
            timestamp_ns=ts_query,
            radius_km=self.radius_km,
            storm_lat=lat,
            storm_lon=lon,
            window_ns=self.window_ns,
        )
        if len(df) < self.min_stations:
            return None

        return self._build_sample(df, lat, lon, CLASS_NO_STORM, 'BACKGROUND', ts, rng)

    # ------------------------------------------------------------------
    # Shared sample builder
    # ------------------------------------------------------------------

    def _build_sample(
        self,
        df:           'pd.DataFrame',
        query_lat:    float,
        query_lon:    float,
        label:        int,
        sid:          str,
        timestamp_ns: int,
        rng:          Optional[np.random.Generator],
    ) -> dict:
        n_available = len(df)

        # Subsample or trim to max_stations
        if n_available > self.max_stations:
            if rng is not None:
                idx = rng.choice(n_available, self.max_stations, replace=False)
                df = df.iloc[idx]
            else:
                # Deterministic: closest stations (DataFrame is sorted by distance)
                df = df.iloc[:self.max_stations]
            n_real = self.max_stations
        else:
            n_real = n_available

        # Physical obs: (n_real, 4), NaN where not reported
        phys = df[_PHYSICAL_OBS_COLS].to_numpy(dtype=np.float32)

        # Geometric features computed from Vincenty output in get_stations_at_time
        dist_km = df['distance_km'].to_numpy(dtype=np.float32)
        fwd_az  = df['forward_azimuth_deg'].to_numpy(dtype=np.float32)
        az_rad  = np.radians(fwd_az)
        log_dist_norm = np.log1p(dist_km) / np.log1p(self.radius_km)  # [0, 1]

        geo = np.stack(
            [np.sin(az_rad), np.cos(az_rad), log_dist_norm], axis=1
        )  # (n_real, 3)

        # Full feature matrix: (n_real, 7)
        obs = np.concatenate([phys, geo], axis=1)

        # Obs validity mask: physical vars may be NaN, geo is always valid
        phys_mask = np.isfinite(phys)                               # (n_real, 4)
        geo_mask  = np.ones((n_real, 3), dtype=bool)
        obs_mask  = np.concatenate([phys_mask, geo_mask], axis=1)  # (n_real, 7)

        # Replace NaN with 0 (masked out by obs_mask in model)
        obs_safe = np.where(np.isnan(obs), 0.0, obs)

        # Pad arrays to max_stations (padding rows are zero, mask=False)
        station_obs  = np.zeros((self.max_stations, OBS_DIM), dtype=np.float32)
        station_mask = np.zeros((self.max_stations,),          dtype=bool)
        obs_mask_pad = np.zeros((self.max_stations, OBS_DIM),  dtype=bool)

        station_obs[:n_real]  = obs_safe
        station_mask[:n_real] = True
        obs_mask_pad[:n_real] = obs_mask

        # Query positional encoding: unit sphere (3D) for consistency with
        # the GaussianFourierEmbedding(input_dim=3) encoder in the model
        query_coords = encode_unit_sphere(
            np.array([query_lat], dtype=np.float32),
            np.array([query_lon], dtype=np.float32),
        )[0]  # (3,)

        return {
            'query_coords': query_coords,           # (3,)          float32
            'station_obs':  station_obs,            # (max_stations, 7)  float32
            'station_mask': station_mask,           # (max_stations,)    bool
            'obs_mask':     obs_mask_pad,           # (max_stations, 7)  bool
            'label':        np.int32(label),
            'n_stations':   np.int32(n_real),
            'sid':          sid,
            'iso_time':     np.int64(timestamp_ns),
        }
