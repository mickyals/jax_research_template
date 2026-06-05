"""
experiments/sparse_obs_cross_attn/dataset.py

TCDataset: pairs IBTrACSDataset + InsituLandDataset into per-sample dicts.

Each sample has a query position (storm center for TC samples, random
point in the domain for background samples) and the set of station
observations within radius_km at the query time.

Sample dict
-----------
    query_coords   : np.float32 (2,)      encoded query position (see below)
    station_obs    : np.float32 (N, F)    obs values, normalised, NaN→0
    station_coords : np.float32 (N, 2)    encoded station positions (see below)
    station_mask   : np.bool_   (N,)      True = real station, False = padding
    obs_mask       : np.bool_   (N, F)    True = measurement was present
    label          : np.int32             0 = no storm, 1–10 = SSHS+offset
    n_stations     : np.int32

Location encoding modes
-----------------------
    'unit_circle'
        station_coords : [normalised_distance, bearing_radians]
            normalised_distance = haversine_km / radius_km  in [0, 1]
            bearing_radians     = bearing from storm to station in [0, 2π)
        query_coords   : [0.0, 0.0]  — sentinel; model replaces with a
                         learned centre token.

    'domain'
        station_coords : [norm_lat_rad, norm_lon_rad]
            lat/lon normalised to FOV bounds → [-1, 1] → scaled by π/2
        query_coords   : same encoding applied to the storm/query position.

N = max_stations (zero-padded).  F = len(obs_vars).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from experiments.sparse_obs_cross_attn.ibtracs import (
    IBTrACSDataset, SSHS_TO_CLASS,
)
from experiments.sparse_obs_cross_attn.insitu_land import (
    InsituLandDataset, DEFAULT_OBS_VARS,
)
from utils.geoscience.geodesic import vincenty_np

if TYPE_CHECKING:
    import pandas as pd

_HALF_PI = np.float32(np.pi / 2.0)


class TCDataset:
    """Assembles per-sample dicts for TC detection + intensity classification.

    Parameters
    ----------
    ibtracs : IBTrACSDataset
        An already-split IBTrACS dataset (one split only).
    insitu : InsituLandDataset
        An already-split InsituLand dataset.
    radius_km : float
        Spatial radius for station search.
    time_window_hours : float
        Half-width of the temporal search window (±).
    max_stations : int
        Fixed sample size for batching — real stations are padded or subsampled
        to exactly this count.
    min_stations : int
        Samples with fewer matching stations return None.
    obs_vars : list[str] or None
        Observation variables to include. None → DEFAULT_OBS_VARS.
    background_timestamps : np.ndarray or None
        Pool of int64 Unix-ns timestamps for background sample draws.
        Required before calling get_background_sample.
    location_encoding : {'unit_circle', 'domain'}
        Coordinate representation fed to the model. Default 'unit_circle'.
    fov_lat : (lat_min, lat_max)
        Field-of-view latitude bounds in degrees. Required for 'domain' encoding.
    fov_lon : (lon_min, lon_max)
        Field-of-view longitude bounds in degrees. Required for 'domain' encoding.
    obs_bounds : dict[str, (min, max)] or None
        Per-variable physical bounds for normalisation of station_obs.
        Keys must match obs_vars. If None, obs values are not normalised.
        For 'standardise', values are interpreted as (mean, std).
    obs_normalisation : {'minmax_01', 'minmax_11', 'standardise'}
        Normalisation applied to station_obs when obs_bounds is provided.
        'minmax_01'  : scale to [0, 1]  using (min, max) bounds.
        'minmax_11'  : scale to [-1, 1] using (min, max) bounds.
        'standardise': z-score using (mean, std) bounds.
        Default 'minmax_01'.
    """

    def __init__(
        self,
        ibtracs:               IBTrACSDataset,
        insitu:                InsituLandDataset,
        radius_km:             float = 500.0,
        time_window_hours:     float = 0.1,
        max_stations:          int   = 64,
        min_stations:          int   = 1,
        obs_vars:              Optional[list[str]] = None,
        background_timestamps: Optional[np.ndarray] = None,
        location_encoding:     str = 'unit_circle',
        fov_lat:               tuple[float, float] = (0.0, 30.0),
        fov_lon:               tuple[float, float] = (-100.0, -45.0),
        obs_bounds:            Optional[dict[str, tuple[float, float]]] = None,
        obs_normalisation:     str = 'minmax_01',
    ) -> None:

        if location_encoding not in ('unit_circle', 'domain'):
            raise ValueError(
                f"location_encoding must be 'unit_circle' or 'domain', "
                f"got '{location_encoding}'."
            )
        if obs_normalisation not in ('minmax_01', 'minmax_11', 'standardise'):
            raise ValueError(
                f"obs_normalisation must be 'minmax_01', 'minmax_11', or 'standardise', "
                f"got '{obs_normalisation}'."
            )

        self.ibtracs               = ibtracs
        self.insitu                = insitu
        self.radius_km             = float(radius_km)
        self.window_ns             = int(time_window_hours * 3600 * 1e9)
        self.max_stations          = int(max_stations)
        self.min_stations          = int(min_stations)
        self.obs_vars              = list(obs_vars) if obs_vars is not None else list(DEFAULT_OBS_VARS)
        self.background_timestamps = background_timestamps
        self.location_encoding     = location_encoding
        self.fov_lat               = tuple(fov_lat)
        self.fov_lon               = tuple(fov_lon)
        self.obs_bounds            = obs_bounds
        self.obs_normalisation     = obs_normalisation

        # Pre-compute per-variable normalisation arrays for fast reuse.
        # _obs_lo/_obs_hi store (min, max) for minmax modes and (mean, std)
        # for 'standardise' — the names reflect the minmax convention.
        if obs_bounds is not None:
            lo = np.array([obs_bounds[v][0] for v in self.obs_vars], dtype=np.float32)
            hi = np.array([obs_bounds[v][1] for v in self.obs_vars], dtype=np.float32)
            self._obs_lo = lo
            self._obs_hi = hi
        else:
            self._obs_lo = None
            self._obs_hi = None

        # Cache frequently accessed arrays for fast sample assembly
        self._lat  = ibtracs['LAT'].astype(np.float32)
        self._lon  = ibtracs['LON'].astype(np.float32)
        self._time = ibtracs['ISO_TIME']                    # int64 Unix-ns
        self._sshs = ibtracs['USA_SSHS'].astype(np.float32)

    def __len__(self) -> int:
        return len(self.ibtracs)

    def __repr__(self) -> str:
        return (
            f"TCDataset("
            f"n_tc={len(self)}, "
            f"obs_vars={self.obs_vars}, "
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
        """Assemble one TC sample from IBTrACS row idx.

        Parameters
        ----------
        idx : int
            Row index into the IBTrACS split.
        rng : np.random.Generator, optional
            When more stations match than max_stations, used for random
            subsampling (train augmentation). If None the closest stations
            by distance are kept.

        Returns
        -------
        dict | None  — None when matching stations < min_stations or SSHS
        value is not in SSHS_TO_CLASS.
        """
        lat      = float(self._lat[idx])
        lon      = float(self._lon[idx])
        ts       = int(self._time[idx])
        sshs_raw = float(self._sshs[idx])

        sshs = int(round(sshs_raw))
        if sshs not in SSHS_TO_CLASS:
            return None
        label = SSHS_TO_CLASS[sshs]

        df = self.insitu.get_obs_near(
            query_lat=lat,
            query_lon=lon,
            timestamp_ns=ts,
            radius_km=self.radius_km,
            window_ns=self.window_ns,
            obs_vars=self.obs_vars,
        )
        if len(df) < self.min_stations:
            return None

        return self._build_sample(df, lat, lon, label, rng)

    # ------------------------------------------------------------------
    # Background sample
    # ------------------------------------------------------------------

    def get_background_sample(
        self,
        rng:     np.random.Generator,
        fov_lat: Optional[tuple[float, float]] = None,
        fov_lon: Optional[tuple[float, float]] = None,
    ) -> Optional[dict]:
        """Assemble one background (no-storm) sample.

        A random timestamp is drawn from the background pool and a random
        query position from within the field-of-view bounds.

        Parameters
        ----------
        rng : np.random.Generator
        fov_lat : (lat_min, lat_max) or None
            Defaults to self.fov_lat.
        fov_lon : (lon_min, lon_max) or None
            Defaults to self.fov_lon.

        Returns
        -------
        dict | None — None when no stations are found.
        """
        if self.background_timestamps is None or len(self.background_timestamps) == 0:
            raise RuntimeError(
                "background_timestamps pool is empty. "
                "Build it via TCDataModule.setup()."
            )

        fov_lat = fov_lat if fov_lat is not None else self.fov_lat
        fov_lon = fov_lon if fov_lon is not None else self.fov_lon
        lat = float(rng.uniform(fov_lat[0], fov_lat[1]))
        lon = float(rng.uniform(fov_lon[0], fov_lon[1]))
        ts  = int(rng.choice(self.background_timestamps))

        df = self.insitu.get_obs_near(
            query_lat=lat,
            query_lon=lon,
            timestamp_ns=ts,
            radius_km=self.radius_km,
            window_ns=self.window_ns,
            obs_vars=self.obs_vars,
        )
        if len(df) < self.min_stations:
            return None

        return self._build_sample(df, lat, lon, 0, rng)

    # ------------------------------------------------------------------
    # Shared sample builder
    # ------------------------------------------------------------------

    def _build_sample(
        self,
        df:        'pd.DataFrame',
        query_lat: float,
        query_lon: float,
        label:     int,
        rng:       Optional[np.random.Generator],
    ) -> dict:
        n_available = len(df)
        F = len(self.obs_vars)

        # Subsample or trim to max_stations
        if n_available > self.max_stations:
            if rng is not None:
                chosen = rng.choice(n_available, self.max_stations, replace=False)
                df = df.iloc[chosen]
            else:
                df = df.iloc[:self.max_stations]
            n_real = self.max_stations
        else:
            n_real = n_available

        obs_vals = df[self.obs_vars].to_numpy(dtype=np.float32)   # (n_real, F)
        obs_mask = np.isfinite(obs_vals)                           # (n_real, F)
        obs_safe = np.where(obs_mask, obs_vals, 0.0)

        # Normalise obs values using pre-computed bounds
        if self._obs_lo is not None:
            if self.obs_normalisation == 'minmax_01':
                span = self._obs_hi - self._obs_lo
                obs_safe = (obs_safe - self._obs_lo) / (span + 1e-12)
            elif self.obs_normalisation == 'minmax_11':
                span = self._obs_hi - self._obs_lo
                obs_safe = (obs_safe - self._obs_lo) / (span + 1e-12) * 2.0 - 1.0
            else:  # standardise: bounds are (mean, std)
                obs_safe = (obs_safe - self._obs_lo) / (self._obs_hi + 1e-8)
            # re-zero positions that were missing before normalisation
            obs_safe = obs_safe * obs_mask

        raw_lats = df['latitude'].to_numpy(dtype=np.float32)    # (n_real,)
        raw_lons = df['longitude'].to_numpy(dtype=np.float32)   # (n_real,)

        # Encode station coordinates and query coordinate
        if self.location_encoding == 'unit_circle':
            dist_km = df['distance_km'].to_numpy(dtype=np.float32)
            norm_dist = np.clip(dist_km / self.radius_km, 0.0, 1.0)

            _, bearing_deg, _, _ = vincenty_np(
                np.full(n_real, query_lat),
                np.full(n_real, query_lon),
                raw_lats.astype(np.float64),
                raw_lons.astype(np.float64),
            )
            bearing_rad = np.radians(bearing_deg).astype(np.float32)
            # NaN from near-coincident points → bearing 0.0
            bearing_rad = np.where(np.isfinite(bearing_rad), bearing_rad, 0.0)

            encoded_coords = np.stack([norm_dist, bearing_rad], axis=-1)  # (n_real, 2)
            query_coords   = np.zeros(2, dtype=np.float32)  # sentinel; model uses learned token

        else:  # domain
            lat_min, lat_max = self.fov_lat
            lon_min, lon_max = self.fov_lon
            lat_span = lat_max - lat_min
            lon_span = lon_max - lon_min

            norm_lat = ((raw_lats - lat_min) / (lat_span + 1e-12) * 2.0 - 1.0) * _HALF_PI
            norm_lon = ((raw_lons - lon_min) / (lon_span + 1e-12) * 2.0 - 1.0) * _HALF_PI
            encoded_coords = np.stack([norm_lat, norm_lon], axis=-1)  # (n_real, 2)

            q_norm_lat = ((query_lat - lat_min) / (lat_span + 1e-12) * 2.0 - 1.0) * float(_HALF_PI)
            q_norm_lon = ((query_lon - lon_min) / (lon_span + 1e-12) * 2.0 - 1.0) * float(_HALF_PI)
            query_coords = np.array([q_norm_lat, q_norm_lon], dtype=np.float32)

        # Allocate padded arrays
        station_obs    = np.zeros((self.max_stations, F), dtype=np.float32)
        station_coords = np.zeros((self.max_stations, 2), dtype=np.float32)
        station_mask   = np.zeros((self.max_stations,),   dtype=bool)
        obs_mask_pad   = np.zeros((self.max_stations, F), dtype=bool)

        station_obs[:n_real]    = obs_safe
        station_coords[:n_real] = encoded_coords
        station_mask[:n_real]   = True
        obs_mask_pad[:n_real]   = obs_mask

        return {
            'query_coords':   query_coords,
            'station_obs':    station_obs,
            'station_coords': station_coords,
            'station_mask':   station_mask,
            'obs_mask':       obs_mask_pad,
            'label':          np.int32(label),
            'n_stations':     np.int32(n_real),
        }
