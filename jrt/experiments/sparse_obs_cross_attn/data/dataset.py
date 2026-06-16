"""
experiments/sparse_obs_cross_attn/data/dataset.py

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
    n_stations     : np.int32             stations actually used (≤ N)

Metadata keys (attribution/diagnostics — collated OUTSIDE the model inputs,
never part of batch['X']):
    sid            : str | None           IBTrACS SID; None for background
    iso_time       : np.int64             query timestamp, Unix-ns
    query_lat      : np.float32           raw query latitude, degrees
    query_lon      : np.float32           raw query longitude, degrees
    n_available    : np.int32             post-dedup candidate stations
                                          before trimming to max_stations

Location encoding modes (see data/encoding.py for the encode/decode pairs)
-----------------------
    'unit_circle'
        station_coords : [x, y] local storm-centred map, each in [-1, 1]
            x = norm_dist · sin(bearing)   (east)
            y = norm_dist · cos(bearing)   (north)
            norm_dist = haversine_km / radius_km in [0, 1]
        query_coords   : [0.0, 0.0] — the storm position on the local map;
                         model adds a learned content token for the query.

    'domain'
        station_coords : [norm_lat_rad, norm_lon_rad]
            lat/lon normalised to FOV bounds → [-1, 1] → scaled by π/2
        query_coords   : same encoding applied to the storm/query position.

N = max_stations (zero-padded).  F = len(obs_vars).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from experiments.sparse_obs_cross_attn.data.sources.ibtracs import IBTrACSDataset
from experiments.sparse_obs_cross_attn.data.sources.insitu_land import (
    InsituLandDataset, DEFAULT_OBS_VARS,
)
from experiments.sparse_obs_cross_attn.data.targets import TargetSpec, resolve_target
from experiments.sparse_obs_cross_attn.data.encoding import (
    encode_domain, encode_unit_circle,
)
from utils.geoscience.geodesic import vincenty_np
from utils.geoscience.met_conversions import wind_to_components

if TYPE_CHECKING:
    import pandas as pd

# Derived obs variables: names that may appear in obs_vars but are computed
# by TCDataset from source columns rather than fetched from InsituLandDataset.
# wind_east/wind_north are the (u, v) decomposition of speed + FROM-direction
# (meteorological convention, see utils.geoscience.met_conversions) — kills
# the 0/360 direction seam and shrinks low-speed direction noise by magnitude.
DERIVED_OBS_VARS: dict[str, tuple[str, ...]] = {
    'wind_east':  ('wind_speed', 'wind_from_direction'),
    'wind_north': ('wind_speed', 'wind_from_direction'),
}


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
        Temporal tolerance (±) when matching station reports to the query
        time. Each station contributes at most one report — the one
        nearest in time (see InsituLandDataset.get_obs_near).
    max_stations : int
        Cap on station tokens per sample (for batching). All stations within
        radius are used; samples with fewer than this are zero-padded, and
        samples with more are trimmed to the nearest max_stations by distance.
    min_stations : int
        Samples with fewer matching stations return None.
    obs_vars : list[str] or None
        Observation variables to include. None → DEFAULT_OBS_VARS.
        May contain derived names (see DERIVED_OBS_VARS): 'wind_east' /
        'wind_north' are computed from wind_speed + wind_from_direction
        (meteorological FROM convention; calm speed==0 → components (0, 0)
        even when direction is missing). Source columns are fetched
        automatically; obs_bounds keys must match obs_vars as listed.
    background_timestamps : np.ndarray or None
        Pool of int64 Unix-ns timestamps for background sample draws.
        Carried for the loader's benefit — TCLoader draws timestamps from
        it; get_background_sample itself takes the timestamp as an
        argument (pure assembly).
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
    target : str or TargetSpec or None
        Prediction target. A name resolved against data/targets.TARGET_SCHEMA,
        a TargetSpec directly, or None → the default ('organisation', the
        9-class ordinal scale). The spec's labeller produces each TC sample's
        label; the model/loss/metrics are selected from it upstream.
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
        target:                Optional[str | TargetSpec] = None,
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
        # Columns actually fetched from InsituLandDataset: derived names are
        # replaced by their source columns (order-preserving, deduped).
        fetch: list[str] = []
        for v in self.obs_vars:
            for src in DERIVED_OBS_VARS.get(v, (v,)):
                if src not in fetch:
                    fetch.append(src)
        self._fetch_vars = fetch
        self.background_timestamps = background_timestamps
        self.location_encoding     = location_encoding
        self.fov_lat               = tuple(fov_lat)
        self.fov_lon               = tuple(fov_lon)
        self.obs_bounds            = obs_bounds
        self.obs_normalisation     = obs_normalisation
        self.target_spec           = (
            target if isinstance(target, TargetSpec) else resolve_target(target)
        )

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

        # Cache frequently accessed arrays for fast sample assembly. The label
        # columns are read by the target spec's labeller straight from
        # `ibtracs`, so they are not cached here (target-agnostic).
        self._lat  = ibtracs['LAT'].astype(np.float32)
        self._lon  = ibtracs['LON'].astype(np.float32)
        self._time = ibtracs['ISO_TIME']                    # int64 Unix-ns
        self._sid  = ibtracs['SID']

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

    def get_tc_sample(self, idx: int) -> Optional[dict]:
        """Assemble one TC sample from IBTrACS row idx.

        All stations within radius_km are used; when more than max_stations
        match, the nearest max_stations by distance are kept (the candidate
        frame from get_obs_near is distance-sorted).

        Parameters
        ----------
        idx : int
            Row index into the IBTrACS split.

        Returns
        -------
        dict | None  — None when matching stations < min_stations or the target
        spec's labeller drops the row (e.g. organisation excludes off-axis
        statuses: extratropical, post-tropical, dissipating, etc.).
        """
        lat = float(self._lat[idx])
        lon = float(self._lon[idx])
        ts  = int(self._time[idx])

        label = self.target_spec.labeller(self.ibtracs, idx)
        if label is None:
            return None

        df = self.insitu.get_obs_near(
            query_lat=lat,
            query_lon=lon,
            timestamp_ns=ts,
            radius_km=self.radius_km,
            window_ns=self.window_ns,
            obs_vars=self._fetch_vars,
        )
        if len(df) < self.min_stations:
            return None

        return self._build_sample(df, lat, lon, label,
                                  sid=str(self._sid[idx]), iso_time=ts)

    # ------------------------------------------------------------------
    # Background sample
    # ------------------------------------------------------------------

    def get_background_sample(
        self,
        lat:          float,
        lon:          float,
        timestamp_ns: int,
    ) -> Optional[dict]:
        """Assemble one background (no-storm) sample at a given point/time.

        Pure assembly: the query position and timestamp are ARGUMENTS —
        sampling policy (uniform vs LHS positions, pool draws, frozen eval
        sets) lives in the loader (TCLoader), not here. Stations are taken
        nearest-first up to max_stations (same as get_tc_sample).

        Parameters
        ----------
        lat, lon : float
            Query position in decimal degrees.
        timestamp_ns : int
            Query timestamp, Unix-ns (typically drawn from the loader's
            background pool).

        Returns
        -------
        dict | None — None when matching stations < min_stations.
        """
        lat = float(lat)
        lon = float(lon)
        ts  = int(timestamp_ns)

        df = self.insitu.get_obs_near(
            query_lat=lat,
            query_lon=lon,
            timestamp_ns=ts,
            radius_km=self.radius_km,
            window_ns=self.window_ns,
            obs_vars=self._fetch_vars,
        )
        if len(df) < self.min_stations:
            return None

        return self._build_sample(df, lat, lon, 0, sid=None, iso_time=ts)

    # ------------------------------------------------------------------
    # Shared sample builder
    # ------------------------------------------------------------------

    def _build_sample(
        self,
        df:        'pd.DataFrame',
        query_lat: float,
        query_lon: float,
        label:     int,
        sid:       Optional[str],
        iso_time:  int,
    ) -> dict:
        # Compute derived obs columns (e.g. wind_east/wind_north from
        # speed + direction) so df[self.obs_vars] below resolves directly.
        if 'wind_east' in self.obs_vars or 'wind_north' in self.obs_vars:
            u, v = wind_to_components(
                df['wind_speed'].to_numpy(dtype=np.float32),
                df['wind_from_direction'].to_numpy(dtype=np.float32),
            )
            df = df.assign(wind_east=u, wind_north=v)

        n_available = len(df)
        F = len(self.obs_vars)

        # Use all stations within radius; when over max_stations keep the
        # NEAREST (df from get_obs_near is sorted by distance). No random
        # subsampling — the region is large and stations sparse, so the cap
        # rarely binds and deterministic selection keeps eval reproducible.
        if n_available > self.max_stations:
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

            x, y = encode_unit_circle(norm_dist, bearing_rad)
            encoded_coords = np.stack([x, y], axis=-1)  # (n_real, 2)
            # (0, 0) = the storm position on the local map; the model adds a
            # learned content token for the query.
            query_coords   = np.zeros(2, dtype=np.float32)

        else:  # domain
            norm_lat, norm_lon = encode_domain(
                raw_lats, raw_lons, self.fov_lat, self.fov_lon,
            )
            encoded_coords = np.stack([norm_lat, norm_lon], axis=-1)  # (n_real, 2)

            q_norm_lat, q_norm_lon = encode_domain(
                query_lat, query_lon, self.fov_lat, self.fov_lon,
            )
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
            # Metadata — kept outside batch['X'] by _collate (decision 13)
            'sid':            sid,
            'iso_time':       np.int64(iso_time),
            'query_lat':      np.float32(query_lat),
            'query_lon':      np.float32(query_lon),
            'n_available':    np.int32(n_available),
        }
