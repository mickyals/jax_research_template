"""
experiments/sparse_obs_encoder/data/dataset.py

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

Location encoding modes (see data/transforms/encoding.py for the encode/decode pairs)
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

from experiments.sparse_obs_encoder.data.sources.ibtracs import IBTrACSDataset
from experiments.sparse_obs_encoder.data.sources.insitu_land import InsituLandDataset
from experiments.sparse_obs_encoder.data.inputs import InputSpec
from experiments.sparse_obs_encoder.data.targets import TargetSpec, resolve_target
from experiments.sparse_obs_encoder.data.transforms.derived import compute_derived

if TYPE_CHECKING:
    import pandas as pd


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
    background_timestamps : np.ndarray or None
        Pool of int64 Unix-ns timestamps for background sample draws.
        Carried for the loader's benefit — TCLoader draws timestamps from
        it; get_background_sample itself takes the timestamp as an
        argument (pure assembly).
    inputs : InputSpec or None
        Declarative input configuration (see data/inputs.py): observation
        variables (incl. derived names), normalisation, coordinate encoding,
        and FOV bounds. None → the default InputSpec. The spec supplies the
        resolved transforms (normaliser, coord_encoder) and the fetch-column
        resolution; the encoder stays input-agnostic.
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
        background_timestamps: Optional[np.ndarray] = None,
        inputs:                Optional[InputSpec] = None,
        target:                Optional[str | TargetSpec] = None,
    ) -> None:

        self.ibtracs               = ibtracs
        self.insitu                = insitu
        self.radius_km             = float(radius_km)
        self.window_ns             = int(time_window_hours * 3600 * 1e9)
        self.max_stations          = int(max_stations)
        self.min_stations          = int(min_stations)
        self.background_timestamps = background_timestamps
        self.inputs                = inputs if inputs is not None else InputSpec()
        self.obs_vars              = list(self.inputs.obs_vars)
        # Source columns to fetch: derived names → their source columns
        # (order-preserving, deduped) — resolved by the InputSpec.
        self._fetch_vars           = self.inputs.fetch_vars
        self.target_spec           = (
            target if isinstance(target, TargetSpec) else resolve_target(target)
        )

        # Pre-compute per-variable normalisation arrays for fast reuse.
        # _obs_lo/_obs_hi store (min, max) for minmax modes and (mean, std)
        # for 'standardise' — the names reflect the minmax convention.
        self._obs_lo, self._obs_hi = self.inputs.bounds_arrays()

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
        df = compute_derived(df, self.obs_vars)

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

        # Normalise obs values using the InputSpec's normaliser + pre-computed
        # bounds. Missingness is structural: re-zero positions that were
        # missing before normalisation (the normaliser never sees the mask).
        if self._obs_lo is not None:
            obs_safe = self.inputs.normaliser(obs_safe, self._obs_lo, self._obs_hi)
            obs_safe = obs_safe * obs_mask

        # Encode station + query coordinates via the InputSpec's encoder.
        encoded_coords, query_coords = self.inputs.coord_encoder(
            df, query_lat, query_lon,
            radius_km=self.radius_km,
            fov_lat=self.inputs.fov_lat,
            fov_lon=self.inputs.fov_lon,
        )

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
