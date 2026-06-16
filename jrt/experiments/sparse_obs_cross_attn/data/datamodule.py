"""
experiments/sparse_obs_cross_attn/data/datamodule.py

TCDataModule: coordinates IBTrACSDataset and InsituLandDataset into
Trainer-compatible loaders for the sparse_obs_cross_attn experiment.

As additional data sources are added (reanalysis, radar, satellite) they
appear alongside ibtracs and insitu_land in setup(), each contributing
observations to the same sample structure via TCDataset.

Batch format
------------
    {'X': dict_of_jnp_arrays, 'y': jnp.int32_array, 'meta': dict}

    X keys: query_coords, station_obs, station_coords,
            station_mask, obs_mask
    y     : (B,) int32 ordinal class labels
    meta  : non-model sample attribution + diagnostics, NEVER part of X —
            sid (list[str | None], None = background), iso_time (int64),
            query_lat / query_lon (float32 degrees), n_available /
            n_used (int32). The Trainer drops 'meta' before its jitted
            steps; evaluate.py uses it for per-storm attribution.

Class balance
-------------
tc_fraction controls the TC share per batch (default 0.5). Set it below
0.5 to oversample background and counteract the natural bias toward storm
samples in the IBTrACS data (~10 K TC rows vs millions of background
timestamps).
"""

from __future__ import annotations

import math
import warnings
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np

from datasets.datamodule import BaseDataModule
from utils.jax_core.helpers import create_rng
from utils.sampling.coordinate import _key_to_seed, lhs_sample_regional

from experiments.sparse_obs_cross_attn.data.sources.ibtracs import IBTrACSDataset
from experiments.sparse_obs_cross_attn.data.sources.insitu_land import InsituLandDataset
from experiments.sparse_obs_cross_attn.data.dataset import TCDataset
from experiments.sparse_obs_cross_attn.data.splits import resolve_splits
from experiments.sparse_obs_cross_attn.data.targets import resolve_target


# ---------------------------------------------------------------------------
# Background pool construction
# ---------------------------------------------------------------------------

# Best-track rows sit on the 3-hourly synoptic grid (00/03/…/21 UTC, exact
# minutes/seconds). Background timestamps are restricted to the same grid so
# time-of-day can never become a class shortcut once time is encoded. A
# handful of off-grid best-track special rows (landfall/peak inserts) remain
# on the TC side — accepted asymmetry.
SYNOPTIC_STEP_NS: int = 3 * 3600 * 1_000_000_000


def _build_background_pool(
    insitu:              InsituLandDataset,
    ibtracs:             IBTrACSDataset,
    exclude_multi_times: Optional[np.ndarray] = None,
    buffer_hours:        float = 6.0,
) -> np.ndarray:
    """Return synoptic-hour InsituLand timestamps clear of any active TC.

    Only timestamps on the 3-hourly synoptic grid survive (see
    SYNOPTIC_STEP_NS); within those, any timestamp within buffer_hours of
    an IBTrACS observation (or in the multi-storm blackout set) is removed.

    Parameters
    ----------
    insitu : InsituLandDataset (already split to a single season range)
    ibtracs : IBTrACSDataset (already split to the same season range)
    exclude_multi_times : int64 array, optional
        Additional timestamps to exclude (multi-storm blackout set).
    buffer_hours : float
        Timestamps within this many hours of any IBTrACS ISO_TIME are excluded.

    Returns
    -------
    np.ndarray  int64 Unix-ns timestamps
    """
    all_ts = np.unique(insitu.timestamps)
    tc_ts  = np.sort(ibtracs['ISO_TIME'])
    buf_ns = int(buffer_hours * 3600 * 1e9)

    lo = np.searchsorted(tc_ts, all_ts - buf_ns, side='left')
    hi = np.searchsorted(tc_ts, all_ts + buf_ns, side='right')
    near_tc = (hi - lo) > 0

    if exclude_multi_times is not None and len(exclude_multi_times) > 0:
        multi_set = set(int(t) for t in exclude_multi_times)
        in_multi  = np.array([int(t) in multi_set for t in all_ts], dtype=bool)
    else:
        in_multi = np.zeros(len(all_ts), dtype=bool)

    # Unix epoch is 00:00 UTC, so grid membership is a plain modulo.
    on_grid = (all_ts % SYNOPTIC_STEP_NS) == 0

    pool = all_ts[~near_tc & ~in_multi & on_grid]
    if len(pool) == 0:
        warnings.warn(
            "Background timestamp pool is empty after filtering. "
            "Check background_buffer_hours and InsituLand split coverage.",
            UserWarning,
            stacklevel=2,
        )
    return pool


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------

def _collate(samples: list[dict]) -> dict:
    """Stack sample dicts into {'X': dict_of_jnp_arrays, 'y': jnp.array, 'meta': dict}.

    'meta' carries sample attribution and diagnostics (SID, timestamp, raw
    query position, station counts) OUTSIDE the model inputs — batch['X']
    contains exactly the model-facing arrays and nothing else.
    """
    X = {
        k: jnp.array(np.stack([s[k] for s in samples], axis=0))
        for k in (
            'query_coords',
            'station_obs', 'station_coords',
            'station_mask', 'obs_mask',
        )
    }
    y = jnp.array(
        np.stack([s['label'] for s in samples], axis=0),
        dtype=jnp.int32,
    )
    meta = {
        'sid':         [s['sid'] for s in samples],   # str | None (background)
        'iso_time':    np.array([s['iso_time']    for s in samples], dtype=np.int64),
        'query_lat':   np.array([s['query_lat']   for s in samples], dtype=np.float32),
        'query_lon':   np.array([s['query_lon']   for s in samples], dtype=np.float32),
        'n_available': np.array([s['n_available'] for s in samples], dtype=np.int32),
        'n_used':      np.array([s['n_stations']  for s in samples], dtype=np.int32),
    }
    return {'X': X, 'y': y, 'meta': meta}


# ---------------------------------------------------------------------------
# TCLoader
# ---------------------------------------------------------------------------

class TCLoader:
    """Re-iterable loader producing balanced {'X', 'y', 'meta'} batches.

    Each batch contains tc_fraction * batch_size TC samples interleaved
    with background samples. Background sampling POLICY lives here — the
    dataset's get_background_sample is pure assembly given (lat, lon, ts).
    'meta' carries per-sample attribution (SID/time/position) and station
    counts outside the model inputs — see _collate.

    Two iteration modes, selected by ``steps_per_epoch``:

    **Sequential mode** (``steps_per_epoch=None``, default — val/test)
        Iterates through all TC samples in the dataset once per epoch,
        in shuffled or fixed order. Every valid TC sample is yielded —
        the final partial buffer is flushed as a smaller batch with
        proportionally fewer backgrounds (one jit recompile for the odd
        shape is accepted; the Trainer accumulates per-sample).

    **Random mode** (``steps_per_epoch=N`` — training)
        Draws TC samples uniformly at random *with replacement* for exactly
        ``steps_per_epoch`` gradient steps.  TC events are reused across
        steps; each reuse receives a fresh background sample and (when
        more stations are available than max_stations) a different random
        station subset, providing implicit augmentation.  Background
        samples are drawn fresh every step (uniform position + random
        pool timestamp), giving maximum diversity.

    Parameters
    ----------
    dataset : TCDataset
    batch_size : int
    tc_fraction : float
        Fraction of each batch that is TC samples. Default 0.5.
    shuffle : bool
        Reshuffle TC order in sequential mode. Ignored in random mode.
    seed : int
    fov_lat, fov_lon : tuple[float, float]
        Field-of-view bounds for background query positions.
    steps_per_epoch : int or None
        If set, enables random mode and controls epoch length exactly.
        If None, sequential mode is used.
    station_selection : {'random', 'nearest'}
        How stations are chosen when a sample has more candidates than
        max_stations. 'random' = epoch-varying random subset (train
        augmentation); 'nearest' = deterministic nearest-N by distance
        (deployment policy — eval default).
    freeze_backgrounds : bool
        If True (val/test), ONE background set is pre-drawn on first
        iteration — positions via Latin Hypercube, timestamps fixed-seed
        from the synoptic pool — assembled once and reused every epoch,
        so eval differences are purely model change. If False (train),
        backgrounds are drawn fresh each step.
    """

    # fold_in constant separating the frozen-background key stream from
    # the per-epoch stream (which folds in 0, 1, 2, ...).
    _FROZEN_BG_FOLD = 0x0F0F

    def __init__(
        self,
        dataset:            TCDataset,
        batch_size:         int,
        tc_fraction:        float = 0.5,
        shuffle:            bool  = True,
        seed:               int   = 0,
        fov_lat:            tuple[float, float] = (0.0, 30.0),
        fov_lon:            tuple[float, float] = (-100.0, -45.0),
        steps_per_epoch:    Optional[int] = None,
        station_selection:  str   = 'random',
        freeze_backgrounds: bool  = False,
    ) -> None:
        if not (0.0 < tc_fraction < 1.0):
            raise ValueError(f"tc_fraction must be in (0, 1), got {tc_fraction}")
        if station_selection not in ('random', 'nearest'):
            raise ValueError(
                f"station_selection must be 'random' or 'nearest', "
                f"got '{station_selection}'."
            )
        self._dataset            = dataset
        self._batch_size         = batch_size
        self._tc_half            = max(1, round(batch_size * tc_fraction))
        self._bg_half            = batch_size - self._tc_half
        self._shuffle            = shuffle
        self._base_seed          = seed
        self._fov_lat            = fov_lat
        self._fov_lon            = fov_lon
        self._steps_per_epoch    = steps_per_epoch
        self._station_selection  = station_selection
        self._freeze_backgrounds = freeze_backgrounds
        self._frozen_bg: Optional[list[dict]] = None
        self._epoch              = 0

    # ------------------------------------------------------------------
    # Helpers shared by both modes
    # ------------------------------------------------------------------

    def _station_rng(
        self, rng: np.random.Generator
    ) -> Optional[np.random.Generator]:
        """Epoch rng for 'random' station selection, None for 'nearest'."""
        return rng if self._station_selection == 'random' else None

    def _background_pool(self) -> np.ndarray:
        pool = self._dataset.background_timestamps
        if pool is None or len(pool) == 0:
            raise RuntimeError(
                "background_timestamps pool is empty. "
                "Build it via TCDataModule.setup()."
            )
        return pool

    def _draw_background(
        self, rng: np.random.Generator
    ) -> tuple[list[dict], bool]:
        """Draw self._bg_half fresh background samples. Returns (buf, success).

        Position policy: uniform in the FOV box; timestamp drawn randomly
        from the synoptic background pool.
        """
        pool     = self._background_pool()
        bg_buf   = []
        bg_tries = 0
        while len(bg_buf) < self._bg_half:
            bg_tries += 1
            if bg_tries > self._bg_half * 50:
                warnings.warn(
                    f"Could not draw {self._bg_half} background samples "
                    f"after {bg_tries} attempts.",
                    UserWarning,
                    stacklevel=2,
                )
                return bg_buf, False
            lat = float(rng.uniform(self._fov_lat[0], self._fov_lat[1]))
            lon = float(rng.uniform(self._fov_lon[0], self._fov_lon[1]))
            ts  = int(rng.choice(pool))
            bg  = self._dataset.get_background_sample(
                lat, lon, ts, rng=self._station_rng(rng),
            )
            if bg is not None:
                bg_buf.append(bg)
        return bg_buf, True

    # ------------------------------------------------------------------
    # Frozen eval backgrounds (decisions 5 + 9)
    # ------------------------------------------------------------------

    def _build_frozen_backgrounds(self) -> list[dict]:
        """Pre-assemble ONE background sample set, reused every epoch.

        Positions come from a Latin Hypercube over the FOV (space-filling,
        low-variance eval metrics); timestamps from a fixed-seed draw over
        the synoptic pool. Positions that yield < min_stations are skipped
        during the over-drawn walk. Station selection uses the
        deterministic nearest-N path so the assembled samples are
        reproducible from (seed, dataset) alone.
        """
        pool      = self._background_pool()
        n_batches = max(1, math.ceil(len(self._dataset) / self._tc_half))
        n_needed  = (n_batches + 1) * self._bg_half   # +1 covers the flush

        key    = jax.random.fold_in(create_rng(self._base_seed),
                                    self._FROZEN_BG_FOLD)
        n_draw = max(2 * n_needed, 16)                # over-draw for rejection
        lons, lats = lhs_sample_regional(
            key, n_draw, lon_bounds=self._fov_lon, lat_bounds=self._fov_lat,
        )
        ts_rng = np.random.default_rng(_key_to_seed(jax.random.fold_in(key, 1)))
        tss    = ts_rng.choice(pool, size=n_draw)

        frozen: list[dict] = []
        for lat, lon, ts in zip(np.asarray(lats), np.asarray(lons), tss):
            if len(frozen) == n_needed:
                break
            bg = self._dataset.get_background_sample(
                float(lat), float(lon), int(ts), rng=None,
            )
            if bg is not None:
                frozen.append(bg)
        if len(frozen) < n_needed:
            warnings.warn(
                f"Frozen background set has {len(frozen)} samples but "
                f"{n_needed} are needed per epoch — samples will repeat "
                f"within an epoch (cyclic reuse).",
                UserWarning,
                stacklevel=2,
            )
        if not frozen:
            raise RuntimeError(
                "Could not assemble any frozen background samples — every "
                "LHS position/timestamp yielded < min_stations stations."
            )
        return frozen

    def _frozen_slice(self, start: int, count: int) -> list[dict]:
        """count frozen samples starting at start, wrapping cyclically."""
        n = len(self._frozen_bg)
        return [self._frozen_bg[(start + i) % n] for i in range(count)]

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        # Per-epoch seed derived from a JAX key — same pattern as ArrayLoader.
        jax_key = jax.random.fold_in(create_rng(self._base_seed), self._epoch)
        np_seed = int(jax_key[0])
        rng     = np.random.default_rng(np_seed)
        self._epoch += 1

        if self._steps_per_epoch is not None:
            yield from self._iter_random(rng)
        else:
            yield from self._iter_sequential(rng)

    def _iter_random(self, rng: np.random.Generator) -> Iterator[dict]:
        """Random mode: draw TC samples with replacement for steps_per_epoch steps."""
        n_tc = len(self._dataset)
        step = 0
        while step < self._steps_per_epoch:
            # Draw tc_half TC samples randomly with replacement.
            # get_tc_sample may return None (invalid SSHS, or < min_stations),
            # so retry until the buffer is full.
            tc_buf: list[dict] = []
            while len(tc_buf) < self._tc_half:
                idx    = int(rng.integers(0, n_tc))
                sample = self._dataset.get_tc_sample(
                    idx, rng=self._station_rng(rng))
                if sample is not None:
                    tc_buf.append(sample)

            # Draw fresh background samples.
            bg_buf, ok = self._draw_background(rng)
            if not ok:
                continue   # retry this step — background pool temporarily dry

            yield _collate(tc_buf + bg_buf)
            step += 1

    def _iter_sequential(self, rng: np.random.Generator) -> Iterator[dict]:
        """Sequential mode: iterate all TC samples once (val/test).

        Every valid TC sample is yielded: the final partial buffer is
        flushed as a smaller batch with proportionally fewer backgrounds,
        and a failed fresh background draw no longer discards the TC
        buffer (the batch is yielded with however many backgrounds were
        obtained). With freeze_backgrounds the background set is the same
        every epoch and a draw can never fail.
        """
        if self._freeze_backgrounds and self._frozen_bg is None:
            self._frozen_bg = self._build_frozen_backgrounds()

        indices = np.arange(len(self._dataset))
        if self._shuffle:
            rng.shuffle(indices)

        tc_buf: list[dict] = []
        bg_cursor = 0

        def _backgrounds(n_bg: int) -> list[dict]:
            nonlocal bg_cursor
            if self._freeze_backgrounds:
                buf = self._frozen_slice(bg_cursor, n_bg)
                bg_cursor += n_bg
                return buf
            buf, _ = self._draw_background(rng)
            return buf[:n_bg]

        for idx in indices:
            sample = self._dataset.get_tc_sample(
                int(idx), rng=self._station_rng(rng))
            if sample is None:
                continue
            tc_buf.append(sample)

            if len(tc_buf) == self._tc_half:
                yield _collate(tc_buf + _backgrounds(self._bg_half))
                tc_buf = []

        # Flush the final partial batch (decision 9 — previously dropped)
        if tc_buf:
            n_bg = round(len(tc_buf) * self._bg_half / self._tc_half)
            yield _collate(tc_buf + _backgrounds(n_bg))

    def __len__(self) -> int:
        """Random mode: exact. Sequential mode: upper-bound estimate
        (rows yielding None shrink the true count; the flushed partial
        batch is included via ceil)."""
        if self._steps_per_epoch is not None:
            return self._steps_per_epoch
        return max(1, math.ceil(len(self._dataset) / max(1, self._tc_half)))


# ---------------------------------------------------------------------------
# TCDataModule
# ---------------------------------------------------------------------------

class TCDataModule(BaseDataModule):
    """DataModule for the sparse_obs_cross_attn experiment.

    Subclasses BaseDataModule. Because samples are assembled on-the-fly from
    two large datasets (74 M obs rows), this module overrides train_loader /
    val_loader / test_loader to return TCLoader instances rather than
    ArrayLoaders. The in-memory array accessors (train_arrays etc.) are not
    applicable and raise NotImplementedError.

    Config keys (data: block in YAML)
    ----------------------------------
        ibtracs_path             str
        multi_storm_path         str   optional
        ibtracs_sid_meta_path    str   optional
        insitu_obs_path          str
        insitu_meta_path         str
        split                    dict  required — see
                                        experiments.sparse_obs_cross_attn.data.splits
        reliability_levels       list  default [always_active, mostly_active]
        obs_vars                 list  default DEFAULT_OBS_VARS (from insitu_land)
        radius_km                float default 500.0
        time_window_hours        float default 3.0
        max_stations             int   default 64
        min_stations             int   default 1
        batch_size               int   default 64
        tc_fraction              float default 0.5
        fov_lat                  list  default [0.0, 30.0]
        fov_lon                  list  default [-100.0, -45.0]
        background_buffer_hours  float default 6.0
        station_selection        str   default 'random' ('random'|'nearest') —
                                        TRAIN loader station subsampling;
                                        val/test loaders are 'nearest' by
                                        default regardless (decision 9)
        location_encoding        str   default 'unit_circle' ('unit_circle'|'domain')
        obs_normalisation        str   default 'minmax_01'
                                        ('minmax_01'|'minmax_11'|'standardise')
        obs_bounds               dict  optional — {var: [min, max]} for minmax_*,
                                        {var: [mean, std]} for standardise
        target                   str   default 'organisation' — prediction
                                        target, resolved against
                                        data/targets.TARGET_SCHEMA
    """

    @classmethod
    def from_config(cls, config: dict) -> TCDataModule:
        dm = cls()
        dm.setup(config)
        return dm

    # ------------------------------------------------------------------
    # BaseDataModule abstract methods — not applicable for on-the-fly sampling
    # ------------------------------------------------------------------

    def train_arrays(self) -> dict:
        raise NotImplementedError(
            "TCDataModule assembles samples on-the-fly. Use train_loader() instead."
        )

    def val_arrays(self) -> dict:
        raise NotImplementedError(
            "TCDataModule assembles samples on-the-fly. Use val_loader() instead."
        )

    def test_arrays(self) -> dict:
        raise NotImplementedError(
            "TCDataModule assembles samples on-the-fly. Use test_loader() instead."
        )

    def setup(self, config: dict) -> None:
        ibtracs_path      = config['ibtracs_path']
        multi_path        = config.get('multi_storm_path')
        sid_meta_path     = config.get('ibtracs_sid_meta_path')
        obs_path          = config['insitu_obs_path']
        meta_path         = config['insitu_meta_path']
        reliability       = config.get('reliability_levels', ['always_active', 'mostly_active'])
        obs_vars          = config.get('obs_vars', None)
        radius_km         = float(config.get('radius_km', 500.0))
        time_window_h     = float(config.get('time_window_hours', 3.0))
        max_stations      = int(config.get('max_stations', 64))
        min_stations      = int(config.get('min_stations', 1))
        buf_hours         = float(config.get('background_buffer_hours', 6.0))
        location_encoding = config.get('location_encoding', 'unit_circle')
        obs_normalisation = config.get('obs_normalisation', 'minmax_01')
        # Train-loader station selection; eval loaders are always 'nearest'
        # by default (deterministic deployment policy, decision 9).
        self._station_selection = config.get('station_selection', 'random')

        # obs_bounds: dict[var, [min, max]] or dict[var, [mean, std]] for standardise
        obs_bounds_raw = config.get('obs_bounds', None)
        obs_bounds: Optional[dict[str, tuple[float, float]]] = None
        if obs_bounds_raw is not None:
            obs_bounds = {k: tuple(v) for k, v in obs_bounds_raw.items()}

        self._batch_size         = int(config.get('batch_size', 64))
        self._tc_fraction        = float(config.get('tc_fraction', 0.5))
        self._fov_lat            = tuple(config.get('fov_lat', [0.0, 30.0]))
        self._fov_lon            = tuple(config.get('fov_lon', [-100.0, -45.0]))
        self._max_stations       = max_stations
        self._min_stations       = min_stations
        self._location_encoding  = location_encoding
        self._obs_normalisation  = obs_normalisation
        # Prediction target (data.target) — drives the label, head size, loss,
        # metrics, and class names downstream (see data/targets.py). None →
        # the default 'organisation' 9-class ordinal scale.
        self._target_spec        = resolve_target(config.get('target'))

        ibtracs_full = IBTrACSDataset(ibtracs_path, multi_path, sid_meta_path)
        insitu_full  = InsituLandDataset(obs_path, meta_path)
        if reliability:
            insitu_full = insitu_full.filter_reliability(reliability)

        multi_times: Optional[np.ndarray] = None
        if multi_path is not None:
            ms = np.load(multi_path, allow_pickle=True)
            multi_times = ms['ISO_TIME']

        resolved = resolve_splits(config['split'], ibtracs_full, insitu_full)
        self._manifest = resolved['manifest']

        for split_name in ('train', 'val', 'test'):
            ib  = resolved[split_name]['ibtracs']
            ins = resolved[split_name]['insitu']
            bg_pool = _build_background_pool(
                insitu=ins,
                ibtracs=ib,
                exclude_multi_times=multi_times,
                buffer_hours=buf_hours,
            )
            ds = TCDataset(
                ibtracs=ib,
                insitu=ins,
                radius_km=radius_km,
                time_window_hours=time_window_h,
                max_stations=max_stations,
                min_stations=min_stations,
                obs_vars=obs_vars,
                background_timestamps=bg_pool,
                location_encoding=location_encoding,
                fov_lat=self._fov_lat,
                fov_lon=self._fov_lon,
                obs_bounds=obs_bounds,
                obs_normalisation=obs_normalisation,
                target=self._target_spec,
            )
            setattr(self, f'_{split_name}_ds', ds)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def train_loader(
        self,
        batch_size:      Optional[int] = None,
        seed:            int  = 0,
        shuffle:         bool = True,
        steps_per_epoch: Optional[int] = None,
    ) -> TCLoader:
        return TCLoader(
            self._train_ds,
            batch_size or self._batch_size,
            tc_fraction        = self._tc_fraction,
            shuffle            = shuffle,
            seed               = seed,
            fov_lat            = self._fov_lat,
            fov_lon            = self._fov_lon,
            steps_per_epoch    = steps_per_epoch,
            station_selection  = self._station_selection,
            freeze_backgrounds = False,
        )

    def val_loader(
        self,
        batch_size:        Optional[int] = None,
        shuffle:           bool = False,
        station_selection: str  = 'nearest',
    ) -> TCLoader:
        """Deterministic eval loader: nearest-N stations + frozen
        backgrounds, so two iterations yield identical batches.
        station_selection='random' override exists for the
        nearest-vs-random comparison runs (decision 13)."""
        return TCLoader(
            self._val_ds,
            batch_size or self._batch_size,
            tc_fraction=self._tc_fraction,
            shuffle=shuffle,
            seed=0,
            fov_lat=self._fov_lat,
            fov_lon=self._fov_lon,
            station_selection=station_selection,
            freeze_backgrounds=True,
        )

    def test_loader(
        self,
        batch_size:        Optional[int] = None,
        shuffle:           bool = False,
        station_selection: str  = 'nearest',
    ) -> TCLoader:
        """Deterministic eval loader — see val_loader."""
        return TCLoader(
            self._test_ds,
            batch_size or self._batch_size,
            tc_fraction=self._tc_fraction,
            shuffle=shuffle,
            seed=0,
            fov_lat=self._fov_lat,
            fov_lon=self._fov_lon,
            station_selection=station_selection,
            freeze_backgrounds=True,
        )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def manifest(self) -> dict:
        """Resolved split seasons/SIDs/row counts — see resolve_splits."""
        return self._manifest

    @property
    def target_spec(self):
        """The resolved TargetSpec (data.target) — single source of truth for
        n_classes, class_names, and the default loss downstream."""
        return self._target_spec

    # ------------------------------------------------------------------
    # Station-count diagnostics
    # ------------------------------------------------------------------

    def station_diagnostics(
        self,
        split:       str = 'train',
        max_samples: int = 256,
    ) -> Optional[dict]:
        """Station-count statistics over TC samples of one split (decision 9).

        Assembles up to max_samples TC samples (evenly spaced rows,
        deterministic nearest-N station path) and reports avg/min/max of
        n_available (post-dedup candidate stations) and n_used (stations
        in the sample after trimming to max_stations), plus the fraction
        of samples capped at max_stations.

        Returns None when the split is empty or no row yields a sample.
        """
        ds = getattr(self, f'_{split}_ds')
        n  = len(ds)
        if n == 0:
            return None

        idx   = np.unique(np.linspace(0, n - 1, min(n, max_samples)).astype(int))
        avail = []
        used  = []
        for i in idx:
            s = ds.get_tc_sample(int(i), rng=None)
            if s is None:
                continue
            avail.append(int(s['n_available']))
            used.append(int(s['n_stations']))
        if not avail:
            return None

        avail_a = np.array(avail)
        used_a  = np.array(used)
        return {
            'n_samples':   len(avail),
            'n_available': {'avg': float(avail_a.mean()),
                            'min': int(avail_a.min()),
                            'max': int(avail_a.max())},
            'n_used':      {'avg': float(used_a.mean()),
                            'min': int(used_a.min()),
                            'max': int(used_a.max())},
            'frac_capped': float((avail_a >= self._max_stations).mean()),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(
        self,
        steps_per_epoch: Optional[int] = None,
        diagnostics:     bool = True,
    ) -> None:
        """Print a formatted data summary to stdout.

        Parameters
        ----------
        steps_per_epoch : int or None
            When set (random training mode), the train row shows this exact
            value instead of the TC-count-based estimate.
        diagnostics : bool
            Also assemble a subset of TC samples per split and print
            station-count statistics (see station_diagnostics). Default True.
        """
        tc_half = max(1, round(self._batch_size * self._tc_fraction))

        print()
        print("─" * 58)
        print(f"Data  ({self._location_encoding} · {self._obs_normalisation})")
        print(f"  {'split':<6}  {'seasons':>16}  {'SIDs':>6}")
        for name in ('train', 'val', 'test'):
            entry = self._manifest[name]
            seasons = entry['seasons']
            season_str = (
                f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1
                else str(seasons[0]) if seasons else "-"
            )
            print(f"  {name:<6}  {season_str:>16}  {entry['n_sids']:>6,}")
        if 'hard_test' in self._manifest:
            ht = self._manifest['hard_test']
            print(f"  {'hard_test (multi-storm)':<24}  rows={ht['n_rows']:,}  SIDs={ht['n_sids']:,}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*16}  {'─'*10}")
        print(f"  {'split':<6}  {'TC rows':>8}  {'background pool':>16}  {'steps/ep':>10}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*16}  {'─'*10}")
        for name in ('train', 'val', 'test'):
            ds   = getattr(self, f'_{name}_ds')
            n_tc = len(ds)
            n_bg = (
                len(ds.background_timestamps)
                if ds.background_timestamps is not None else 0
            )
            if name == 'train' and steps_per_epoch is not None:
                # Random mode — epoch length is exact and controlled by config
                steps_str = f"{steps_per_epoch:>9,} "
            else:
                # Sequential mode — estimate from TC count
                steps_str = f"{max(1, n_tc // tc_half):>9,}~"
            print(f"  {name:<6}  {n_tc:>8,}  {n_bg:>16,}  {steps_str}")
        mode_str = (
            f"random ({steps_per_epoch:,} steps/ep)"
            if steps_per_epoch is not None else "sequential (1 pass)"
        )
        print(
            f"  train mode: {mode_str}  |  "
            f"batch_size={self._batch_size}  "
            f"max_stations={self._max_stations}  "
            f"tc_fraction={self._tc_fraction}"
        )
        if diagnostics:
            print(f"  {'─'*6}  {'─'*20}  {'─'*20}  {'─'*6}")
            print(f"  {'split':<6}  {'n_avail avg/min/max':>20}  "
                  f"{'n_used avg/min/max':>20}  {'capped':>6}")
            for name in ('train', 'val', 'test'):
                d = self.station_diagnostics(name)
                if d is None:
                    print(f"  {name:<6}  {'(no TC samples)':>20}")
                    continue
                na, nu = d['n_available'], d['n_used']
                print(f"  {name:<6}  "
                      f"{na['avg']:>10.1f} /{na['min']:>3} /{na['max']:>4}  "
                      f"{nu['avg']:>10.1f} /{nu['min']:>3} /{nu['max']:>4}  "
                      f"{d['frac_capped']:>5.0%}")
        print("─" * 58)
        print()
