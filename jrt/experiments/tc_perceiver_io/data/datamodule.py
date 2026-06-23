"""
experiments/tc_perceiver_io/data/datamodule.py

TCDataModule: coordinates IBTrACSDataset and InsituLandDataset into
Trainer-compatible loaders for the tc_perceiver_io experiment.

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
tc_fraction controls the TC share per batch. It is either a scalar (all splits)
or a {train, val, test} mapping resolved by _tc_fraction_for — typically a low
train fraction with a balanced 0.5 val/test so eval metrics are honest and
comparable. Raising it oversamples storms against the natural bias toward
background (~10 K TC rows vs millions of background timestamps).

tc_fraction oversampling and a class-balancing loss (data.class_weight_scheme)
attack the same imbalance, so running both stacks the corrections. When a
weighting scheme is active, the TRAIN loader drops the oversampling and reverts
to natural prevalence — background is recorded as label 0 in the split manifest
(its pool size) and the per-class weights do the balancing. val/test keep their
configured tc_fraction so eval composition stays stable. See _train_tc_fraction
and setup().

Background cleanliness
----------------------
data.background_sampling selects how a (point, time) draw is judged storm-free:
'time' (default) bakes a basin-wide near-TC time exclusion into the pool (see
_build_background_pool); 'spatial' keeps every synoptic grid timestamp and
validates each draw by storm proximity (TCDataset.storm_within) — background iff
no storm within background_exclusion_radius_km at ±background_buffer_hours. See
TCLoader and setup().
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
from utils.sampling.spatial import _key_to_seed, lhs_sample_regional

from experiments.tc_perceiver_io.data.sources.ibtracs import IBTrACSDataset
from experiments.tc_perceiver_io.data.sources.insitu_land import InsituLandDataset
from experiments.tc_perceiver_io.data.dataset import TCDataset
from experiments.tc_perceiver_io.data.splits import resolve_splits
from experiments.tc_perceiver_io.data.inputs import resolve_input
from experiments.tc_perceiver_io.data.targets import resolve_target


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
    exclude_near_tc:     bool  = True,
) -> np.ndarray:
    """Return synoptic-hour InsituLand timestamps for background draws.

    Only timestamps on the 3-hourly synoptic grid survive (see
    SYNOPTIC_STEP_NS); within those, the multi-storm blackout set is always
    removed. The basin-wide near-TC time exclusion is OPTIONAL:

      * ``exclude_near_tc=True`` (default, ``background_sampling='time'``) drops
        any timestamp within buffer_hours of an IBTrACS observation — a
        basin-wide, time-only exclusion. In peak season (a storm almost always
        active *somewhere*) this shrinks the pool sharply.
      * ``exclude_near_tc=False`` (``background_sampling='spatial'``) keeps every
        grid timestamp; cleanliness is then decided per draw by spatial
        proximity (TCDataset.storm_within) rather than basin-wide by time, so a
        point far from every active storm stays a valid background even mid-season.

    Parameters
    ----------
    insitu : InsituLandDataset (already split to a single season range)
    ibtracs : IBTrACSDataset (already split to the same season range)
    exclude_multi_times : int64 array, optional
        Additional timestamps to exclude (multi-storm blackout set).
    buffer_hours : float
        Timestamps within this many hours of any IBTrACS ISO_TIME are excluded
        (only when ``exclude_near_tc`` is True).
    exclude_near_tc : bool
        Apply the basin-wide near-TC time exclusion (default True).

    Returns
    -------
    np.ndarray  int64 Unix-ns timestamps
    """
    all_ts = np.unique(insitu.timestamps)

    if exclude_near_tc:
        tc_ts  = np.sort(ibtracs['ISO_TIME'])
        buf_ns = int(buffer_hours * 3600 * 1e9)
        lo = np.searchsorted(tc_ts, all_ts - buf_ns, side='left')
        hi = np.searchsorted(tc_ts, all_ts + buf_ns, side='right')
        near_tc = (hi - lo) > 0
    else:
        near_tc = np.zeros(len(all_ts), dtype=bool)

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
        steps.  Backgrounds come from a reusable buffer (see
        ``bg_refresh_every`` / ``bg_buffer_size``): by default it is refreshed
        every step (maximum diversity), but raising the interval reuses a pool
        of pre-assembled backgrounds across steps to cut per-step assembly cost
        when batches are background-heavy.

    Station selection up to max_stations is controlled by ``station_selection``:
    'nearest' (default, deterministic — always used for val/test so eval is
    reproducible) or 'random' (train-only augmentation — each draw of a TC row
    yields a different station view; only bites when a sample has more than
    max_stations candidates).

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
    freeze_backgrounds : bool
        If True (val/test), ONE background set is pre-drawn on first
        iteration — positions via Latin Hypercube, timestamps fixed-seed
        from the synoptic pool — assembled once and reused every epoch,
        so eval differences are purely model change. If False (train),
        backgrounds come from the refreshable buffer below.
    bg_refresh_every : int
        Random mode only. Steps between full refreshes of the background
        buffer. Default 1 = assemble fresh backgrounds every step. Larger
        values reuse pre-assembled backgrounds across steps, cutting per-step
        assembly cost (~bg_buffer_size / bg_refresh_every draws per step
        instead of bg_half) at the cost of some background diversity.
    bg_buffer_size : int or None
        Random mode only. Size of the reusable background buffer to sample
        each step's bg_half from. None (default) = bg_half (with
        bg_refresh_every=1 this is exactly fresh-every-step). Floored at
        bg_half so a batch can be filled without replacement.
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
        freeze_backgrounds: bool  = False,
        station_selection:  str   = 'nearest',
        bg_refresh_every:   int   = 1,
        bg_buffer_size:     Optional[int] = None,
        background_sampling:        str   = 'time',
        storm_exclusion_radius_km:  Optional[float] = None,
        storm_time_tol_ns:          Optional[int]   = None,
    ) -> None:
        if not (0.0 < tc_fraction < 1.0):
            raise ValueError(f"tc_fraction must be in (0, 1), got {tc_fraction}")
        if bg_refresh_every < 1:
            raise ValueError(
                f"bg_refresh_every must be >= 1, got {bg_refresh_every}")
        if background_sampling not in ('time', 'spatial'):
            raise ValueError(
                f"background_sampling must be 'time' or 'spatial', "
                f"got {background_sampling!r}")
        if background_sampling == 'spatial' and (
                storm_exclusion_radius_km is None or storm_time_tol_ns is None):
            raise ValueError(
                "background_sampling='spatial' requires "
                "storm_exclusion_radius_km and storm_time_tol_ns.")
        self._dataset            = dataset
        self._batch_size         = batch_size
        self._tc_half            = max(1, round(batch_size * tc_fraction))
        self._bg_half            = batch_size - self._tc_half
        # Reusable train-background buffer (random mode only). Each step samples
        # _bg_half backgrounds from a buffer of _bg_buffer_size pre-assembled
        # samples, refreshed every _bg_refresh_every steps, instead of
        # assembling _bg_half fresh samples every step. The defaults
        # (refresh_every=1, buffer=_bg_half) reproduce fresh-every-step exactly.
        # A larger buffer + interval trades some background diversity for far
        # less per-step assembly cost — assembly drops from _bg_half draws/step
        # to ~_bg_buffer_size/_bg_refresh_every. Eval is unaffected (it uses
        # frozen backgrounds in sequential mode, never this path).
        self._bg_refresh_every   = bg_refresh_every
        self._bg_buffer_size     = (
            max(self._bg_half, int(bg_buffer_size)) if bg_buffer_size
            else self._bg_half
        )
        self._shuffle            = shuffle
        self._base_seed          = seed
        self._fov_lat            = fov_lat
        self._fov_lon            = fov_lon
        self._steps_per_epoch    = steps_per_epoch
        self._freeze_backgrounds = freeze_backgrounds
        # Station-selection policy when a sample has more than max_stations
        # candidates: 'nearest' (deterministic) or 'random' (train-only view
        # augmentation). Applied to BOTH TC and background draws so the two
        # channels share the same selection statistics (no shortcut).
        self._station_selection  = station_selection
        # Background sampling policy: 'time' (basin-wide near-TC time exclusion
        # baked into the pool) or 'spatial' (a draw is background iff no storm is
        # within storm_exclusion_radius_km at ±storm_time_tol_ns of the point/
        # time — see TCDataset.storm_within). Spatial validation gates BOTH the
        # random train draws and the frozen eval set.
        self._background_sampling   = background_sampling
        self._storm_excl_radius_km  = storm_exclusion_radius_km
        self._storm_time_tol_ns     = storm_time_tol_ns
        self._frozen_bg: Optional[list[dict]] = None
        self._epoch              = 0

    # ------------------------------------------------------------------
    # Helpers shared by both modes
    # ------------------------------------------------------------------

    def _background_pool(self) -> np.ndarray:
        pool = self._dataset.background_timestamps
        if pool is None or len(pool) == 0:
            raise RuntimeError(
                "background_timestamps pool is empty. "
                "Build it via TCDataModule.setup()."
            )
        return pool

    def _storm_blocks(self, lat: float, lon: float, ts: int) -> bool:
        """True when spatial mode rejects this (point, time) — a storm is too
        close. Always False in 'time' mode (the pool already did the exclusion)."""
        if self._background_sampling != 'spatial':
            return False
        return self._dataset.storm_within(
            lat, lon, ts,
            radius_km   = self._storm_excl_radius_km,
            time_tol_ns = self._storm_time_tol_ns,
        )

    def _draw_background(
        self, rng: np.random.Generator, n: Optional[int] = None,
    ) -> tuple[list[dict], bool]:
        """Draw n fresh background samples (default self._bg_half).

        Returns (buf, success). Position policy: uniform in the FOV box;
        timestamp drawn randomly from the synoptic background pool. In spatial
        mode a drawn (point, time) with a storm too close is rejected (a try,
        not a sample) — see _storm_blocks.
        """
        count    = self._bg_half if n is None else n
        pool     = self._background_pool()
        bg_buf   = []
        bg_tries = 0
        while len(bg_buf) < count:
            bg_tries += 1
            if bg_tries > count * 50:
                warnings.warn(
                    f"Could not draw {count} background samples "
                    f"after {bg_tries} attempts.",
                    UserWarning,
                    stacklevel=2,
                )
                return bg_buf, False
            lat = float(rng.uniform(self._fov_lat[0], self._fov_lat[1]))
            lon = float(rng.uniform(self._fov_lon[0], self._fov_lon[1]))
            ts  = int(rng.choice(pool))
            if self._storm_blocks(lat, lon, ts):
                continue
            bg  = self._dataset.get_background_sample(
                lat, lon, ts,
                station_selection=self._station_selection, rng=rng)
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
        # Over-draw for rejection (< min_stations, and storm-too-close in
        # spatial mode — which rejects more, so over-draw harder there).
        overdraw = 4 if self._background_sampling == 'spatial' else 2
        n_draw   = max(overdraw * n_needed, 16)
        lons, lats = lhs_sample_regional(
            key, n_draw, lon_bounds=self._fov_lon, lat_bounds=self._fov_lat,
        )
        ts_rng = np.random.default_rng(_key_to_seed(jax.random.fold_in(key, 1)))
        tss    = ts_rng.choice(pool, size=n_draw)

        frozen: list[dict] = []
        for lat, lon, ts in zip(np.asarray(lats), np.asarray(lons), tss):
            if len(frozen) == n_needed:
                break
            if self._storm_blocks(float(lat), float(lon), int(ts)):
                continue
            bg = self._dataset.get_background_sample(
                float(lat), float(lon), int(ts),
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
        """Random mode: draw TC samples with replacement for steps_per_epoch steps.

        Backgrounds come from a reusable buffer refreshed every
        _bg_refresh_every steps (see __init__): each step samples _bg_half from
        a buffer of _bg_buffer_size pre-assembled samples rather than assembling
        _bg_half fresh ones every step. With the defaults (refresh_every=1,
        buffer=_bg_half) this is exactly fresh-every-step.
        """
        n_tc = len(self._dataset)
        bg_buffer: list[dict] = []
        step = 0
        while step < self._steps_per_epoch:
            # Draw tc_half TC samples randomly with replacement.
            # get_tc_sample may return None (invalid SSHS, or < min_stations),
            # so retry until the buffer is full.
            tc_buf: list[dict] = []
            while len(tc_buf) < self._tc_half:
                idx    = int(rng.integers(0, n_tc))
                sample = self._dataset.get_tc_sample(
                    idx, station_selection=self._station_selection, rng=rng)
                if sample is not None:
                    tc_buf.append(sample)

            # Refresh the background buffer on the cadence, then sample this
            # step's backgrounds from it (without replacement when the buffer is
            # large enough, so a single batch has no duplicate backgrounds).
            if step % self._bg_refresh_every == 0:
                bg_buffer, ok = self._draw_background(rng, n=self._bg_buffer_size)
                if not ok:
                    continue   # retry this step — background pool temporarily dry
            if not bg_buffer:
                continue
            sel    = rng.choice(
                len(bg_buffer), size=self._bg_half,
                replace=len(bg_buffer) < self._bg_half,
            )
            bg_buf = [bg_buffer[i] for i in sel]

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
                int(idx), station_selection=self._station_selection, rng=rng)
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
    """DataModule for the tc_perceiver_io experiment.

    Subclasses BaseDataModule. Because samples are assembled on-the-fly from
    two large datasets (74 M obs rows), this module implements train_loader /
    val_loader / test_loader to return TCLoader instances rather than
    ArrayLoaders. There are no in-memory array accessors — the base contract is
    loaders only (r20), so nothing to stub out.

    Config keys (data: block in YAML)
    ----------------------------------
        ibtracs_path             str
        multi_storm_path         str   optional
        ibtracs_sid_meta_path    str   optional
        insitu_obs_path          str
        insitu_meta_path         str
        split                    dict  required — see
                                        experiments.tc_perceiver_io.data.splits
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
        location_encoding        str   default 'unit_circle' ('unit_circle'|'domain')
        obs_normalisation        str   default 'minmax_01'
                                        ('minmax_01'|'minmax_11'|'standardise')
        obs_bounds               dict  optional — {var: [min, max]} for minmax_*,
                                        {var: [mean, std]} for standardise
        target                   str   default 'organisation' — prediction
                                        target, resolved against
                                        data/targets.TARGET_SCHEMA
    """

    # Whether a class-balancing loss is active (data.class_weight_scheme !=
    # 'none'). Set in setup(); the default keeps stub instances that bypass
    # setup (e.g. tests) on the configured tc_fraction. When True, the train
    # loader stops oversampling TC — see _train_tc_fraction.
    _cw_active: bool = False

    # Train-background buffer defaults (set in setup()); the class defaults keep
    # stub instances that bypass setup on fresh-every-step behaviour. See
    # TCLoader for semantics.
    _bg_refresh_every: int = 1
    _bg_buffer_size: Optional[int] = None

    # Background sampling policy defaults (set in setup()); class defaults keep
    # stub instances that bypass setup on the legacy time-only behaviour.
    _bg_sampling: str = 'time'
    _bg_excl_radius_km: Optional[float] = None
    _bg_time_tol_ns: int = 0

    @classmethod
    def from_config(cls, config: dict) -> TCDataModule:
        dm = cls()
        dm.setup(config)
        return dm

    def setup(self, config: dict) -> None:
        ibtracs_path      = config['ibtracs_path']
        multi_path        = config.get('multi_storm_path')
        sid_meta_path     = config.get('ibtracs_sid_meta_path')
        obs_path          = config['insitu_obs_path']
        meta_path         = config['insitu_meta_path']
        reliability       = config.get('reliability_levels', ['always_active', 'mostly_active'])
        radius_km         = float(config.get('radius_km', 500.0))
        time_window_h     = float(config.get('time_window_hours', 3.0))
        max_stations      = int(config.get('max_stations', 64))
        min_stations      = int(config.get('min_stations', 1))
        buf_hours         = float(config.get('background_buffer_hours', 6.0))

        # Input configuration (data.obs_vars / obs_normalisation / obs_bounds /
        # location_encoding / fov_*) — drives the observation variables, their
        # normalisation, the coordinate encoding, and the FOV bounds (see
        # data/inputs.py). The encoder stays input-agnostic.
        self._input_spec         = resolve_input(config)

        self._batch_size         = int(config.get('batch_size', 64))
        # tc_fraction may be a scalar (all splits) or a {train, val, test} dict
        # (e.g. natural-ish train + balanced val/test for honest eval metrics).
        # Stored raw; _tc_fraction_for(split) resolves either form.
        self._tc_fraction        = config.get('tc_fraction', 0.5)
        # Background sampling policy (see _build_background_pool / TCLoader):
        # 'time' (default) bakes a basin-wide near-TC time exclusion into the
        # pool; 'spatial' keeps every grid timestamp and validates each draw by
        # storm proximity (no storm within background_exclusion_radius_km — null
        # → radius_km — at ±background_buffer_hours of the point/time).
        self._bg_sampling        = str(config.get('background_sampling', 'time'))
        _excl                    = config.get('background_exclusion_radius_km')
        self._bg_excl_radius_km  = float(_excl) if _excl else radius_km
        self._bg_time_tol_ns     = int(buf_hours * 3600 * 1e9)
        # A class-balancing loss and tc_fraction oversampling both correct the
        # same imbalance; running both stacks the corrections. When weighting is
        # on, the train loader reverts to natural prevalence and the weights do
        # the balancing — see _train_tc_fraction.
        self._cw_active          = str(config.get('class_weight_scheme', 'none')) != 'none'
        # Reusable train-background buffer (random mode). Defaults reproduce
        # fresh-every-step assembly; raise bg_refresh_every (+ optionally
        # bg_buffer_size) to cut per-step background-assembly cost when the
        # batch is background-heavy (e.g. natural prevalence). See TCLoader.
        self._bg_refresh_every   = int(config.get('bg_refresh_every', 1))
        self._bg_buffer_size     = config.get('bg_buffer_size')
        # Train-only station-selection policy ('nearest' | 'random'); val/test
        # always use 'nearest' (deterministic eval). See TCLoader.
        self._station_selection  = str(config.get('station_selection', 'nearest'))
        # FOV bounds live on the InputSpec (single source of truth) — used here
        # for the loaders' background sampling and the background pool.
        self._fov_lat            = self._input_spec.fov_lat
        self._fov_lon            = self._input_spec.fov_lon
        self._max_stations       = max_stations
        self._min_stations       = min_stations
        # Prediction target (data.target) — drives the label, head size, loss,
        # metrics, and class names downstream (see data/targets.py). None →
        # the default 'organisation' 9-class ordinal scale.
        self._target_spec        = resolve_target(config.get('target'))

        ibtracs_full = IBTrACSDataset(ibtracs_path, multi_path, sid_meta_path)
        # cache_sorted: on a slow .npz load, build (and reuse) a sibling
        # mmap-able sorted cache so future runs start in seconds (default on).
        insitu_full  = InsituLandDataset(
            obs_path, meta_path,
            cache_sorted=bool(config.get('cache_sorted_obs', True)))
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
                # Spatial mode keeps every grid timestamp and validates draws by
                # storm proximity instead of basin-wide time exclusion.
                exclude_near_tc=(self._bg_sampling != 'spatial'),
            )
            # Background (label 0) is a real class but never appears as an
            # IBTrACS row, so resolve_splits leaves its count at 0. Record its
            # natural prevalence — the size of the TC-free synoptic pool — so
            # class-balancing weights treat background as the (dominant) class
            # it is, rather than holding it neutral at 1.0.
            if split_name in self._manifest:
                self._manifest[split_name]['class_counts']['0'] = int(len(bg_pool))
            ds = TCDataset(
                ibtracs=ib,
                insitu=ins,
                radius_km=radius_km,
                time_window_hours=time_window_h,
                max_stations=max_stations,
                min_stations=min_stations,
                background_timestamps=bg_pool,
                inputs=self._input_spec,
                target=self._target_spec,
            )
            setattr(self, f'_{split_name}_ds', ds)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _tc_fraction_for(self, split: str) -> float:
        """Resolve data.tc_fraction for one split.

        Accepts a scalar (applies to every split) or a {train, val, test} dict;
        a split missing from the dict falls back to 0.5. val/test typically use
        a balanced 0.5 for honest metrics while train sits lower (or reverts to
        natural prevalence under class weighting — see _train_tc_fraction).
        """
        tf = self._tc_fraction
        if isinstance(tf, dict):
            return float(tf.get(split, 0.5))
        return float(tf)

    def _train_tc_fraction(self) -> float:
        """TC fraction per batch for the TRAIN loader.

        When a class-balancing loss is active (data.class_weight_scheme !=
        'none') the per-class weights carry the imbalance correction, so the
        sampler must not *also* oversample TC — otherwise the two corrections
        stack. The train batch then reverts to the natural TC:background
        prevalence (from the train manifest's class counts) and the weights do
        the balancing. The TCLoader's max(1, ...) floor still guarantees at
        least one TC sample per batch.

        val/test always use the configured tc_fraction so eval composition
        stays stable and comparable across runs (see val_loader/test_loader).

        Falls back to the configured tc_fraction when weighting is off, or when
        the natural fraction is undefined (no background or no TC in the split).
        """
        if not self._cw_active:
            return self._tc_fraction_for('train')
        counts = self._manifest['train']['class_counts']
        n_bg = int(counts.get('0', 0))
        n_tc = sum(int(v) for k, v in counts.items() if k != '0')
        if n_bg == 0 or n_tc == 0:
            return self._tc_fraction_for('train')
        return n_tc / (n_tc + n_bg)

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
            tc_fraction        = self._train_tc_fraction(),
            shuffle            = shuffle,
            seed               = seed,
            fov_lat            = self._fov_lat,
            fov_lon            = self._fov_lon,
            steps_per_epoch    = steps_per_epoch,
            freeze_backgrounds = False,
            station_selection  = self._station_selection,   # train-only random views
            bg_refresh_every   = self._bg_refresh_every,
            bg_buffer_size     = self._bg_buffer_size,
            background_sampling       = self._bg_sampling,
            storm_exclusion_radius_km = self._bg_excl_radius_km,
            storm_time_tol_ns         = self._bg_time_tol_ns,
        )

    def val_loader(
        self,
        batch_size: Optional[int] = None,
        shuffle:    bool = False,
    ) -> TCLoader:
        """Deterministic eval loader: nearest-N stations + frozen backgrounds,
        so two iterations yield identical batches."""
        return TCLoader(
            self._val_ds,
            batch_size or self._batch_size,
            tc_fraction=self._tc_fraction_for('val'),
            shuffle=shuffle,
            seed=0,
            fov_lat=self._fov_lat,
            fov_lon=self._fov_lon,
            freeze_backgrounds=True,
            background_sampling       = self._bg_sampling,
            storm_exclusion_radius_km = self._bg_excl_radius_km,
            storm_time_tol_ns         = self._bg_time_tol_ns,
        )

    def test_loader(
        self,
        batch_size: Optional[int] = None,
        shuffle:    bool = False,
    ) -> TCLoader:
        """Deterministic eval loader — see val_loader."""
        return TCLoader(
            self._test_ds,
            batch_size or self._batch_size,
            tc_fraction=self._tc_fraction_for('test'),
            shuffle=shuffle,
            seed=0,
            fov_lat=self._fov_lat,
            fov_lon=self._fov_lon,
            freeze_backgrounds=True,
            background_sampling       = self._bg_sampling,
            storm_exclusion_radius_km = self._bg_excl_radius_km,
            storm_time_tol_ns         = self._bg_time_tol_ns,
        )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def manifest(self) -> dict:
        """Resolved split years/SIDs/row counts — see resolve_splits."""
        return self._manifest

    @property
    def target_spec(self):
        """The resolved TargetSpec (data.target) — single source of truth for
        n_classes, class_names, and the default loss downstream."""
        return self._target_spec

    @property
    def input_spec(self):
        """The resolved InputSpec (data.obs_vars/obs_normalisation/obs_bounds/
        location_encoding/fov_*) — single source of truth for the observation
        variables, normalisation, coordinate encoding, and FOV bounds."""
        return self._input_spec

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
            s = ds.get_tc_sample(int(i))
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

    def coverage_figure(
        self,
        split:       str  = 'train',
        geo:         bool = False,
        max_batches: Optional[int] = None,
    ):
        """FOV map of sample positions coloured by TRUE class (data diagnostic).

        Iterates the split's loader collecting each sample's query lat/lon and
        true label from the batch ``meta``/``y`` (no model), then composes the
        experiment's ``plot_class_coverage_map`` over the template
        ``plot_categorical_scatter`` primitive. Background (class 0) appears at
        its draw positions alongside the storm classes, so the spatial coverage
        and imbalance across the region are visible at a glance.

        Parameters
        ----------
        split : {'train', 'val', 'test'}
        geo : bool
            Draw on a cartopy FOV map (requires cartopy); else a plain lat/lon
            scatter. Default False.
        max_batches : int, optional
            Cap on batches drawn — required-ish for the random train loader
            (otherwise it yields steps_per_epoch batches). None = exhaust the
            loader (val/test are finite).
        """
        from experiments.tc_perceiver_io.plotting.plotting import (
            plot_class_coverage_map,
        )
        loader = getattr(self, f'{split}_loader')()
        lats, lons, labels = [], [], []
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            m = batch['meta']
            lats.append(np.asarray(m['query_lat']))
            lons.append(np.asarray(m['query_lon']))
            labels.append(np.asarray(batch['y']))
        return plot_class_coverage_map(
            np.concatenate(lats), np.concatenate(lons), np.concatenate(labels),
            self._target_spec.class_names,
            fov_lat=self._fov_lat, fov_lon=self._fov_lon,
            geo=geo, n_classes=self._target_spec.n_classes,
        )

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
        print()
        print("─" * 58)
        print(f"Data  ({self._input_spec.location_encoding} · {self._input_spec.normalisation})")
        print(f"  {'split':<6}  {'years':>16}  {'SIDs':>6}")
        for name in ('train', 'val', 'test'):
            entry = self._manifest[name]
            years = entry['years']
            year_str = (
                f"{years[0]}-{years[-1]}" if len(years) > 1
                else str(years[0]) if years else "-"
            )
            print(f"  {name:<6}  {year_str:>16}  {entry['n_sids']:>6,}")
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
            # Train reverts to natural prevalence when class weighting is on;
            # val/test always use the configured tc_fraction (see
            # _train_tc_fraction).
            frac    = self._train_tc_fraction() if name == 'train' else self._tc_fraction_for(name)
            tc_half = max(1, round(self._batch_size * frac))
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
        tc_frac_str = f"tc_fraction={self._tc_fraction}"
        if self._cw_active:
            # Train sampler is bypassed in favour of natural prevalence; the
            # class-balancing loss carries the imbalance correction.
            tc_frac_str += (
                f" (train→{self._train_tc_fraction():.4g}, natural prevalence: "
                f"class weights active)"
            )
        print(
            f"  train mode: {mode_str}  |  "
            f"batch_size={self._batch_size}  "
            f"max_stations={self._max_stations}  "
            f"{tc_frac_str}"
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
