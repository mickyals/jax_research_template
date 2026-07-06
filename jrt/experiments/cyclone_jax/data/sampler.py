"""
experiments/cyclone_jax/data/sampler.py

Two layers, one file (split if either grows):

    Loader   — DETERMINISTIC assembly: fix index -> one ragged named
               sample. No RNG, no epochs, no padding; equally usable for
               per-entity/notebook inspection and training.
    Sampler  — index policy: year splits, stratified subsets, seeded
               re-iterable epoch streams of batch indices. Owns ALL data
               randomness.

A sample is {'x': ..., 'y': ...} with named ragged fields:

    x = {lat (n,), lon (n,), level (n,), time (n,),
         obs (n, C), missing (n, C) bool, id (n,)}
    y = {'target', 'sid', 'lat', 'lon', 'time'}          (targets.build_y)

x['time'] is SECONDS RELATIVE to the fix time, <= 0 by construction
(causality inherited from the bookshelf lookback edges); the absolute fix
time is y['time']. x['level'] is each volume's vertical coordinate (land:
station pressure, marine: SLP, upper: z) — finite by the build-time
vertical gate. Channels a source lacks ride NaN -> 0 with missing False,
so every variable owns one fixed union position (inputs.CHANNEL_ORDER).

Normalisation (normalise.NormSpec, attached by interface.build_data as
loader.norms; None = raw): obs are scaled INSIDE _source_x, before the
NaN->0 missingness fill, so a filled zero sits at the channel mean (the
scaling is NaN-propagating — the mask is unaffected). lat/lon/time/level
are scaled at the END of build, after max_stations selection, because
haversine needs real degrees. y is NEVER normalised (eval metadata).
Everything x carries after build is float32 either way.

Leakage allowlist holds by construction: x is built from obs volumes plus
fix position/time only — no CYC_TARGETS field enters x.

Seeding: one config seed (trainer.seed) populates BOTH sides — jax model
init via utils.jax_core.helpers.create_rng(seed), and data order here via
numpy (this module is jax-free so multiprocess workers stay cheap). Epoch
streams derive rng = default_rng([seed, epoch]): any epoch is re-iterable
without storing state.
"""

from __future__ import annotations

import numpy as np

from datasets.splitting import group_mask, validate_disjoint_groups
from utils.geoscience.geodesic import haversine_np

from experiments.cyclone_jax.data.inputs import SOURCE_SCHEMAS
from experiments.cyclone_jax.data.transformations import build_missingness
from experiments.cyclone_jax.data.sources.shelf import load_lookback
from experiments.cyclone_jax.data.sources.library import (
    CYC_SSHS, TROPICAL_STORM, get_fixes,
)

X_FIELDS = ('lat', 'lon', 'level', 'time', 'obs', 'missing', 'id')


# ---------------------------------------------------------------------------
# Loader — deterministic fix -> sample assembly
# ---------------------------------------------------------------------------

class Loader:
    """Assembles one ragged named sample per driver fix.

    Parameters
    ----------
    lib : dict
        From library.load_library (volumes + shelves, guards passed).
    inputs : InputSpec
        Sources / channel union / selection policy (inputs.resolve_input).
    targets : TargetSpec
        y policy (targets.resolve_target).
    sshs_min : int
        Driver threshold for the fix table (matches the bookshelf build).
    """

    def __init__(self, lib, inputs, targets, sshs_min=TROPICAL_STORM,
                 drop_subtropical=False, norms=None):
        self.lib     = lib
        self.inputs  = inputs
        self.targets = targets
        # build_data attaches this AFTER computing train-split stats (the
        # stats pass itself needs raw samples, i.e. norms is None).
        self.norms   = norms

        self.fixes = get_fixes(lib['volumes']['cyclone'], sshs_min=sshs_min,
                               drop_subtropical=drop_subtropical)
        if targets.kind == 'categorical':
            # Only fixes whose category is IN the label space are samples —
            # a class_set narrower than sshs_min must not crash build_y.
            keep = np.isin(np.rint(np.asarray(self.fixes[CYC_SSHS])),
                           targets.class_set)
            self.fixes = {k: v[keep] for k, v in self.fixes.items()}
        self.storm_times = np.asarray(lib['shelves']['cyclone']['storm_times'])
        self._edges = {}
        for s in inputs.sources:
            edges, _ = load_lookback(lib['shelves'][s])
            if edges is None:
                raise RuntimeError(f"no lookback edges for {s!r} — rebuild "
                                   f"the bookshelf.")
            self._edges[s] = edges

    def __len__(self):
        return len(self.fixes['time'])

    # ------------------------------------------------------------------

    def _source_x(self, s, ti, T):
        """One source's ragged x fields at storm-time index ti."""
        e = self._edges[s]
        lo, hi = int(e[ti, 0]), int(e[ti, -1])
        obs = self.lib['volumes'][s]['obs']
        n = hi - lo

        vals = np.full((n, self.inputs.n_channels), np.nan, np.float32)
        ch = self.inputs.channel_index
        schema = SOURCE_SCHEMAS[s]
        # channels the yaml's `channels:` filtered out are simply skipped
        for col, channel in schema.direct.items():
            if channel in ch:
                vals[:, ch[channel]] = np.asarray(obs[col][lo:hi],
                                                  np.float32)
        for d in schema.derived:
            if not any(c in ch for c in d.channels):
                continue
            cols = (np.asarray(obs[c][lo:hi]) for c in d.columns)
            for channel, arr in zip(d.channels, d.compute(*cols)):
                if channel in ch:
                    vals[:, ch[channel]] = arr
        if self.norms is not None:
            vals = self.norms.obs(vals)      # pre-fill: zero-fill == mean
        vals, missing = build_missingness(vals)

        dt = ((np.asarray(obs['report_timestamp'][lo:hi]).astype('int64')
               - T.astype('int64')) / 1e9).astype(np.float32)
        return {
            'lat':     np.asarray(obs['lat'][lo:hi], np.float32),
            'lon':     np.asarray(obs['lon'][lo:hi], np.float32),
            'level':   np.asarray(obs['level'][lo:hi], np.float32),
            'time':    dt,
            'obs':     vals,
            'missing': missing,
            'id':      np.full(n, self.inputs.source_id[s], np.float32),
        }

    def build(self, i):
        """Sample {'x', 'y'} for fix i (see module docstring for schema)."""
        T = self.fixes['time'][i]
        ti = int(np.searchsorted(self.storm_times, T))
        parts = [self._source_x(s, ti, T) for s in self.inputs.sources]
        x = {k: np.concatenate([p[k] for p in parts], axis=0)
             for k in X_FIELDS}

        k = self.inputs.max_stations
        if self.inputs.selection == 'max_stations' and len(x['lat']) > k:
            d = haversine_np(np.float32(self.fixes['lat'][i]),
                             np.float32(self.fixes['lon'][i]),
                             x['lat'], x['lon'])
            keep = np.argsort(d, kind='stable')[:k]
            x = {f: v[keep] for f, v in x.items()}

        if self.norms is not None:
            x = self.norms.apply_tail(x)     # post-selection: coords/time/level

        return {'x': x, 'y': self.targets.build_y(self.fixes, i)}


# ---------------------------------------------------------------------------
# Index selections: splits + overfit sets
# ---------------------------------------------------------------------------

def split_by_year(fix_times, train_years, val_years, test_years):
    """Disjoint year-based split over fix timestamps -> index arrays.

    Disjointness and masking via datasets.splitting; raises on overlap.
    """
    years = (np.asarray(fix_times).astype('datetime64[Y]').astype(int)
             + 1970)
    sets = {'train': list(train_years), 'val': list(val_years),
            'test': list(test_years)}
    validate_disjoint_groups(sets)
    return {k: np.nonzero(group_mask(years, v))[0] for k, v in sets.items()}


def stratified_fixes(loader, n_per_class, seed=0, classes=None):
    """Balanced overfit set: n_per_class seeded random fixes per remapped
    category (default: the loader's target class_set). Classes short of
    n_per_class contribute everything they have. Sorted index array."""
    classes = loader.targets.class_set if classes is None else classes
    rng = np.random.default_rng(seed)
    sshs = np.asarray(loader.fixes[CYC_SSHS]).astype(int)
    picks = []
    for c in classes:
        idx = np.nonzero(sshs == c)[0]
        take = min(n_per_class, len(idx))
        picks.append(rng.choice(idx, take, replace=False))
    return np.sort(np.concatenate(picks))


# ---------------------------------------------------------------------------
# Sampler — seeded epoch streams of batch indices
# ---------------------------------------------------------------------------

class Sampler:
    """Yields batch index arrays over a fixed index set, deterministically
    in (seed, epoch).

    numpy only — no jax (worker purity), no dependence on the Loader (the
    consumer maps loader.build over the yielded indices and collates).
    drop_last=True keeps every batch full-size so the jitted step sees one
    shape (pair with batching.collate's fixed pad_to).
    """

    def __init__(self, indices, batch_size, seed=0, shuffle=True,
                 drop_last=True):
        self.indices = np.asarray(indices, np.int64)
        if self.indices.ndim != 1 or len(self.indices) == 0:
            raise ValueError("indices must be a non-empty 1-D index array.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        self.batch_size = int(batch_size)
        self.seed       = int(seed)
        self.shuffle    = bool(shuffle)
        self.drop_last  = bool(drop_last)

    def __len__(self):
        """Batches per epoch."""
        n, b = len(self.indices), self.batch_size
        return n // b if self.drop_last else -(-n // b)

    def epoch(self, epoch=0):
        """Yield batch index arrays for one epoch (re-iterable: the order
        is a pure function of (seed, epoch))."""
        idx = self.indices
        if self.shuffle:
            rng = np.random.default_rng([self.seed, int(epoch)])
            idx = rng.permutation(idx)
        stop = len(self) * self.batch_size if self.drop_last else len(idx)
        for b0 in range(0, stop, self.batch_size):
            yield idx[b0:b0 + self.batch_size]

    def __iter__(self):
        return self.epoch(0)
