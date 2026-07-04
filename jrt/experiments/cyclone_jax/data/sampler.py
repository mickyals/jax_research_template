"""
experiments/cyclone_jax/data/sampler.py

Sample assembly: one storm FIX -> the model-ready token arrays. Replaces
the old splits.py role too (splits + overfit sets are fix-index selections
over the bookshelf). Batching (batch.py) only adds the batch dimension.

A sample is Station x [location | dt | obs | missingness | ID]:
    location    lat, lon (degrees — Fourier encoding is model-side)
    dt          seconds before the fix time, <= 0 (carried even though the
                v1 model may ignore it — cache cheap, consume selectively)
    obs         the CHANNELS union (u/v-unified wind), NaN -> 0
    missingness one mask bit per channel (True = measured)
    ID          scalar source code (-1 land / +1 marine / 0 upper)

Causality is inherited from the bookshelf lookback edges: only rows in
[T - reach, T] are ever gathered. The leakage allowlist holds by
construction — tokens are built from obs volumes + fix position/time only;
no CYC_TARGETS field enters the token array.
"""

from __future__ import annotations

import numpy as np

from experiments.cyclone_jax.data.sources.shelf import load_lookback
from utils.geoscience.geodesic import haversine_np

from experiments.cyclone_jax.data.inputs import (
    DEFAULT_SOURCE_ID, SOURCE_SCHEMAS, union_channels,
)
from experiments.cyclone_jax.data.transformations import (
    build_missingness, stamp_source_id,
)
from experiments.cyclone_jax.data.sources.library import (
    CYC_SSHS, TROPICAL_STORM, get_fixes,
)

# v1 channel union (land + marine); the schemas live in inputs.py. P3 makes
# this per-instance via InputSpec.
CHANNELS = union_channels(('land', 'marine'))

# Token column layout: [lat, lon, dt, obs(C), mask(C), id]
N_LOC = 3
TOKEN_DIM = N_LOC + 2 * len(CHANNELS) + 1


class FixSampler:
    """Assembles one sample per driver fix from a loaded library.

    Parameters
    ----------
    lib : dict
        From library.load_library (volumes + shelves, guards passed).
    sources : sequence
        Obs volumes to gather ('land', 'marine' for v1).
    pad_to : int
        Fixed station-token count (1536 covers the v1 max of 1534).
    sshs_min / class_min : int
        Driver threshold and label origin: label = sshs - class_min.
    selection : 'all' | 'max_stations'
        'all' = every measurement in the FOV (default). 'max_stations' =
        the nearest max_stations tokens by haversine distance to the fix.
    """

    def __init__(self, lib, sources=('land', 'marine'), pad_to=1536,
                 sshs_min=TROPICAL_STORM, class_min=TROPICAL_STORM,
                 source_id=None, selection='all', max_stations=None):
        if selection not in ('all', 'max_stations'):
            raise ValueError(f"selection must be 'all' or 'max_stations', "
                             f"got {selection!r}")
        if selection == 'max_stations' and not max_stations:
            raise ValueError("selection='max_stations' requires max_stations.")
        self.lib          = lib
        self.sources      = tuple(sources)
        self.pad_to       = int(pad_to)
        self.class_min    = int(class_min)
        self.selection    = selection
        self.max_stations = int(max_stations) if max_stations else None
        self.source_id    = dict(source_id or DEFAULT_SOURCE_ID)

        self.fixes = get_fixes(lib['volumes']['cyclone'], sshs_min=sshs_min)
        self.storm_times = np.asarray(lib['shelves']['cyclone']['storm_times'])
        self._edges, self._deltas = {}, {}
        for s in self.sources:
            edges, deltas = load_lookback(lib['shelves'][s])
            if edges is None:
                raise RuntimeError(f"no lookback edges for {s!r} — rebuild "
                                   f"the bookshelf.")
            self._edges[s], self._deltas[s] = edges, deltas

    def __len__(self):
        return len(self.fixes['time'])

    # ------------------------------------------------------------------

    def _source_tokens(self, s, ti, T):
        """(n, TOKEN_DIM) float32 token block for one source at time idx ti."""
        e = self._edges[s]
        lo, hi = int(e[ti, 0]), int(e[ti, -1])
        obs = self.lib['volumes'][s]['obs']
        n = hi - lo
        if n == 0:
            return np.zeros((0, TOKEN_DIM), np.float32)

        vals = np.full((n, len(CHANNELS)), np.nan, np.float32)
        ch = {c: j for j, c in enumerate(CHANNELS)}
        schema = SOURCE_SCHEMAS[s]
        for col, channel in schema.direct.items():
            vals[:, ch[channel]] = np.asarray(obs[col][lo:hi], np.float32)
        for d in schema.derived:
            cols = (np.asarray(obs[c][lo:hi]) for c in d.columns)
            for channel, arr in zip(d.channels, d.compute(*cols)):
                vals[:, ch[channel]] = arr

        vals, mask = build_missingness(vals)
        dt = ((np.asarray(obs['report_timestamp'][lo:hi]).astype('int64')
               - T.astype('int64')) / 1e9).astype(np.float32)[:, None]
        loc = np.stack([np.asarray(obs['lat'][lo:hi], np.float32),
                        np.asarray(obs['lon'][lo:hi], np.float32)], axis=1)
        sid_col = stamp_source_id(n, self.source_id[s])
        return np.concatenate(
            [loc, dt, vals, mask.astype(np.float32), sid_col], axis=1)

    def build(self, i):
        """Sample dict for fix i: tokens (pad_to, TOKEN_DIM), station_mask
        (pad_to,), label int32, + meta (sid / time / sshs / n_stations)."""
        T = self.fixes['time'][i]
        ti = int(np.searchsorted(self.storm_times, T))
        blocks = [self._source_tokens(s, ti, T) for s in self.sources]
        tok = np.concatenate(blocks, axis=0) if blocks else \
            np.zeros((0, TOKEN_DIM), np.float32)

        if self.selection == 'max_stations' and len(tok) > self.max_stations:
            d = haversine_np(np.float32(self.fixes['lat'][i]),
                             np.float32(self.fixes['lon'][i]),
                             tok[:, 0], tok[:, 1])
            tok = tok[np.argsort(d, kind='stable')[:self.max_stations]]

        n = min(len(tok), self.pad_to)
        tokens = np.zeros((self.pad_to, TOKEN_DIM), np.float32)
        tokens[:n] = tok[:n]
        station_mask = np.zeros(self.pad_to, bool)
        station_mask[:n] = True

        return {
            'tokens':       tokens,
            'station_mask': station_mask,
            'label':        np.int32(int(self.fixes[CYC_SSHS][i])
                                     - self.class_min),
            'sid':          str(self.fixes['sid'][i]),
            'time':         T,
            'sshs':         float(self.fixes[CYC_SSHS][i]),
            'n_stations':   np.int32(n),
        }


# ---------------------------------------------------------------------------
# Fix-index selections: splits + overfit sets
# ---------------------------------------------------------------------------

def split_by_year(fix_times, train_years, val_years, test_years):
    """Disjoint year-based split over fix timestamps -> index arrays."""
    years = fix_times.astype('datetime64[Y]').astype(int) + 1970
    sets = {'train': train_years, 'val': val_years, 'test': test_years}
    out = {k: np.nonzero(np.isin(years, list(v)))[0] for k, v in sets.items()}
    for a in ('train', 'val'):
        for b in ('val', 'test'):
            if a != b:
                assert not set(sets[a]) & set(sets[b]), 'split years overlap'
    return out


def stratified_fixes(sampler, n_per_class, seed=0, classes=range(3, 9)):
    """Balanced overfit set: n_per_class random fixes per remapped category
    (drawn from the sampler's fix table). Classes short of n_per_class
    contribute everything they have."""
    rng = np.random.default_rng(seed)
    sshs = np.asarray(sampler.fixes[CYC_SSHS]).astype(int)
    picks = []
    for c in classes:
        idx = np.nonzero(sshs == c)[0]
        take = min(n_per_class, len(idx))
        picks.append(rng.choice(idx, take, replace=False))
    return np.sort(np.concatenate(picks))
