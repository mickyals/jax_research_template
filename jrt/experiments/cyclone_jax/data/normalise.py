"""
experiments/cyclone_jax/data/normalise.py

NormSpec — the declarative NORMALISATION side of a sample (third sibling of
inputs.py / targets.py: policy here, mechanics in utils/normalise).

What gets scaled (x only — y and meta stay RAW, they are eval metadata):

    obs      per-channel by `method`, BEFORE the NaN->0 missingness fill,
             so a zero-filled missing value sits at the channel mean
    level    by `method`
    time     divided by time scale -> [-1, 0]   (relative seconds, <= 0)
    lat/lon  minmax_11 -> [-1, 1] over the domain bounds (SIREN/FINER need
             [-1, 1]; radians only matter for angle-consuming embeddings)
    id       untouched (already {-1, 0, 1}), missing untouched (bool)

Stats policy (user ruling 2026-07-04): computed over the TRAIN split at
train time and LOGGED — train.py writes NormSpec.to_json() to
run_dir/norm_stats.json; evaluation REUSES a training run's saved stats.
Scenarios without a train/all split (e.g. multistorm) cannot self-compute:
they must carry inline stats in the yaml or be evaluated with a stats
pointer to the training run — the point is to name WHICH training
distribution the numbers are relative to.

Config block (configs/data/*.yaml; no block or method 'none' = raw):

    normalise:
      method: standardise        # standardise | minmax_01 | minmax_11
      stats: auto                # auto | inline json-shaped block
    domain:                      # optional; fallback = train-split min/max
      lat: [0, 35]
      lon: [-100, -30]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from utils.normalise import NORMALISERS, StatsAccumulator, get_normaliser

_STAT_KEYS = ('mean', 'std', 'min', 'max', 'count')

# Coordinate-convention guard rails: the store's canonical convention is
# degrees with lon in -180..180 (sources/build.py stores source lon
# verbatim — a 0..360 source must be canonicalised at volume build, this
# layer only refuses to scale it silently).
_GEO_LIMITS = {'lat': (-90.0, 90.0), 'lon': (-180.0, 180.0)}


def _bounds(entry: Mapping, method: str) -> tuple[float, float]:
    """One field's (lo, hi) for `method`; never-observed -> identity-ish
    (0, 1) — such values are all-NaN raw, so the choice is inert."""
    lo, hi = (('mean', 'std') if method == 'standardise' else ('min', 'max'))
    a, b = float(entry[lo]), float(entry[hi])
    if not (np.isfinite(a) and np.isfinite(b)):
        return 0.0, 1.0
    return a, b


@dataclass(frozen=True)
class NormSpec:
    """Materialised normalisation: method + concrete per-field stats.

    Built by NormPolicy.materialise (or from_json for a saved run) —
    construct directly only in tests. `stats` is the json-shaped record
    (module docstring) and THE thing train.py logs.
    """
    method:   str
    channels: tuple[str, ...]          # obs column alignment (InputSpec)
    stats:    Mapping = field(repr=False)

    def __post_init__(self):
        if self.method not in NORMALISERS:
            raise ValueError(f"unknown normalise method {self.method!r} — "
                             f"one of {NORMALISERS.names()} or 'none'.")
        missing = set(self.channels) - set(self.stats.get('obs', {}))
        if missing:
            raise ValueError(f"stats have no obs entry for channel(s) "
                             f"{sorted(missing)}.")
        for f in ('level', 'time', 'lat', 'lon'):
            if f not in self.stats:
                raise ValueError(f"stats missing the {f!r} block.")
        fn = get_normaliser(self.method)
        obs = self.stats['obs']
        lo, hi = zip(*(_bounds(obs[c], self.method) for c in self.channels))
        object.__setattr__(self, '_fn', fn)
        object.__setattr__(self, '_obs_lo', np.asarray(lo, np.float32))
        object.__setattr__(self, '_obs_hi', np.asarray(hi, np.float32))
        object.__setattr__(self, '_level',
                           _bounds(self.stats['level'], self.method))
        scale = float(self.stats['time']['scale'])
        object.__setattr__(self, 'time_scale', scale if scale > 0 else 1.0)
        object.__setattr__(self, '_lat', (float(self.stats['lat']['min']),
                                          float(self.stats['lat']['max'])))
        object.__setattr__(self, '_lon', (float(self.stats['lon']['min']),
                                          float(self.stats['lon']['max'])))
        for f in ('lat', 'lon'):
            lo, hi = getattr(self, f'_{f}')
            glo, ghi = _GEO_LIMITS[f]
            if np.isfinite(lo) and np.isfinite(hi) and (lo < glo or hi > ghi):
                hint = (' — a 0..360-convention source? Canonicalise at '
                        'volume build: ((lon + 180) % 360) - 180'
                        if f == 'lon' else '')
                raise ValueError(
                    f"{f} stats [{lo:g}, {hi:g}] fall outside the "
                    f"geographic range [{glo:g}, {ghi:g}]{hint}.")

    # ------------------------------------------------------------------
    # Application (sampler.Loader calls these; see its docstring for WHERE)
    # ------------------------------------------------------------------

    def obs(self, vals) -> np.ndarray:
        """Scale a raw (n, C) obs matrix. NaN-propagating — call BEFORE
        build_missingness, so zero-fill lands on the channel mean."""
        return self._fn(vals, self._obs_lo, self._obs_hi).astype(np.float32)

    def apply_tail(self, x: dict) -> dict:
        """Scale lat/lon/time/level in place — AFTER station selection
        (haversine needs real degrees). Returns x for chaining."""
        mm = get_normaliser('minmax_11')
        x['lat'] = mm(x['lat'], *self._lat).astype(np.float32)
        x['lon'] = mm(x['lon'], *self._lon).astype(np.float32)
        x['time'] = (x['time'] / self.time_scale).astype(np.float32)
        x['level'] = self._fn(x['level'], *self._level).astype(np.float32)
        return x

    def invert_coords(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Normalised [-1, 1] lat/lon back to degrees.

        Inverse of apply_tail's minmax_11 coordinate scaling — for figures
        that plot station positions from an already-normalised batch (the
        storm panel callback; y/meta coordinates stay raw and never need
        this).
        """
        la_lo, la_hi = self._lat
        lo_lo, lo_hi = self._lon
        lat = la_lo + (np.asarray(lat, np.float32) + 1.0) * (la_hi - la_lo) / 2.0
        lon = lo_lo + (np.asarray(lon, np.float32) + 1.0) * (lo_hi - lo_lo) / 2.0
        return lat.astype(np.float32), lon.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence (train.py -> run_dir/norm_stats.json; evaluate reads)
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        return {'method': self.method, 'channels': list(self.channels),
                'stats': _plain(self.stats)}

    @classmethod
    def from_json(cls, d: Mapping) -> 'NormSpec':
        return cls(method=d['method'], channels=tuple(d['channels']),
                   stats=d['stats'])


def _plain(obj):
    """Numpy scalars/arrays -> json-serialisable python."""
    if isinstance(obj, Mapping):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Policy: config block -> (materialise at build_data time) -> NormSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormPolicy:
    """Resolved `normalise:` block, before stats exist."""
    method: str
    inline: Mapping | None      # stats block, or None = auto (train split)
    domain: Mapping | None      # {'lat': (lo, hi), 'lon': (lo, hi)}

    @property
    def auto(self) -> bool:
        return self.inline is None

    def materialise(self, loader, indices=None) -> NormSpec:
        """Concrete NormSpec — from inline stats, or one raw pass over
        loader samples at `indices` (the train split; interface.py decides
        which indices and raises the no-train-split error before this)."""
        channels = loader.inputs.channels
        if not self.auto:
            stats = dict(self.inline)
        else:
            if indices is None or not len(indices):
                raise ValueError("stats: auto needs a non-empty index set "
                                 "to compute from.")
            if loader.norms is not None:
                raise RuntimeError("stats must be computed on RAW samples — "
                                   "loader already has norms attached.")
            stats = _collect_stats(loader, indices, channels)
        if self.domain:
            for f in ('lat', 'lon'):
                if f in self.domain:
                    dlo, dhi = self.domain[f]
                    if self.auto:
                        # observed coords must fit the declared domain, or
                        # the [-1,1] scaling silently leaves the unit range
                        lo = float(stats[f]['min'])
                        hi = float(stats[f]['max'])
                        if np.isfinite(lo) and np.isfinite(hi) \
                                and (lo < dlo or hi > dhi):
                            raise ValueError(
                                f"observed {f} range [{lo:g}, {hi:g}] "
                                f"exceeds the declared domain "
                                f"[{dlo:g}, {dhi:g}] — coords would scale "
                                f"outside [-1, 1]; widen the domain block "
                                f"or fix the data.")
                    stats[f] = {'min': float(dlo), 'max': float(dhi)}
        return NormSpec(method=self.method, channels=channels, stats=stats)


def _collect_stats(loader, indices, channels) -> dict:
    """One raw pass over the given fixes -> json-shaped stats dict."""
    obs_acc, tail_acc = StatsAccumulator(), StatsAccumulator()
    for i in indices:
        x = loader.build(int(i))['x']
        # x['missing'] is a PRESENT mask; restore NaN where absent so the
        # zero-fill never poisons the statistics.
        obs_acc.update(np.where(x['missing'], x['obs'], np.nan))
        tail_acc.update(np.column_stack(
            [x['level'], x['time'], x['lat'], x['lon']]))
    obs = obs_acc.result()
    tail = tail_acc.result()

    def _entry(res, j):
        return {k: res[k][j] for k in _STAT_KEYS}

    return {
        'obs':   {c: _entry(obs, j) for j, c in enumerate(channels)},
        'level': _entry(tail, 0),
        'time':  {'scale': float(max(abs(tail['min'][1]), 1.0))},
        'lat':   {'min': tail['min'][2], 'max': tail['max'][2]},
        'lon':   {'min': tail['min'][3], 'max': tail['max'][3]},
    }


def resolve_normalise(config: dict) -> NormPolicy | None:
    """Build the NormPolicy from the data config block.

    Keys read: normalise {method, stats}, domain. No block, or
    method 'none', means raw values (None). Value errors surface here;
    the no-train-split case surfaces in interface.build_data.
    """
    block = config.get('normalise')
    if not block:
        return None
    method = block.get('method', 'standardise')
    if method == 'none':
        return None
    if method not in NORMALISERS:
        raise ValueError(f"unknown normalise.method {method!r} — one of "
                         f"{NORMALISERS.names()} or 'none'.")
    stats = block.get('stats', 'auto')
    inline = None if (stats is None or stats == 'auto') else dict(stats)
    domain = config.get('domain')
    if domain:
        unknown = set(domain) - {'lat', 'lon'}
        if unknown:
            raise ValueError(f"domain block has unknown key(s) "
                             f"{sorted(unknown)} — only lat/lon.")
        domain = {f: (float(v[0]), float(v[1])) for f, v in domain.items()}
        for f, (lo, hi) in domain.items():
            glo, ghi = _GEO_LIMITS[f]
            if not (glo <= lo < hi <= ghi):
                raise ValueError(
                    f"domain.{f} [{lo:g}, {hi:g}] must satisfy "
                    f"{glo:g} <= lo < hi <= {ghi:g} (degrees; lon in "
                    f"the -180..180 convention).")
    return NormPolicy(method=method, inline=inline, domain=domain)
