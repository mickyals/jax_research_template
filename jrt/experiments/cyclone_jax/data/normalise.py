"""
experiments/cyclone_jax/data/normalise.py

NormSpec — the declarative NORMALISATION side of a sample (third sibling of
inputs.py / targets.py: policy here, stats mechanics in utils/normalise).

PHYSICAL DECLARED BOUNDS (user ruling 2026-07-06, supersedes the flat
method/stats form): the yaml declares each field's scaling, so tokens are
exactly identical across land/marine/channel-subset scenarios — no
per-scenario stats drift. The train split is consulted only for fields
explicitly marked ``auto``.

Config block — grouped by coordinate kind; the KEYED PAIR selects the
method per field:

    normalise:
      surface_coordinate:               # -> minmax_11, [-1, 1]
        lat: {min: 0.0, max: 30.0}
        lon: {min: -100.0, max: -30.0}
      vertical_coordinate:
        level: {min: 70000.0, max: 108000.0}
      time_coordinate:
        time: {scale: 10800.0}          # dt / scale -> [-1, 0]
      variables:                        # the obs channels
        slp:      {min: 87000.0, max: 105000.0}    # minmax_11
        air_temp: {mean: 300.0, std: 5.0}          # standardise
        sst:      auto                  # train-split mean/std

Entry forms:
    {min, max}    affine map onto [-1, 1] (signed-symmetric bounds keep
                  0 at 0 — the u/v wind case)
    {mean, std}   z-score
    {scale}       time only: dt/scale (dt is relative seconds <= 0)
    auto          computed over the TRAIN split at build_data time —
                  variables/level get mean/std, lat/lon observed min/max,
                  time scale = max |dt|. ANY auto entry costs one raw
                  pass over the split; an all-declared block builds with
                  NO data pass (and no observed-coverage check — declared
                  bounds are trusted).

Every ACTIVE channel (after ``channels:`` filtering) must have a
variables entry — missing is a config ERROR ("computed if not provided"
requires an EXPLICIT auto). Entries for known-but-inactive channels are
ignored, so one block can serve source/channel variants. x is scaled;
y and meta stay RAW (eval metadata).

Every scaling is affine, (value - shift) / scale, NaN-propagating: obs
are scaled BEFORE the NaN->0 missingness fill, so a filled zero sits at
the declared midpoint (minmax) or mean (standardise) and the missing
flag disambiguates it. Values outside declared bounds scale slightly
outside [-1, 1] — no clipping, QC outliers stay visible.

The resolved record (autos filled with numbers) is what train.py writes
to run_dir/norm_stats.json and evaluation reuses. Records written before
this schema (flat method/stats, pre-2026-07-06) are not readable — those
runs predate the bounds ruling and its comparison line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from utils.normalise import StatsAccumulator

from experiments.cyclone_jax.data.inputs import CHANNEL_ORDER

_COORD_FIELDS = {'surface_coordinate':  ('lat', 'lon'),
                 'vertical_coordinate': ('level',),
                 'time_coordinate':     ('time',)}
_GROUPS = (*_COORD_FIELDS, 'variables')

# Coordinate-convention guard rails: the store's canonical convention is
# degrees with lon in -180..180 (sources/build.py stores source lon
# verbatim — a 0..360 source must be canonicalised at volume build, this
# layer only refuses to scale it silently).
_GEO_LIMITS = {'lat': (-90.0, 90.0), 'lon': (-180.0, 180.0)}


def _check_entry(entry, where, kinds=('minmax', 'standardise')):
    """Validate one field entry; return it with floats (or 'auto').

    ``kinds`` = the keyed forms this field admits ('minmax' | 'standardise'
    | 'scale'); 'auto' is always admitted.
    """
    if entry == 'auto':
        return 'auto'
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where}: entry must be a keyed dict or 'auto', "
                         f"got {entry!r}.")
    keys = frozenset(entry)
    if keys == {'min', 'max'} and 'minmax' in kinds:
        lo, hi = float(entry['min']), float(entry['max'])
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"{where}: need finite min < max, "
                             f"got [{lo!r}, {hi!r}].")
        return {'min': lo, 'max': hi}
    if keys == {'mean', 'std'} and 'standardise' in kinds:
        mean, std = float(entry['mean']), float(entry['std'])
        if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
            raise ValueError(f"{where}: need finite mean and std > 0, "
                             f"got mean {mean!r}, std {std!r}.")
        return {'mean': mean, 'std': std}
    if keys == {'scale'} and 'scale' in kinds:
        s = float(entry['scale'])
        if not (np.isfinite(s) and s > 0):
            raise ValueError(f"{where}: need finite scale > 0, got {s!r}.")
        return {'scale': s}
    forms = {'minmax': '{min, max}', 'standardise': '{mean, std}',
             'scale': '{scale}'}
    raise ValueError(f"{where}: the keyed pair selects the method — "
                     f"{' or '.join(forms[k] for k in kinds)} or 'auto', "
                     f"got keys {sorted(entry)}.")


def _check_geo(f, entry, where):
    """Declared lat/lon bounds must sit inside the geographic range."""
    lo, hi = entry['min'], entry['max']
    glo, ghi = _GEO_LIMITS[f]
    if lo < glo or hi > ghi:
        hint = (' — a 0..360-convention source? Canonicalise at volume '
                'build: ((lon + 180) % 360) - 180' if f == 'lon' else '')
        raise ValueError(f"{where}: {f} bounds [{lo:g}, {hi:g}] fall "
                         f"outside the geographic range "
                         f"[{glo:g}, {ghi:g}]{hint}.")


def _affine(entry) -> tuple[float, float]:
    """One resolved entry -> (shift, scale) for (v - shift) / scale.

    {min, max} -> (midpoint, half-width): exactly minmax_11. {mean, std}
    -> z-score. Auto-resolved entries whose split never observed the
    field carry NaN — those scale as identity (0, 1): such values are
    all-NaN raw, so the choice is inert (the record keeps the honest NaN).
    """
    if 'min' in entry:
        lo, hi = float(entry['min']), float(entry['max'])
        shift, scale = (lo + hi) / 2.0, (hi - lo) / 2.0
    else:
        shift, scale = float(entry['mean']), float(entry['std'])
    if not (np.isfinite(shift) and np.isfinite(scale) and scale > 0):
        return 0.0, 1.0
    return shift, scale


@dataclass(frozen=True)
class NormSpec:
    """Materialised normalisation: concrete per-field scalings.

    Built by NormPolicy.materialise (or from_json for a saved run) —
    construct directly only in tests. ``stats`` is the RESOLVED grouped
    record (module docstring shape, autos filled with numbers) and THE
    thing train.py logs.
    """
    channels: tuple[str, ...]          # obs column alignment (InputSpec)
    stats:    Mapping = field(repr=False)

    def __post_init__(self):
        missing_groups = set(_GROUPS) - set(self.stats)
        if missing_groups:
            raise ValueError(f"stats missing group(s) "
                             f"{sorted(missing_groups)}.")
        v = self.stats['variables']
        missing = set(self.channels) - set(v)
        if missing:
            raise ValueError(f"stats have no variables entry for "
                             f"channel(s) {sorted(missing)}.")
        shift, scale = zip(*(_affine(v[c]) for c in self.channels))
        object.__setattr__(self, '_obs_shift', np.asarray(shift, np.float32))
        object.__setattr__(self, '_obs_scale', np.asarray(scale, np.float32))
        object.__setattr__(self, '_level', _affine(
            self.stats['vertical_coordinate']['level']))
        s = float(self.stats['time_coordinate']['time']['scale'])
        object.__setattr__(self, 'time_scale', s if s > 0 else 1.0)
        for f in ('lat', 'lon'):
            entry = self.stats['surface_coordinate'][f]
            lo, hi = float(entry['min']), float(entry['max'])
            _check_geo(f, {'min': lo, 'max': hi}, 'stats')
            object.__setattr__(self, f'_{f}', (lo, hi))

    # ------------------------------------------------------------------
    # Application (sampler.Loader calls these; see its docstring for WHERE)
    # ------------------------------------------------------------------

    def obs(self, vals) -> np.ndarray:
        """Scale a raw (n, C) obs matrix. NaN-propagating — call BEFORE
        build_missingness, so zero-fill lands on the midpoint/mean."""
        return ((np.asarray(vals, np.float32) - self._obs_shift)
                / self._obs_scale).astype(np.float32)

    def apply_tail(self, x: dict) -> dict:
        """Scale lat/lon/time/level in place — AFTER station selection
        (haversine needs real degrees). Returns x for chaining."""
        for f in ('lat', 'lon'):
            lo, hi = getattr(self, f'_{f}')
            x[f] = ((x[f] - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)
        x['time'] = (x['time'] / self.time_scale).astype(np.float32)
        shift, scale = self._level
        x['level'] = ((x['level'] - shift) / scale).astype(np.float32)
        return x

    def invert_coords(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Normalised [-1, 1] lat/lon back to degrees.

        Inverse of apply_tail's coordinate scaling — for figures that
        plot station positions from an already-normalised batch (the
        storm panel callback; y/meta coordinates stay raw and never need
        this).
        """
        la_lo, la_hi = self._lat
        lo_lo, lo_hi = self._lon
        lat = la_lo + (np.asarray(lat, np.float32) + 1.0) * (la_hi - la_lo) / 2.0
        lon = lo_lo + (np.asarray(lon, np.float32) + 1.0) * (lo_hi - lo_lo) / 2.0
        return lat.astype(np.float32), lon.astype(np.float32)

    @property
    def domain(self) -> dict:
        """The declared FOV, {'lat': [lo, hi], 'lon': [lo, hi]} — feeds
        network_sparsity and the figure extents (train/log.py)."""
        return {'lat': [self._lat[0], self._lat[1]],
                'lon': [self._lon[0], self._lon[1]]}

    def describe(self) -> str:
        """One banner line: per-method channel groups + coord bounds."""
        v = self.stats['variables']
        parts = []
        for kind, key in (('minmax', 'min'), ('standardise', 'mean')):
            named = [c for c in self.channels if key in v[c]]
            if named:
                parts.append(f"{kind}({', '.join(named)})")
        lv = self.stats['vertical_coordinate']['level']
        parts.append(f"lat [{self._lat[0]:g}, {self._lat[1]:g}]  "
                     f"lon [{self._lon[0]:g}, {self._lon[1]:g}]")
        parts.append(f"level {'minmax' if 'min' in lv else 'standardise'}")
        parts.append(f"time /{self.time_scale:g}s")
        return '  '.join(parts)

    # ------------------------------------------------------------------
    # Persistence (train.py -> run_dir/norm_stats.json; evaluate reads)
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        return {'channels': list(self.channels), 'stats': _plain(self.stats)}

    @classmethod
    def from_json(cls, d: Mapping) -> 'NormSpec':
        return cls(channels=tuple(d['channels']), stats=d['stats'])


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
    """Resolved ``normalise:`` block — declared entries plus 'auto'
    markers, before any train-split numbers exist."""
    entries: Mapping                   # grouped, validated (floats/'auto')

    @property
    def needs_stats(self) -> bool:
        """True when ANY entry is 'auto' — materialise then needs a
        train-split index set (one raw pass; interface.py supplies it)."""
        coords = any(self.entries[g][f] == 'auto'
                     for g, fields in _COORD_FIELDS.items() for f in fields)
        return coords or any(e == 'auto'
                             for e in self.entries['variables'].values())

    def materialise(self, loader, indices=None) -> NormSpec:
        """Concrete NormSpec — declared entries pass through; 'auto'
        entries are filled from one raw pass over loader samples at
        ``indices`` (the train split; interface.py decides which indices
        and raises the no-train-split error before this)."""
        channels = loader.inputs.channels
        variables = self.entries['variables']
        missing = [c for c in channels if c not in variables]
        if missing:
            raise ValueError(
                f"active channel(s) {missing} have no normalise.variables "
                f"entry — declare {{min, max}}/{{mean, std}} bounds or an "
                f"explicit 'auto' (config error, not a silent fallback).")
        resolved = {g: {f: self.entries[g][f] for f in fields}
                    for g, fields in _COORD_FIELDS.items()}
        resolved['variables'] = {c: variables[c] for c in channels}

        if self.needs_stats:
            if indices is None or not len(indices):
                raise ValueError("'auto' normalise entries need a non-empty "
                                 "index set to compute from.")
            if loader.norms is not None:
                raise RuntimeError("stats must be computed on RAW samples — "
                                   "loader already has norms attached.")
            stats = _collect_stats(loader, indices, channels)
            obs = stats['obs']
            for c in channels:
                if resolved['variables'][c] == 'auto':
                    resolved['variables'][c] = {'mean': obs[c]['mean'],
                                                'std':  obs[c]['std']}
            if resolved['vertical_coordinate']['level'] == 'auto':
                resolved['vertical_coordinate']['level'] = {
                    'mean': stats['level']['mean'],
                    'std':  stats['level']['std']}
            if resolved['time_coordinate']['time'] == 'auto':
                resolved['time_coordinate']['time'] = stats['time']
            for f in ('lat', 'lon'):
                observed = stats[f]
                if resolved['surface_coordinate'][f] == 'auto':
                    resolved['surface_coordinate'][f] = dict(observed)
                else:
                    # the pass ran anyway — declared bounds must cover the
                    # observed range, or coords silently leave [-1, 1]
                    d = resolved['surface_coordinate'][f]
                    lo, hi = float(observed['min']), float(observed['max'])
                    if np.isfinite(lo) and np.isfinite(hi) \
                            and (lo < d['min'] or hi > d['max']):
                        raise ValueError(
                            f"observed {f} range [{lo:g}, {hi:g}] exceeds "
                            f"the declared surface_coordinate bounds "
                            f"[{d['min']:g}, {d['max']:g}] — coords would "
                            f"scale outside [-1, 1]; widen the bounds or "
                            f"fix the data.")
        return NormSpec(channels=channels, stats=resolved)


def _collect_stats(loader, indices, channels) -> dict:
    """One raw pass over the given fixes -> per-field stats for 'auto'."""
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
        return {'mean': res['mean'][j], 'std': res['std'][j],
                'min': res['min'][j], 'max': res['max'][j]}

    return {
        'obs':   {c: _entry(obs, j) for j, c in enumerate(channels)},
        'level': _entry(tail, 0),
        'time':  {'scale': float(max(abs(tail['min'][1]), 1.0))},
        'lat':   {'min': tail['min'][2], 'max': tail['max'][2]},
        'lon':   {'min': tail['min'][3], 'max': tail['max'][3]},
    }


def resolve_normalise(config: dict) -> NormPolicy | None:
    """Data-config ``normalise:`` block -> NormPolicy (None = raw).

    Guards the grouped surface (module docstring): the four groups, exact
    coordinate field sets, keyed-pair entries, geographic limits on
    declared lat/lon, and variables names against the canonical channel
    union (typo guard — ACTIVE coverage is checked at materialise, where
    the resolved channel set exists).
    """
    if config.get('domain'):
        raise ValueError("the top-level domain: block moved into "
                         "normalise.surface_coordinate (lat/lon "
                         "{min, max}) — physical-bounds schema, "
                         "2026-07-06.")
    block = config.get('normalise')
    if not block:
        return None
    unknown = set(block) - set(_GROUPS)
    if unknown:
        raise ValueError(f"unknown normalise group(s) {sorted(unknown)} — "
                         f"the block is grouped as {sorted(_GROUPS)} "
                         f"(the flat method/stats form was replaced by "
                         f"per-field physical bounds, 2026-07-06).")
    missing = set(_GROUPS) - set(block)
    if missing:
        raise ValueError(f"normalise block missing group(s) "
                         f"{sorted(missing)} — every group is declared "
                         f"explicitly (entries may be 'auto').")

    entries = {}
    for g, fields in _COORD_FIELDS.items():
        sub = block[g] or {}
        if set(sub) != set(fields):
            raise ValueError(f"normalise.{g} must declare exactly "
                             f"{sorted(fields)}, got {sorted(sub)}.")
        kinds = {'lat': ('minmax',), 'lon': ('minmax',),
                 'level': ('minmax', 'standardise'), 'time': ('scale',)}
        entries[g] = {f: _check_entry(sub[f], f'normalise.{g}.{f}',
                                      kinds=kinds[f]) for f in fields}
        for f in fields:
            if f in _GEO_LIMITS and entries[g][f] != 'auto':
                _check_geo(f, entries[g][f], f'normalise.{g}')

    variables = block['variables'] or {}
    if not variables:
        raise ValueError("normalise.variables is empty — every active "
                         "channel needs an entry.")
    unknown = set(variables) - set(CHANNEL_ORDER)
    if unknown:
        raise ValueError(f"normalise.variables name(s) {sorted(unknown)} "
                         f"are not in the channel union {CHANNEL_ORDER}.")
    entries['variables'] = {c: _check_entry(e, f'normalise.variables.{c}')
                            for c, e in variables.items()}
    return NormPolicy(entries=entries)
