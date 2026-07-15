"""
experiments/cyclone_jax/data/inputs.py

InputSpec — the declarative INPUT side of a sample (mirror of targets.py).

Owns the channel vocabulary: which volume columns each source contributes,
how derived channels (u/v wind) are computed, and the uniform channel union
the model sees. Combining sources yields one fixed-length obs vector per
token — channels a source lacks (e.g. sst on land) stay NaN and are carried
by the missingness mask, so no per-source branching survives past assembly.

resolve_input builds the spec from the data config block; policy lives in
yaml, mechanics live here, and the sampler consumes the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from utils.geoscience.met_conversions import wind_to_components


# ---------------------------------------------------------------------------
# Per-source channel schemas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DerivedChannels:
    """Channels computed from volume columns rather than read directly.

    compute(*columns) -> tuple of channel arrays, NaN-propagating (missing
    inputs stay missing so build_missingness sees them).
    """
    channels: tuple[str, ...]   # channel names produced, in compute's order
    columns:  tuple[str, ...]   # volume columns consumed
    compute:  Callable


@dataclass(frozen=True)
class SourceSchema:
    """One source's contribution to the channel union."""
    direct:  Mapping[str, str]              # volume column -> channel name
    derived: tuple[DerivedChannels, ...] = ()

    @property
    def channels(self) -> tuple[str, ...]:
        out = tuple(self.direct.values())
        for d in self.derived:
            out += d.channels
        return out


# Meteorological (speed, FROM-direction) -> (u, v); calm -> (0, 0), NaN
# speed (or NaN direction at speed > 0) propagates NaN.
WIND_UV = DerivedChannels(channels=('u_wind', 'v_wind'),
                          columns=('wind_speed', 'wind_dir'),
                          compute=wind_to_components)

# 'upper' (sky-arcanum) joins when its pressure-coordinate encoding is
# designed; its volume stores u_wind/v_wind directly (no derivation).
SOURCE_SCHEMAS = {
    'land':   SourceSchema(direct={'station_pressure': 'station_pressure',
                                   'slp': 'slp', 'air_temp': 'air_temp',
                                   'dewpoint': 'dewpoint'},
                           derived=(WIND_UV,)),
    'marine': SourceSchema(direct={'slp': 'slp', 'air_temp': 'air_temp',
                                   'dewpoint': 'dewpoint', 'sst': 'sst'},
                           derived=(WIND_UV,)),
}

# Canonical channel ordering — every union is a subsequence of this, so the
# obs column layout is stable under source subsets. Extend when upper joins.
CHANNEL_ORDER = ('station_pressure', 'slp', 'air_temp', 'dewpoint', 'sst',
                 'u_wind', 'v_wind')

DEFAULT_SOURCE_ID = {'land': -1.0, 'upper': 0.0, 'marine': 1.0}


def union_channels(sources) -> tuple[str, ...]:
    """The uniform channel vector for a source combination, in canonical
    order. Raises for unknown sources or channels missing from CHANNEL_ORDER
    (a schema/order mismatch is a bug, not a config error)."""
    contributed = set()
    for s in sources:
        if s not in SOURCE_SCHEMAS:
            raise ValueError(f"unknown source {s!r} — schemas exist for "
                             f"{sorted(SOURCE_SCHEMAS)}")
        contributed.update(SOURCE_SCHEMAS[s].channels)
    unordered = contributed - set(CHANNEL_ORDER)
    if unordered:
        raise RuntimeError(f"channels {sorted(unordered)} missing from "
                           f"CHANNEL_ORDER — extend the canonical ordering.")
    return tuple(c for c in CHANNEL_ORDER if c in contributed)


# ---------------------------------------------------------------------------
# InputSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputSpec:
    """Resolved input policy: which sources, the resulting channel union,
    station selection, and the fixed token budget.

    Built by resolve_input from the data config — construct directly only
    in tests.
    """
    sources:      tuple[str, ...]
    channels:     tuple[str, ...]
    source_id:    Mapping[str, float]
    selection:    str                   # 'all' | 'max_stations'
    max_stations: int | None
    pad_to:       int

    def __post_init__(self):
        if not self.sources:
            raise ValueError("at least one source is required.")
        for s in self.sources:
            if s not in SOURCE_SCHEMAS:
                raise ValueError(f"unknown source {s!r} — schemas exist for "
                                 f"{sorted(SOURCE_SCHEMAS)}")
            if s not in self.source_id:
                raise ValueError(f"source_id has no code for {s!r}.")
        if self.selection not in ('all', 'max_stations'):
            raise ValueError(f"selection must be 'all' or 'max_stations', "
                             f"got {self.selection!r}")
        if self.selection == 'max_stations' and not self.max_stations:
            raise ValueError("selection='max_stations' requires max_stations.")
        if self.pad_to <= 0:
            raise ValueError(f"pad_to must be positive, got {self.pad_to}.")

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def channel_index(self) -> dict[str, int]:
        """{channel name: obs column index} for assembly and notebooks."""
        return {c: j for j, c in enumerate(self.channels)}


def select_channels(sources, selected) -> tuple[str, ...]:
    """Filter the sources' channel union down to ``selected`` names.

    GLOBAL selection (one list for all sources — per-source absence is
    already the missingness mask's job): every name must exist in the
    union the chosen sources contribute, otherwise the channel would be
    all-NaN and the config is lying about its inputs. The result keeps
    CANONICAL order regardless of the yaml's listing order.
    """
    if not selected:
        raise ValueError("channels: must name at least one channel — "
                         "omit the key entirely for the full union.")
    union = union_channels(sources)
    bad = set(selected) - set(union)
    if bad:
        raise ValueError(f"channels {sorted(bad)} not contributed by "
                         f"sources {list(sources)} — union: {list(union)}")
    keep = set(selected)
    return tuple(c for c in union if c in keep)


def resolve_input(config: dict) -> InputSpec:
    """Build the InputSpec from the data config block (configs/data/*.yaml).

    Keys read: sources, channels, selection, max_stations, pad_to,
    source_id — defaults match the v1 land+marine setup. ``channels``
    (optional) restricts the union to an explicit list (see
    select_channels); omitted = the full union.
    """
    sources = tuple(config.get('sources', ('land', 'marine')))
    selected = config.get('channels')
    max_stations = config.get('max_stations')
    return InputSpec(
        sources=sources,
        channels=(select_channels(sources, selected)
                  if selected is not None else union_channels(sources)),
        source_id=dict(config.get('source_id', DEFAULT_SOURCE_ID)),
        selection=config.get('selection', 'all'),
        max_stations=int(max_stations) if max_stations else None,
        pad_to=int(config.get('pad_to', 1536)),
    )
