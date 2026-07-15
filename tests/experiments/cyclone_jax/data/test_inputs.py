"""
Tests for cyclone_jax inputs — the InputSpec contract: canonical channel
union across source combinations, config resolution + defaults, validation
guards, and the derived-wind schema semantics.
"""

import dataclasses

import numpy as np
import pytest

from experiments.cyclone_jax.data.inputs import (
    CHANNEL_ORDER, DEFAULT_SOURCE_ID, SOURCE_SCHEMAS, WIND_UV,
    InputSpec, resolve_input, select_channels, union_channels,
)


# ---------------------------------------------------------------------------
# Channel union
# ---------------------------------------------------------------------------

class TestUnionChannels:

    def test_land_marine_is_full_canonical_order(self):
        assert union_channels(('land', 'marine')) == CHANNEL_ORDER

    def test_land_only_has_no_sst(self):
        ch = union_channels(('land',))
        assert 'sst' not in ch and 'station_pressure' in ch

    def test_marine_only_has_no_station_pressure(self):
        ch = union_channels(('marine',))
        assert 'station_pressure' not in ch and 'sst' in ch

    def test_order_is_canonical_subsequence(self):
        for sources in (('land',), ('marine',), ('marine', 'land')):
            ch = union_channels(sources)
            positions = [CHANNEL_ORDER.index(c) for c in ch]
            assert positions == sorted(positions)

    def test_source_order_does_not_matter(self):
        assert union_channels(('marine', 'land')) == \
            union_channels(('land', 'marine'))

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match='unknown source'):
            union_channels(('land', 'satellite'))

    def test_every_schema_channel_is_ordered(self):
        for schema in SOURCE_SCHEMAS.values():
            assert set(schema.channels) <= set(CHANNEL_ORDER)


# ---------------------------------------------------------------------------
# Derived wind schema
# ---------------------------------------------------------------------------

class TestWindDerived:

    def test_both_surface_sources_derive_wind(self):
        for name in ('land', 'marine'):
            assert WIND_UV in SOURCE_SCHEMAS[name].derived

    def test_compute_calm_and_nan_semantics(self):
        u, v = WIND_UV.compute(np.array([0.0, np.nan, 10.0]),
                               np.array([np.nan, 90.0, 0.0]))
        assert u[0] == 0.0 and v[0] == 0.0            # calm -> (0, 0)
        assert np.isnan(u[1]) and np.isnan(v[1])      # NaN speed propagates
        np.testing.assert_allclose([u[2], v[2]], [0.0, -10.0], atol=1e-5)


# ---------------------------------------------------------------------------
# InputSpec + resolve_input
# ---------------------------------------------------------------------------

def _cfg(**over):
    cfg = {'sources': ['land', 'marine'], 'selection': 'all',
           'max_stations': None, 'pad_to': 1536,
           'source_id': {'land': -1, 'upper': 0, 'marine': 1}}
    cfg.update(over)
    return cfg


class TestResolveInput:

    def test_resolves_data_yaml_shape(self):
        spec = resolve_input(_cfg())
        assert spec.sources == ('land', 'marine')
        assert spec.channels == CHANNEL_ORDER
        assert spec.selection == 'all' and spec.max_stations is None
        assert spec.pad_to == 1536
        assert spec.source_id['marine'] == 1.0

    def test_defaults_match_v1(self):
        spec = resolve_input({})
        assert spec.sources == ('land', 'marine')
        assert spec.pad_to == 1536
        assert spec.source_id == DEFAULT_SOURCE_ID

    def test_channel_accessors(self):
        spec = resolve_input(_cfg())
        assert spec.n_channels == len(CHANNEL_ORDER)
        assert spec.channel_index['sst'] == CHANNEL_ORDER.index('sst')

    def test_max_stations_selection(self):
        spec = resolve_input(_cfg(selection='max_stations', max_stations=256))
        assert spec.max_stations == 256

    def test_frozen(self):
        spec = resolve_input(_cfg())
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.pad_to = 64


class TestChannelSelection:

    def test_channels_key_filters_the_union(self):
        spec = resolve_input(_cfg(channels=['slp', 'u_wind', 'v_wind']))
        assert spec.channels == ('slp', 'u_wind', 'v_wind')
        assert spec.n_channels == 3

    def test_yaml_order_does_not_matter(self):
        # result keeps CANONICAL order regardless of listing order
        assert select_channels(('land', 'marine'),
                               ['v_wind', 'slp', 'u_wind']) \
            == ('slp', 'u_wind', 'v_wind')

    def test_full_union_listed_explicitly_is_identity(self):
        spec = resolve_input(_cfg(channels=list(CHANNEL_ORDER)))
        assert spec.channels == CHANNEL_ORDER

    def test_channel_outside_source_union_raises(self):
        # land contributes no sst — an all-NaN channel is a config lie
        with pytest.raises(ValueError, match='sst'):
            select_channels(('land',), ['slp', 'sst'])

    def test_unknown_channel_raises(self):
        with pytest.raises(ValueError, match='geopotential'):
            select_channels(('land', 'marine'), ['slp', 'geopotential'])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match='at least one'):
            resolve_input(_cfg(channels=[]))

    def test_omitted_key_gives_full_union(self):
        assert resolve_input(_cfg()).channels == CHANNEL_ORDER


class TestInputSpecGuards:

    def test_invalid_selection_raises(self):
        with pytest.raises(ValueError, match='selection'):
            resolve_input(_cfg(selection='nearest'))

    def test_max_stations_required(self):
        with pytest.raises(ValueError, match='max_stations'):
            resolve_input(_cfg(selection='max_stations'))

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match='unknown source'):
            resolve_input(_cfg(sources=['land', 'satellite']))

    def test_empty_sources_raises(self):
        with pytest.raises(ValueError, match='at least one source'):
            resolve_input(_cfg(sources=[]))

    def test_source_id_must_cover_sources(self):
        with pytest.raises(ValueError, match='source_id'):
            resolve_input(_cfg(source_id={'land': -1}))

    def test_pad_to_positive(self):
        with pytest.raises(ValueError, match='pad_to'):
            resolve_input(_cfg(pad_to=0))
