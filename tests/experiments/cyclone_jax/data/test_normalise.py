"""
Tests for cyclone_jax data/normalise.py — the physical-bounds NormSpec
layer (2026-07-06 schema): grouped keyed-dict resolve, per-field method
selection, explicit-auto train-split fill, application inside the Loader
(obs pre-fill, tail post-selection), json round-trip, and the guards.
Stats mechanics (accumulator) are covered in tests/utils/test_normalise.
"""

import numpy as np
import pytest

from experiments.cyclone_jax.data.inputs import resolve_input
from experiments.cyclone_jax.data.targets import resolve_target
from experiments.cyclone_jax.data.sampler import Loader
from experiments.cyclone_jax.data.normalise import (
    NormSpec, resolve_normalise,
)

# fixture library: station lat 0..30, lon -100..-30 (matches the real
# library's verified coverage); default input spec = the 7-channel union
COORDS = {
    'surface_coordinate':  {'lat': {'min': 0.0, 'max': 30.0},
                            'lon': {'min': -100.0, 'max': -30.0}},
    'vertical_coordinate': {'level': {'min': 70000.0, 'max': 108000.0}},
    'time_coordinate':     {'time': {'scale': 10800.0}},
}
DECLARED_VARS = {
    'station_pressure': {'min': 87000.0, 'max': 105000.0},
    'slp':              {'min': 87000.0, 'max': 105000.0},
    'air_temp':         {'min': 193.0, 'max': 333.0},
    'dewpoint':         {'mean': 290.0, 'std': 8.0},      # z-score entry
    'sst':              {'mean': 300.0, 'std': 3.0},
    'u_wind':           {'min': -115.0, 'max': 115.0},
    'v_wind':           {'min': -115.0, 'max': 115.0},
}

DECLARED = {'normalise': {**COORDS, 'variables': dict(DECLARED_VARS)}}
# same block with two obs channels + every coordinate left to the split
AUTO = {'normalise': {
    'surface_coordinate':  {'lat': 'auto', 'lon': 'auto'},
    'vertical_coordinate': {'level': 'auto'},
    'time_coordinate':     {'time': 'auto'},
    'variables': {**{c: 'auto' for c in ('slp', 'air_temp')},
                  **{c: v for c, v in DECLARED_VARS.items()
                     if c not in ('slp', 'air_temp')}},
}}


@pytest.fixture(scope='module')
def raw_loader(library):
    return Loader(library, resolve_input({}), resolve_target({}))


@pytest.fixture(scope='module')
def spec(raw_loader):
    """Fully declared — materialises with NO data pass."""
    return resolve_normalise(DECLARED).materialise(raw_loader)


@pytest.fixture(scope='module')
def auto_spec(raw_loader):
    """Auto entries filled over every fix (the fixture's 'train split')."""
    policy = resolve_normalise(AUTO)
    return policy.materialise(raw_loader, np.arange(len(raw_loader)))


@pytest.fixture(scope='module')
def norm_loader(library, spec):
    return Loader(library, resolve_input({}), resolve_target({}),
                  norms=spec)


# ---------------------------------------------------------------------------
# resolve_normalise — the grouped keyed-dict surface
# ---------------------------------------------------------------------------

class TestResolve:

    def test_absent_block_is_none(self):
        assert resolve_normalise({}) is None

    def test_old_flat_form_rejected_with_hint(self):
        with pytest.raises(ValueError, match='physical bounds'):
            resolve_normalise({'normalise': {'method': 'standardise',
                                             'stats': 'auto'}})

    def test_old_domain_block_rejected_with_hint(self):
        with pytest.raises(ValueError, match='surface_coordinate'):
            resolve_normalise({**DECLARED, 'domain': {'lat': [0, 35]}})

    def test_missing_group_raises(self):
        block = {k: v for k, v in DECLARED['normalise'].items()
                 if k != 'time_coordinate'}
        with pytest.raises(ValueError, match='time_coordinate'):
            resolve_normalise({'normalise': block})

    def test_wrong_coord_fields_raise(self):
        block = dict(DECLARED['normalise'],
                     surface_coordinate={'lat': {'min': 0, 'max': 30}})
        with pytest.raises(ValueError, match='lon'):
            resolve_normalise({'normalise': block})

    def test_needs_stats_flag(self):
        assert resolve_normalise(AUTO).needs_stats
        assert not resolve_normalise(DECLARED).needs_stats

    def test_mixed_key_entry_raises(self):
        block = dict(DECLARED['normalise'],
                     variables=dict(DECLARED_VARS,
                                    slp={'min': 0.0, 'std': 1.0}))
        with pytest.raises(ValueError, match='keyed pair'):
            resolve_normalise({'normalise': block})

    def test_inverted_minmax_raises(self):
        block = dict(DECLARED['normalise'],
                     variables=dict(DECLARED_VARS,
                                    slp={'min': 9.0, 'max': 1.0}))
        with pytest.raises(ValueError, match='min < max'):
            resolve_normalise({'normalise': block})

    def test_nonpositive_std_raises(self):
        block = dict(DECLARED['normalise'],
                     variables=dict(DECLARED_VARS,
                                    sst={'mean': 300.0, 'std': 0.0}))
        with pytest.raises(ValueError, match='std > 0'):
            resolve_normalise({'normalise': block})

    def test_time_takes_scale_only(self):
        block = dict(DECLARED['normalise'],
                     time_coordinate={'time': {'min': -1e4, 'max': 0.0}})
        with pytest.raises(ValueError, match='scale'):
            resolve_normalise({'normalise': block})

    def test_lat_rejects_mean_std(self):
        block = dict(DECLARED['normalise'],
                     surface_coordinate={'lat': {'mean': 15.0, 'std': 5.0},
                                         'lon': COORDS['surface_coordinate']['lon']})
        with pytest.raises(ValueError, match='lat'):
            resolve_normalise({'normalise': block})

    def test_unknown_variable_name_raises(self):
        block = dict(DECLARED['normalise'],
                     variables=dict(DECLARED_VARS,
                                    air_tmp={'min': 0.0, 'max': 1.0}))
        with pytest.raises(ValueError, match='air_tmp'):
            resolve_normalise({'normalise': block})

    def test_0360_lon_bounds_raise_with_hint(self):
        block = dict(DECLARED['normalise'],
                     surface_coordinate={'lat': COORDS['surface_coordinate']['lat'],
                                         'lon': {'min': 260.0, 'max': 330.0}})
        with pytest.raises(ValueError, match='-180'):
            resolve_normalise({'normalise': block})


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------

class TestMaterialise:

    def test_declared_needs_no_indices(self, spec, raw_loader):
        assert spec.channels == raw_loader.inputs.channels

    def test_declared_passthrough(self, spec):
        v = spec.stats['variables']
        assert v['slp'] == {'min': 87000.0, 'max': 105000.0}
        assert v['sst'] == {'mean': 300.0, 'std': 3.0}

    def test_active_channel_without_entry_raises(self, raw_loader):
        block = dict(DECLARED['normalise'],
                     variables={c: v for c, v in DECLARED_VARS.items()
                                if c != 'slp'})
        with pytest.raises(ValueError, match='slp'):
            resolve_normalise({'normalise': block}).materialise(raw_loader)

    def test_inactive_entries_ignored(self, library):
        """A shared block may carry entries for channels the scenario
        filtered out — the record subsets to the active set."""
        loader = Loader(library,
                        resolve_input({'channels': ['slp', 'air_temp']}),
                        resolve_target({}))
        s = resolve_normalise(DECLARED).materialise(loader)
        assert set(s.stats['variables']) == {'slp', 'air_temp'}

    def test_auto_needs_indices(self, raw_loader):
        with pytest.raises(ValueError, match='index set'):
            resolve_normalise(AUTO).materialise(raw_loader)

    def test_auto_refuses_normalised_loader(self, norm_loader):
        with pytest.raises(RuntimeError, match='RAW'):
            resolve_normalise(AUTO).materialise(norm_loader, np.arange(4))

    def test_auto_fills_mean_std(self, auto_spec, raw_loader):
        """auto == nan-stats over raw samples (mask-restored NaN)."""
        vals = []
        for i in range(len(raw_loader)):
            x = raw_loader.build(i)['x']
            vals.append(np.where(x['missing'], x['obs'], np.nan))
        v = np.concatenate(vals)
        j = auto_spec.channels.index('slp')
        s = auto_spec.stats['variables']['slp']
        assert set(s) == {'mean', 'std'}
        np.testing.assert_allclose(s['mean'], np.nanmean(v[:, j]), rtol=1e-6)
        np.testing.assert_allclose(s['std'], np.nanstd(v[:, j]), rtol=1e-4)

    def test_auto_coords_are_observed_minmax(self, auto_spec):
        lat = auto_spec.stats['surface_coordinate']['lat']
        assert set(lat) == {'min', 'max'}
        assert 0.0 <= lat['min'] < lat['max'] <= 30.0

    def test_auto_time_scale_covers_the_lookback(self, auto_spec,
                                                 raw_loader):
        dts = [raw_loader.build(i)['x']['time'].min()
               for i in range(0, len(raw_loader), 5)]
        assert auto_spec.time_scale >= abs(min(dts))

    def test_declared_coords_must_cover_observed_when_pass_runs(
            self, raw_loader):
        """A stats pass (any auto) validates declared lat/lon coverage."""
        lats = np.concatenate([raw_loader.build(i)['x']['lat']
                               for i in range(8)])
        lo, hi = float(lats.min()), float(lats.max())
        block = dict(AUTO['normalise'])
        block['surface_coordinate'] = {
            'lat': {'min': lo + (hi - lo) / 2, 'max': hi},
            'lon': 'auto'}
        with pytest.raises(ValueError, match='declared'):
            resolve_normalise({'normalise': block}).materialise(
                raw_loader, np.arange(8))

    def test_json_round_trip(self, spec):
        clone = NormSpec.from_json(spec.to_json())
        assert clone.channels == spec.channels
        assert clone.time_scale == spec.time_scale
        v = np.array([[1.0e5] * len(spec.channels)], np.float32)
        np.testing.assert_array_equal(spec.obs(v), clone.obs(v))

    def test_json_is_serialisable(self, auto_spec):
        import json
        json.dumps(auto_spec.to_json())              # numpy fully stripped

    def test_domain_property(self, spec):
        assert spec.domain == {'lat': [0.0, 30.0], 'lon': [-100.0, -30.0]}

    def test_describe_names_methods(self, spec):
        d = spec.describe()
        assert 'minmax(' in d and 'standardise(' in d
        assert 'slp' in d and 'sst' in d and 'time /10800s' in d


# ---------------------------------------------------------------------------
# NormSpec direct-construction guards (the from_json path)
# ---------------------------------------------------------------------------

class TestSpecGuards:

    def test_missing_channel_raises(self, spec):
        stats = spec.to_json()['stats']
        stats['variables'].pop('sst')
        with pytest.raises(ValueError, match='sst'):
            NormSpec(channels=spec.channels, stats=stats)

    def test_missing_group_raises(self, spec):
        stats = spec.to_json()['stats']
        stats.pop('time_coordinate')
        with pytest.raises(ValueError, match='time_coordinate'):
            NormSpec(channels=spec.channels, stats=stats)

    def test_lon_0360_stats_raise_with_hint(self, spec):
        stats = spec.to_json()['stats']
        stats['surface_coordinate']['lon'] = {'min': 260.0, 'max': 330.0}
        with pytest.raises(ValueError, match='180'):
            NormSpec(channels=spec.channels, stats=stats)

    def test_bad_lat_stats_raise(self, spec):
        stats = spec.to_json()['stats']
        stats['surface_coordinate']['lat'] = {'min': -95.0, 'max': 30.0}
        with pytest.raises(ValueError, match='lat'):
            NormSpec(channels=spec.channels, stats=stats)


# ---------------------------------------------------------------------------
# Application inside the Loader
# ---------------------------------------------------------------------------

class TestAppliedLoader:

    def test_x_scaled_y_raw(self, raw_loader, norm_loader):
        raw, norm = raw_loader.build(5), norm_loader.build(5)
        for f in ('target', 'sid', 'lat', 'lon', 'time'):
            assert norm['y'][f] == raw['y'][f]       # y NEVER normalised
        assert not np.allclose(norm['x']['obs'], raw['x']['obs'])

    def test_coords_in_unit_range(self, norm_loader):
        x = norm_loader.build(3)['x']
        for f in ('lat', 'lon'):
            assert x[f].min() >= -1.0 - 1e-6 and x[f].max() <= 1.0 + 1e-6

    def test_time_in_minus_one_zero(self, norm_loader):
        t = norm_loader.build(3)['x']['time']
        assert t.min() >= -1.0 - 1e-6 and t.max() <= 0.0

    def test_missing_fill_sits_at_zero(self, norm_loader):
        """Scaling runs BEFORE the fill: filled values are exactly 0 —
        the declared midpoint (minmax) or mean (standardise); the missing
        flag is what disambiguates them from a real mid-range value."""
        x = norm_loader.build(5)['x']
        assert not x['missing'].all()                # fixture has gaps
        np.testing.assert_array_equal(x['obs'][~x['missing']], 0.0)

    def test_minmax_channel_lands_in_declared_band(self, norm_loader):
        """Observed slp scales inside [-1, 1] iff the raw values respect
        the declared physical bounds (the fixture's do)."""
        j = norm_loader.norms.channels.index('slp')
        vals = []
        for i in range(0, len(norm_loader), 3):
            x = norm_loader.build(i)['x']
            vals.append(x['obs'][x['missing'][:, j], j])
        v = np.concatenate(vals)
        assert len(v) and v.min() >= -1.0 - 1e-6 and v.max() <= 1.0 + 1e-6

    def test_signed_symmetric_wind_keeps_zero(self, spec):
        """u/v bounds [-115, 115]: raw 0 m/s must scale to exactly 0."""
        j = spec.channels.index('u_wind')
        v = np.zeros((1, len(spec.channels)), np.float32)
        assert spec.obs(v)[0, j] == 0.0

    def test_mask_and_dtypes_unchanged(self, raw_loader, norm_loader):
        raw, norm = raw_loader.build(7)['x'], norm_loader.build(7)['x']
        np.testing.assert_array_equal(raw['missing'], norm['missing'])
        np.testing.assert_array_equal(raw['id'], norm['id'])
        for f in ('lat', 'lon', 'level', 'time', 'obs'):
            assert norm[f].dtype == np.float32

    def test_invert_coords_roundtrip(self, raw_loader, norm_loader, spec):
        """Normalised batch coords -> degrees (storm-panel plotting)."""
        raw, norm = raw_loader.build(3)['x'], norm_loader.build(3)['x']
        lat, lon = spec.invert_coords(norm['lat'], norm['lon'])
        np.testing.assert_allclose(lat, raw['lat'], atol=1e-3)
        np.testing.assert_allclose(lon, raw['lon'], atol=1e-3)

    def test_station_selection_unaffected(self, library, spec):
        """max_stations picks the SAME stations with norms on (haversine
        runs on real degrees — tail scaling is post-selection)."""
        kw = dict(selection='max_stations', max_stations=32)
        raw = Loader(library, resolve_input(kw), resolve_target({}))
        norm = Loader(library, resolve_input(kw), resolve_target({}),
                      norms=spec)
        traw, tnorm = raw.build(9)['x']['time'], norm.build(9)['x']['time']
        np.testing.assert_allclose(tnorm * spec.time_scale, traw, atol=1e-3)
