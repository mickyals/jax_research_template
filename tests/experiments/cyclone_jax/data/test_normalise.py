"""
Tests for cyclone_jax data/normalise.py — the NormSpec policy layer:
resolve, train-split stats materialisation, application inside the Loader
(obs pre-fill, tail post-selection), json round-trip, and the guards.
Mechanics (registry, accumulator) are covered in tests/utils/test_normalise.
"""

import numpy as np
import pytest

from experiments.cyclone_jax.data.inputs import resolve_input
from experiments.cyclone_jax.data.targets import resolve_target
from experiments.cyclone_jax.data.sampler import Loader
from experiments.cyclone_jax.data.normalise import (
    NormSpec, resolve_normalise,
)

CFG = {'normalise': {'method': 'standardise', 'stats': 'auto'}}


@pytest.fixture(scope='module')
def raw_loader(library):
    return Loader(library, resolve_input({}), resolve_target({}))


@pytest.fixture(scope='module')
def spec(raw_loader):
    """Stats over every fix (the fixture's 'train split')."""
    policy = resolve_normalise(CFG)
    return policy.materialise(raw_loader, np.arange(len(raw_loader)))


@pytest.fixture(scope='module')
def norm_loader(library, spec):
    return Loader(library, resolve_input({}), resolve_target({}),
                  norms=spec)


# ---------------------------------------------------------------------------
# resolve_normalise
# ---------------------------------------------------------------------------

class TestResolve:

    def test_absent_block_is_none(self):
        assert resolve_normalise({}) is None

    def test_method_none_is_none(self):
        assert resolve_normalise({'normalise': {'method': 'none'}}) is None

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match='zscore'):
            resolve_normalise({'normalise': {'method': 'zscore'}})

    def test_auto_flag(self):
        assert resolve_normalise(CFG).auto

    def test_inline_stats_not_auto(self, spec):
        cfg = {'normalise': {'method': 'standardise',
                             'stats': spec.to_json()['stats']}}
        assert not resolve_normalise(cfg).auto

    def test_domain_parsed_and_validated(self):
        p = resolve_normalise({**CFG, 'domain': {'lat': [0, 35]}})
        assert p.domain == {'lat': (0.0, 35.0)}
        with pytest.raises(ValueError, match='alt'):
            resolve_normalise({**CFG, 'domain': {'alt': [0, 1]}})


# ---------------------------------------------------------------------------
# Materialisation (train-split stats)
# ---------------------------------------------------------------------------

class TestMaterialise:

    def test_channels_align_with_input_spec(self, spec, raw_loader):
        assert spec.channels == raw_loader.inputs.channels

    def test_obs_stats_match_manual_nan_stats(self, spec, raw_loader):
        """Spec stats == nan-stats over raw samples (mask-restored NaN)."""
        vals = []
        for i in range(len(raw_loader)):
            x = raw_loader.build(i)['x']
            vals.append(np.where(x['missing'], x['obs'], np.nan))
        v = np.concatenate(vals)
        j = spec.channels.index('slp')
        s = spec.stats['obs']['slp']
        # accumulator runs float64; the manual reference is float32 samples
        np.testing.assert_allclose(s['mean'], np.nanmean(v[:, j]), rtol=1e-6)
        np.testing.assert_allclose(s['std'], np.nanstd(v[:, j]), rtol=1e-4)

    def test_time_scale_covers_the_lookback(self, spec, raw_loader):
        dts = [raw_loader.build(i)['x']['time'].min()
               for i in range(0, len(raw_loader), 5)]
        assert spec.time_scale >= abs(min(dts))

    def test_domain_overrides_observed_coord_range(self, raw_loader):
        p = resolve_normalise({**CFG, 'domain': {'lat': [-90, 90]}})
        s = p.materialise(raw_loader, np.arange(8))
        assert s.stats['lat'] == {'min': -90.0, 'max': 90.0}

    def test_auto_needs_indices(self, raw_loader):
        with pytest.raises(ValueError, match='index set'):
            resolve_normalise(CFG).materialise(raw_loader)

    def test_stats_pass_refuses_normalised_loader(self, norm_loader):
        with pytest.raises(RuntimeError, match='RAW'):
            resolve_normalise(CFG).materialise(norm_loader, np.arange(4))

    def test_inline_missing_channel_raises(self, spec):
        stats = spec.to_json()['stats']
        stats['obs'].pop('sst')
        with pytest.raises(ValueError, match='sst'):
            NormSpec(method='standardise', channels=spec.channels,
                     stats=stats)

    def test_json_round_trip(self, spec):
        clone = NormSpec.from_json(spec.to_json())
        assert clone.channels == spec.channels
        assert clone.time_scale == spec.time_scale
        v = np.array([[1.0e5] * len(spec.channels)], np.float32)
        np.testing.assert_array_equal(spec.obs(v), clone.obs(v))

    def test_json_is_serialisable(self, spec):
        import json
        json.dumps(spec.to_json())                   # numpy fully stripped


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

    def test_missing_fill_sits_at_channel_mean(self, norm_loader):
        """Standardised mean = 0; zero-fill therefore == mean-fill."""
        x = norm_loader.build(5)['x']
        assert not x['missing'].all()                # fixture has gaps
        np.testing.assert_array_equal(x['obs'][~x['missing']], 0.0)

    def test_observed_values_roughly_standardised(self, norm_loader):
        vals = []
        for i in range(0, len(norm_loader), 3):
            x = norm_loader.build(i)['x']
            vals.append(np.where(x['missing'], x['obs'], np.nan))
        v = np.concatenate(vals)
        assert abs(np.nanmean(v)) < 0.2
        assert 0.5 < np.nanstd(v) < 2.0

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
