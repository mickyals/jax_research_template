"""
Tests for experiments/sparse_obs_encoder/data/inputs.py and the swappable
input transforms (data/transforms/normalise.py + derived.py).

No real data files required — InputSpec is pure-declarative; the transforms
operate on small in-memory arrays / frames.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.sparse_obs_encoder.data.inputs import InputSpec, resolve_input
from experiments.sparse_obs_encoder.data.sources.insitu_land import DEFAULT_OBS_VARS
from experiments.sparse_obs_encoder.data.transforms.normalise import (
    NORMALISERS, get_normaliser,
)
from experiments.sparse_obs_encoder.data.transforms.derived import (
    DERIVED_VARS, resolve_fetch_vars, compute_derived,
)


# ---------------------------------------------------------------------------
# resolve_input
# ---------------------------------------------------------------------------

class TestResolveInput:

    def test_defaults_from_empty_config(self):
        spec = resolve_input({})
        assert spec.obs_vars == tuple(DEFAULT_OBS_VARS)
        assert spec.normalisation == 'minmax_01'
        assert spec.location_encoding == 'unit_circle'
        assert spec.obs_bounds is None
        assert spec.fov_lat == (0.0, 30.0)
        assert spec.fov_lon == (-100.0, -45.0)

    def test_reads_data_block_keys(self):
        spec = resolve_input({
            'obs_vars': ['air_temperature', 'wind_east', 'wind_north'],
            'obs_normalisation': 'minmax_11',
            'obs_bounds': {'air_temperature': [200, 330],
                           'wind_east': [-50, 50], 'wind_north': [-50, 50]},
            'location_encoding': 'domain',
            'fov_lat': [5.0, 25.0],
            'fov_lon': [-90.0, -50.0],
        })
        assert spec.obs_vars == ('air_temperature', 'wind_east', 'wind_north')
        assert spec.normalisation == 'minmax_11'
        assert spec.location_encoding == 'domain'
        assert spec.obs_bounds['wind_east'] == (-50, 50)
        assert spec.fov_lat == (5.0, 25.0)
        assert spec.fov_lon == (-90.0, -50.0)


# ---------------------------------------------------------------------------
# InputSpec construction guards
# ---------------------------------------------------------------------------

class TestInputSpecGuards:

    def test_empty_obs_vars_raises(self):
        with pytest.raises(ValueError, match='obs_vars must be non-empty'):
            InputSpec(obs_vars=())

    def test_invalid_normalisation_raises(self):
        with pytest.raises(ValueError, match='obs_normalisation'):
            InputSpec(normalisation='zscore')

    def test_invalid_location_encoding_raises(self):
        with pytest.raises(ValueError, match='location_encoding'):
            InputSpec(location_encoding='polar')


# ---------------------------------------------------------------------------
# InputSpec accessors
# ---------------------------------------------------------------------------

class TestInputSpecAccessors:

    def test_feature_dim(self):
        spec = InputSpec(obs_vars=('a', 'b', 'c'),
                         obs_bounds={'a': (0, 1), 'b': (0, 1), 'c': (0, 1)},
                         normalisation='minmax_01')
        # normalisation requires bounds keys to exist only when used; feature_dim
        # is purely structural.
        assert spec.feature_dim == 3

    def test_fetch_vars_expands_derived(self):
        spec = InputSpec(obs_vars=('air_temperature', 'wind_east', 'wind_north'))
        assert spec.fetch_vars == [
            'air_temperature', 'wind_speed', 'wind_from_direction',
        ]

    def test_bounds_arrays_none_when_unset(self):
        lo, hi = InputSpec().bounds_arrays()
        assert lo is None and hi is None

    def test_bounds_arrays_aligned_to_obs_vars(self):
        spec = InputSpec(obs_vars=('air_temperature', 'wind_speed'),
                         obs_bounds={'air_temperature': (200.0, 330.0),
                                     'wind_speed': (0.0, 115.0)})
        lo, hi = spec.bounds_arrays()
        assert np.allclose(lo, [200.0, 0.0])
        assert np.allclose(hi, [330.0, 115.0])

    def test_resolved_transforms(self):
        spec = InputSpec(normalisation='minmax_11', location_encoding='domain')
        assert spec.normaliser is get_normaliser('minmax_11')
        assert callable(spec.coord_encoder)
        assert callable(spec.coord_decoder)


# ---------------------------------------------------------------------------
# Normaliser registry
# ---------------------------------------------------------------------------

class TestNormalisers:

    def test_registry_members(self):
        assert set(NORMALISERS.names()) == {'MINMAX_01', 'MINMAX_11', 'STANDARDISE'}

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match='not a registered normaliser'):
            get_normaliser('zscore')

    def test_minmax_01(self):
        lo = np.array([0.0], dtype=np.float32)
        hi = np.array([10.0], dtype=np.float32)
        out = get_normaliser('minmax_01')(np.array([[5.0]], dtype=np.float32), lo, hi)
        assert np.allclose(out, 0.5, atol=1e-5)

    def test_minmax_11(self):
        lo = np.array([0.0], dtype=np.float32)
        hi = np.array([10.0], dtype=np.float32)
        out = get_normaliser('minmax_11')(np.array([[5.0]], dtype=np.float32), lo, hi)
        assert np.allclose(out, 0.0, atol=1e-5)

    def test_standardise(self):
        mean = np.array([5.0], dtype=np.float32)
        std  = np.array([2.0], dtype=np.float32)
        out = get_normaliser('standardise')(np.array([[7.0]], dtype=np.float32), mean, std)
        assert np.allclose(out, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Derived variables registry
# ---------------------------------------------------------------------------

class TestDerived:

    def test_registry_members(self):
        assert set(DERIVED_VARS.names()) == {'WIND_EAST', 'WIND_NORTH'}

    def test_resolve_fetch_vars_passthrough_and_dedup(self):
        # plain vars pass through; derived vars expand; sources deduped in order
        out = resolve_fetch_vars(['air_temperature', 'wind_east', 'wind_north'])
        assert out == ['air_temperature', 'wind_speed', 'wind_from_direction']

    def test_resolve_fetch_vars_plain_only(self):
        assert resolve_fetch_vars(['air_temperature', 'wind_speed']) == [
            'air_temperature', 'wind_speed',
        ]

    def test_compute_derived_wind_components(self):
        # wind 10 m/s FROM east (90°) → u = -10, v = 0
        df = pd.DataFrame({
            'wind_speed':          np.array([10.0], dtype=np.float32),
            'wind_from_direction': np.array([90.0], dtype=np.float32),
        })
        out = compute_derived(df, ['wind_east', 'wind_north'])
        assert np.allclose(out['wind_east'].to_numpy(), -10.0, atol=1e-4)
        assert np.allclose(out['wind_north'].to_numpy(), 0.0, atol=1e-4)

    def test_compute_derived_noop_without_derived(self):
        df = pd.DataFrame({'air_temperature': np.array([300.0], dtype=np.float32)})
        out = compute_derived(df, ['air_temperature'])
        assert out is df
