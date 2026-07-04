"""
Tests for cyclone_jax targets — the TargetSpec contract: config resolution,
label mapping (scalar + vectorised, non-contiguous sets), validation guards,
and the named-y build.
"""

import dataclasses

import numpy as np
import pytest

from experiments.cyclone_jax.data.targets import (
    DEFAULT_CLASS_SET, SSHS_NAMES, TargetSpec, resolve_target,
)


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------

class TestResolveTarget:

    def test_defaults_match_v1(self):
        spec = resolve_target({})
        assert spec.variable == 'usa_sshs'
        assert spec.kind == 'categorical'
        assert spec.class_set == DEFAULT_CLASS_SET
        assert spec.n_classes == 6

    def test_reads_data_yaml_keys(self):
        spec = resolve_target({'target': 'usa_sshs', 'class_set': [4, 5, 6]})
        assert spec.class_set == (4, 5, 6) and spec.n_classes == 3

    def test_continuous_reserved_raises(self):
        with pytest.raises(NotImplementedError, match='reserved'):
            resolve_target({'target': 'usa_wind'})

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match='unknown target'):
            resolve_target({'target': 'usa_sshs_raw'})

    def test_frozen(self):
        spec = resolve_target({})
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.class_set = (0, 1)


class TestClassSetGuards:

    def test_empty_raises(self):
        with pytest.raises(ValueError, match='class_set'):
            resolve_target({'class_set': []})

    def test_unsorted_raises(self):
        with pytest.raises(ValueError, match='ascending'):
            resolve_target({'class_set': [5, 4, 3]})

    def test_duplicates_raise(self):
        with pytest.raises(ValueError, match='ascending'):
            resolve_target({'class_set': [3, 3, 4]})

    def test_out_of_scheme_raises(self):
        with pytest.raises(ValueError, match='remapped scheme'):
            resolve_target({'class_set': [3, 9]})

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match='kind'):
            TargetSpec(variable='usa_sshs', kind='ordinal',
                       class_set=(3, 4))


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

class TestLabels:

    def test_label_is_position_in_class_set(self):
        spec = resolve_target({})
        assert spec.label(3) == 0 and spec.label(8) == 5

    def test_non_contiguous_class_set(self):
        spec = resolve_target({'class_set': [3, 5, 7]})
        assert [spec.label(c) for c in (3, 5, 7)] == [0, 1, 2]
        with pytest.raises(ValueError, match='not in class_set'):
            spec.label(4)

    def test_label_outside_set_raises(self):
        with pytest.raises(ValueError, match='not in class_set'):
            resolve_target({}).label(2)

    def test_labels_vectorised_matches_scalar(self):
        spec = resolve_target({})
        vals = np.array([3.0, 8.0, 5.0, 3.0], np.float32)
        out = spec.labels(vals)
        assert out.dtype == np.int32
        assert out.tolist() == [spec.label(v) for v in vals]

    def test_labels_vectorised_bad_value_raises(self):
        with pytest.raises(ValueError, match='not in class_set'):
            resolve_target({}).labels(np.array([3.0, 2.0]))

    def test_class_names_default_set(self):
        spec = resolve_target({})
        assert spec.class_names == ('Tropical Storm', 'Cat 1', 'Cat 2',
                                    'Cat 3', 'Cat 4', 'Cat 5')
        assert len(SSHS_NAMES) == 9


# ---------------------------------------------------------------------------
# y build
# ---------------------------------------------------------------------------

class TestBuildY:

    def test_named_dict_contract(self):
        spec = resolve_target({})
        fixes = {
            'usa_sshs': np.array([4.0], np.float32),
            'sid':      np.array(['AL012020']),
            'lat':      np.array([15.5], np.float32),
            'lon':      np.array([-60.0], np.float32),
            'time':     np.array(['2020-08-01T12:00'], 'datetime64[ns]'),
        }
        y = spec.build_y(fixes, 0)
        assert set(y) == {'target', 'sid', 'lat', 'lon', 'time'}
        assert y['target'] == 1 and y['target'].dtype == np.int32
        assert y['sid'] == 'AL012020' and isinstance(y['sid'], str)
        assert y['lat'].dtype == np.float32
        assert y['time'] == np.datetime64('2020-08-01T12:00', 'ns')
