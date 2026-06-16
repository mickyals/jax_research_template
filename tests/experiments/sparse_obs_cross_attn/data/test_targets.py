"""
Tests for experiments/sparse_obs_encoder/data/targets.py.

No real data files required — the organisation labeller is exercised against a
plain dict (it only does ``ibtracs[col][idx]`` lookups).
"""

import numpy as np
import pytest

from experiments.sparse_obs_encoder.data.targets import (
    TargetSpec,
    TARGET_SCHEMA,
    DEFAULT_TARGET,
    resolve_target,
    NOMINAL,
    CONTINUOUS,
)
from experiments.sparse_obs_encoder.data.sources.ibtracs import (
    CLASS_NAMES, N_CLASSES, status_sshs_to_class,
)


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------

class TestResolveTarget:

    def test_default_is_organisation(self):
        assert DEFAULT_TARGET == 'organisation'
        assert resolve_target(None).name == 'organisation'

    def test_resolve_by_name(self):
        assert resolve_target('organisation') is TARGET_SCHEMA['organisation']

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match='unknown target'):
            resolve_target('does_not_exist')


# ---------------------------------------------------------------------------
# organisation spec
# ---------------------------------------------------------------------------

class TestOrganisationSpec:

    @pytest.fixture
    def spec(self):
        return resolve_target('organisation')

    def test_kind_and_loss(self, spec):
        assert spec.kind == NOMINAL
        assert spec.loss == 'cross_entropy'

    def test_n_classes_and_names_match_ibtracs(self, spec):
        assert spec.n_classes == N_CLASSES
        assert spec.class_names is CLASS_NAMES
        assert spec.include_background is True

    def test_labeller_matches_status_sshs_to_class(self, spec):
        row = {'USA_STATUS': np.array(['HU']), 'USA_SSHS': np.array([2.0])}
        assert spec.labeller(row, 0) == status_sshs_to_class('HU', 2.0) == 5

    def test_labeller_excludes_offaxis(self, spec):
        row = {'USA_STATUS': np.array(['EX']), 'USA_SSHS': np.array([-4.0])}
        assert spec.labeller(row, 0) is None


# ---------------------------------------------------------------------------
# TargetSpec construction guards
# ---------------------------------------------------------------------------

class TestTargetSpecGuards:

    def test_nominal_requires_n_classes_and_labeller(self):
        with pytest.raises(ValueError, match='needs n_classes and labeller'):
            TargetSpec(name='bad', kind=NOMINAL)

    def test_continuous_is_reserved(self):
        with pytest.raises(NotImplementedError, match='continuous'):
            TargetSpec(name='USA_WIND', kind=CONTINUOUS,
                       column='USA_WIND', bounds=(0.0, 115.0))

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match='unknown target kind'):
            TargetSpec(name='bad', kind='ordinal-ish')
