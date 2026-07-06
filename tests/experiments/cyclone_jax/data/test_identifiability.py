"""
Tests for data/identifiability.py — input-hash collision detection on a
controlled fake loader (real-loader behaviour is byte-exact hashing of
whatever build() returns, exercised here directly).
"""

import numpy as np
import pytest

from experiments.cyclone_jax.data.identifiability import input_collisions

# fixes 0,1: identical inputs, DIFFERENT targets (conflict)
# fix 2:     unique input
# fixes 3,4: identical inputs, SAME target (collision, no conflict)
_X = {
    0: {'lat': np.float32([1, 2]), 'obs': np.float32([[1.0], [2.0]])},
    2: {'lat': np.float32([9, 9]), 'obs': np.float32([[1.0], [2.0]])},
    3: {'lat': np.float32([5, 6]), 'obs': np.float32([[0.0], [np.nan]])},
}
_X[1] = _X[0]
_X[4] = _X[3]
_Y = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}


class FakeLoader:
    fixes = {'sid': np.asarray(['A', 'B', 'C', 'D', 'E']),
             'time': np.arange(5).astype('datetime64[h]')}

    def __len__(self):
        return 5

    def build(self, i):
        return {'x': {k: v.copy() for k, v in _X[i].items()},
                'y': {'target': _Y[i]}}


class TestInputCollisions:

    @pytest.fixture()
    def report(self):
        return input_collisions(FakeLoader())

    def test_counts(self, report):
        assert report['n_fixes'] == 5
        assert report['n_unique_inputs'] == 3
        assert len(report['collisions']) == 2
        assert len(report['conflicts']) == 1

    def test_conflict_group_details(self, report):
        g, = report['conflicts']
        assert g['indices'] == [0, 1]
        assert g['sids'] == ['A', 'B']
        assert sorted(g['targets']) == [0, 1]

    def test_max_accuracy_ceiling(self, report):
        # one of fixes 0/1 must be wrong -> ceiling 4/5
        assert report['n_unmemorisable'] == 1
        assert report['max_accuracy'] == pytest.approx(0.8)

    def test_same_target_collision_is_not_a_conflict(self, report):
        no_conflict = [g for g in report['collisions'] if not g['conflict']]
        assert len(no_conflict) == 1 and no_conflict[0]['indices'] == [3, 4]

    def test_indices_subset(self):
        report = input_collisions(FakeLoader(), indices=[2, 3, 4])
        assert report['n_fixes'] == 3
        assert report['conflicts'] == []
        assert report['max_accuracy'] == 1.0

    def test_nan_inputs_hash_equal(self):
        # 3 and 4 share a NaN cell — byte-exact hashing must still group
        report = input_collisions(FakeLoader(), indices=[3, 4])
        assert report['n_unique_inputs'] == 1
