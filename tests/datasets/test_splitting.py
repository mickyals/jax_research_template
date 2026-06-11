"""
Tests for datasets/splitting.py.
"""

import numpy as np
import pytest

from datasets.splitting import (
    assign_groups_by_fraction,
    group_mask,
    validate_disjoint_groups,
)


# ---------------------------------------------------------------------------
# validate_disjoint_groups
# ---------------------------------------------------------------------------

class TestValidateDisjointGroups:

    def test_disjoint_ok(self):
        validate_disjoint_groups({
            'train': [2010, 2011],
            'val':   [2021],
            'test':  [2023],
        })

    def test_overlap_raises(self):
        with pytest.raises(ValueError, match='disjoint'):
            validate_disjoint_groups({
                'train': [2010, 2021],
                'val':   [2021],
                'test':  [2023],
            })

    def test_works_with_non_int_values(self):
        validate_disjoint_groups({
            'train': ['a', 'b'],
            'val':   ['c'],
        })


# ---------------------------------------------------------------------------
# group_mask
# ---------------------------------------------------------------------------

class TestGroupMask:

    def test_basic(self):
        row_groups = np.array(['a', 'a', 'b', 'c', 'b'])
        mask = group_mask(row_groups, ['a', 'c'])
        assert mask.tolist() == [True, True, False, True, False]

    def test_empty_groups_selects_nothing(self):
        row_groups = np.array(['a', 'b'])
        mask = group_mask(row_groups, [])
        assert not mask.any()

    def test_dtype_is_bool(self):
        row_groups = np.array([1, 2, 3])
        mask = group_mask(row_groups, [2])
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# assign_groups_by_fraction — unstratified
# ---------------------------------------------------------------------------

class TestAssignUnstratified:

    def test_shape_and_dtype(self):
        groups = np.arange(100)
        sel = assign_groups_by_fraction(groups, fraction=0.2, seed=0)
        assert sel.shape == (100,)
        assert sel.dtype == bool

    def test_count_matches_fraction(self):
        groups = np.arange(100)
        sel = assign_groups_by_fraction(groups, fraction=0.2, seed=0)
        assert sel.sum() == 20

    def test_deterministic_with_same_seed(self):
        groups = np.arange(50)
        sel1 = assign_groups_by_fraction(groups, fraction=0.3, seed=42)
        sel2 = assign_groups_by_fraction(groups, fraction=0.3, seed=42)
        assert np.array_equal(sel1, sel2)

    def test_different_seeds_differ(self):
        groups = np.arange(50)
        sel1 = assign_groups_by_fraction(groups, fraction=0.3, seed=1)
        sel2 = assign_groups_by_fraction(groups, fraction=0.3, seed=2)
        assert not np.array_equal(sel1, sel2)

    def test_fraction_out_of_range_raises(self):
        groups = np.arange(10)
        with pytest.raises(ValueError, match='fraction'):
            assign_groups_by_fraction(groups, fraction=1.0, seed=0)
        with pytest.raises(ValueError, match='fraction'):
            assign_groups_by_fraction(groups, fraction=0.0, seed=0)

    def test_duplicate_groups_raise(self):
        groups = np.array([1, 2, 2, 3])
        with pytest.raises(ValueError, match='duplicate'):
            assign_groups_by_fraction(groups, fraction=0.5, seed=0)


# ---------------------------------------------------------------------------
# assign_groups_by_fraction — stratified + floor rule
# ---------------------------------------------------------------------------

class TestAssignStratified:

    def test_floor_rule_small_stratum(self):
        # Stratum 'rare' has 4 groups; int(4 * 0.2) == 0, floor rule -> 1.
        groups = np.array([f'g{i}' for i in range(10)])
        strata = np.array(['common'] * 6 + ['rare'] * 4)
        sel = assign_groups_by_fraction(groups, fraction=0.2, seed=0, stratify_by=strata)
        rare_selected = sel[strata == 'rare']
        assert rare_selected.sum() == 1

    def test_per_stratum_count_without_floor(self):
        # 'common' has 6 groups; int(6 * 0.5) == 3, no floor needed.
        groups = np.array([f'g{i}' for i in range(10)])
        strata = np.array(['common'] * 6 + ['rare'] * 4)
        sel = assign_groups_by_fraction(groups, fraction=0.5, seed=0, stratify_by=strata)
        common_selected = sel[strata == 'common']
        assert common_selected.sum() == 3

    def test_size_one_stratum_fully_selected(self):
        groups = np.array([f'g{i}' for i in range(5)])
        strata = np.array(['common'] * 4 + ['unique'])
        sel = assign_groups_by_fraction(groups, fraction=0.2, seed=0, stratify_by=strata)
        assert sel[strata == 'unique'].all()

    def test_every_nonempty_stratum_represented(self):
        groups = np.arange(20)
        strata = np.array([0, 1, 2, 3] * 5)   # 4 strata x 5 groups each
        sel = assign_groups_by_fraction(groups, fraction=0.1, seed=7, stratify_by=strata)
        for stratum in np.unique(strata):
            assert sel[strata == stratum].sum() >= 1

    def test_deterministic_with_same_seed(self):
        groups = np.arange(30)
        strata = np.array([0, 1] * 15)
        sel1 = assign_groups_by_fraction(groups, fraction=0.2, seed=5, stratify_by=strata)
        sel2 = assign_groups_by_fraction(groups, fraction=0.2, seed=5, stratify_by=strata)
        assert np.array_equal(sel1, sel2)
