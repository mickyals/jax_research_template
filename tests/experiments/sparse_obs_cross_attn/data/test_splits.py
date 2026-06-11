"""
Tests for experiments/sparse_obs_cross_attn/data/splits.py.

Synthetic-data tests use the in-memory fixture builder from
test_ibtracs.py. The referee test at the bottom uses the real
ibtracs_full.npz / ibtracs_multi_storm_times.npz files and is skipped if
they are not present on disk.
"""

from pathlib import Path

import pytest

from experiments.sparse_obs_cross_attn.data.sources.ibtracs import (
    IBTrACSDataset,
    IBTRACS_TRAIN_SEASONS,
    IBTRACS_VAL_SEASONS,
    IBTRACS_TEST_SEASONS,
)
from experiments.sparse_obs_cross_attn.data.splits import resolve_splits

from tests.experiments.sparse_obs_cross_attn.data.sources.test_ibtracs import (
    _make_ibtracs_npz,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ds(tmp_path):
    npz, ms, *_ = _make_ibtracs_npz(tmp_path)
    return IBTrACSDataset(npz, ms)


@pytest.fixture
def ds_no_multi(tmp_path):
    npz, *_ = _make_ibtracs_npz(tmp_path)
    return IBTrACSDataset(npz)


SEASON_CONFIG = {
    'strategy': 'season',
    'train': {'seasons': [2019]},
    'val':   {'seasons': [2021]},
    'test':  {'seasons': [2023], 'hard_test': 'multi_storm'},
}


# ---------------------------------------------------------------------------
# 'season' strategy — synthetic data
# ---------------------------------------------------------------------------

class TestSeasonStrategy:

    def test_returns_train_val_test_keys(self, ds):
        result = resolve_splits(SEASON_CONFIG, ds)
        for name in ('train', 'val', 'test', 'manifest'):
            assert name in result

    def test_multi_storm_excluded_from_train(self, ds):
        result = resolve_splits(SEASON_CONFIG, ds)
        # Season 2019 has 8 rows, 3 of which are multi-storm.
        assert result['manifest']['train']['n_rows'] == 5
        assert result['manifest']['train']['n_sids'] == 2

    def test_val_unaffected_by_multi_storm(self, ds):
        result = resolve_splits(SEASON_CONFIG, ds)
        assert result['manifest']['val']['n_rows'] == 8
        assert result['manifest']['val']['n_sids'] == 2

    def test_manifest_seasons_recorded(self, ds):
        result = resolve_splits(SEASON_CONFIG, ds)
        assert result['manifest']['train']['seasons'] == [2019]
        assert result['manifest']['val']['seasons']   == [2021]
        assert result['manifest']['test']['seasons']  == [2023]

    def test_manifest_sids_sorted(self, ds):
        result = resolve_splits(SEASON_CONFIG, ds)
        sids = result['manifest']['train']['sids']
        assert sids == sorted(sids)

    def test_class_counts_sum_to_n_rows(self, ds):
        result = resolve_splits(SEASON_CONFIG, ds)
        for name in ('train', 'val', 'test'):
            entry = result['manifest'][name]
            assert sum(entry['class_counts'].values()) == entry['n_rows']

    def test_hard_test_is_multi_storm_only(self, ds):
        result = resolve_splits(SEASON_CONFIG, ds)
        # Multi-storm rows are all in season 2019, not 2023 -> empty hard_test.
        assert result['hard_test']['ibtracs'].is_multi_storm.all()
        assert result['manifest']['hard_test']['n_rows'] == 0

    def test_insitu_none_when_not_provided(self, ds):
        result = resolve_splits(SEASON_CONFIG, ds)
        for name in ('train', 'val', 'test'):
            assert result[name]['insitu'] is None

    def test_ibtracs_returns_ibtracs_type(self, ds):
        result = resolve_splits(SEASON_CONFIG, ds)
        for name in ('train', 'val', 'test'):
            assert type(result[name]['ibtracs']) is IBTrACSDataset


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_unknown_strategy_raises(self, ds):
        cfg = {**SEASON_CONFIG, 'strategy': 'hybrid'}
        with pytest.raises(NotImplementedError, match='hybrid'):
            resolve_splits(cfg, ds)

    def test_missing_seasons_key_raises(self, ds):
        cfg = {
            'strategy': 'season',
            'train': {},
            'val':   {'seasons': [2021]},
            'test':  {'seasons': [2023]},
        }
        with pytest.raises(KeyError, match='train'):
            resolve_splits(cfg, ds)

    def test_overlapping_seasons_raise(self, ds):
        cfg = {
            'strategy': 'season',
            'train': {'seasons': [2019, 2021]},
            'val':   {'seasons': [2021]},
            'test':  {'seasons': [2023]},
        }
        with pytest.raises(ValueError, match='disjoint'):
            resolve_splits(cfg, ds)

    def test_unknown_hard_test_value_raises(self, ds):
        cfg = {**SEASON_CONFIG, 'test': {'seasons': [2023], 'hard_test': 'bogus'}}
        with pytest.raises(ValueError, match='hard_test'):
            resolve_splits(cfg, ds)

    def test_hard_test_without_multi_path_raises(self, ds_no_multi):
        with pytest.warns(UserWarning, match='multi_storm_path'):
            with pytest.raises(ValueError, match='multi_storm_path'):
                resolve_splits(SEASON_CONFIG, ds_no_multi)

    def test_no_hard_test_without_multi_path_ok(self, ds_no_multi):
        cfg = {**SEASON_CONFIG, 'test': {'seasons': [2023]}}
        with pytest.warns(UserWarning, match='multi_storm_path'):
            result = resolve_splits(cfg, ds_no_multi)
        assert 'hard_test' not in result


# ---------------------------------------------------------------------------
# Referee — season-strategy config must reproduce the previous hardcoded
# IBTRACS_TRAIN/VAL/TEST_SEASONS split sizes exactly (decision 3, plan
# implementation order step 1 referee). Requires real data files.
# ---------------------------------------------------------------------------

IBTRACS_FULL_PATH = Path('E:/sparse_obs/ibtracs/ibtracs_full.npz')
MULTI_STORM_PATH  = Path('E:/sparse_obs/ibtracs/ibtracs_multi_storm_times.npz')

_real_data_available = IBTRACS_FULL_PATH.exists() and MULTI_STORM_PATH.exists()


@pytest.mark.skipif(not _real_data_available, reason="real ibtracs npz files not found")
class TestSeasonStrategyReferee:

    @pytest.fixture(scope='class')
    def real_result(self):
        ibtracs_full = IBTrACSDataset(IBTRACS_FULL_PATH, MULTI_STORM_PATH)
        cfg = {
            'strategy': 'season',
            'train': {'seasons': IBTRACS_TRAIN_SEASONS},
            'val':   {'seasons': IBTRACS_VAL_SEASONS},
            'test':  {'seasons': IBTRACS_TEST_SEASONS, 'hard_test': 'multi_storm'},
        }
        return resolve_splits(cfg, ibtracs_full)

    def test_train_size(self, real_result):
        assert real_result['manifest']['train']['n_rows'] == 6364
        assert real_result['manifest']['train']['n_sids'] == 225

    def test_val_size(self, real_result):
        assert real_result['manifest']['val']['n_rows'] == 757
        assert real_result['manifest']['val']['n_sids'] == 23

    def test_test_size(self, real_result):
        assert real_result['manifest']['test']['n_rows'] == 1226
        assert real_result['manifest']['test']['n_sids'] == 44
