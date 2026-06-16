"""
Tests for experiments/sparse_obs_encoder/data/splits.py.

Synthetic-data tests use the in-memory fixture builder from test_ibtracs.py.
The referee tests at the bottom use the real ibtracs_full.npz /
ibtracs_multi_storm_times.npz files and are skipped if absent.
"""

from pathlib import Path

import pytest

from experiments.sparse_obs_encoder.data.sources.ibtracs import (
    IBTrACSDataset,
    IBTRACS_TRAIN_SEASONS,
    IBTRACS_VAL_SEASONS,
    IBTRACS_TEST_SEASONS,
)
from experiments.sparse_obs_encoder.data.splits import resolve_splits

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


# Synthetic data: years 2019 / 2021 / 2023, 8 rows each; multi-storm = the
# first 3 rows of 2019 (all SID 2019A).
YEAR_CONFIG = {
    'strategy': 'year',
    'train': {'years': [2019]},
    'val':   {'years': [2021]},
    'test':  {'years': [2023], 'hard_test': 'multi_storm'},
}


# ---------------------------------------------------------------------------
# 'year' strategy — synthetic data
# ---------------------------------------------------------------------------

class TestYearStrategy:

    def test_returns_train_val_test_keys(self, ds):
        result = resolve_splits(YEAR_CONFIG, ds)
        for name in ('train', 'val', 'test', 'manifest'):
            assert name in result

    def test_multi_storm_excluded_from_train(self, ds):
        result = resolve_splits(YEAR_CONFIG, ds)
        # Year 2019 has 8 rows, 3 of which are multi-storm.
        assert result['manifest']['train']['n_rows'] == 5
        assert result['manifest']['train']['n_sids'] == 2

    def test_val_unaffected_by_multi_storm(self, ds):
        result = resolve_splits(YEAR_CONFIG, ds)
        assert result['manifest']['val']['n_rows'] == 8
        assert result['manifest']['val']['n_sids'] == 2

    def test_manifest_years_recorded(self, ds):
        result = resolve_splits(YEAR_CONFIG, ds)
        assert result['manifest']['train']['years'] == [2019]
        assert result['manifest']['val']['years']   == [2021]
        assert result['manifest']['test']['years']  == [2023]
        assert result['manifest']['strategy'] == 'year'

    def test_manifest_sids_sorted(self, ds):
        result = resolve_splits(YEAR_CONFIG, ds)
        sids = result['manifest']['train']['sids']
        assert sids == sorted(sids)

    def test_class_counts_sum_to_n_rows(self, ds):
        result = resolve_splits(YEAR_CONFIG, ds)
        for name in ('train', 'val', 'test'):
            entry = result['manifest'][name]
            assert sum(entry['class_counts'].values()) == entry['n_rows']

    def test_hard_test_is_multi_storm_only(self, ds):
        result = resolve_splits(YEAR_CONFIG, ds)
        # Multi-storm rows are all in year 2019, not 2023 -> empty hard_test.
        assert result['hard_test']['ibtracs'].is_multi_storm.all()
        assert result['manifest']['hard_test']['n_rows'] == 0

    def test_insitu_none_when_not_provided(self, ds):
        result = resolve_splits(YEAR_CONFIG, ds)
        for name in ('train', 'val', 'test'):
            assert result[name]['insitu'] is None

    def test_ibtracs_returns_ibtracs_type(self, ds):
        result = resolve_splits(YEAR_CONFIG, ds)
        for name in ('train', 'val', 'test'):
            assert type(result[name]['ibtracs']) is IBTrACSDataset


# ---------------------------------------------------------------------------
# 'year_random' strategy — synthetic data
# ---------------------------------------------------------------------------

# train_val = [2019, 2021]: 2019 has 8 rows (3 multi-storm excluded -> 5) plus
# 2021's 8 rows = 13 pooled single-storm rows, split 50/50 -> val 6, train 7.
# test = [2023]: 8 single-storm rows.
YEAR_RANDOM_CONFIG = {
    'strategy': 'year_random',
    'train_val': {'years': [2019, 2021]},
    'test':      {'years': [2023], 'hard_test': 'multi_storm'},
    'val':       {'fraction': 0.5, 'seed': 0},
}


class TestYearRandomStrategy:

    def test_returns_keys(self, ds):
        result = resolve_splits(YEAR_RANDOM_CONFIG, ds)
        for name in ('train', 'val', 'test', 'manifest'):
            assert name in result

    def test_train_val_partition_pool(self, ds):
        m = resolve_splits(YEAR_RANDOM_CONFIG, ds)['manifest']
        # 13 pooled single-storm rows split into train + val (disjoint, complete)
        assert m['train']['n_rows'] + m['val']['n_rows'] == 13

    def test_val_fraction_holds(self, ds):
        m = resolve_splits(YEAR_RANDOM_CONFIG, ds)['manifest']
        assert m['val']['n_rows'] == int(13 * 0.5)   # 6
        assert m['train']['n_rows'] == 13 - int(13 * 0.5)

    def test_deterministic_for_fixed_seed(self, ds):
        m1 = resolve_splits(YEAR_RANDOM_CONFIG, ds)['manifest']
        m2 = resolve_splits(YEAR_RANDOM_CONFIG, ds)['manifest']
        for n in ('train', 'val', 'test'):
            assert m1[n]['n_rows'] == m2[n]['n_rows']
            assert m1[n]['sids'] == m2[n]['sids']

    def test_test_is_separate_years_single_storm(self, ds):
        m = resolve_splits(YEAR_RANDOM_CONFIG, ds)['manifest']
        assert m['test']['n_rows'] == 8        # 2023, no multi-storm
        assert m['test']['years'] == [2023]

    def test_train_val_share_years_manifest(self, ds):
        m = resolve_splits(YEAR_RANDOM_CONFIG, ds)['manifest']
        assert m['train']['years'] == [2019, 2021]
        assert m['val']['years']   == [2019, 2021]

    def test_manifest_records_assignment(self, ds):
        m = resolve_splits(YEAR_RANDOM_CONFIG, ds)['manifest']
        assert m['strategy'] == 'year_random'
        assert m['assignment'] == {'fraction': 0.5, 'seed': 0, 'level': 'row'}

    def test_class_counts_sum_to_n_rows(self, ds):
        m = resolve_splits(YEAR_RANDOM_CONFIG, ds)['manifest']
        for name in ('train', 'val', 'test'):
            entry = m[name]
            assert sum(entry['class_counts'].values()) == entry['n_rows']

    def test_hard_test_is_multi_storm_only(self, ds):
        result = resolve_splits(YEAR_RANDOM_CONFIG, ds)
        # Multi-storm rows are in 2019, not the 2023 test years -> empty.
        assert result['hard_test']['ibtracs'].is_multi_storm.all()
        assert result['manifest']['hard_test']['n_rows'] == 0

    def test_explicit_train_block_raises(self, ds):
        cfg = {**YEAR_RANDOM_CONFIG, 'train': {'years': [2019]}}
        with pytest.raises(ValueError, match='remainder'):
            resolve_splits(cfg, ds)

    def test_missing_train_val_years_raises(self, ds):
        cfg = {**YEAR_RANDOM_CONFIG, 'train_val': {}}
        with pytest.raises(KeyError, match='train_val.years'):
            resolve_splits(cfg, ds)

    def test_missing_val_keys_raise(self, ds):
        cfg = {**YEAR_RANDOM_CONFIG, 'val': {'fraction': 0.5}}
        with pytest.raises(KeyError, match='seed'):
            resolve_splits(cfg, ds)

    def test_overlapping_train_val_and_test_raises(self, ds):
        cfg = {**YEAR_RANDOM_CONFIG,
               'train_val': {'years': [2019, 2021, 2023]}}
        with pytest.raises(ValueError, match='disjoint'):
            resolve_splits(cfg, ds)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_unknown_strategy_raises(self, ds):
        cfg = {**YEAR_CONFIG, 'strategy': 'hybrid'}
        with pytest.raises(NotImplementedError, match='hybrid'):
            resolve_splits(cfg, ds)

    def test_missing_years_key_raises(self, ds):
        cfg = {
            'strategy': 'year',
            'train': {},
            'val':   {'years': [2021]},
            'test':  {'years': [2023]},
        }
        with pytest.raises(KeyError, match='train'):
            resolve_splits(cfg, ds)

    def test_overlapping_years_raise(self, ds):
        cfg = {
            'strategy': 'year',
            'train': {'years': [2019, 2021]},
            'val':   {'years': [2021]},
            'test':  {'years': [2023]},
        }
        with pytest.raises(ValueError, match='disjoint'):
            resolve_splits(cfg, ds)

    def test_unknown_hard_test_value_raises(self, ds):
        cfg = {**YEAR_CONFIG, 'test': {'years': [2023], 'hard_test': 'bogus'}}
        with pytest.raises(ValueError, match='hard_test'):
            resolve_splits(cfg, ds)

    def test_hard_test_without_multi_path_raises(self, ds_no_multi):
        with pytest.warns(UserWarning, match='multi_storm_path'):
            with pytest.raises(ValueError, match='multi_storm_path'):
                resolve_splits(YEAR_CONFIG, ds_no_multi)

    def test_no_hard_test_without_multi_path_ok(self, ds_no_multi):
        cfg = {**YEAR_CONFIG, 'test': {'years': [2023]}}
        with pytest.warns(UserWarning, match='multi_storm_path'):
            result = resolve_splits(cfg, ds_no_multi)
        assert 'hard_test' not in result


# ---------------------------------------------------------------------------
# Referee — 'year' strategy config must reproduce the previous hardcoded
# IBTRACS_TRAIN/VAL/TEST_SEASONS split sizes exactly. Requires real data files.
# ---------------------------------------------------------------------------

IBTRACS_FULL_PATH = Path('E:/sparse_obs/ibtracs/ibtracs_full.npz')
MULTI_STORM_PATH  = Path('E:/sparse_obs/ibtracs/ibtracs_multi_storm_times.npz')

_real_data_available = IBTRACS_FULL_PATH.exists() and MULTI_STORM_PATH.exists()


@pytest.mark.skipif(not _real_data_available, reason="real ibtracs npz files not found")
class TestYearStrategyReferee:

    @pytest.fixture(scope='class')
    def real_result(self):
        ibtracs_full = IBTrACSDataset(IBTRACS_FULL_PATH, MULTI_STORM_PATH)
        cfg = {
            'strategy': 'year',
            'train': {'years': IBTRACS_TRAIN_SEASONS},
            'val':   {'years': IBTRACS_VAL_SEASONS},
            'test':  {'years': IBTRACS_TEST_SEASONS, 'hard_test': 'multi_storm'},
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


# ---------------------------------------------------------------------------
# Referee — 'year_random' strategy on real data: determinism + the pooled
# train+val rows are partitioned exactly into train/val.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _real_data_available, reason="real ibtracs npz files not found")
class TestYearRandomStrategyReferee:

    @pytest.fixture(scope='class')
    def cfg(self):
        return {
            'strategy': 'year_random',
            'train_val': {'years': IBTRACS_TRAIN_SEASONS + IBTRACS_VAL_SEASONS},
            'test':      {'years': IBTRACS_TEST_SEASONS, 'hard_test': 'multi_storm'},
            'val':       {'fraction': 0.2, 'seed': 42},
        }

    @pytest.fixture(scope='class')
    def ibtracs_full(self):
        return IBTrACSDataset(IBTRACS_FULL_PATH, MULTI_STORM_PATH)

    def test_deterministic(self, ibtracs_full, cfg):
        m1 = resolve_splits(cfg, ibtracs_full)['manifest']
        m2 = resolve_splits(cfg, ibtracs_full)['manifest']
        for n in ('train', 'val', 'test'):
            assert m1[n]['n_rows'] == m2[n]['n_rows']

    def test_train_val_partition_pool(self, ibtracs_full, cfg):
        m = resolve_splits(cfg, ibtracs_full)['manifest']
        # train+val rows partition the pooled single-storm rows (6364 + 757).
        assert m['train']['n_rows'] + m['val']['n_rows'] == 6364 + 757
        assert m['val']['n_rows'] == int((6364 + 757) * 0.2)
