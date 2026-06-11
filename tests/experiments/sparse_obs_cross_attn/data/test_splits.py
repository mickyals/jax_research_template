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
    _make_sid_meta_npz,
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
# 'sid' strategy — synthetic data
# ---------------------------------------------------------------------------

# Fixture SIDs: 2019A/2019B, 2021A/2021B, 2023A/2023B (4 rows each).
# Track years: 2019B is the Zeta case — SEASON 2019 but track entirely in
# the interior year 2021, so it must land in the interior pool.
SID_TRACK_YEARS = {
    '2019A': (2019, 2019),
    '2019B': (2021, 2021),   # Zeta case
    '2021A': (2021, 2021),
    '2021B': (2021, 2021),
    '2023A': (2023, 2023),
    '2023B': (2023, 2023),
}
SID_PEAK_SSHS = {
    '2019A': 0, '2019B': 0, '2021A': 0, '2021B': 3,
    '2023A': 1, '2023B': 2,
}

SID_CONFIG = {
    'strategy': 'sid',
    'test': {'seasons': [2019, 2023], 'hard_test': 'multi_storm'},
    'val':  {'fraction': 0.5, 'seed': 0, 'stratify_by': 'peak_sshs'},
}


@pytest.fixture
def ds_sid(tmp_path):
    npz, ms, _, _, sids = _make_ibtracs_npz(tmp_path)
    meta = _make_sid_meta_npz(tmp_path, sids, n_per_sid=4,
                              peak_sshs=SID_PEAK_SSHS,
                              track_years=SID_TRACK_YEARS)
    return IBTrACSDataset(npz, ms, sid_meta_path=meta)


class TestSidStrategy:

    def test_track_year_membership(self, ds_sid):
        result = resolve_splits(SID_CONFIG, ds_sid)
        m = result['manifest']
        # 2019B (track year 2021) is interior despite SEASON 2019.
        interior = set(m['train']['sids']) | set(m['val']['sids'])
        assert '2019B' in interior
        assert set(m['test']['sids']) == {'2019A', '2023A', '2023B'}

    def test_sid_sets_disjoint_and_complete(self, ds_sid):
        m = resolve_splits(SID_CONFIG, ds_sid)['manifest']
        sets = [set(m[n]['sids']) for n in ('train', 'val', 'test')]
        assert sets[0] | sets[1] | sets[2] == set(SID_TRACK_YEARS)
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) \
            and not (sets[1] & sets[2])

    def test_deterministic_for_fixed_seed(self, ds_sid):
        m1 = resolve_splits(SID_CONFIG, ds_sid)['manifest']
        m2 = resolve_splits(SID_CONFIG, ds_sid)['manifest']
        for n in ('train', 'val', 'test'):
            assert m1[n]['sids'] == m2[n]['sids']

    def test_every_stratum_has_val_storm(self, ds_sid):
        m = resolve_splits(SID_CONFIG, ds_sid)['manifest']
        val = set(m['val']['sids'])
        # Interior strata: {0: [2019B, 2021A], 3: [2021B]} — floor rule
        # guarantees stratum 3 (size 1) contributes its storm to val.
        assert '2021B' in val
        assert val & {'2019B', '2021A'}

    def test_insitu_years_shared_interior_disjoint_test(self, ds_sid):
        m = resolve_splits(SID_CONFIG, ds_sid)['manifest']
        assert m['train']['insitu_years'] == m['val']['insitu_years'] == [2021]
        assert m['test']['insitu_years'] == [2019, 2023]
        assert not set(m['train']['insitu_years']) & set(m['test']['insitu_years'])

    def test_manifest_records_assignment(self, ds_sid):
        m = resolve_splits(SID_CONFIG, ds_sid)['manifest']
        assert m['strategy'] == 'sid'
        assert m['assignment'] == {'fraction': 0.5, 'seed': 0,
                                   'stratify_by': 'peak_sshs'}

    def test_manifest_entry_shape(self, ds_sid):
        m = resolve_splits(SID_CONFIG, ds_sid)['manifest']
        for n in ('train', 'val', 'test'):
            entry = m[n]
            for key in ('seasons', 'sids', 'n_rows', 'n_sids',
                        'class_counts', 'insitu_years'):
                assert key in entry, f"{n} missing {key}"
            assert sum(entry['class_counts'].values()) == entry['n_rows']

    def test_hard_test_is_multi_storm_only(self, ds_sid):
        result = resolve_splits(SID_CONFIG, ds_sid)
        # Multi-storm rows are the first 3 rows of season 2019 = SID 2019A,
        # which is a test storm under SID_TRACK_YEARS.
        assert result['hard_test']['ibtracs'].is_multi_storm.all()
        assert result['manifest']['hard_test']['n_rows'] == 3

    def test_multi_storm_rows_excluded_from_test(self, ds_sid):
        m = resolve_splits(SID_CONFIG, ds_sid)['manifest']
        # 2019A has 4 rows, 3 multi-storm -> 1 survives in test.
        assert m['test']['n_rows'] == 4 + 4 + 1

    def test_requires_sid_meta(self, ds):
        with pytest.raises(ValueError, match='sid_meta_path'):
            resolve_splits(SID_CONFIG, ds)

    def test_explicit_train_block_raises(self, ds_sid):
        cfg = {**SID_CONFIG, 'train': {'seasons': [2021]}}
        with pytest.raises(ValueError, match='remainder'):
            resolve_splits(cfg, ds_sid)

    def test_missing_val_key_raises(self, ds_sid):
        cfg = {**SID_CONFIG, 'val': {'fraction': 0.5, 'seed': 0}}
        with pytest.raises(KeyError, match='stratify_by'):
            resolve_splits(cfg, ds_sid)

    def test_unknown_stratify_column_raises(self, ds_sid):
        cfg = {**SID_CONFIG,
               'val': {'fraction': 0.5, 'seed': 0, 'stratify_by': 'bogus'}}
        with pytest.raises(ValueError, match='bogus'):
            resolve_splits(cfg, ds_sid)

    def test_empty_test_raises(self, ds_sid):
        cfg = {**SID_CONFIG, 'test': {'seasons': [1900]}}
        with pytest.raises(ValueError, match='empty test'):
            resolve_splits(cfg, ds_sid)

    def test_all_storms_edge_raises(self, ds_sid):
        cfg = {**SID_CONFIG, 'test': {'seasons': [2019, 2021, 2023]}}
        with pytest.raises(ValueError, match='interior'):
            resolve_splits(cfg, ds_sid)


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
SID_META_PATH     = Path('E:/sparse_obs/ibtracs/ibtracs_sid_meta.npz')

_real_data_available = IBTRACS_FULL_PATH.exists() and MULTI_STORM_PATH.exists()
_sid_meta_available  = _real_data_available and SID_META_PATH.exists()


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


# ---------------------------------------------------------------------------
# Referee — 'sid' strategy on real data: deterministic assignment, per-stratum
# fractions with floor rule, disjoint SID sets, Zeta in the interior pool.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _sid_meta_available, reason="real ibtracs npz files not found")
class TestSidStrategyReferee:

    FRACTION = 0.2
    SEED     = 42

    # NOTE: no multi_storm_path — filter_single_storm drops storms whose
    # every timestep is multi-storm from the manifest SID lists, which would
    # confound the assignment invariants checked here. The multi-storm
    # interaction is covered by the synthetic TestSidStrategy tests.
    @pytest.fixture(scope='class')
    def real_ds(self):
        return IBTrACSDataset(IBTRACS_FULL_PATH, sid_meta_path=SID_META_PATH)

    @pytest.fixture(scope='class')
    def cfg(self):
        return {
            'strategy': 'sid',
            'test': {'seasons': [2005, 2025]},
            'val':  {'fraction': self.FRACTION, 'seed': self.SEED,
                     'stratify_by': 'peak_sshs'},
        }

    @pytest.fixture(scope='class')
    def real_result(self, real_ds, cfg):
        with pytest.warns(UserWarning, match='multi_storm_path'):
            return resolve_splits(cfg, real_ds)

    def test_deterministic(self, real_ds, cfg, real_result):
        with pytest.warns(UserWarning, match='multi_storm_path'):
            again = resolve_splits(cfg, real_ds)
        for n in ('train', 'val', 'test'):
            assert again['manifest'][n]['sids'] == \
                real_result['manifest'][n]['sids']

    def test_sid_sets_disjoint_and_complete(self, real_ds, real_result):
        m = real_result['manifest']
        sets = {n: set(m[n]['sids']) for n in ('train', 'val', 'test')}
        all_sids = set(real_ds._sid_meta['SID'].tolist())
        assert sets['train'] | sets['val'] | sets['test'] == all_sids
        assert not (sets['train'] & sets['val'])
        assert not (sets['train'] & sets['test'])
        assert not (sets['val'] & sets['test'])

    def test_zeta_is_interior(self, real_result):
        # SEASON 2005 but track entirely Jan 2006 -> interior, not test.
        m = real_result['manifest']
        assert '2005364N24324' not in m['test']['sids']
        assert '2005364N24324' in m['train']['sids'] + m['val']['sids']

    def test_fractions_hold_per_stratum(self, real_ds, real_result):
        import numpy as np
        meta = real_ds._sid_meta
        m = real_result['manifest']
        val_set      = set(m['val']['sids'])
        interior_set = val_set | set(m['train']['sids'])
        sshs = {s: int(p) for s, p in zip(meta['SID'], meta['peak_sshs'])}
        strata: dict[int, list[str]] = {}
        for s in interior_set:
            strata.setdefault(sshs[s], []).append(s)
        for stratum, sids in strata.items():
            expected = max(1, int(len(sids) * self.FRACTION))
            got = sum(1 for s in sids if s in val_set)
            assert got == expected, f"stratum {stratum}: {got} != {expected}"

    def test_every_stratum_has_val_storm(self, real_ds, real_result):
        meta = real_ds._sid_meta
        m = real_result['manifest']
        val_set      = set(m['val']['sids'])
        interior_set = val_set | set(m['train']['sids'])
        sshs = {s: int(p) for s, p in zip(meta['SID'], meta['peak_sshs'])}
        interior_strata = {sshs[s] for s in interior_set}
        val_strata      = {sshs[s] for s in val_set}
        assert interior_strata == val_strata

    def test_insitu_years_disjoint(self, real_result):
        m = real_result['manifest']
        assert m['train']['insitu_years'] == m['val']['insitu_years']
        assert not set(m['train']['insitu_years']) & set(m['test']['insitu_years'])
