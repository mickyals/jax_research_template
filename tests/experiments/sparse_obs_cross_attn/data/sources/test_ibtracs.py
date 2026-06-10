"""
Tests for experiments/sparse_obs_cross_attn/data/sources/ibtracs.py.

All fixtures are built in memory — no real data files required.
"""

import numpy as np
import pytest

from experiments.sparse_obs_cross_attn.data.sources.ibtracs import (
    IBTrACSDataset,
    IBTRACS_PRIMARY_TARGET_COLS,
    IBTRACS_SECONDARY_TARGET_COLS,
    IBTRACS_ALL_TARGET_COLS,
    IBTRACS_TRAIN_SEASONS,
    IBTRACS_VAL_SEASONS,
    IBTRACS_TEST_SEASONS,
    SSHS_TO_CLASS,
    N_CLASSES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ibtracs_npz(tmp_path, n_per_season=8):
    """Synthetic IBTrACS covering seasons 2019, 2021, 2023 (train/val/test)."""
    seasons_list = (
        [2019] * n_per_season +
        [2021] * n_per_season +
        [2023] * n_per_season
    )
    n   = len(seasons_list)
    rng = np.random.default_rng(0)

    sids = []
    for s in [2019, 2021, 2023]:
        for i in range(n_per_season):
            sids.append(f"{s}A" if i < n_per_season // 2 else f"{s}B")

    iso_times = np.array([
        1118253600000000000 + i * 21600_000_000_000
        for i in range(n)
    ], dtype=np.int64)

    data = {
        'SID':         np.array(sids),
        'NAME':        np.array([f'STORM{i}' for i in range(n)]),
        'SEASON':      np.array(seasons_list, dtype=np.float32),
        'BASIN':       np.full(n, 'NA'),
        'SUBBASIN':    np.full(n, 'CS'),
        'ISO_TIME':    iso_times,
        'LAT':         rng.uniform(10, 28, n).astype(np.float32),
        'LON':         rng.uniform(-95, -50, n).astype(np.float32),
        'TRACK_TYPE':  np.full(n, 'main'),
        'IFLAG':       np.full(n, 'original'),
        'USA_AGENCY':  np.full(n, 'hurdat_atl'),
        'USA_ATCF_ID': np.array([f'AL01{i:04d}' for i in range(n)]),
        'USA_RECORD':  np.full(n, ' '),
        'USA_STATUS':  np.full(n, 'HU'),
        'USA_SSHS':    rng.choice([-3, -1, 0, 1, 2], n).astype(np.float32),
        'USA_WIND':    rng.uniform(20, 70, n).astype(np.float32),
        'USA_PRES':    rng.uniform(95000, 101000, n).astype(np.float32),
        'USA_POCI':    rng.uniform(100000, 102000, n).astype(np.float32),
        'USA_RMW':     rng.uniform(20000, 80000, n).astype(np.float32),
        'STORM_SPEED': rng.uniform(2, 10, n).astype(np.float32),
        'STORM_DIR':   rng.uniform(0, 360, n).astype(np.float32),
        'USA_R17MS_NE': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R17MS_SE': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R17MS_SW': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R17MS_NW': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R26MS_NE': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R26MS_SE': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R26MS_SW': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R26MS_NW': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R33MS_NE': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R33MS_SE': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R33MS_SW': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_R33MS_NW': rng.uniform(1e5, 3e5, n).astype(np.float32),
        'USA_ROCI':    rng.uniform(2e5, 6e5, n).astype(np.float32),
        'USA_EYE':     np.full(n, np.nan, dtype=np.float32),
        'USA_SEAHGT':  np.full(n, np.nan, dtype=np.float32),
        'USA_SEARAD_NE': np.full(n, np.nan, dtype=np.float32),
        'USA_SEARAD_SE': np.full(n, np.nan, dtype=np.float32),
        'USA_SEARAD_SW': np.full(n, np.nan, dtype=np.float32),
        'USA_SEARAD_NW': np.full(n, np.nan, dtype=np.float32),
    }

    npz_path = tmp_path / 'ibtracs_full.npz'
    np.savez(npz_path, **data)

    # Multi-storm file: first 3 rows of season 2019
    ms_path = tmp_path / 'ibtracs_multi_storm_times.npz'
    np.savez(ms_path, ISO_TIME=iso_times[:3], n_active=np.full(3, 2, dtype=np.int32))

    return npz_path, ms_path, n, iso_times


@pytest.fixture
def ibtracs_paths(tmp_path):
    return _make_ibtracs_npz(tmp_path)


@pytest.fixture
def ds(ibtracs_paths):
    npz, ms, *_ = ibtracs_paths
    return IBTrACSDataset(npz, ms)


@pytest.fixture
def ds_no_multi(ibtracs_paths):
    npz, *_ = ibtracs_paths
    return IBTrACSDataset(npz)


# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------

class TestColumnConstants:

    def test_primary_count(self):
        assert len(IBTRACS_PRIMARY_TARGET_COLS) == 6

    def test_secondary_count(self):
        assert len(IBTRACS_SECONDARY_TARGET_COLS) == 19

    def test_all_is_concatenation(self):
        assert IBTRACS_ALL_TARGET_COLS == (
            IBTRACS_PRIMARY_TARGET_COLS + IBTRACS_SECONDARY_TARGET_COLS
        )

    def test_no_duplicates(self):
        assert len(IBTRACS_ALL_TARGET_COLS) == len(set(IBTRACS_ALL_TARGET_COLS))

    def test_sshs_to_class_range(self):
        assert set(SSHS_TO_CLASS.keys()) == {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
        assert set(SSHS_TO_CLASS.values()) == set(range(1, 11))

    def test_n_classes(self):
        assert N_CLASSES == 11


# ---------------------------------------------------------------------------
# Season splits
# ---------------------------------------------------------------------------

class TestSeasonConstants:

    def test_train_range(self):
        assert IBTRACS_TRAIN_SEASONS == list(range(2005, 2021))

    def test_val(self):
        assert IBTRACS_VAL_SEASONS == [2021, 2022]

    def test_test_range(self):
        assert IBTRACS_TEST_SEASONS == list(range(2023, 2026))

    def test_disjoint(self):
        all_s = (
            set(IBTRACS_TRAIN_SEASONS) |
            set(IBTRACS_VAL_SEASONS) |
            set(IBTRACS_TEST_SEASONS)
        )
        assert len(all_s) == (
            len(IBTRACS_TRAIN_SEASONS) +
            len(IBTRACS_VAL_SEASONS) +
            len(IBTRACS_TEST_SEASONS)
        )

    def test_temporal_order(self):
        assert max(IBTRACS_TRAIN_SEASONS) < min(IBTRACS_VAL_SEASONS)
        assert max(IBTRACS_VAL_SEASONS)   < min(IBTRACS_TEST_SEASONS)


# ---------------------------------------------------------------------------
# IBTrACSDataset init + properties
# ---------------------------------------------------------------------------

class TestIBTrACSInit:

    def test_len(self, ds, ibtracs_paths):
        _, _, n, _ = ibtracs_paths
        assert len(ds) == n

    def test_seasons_dtype(self, ds):
        assert ds.seasons.dtype == np.int32

    def test_seasons_values(self, ds):
        assert set(ds.seasons.tolist()) == {2019, 2021, 2023}

    def test_n_sids(self, ds):
        assert ds.n_sids == 6   # 2 SIDs per season × 3 seasons

    def test_is_multi_storm_shape(self, ds, ibtracs_paths):
        _, _, n, _ = ibtracs_paths
        assert ds.is_multi_storm.shape == (n,)
        assert ds.is_multi_storm.dtype == bool

    def test_is_multi_storm_count(self, ds, ibtracs_paths):
        assert ds.is_multi_storm.sum() == 3

    def test_is_multi_storm_raises_without_path(self, ds_no_multi):
        with pytest.raises(ValueError, match='multi_storm_path'):
            _ = ds_no_multi.is_multi_storm

    def test_iso_time_is_datetimeindex(self, ds):
        import pandas as pd
        assert isinstance(ds.iso_time, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFiltering:

    def test_filter_seasons_count(self, ds):
        sub = ds.filter_seasons([2019])
        assert len(sub) == 8

    def test_filter_single_storm(self, ds):
        sub = ds.filter_single_storm()
        assert sub.is_multi_storm.sum() == 0
        assert len(sub) == len(ds) - 3

    def test_filter_multi_storm(self, ds):
        sub = ds.filter_multi_storm()
        assert sub.is_multi_storm.all()
        assert len(sub) == 3

    def test_filter_preserves_multi_times(self, ds):
        sub = ds.filter_seasons([2019])
        assert sub._multi_times is not None

    def test_filter_returns_ibtracs_type(self, ds):
        assert type(ds.filter_seasons([2019])) is IBTrACSDataset


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

class TestSplits:

    def test_train_seasons(self, ds):
        tr = ds.split('train')
        assert set(tr.seasons.tolist()).issubset(set(IBTRACS_TRAIN_SEASONS))

    def test_train_no_multi_storm(self, ds):
        assert ds.split('train').is_multi_storm.sum() == 0

    def test_val_seasons(self, ds):
        val = ds.split('val')
        assert set(val.seasons.tolist()).issubset(set(IBTRACS_VAL_SEASONS))

    def test_test_seasons(self, ds):
        tst = ds.split('test')
        assert set(tst.seasons.tolist()).issubset(set(IBTRACS_TEST_SEASONS))

    def test_hard_test_all_multi(self, ds):
        ht = ds.split('hard_test')
        assert ht.is_multi_storm.all()
        assert len(ht) == 3

    def test_unknown_split_raises(self, ds):
        with pytest.raises(ValueError, match='Unknown split'):
            ds.split('bogus')

    def test_split_without_multi_path_warns(self, ds_no_multi):
        with pytest.warns(UserWarning, match='multi_storm_path'):
            result = ds_no_multi.split('train')
        assert len(result) > 0

    def test_split_returns_ibtracs_type(self, ds):
        assert type(ds.split('train')) is IBTrACSDataset
