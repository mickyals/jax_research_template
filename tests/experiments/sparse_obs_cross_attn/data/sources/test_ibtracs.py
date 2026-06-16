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
    status_sshs_to_class,
    CLASS_NAMES,
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

    # ISO_TIME year aligned with SEASON (filter_years filters on the
    # observation year, not the SEASON column).
    iso_times = np.array([
        np.datetime64(f"{s}-09-01", "ns").astype(np.int64)
        + (i % n_per_season) * 21600_000_000_000
        for i, s in enumerate(seasons_list)
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

    return npz_path, ms_path, n, iso_times, sids


def _make_sid_meta_npz(tmp_path, sids, n_per_sid=4,
                       peak_sshs=None, track_years=None):
    """Synthetic ibtracs_sid_meta.npz matching _make_ibtracs_npz's SIDs.

    peak_sshs : optional dict {sid: int} — per-storm stratum label.
    track_years : optional dict {sid: (start_year, end_year)} — track span;
        track_start/track_end become mid-year timestamps in those years.
    """
    import pandas as pd

    unique_sids = list(dict.fromkeys(sids))   # preserve order, dedupe
    n = len(unique_sids)
    if track_years is None:
        track_start = np.full(n, 0, dtype=np.int64)
        track_end   = np.full(n, 0, dtype=np.int64)
    else:
        track_start = np.array([pd.Timestamp(f'{track_years[s][0]}-06-01').value
                                for s in unique_sids], dtype=np.int64)
        track_end   = np.array([pd.Timestamp(f'{track_years[s][1]}-06-08').value
                                for s in unique_sids], dtype=np.int64)
    data = {
        'SID':         np.array(unique_sids),
        'NAME':        np.array([f'STORM_{s}' for s in unique_sids]),
        'SEASON':      np.array([int(s[:4]) for s in unique_sids], dtype=np.int32),
        'BASIN':       np.full(n, 'NA'),
        'SUBBASIN':    np.full(n, 'CS'),
        'USA_AGENCY':  np.full(n, 'hurdat_atl'),
        'USA_ATCF_ID': np.array([f'AL{i:02d}' for i in range(n)]),
        'peak_wind':   np.full(n, 50.0, dtype=np.float32),
        'peak_sshs':   (np.full(n, 1, dtype=np.int32) if peak_sshs is None
                        else np.array([peak_sshs[s] for s in unique_sids],
                                      dtype=np.int32)),
        'min_pres':    np.full(n, 99000.0, dtype=np.float32),
        'n_timesteps': np.full(n, n_per_sid, dtype=np.int32),
        'track_start': track_start,
        'track_end':   track_end,
    }
    path = tmp_path / 'ibtracs_sid_meta.npz'
    np.savez(path, **data)
    return path


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


@pytest.fixture
def sid_meta_path(tmp_path, ibtracs_paths):
    *_, sids = ibtracs_paths
    return _make_sid_meta_npz(tmp_path, sids, n_per_sid=4)


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

    def test_n_classes(self):
        assert N_CLASSES == 9
        assert len(CLASS_NAMES) == N_CLASSES


# ---------------------------------------------------------------------------
# Ordinal organisation label mapping (status_sshs_to_class)
# ---------------------------------------------------------------------------

class TestLabelMapping:

    def test_disturbance_statuses(self):
        for st in ('DB', 'LO', 'WV', 'MD'):
            assert status_sshs_to_class(st, -3) == 1   # Disturbance

    def test_depression_statuses(self):
        assert status_sshs_to_class('TD', -1) == 2     # tropical depression
        assert status_sshs_to_class('SD', -2) == 2     # subtropical depression

    def test_storm_statuses(self):
        assert status_sshs_to_class('TS', 0) == 3      # tropical storm
        assert status_sshs_to_class('SS', -2) == 3     # subtropical storm

    def test_hurricane_category_from_sshs(self):
        # Hurricane status: category number comes from USA_SSHS (1..5 → 4..8).
        for cat in range(1, 6):
            assert status_sshs_to_class('HU', cat) == 3 + cat

    def test_status_drives_over_sshs(self):
        # Agency status wins on disagreement: STATUS=TS but SSHS=1 → Storm,
        # not Category 1 (decision r2 — STATUS-driven).
        assert status_sshs_to_class('TS', 1) == 3

    def test_hurricane_status_below_cat1_falls_back_to_storm(self):
        assert status_sshs_to_class('HU', 0) == 3

    def test_offaxis_statuses_excluded(self):
        # Extratropical / post-tropical / dissipating / unknown → None.
        for st in ('EX', 'ET', 'PT', 'DS', 'IN', 'XX'):
            assert status_sshs_to_class(st, 0) is None

    def test_case_insensitive(self):
        assert status_sshs_to_class('td', -1) == 2
        assert status_sshs_to_class(' Hu ', 3) == 6


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
        _, _, n, _, _ = ibtracs_paths
        assert len(ds) == n

    def test_seasons_dtype(self, ds):
        assert ds.seasons.dtype == np.int32

    def test_seasons_values(self, ds):
        assert set(ds.seasons.tolist()) == {2019, 2021, 2023}

    def test_n_sids(self, ds):
        assert ds.n_sids == 6   # 2 SIDs per season × 3 seasons

    def test_is_multi_storm_shape(self, ds, ibtracs_paths):
        _, _, n, _, _ = ibtracs_paths
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

    def test_filter_years_count(self, ds):
        sub = ds.filter_years([2019])
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
        sub = ds.filter_years([2019])
        assert sub._multi_times is not None

    def test_filter_returns_ibtracs_type(self, ds):
        assert type(ds.filter_years([2019])) is IBTrACSDataset

    def test_filter_sids(self, ds):
        sub = ds.filter_sids(['2019A'])
        assert len(sub) == 4
        assert set(sub['SID'].tolist()) == {'2019A'}

    def test_filter_sids_multiple(self, ds):
        sub = ds.filter_sids(['2019A', '2021B'])
        assert set(sub['SID'].tolist()) == {'2019A', '2021B'}

    def test_filter_sids_returns_ibtracs_type(self, ds):
        assert type(ds.filter_sids(['2019A'])) is IBTrACSDataset


# ---------------------------------------------------------------------------
# SID metadata validation
# ---------------------------------------------------------------------------

class TestSidMeta:

    def test_loads_with_valid_meta(self, ibtracs_paths, sid_meta_path):
        npz, ms, *_ = ibtracs_paths
        ds = IBTrACSDataset(npz, ms, sid_meta_path)
        assert ds._sid_meta is not None

    def test_raises_on_sid_set_mismatch(self, tmp_path, ibtracs_paths):
        npz, ms, n, iso_times, sids = ibtracs_paths
        bad_meta = _make_sid_meta_npz(tmp_path, sids + ['9999Z'], n_per_sid=4)
        with pytest.raises(ValueError, match='SID set'):
            IBTrACSDataset(npz, ms, bad_meta)

    def test_raises_on_n_timesteps_mismatch(self, tmp_path, ibtracs_paths):
        npz, ms, n, iso_times, sids = ibtracs_paths
        bad_meta = _make_sid_meta_npz(tmp_path, sids, n_per_sid=999)
        with pytest.raises(ValueError, match='n_timesteps'):
            IBTrACSDataset(npz, ms, bad_meta)

    def test_filter_preserves_sid_meta(self, ibtracs_paths, sid_meta_path):
        npz, ms, *_ = ibtracs_paths
        ds = IBTrACSDataset(npz, ms, sid_meta_path)
        sub = ds.filter_years([2019])
        assert sub._sid_meta is not None

    def test_no_validation_without_sid_meta_path(self, ds):
        assert ds._sid_meta is None



# ---------------------------------------------------------------------------
# Registry self-registration
# ---------------------------------------------------------------------------

def test_ibtracs_self_registers_with_generic_registry():
    """Importing the experiment's ibtracs module registers the IBTRACS factory
    with the generic datasets registry (experiment -> jrt dependency only)."""
    # The module is already imported at the top of this test file.
    from datasets.datamodule import list_datasets
    assert "IBTRACS" in list_datasets()


def test_ibtracs_factory_builds_dataset_via_registry(ibtracs_paths):
    from datasets.datamodule import DATASETS
    npz, ms, *_ = ibtracs_paths
    ds = DATASETS["IBTRACS"]({"npz_path": str(npz), "multi_storm_path": str(ms)})
    assert isinstance(ds, IBTrACSDataset)
