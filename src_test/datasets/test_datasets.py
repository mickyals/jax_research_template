"""
Tests for src/datasets/base.py, src/datasets/schema.py,
and src/datasets/ibtracs/dataset.py.

All tests are self-contained — synthetic .npz fixtures are built in
memory via pytest's tmp_path fixture so no real data files are required.
"""

import numpy as np
import pandas as pd
import pytest

from datasets.base import NpzDataset
from datasets.schema import (
    IBTRACS_ALL_TARGET_COLS,
    IBTRACS_PRIMARY_TARGET_COLS,
    IBTRACS_SECONDARY_TARGET_COLS,
    IBTRACS_TRAIN_SEASONS,
    IBTRACS_VAL_SEASONS,
    IBTRACS_TEST_SEASONS,
)
from datasets.ibtracs.dataset import IBTrACSDataset


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_ibtracs_npz(tmp_path, n_per_season=10):
    """
    Build a minimal synthetic IBTrACS npz covering four seasons.

    Seasons: 2019, 2020, 2021, 2023  (train, train, val, test)
    Two SIDs per season → 8 unique storms total.
    First 5 rows of 2019 are labelled multi-storm in the companion file.
    """
    seasons_list  = [2019] * n_per_season + [2020] * n_per_season \
                  + [2021] * n_per_season + [2023] * n_per_season
    n             = len(seasons_list)
    rng           = np.random.default_rng(0)

    # Two SIDs per season, alternating
    sids = []
    for s in [2019, 2020, 2021, 2023]:
        for i in range(n_per_season):
            sids.append(f"{s}A" if i < n_per_season // 2 else f"{s}B")

    iso_times = np.array([
        f"{s}-09-{(i % 28) + 1:02d}T00:00:00"
        for s, i in zip(seasons_list, range(n))
    ])

    data = {
        "SID":        np.array(sids),
        "NAME":       np.array([f"STORM{i}" for i in range(n)]),
        "SEASON":     np.array(seasons_list),
        "BASIN":      np.full(n, "NA"),
        "SUBBASIN":   np.full(n, "GM"),
        "ISO_TIME":   iso_times,
        "LAT":        rng.uniform(10, 28, n).astype(np.float32),
        "LON":        rng.uniform(-95, -50, n).astype(np.float32),
        "TRACK_TYPE": np.full(n, "MAIN"),
        "IFLAG":      np.full(n, "I"),
        "USA_AGENCY": np.full(n, "NHC"),
        "USA_ATCF_ID":np.array([f"AL0{i%9+1}{s}" for i, s in
                                  zip(range(n), seasons_list)]),
        "USA_RECORD": np.full(n, ""),
        "USA_STATUS": np.full(n, "HU"),
        "USA_SSHS":   rng.integers(1, 4, n).astype(np.float32),
        # Primary targets
        "USA_WIND":   rng.uniform(20, 70, n).astype(np.float32),
        "USA_PRES":   rng.uniform(95000, 101000, n).astype(np.float32),
        "USA_POCI":   rng.uniform(100000, 102000, n).astype(np.float32),
        "USA_RMW":    rng.uniform(20000, 80000, n).astype(np.float32),
        "STORM_SPEED":rng.uniform(2, 10, n).astype(np.float32),
        "STORM_DIR":  rng.uniform(0, 360, n).astype(np.float32),
        # A few secondary targets (not exhaustive)
        "USA_R17MS_NE": rng.uniform(100000, 300000, n).astype(np.float32),
        "USA_R17MS_SE": rng.uniform(100000, 300000, n).astype(np.float32),
        "USA_R17MS_SW": rng.uniform(100000, 300000, n).astype(np.float32),
        "USA_R17MS_NW": rng.uniform(100000, 300000, n).astype(np.float32),
        "USA_ROCI":     rng.uniform(200000, 600000, n).astype(np.float32),
    }

    npz_path = tmp_path / "ibtracs_tc_clean.npz"
    np.savez(npz_path, **data)

    # Multi-storm file: first 5 rows of 2019
    multi_times = iso_times[:5]
    ms_path = tmp_path / "ibtracs_multi_storm_times.npz"
    np.savez(ms_path, ISO_TIME=multi_times)

    return npz_path, ms_path, n, iso_times, multi_times


def _make_minimal_npz(tmp_path, name="minimal.npz"):
    """Build a tiny 3-row npz for NpzDataset base tests."""
    path = tmp_path / name
    np.savez(
        path,
        X=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        Y=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ISO_TIME=np.array(["2020-01-01", "2020-01-02", "2020-01-03"]),
        SEASON=np.array([2020, 2020, 2021]),
    )
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_npz(tmp_path):
    return _make_minimal_npz(tmp_path)


@pytest.fixture
def ibtracs_paths(tmp_path):
    npz, ms, n, iso_times, multi_times = _make_ibtracs_npz(tmp_path)
    return npz, ms, n, iso_times, multi_times


@pytest.fixture
def ds_full(ibtracs_paths):
    npz, ms, *_ = ibtracs_paths
    return IBTrACSDataset(npz, ms)


@pytest.fixture
def ds_no_multi(ibtracs_paths):
    npz, *_ = ibtracs_paths
    return IBTrACSDataset(npz)


# ===========================================================================
# Schema
# ===========================================================================

class TestSchema:

    def test_primary_targets_count(self):
        assert len(IBTRACS_PRIMARY_TARGET_COLS) == 6

    def test_secondary_targets_count(self):
        assert len(IBTRACS_SECONDARY_TARGET_COLS) == 19

    def test_all_targets_is_concatenation(self):
        assert IBTRACS_ALL_TARGET_COLS == (
            IBTRACS_PRIMARY_TARGET_COLS + IBTRACS_SECONDARY_TARGET_COLS
        )

    def test_no_duplicates_in_all_targets(self):
        assert len(IBTRACS_ALL_TARGET_COLS) == len(set(IBTRACS_ALL_TARGET_COLS))

    def test_primary_targets_are_strings(self):
        assert all(isinstance(c, str) for c in IBTRACS_PRIMARY_TARGET_COLS)

    def test_train_seasons_range(self):
        assert IBTRACS_TRAIN_SEASONS == list(range(2005, 2021))

    def test_val_seasons(self):
        assert IBTRACS_VAL_SEASONS == [2021, 2022]

    def test_test_seasons_range(self):
        assert IBTRACS_TEST_SEASONS == list(range(2023, 2026))

    def test_splits_are_disjoint(self):
        all_splits = set(IBTRACS_TRAIN_SEASONS) | set(IBTRACS_VAL_SEASONS) | set(IBTRACS_TEST_SEASONS)
        assert len(all_splits) == (
            len(IBTRACS_TRAIN_SEASONS) + len(IBTRACS_VAL_SEASONS) + len(IBTRACS_TEST_SEASONS)
        )

    def test_train_before_val_before_test(self):
        assert max(IBTRACS_TRAIN_SEASONS) < min(IBTRACS_VAL_SEASONS)
        assert max(IBTRACS_VAL_SEASONS) < min(IBTRACS_TEST_SEASONS)


# ===========================================================================
# NpzDataset (base)
# ===========================================================================

class TestNpzDataset:

    def test_loads_from_path(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        assert len(ds) == 3

    def test_getitem_returns_array(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        assert isinstance(ds["X"], np.ndarray)
        assert np.allclose(ds["X"], [1.0, 2.0, 3.0])

    def test_getitem_unknown_key_raises(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        with pytest.raises(KeyError, match="MISSING"):
            ds["MISSING"]

    def test_columns_lists_all_keys(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        assert set(ds.columns) == {"X", "Y", "ISO_TIME", "SEASON"}

    def test_repr_contains_class_and_n(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        r = repr(ds)
        assert "NpzDataset" in r
        assert "n=3" in r

    def test_mask_to_dataset_filters_rows(self, minimal_npz):
        ds   = NpzDataset(minimal_npz)
        mask = np.array([True, False, True])
        sub  = ds._mask_to_dataset(mask)
        assert len(sub) == 2
        assert np.allclose(sub["X"], [1.0, 3.0])

    def test_mask_to_dataset_returns_same_type(self, minimal_npz):
        ds  = NpzDataset(minimal_npz)
        sub = ds._mask_to_dataset(np.array([True, False, True]))
        assert type(sub) is NpzDataset

    def test_filter_column(self, minimal_npz):
        ds  = NpzDataset(minimal_npz)
        sub = ds.filter_column("SEASON", [2020])
        assert len(sub) == 2
        assert all(sub["SEASON"] == 2020)

    def test_to_dataframe_shape(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        df = ds.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 4)

    def test_to_dataframe_iso_time_is_datetime(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        df = ds.to_dataframe()
        assert pd.api.types.is_datetime64_any_dtype(df["ISO_TIME"])

    def test_to_dataframe_col_subset(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        df = ds.to_dataframe(cols=["X", "Y"])
        assert list(df.columns) == ["X", "Y"]

    def test_to_Xy_shapes(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        X, y = ds.to_Xy(target_cols=["Y"], feature_cols=["X"])
        assert X.shape == (3, 1)
        assert y.shape == (3, 1)

    def test_to_Xy_dtype_float32(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        X, y = ds.to_Xy(target_cols=["Y"], feature_cols=["X"])
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_split_raises_not_implemented(self, minimal_npz):
        ds = NpzDataset(minimal_npz)
        with pytest.raises(NotImplementedError):
            ds.split("train")

    def test_summary_runs(self, minimal_npz, capsys):
        NpzDataset(minimal_npz).summary()
        out = capsys.readouterr().out
        assert "NpzDataset" in out
        assert "rows" in out


# ===========================================================================
# IBTrACSDataset
# ===========================================================================

class TestIBTrACSDatasetInit:

    def test_loads_without_multi_storm(self, ds_no_multi):
        assert len(ds_no_multi) == 40

    def test_loads_with_multi_storm(self, ds_full):
        assert len(ds_full) == 40

    def test_repr(self, ds_full):
        r = repr(ds_full)
        assert "IBTrACSDataset" in r
        assert "n=40" in r
        assert "SIDs=" in r

    def test_n_sids(self, ds_full):
        # 2 SIDs per season × 4 seasons = 8
        assert ds_full.n_sids == 8

    def test_iso_time_is_datetimeindex(self, ds_full):
        assert isinstance(ds_full.iso_time, pd.DatetimeIndex)
        assert len(ds_full.iso_time) == 40

    def test_seasons_dtype_int(self, ds_full):
        assert ds_full.seasons.dtype in (np.int32, np.int64)

    def test_is_multi_storm_raises_without_path(self, ds_no_multi):
        with pytest.raises(ValueError, match="multi_storm_path"):
            _ = ds_no_multi.is_multi_storm

    def test_is_multi_storm_shape(self, ds_full):
        mask = ds_full.is_multi_storm
        assert mask.shape == (40,)
        assert mask.dtype == bool

    def test_is_multi_storm_count(self, ds_full, ibtracs_paths):
        _, _, _, _, multi_times = ibtracs_paths
        assert ds_full.is_multi_storm.sum() == len(multi_times)

    def test_issubclass_of_npzdataset(self):
        assert issubclass(IBTrACSDataset, NpzDataset)


class TestIBTrACSFilters:

    def test_filter_seasons_count(self, ds_full):
        sub = ds_full.filter_seasons([2019])
        assert len(sub) == 10

    def test_filter_seasons_values(self, ds_full):
        sub = ds_full.filter_seasons([2019, 2020])
        assert set(sub.seasons.tolist()) == {2019, 2020}

    def test_filter_sids(self, ds_full):
        sub = ds_full.filter_sids(["2019A"])
        assert len(sub) == 5
        assert all(sub["SID"] == "2019A")

    def test_filter_single_storm_removes_multi(self, ds_full, ibtracs_paths):
        _, _, _, _, multi_times = ibtracs_paths
        sub = ds_full.filter_single_storm()
        assert len(sub) == 40 - len(multi_times)
        assert sub.is_multi_storm.sum() == 0

    def test_filter_multi_storm_keeps_only_multi(self, ds_full, ibtracs_paths):
        _, _, _, _, multi_times = ibtracs_paths
        sub = ds_full.filter_multi_storm()
        assert len(sub) == len(multi_times)
        assert sub.is_multi_storm.all()

    def test_filter_preserves_multi_times(self, ds_full):
        """_multi_times must survive a mask so split() still works on subsets."""
        sub = ds_full.filter_seasons([2019, 2020])
        assert sub._multi_times is not None
        assert sub.multi_storm_path == ds_full.multi_storm_path

    def test_filter_returns_ibtracs_type(self, ds_full):
        sub = ds_full.filter_seasons([2019])
        assert type(sub) is IBTrACSDataset


class TestIBTrACSSplits:

    def test_train_seasons(self, ds_full):
        train = ds_full.split("train")
        assert set(train.seasons.tolist()).issubset(set(IBTRACS_TRAIN_SEASONS))

    def test_train_no_multi_storm(self, ds_full):
        train = ds_full.split("train")
        assert train.is_multi_storm.sum() == 0

    def test_val_seasons(self, ds_full):
        val = ds_full.split("val")
        assert set(val.seasons.tolist()).issubset(set(IBTRACS_VAL_SEASONS))

    def test_val_no_multi_storm(self, ds_full):
        assert ds_full.split("val").is_multi_storm.sum() == 0

    def test_test_seasons(self, ds_full):
        tst = ds_full.split("test")
        assert set(tst.seasons.tolist()).issubset(set(IBTRACS_TEST_SEASONS))

    def test_test_no_multi_storm(self, ds_full):
        assert ds_full.split("test").is_multi_storm.sum() == 0

    def test_hard_test_all_multi_storm(self, ds_full, ibtracs_paths):
        _, _, _, _, multi_times = ibtracs_paths
        hard = ds_full.split("hard_test")
        assert len(hard) == len(multi_times)
        assert hard.is_multi_storm.all()

    def test_unknown_split_raises(self, ds_full):
        with pytest.raises(ValueError, match="Unknown split"):
            ds_full.split("blah")

    def test_split_without_multi_storm_raises(self, ds_no_multi):
        with pytest.raises(ValueError):
            ds_no_multi.split("train")

    def test_split_returns_ibtracs_type(self, ds_full):
        assert type(ds_full.split("train")) is IBTrACSDataset

    def test_train_val_test_disjoint(self, ds_full):
        train_times = set(ds_full.split("train")["ISO_TIME"].tolist())
        val_times   = set(ds_full.split("val")["ISO_TIME"].tolist())
        test_times  = set(ds_full.split("test")["ISO_TIME"].tolist())
        assert train_times.isdisjoint(val_times)
        assert train_times.isdisjoint(test_times)
        assert val_times.isdisjoint(test_times)


class TestIBTrACSExport:

    def test_to_dataframe(self, ds_full):
        df = ds_full.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40
        assert pd.api.types.is_datetime64_any_dtype(df["ISO_TIME"])

    def test_to_dataframe_col_subset(self, ds_full):
        df = ds_full.to_dataframe(cols=["SID", "LAT", "LON"])
        assert list(df.columns) == ["SID", "LAT", "LON"]

    def test_to_Xy_shapes(self, ds_full):
        X, y = ds_full.to_Xy(
            target_cols=IBTRACS_PRIMARY_TARGET_COLS[:2],
            feature_cols=["LAT", "LON"],
        )
        assert X.shape == (40, 2)
        assert y.shape == (40, 2)

    def test_to_Xy_dtype(self, ds_full):
        X, y = ds_full.to_Xy(
            target_cols=["USA_WIND"],
            feature_cols=["LAT", "LON"],
        )
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_summary_runs(self, ds_full, capsys):
        ds_full.summary()
        out = capsys.readouterr().out
        assert "IBTrACSDataset" in out
        assert "rows" in out
        assert "SIDs" in out
        assert "multi-storm" in out
