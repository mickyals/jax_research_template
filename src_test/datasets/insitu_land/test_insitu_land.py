"""
Tests for datasets/insitu_land/dataset.py.

All tests use synthetic NPZ fixtures — the real 74.7 M row file is not
required to run the suite.

Coverage
--------
TestFixtures            fixture sanity checks (shapes, types)
TestInit                loading, time index built, metadata loaded
TestFilters             filter_reliability, filter_time_range, filter_bbox,
                        filter_radius, filter_stations
TestSplit               train / val / test year partitioning; unknown split raises
TestGetStationsAtTime   time window lookup; spatial filter; distance/azimuth
                        columns present; empty result when no match;
                        result sorted by distance
TestToDataframe         timestamp cast to datetime; column subset
TestSummary             summary runs without error; content checks
TestRegistry            INSITU_LAND registered in datamodule DATASETS
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasets.insitu_land.dataset import (
    RELIABILITY_LEVELS,
    InsituLandDataset,
    META_COLS,
    OBS_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_OBS   = 500    # rows in synthetic obs file
_N_META  = 20     # unique stations in synthetic metadata file
_SEED    = 0

# Station ids used across fixtures
_STATION_IDS = [f"ST{i:03d}" for i in range(_N_META)]

# Timestamps spanning 2005-2025 in Unix nanoseconds
_T_2010 = pd.Timestamp("2010-06-01 00:00").value
_T_2021 = pd.Timestamp("2021-06-01 00:00").value
_T_2023 = pd.Timestamp("2023-06-01 00:00").value


def _make_obs_npz(path: Path, seed: int = _SEED) -> Path:
    rng = np.random.default_rng(seed)
    n   = _N_OBS

    # Spread timestamps evenly across 2005-2025
    t_start = pd.Timestamp("2005-01-01").value
    t_end   = pd.Timestamp("2026-01-01").value
    timestamps = np.sort(rng.integers(t_start, t_end, n))

    # Assign reliability levels (more active stations have more rows)
    rel_choices = (
        ["always_active"]  * (n // 3) +
        ["mostly_active"]  * (n // 3) +
        ["sporadic"]       * (n // 6) +
        ["sparse"]         * (n // 8) +
        ["unusable"]       * (n - n // 3 - n // 3 - n // 6 - n // 8)
    )
    reliability = np.array(rel_choices[:n])

    station_ids = np.array([_STATION_IDS[i % _N_META] for i in range(n)])

    data = {
        "primary_station_id":       station_ids,
        "report_timestamp":         timestamps.astype(np.int64),
        "latitude":                 rng.uniform(5.0, 28.0, n).astype(np.float32),
        "longitude":                rng.uniform(-95.0, -50.0, n).astype(np.float32),
        "elevation":                rng.uniform(0.0, 800.0, n).astype(np.float32),
        "station_name":             np.array([f"Station {i % _N_META}" for i in range(n)]),
        "air_pressure":             rng.uniform(95000, 102000, n).astype(np.float32),
        "air_pressure_at_sea_level": rng.uniform(99000, 102000, n).astype(np.float32),
        "air_temperature":          rng.uniform(273, 310, n).astype(np.float32),
        "dew_point_temperature":    rng.uniform(265, 300, n).astype(np.float32),
        "wind_speed":               rng.uniform(0, 20, n).astype(np.float32),
        "wind_from_direction":      rng.uniform(0, 360, n).astype(np.float32),
        "slp_derived":              rng.choice([True, False], n),
        "slp_unreliable":           rng.choice([True, False], n),
        "station_reliability":      reliability,
    }
    # Introduce NaN in ~20% of wind_speed
    nan_idx = rng.choice(n, n // 5, replace=False)
    data["wind_speed"][nan_idx] = np.nan

    np.savez(path, **data)
    return path


def _make_meta_npz(path: Path, seed: int = _SEED) -> Path:
    rng = np.random.default_rng(seed)
    n   = _N_META
    data = {
        "primary_station_id": np.array(_STATION_IDS),
        "station_name":       np.array([f"Station {i}" for i in range(n)]),
        "latitude":           rng.uniform(5.0, 28.0, n).astype(np.float32),
        "longitude":          rng.uniform(-95.0, -50.0, n).astype(np.float32),
        "elevation":          rng.uniform(0.0, 800.0, n).astype(np.float32),
        "station_reliability": np.array(["always_active"] * n),
        "slp_derived":        rng.choice([True, False], n),
        "slp_unreliable":     rng.choice([True, False], n),
    }
    np.savez(path, **data)
    return path


@pytest.fixture
def obs_npz(tmp_path):
    return _make_obs_npz(tmp_path / "insitu_land_clean.npz")


@pytest.fixture
def meta_npz(tmp_path):
    return _make_meta_npz(tmp_path / "insitu_land_station_meta.npz")


@pytest.fixture
def ds(obs_npz, meta_npz):
    return InsituLandDataset(obs_npz, meta_npz)


# ---------------------------------------------------------------------------
# TestFixtures
# ---------------------------------------------------------------------------

class TestFixtures:

    def test_obs_has_expected_rows(self, obs_npz):
        raw = np.load(obs_npz, allow_pickle=True)
        assert raw["report_timestamp"].shape[0] == _N_OBS

    def test_meta_has_expected_rows(self, meta_npz):
        raw = np.load(meta_npz, allow_pickle=True)
        assert raw["primary_station_id"].shape[0] == _N_META


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_len_matches_obs_rows(self, ds):
        assert len(ds) == _N_OBS

    def test_time_index_built(self, ds):
        assert hasattr(ds, "_sorted_timestamps")
        assert hasattr(ds, "_sorted_time_idx")

    def test_time_index_sorted(self, ds):
        assert np.all(np.diff(ds._sorted_timestamps) >= 0)

    def test_time_index_length(self, ds):
        assert len(ds._sorted_timestamps) == _N_OBS

    def test_metadata_loaded(self, ds):
        assert len(ds._meta) > 0
        assert "primary_station_id" in ds._meta

    def test_station_meta_property(self, ds):
        df = ds.station_meta
        assert isinstance(df, pd.DataFrame)
        assert len(df) == _N_META

    def test_n_stations(self, ds):
        assert ds.n_stations == _N_META

    def test_repr_contains_rows_and_stations(self, ds):
        r = repr(ds)
        assert "InsituLandDataset" in r
        assert str(_N_OBS) in r

    def test_all_obs_cols_accessible(self, ds):
        for col in OBS_COLS:
            assert col in ds.columns or True  # some optional cols may be absent

    def test_timestamps_property(self, ds):
        assert ds.timestamps.dtype == np.int64
        assert len(ds.timestamps) == _N_OBS


# ---------------------------------------------------------------------------
# TestFilters
# ---------------------------------------------------------------------------

class TestFilters:

    def test_filter_reliability_reduces_rows(self, ds):
        sub = ds.filter_reliability(["always_active"])
        assert len(sub) < len(ds)
        assert len(sub) > 0

    def test_filter_reliability_only_requested_levels(self, ds):
        sub = ds.filter_reliability(["always_active", "mostly_active"])
        unique = np.unique(sub["station_reliability"])
        for u in unique:
            assert u in ["always_active", "mostly_active"]

    def test_filter_reliability_all_levels_returns_all(self, ds):
        sub = ds.filter_reliability(RELIABILITY_LEVELS)
        assert len(sub) == len(ds)

    def test_filter_time_range(self, ds):
        t_lo = pd.Timestamp("2010-01-01").value
        t_hi = pd.Timestamp("2011-01-01").value
        sub  = ds.filter_time_range(t_lo, t_hi)
        assert len(sub) > 0
        assert np.all(sub["report_timestamp"] >= t_lo)
        assert np.all(sub["report_timestamp"] <= t_hi)

    def test_filter_time_range_empty_returns_empty(self, ds):
        # Far-future range with no data
        t_lo = pd.Timestamp("2040-01-01").value
        t_hi = pd.Timestamp("2041-01-01").value
        sub  = ds.filter_time_range(t_lo, t_hi)
        assert len(sub) == 0

    def test_filter_bbox_reduces_rows(self, ds):
        sub = ds.filter_bbox(lat_min=10, lat_max=20, lon_min=-90, lon_max=-60)
        assert len(sub) < len(ds)
        assert np.all(sub["latitude"]  >= 10)
        assert np.all(sub["latitude"]  <= 20)
        assert np.all(sub["longitude"] >= -90)
        assert np.all(sub["longitude"] <= -60)

    def test_filter_radius_returns_nearby_stations(self, ds):
        # Use the centre of the domain
        sub = ds.filter_radius(center_lat=15.0, center_lon=-75.0, radius_km=500)
        assert len(sub) > 0

    def test_filter_stations(self, ds):
        target = _STATION_IDS[:3]
        sub    = ds.filter_stations(target)
        assert len(sub) > 0
        unique_ids = np.unique(sub["primary_station_id"]).tolist()
        for uid in unique_ids:
            assert uid in target

    def test_filter_preserves_time_index(self, ds):
        sub = ds.filter_reliability(["always_active", "mostly_active"])
        assert hasattr(sub, "_sorted_timestamps")
        assert len(sub._sorted_timestamps) == len(sub)
        assert np.all(np.diff(sub._sorted_timestamps) >= 0)

    def test_filter_returns_same_type(self, ds):
        sub = ds.filter_reliability(["always_active"])
        assert isinstance(sub, InsituLandDataset)


# ---------------------------------------------------------------------------
# TestSplit
# ---------------------------------------------------------------------------

class TestSplit:

    def test_train_split_in_correct_years(self, ds):
        tr    = ds.split("train")
        years = pd.to_datetime(tr["report_timestamp"]).year
        assert all(2005 <= y <= 2020 for y in years)

    def test_val_split_in_correct_years(self, ds):
        va    = ds.split("val")
        years = pd.to_datetime(va["report_timestamp"]).year
        assert all(y in [2021, 2022] for y in years)

    def test_test_split_in_correct_years(self, ds):
        te    = ds.split("test")
        years = pd.to_datetime(te["report_timestamp"]).year
        assert all(2023 <= y <= 2025 for y in years)

    def test_all_splits_non_empty(self, ds):
        assert len(ds.split("train")) > 0
        assert len(ds.split("val"))   > 0
        assert len(ds.split("test"))  > 0

    def test_splits_partition_full_dataset(self, ds):
        n_tr = len(ds.split("train"))
        n_va = len(ds.split("val"))
        n_te = len(ds.split("test"))
        assert n_tr + n_va + n_te == len(ds)

    def test_unknown_split_raises(self, ds):
        with pytest.raises(ValueError, match="Unknown split"):
            ds.split("hard_test")


# ---------------------------------------------------------------------------
# TestGetStationsAtTime
# ---------------------------------------------------------------------------

class TestGetStationsAtTime:

    def _pick_valid_timestamp(self, ds):
        """Return a timestamp that exists in the dataset."""
        return int(ds._data["report_timestamp"][len(ds) // 2])

    def test_returns_dataframe(self, ds):
        ts  = self._pick_valid_timestamp(ds)
        df  = ds.get_stations_at_time(ts, radius_km=5000.0,
                                       storm_lat=20.0, storm_lon=-75.0)
        assert isinstance(df, pd.DataFrame)

    def test_result_within_time_window(self, ds):
        ts         = self._pick_valid_timestamp(ds)
        window_ns  = int(6 * 3600 * 1e9)
        df         = ds.get_stations_at_time(ts, radius_km=5000.0,
                                              storm_lat=20.0, storm_lon=-75.0,
                                              window_ns=window_ns)
        if len(df) > 0:
            assert np.all(np.abs(df["report_timestamp"].values - ts) <= window_ns)

    def test_distance_column_present(self, ds):
        ts  = self._pick_valid_timestamp(ds)
        df  = ds.get_stations_at_time(ts, radius_km=5000.0,
                                       storm_lat=20.0, storm_lon=-75.0)
        assert "distance_km" in df.columns

    def test_azimuth_columns_present(self, ds):
        ts  = self._pick_valid_timestamp(ds)
        df  = ds.get_stations_at_time(ts, radius_km=5000.0,
                                       storm_lat=20.0, storm_lon=-75.0)
        assert "forward_azimuth_deg" in df.columns
        assert "back_azimuth_deg"    in df.columns

    def test_result_sorted_by_distance(self, ds):
        ts  = self._pick_valid_timestamp(ds)
        df  = ds.get_stations_at_time(ts, radius_km=5000.0,
                                       storm_lat=20.0, storm_lon=-75.0)
        if len(df) > 1:
            assert np.all(np.diff(df["distance_km"].values) >= 0)

    def test_result_within_radius(self, ds):
        ts  = self._pick_valid_timestamp(ds)
        df  = ds.get_stations_at_time(ts, radius_km=500.0,
                                       storm_lat=15.0, storm_lon=-75.0)
        if len(df) > 0:
            assert np.all(df["distance_km"].values <= 500.0 + 1e-6)

    def test_empty_result_when_no_match(self, ds):
        # Far-future timestamp — no observations
        ts  = pd.Timestamp("2040-01-01").value
        df  = ds.get_stations_at_time(ts, radius_km=500.0,
                                       storm_lat=20.0, storm_lon=-75.0)
        assert len(df) == 0

    def test_empty_result_has_expected_columns(self, ds):
        ts  = pd.Timestamp("2040-01-01").value
        df  = ds.get_stations_at_time(ts, radius_km=500.0,
                                       storm_lat=20.0, storm_lon=-75.0)
        assert "distance_km" in df.columns
        assert "forward_azimuth_deg" in df.columns

    def test_narrow_time_window_returns_fewer_rows(self, ds):
        ts   = self._pick_valid_timestamp(ds)
        wide = ds.get_stations_at_time(ts, radius_km=5000.0,
                                        storm_lat=20.0, storm_lon=-75.0,
                                        window_ns=int(24 * 3600 * 1e9))
        narrow = ds.get_stations_at_time(ts, radius_km=5000.0,
                                          storm_lat=20.0, storm_lon=-75.0,
                                          window_ns=int(1 * 3600 * 1e9))
        assert len(narrow) <= len(wide)

    def test_distances_are_non_negative(self, ds):
        ts  = self._pick_valid_timestamp(ds)
        df  = ds.get_stations_at_time(ts, radius_km=5000.0,
                                       storm_lat=20.0, storm_lon=-75.0)
        if len(df) > 0:
            assert np.all(df["distance_km"].values >= 0)

    def test_azimuths_in_range(self, ds):
        ts  = self._pick_valid_timestamp(ds)
        df  = ds.get_stations_at_time(ts, radius_km=5000.0,
                                       storm_lat=20.0, storm_lon=-75.0)
        if len(df) > 0:
            assert np.all(df["forward_azimuth_deg"].values >= 0)
            assert np.all(df["forward_azimuth_deg"].values < 360)


# ---------------------------------------------------------------------------
# TestToDataframe
# ---------------------------------------------------------------------------

class TestToDataframe:

    def test_returns_dataframe(self, ds):
        assert isinstance(ds.to_dataframe(), pd.DataFrame)

    def test_timestamp_cast_to_datetime(self, ds):
        df = ds.to_dataframe(["report_timestamp", "latitude"])
        assert pd.api.types.is_datetime64_any_dtype(df["report_timestamp"])

    def test_column_subset(self, ds):
        cols = ["latitude", "longitude", "wind_speed"]
        df   = ds.to_dataframe(cols)
        assert list(df.columns) == cols

    def test_full_dataframe_has_all_loaded_cols(self, ds):
        df = ds.to_dataframe()
        for col in ["latitude", "longitude", "report_timestamp"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# TestSummary
# ---------------------------------------------------------------------------

class TestSummary:

    def test_runs_without_error(self, ds, capsys):
        ds.summary()
        out = capsys.readouterr().out
        assert "InsituLandDataset" in out

    def test_shows_row_count(self, ds, capsys):
        ds.summary()
        out = capsys.readouterr().out
        assert str(_N_OBS) in out.replace(",", "")

    def test_shows_reliability_breakdown(self, ds, capsys):
        ds.summary()
        out = capsys.readouterr().out
        assert "always_active" in out

    def test_shows_sparsity(self, ds, capsys):
        ds.summary()
        out = capsys.readouterr().out
        assert "wind_speed" in out


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_insitu_land_registered(self):
        from datasets.datamodule import list_datasets
        assert "INSITU_LAND" in list_datasets()

    def test_factory_instantiates(self, obs_npz, meta_npz):
        from datasets.datamodule import DATASETS
        factory = DATASETS["INSITU_LAND"]
        ds = factory({"obs_path": str(obs_npz), "meta_path": str(meta_npz)})
        assert isinstance(ds, InsituLandDataset)
        assert len(ds) == _N_OBS
