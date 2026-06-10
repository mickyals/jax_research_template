"""
Tests for experiments/sparse_obs_cross_attn/data/sources/insitu_land.py.

All fixtures are synthetic and built in memory.
"""

import numpy as np
import pytest

from experiments.sparse_obs_cross_attn.data.sources.insitu_land import (
    InsituLandDataset, ALL_OBS_VARS, DEFAULT_OBS_VARS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_insitu_npz(tmp_path):
    """
    3 stations at fixed positions, 12 hourly obs each.
    Station IDs: 'STA_A' (always_active), 'STA_B' (mostly_active), 'STA_C' (sporadic).

    Positions chosen so that a query at (15.0, -75.0) with 300 km radius hits STA_A and
    STA_B but not STA_C, which is placed ~500 km away.
    """
    rng = np.random.default_rng(42)

    stations = [
        ('STA_A', 15.0,  -75.0, 5.0, 'always_active'),
        ('STA_B', 16.5,  -74.0, 10.0, 'mostly_active'),
        ('STA_C', 20.0,  -70.0, 50.0, 'sporadic'),    # ~560 km from query
    ]

    # Base timestamp: 2019-09-01T00:00:00 UTC in Unix-ns
    base_ns = 1567296000_000_000_000
    hour_ns = 3_600_000_000_000

    rows_per_station = 12
    n = len(stations) * rows_per_station

    sids, lats, lons, elevs, timestamps, reliabilities = [], [], [], [], [], []
    for sid, lat, lon, elev, rel in stations:
        for h in range(rows_per_station):
            sids.append(sid)
            lats.append(lat)
            lons.append(lon)
            elevs.append(elev)
            timestamps.append(base_ns + h * hour_ns)
            reliabilities.append(rel)

    obs_data = {
        'primary_station_id':       np.array(sids),
        'latitude':                 np.array(lats,       dtype=np.float32),
        'longitude':                np.array(lons,       dtype=np.float32),
        'elevation':                np.array(elevs,      dtype=np.float32),
        'report_timestamp':         np.array(timestamps, dtype=np.int64),
        'slp_derived':              np.zeros(n, dtype=bool),
        'slp_unreliable':           np.zeros(n, dtype=bool),
        'station_name':             np.array([s[0] for s in stations for _ in range(rows_per_station)]),
        'station_reliability':      np.array(reliabilities),
        'air_pressure':             rng.uniform(99000, 101000, n).astype(np.float32),
        'air_pressure_at_sea_level':rng.uniform(100900, 101500, n).astype(np.float32),
        'air_temperature':          rng.uniform(295, 310, n).astype(np.float32),
        'dew_point_temperature':    rng.uniform(285, 300, n).astype(np.float32),
        'wind_speed':               rng.uniform(2, 15, n).astype(np.float32),
        'wind_from_direction':      rng.uniform(0, 360, n).astype(np.float32),
    }
    # Introduce some NaN values in one variable for one station
    nan_mask = np.array(sids) == 'STA_B'
    obs_data['dew_point_temperature'][nan_mask] = np.nan

    obs_path = tmp_path / 'insitu_land_clean.npz'
    np.savez(obs_path, **obs_data)

    meta_data = {
        'primary_station_id': np.array(['STA_A', 'STA_B', 'STA_C']),
        'station_name':       np.array(['Station A', 'Station B', 'Station C']),
        'latitude':           np.array([15.0, 16.5, 20.0], dtype=np.float32),
        'longitude':          np.array([-75.0, -74.0, -70.0], dtype=np.float32),
        'elevation':          np.array([5.0, 10.0, 50.0], dtype=np.float32),
        'slp_derived':        np.array([False, False, False]),
        'slp_unreliable':     np.array([False, False, False]),
        'station_reliability':np.array(['always_active', 'mostly_active', 'sporadic']),
    }
    meta_path = tmp_path / 'insitu_land_station_meta.npz'
    np.savez(meta_path, **meta_data)

    return obs_path, meta_path, base_ns, hour_ns


@pytest.fixture
def paths(tmp_path):
    return _make_insitu_npz(tmp_path)


@pytest.fixture
def ds(paths):
    obs_path, meta_path, *_ = paths
    return InsituLandDataset(obs_path, meta_path)


# ---------------------------------------------------------------------------
# Init + properties
# ---------------------------------------------------------------------------

class TestInit:

    def test_n_stations(self, ds):
        assert ds.n_stations == 3

    def test_timestamps_sorted(self, ds):
        ts = ds.timestamps
        assert np.all(ts[:-1] <= ts[1:])

    def test_timestamps_length(self, ds):
        # 3 stations × 12 hours, all unique (but same timestamp repeated per station)
        assert len(ds.timestamps) == 3 * 12

    def test_repr_does_not_raise(self, ds):
        # summary() should run without error
        ds.summary()


# ---------------------------------------------------------------------------
# Reliability filtering
# ---------------------------------------------------------------------------

class TestReliabilityFilter:

    def test_filter_always_active(self, ds):
        sub = ds.filter_reliability(['always_active'])
        assert sub.n_stations == 1

    def test_filter_always_mostly(self, ds):
        sub = ds.filter_reliability(['always_active', 'mostly_active'])
        assert sub.n_stations == 2

    def test_filter_reduces_obs_rows(self, ds):
        full_rows = len(ds.timestamps)
        sub = ds.filter_reliability(['always_active'])
        assert len(sub.timestamps) < full_rows

    def test_filter_returns_insitu_type(self, ds):
        assert type(ds.filter_reliability(['always_active'])) is InsituLandDataset


# ---------------------------------------------------------------------------
# Spatial + temporal query
# ---------------------------------------------------------------------------

class TestGetObsNear:

    def test_returns_dataframe(self, ds, paths):
        _, _, base_ns, hour_ns = paths
        df = ds.get_obs_near(15.0, -75.0, base_ns, 400.0, hour_ns, DEFAULT_OBS_VARS)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_hits_close_stations(self, ds, paths):
        _, _, base_ns, hour_ns = paths
        # 300 km radius: should hit STA_A (~0 km) and STA_B (~190 km) but not STA_C (~560 km)
        df = ds.get_obs_near(15.0, -75.0, base_ns, 300.0, hour_ns, DEFAULT_OBS_VARS)
        sids = set(df['primary_station_id'].unique())
        assert 'STA_A' in sids
        assert 'STA_B' in sids
        assert 'STA_C' not in sids

    def test_empty_when_no_stations_in_radius(self, ds, paths):
        _, _, base_ns, hour_ns = paths
        # Tiny radius in the middle of the ocean far from any station
        df = ds.get_obs_near(0.0, -50.0, base_ns, 10.0, hour_ns, DEFAULT_OBS_VARS)
        assert len(df) == 0

    def test_empty_when_outside_time_window(self, ds, paths):
        _, _, base_ns, hour_ns = paths
        # Timestamp 100 days after any obs
        far_future = base_ns + 100 * 24 * hour_ns
        df = ds.get_obs_near(15.0, -75.0, far_future, 500.0, hour_ns, DEFAULT_OBS_VARS)
        assert len(df) == 0

    def test_obs_vars_in_output(self, ds, paths):
        _, _, base_ns, hour_ns = paths
        vars_to_request = ['air_pressure_at_sea_level', 'air_temperature']
        df = ds.get_obs_near(15.0, -75.0, base_ns, 300.0, hour_ns, vars_to_request)
        for v in vars_to_request:
            assert v in df.columns

    def test_sorted_by_distance(self, ds, paths):
        _, _, base_ns, hour_ns = paths
        df = ds.get_obs_near(15.0, -75.0, base_ns, 300.0, hour_ns, DEFAULT_OBS_VARS)
        assert (df['distance_km'].diff().dropna() >= 0).all()

    def test_nan_preserved_for_missing_obs(self, ds, paths):
        _, _, base_ns, hour_ns = paths
        # STA_B has NaN dew_point_temperature
        df = ds.get_obs_near(15.0, -75.0, base_ns, 300.0, hour_ns, DEFAULT_OBS_VARS)
        sta_b = df[df['primary_station_id'] == 'STA_B']
        if len(sta_b) > 0:
            assert sta_b['dew_point_temperature'].isna().any()


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

class TestSplit:

    def test_split_returns_insitu_type(self, ds):
        sub = ds.split('train')
        assert type(sub) is InsituLandDataset

    def test_unknown_split_raises(self, ds):
        with pytest.raises(ValueError, match='Unknown split'):
            ds.split('bad_split')
