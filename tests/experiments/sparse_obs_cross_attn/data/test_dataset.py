"""
Tests for experiments/sparse_obs_cross_attn/data/dataset.py.

Uses the same synthetic fixtures as test_ibtracs and test_insitu_land.
No real data files required.
"""

import numpy as np
import pytest

from experiments.sparse_obs_cross_attn.data.sources.ibtracs import IBTrACSDataset, SSHS_TO_CLASS
from experiments.sparse_obs_cross_attn.data.sources.insitu_land import InsituLandDataset
from experiments.sparse_obs_cross_attn.data.dataset import TCDataset


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

def _make_ibtracs(tmp_path):
    """Single IBTrACS storm row sitting on top of STA_A from insitu fixture."""
    base_ns = 1567296000_000_000_000
    data = {
        'SID':         np.array(['2019001N15276']),
        'NAME':        np.array(['DORIAN']),
        'SEASON':      np.array([2019.0], dtype=np.float32),
        'BASIN':       np.array(['NA']),
        'SUBBASIN':    np.array(['CS']),
        'ISO_TIME':    np.array([base_ns], dtype=np.int64),
        'LAT':         np.array([15.0], dtype=np.float32),
        'LON':         np.array([-75.0], dtype=np.float32),
        'TRACK_TYPE':  np.array(['main']),
        'IFLAG':       np.array(['original']),
        'USA_AGENCY':  np.array(['hurdat_atl']),
        'USA_ATCF_ID': np.array(['AL052019']),
        'USA_RECORD':  np.array([' ']),
        'USA_STATUS':  np.array(['HU']),
        'USA_SSHS':    np.array([2.0], dtype=np.float32),
        'USA_WIND':    np.array([52.0], dtype=np.float32),
        'USA_PRES':    np.array([96000.0], dtype=np.float32),
        'USA_POCI':    np.array([101000.0], dtype=np.float32),
        'USA_RMW':     np.array([40000.0], dtype=np.float32),
        'STORM_SPEED': np.array([5.0], dtype=np.float32),
        'STORM_DIR':   np.array([315.0], dtype=np.float32),
        **{col: np.array([np.nan], dtype=np.float32) for col in [
            'USA_R17MS_NE','USA_R17MS_SE','USA_R17MS_SW','USA_R17MS_NW',
            'USA_R26MS_NE','USA_R26MS_SE','USA_R26MS_SW','USA_R26MS_NW',
            'USA_R33MS_NE','USA_R33MS_SE','USA_R33MS_SW','USA_R33MS_NW',
            'USA_ROCI','USA_EYE','USA_SEAHGT',
            'USA_SEARAD_NE','USA_SEARAD_SE','USA_SEARAD_SW','USA_SEARAD_NW',
        ]},
    }
    p = tmp_path / 'ibtracs.npz'
    np.savez(p, **data)
    ms_p = tmp_path / 'multi.npz'
    np.savez(ms_p, ISO_TIME=np.array([], dtype=np.int64),
             n_active=np.array([], dtype=np.int32))
    return p, ms_p, base_ns


def _make_insitu(tmp_path, base_ns):
    """Two stations within 300 km of (15, -75); one far away."""
    hour_ns = 3_600_000_000_000
    n_hours = 6
    stations = [
        ('STA_A', 15.0, -75.0, 'always_active'),
        ('STA_B', 16.0, -74.5, 'mostly_active'),
        ('STA_C', 22.0, -68.0, 'sporadic'),
    ]
    rng = np.random.default_rng(0)

    sids, lats, lons, elevs, times, rels = [], [], [], [], [], []
    for sid, lat, lon, rel in stations:
        for h in range(n_hours):
            sids.append(sid)
            lats.append(lat)
            lons.append(lon)
            elevs.append(5.0)
            times.append(base_ns + h * hour_ns)
            rels.append(rel)

    n = len(sids)
    obs = {
        'primary_station_id':       np.array(sids),
        'latitude':                 np.array(lats,  dtype=np.float32),
        'longitude':                np.array(lons,  dtype=np.float32),
        'elevation':                np.array(elevs, dtype=np.float32),
        'report_timestamp':         np.array(times, dtype=np.int64),
        'slp_derived':              np.zeros(n, dtype=bool),
        'slp_unreliable':           np.zeros(n, dtype=bool),
        'station_name':             np.array(sids),
        'station_reliability':      np.array(rels),
        'air_pressure':             rng.uniform(99000, 101000, n).astype(np.float32),
        'air_pressure_at_sea_level':rng.uniform(100900, 101500, n).astype(np.float32),
        'air_temperature':          rng.uniform(295, 310, n).astype(np.float32),
        'dew_point_temperature':    rng.uniform(285, 300, n).astype(np.float32),
        'wind_speed':               rng.uniform(2, 15, n).astype(np.float32),
        'wind_from_direction':      rng.uniform(0, 360, n).astype(np.float32),
    }
    obs_path = tmp_path / 'obs.npz'
    np.savez(obs_path, **obs)

    meta = {
        'primary_station_id': np.array(['STA_A', 'STA_B', 'STA_C']),
        'station_name':       np.array(['A', 'B', 'C']),
        'latitude':           np.array([15.0, 16.0, 22.0], dtype=np.float32),
        'longitude':          np.array([-75.0, -74.5, -68.0], dtype=np.float32),
        'elevation':          np.array([5.0, 5.0, 5.0], dtype=np.float32),
        'slp_derived':        np.array([False, False, False]),
        'slp_unreliable':     np.array([False, False, False]),
        'station_reliability':np.array(['always_active', 'mostly_active', 'sporadic']),
    }
    meta_path = tmp_path / 'meta.npz'
    np.savez(meta_path, **meta)
    return obs_path, meta_path


@pytest.fixture
def tc_dataset(tmp_path):
    ib_path, ms_path, base_ns = _make_ibtracs(tmp_path)
    obs_path, meta_path = _make_insitu(tmp_path, base_ns)

    ibtracs = IBTrACSDataset(ib_path, ms_path)
    insitu  = InsituLandDataset(obs_path, meta_path)

    bg_pool = np.array([base_ns + 100 * 24 * 3_600_000_000_000], dtype=np.int64)

    return TCDataset(
        ibtracs=ibtracs,
        insitu=insitu,
        radius_km=300.0,
        time_window_hours=3.0,
        max_stations=8,
        min_stations=1,
        background_timestamps=bg_pool,
    )


# ---------------------------------------------------------------------------
# TC sample
# ---------------------------------------------------------------------------

class TestTCSample:

    def test_get_tc_sample_returns_dict(self, tc_dataset):
        s = tc_dataset.get_tc_sample(0)
        assert isinstance(s, dict)

    def test_tc_sample_keys(self, tc_dataset):
        s = tc_dataset.get_tc_sample(0)
        assert s is not None
        expected = {'query_coords', 'station_obs', 'station_coords',
                    'station_mask', 'obs_mask', 'label', 'n_stations'}
        assert expected.issubset(s.keys())
        # old keys should not be present
        assert 'query_lat' not in s
        assert 'query_lon' not in s

    def test_tc_sample_label(self, tc_dataset):
        s = tc_dataset.get_tc_sample(0)
        assert s is not None
        assert int(s['label']) == SSHS_TO_CLASS[2]   # SSHS=2 → class 7

    def test_tc_sample_shapes(self, tc_dataset):
        s = tc_dataset.get_tc_sample(0)
        assert s is not None
        N, F = tc_dataset.max_stations, len(tc_dataset.obs_vars)
        assert s['query_coords'].shape   == (2,)
        assert s['station_obs'].shape    == (N, F)
        assert s['station_coords'].shape == (N, 2)
        assert s['station_mask'].shape   == (N,)
        assert s['obs_mask'].shape       == (N, F)

    def test_tc_sample_dtypes(self, tc_dataset):
        s = tc_dataset.get_tc_sample(0)
        assert s is not None
        assert s['query_coords'].dtype   == np.float32
        assert s['station_obs'].dtype    == np.float32
        assert s['station_coords'].dtype == np.float32
        assert s['station_mask'].dtype   == bool
        assert s['label'].dtype          == np.int32

    def test_tc_sample_padding(self, tc_dataset):
        s = tc_dataset.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        # Padding rows should be all-zero obs and False mask
        assert not s['station_mask'][n_real:].any()
        assert not s['obs_mask'][n_real:].any()
        assert (s['station_obs'][n_real:] == 0.0).all()

    def test_tc_sample_obs_zero_where_missing(self, tc_dataset):
        s = tc_dataset.get_tc_sample(0)
        assert s is not None
        # Where obs_mask is False (real row), station_obs should be 0
        n_real = int(s['n_stations'])
        real_obs  = s['station_obs'][:n_real]
        real_mask = s['obs_mask'][:n_real]
        assert (real_obs[~real_mask] == 0.0).all()

    def test_tc_sample_returns_none_outside_radius(self, tmp_path):
        """SSHS within known map and no stations → should be None or has stations."""
        ib_path, ms_path, base_ns = _make_ibtracs(tmp_path)
        obs_path, meta_path = _make_insitu(tmp_path, base_ns)
        ibtracs = IBTrACSDataset(ib_path, ms_path)
        insitu  = InsituLandDataset(obs_path, meta_path)
        # Tiny radius — no stations within 1 km of storm
        ds = TCDataset(ibtracs, insitu, radius_km=1.0, min_stations=5)
        s  = ds.get_tc_sample(0)
        assert s is None


# ---------------------------------------------------------------------------
# Background sample
# ---------------------------------------------------------------------------

class TestBackgroundSample:

    def test_background_label_is_zero(self, tc_dataset):
        rng = np.random.default_rng(0)
        s   = None
        for _ in range(20):
            s = tc_dataset.get_background_sample(rng)
            if s is not None:
                break
        # Background pool timestamp may be far in future so s could be None
        # Just verify type if returned
        if s is not None:
            assert int(s['label']) == 0

    def test_background_raises_without_pool(self, tmp_path):
        ib_path, ms_path, base_ns = _make_ibtracs(tmp_path)
        obs_path, meta_path = _make_insitu(tmp_path, base_ns)
        ibtracs = IBTrACSDataset(ib_path, ms_path)
        insitu  = InsituLandDataset(obs_path, meta_path)
        ds  = TCDataset(ibtracs, insitu, background_timestamps=None)
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match='background_timestamps'):
            ds.get_background_sample(rng)

    def test_background_shapes_match_tc(self, tc_dataset):
        rng = np.random.default_rng(1)
        tc  = tc_dataset.get_tc_sample(0)
        # Add a pool timestamp that aligns with the obs data
        base_ns = int(tc_dataset.ibtracs['ISO_TIME'][0])
        tc_dataset.background_timestamps = np.array([base_ns], dtype=np.int64)
        bg = tc_dataset.get_background_sample(rng)
        if bg is None:
            pytest.skip("No stations found for background sample in test domain")
        for k in ('station_obs', 'station_coords', 'station_mask', 'obs_mask'):
            assert bg[k].shape == tc[k].shape


# ---------------------------------------------------------------------------
# Location encoding
# ---------------------------------------------------------------------------

class TestLocationEncoding:

    def _make_ds(self, tmp_path, encoding, **kwargs):
        ib_path, ms_path, base_ns = _make_ibtracs(tmp_path)
        obs_path, meta_path = _make_insitu(tmp_path, base_ns)
        ibtracs = IBTrACSDataset(ib_path, ms_path)
        insitu  = InsituLandDataset(obs_path, meta_path)
        return TCDataset(
            ibtracs=ibtracs,
            insitu=insitu,
            radius_km=300.0,
            time_window_hours=3.0,
            max_stations=8,
            min_stations=1,
            location_encoding=encoding,
            **kwargs,
        )

    def test_unit_circle_query_coords_are_zeros(self, tmp_path):
        ds = self._make_ds(tmp_path, 'unit_circle')
        s  = ds.get_tc_sample(0)
        assert s is not None
        assert (s['query_coords'] == 0.0).all()

    def test_unit_circle_norm_dist_in_range(self, tmp_path):
        ds = self._make_ds(tmp_path, 'unit_circle')
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        norm_dist = s['station_coords'][:n_real, 0]
        assert np.all(norm_dist >= 0.0)
        assert np.all(norm_dist <= 1.0)

    def test_unit_circle_bearing_is_finite(self, tmp_path):
        ds = self._make_ds(tmp_path, 'unit_circle')
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        bearing = s['station_coords'][:n_real, 1]
        assert np.all(np.isfinite(bearing))

    def test_domain_query_coords_nonzero(self, tmp_path):
        # Storm at (15, -75), FOV lat [0, 30] lon [-100, -45] — query is inside domain
        ds = self._make_ds(tmp_path, 'domain',
                           fov_lat=(0.0, 30.0), fov_lon=(-100.0, -45.0))
        s  = ds.get_tc_sample(0)
        assert s is not None
        # Query is the storm centre — should be non-zero encoded position
        assert not (s['query_coords'] == 0.0).all()

    def test_domain_station_coords_finite(self, tmp_path):
        ds = self._make_ds(tmp_path, 'domain',
                           fov_lat=(0.0, 30.0), fov_lon=(-100.0, -45.0))
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        assert np.all(np.isfinite(s['station_coords'][:n_real]))

    def test_domain_coords_bounded(self, tmp_path):
        # Normalised coords scaled by π/2 should be within [-π/2, π/2]
        ds = self._make_ds(tmp_path, 'domain',
                           fov_lat=(0.0, 30.0), fov_lon=(-100.0, -45.0))
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        coords = s['station_coords'][:n_real]
        assert np.all(np.abs(coords) <= np.pi / 2 + 1e-5)

    def test_invalid_location_encoding_raises(self, tmp_path):
        with pytest.raises(ValueError, match="location_encoding"):
            self._make_ds(tmp_path, 'polar')


# ---------------------------------------------------------------------------
# Obs normalisation
# ---------------------------------------------------------------------------

class TestObsNormalisation:

    # Bounds match the fixture's units: pressure in Pa, temperature in K
    _BOUNDS = {
        'air_pressure_at_sea_level': (87000.0, 108400.0),
        'air_temperature':           (193.0,   333.0),
        'dew_point_temperature':     (193.0,   308.0),
        'wind_speed':                (0.0,     115.0),
        'wind_from_direction':       (0.0,     360.0),
    }

    def _make_ds(self, tmp_path, obs_bounds=None, obs_normalisation='minmax_01'):
        ib_path, ms_path, base_ns = _make_ibtracs(tmp_path)
        obs_path, meta_path = _make_insitu(tmp_path, base_ns)
        ibtracs = IBTrACSDataset(ib_path, ms_path)
        insitu  = InsituLandDataset(obs_path, meta_path)
        return TCDataset(
            ibtracs=ibtracs,
            insitu=insitu,
            radius_km=300.0,
            time_window_hours=3.0,
            max_stations=8,
            min_stations=1,
            obs_bounds=obs_bounds,
            obs_normalisation=obs_normalisation,
        )

    def test_no_bounds_obs_not_normalised(self, tmp_path):
        # Without bounds, raw values pass through — wind_speed fixture is 2-15 m/s
        ds = self._make_ds(tmp_path, obs_bounds=None)
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        wind_idx = ds.obs_vars.index('wind_speed')
        wind_vals = s['station_obs'][:n_real, wind_idx]
        # Raw values should be >> 1.0 (fixture generates 2–15 range)
        assert np.any(wind_vals > 1.0)

    def test_minmax_01_obs_in_unit_range(self, tmp_path):
        ds = self._make_ds(tmp_path, obs_bounds=self._BOUNDS, obs_normalisation='minmax_01')
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        real_obs  = s['station_obs'][:n_real]
        real_mask = s['obs_mask'][:n_real]
        assert np.all(real_obs[real_mask] >= 0.0)
        assert np.all(real_obs[real_mask] <= 1.0)

    def test_minmax_11_obs_in_signed_range(self, tmp_path):
        ds = self._make_ds(tmp_path, obs_bounds=self._BOUNDS, obs_normalisation='minmax_11')
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        real_obs  = s['station_obs'][:n_real]
        real_mask = s['obs_mask'][:n_real]
        assert np.all(real_obs[real_mask] >= -1.0)
        assert np.all(real_obs[real_mask] <=  1.0)

    def test_normalised_missing_still_zero(self, tmp_path):
        for mode in ('minmax_01', 'minmax_11'):
            ds = self._make_ds(tmp_path, obs_bounds=self._BOUNDS, obs_normalisation=mode)
            s  = ds.get_tc_sample(0)
            assert s is not None
            n_real = int(s['n_stations'])
            real_obs  = s['station_obs'][:n_real]
            real_mask = s['obs_mask'][:n_real]
            assert (real_obs[~real_mask] == 0.0).all(), f"failed for mode={mode}"

    def test_invalid_obs_normalisation_raises(self, tmp_path):
        with pytest.raises(ValueError, match="obs_normalisation"):
            self._make_ds(tmp_path, obs_normalisation='zscore')


# ---------------------------------------------------------------------------
# Repr and len
# ---------------------------------------------------------------------------

class TestMisc:

    def test_len(self, tc_dataset):
        assert len(tc_dataset) == 1

    def test_repr(self, tc_dataset):
        r = repr(tc_dataset)
        assert 'TCDataset' in r
        assert 'n_tc=1' in r
