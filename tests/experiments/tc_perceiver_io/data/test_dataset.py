"""
Tests for experiments/tc_perceiver_io/data/dataset.py.

Uses the same synthetic fixtures as test_ibtracs and test_insitu_land.
No real data files required.
"""

import numpy as np
import pytest

from experiments.tc_perceiver_io.data.sources.ibtracs import IBTrACSDataset, status_sshs_to_class
from experiments.tc_perceiver_io.data.sources.insitu_land import InsituLandDataset
from experiments.tc_perceiver_io.data.dataset import TCDataset
from experiments.tc_perceiver_io.data.inputs import InputSpec


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
                    'station_mask', 'obs_mask', 'label', 'n_stations',
                    'sid', 'iso_time', 'query_lat', 'query_lon',
                    'n_available'}
        assert expected.issubset(s.keys())

    def test_tc_sample_label(self, tc_dataset):
        s = tc_dataset.get_tc_sample(0)
        assert s is not None
        # Fixture row: USA_STATUS='HU', USA_SSHS=2 → Category 2 = class 5
        assert int(s['label']) == status_sshs_to_class('HU', 2)
        assert int(s['label']) == 5


# ---------------------------------------------------------------------------
# Sample metadata (decision 13) + post-dedup n_available (decisions 9/10)
# ---------------------------------------------------------------------------

class TestSampleMetadata:

    def test_tc_sample_metadata_values(self, tc_dataset):
        base_ns = 1567296000_000_000_000
        s = tc_dataset.get_tc_sample(0)
        assert s['sid'] == '2019001N15276'
        assert int(s['iso_time']) == base_ns
        assert float(s['query_lat']) == pytest.approx(15.0)
        assert float(s['query_lon']) == pytest.approx(-75.0)

    def test_n_available_is_post_dedup_station_count(self, tc_dataset):
        # ±3 h window over hourly stations: several reports per station in
        # the window, but only STA_A and STA_B lie within 300 km — after
        # per-station dedup exactly 2 candidates remain.
        s = tc_dataset.get_tc_sample(0)
        assert int(s['n_available']) == 2
        assert int(s['n_available']) >= int(s['n_stations'])

    def test_background_sample_has_null_sid(self, tc_dataset):
        base_ns = 1567296000_000_000_000
        hour_ns = 3_600_000_000_000
        # Pure assembly: position and timestamp are arguments
        s = tc_dataset.get_background_sample(
            15.5, -75.0, base_ns + 2 * hour_ns,
        )
        assert s is not None
        assert s['sid'] is None
        assert int(s['iso_time']) == base_ns + 2 * hour_ns
        assert float(s['query_lat']) == pytest.approx(15.5)
        assert float(s['query_lon']) == pytest.approx(-75.0)

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
        base_ns = int(tc_dataset.ibtracs['ISO_TIME'][0])
        s = tc_dataset.get_background_sample(15.0, -75.0, base_ns)
        assert s is not None
        assert int(s['label']) == 0

    def test_background_is_pure_assembly_without_pool(self, tmp_path):
        # Pool checks belong to the loader — the dataset assembles a
        # sample from explicit (lat, lon, ts) even with no pool attached.
        ib_path, ms_path, base_ns = _make_ibtracs(tmp_path)
        obs_path, meta_path = _make_insitu(tmp_path, base_ns)
        ibtracs = IBTrACSDataset(ib_path, ms_path)
        insitu  = InsituLandDataset(obs_path, meta_path)
        ds = TCDataset(ibtracs, insitu, radius_km=300.0,
                       time_window_hours=3.0, background_timestamps=None)
        s  = ds.get_background_sample(15.0, -75.0, base_ns)
        assert s is not None
        assert s['sid'] is None

    def test_background_shapes_match_tc(self, tc_dataset):
        tc      = tc_dataset.get_tc_sample(0)
        base_ns = int(tc_dataset.ibtracs['ISO_TIME'][0])
        bg = tc_dataset.get_background_sample(15.0, -75.0, base_ns)
        assert bg is not None
        for k in ('station_obs', 'station_coords', 'station_mask', 'obs_mask'):
            assert bg[k].shape == tc[k].shape


# ---------------------------------------------------------------------------
# Location encoding
# ---------------------------------------------------------------------------

class TestLocationEncoding:

    def _make_ds(self, tmp_path, encoding,
                 fov_lat=(0.0, 30.0), fov_lon=(-100.0, -45.0)):
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
            inputs=InputSpec(
                location_encoding=encoding, fov_lat=fov_lat, fov_lon=fov_lon,
            ),
        )

    def test_unit_circle_query_coords_are_zeros(self, tmp_path):
        ds = self._make_ds(tmp_path, 'unit_circle')
        s  = ds.get_tc_sample(0)
        assert s is not None
        assert (s['query_coords'] == 0.0).all()

    def test_unit_circle_coords_within_unit_disk(self, tmp_path):
        ds = self._make_ds(tmp_path, 'unit_circle')
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        xy = s['station_coords'][:n_real]              # local (x, y)
        assert np.all(np.isfinite(xy))
        assert np.all(np.hypot(xy[:, 0], xy[:, 1]) <= 1.0 + 1e-5)

    def test_unit_circle_xy_matches_distance(self, tmp_path):
        # hypot(x, y) recovers the normalised haversine distance
        ds = self._make_ds(tmp_path, 'unit_circle')
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        xy = s['station_coords'][:n_real]
        df = ds.insitu.get_obs_near(
            query_lat=15.0, query_lon=-75.0,
            timestamp_ns=int(ds.ibtracs['ISO_TIME'][0]),
            radius_km=300.0, window_ns=ds.window_ns,
            obs_vars=ds._fetch_vars,
        )
        expected = np.clip(
            df['distance_km'].to_numpy()[:n_real] / 300.0, 0.0, 1.0
        )
        assert np.allclose(np.hypot(xy[:, 0], xy[:, 1]), expected, atol=1e-3)

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
            inputs=InputSpec(
                obs_bounds=obs_bounds, normalisation=obs_normalisation,
            ),
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


# ---------------------------------------------------------------------------
# Derived wind components (decision 18)
# ---------------------------------------------------------------------------

class TestWindDecomposition:

    _WIND_VARS = ['air_pressure_at_sea_level', 'wind_east', 'wind_north']
    _BOUNDS = {
        'air_pressure_at_sea_level': (87000.0, 108400.0),
        'wind_east':                 (-115.0,  115.0),
        'wind_north':                (-115.0,  115.0),
    }

    def _make_ds(self, tmp_path, wind_speed, wind_dir,
                 obs_bounds=None, obs_normalisation='minmax_01'):
        """TCDataset over a fixture with controlled wind values on every row."""
        ib_path, ms_path, base_ns = _make_ibtracs(tmp_path)
        obs_path, meta_path = _make_insitu(tmp_path, base_ns)
        # Overwrite the wind columns with controlled values
        raw = dict(np.load(obs_path, allow_pickle=True))
        n = len(raw['wind_speed'])
        raw['wind_speed']          = np.full(n, wind_speed, dtype=np.float32)
        raw['wind_from_direction'] = np.full(n, wind_dir,   dtype=np.float32)
        np.savez(obs_path, **raw)

        ibtracs = IBTrACSDataset(ib_path, ms_path)
        insitu  = InsituLandDataset(obs_path, meta_path)
        return TCDataset(
            ibtracs=ibtracs,
            insitu=insitu,
            radius_km=300.0,
            time_window_hours=3.0,
            max_stations=8,
            min_stations=1,
            inputs=InputSpec(
                obs_vars=tuple(self._WIND_VARS),
                obs_bounds=obs_bounds,
                normalisation=obs_normalisation,
            ),
        )

    def test_fetch_vars_expand_derived_names(self, tmp_path):
        ds = self._make_ds(tmp_path, 10.0, 90.0)
        assert ds._fetch_vars == [
            'air_pressure_at_sea_level', 'wind_speed', 'wind_from_direction'
        ]
        assert ds.obs_vars == self._WIND_VARS

    def test_components_match_decomposition(self, tmp_path):
        # Wind 10 m/s FROM east → u = -10, v = 0
        ds = self._make_ds(tmp_path, 10.0, 90.0)
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        u = s['station_obs'][:n_real, self._WIND_VARS.index('wind_east')]
        v = s['station_obs'][:n_real, self._WIND_VARS.index('wind_north')]
        assert np.allclose(u, -10.0, atol=1e-4)
        assert np.allclose(v, 0.0, atol=1e-4)
        assert s['obs_mask'][:n_real].all()

    def test_components_normalised_minmax_11(self, tmp_path):
        # u = -10 m/s over [-115, 115] → -10/115 in minmax_11
        ds = self._make_ds(tmp_path, 10.0, 90.0,
                           obs_bounds=self._BOUNDS, obs_normalisation='minmax_11')
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        u = s['station_obs'][:n_real, self._WIND_VARS.index('wind_east')]
        v = s['station_obs'][:n_real, self._WIND_VARS.index('wind_north')]
        assert np.allclose(u, -10.0 / 115.0, atol=1e-4)
        assert np.allclose(v, 0.0, atol=1e-4)

    def test_calm_with_missing_direction_kept_as_zero_vector(self, tmp_path):
        # speed 0 + NaN direction must NOT become missing — calm rule
        ds = self._make_ds(tmp_path, 0.0, np.nan,
                           obs_bounds=self._BOUNDS, obs_normalisation='minmax_11')
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        for var in ('wind_east', 'wind_north'):
            col = self._WIND_VARS.index(var)
            assert s['obs_mask'][:n_real, col].all()
            # symmetric bounds → 0 m/s normalises to exactly 0
            assert np.allclose(s['station_obs'][:n_real, col], 0.0, atol=1e-7)

    def test_noncalm_missing_direction_becomes_missing(self, tmp_path):
        # speed 5 + NaN direction → unknown vector → masked out
        ds = self._make_ds(tmp_path, 5.0, np.nan)
        s  = ds.get_tc_sample(0)
        assert s is not None
        n_real = int(s['n_stations'])
        for var in ('wind_east', 'wind_north'):
            col = self._WIND_VARS.index(var)
            assert not s['obs_mask'][:n_real, col].any()
            assert (s['station_obs'][:n_real, col] == 0.0).all()


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


# ---------------------------------------------------------------------------
# Station selection (nearest vs random; train-only view augmentation)
# ---------------------------------------------------------------------------

def _dense_insitu(tmp_path, base_ns):
    """Five stations all within ~30 km of (15, -75) — a candidate pool larger
    than the small max_stations used below, so the selection policy bites."""
    offs = [(0.0, 0.0), (0.2, 0.0), (-0.2, 0.0), (0.0, 0.2), (0.0, -0.2)]
    sids = [f'STA_{i}' for i in range(len(offs))]
    rng  = np.random.default_rng(0)
    cols = {k: [] for k in (
        'primary_station_id', 'latitude', 'longitude', 'elevation',
        'report_timestamp', 'station_name', 'station_reliability')}
    for sid, (dla, dlo) in zip(sids, offs):
        cols['primary_station_id'].append(sid)
        cols['latitude'].append(15.0 + dla)
        cols['longitude'].append(-75.0 + dlo)
        cols['elevation'].append(5.0)
        cols['report_timestamp'].append(base_ns)
        cols['station_name'].append(sid)
        cols['station_reliability'].append('always_active')
    n = len(sids)
    obs = {
        'primary_station_id':        np.array(cols['primary_station_id']),
        'latitude':                  np.array(cols['latitude'],  dtype=np.float32),
        'longitude':                 np.array(cols['longitude'], dtype=np.float32),
        'elevation':                 np.array(cols['elevation'], dtype=np.float32),
        'report_timestamp':          np.array(cols['report_timestamp'], dtype=np.int64),
        'slp_derived':               np.zeros(n, dtype=bool),
        'slp_unreliable':            np.zeros(n, dtype=bool),
        'station_name':              np.array(cols['station_name']),
        'station_reliability':       np.array(cols['station_reliability']),
        'air_pressure':              rng.uniform(99000, 101000, n).astype(np.float32),
        'air_pressure_at_sea_level': rng.uniform(100900, 101500, n).astype(np.float32),
        'air_temperature':           rng.uniform(295, 310, n).astype(np.float32),
        'dew_point_temperature':     rng.uniform(285, 300, n).astype(np.float32),
        'wind_speed':                rng.uniform(2, 15, n).astype(np.float32),
        'wind_from_direction':       rng.uniform(0, 360, n).astype(np.float32),
    }
    obs_path = tmp_path / 'dense_obs.npz'
    np.savez(obs_path, **obs)
    meta = {
        'primary_station_id': np.array(sids),
        'station_name':       np.array(sids),
        'latitude':           np.array(cols['latitude'],  dtype=np.float32),
        'longitude':          np.array(cols['longitude'], dtype=np.float32),
        'elevation':          np.array([5.0] * n, dtype=np.float32),
        'slp_derived':        np.zeros(n, dtype=bool),
        'slp_unreliable':     np.zeros(n, dtype=bool),
        'station_reliability':np.array(['always_active'] * n),
    }
    meta_path = tmp_path / 'dense_meta.npz'
    np.savez(meta_path, **meta)
    return obs_path, meta_path


def _dense_dataset(tmp_path, max_stations):
    ib_path, ms_path, base_ns = _make_ibtracs(tmp_path)
    obs_path, meta_path = _dense_insitu(tmp_path, base_ns)
    return TCDataset(
        ibtracs=IBTrACSDataset(ib_path, ms_path),
        insitu=InsituLandDataset(obs_path, meta_path),
        radius_km=300.0, time_window_hours=3.0,
        max_stations=max_stations, min_stations=1,
    )


def _coord_set(s):
    """Identity of the chosen REAL stations: set of rounded (x, y) coords."""
    m = np.asarray(s['station_mask'])
    return frozenset(map(tuple, np.round(np.asarray(s['station_coords'])[m], 4)))


class TestStationSelection:

    def test_pool_exceeds_cap(self, tmp_path):
        # Sanity: 5 candidates > max_stations=2, so selection actually bites.
        ds = _dense_dataset(tmp_path, max_stations=2)
        s  = ds.get_tc_sample(0)
        assert int(s['n_available']) == 5
        assert int(s['n_stations']) == 2

    def test_nearest_is_deterministic(self, tmp_path):
        ds = _dense_dataset(tmp_path, max_stations=2)
        a = _coord_set(ds.get_tc_sample(0, station_selection='nearest'))
        b = _coord_set(ds.get_tc_sample(0, station_selection='nearest'))
        assert a == b

    def test_random_varies_across_rngs(self, tmp_path):
        ds = _dense_dataset(tmp_path, max_stations=2)
        seen = set()
        for seed in range(25):
            s = ds.get_tc_sample(0, station_selection='random',
                                 rng=np.random.default_rng(seed))
            cs = _coord_set(s)
            assert len(cs) == 2                       # always max_stations kept
            seen.add(cs)
        assert len(seen) > 1                          # different subsets appear

    def test_random_without_rng_falls_back_to_nearest(self, tmp_path):
        ds = _dense_dataset(tmp_path, max_stations=2)
        rand_norng = _coord_set(ds.get_tc_sample(0, station_selection='random'))
        nearest    = _coord_set(ds.get_tc_sample(0, station_selection='nearest'))
        assert rand_norng == nearest

    def test_no_op_when_pool_within_cap(self, tmp_path):
        # 5 candidates <= max_stations=8 → random keeps all, same as nearest.
        ds = _dense_dataset(tmp_path, max_stations=8)
        nearest = _coord_set(ds.get_tc_sample(0, station_selection='nearest'))
        rand    = _coord_set(ds.get_tc_sample(0, station_selection='random',
                                              rng=np.random.default_rng(3)))
        assert nearest == rand and len(nearest) == 5


# ---------------------------------------------------------------------------
# Storm proximity (spatial background validation)
# ---------------------------------------------------------------------------

class TestStormWithin:
    """TCDataset.storm_within — the per-draw check behind spatial backgrounds.

    The tc_dataset fixture has one storm at (15.0, -75.0) at base_ns.
    """

    _HOUR_NS = 3_600_000_000_000

    def test_same_point_and_time_is_within(self, tc_dataset):
        ts = int(tc_dataset._time[0])
        assert tc_dataset.storm_within(
            15.0, -75.0, ts, radius_km=100.0, time_tol_ns=self._HOUR_NS) is True

    def test_far_point_is_not_within(self, tc_dataset):
        # ~5500 km away — well outside any sane radius at the same time.
        ts = int(tc_dataset._time[0])
        assert tc_dataset.storm_within(
            40.0, -10.0, ts, radius_km=500.0, time_tol_ns=self._HOUR_NS) is False

    def test_far_time_is_not_within(self, tc_dataset):
        # Same point, 100 days later — outside the time tolerance.
        ts = int(tc_dataset._time[0]) + 100 * 24 * self._HOUR_NS
        assert tc_dataset.storm_within(
            15.0, -75.0, ts, radius_km=100.0, time_tol_ns=self._HOUR_NS) is False

    def test_radius_boundary(self, tc_dataset):
        # A point just inside vs outside ~110 km (1 deg latitude) of the storm.
        ts = int(tc_dataset._time[0])
        assert tc_dataset.storm_within(
            16.0, -75.0, ts, radius_km=150.0, time_tol_ns=self._HOUR_NS) is True
        assert tc_dataset.storm_within(
            16.0, -75.0, ts, radius_km=50.0,  time_tol_ns=self._HOUR_NS) is False

    def test_returns_plain_bool(self, tc_dataset):
        ts = int(tc_dataset._time[0])
        out = tc_dataset.storm_within(
            15.0, -75.0, ts, radius_km=100.0, time_tol_ns=self._HOUR_NS)
        assert isinstance(out, bool)
