"""
Tests for experiments/tc_perceiver_io/data/datamodule.py.

Verifies background pool construction, collation, and loader batch shapes.
No real data files required.
"""

import numpy as np
import pytest

from experiments.tc_perceiver_io.data.sources.ibtracs import IBTrACSDataset
from experiments.tc_perceiver_io.data.sources.insitu_land import InsituLandDataset
from experiments.tc_perceiver_io.data.dataset import TCDataset
from experiments.tc_perceiver_io.data.datamodule import (
    _build_background_pool,
    _collate,
    TCLoader,
)


# ---------------------------------------------------------------------------
# Minimal synthetic dataset helpers
# ---------------------------------------------------------------------------

def _make_ibtracs(tmp_path, seasons=(2019,), n_per_season=4):
    base_ns = 1_567_296_000_000_000_000
    hour_ns = 3_600_000_000_000

    sids, season_arr, iso_times, sshs_arr = [], [], [], []
    for s in seasons:
        for i in range(n_per_season):
            sids.append(f'{s}A')
            season_arr.append(float(s))
            iso_times.append(base_ns + (i + (s - 2019) * 100) * hour_ns)
            sshs_arr.append(1.0)  # Cat-1

    n = len(sids)
    rng = np.random.default_rng(0)
    data = {
        'SID':         np.array(sids),
        'NAME':        np.full(n, 'TEST'),
        'SEASON':      np.array(season_arr, dtype=np.float32),
        'BASIN':       np.full(n, 'NA'),
        'SUBBASIN':    np.full(n, 'CS'),
        'ISO_TIME':    np.array(iso_times, dtype=np.int64),
        'LAT':         np.full(n, 15.0, dtype=np.float32),
        'LON':         np.full(n, -75.0, dtype=np.float32),
        'TRACK_TYPE':  np.full(n, 'main'),
        'IFLAG':       np.full(n, 'original'),
        'USA_AGENCY':  np.full(n, 'hurdat_atl'),
        'USA_ATCF_ID': np.full(n, 'AL052019'),
        'USA_RECORD':  np.full(n, ' '),
        'USA_STATUS':  np.full(n, 'HU'),
        'USA_SSHS':    np.array(sshs_arr, dtype=np.float32),
        'USA_WIND':    rng.uniform(30, 60, n).astype(np.float32),
        'USA_PRES':    rng.uniform(95000, 100000, n).astype(np.float32),
        'USA_POCI':    rng.uniform(100000, 102000, n).astype(np.float32),
        'USA_RMW':     rng.uniform(20000, 60000, n).astype(np.float32),
        'STORM_SPEED': rng.uniform(2, 8, n).astype(np.float32),
        'STORM_DIR':   rng.uniform(0, 360, n).astype(np.float32),
        **{c: np.full(n, np.nan, dtype=np.float32) for c in [
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
    return p, ms_p, np.array(iso_times, dtype=np.int64)


def _make_insitu(tmp_path, base_ns, n_hours=20):
    hour_ns = 3_600_000_000_000
    rng = np.random.default_rng(1)
    n = n_hours
    sids  = np.full(n, 'STA_A')
    times = np.array([base_ns + i * hour_ns for i in range(n)], dtype=np.int64)

    obs = {
        'primary_station_id':       sids,
        'latitude':                 np.full(n, 15.0, dtype=np.float32),
        'longitude':                np.full(n, -75.0, dtype=np.float32),
        'elevation':                np.full(n, 5.0, dtype=np.float32),
        'report_timestamp':         times,
        'slp_derived':              np.zeros(n, dtype=bool),
        'slp_unreliable':           np.zeros(n, dtype=bool),
        'station_name':             sids,
        'station_reliability':      np.full(n, 'always_active'),
        'air_pressure':             rng.uniform(99000, 101000, n).astype(np.float32),
        'air_pressure_at_sea_level':rng.uniform(100900, 101500, n).astype(np.float32),
        'air_temperature':          rng.uniform(295, 310, n).astype(np.float32),
        'dew_point_temperature':    rng.uniform(285, 300, n).astype(np.float32),
        'wind_speed':               rng.uniform(2, 15, n).astype(np.float32),
        'wind_from_direction':      rng.uniform(0, 360, n).astype(np.float32),
    }
    obs_p = tmp_path / 'obs.npz'
    np.savez(obs_p, **obs)

    meta = {
        'primary_station_id': np.array(['STA_A']),
        'station_name':       np.array(['Station A']),
        'latitude':           np.array([15.0], dtype=np.float32),
        'longitude':          np.array([-75.0], dtype=np.float32),
        'elevation':          np.array([5.0], dtype=np.float32),
        'slp_derived':        np.array([False]),
        'slp_unreliable':     np.array([False]),
        'station_reliability':np.array(['always_active']),
    }
    meta_p = tmp_path / 'meta.npz'
    np.savez(meta_p, **meta)
    return obs_p, meta_p, times


# ---------------------------------------------------------------------------
# Background pool
# ---------------------------------------------------------------------------

class TestBackgroundPool:

    def test_pool_excludes_near_tc_times(self, tmp_path):
        ib_p, ms_p, tc_times = _make_ibtracs(tmp_path, seasons=(2019,), n_per_season=2)
        obs_p, meta_p, insitu_times = _make_insitu(tmp_path, tc_times[0])

        ibtracs = IBTrACSDataset(ib_p, ms_p)
        insitu  = InsituLandDataset(obs_p, meta_p)

        pool = _build_background_pool(insitu, ibtracs, buffer_hours=6.0)

        # None of the pool timestamps should be within 6 h of a TC timestamp
        tc_ts   = np.sort(ibtracs['ISO_TIME'])
        buf_ns  = int(6.0 * 3600 * 1e9)
        for t in pool:
            lo = np.searchsorted(tc_ts, t - buf_ns, side='left')
            hi = np.searchsorted(tc_ts, t + buf_ns, side='right')
            assert hi - lo == 0, f"Pool timestamp {t} is within 6 h of a TC obs"

    def test_pool_is_int64(self, tmp_path):
        ib_p, ms_p, tc_times = _make_ibtracs(tmp_path)
        obs_p, meta_p, _ = _make_insitu(tmp_path, tc_times[0])
        ibtracs = IBTrACSDataset(ib_p, ms_p)
        insitu  = InsituLandDataset(obs_p, meta_p)
        pool = _build_background_pool(insitu, ibtracs)
        assert pool.dtype == np.int64

    def test_pool_only_synoptic_hours(self, tmp_path):
        # Insitu fixture is hourly; only timestamps on the exact 3-hour
        # grid (00/03/.../21 UTC, zero minutes/seconds) may survive.
        from experiments.tc_perceiver_io.data.datamodule import (
            SYNOPTIC_STEP_NS,
        )
        ib_p, ms_p, tc_times = _make_ibtracs(tmp_path, seasons=(2019,),
                                             n_per_season=2)
        obs_p, meta_p, insitu_times = _make_insitu(tmp_path, tc_times[0],
                                                   n_hours=40)
        ibtracs = IBTrACSDataset(ib_p, ms_p)
        insitu  = InsituLandDataset(obs_p, meta_p)

        pool = _build_background_pool(insitu, ibtracs, buffer_hours=6.0)
        assert len(pool) > 0
        assert np.all(pool % SYNOPTIC_STEP_NS == 0)
        # Off-grid hours existed in the input and were removed, not absent.
        assert np.any(insitu_times % SYNOPTIC_STEP_NS != 0)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

class TestCollate:

    def _make_sample(self, label, n=4, f=3, sid='2019001N15276'):
        return {
            'query_coords':   np.zeros(2, dtype=np.float32),
            'station_obs':    np.zeros((n, f), dtype=np.float32),
            'station_coords': np.zeros((n, 2), dtype=np.float32),
            'station_mask':   np.ones(n, dtype=bool),
            'obs_mask':       np.ones((n, f), dtype=bool),
            'label':          np.int32(label),
            'n_stations':     np.int32(n),
            'sid':            sid,
            'iso_time':       np.int64(1_567_296_000_000_000_000),
            'query_lat':      np.float32(15.0),
            'query_lon':      np.float32(-75.0),
            'n_available':    np.int32(n + 3),
        }

    def test_collate_y_shape(self):
        samples = [self._make_sample(i) for i in range(4)]
        batch   = _collate(samples)
        assert batch['y'].shape == (4,)

    def test_collate_x_keys(self):
        samples = [self._make_sample(0)]
        batch   = _collate(samples)
        assert set(batch['X'].keys()) == {
            'query_coords',
            'station_obs', 'station_coords',
            'station_mask', 'obs_mask',
        }

    def test_collate_station_obs_shape(self):
        n, f = 8, 5
        samples = [self._make_sample(1, n=n, f=f) for _ in range(3)]
        batch = _collate(samples)
        assert batch['X']['station_obs'].shape == (3, n, f)

    def test_collate_labels_values(self):
        samples = [self._make_sample(i) for i in [0, 5, 10]]
        batch   = _collate(samples)
        import jax.numpy as jnp
        assert list(batch['y'].tolist()) == [0, 5, 10]

    # --- metadata threading (decision 13): present in meta, absent from X ---

    def test_collate_meta_outside_x(self):
        samples = [self._make_sample(0), self._make_sample(1, sid=None)]
        batch   = _collate(samples)
        # X carries exactly the model-facing arrays — nothing else.
        assert set(batch['X'].keys()) == {
            'query_coords',
            'station_obs', 'station_coords',
            'station_mask', 'obs_mask',
        }
        assert set(batch.keys()) == {'X', 'y', 'meta'}
        for k in ('sid', 'iso_time', 'query_lat', 'query_lon',
                  'n_available', 'n_used'):
            assert k in batch['meta']

    def test_collate_meta_values(self):
        s1 = self._make_sample(5)
        s2 = self._make_sample(0, sid=None)
        s2['n_stations']   = np.int32(2)
        s2['n_available']  = np.int32(5)
        batch = _collate([s1, s2])
        meta  = batch['meta']
        assert meta['sid'] == ['2019001N15276', None]
        assert meta['iso_time'].dtype == np.int64
        assert meta['query_lat'][0] == np.float32(15.0)
        assert meta['n_used'].tolist()      == [4, 2]
        assert meta['n_available'].tolist() == [7, 5]


# ---------------------------------------------------------------------------
# TCLoader
# ---------------------------------------------------------------------------

class TestTCLoader:

    def _make_loader(self, tmp_path):
        base_ns = 1_567_296_000_000_000_000
        hour_ns = 3_600_000_000_000

        ib_p, ms_p, tc_times = _make_ibtracs(tmp_path, seasons=(2019,), n_per_season=6)
        obs_p, meta_p, insitu_times = _make_insitu(tmp_path, base_ns, n_hours=50)

        ibtracs = IBTrACSDataset(ib_p, ms_p)
        insitu  = InsituLandDataset(obs_p, meta_p)

        # Build a background pool from times far from TC obs
        bg_pool = insitu_times[40:]  # well past any TC timestamp

        ds = TCDataset(
            ibtracs=ibtracs,
            insitu=insitu,
            radius_km=300.0,
            time_window_hours=3.0,
            max_stations=4,
            min_stations=1,
            background_timestamps=bg_pool,
        )
        return TCLoader(ds, batch_size=4, tc_fraction=0.5, shuffle=False, seed=0)

    def test_loader_yields_dict(self, tmp_path):
        loader = self._make_loader(tmp_path)
        for batch in loader:
            assert isinstance(batch, dict)
            assert 'X' in batch and 'y' in batch
            break

    def test_batch_y_dtype(self, tmp_path):
        import jax.numpy as jnp
        loader = self._make_loader(tmp_path)
        for batch in loader:
            assert batch['y'].dtype == jnp.int32
            break

    def test_batch_x_has_all_keys(self, tmp_path):
        loader = self._make_loader(tmp_path)
        for batch in loader:
            assert set(batch['X'].keys()) == {
                'query_coords',
                'station_obs', 'station_coords',
                'station_mask', 'obs_mask',
            }
            break

    def test_batch_meta_sids(self, tmp_path):
        loader = self._make_loader(tmp_path)
        for batch in loader:
            sids = batch['meta']['sid']
            # tc_half=2 TC samples first (string SIDs), then background
            # (None). Background count may fall short of bg_half in
            # non-frozen sequential mode when draws fail (the TC buffer
            # is no longer discarded) — this fixture's tiny station set
            # makes that common.
            assert 2 <= len(sids) <= 4
            assert all(isinstance(s, str) for s in sids[:2])
            assert all(s is None for s in sids[2:])
            break

    def test_invalid_tc_fraction_raises(self, tmp_path):
        ib_p, ms_p, _ = _make_ibtracs(tmp_path)
        obs_p, meta_p, times = _make_insitu(tmp_path, 1_567_296_000_000_000_000)
        ibtracs = IBTrACSDataset(ib_p, ms_p)
        insitu  = InsituLandDataset(obs_p, meta_p)
        ds = TCDataset(ibtracs, insitu, background_timestamps=times[:5])
        with pytest.raises(ValueError, match='tc_fraction'):
            TCLoader(ds, batch_size=4, tc_fraction=1.0)


# ---------------------------------------------------------------------------
# TCLoader — random mode (steps_per_epoch)
# ---------------------------------------------------------------------------

class TestTCLoaderRandomMode:
    """TCLoader with steps_per_epoch set (random sampling with replacement)."""

    def _make_random_loader(self, tmp_path, steps: int) -> TCLoader:
        base_ns = 1_567_296_000_000_000_000
        ib_p, ms_p, tc_times = _make_ibtracs(tmp_path, seasons=(2019,), n_per_season=6)
        obs_p, meta_p, insitu_times = _make_insitu(tmp_path, base_ns, n_hours=50)

        ibtracs = IBTrACSDataset(ib_p, ms_p)
        insitu  = InsituLandDataset(obs_p, meta_p)
        bg_pool = insitu_times[40:]   # well past TC timestamps

        ds = TCDataset(
            ibtracs=ibtracs,
            insitu=insitu,
            radius_km=300.0,
            time_window_hours=3.0,
            max_stations=4,
            min_stations=1,
            background_timestamps=bg_pool,
        )
        return TCLoader(
            ds, batch_size=4, tc_fraction=0.5,
            seed=0, steps_per_epoch=steps,
        )

    def test_len_returns_steps_per_epoch(self, tmp_path):
        loader = self._make_random_loader(tmp_path, steps=7)
        assert len(loader) == 7

    def test_yields_exactly_steps_per_epoch_batches(self, tmp_path):
        loader = self._make_random_loader(tmp_path, steps=5)
        batches = list(loader)
        assert len(batches) == 5

    def test_each_batch_has_correct_keys(self, tmp_path):
        loader = self._make_random_loader(tmp_path, steps=3)
        for batch in loader:
            assert set(batch['X'].keys()) == {
                'query_coords', 'station_obs', 'station_coords',
                'station_mask', 'obs_mask',
            }
            break

    def test_reuse_allowed_across_steps(self, tmp_path):
        """Dataset has only 6 TC samples; steps=20 > 6 so reuse must occur."""
        # The loader should complete all 20 steps without error —
        # verifying that random-with-replacement (reuse) works correctly.
        loader = self._make_random_loader(tmp_path, steps=20)
        batches = list(loader)
        assert len(batches) == 20

    def test_loader_is_re_iterable(self, tmp_path):
        """Iterating a second time restarts from a fresh per-epoch seed."""
        loader  = self._make_random_loader(tmp_path, steps=3)
        epoch1  = list(loader)
        epoch2  = list(loader)
        # Both epochs should produce exactly steps_per_epoch batches.
        assert len(epoch1) == 3
        assert len(epoch2) == 3


# ---------------------------------------------------------------------------
# TCDataModule.summary() and stored attributes
# ---------------------------------------------------------------------------

class _MockDS:
    """Minimal TCDataset stub for summary() tests — no file I/O needed."""
    def __init__(self, n_tc: int, n_bg: int):
        self.background_timestamps = np.zeros(n_bg, dtype=np.int64)
        self._n = n_tc

    def __len__(self) -> int:
        return self._n

    def get_tc_sample(self, idx: int):
        # Deterministic fake counts for station_diagnostics: candidates
        # cycle 5..14, capped at 8 used stations.
        n_avail = 5 + (idx % 10)
        return {
            'n_available': np.int32(n_avail),
            'n_stations':  np.int32(min(n_avail, 8)),
        }


def _make_summary_dm():
    """Build a TCDataModule with injected stub datasets."""
    from experiments.tc_perceiver_io.data.datamodule import TCDataModule
    from experiments.tc_perceiver_io.data.inputs import InputSpec
    dm = TCDataModule()
    dm._input_spec        = InputSpec(location_encoding='unit_circle',
                                      normalisation='minmax_11')
    dm._max_stations      = 16
    dm._min_stations      = 1
    dm._batch_size        = 4
    dm._tc_fraction       = 0.5
    dm._train_ds          = _MockDS(n_tc=100, n_bg=500)
    dm._val_ds            = _MockDS(n_tc=20,  n_bg=100)
    dm._test_ds           = _MockDS(n_tc=10,  n_bg=50)
    dm._manifest = {
        'strategy': 'year',
        'train': {'years': [2019], 'sids': [], 'n_rows': 100, 'n_sids': 30, 'class_counts': {}},
        'val':   {'years': [2021], 'sids': [], 'n_rows': 20,  'n_sids': 8,  'class_counts': {}},
        'test':  {'years': [2023], 'sids': [], 'n_rows': 10,  'n_sids': 4,  'class_counts': {}},
    }
    return dm


class TestTCDataModuleSummary:

    def test_summary_produces_output(self, capsys):
        _make_summary_dm().summary()
        assert len(capsys.readouterr().out) > 0

    def test_summary_contains_split_names(self, capsys):
        _make_summary_dm().summary()
        out = capsys.readouterr().out
        for name in ('train', 'val', 'test'):
            assert name in out

    def test_summary_contains_location_encoding(self, capsys):
        _make_summary_dm().summary()
        assert 'unit_circle' in capsys.readouterr().out

    def test_summary_contains_obs_normalisation(self, capsys):
        _make_summary_dm().summary()
        assert 'minmax_11' in capsys.readouterr().out

    def test_summary_contains_tc_row_counts(self, capsys):
        _make_summary_dm().summary()
        out = capsys.readouterr().out
        assert '100' in out    # train TC count
        assert '20'  in out    # val TC count
        assert '10'  in out    # test TC count

    def test_summary_contains_batch_size(self, capsys):
        _make_summary_dm().summary()
        assert 'batch_size=4' in capsys.readouterr().out

    def test_summary_contains_max_stations(self, capsys):
        _make_summary_dm().summary()
        assert 'max_stations=16' in capsys.readouterr().out

    def test_summary_none_background_timestamps_does_not_raise(self, capsys):
        dm = _make_summary_dm()
        dm._train_ds.background_timestamps = None
        dm.summary()   # must not raise; shows 0 background rows
        out = capsys.readouterr().out
        assert 'train' in out

    def test_summary_steps_per_epoch_shown_for_train(self, capsys):
        _make_summary_dm().summary(steps_per_epoch=250)
        out = capsys.readouterr().out
        assert '250' in out
        assert 'random' in out.lower()

    def test_summary_sequential_mode_when_steps_none(self, capsys):
        _make_summary_dm().summary(steps_per_epoch=None)
        out = capsys.readouterr().out
        assert 'sequential' in out.lower()

    # --- station-count diagnostics (decision 9) ---

    def test_station_diagnostics_stats(self):
        dm = _make_summary_dm()   # _max_stations = 16
        d  = dm.station_diagnostics('train')
        assert d is not None
        assert d['n_samples'] == 100
        # _MockDS cycles n_available 5..14, n_used = min(n_available, 8)
        assert d['n_available']['min'] == 5
        assert d['n_available']['max'] == 14
        assert d['n_used']['max'] == 8
        assert d['n_available']['avg'] >= d['n_used']['avg']
        # max_stations=16 > all candidate counts: nothing is capped
        assert d['frac_capped'] == 0.0

    def test_station_diagnostics_frac_capped(self):
        dm = _make_summary_dm()
        dm._max_stations = 8
        d  = dm.station_diagnostics('train')
        # n_available cycles 5..14 → 7 of 10 values are >= 8
        assert d['frac_capped'] == pytest.approx(0.7)

    def test_summary_prints_diagnostics(self, capsys):
        _make_summary_dm().summary()
        out = capsys.readouterr().out
        assert 'n_avail' in out
        assert 'capped' in out

    def test_summary_diagnostics_can_be_disabled(self, capsys):
        _make_summary_dm().summary(diagnostics=False)
        out = capsys.readouterr().out
        assert 'n_avail' not in out

    def test_stored_attributes_after_setup(self, tmp_path):
        """Attributes added to setup() are present on the datamodule."""
        from experiments.tc_perceiver_io.data.datamodule import TCDataModule
        # Use the synthetic builders to create real npz files
        ib_p, ms_p, tc_times = _make_ibtracs(tmp_path, seasons=(2019, 2021, 2023))
        obs_p, meta_p, _     = _make_insitu(tmp_path, tc_times[0], n_hours=60)
        cfg = {
            'ibtracs_path':    str(ib_p),
            'multi_storm_path': str(ms_p),
            'insitu_obs_path':  str(obs_p),
            'insitu_meta_path': str(meta_p),
            'location_encoding': 'domain',
            'obs_normalisation': 'minmax_01',
            'max_stations': 8,
            'min_stations': 1,
            'batch_size':   4,
            'tc_fraction':  0.5,
            'split': {
                'strategy': 'year',
                'train': {'years': [2019]},
                'val':   {'years': [2021]},
                'test':  {'years': [2023], 'hard_test': 'multi_storm'},
            },
        }
        dm = TCDataModule.from_config(cfg)
        # InputSpec is the single source of truth for the input configuration.
        assert dm.input_spec.location_encoding == 'domain'
        assert dm.input_spec.normalisation     == 'minmax_01'
        assert dm._max_stations                == 8
        assert dm._min_stations                == 1


# ---------------------------------------------------------------------------
# Eval determinism (phase 3 subset — decisions 5 + 9)
# ---------------------------------------------------------------------------

class TestEvalDeterminism:
    """Frozen backgrounds + nearest station selection + partial-batch flush."""

    # FOV tight around the single fixture station so every LHS background
    # position finds it within radius_km.
    _FOV_LAT = (14.5, 15.5)
    _FOV_LON = (-75.5, -74.5)

    def _make_dataset(self, tmp_path, n_per_season=5):
        ib_p, ms_p, tc_times = _make_ibtracs(
            tmp_path, seasons=(2019,), n_per_season=n_per_season)
        obs_p, meta_p, insitu_times = _make_insitu(
            tmp_path, tc_times[0], n_hours=50)
        ibtracs = IBTrACSDataset(ib_p, ms_p)
        insitu  = InsituLandDataset(obs_p, meta_p)
        return TCDataset(
            ibtracs=ibtracs,
            insitu=insitu,
            radius_km=300.0,
            time_window_hours=3.0,
            max_stations=4,
            min_stations=1,
            background_timestamps=insitu_times[40:],
        )

    def _make_eval_loader(self, tmp_path, n_per_season=5):
        ds = self._make_dataset(tmp_path, n_per_season)
        return TCLoader(
            ds, batch_size=4, tc_fraction=0.5, shuffle=False, seed=0,
            fov_lat=self._FOV_LAT, fov_lon=self._FOV_LON,
            freeze_backgrounds=True,
        )

    @staticmethod
    def _assert_batches_identical(b1, b2):
        for k in b1['X']:
            assert np.array_equal(np.asarray(b1['X'][k]),
                                  np.asarray(b2['X'][k])), k
        assert np.array_equal(np.asarray(b1['y']), np.asarray(b2['y']))
        assert b1['meta']['sid'] == b2['meta']['sid']
        for k in ('iso_time', 'query_lat', 'query_lon',
                  'n_available', 'n_used'):
            assert np.array_equal(b1['meta'][k], b2['meta'][k]), k

    def test_two_iterations_yield_identical_batches(self, tmp_path):
        loader = self._make_eval_loader(tmp_path)
        epoch1 = list(loader)
        epoch2 = list(loader)
        assert len(epoch1) == len(epoch2) > 0
        for b1, b2 in zip(epoch1, epoch2):
            self._assert_batches_identical(b1, b2)

    def test_flush_yields_all_tc_rows(self, tmp_path):
        # 5 valid TC rows, tc_half=2 → 2 full batches + flushed partial of 1
        loader  = self._make_eval_loader(tmp_path, n_per_season=5)
        batches = list(loader)
        n_tc = sum(
            sum(1 for s in b['meta']['sid'] if s is not None)
            for b in batches
        )
        assert n_tc == 5
        assert len(batches) == 3
        # Flushed batch: 1 TC + proportional background (1) = 2 samples
        assert batches[-1]['y'].shape[0] == 2
        assert sum(1 for s in batches[-1]['meta']['sid'] if s is not None) == 1

    def test_no_flush_when_batches_divide_evenly(self, tmp_path):
        loader  = self._make_eval_loader(tmp_path, n_per_season=4)
        batches = list(loader)
        assert len(batches) == 2
        assert all(b['y'].shape[0] == 4 for b in batches)

    def test_frozen_background_positions_within_fov(self, tmp_path):
        loader = self._make_eval_loader(tmp_path)
        for batch in loader:
            sids = batch['meta']['sid']
            bg = [i for i, s in enumerate(sids) if s is None]
            lats = batch['meta']['query_lat'][bg]
            lons = batch['meta']['query_lon'][bg]
            assert np.all(lats >= self._FOV_LAT[0]) and np.all(lats <= self._FOV_LAT[1])
            assert np.all(lons >= self._FOV_LON[0]) and np.all(lons <= self._FOV_LON[1])

    def test_frozen_background_timestamps_from_pool(self, tmp_path):
        loader = self._make_eval_loader(tmp_path)
        pool   = set(int(t) for t in loader._dataset.background_timestamps)
        for batch in loader:
            sids = batch['meta']['sid']
            for i, s in enumerate(sids):
                if s is None:
                    assert int(batch['meta']['iso_time'][i]) in pool

    def test_train_loader_varies_by_epoch(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        loader = TCLoader(
            ds, batch_size=4, tc_fraction=0.5, seed=0,
            fov_lat=self._FOV_LAT, fov_lon=self._FOV_LON,
            steps_per_epoch=3,
            freeze_backgrounds=False,
        )
        epoch1 = list(loader)
        epoch2 = list(loader)
        # Background query positions are fresh draws — epochs must differ
        lats1 = np.concatenate([b['meta']['query_lat'] for b in epoch1])
        lats2 = np.concatenate([b['meta']['query_lat'] for b in epoch2])
        assert not np.array_equal(lats1, lats2)

    def test_len_includes_flush_batch(self, tmp_path):
        loader = self._make_eval_loader(tmp_path, n_per_season=5)
        assert len(loader) == 3
