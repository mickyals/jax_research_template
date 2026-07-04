"""
Tests for datasets/shelf.py — the _BOOKSHELF cross-volume index layer.

Guardrails covered here (the causality invariants):
  * no lookback window ever reaches past the driver time T
  * edges are monotonic non-decreasing
  * empty windows (lo == hi) are handled and surfaced
  * freshness fingerprint mismatch is detected
"""

import numpy as np
import pytest

from experiments.cyclone_jax.data.sources.shelf import (
    build_lookback_pointers,
    build_time_index,
    check_shelf_fresh,
    load_lookback,
    load_shelf,
    load_all_shelves,
    report_occupancy,
    rows_at_shelf,
    time_to_idx,
    window_obs,
    window_temporal_encoding,
    write_lookback,
    write_shelf,
)
from experiments.cyclone_jax.data.sources.volume import build_entity_spine, write_volume, load_volume


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE = np.datetime64('2020-06-01T00:00', 'ns')


def _obs_at(offsets_s, **extra_cols):
    """Column dict with report_timestamp at BASE + given second offsets."""
    off = np.asarray(sorted(offsets_s))
    t = BASE + off.astype('timedelta64[s]').astype('timedelta64[ns]')
    n = len(t)
    obs = {'report_timestamp': t,
           'value': np.arange(n, dtype=np.float32)}
    obs.update(extra_cols)
    return obs


@pytest.fixture
def rng():
    return np.random.default_rng(1)


@pytest.fixture
def obs(rng):
    """200 rows over 24 h, irregular but deterministic."""
    return _obs_at(np.sort(rng.integers(0, 24 * 3600, 200)))


@pytest.fixture
def storm_times():
    """Driver times at +6h, +12h, +18h (all inside the obs span)."""
    hours = np.array([6, 12, 18])
    return BASE + (hours * 3600).astype('timedelta64[s]').astype('timedelta64[ns]')


DELTAS = [np.timedelta64(2, 'h'), np.timedelta64(1, 'h'),
          np.timedelta64(30, 'm'), np.timedelta64(10, 'm')]


def _brute_edges(storm_times, t, deltas):
    """O(S*N) reference implementation of the lookback edges."""
    S, W = len(storm_times), len(deltas)
    edges = np.zeros((S, W + 1), np.int64)
    for i, T in enumerate(storm_times):
        for w, d in enumerate(deltas):
            edges[i, w] = int(np.sum(t < T - d))
        edges[i, W] = int(np.sum(t <= T))
    return edges


# ---------------------------------------------------------------------------
# Shelf write / load / freshness
# ---------------------------------------------------------------------------

class TestShelfRoundTrip:

    def test_write_then_load(self, tmp_path, obs):
        write_shelf(tmp_path, 'land', obs)
        shelf = load_shelf(tmp_path, 'land')
        assert 'uniq_t' in shelf and 'time_offsets' in shelf
        assert shelf['meta']['n_rows'] == len(obs['value'])

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_shelf(tmp_path, 'nope') == {}

    def test_load_all(self, tmp_path, obs):
        write_shelf(tmp_path, 'land', obs)
        write_shelf(tmp_path, 'marine', obs)
        shelves = load_all_shelves(tmp_path, ('land', 'marine', 'upper'))
        assert 'uniq_t' in shelves['land'] and 'uniq_t' in shelves['marine']
        assert shelves['upper'] == {}

    def test_rows_at_shelf(self, tmp_path, obs):
        write_shelf(tmp_path, 'land', obs)
        shelf = load_shelf(tmp_path, 'land')
        t0 = obs['report_timestamp'][0]
        got = rows_at_shelf(obs, shelf, t0)
        assert len(got['value']) == int(np.sum(obs['report_timestamp'] == t0))


class TestFreshness:

    def _vol(self, tmp_path, obs, name='v'):
        sid = np.array(['S0'] * len(obs['value']))
        eids, eint, eorder, eoff = build_entity_spine(sid)
        d = write_volume(tmp_path / name, obs, eint, eids, eorder, eoff)
        return load_volume(d)

    def test_fresh_shelf_passes(self, tmp_path, obs):
        vol = self._vol(tmp_path, obs)
        write_shelf(tmp_path, 'land', obs)
        fresh, why = check_shelf_fresh(vol, load_shelf(tmp_path, 'land'))
        assert fresh, why

    def test_stale_shelf_detected(self, tmp_path, obs):
        write_shelf(tmp_path, 'land', obs)          # shelf for the FULL obs
        truncated = {k: v[:-5] for k, v in obs.items()}
        vol = self._vol(tmp_path, truncated, name='v2')
        fresh, why = check_shelf_fresh(vol, load_shelf(tmp_path, 'land'))
        assert not fresh
        assert 'n_rows' in why or 't_last' in why

    def test_no_fingerprint_skips(self, tmp_path, obs):
        vol = self._vol(tmp_path, obs)
        fresh, why = check_shelf_fresh(vol, {})     # never-written shelf
        assert fresh and 'no fingerprint' in why


# ---------------------------------------------------------------------------
# Driver manifest
# ---------------------------------------------------------------------------

class TestDriverManifest:

    def test_single_multi_split(self, tmp_path):
        # 3 driver rows at T1 (multi), 1 at T2 (single), 1 below threshold
        offs = [100, 100, 100, 200, 300]
        obs = _obs_at(offs, usa_sshs=np.array([3, 4, 5, 6, 1], np.float32))
        write_shelf(tmp_path, 'cyclone', obs,
                    driver_col='usa_sshs', driver_min=3)
        shelf = load_shelf(tmp_path, 'cyclone')
        assert shelf['meta']['n_storm_times'] == 2
        assert len(shelf['single_times']) == 1
        assert len(shelf['multi_times']) == 1
        np.testing.assert_array_equal(np.asarray(shelf['n_storms']), [3, 1])

    def test_exclude_col(self, tmp_path):
        offs = [100, 200]
        obs = _obs_at(offs,
                      usa_sshs=np.array([3, 3], np.float32),
                      is_subtropical=np.array([False, True]))
        write_shelf(tmp_path, 'cyclone', obs,
                    driver_col='usa_sshs', driver_min=3,
                    exclude_col='is_subtropical')
        shelf = load_shelf(tmp_path, 'cyclone')
        assert shelf['meta']['n_storm_times'] == 1

    def test_no_driver_when_unconfigured(self, tmp_path, obs):
        write_shelf(tmp_path, 'land', obs)
        shelf = load_shelf(tmp_path, 'land')
        assert 'storm_times' not in shelf


# ---------------------------------------------------------------------------
# Union time index
# ---------------------------------------------------------------------------

class TestTimeIndex:

    def test_union_sorted_dedup(self):
        a = _obs_at([0, 100, 200])
        b = _obs_at([100, 300])
        times = build_time_index(a, b)
        assert len(times) == 4
        assert np.all(times[:-1] < times[1:])

    def test_time_to_idx(self):
        a = _obs_at([0, 100, 200])
        times = build_time_index(a)
        assert time_to_idx(times, a['report_timestamp'][1]) == 1
        assert time_to_idx(times, np.datetime64('1999-01-01')) == -1
        assert time_to_idx(np.array([], dtype='datetime64[ns]'), BASE) == -1


# ---------------------------------------------------------------------------
# Lookback pointers — the causality guardrails
# ---------------------------------------------------------------------------

class TestLookback:

    def test_matches_brute_force(self, obs, storm_times):
        edges = build_lookback_pointers(storm_times, obs, DELTAS)
        brute = _brute_edges(storm_times, obs['report_timestamp'], DELTAS)
        np.testing.assert_array_equal(edges, brute)

    def test_no_window_reaches_past_T(self, obs, storm_times):
        """GUARDRAIL: nothing after the driver time is ever indexed."""
        edges = build_lookback_pointers(storm_times, obs, DELTAS)
        t = obs['report_timestamp']
        for i, T in enumerate(storm_times):
            hi = edges[i, -1]
            if hi > 0:
                assert t[hi - 1] <= T
            if hi < len(t):
                assert t[hi] > T

    def test_edges_monotonic_non_decreasing(self, obs, storm_times):
        """GUARDRAIL: window boundaries never go backwards."""
        edges = build_lookback_pointers(storm_times, obs, DELTAS)
        assert np.all(np.diff(edges, axis=1) >= 0)

    def test_windows_partition_the_reach(self, obs, storm_times):
        edges = build_lookback_pointers(storm_times, obs, DELTAS)
        t = obs['report_timestamp']
        for i, T in enumerate(storm_times):
            rows = t[edges[i, 0]:edges[i, -1]]
            assert np.all(rows >= T - DELTAS[0])
            assert np.all(rows <= T)

    def test_empty_window_lo_equals_hi(self):
        """GUARDRAIL: a gap in the data yields lo == hi, and window_obs
        returns empty column slices (informative absence, not an error)."""
        obs = _obs_at([0, 10 * 3600])                 # 10-hour hole
        T = BASE + np.timedelta64(6, 'h')
        edges = build_lookback_pointers(np.array([T]), obs, DELTAS)
        counts = np.diff(edges, axis=1)
        assert counts.sum() == 0                      # every window empty
        got = window_obs(obs, edges, 0, 0)
        assert len(got['value']) == 0

    def test_non_descending_deltas_raise(self, obs, storm_times):
        with pytest.raises(ValueError, match='descending'):
            build_lookback_pointers(
                storm_times, obs,
                [np.timedelta64(1, 'h'), np.timedelta64(2, 'h')])

    def test_write_load_round_trip(self, tmp_path, obs, storm_times):
        write_shelf(tmp_path, 'land', obs)
        edges = write_lookback(tmp_path, 'land', storm_times, obs, DELTAS)
        shelf = load_shelf(tmp_path, 'land')
        loaded, deltas = load_lookback(shelf)
        np.testing.assert_array_equal(np.asarray(loaded), edges)
        assert deltas == DELTAS

    def test_temporal_encoding_in_range(self, obs, storm_times):
        edges = build_lookback_pointers(storm_times, obs, DELTAS)
        for i in range(len(storm_times)):
            for w in range(len(DELTAS)):
                enc = window_temporal_encoding(
                    obs, edges, storm_times, i, w, DELTAS[0])
                assert enc.dtype == np.float32
                if len(enc):
                    assert np.all(enc <= 0.0) and np.all(enc >= -1.0)

    def test_temporal_encoding_empty_window(self):
        obs = _obs_at([0])
        T = BASE + np.timedelta64(12, 'h')
        edges = build_lookback_pointers(np.array([T]), obs, DELTAS)
        enc = window_temporal_encoding(obs, edges, np.array([T]), 0, 0, DELTAS[0])
        assert len(enc) == 0


# ---------------------------------------------------------------------------
# Occupancy report (smoke)
# ---------------------------------------------------------------------------

class TestOccupancy:

    def test_counts_shape_and_sum(self, obs, storm_times, capsys):
        counts = report_occupancy('land', storm_times, obs, DELTAS)
        assert counts.shape == (len(storm_times), len(DELTAS))
        edges = build_lookback_pointers(storm_times, obs, DELTAS)
        np.testing.assert_array_equal(counts.sum(axis=1),
                                      edges[:, -1] - edges[:, 0])
        out = capsys.readouterr().out
        assert 'per-fix total' in out
