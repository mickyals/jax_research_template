"""
Tests for datasets/volume.py — generic volume_v1 store.

Guardrails covered here: write/load round-trip through mmap, entity-spine
slice == brute-force filter, time_slice bounds on a known fixture.
"""

import json

import numpy as np
import pytest

from datasets.volume import (
    VOLUME_FORMAT,
    build_entity_spine,
    build_volume_time_index,
    get_entity,
    load_volume,
    rows_at,
    time_slice,
    write_volume,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ROWS = 120
STATIONS = ['ALPHA', 'BRAVO', 'CHARLIE', 'DELTA', 'ECHO']


def _make_obs(rng):
    """Synthetic time-sorted column dict + per-row station ids."""
    base = np.datetime64('2020-06-01T00:00', 'ns')
    offsets = np.sort(rng.integers(0, 72 * 3600, N_ROWS))          # 3 days
    t = base + offsets.astype('timedelta64[s]').astype('timedelta64[ns]')
    obs = {
        'report_timestamp': t,
        'lat':   rng.uniform(0, 30, N_ROWS).astype(np.float32),
        'lon':   rng.uniform(-100, -30, N_ROWS).astype(np.float32),
        'level': rng.uniform(90000, 103000, N_ROWS).astype(np.float32),
        'air_temp': rng.normal(300, 5, N_ROWS).astype(np.float32),
    }
    sid = rng.choice(STATIONS, N_ROWS)
    return obs, sid


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def volume_dir(tmp_path, rng):
    """A synthetic volume written to disk; returns (dir, obs, sid)."""
    obs, sid = _make_obs(rng)
    eids, eint, eorder, eoff = build_entity_spine(sid)
    d = write_volume(tmp_path / 'vol', obs, eint, eids, eorder, eoff)
    return d, obs, sid


# ---------------------------------------------------------------------------
# Entity spine
# ---------------------------------------------------------------------------

class TestEntitySpine:

    def test_ids_sorted_unique(self, rng):
        _, sid = _make_obs(rng)
        eids, eint, eorder, eoff = build_entity_spine(sid)
        assert list(eids) == sorted(set(sid))
        assert eint.dtype == np.int32
        assert eoff[0] == 0 and eoff[-1] == len(sid)

    def test_blocks_match_brute_force(self, rng):
        obs, sid = _make_obs(rng)
        eids, eint, eorder, eoff = build_entity_spine(sid)
        for e, name in enumerate(eids):
            rows = eorder[eoff[e]:eoff[e + 1]]
            brute = np.nonzero(sid == name)[0]
            np.testing.assert_array_equal(np.sort(rows), brute)

    def test_time_order_within_block(self, rng):
        obs, sid = _make_obs(rng)
        eids, eint, eorder, eoff = build_entity_spine(sid)
        t = obs['report_timestamp']
        for e in range(len(eids)):
            rows = eorder[eoff[e]:eoff[e + 1]]
            block_t = t[rows]
            assert np.all(block_t[:-1] <= block_t[1:])


# ---------------------------------------------------------------------------
# Write / load round-trip
# ---------------------------------------------------------------------------

class TestWriteLoad:

    def test_round_trip_values(self, volume_dir):
        d, obs, sid = volume_dir
        vol = load_volume(d)
        for k, v in obs.items():
            np.testing.assert_array_equal(np.asarray(vol['obs'][k]), v)

    def test_arrays_are_memory_mapped(self, volume_dir):
        d, obs, _ = volume_dir
        vol = load_volume(d)
        assert isinstance(vol['obs']['air_temp'], np.memmap)
        assert isinstance(vol['entity_order'], np.memmap)

    def test_manifest_contents(self, volume_dir):
        d, obs, _ = volume_dir
        man = json.loads((d / 'manifest.json').read_text())
        assert man['format'] == VOLUME_FORMAT
        assert man['n_rows'] == N_ROWS
        assert set(man['columns']) == set(obs.keys())

    def test_wrong_format_raises(self, tmp_path, volume_dir):
        d, _, _ = volume_dir
        man = json.loads((d / 'manifest.json').read_text())
        man['format'] = 'something_else'
        (d / 'manifest.json').write_text(json.dumps(man))
        with pytest.raises(ValueError, match='volume_v1'):
            load_volume(d)

    def test_category_spine_both_or_neither(self, tmp_path, rng):
        obs, sid = _make_obs(rng)
        eids, eint, eorder, eoff = build_entity_spine(sid)
        with pytest.raises(ValueError, match='together'):
            write_volume(tmp_path / 'v', obs, eint, eids, eorder, eoff,
                         cat_order=np.arange(3))

    def test_optional_category_spine_round_trip(self, tmp_path, rng):
        obs, sid = _make_obs(rng)
        eids, eint, eorder, eoff = build_entity_spine(sid)
        cat_order = np.arange(N_ROWS, dtype=np.int64)
        cat_offsets = np.array([0, N_ROWS], dtype=np.int64)
        d = write_volume(tmp_path / 'v', obs, eint, eids, eorder, eoff,
                         cat_order=cat_order, cat_offsets=cat_offsets)
        vol = load_volume(d)
        np.testing.assert_array_equal(np.asarray(vol['cat_order']), cat_order)
        np.testing.assert_array_equal(vol['cat_offsets'], cat_offsets)


# ---------------------------------------------------------------------------
# Time spine
# ---------------------------------------------------------------------------

class TestTimeSpine:

    def test_time_slice_bounds(self, volume_dir):
        d, obs, _ = volume_dir
        vol = load_volume(d)
        t = obs['report_timestamp']
        t_lo, t_hi = t[10], t[50]
        lo, hi = time_slice(vol['obs'], t_lo, t_hi)
        window = t[lo:hi]
        assert np.all(window >= t_lo) and np.all(window <= t_hi)
        # closed range: rows just outside are excluded
        if lo > 0:
            assert t[lo - 1] < t_lo
        if hi < len(t):
            assert t[hi] > t_hi

    def test_time_slice_empty_range(self, volume_dir):
        d, obs, _ = volume_dir
        vol = load_volume(d)
        lo, hi = time_slice(vol['obs'],
                            np.datetime64('2030-01-01'),
                            np.datetime64('2030-01-02'))
        assert lo == hi

    def test_rows_at_hit_and_miss(self, volume_dir):
        d, obs, _ = volume_dir
        vol = load_volume(d)
        uniq_t, offsets = build_volume_time_index(vol['obs'])

        hit = uniq_t[3]
        got = rows_at(vol['obs'], uniq_t, offsets, hit)
        expect = np.sum(obs['report_timestamp'] == hit)
        assert len(got['air_temp']) == expect > 0

        miss = np.datetime64('1999-01-01')
        got = rows_at(vol['obs'], uniq_t, offsets, miss)
        assert len(got['air_temp']) == 0

    def test_offsets_partition_all_rows(self, volume_dir):
        d, obs, _ = volume_dir
        vol = load_volume(d)
        uniq_t, offsets = build_volume_time_index(vol['obs'])
        assert offsets[0] == 0 and offsets[-1] == N_ROWS
        assert np.all(np.diff(offsets) >= 1)
        assert len(uniq_t) == len(np.unique(obs['report_timestamp']))


# ---------------------------------------------------------------------------
# Entity access on a loaded volume
# ---------------------------------------------------------------------------

class TestGetEntity:

    def test_by_string_matches_brute_force(self, volume_dir):
        d, obs, sid = volume_dir
        vol = load_volume(d)
        got = get_entity(vol, 'CHARLIE')
        brute_rows = np.nonzero(sid == 'CHARLIE')[0]
        np.testing.assert_array_equal(
            np.sort(got['air_temp']),
            np.sort(obs['air_temp'][brute_rows]))
        # time order preserved
        t = got['report_timestamp']
        assert np.all(t[:-1] <= t[1:])

    def test_by_int_equals_by_string(self, volume_dir):
        d, obs, sid = volume_dir
        vol = load_volume(d)
        e = int(np.searchsorted(vol['entity_ids'], 'BRAVO'))
        by_int = get_entity(vol, e)
        by_str = get_entity(vol, 'BRAVO')
        np.testing.assert_array_equal(by_int['air_temp'], by_str['air_temp'])
