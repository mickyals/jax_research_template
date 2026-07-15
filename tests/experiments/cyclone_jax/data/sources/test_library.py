"""
Tests for experiments/cyclone_jax/data/sources/interface.py — train-time
library access: load_library staleness/delta guards, driver fixes, category
spine access, plus a gated smoke test against the real E:/Caribbean-Obs
library (set RUN_REAL_DATA_TESTS=1 to enable).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from experiments.cyclone_jax.data.sources.volume import build_entity_spine, write_volume
from experiments.cyclone_jax.data.sources.shelf import write_lookback
from experiments.cyclone_jax.data.sources.library import (
    CYC_TARGETS,
    LOOKBACK_DELTAS,
    OBS_VOLUMES,
    TROPICAL_STORM,
    VOLUMES,
    build_bookshelf,
    category_counts,
    get_category,
    get_category_atleast,
    get_fixes,
    load_library,
    load_lookback,
    window_obs,
)
from experiments.cyclone_jax.data.sources.build import build_category_index


BASE = np.datetime64('2020-08-01T00:00', 'ns')
REAL_ROOT = Path('E:/Caribbean-Obs')

_skip_real = pytest.mark.skipif(
    not REAL_ROOT.exists() or not os.environ.get('RUN_REAL_DATA_TESTS'),
    reason="Real data tests disabled. Set RUN_REAL_DATA_TESTS=1 to enable.",
)


# ---------------------------------------------------------------------------
# Synthetic mini-library
# ---------------------------------------------------------------------------

def _ts(seconds):
    off = np.asarray(seconds)
    return BASE + off.astype('timedelta64[s]').astype('timedelta64[ns]')


def _mini_library(tmp_path, rng):
    """Four tiny volumes + nothing else; bookshelf built separately."""
    root = tmp_path / 'lib'

    # --- three obs volumes over ~5 days -------------------------------
    for name in OBS_VOLUMES:
        n = 400
        obs = {
            'report_timestamp': _ts(np.sort(rng.integers(0, 5 * 24 * 3600, n))),
            'lat':      rng.uniform(0, 30, n).astype(np.float32),
            'lon':      rng.uniform(-100, -30, n).astype(np.float32),
            'level':    rng.uniform(90000, 103000, n).astype(np.float32),
            'air_temp': rng.normal(300, 5, n).astype(np.float32),
        }
        sid = rng.choice([f'{name[:2].upper()}{i}' for i in range(8)], n)
        eids, eint, eorder, eoff = build_entity_spine(sid)
        write_volume(root / VOLUMES[name], obs, eint, eids, eorder, eoff)

    # --- cyclone driver volume: 2 storms, 3-hourly fixes --------------
    hours_a = np.arange(24, 60, 3)          # storm ALPHA
    hours_b = np.arange(48, 78, 3)          # storm BETA (overlaps -> multi)
    t = _ts(np.concatenate([hours_a, hours_b]) * 3600)
    sid = np.array(['AL012020'] * len(hours_a) + ['AL022020'] * len(hours_b))
    n = len(t)
    sshs = rng.integers(3, 9, n).astype(np.float32)
    sshs[0] = 1.0                            # one below-threshold fix
    sshs[1] = np.nan                         # one unclassifiable fix
    cyc = {
        'report_timestamp': t,
        'lat':  rng.uniform(10, 25, n).astype(np.float32),
        'lon':  rng.uniform(-80, -50, n).astype(np.float32),
        'level': np.full(n, np.nan, np.float32),
        'sid':  sid,
        'usa_sshs': sshs,
        'usa_wind': rng.uniform(30, 140, n).astype(np.float32),
        'usa_pres': rng.uniform(900, 1010, n).astype(np.float32),
        'is_subtropical': np.zeros(n, bool),
    }
    order = np.argsort(cyc['report_timestamp'], kind='stable')
    cyc = {k: np.asarray(v)[order] for k, v in cyc.items()}
    eids, eint, eorder, eoff = build_entity_spine(cyc['sid'])
    cat_order, cat_offsets = build_category_index(cyc)
    write_volume(root / VOLUMES['cyclone'], cyc, eint, eids, eorder, eoff,
                 cat_order=cat_order, cat_offsets=cat_offsets)
    return root


@pytest.fixture
def rng():
    return np.random.default_rng(2)


@pytest.fixture
def library(tmp_path, rng):
    root = _mini_library(tmp_path, rng)
    build_bookshelf(root, verbose=False)
    return root


# ---------------------------------------------------------------------------
# load_library + guards
# ---------------------------------------------------------------------------

class TestLoadLibrary:

    def test_loads_all_volumes_and_shelves(self, library):
        lib = load_library(library)
        assert set(lib['volumes']) == set(VOLUMES)
        assert 'storm_times' in lib['shelves']['cyclone']
        for n in OBS_VOLUMES:
            edges, deltas = load_lookback(lib['shelves'][n])
            assert edges is not None
            assert deltas == LOOKBACK_DELTAS[n]

    def test_multi_storm_times_detected(self, library):
        lib = load_library(library)
        m = lib['shelves']['cyclone']['meta']
        assert m['n_multi'] > 0                      # overlap window exists
        assert m['n_single'] + m['n_multi'] == m['n_storm_times']

    # NOTE (Windows): these tamper tests must not hold mmaps on the files
    # they overwrite — np.save on a memory-mapped file fails with EINVAL.
    # Tamper state is built WITHOUT load_library; the guard is checked after.

    def test_stale_volume_raises(self, library, rng):
        # rebuild the land volume smaller, from scratch -> fingerprint mismatch
        n = 50
        obs = {
            'report_timestamp': _ts(np.sort(rng.integers(0, 24 * 3600, n))),
            'lat':      rng.uniform(0, 30, n).astype(np.float32),
            'lon':      rng.uniform(-100, -30, n).astype(np.float32),
            'level':    rng.uniform(90000, 103000, n).astype(np.float32),
            'air_temp': rng.normal(300, 5, n).astype(np.float32),
        }
        sid = np.array(['XX0'] * n)
        eids, eint, eorder, eoff = build_entity_spine(sid)
        write_volume(library / VOLUMES['land'], obs, eint, eids, eorder, eoff)
        with pytest.raises(RuntimeError, match='stale shelf'):
            load_library(library)

    def _bake_wrong_land_deltas(self, library):
        """Overwrite land's lookback with a wrong schedule (no mmaps held
        on the shelf files being rewritten)."""
        from experiments.cyclone_jax.data.sources.volume import load_volume
        storm_times = np.load(
            library / '_BOOKSHELF' / 'cyclone_storm_times.npy')  # RAM copy
        land = load_volume(library / VOLUMES['land'])            # columns only
        write_lookback(library, 'land', storm_times, land['obs'],
                       [np.timedelta64(1, 'h')])

    def test_delta_mismatch_raises(self, library):
        self._bake_wrong_land_deltas(library)
        with pytest.raises(RuntimeError, match='lookback deltas'):
            load_library(library)

    def test_checks_can_be_disabled(self, library):
        self._bake_wrong_land_deltas(library)
        lib = load_library(library, check_deltas=False)   # no raise
        assert 'land' in lib['volumes']


# ---------------------------------------------------------------------------
# Driver fixes + category spine
# ---------------------------------------------------------------------------

class TestFixesAndCategories:

    def test_get_fixes_threshold(self, library):
        lib = load_library(library)
        cyc = lib['volumes']['cyclone']
        fixes = get_fixes(cyc, sshs_min=TROPICAL_STORM)
        assert np.all(np.asarray(fixes['usa_sshs']) >= TROPICAL_STORM)
        # below-threshold + NaN rows are excluded
        n_valid = int(np.sum(np.asarray(cyc['obs']['usa_sshs'])
                             >= TROPICAL_STORM))
        assert len(fixes['time']) == n_valid
        # target columns ride along (metadata, never model input)
        assert 'usa_wind' in fixes and 'usa_pres' in fixes
        assert set(fixes).issuperset({'time', 'lat', 'lon', 'sid'})

    def test_category_access(self, library):
        lib = load_library(library)
        cyc = lib['volumes']['cyclone']
        counts = category_counts(cyc)
        got = get_category(cyc, 3)
        assert len(got['sid']) == counts[3]
        at_least = get_category_atleast(cyc, 4)
        assert len(at_least['sid']) == sum(counts[c] for c in range(4, 9))

    def test_window_obs_from_baked_edges(self, library):
        lib = load_library(library)
        edges, deltas = load_lookback(lib['shelves']['land'])
        obs = lib['volumes']['land']['obs']
        storm_times = np.asarray(lib['shelves']['cyclone']['storm_times'])
        edges = np.asarray(edges)
        got = window_obs(obs, edges, 5, len(deltas) - 1)   # nearest window
        T = storm_times[5]
        if len(got['report_timestamp']):
            assert np.all(np.asarray(got['report_timestamp']) <= T)

    def test_leakage_allowlist_is_disjoint(self):
        """CYC_TARGETS (target/metadata) must never appear among the obs
        channel schemas that feed the model."""
        from experiments.cyclone_jax.data.sources.build import (
            LAND_VARS, MARINE_VARS, UPPER_VARS,
        )
        obs_channels = (set(LAND_VARS.values()) | set(MARINE_VARS.values())
                        | set(UPPER_VARS.values()))
        assert obs_channels.isdisjoint(set(CYC_TARGETS))


# ---------------------------------------------------------------------------
# Real library smoke (gated)
# ---------------------------------------------------------------------------

@_skip_real
class TestRealLibrary:

    def test_load_and_pull_one_fix(self):
        lib = load_library(REAL_ROOT)                  # freshness+delta guards
        counts = category_counts(lib['volumes']['cyclone'])
        assert sum(counts.values()) == 10_258          # known build

        storm_times = np.asarray(lib['shelves']['cyclone']['storm_times'])
        assert len(storm_times) == 6_618

        total = 0
        for name in ('land', 'marine'):
            edges, deltas = load_lookback(lib['shelves'][name])
            edges = np.asarray(edges)
            assert edges.shape == (6_618, len(LOOKBACK_DELTAS[name]) + 1)
            i = 1000
            total += int(edges[i, -1] - edges[i, 0])
            got = window_obs(lib['volumes'][name]['obs'], edges, i,
                             len(deltas) - 1)
            T = storm_times[i]
            ts = np.asarray(got['report_timestamp'])
            if len(ts):
                assert np.all(ts <= T)                 # causality on real data
        assert total <= 1536                           # v1 pad bound holds
