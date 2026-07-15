"""
Shared mini-library fixture for cyclone_jax data tests: four volumes with
the REAL surface columns + a built bookshelf, in a session tmp dir.
"""

import numpy as np
import pytest

from experiments.cyclone_jax.data.sources.volume import (
    build_entity_spine, write_volume,
)
from experiments.cyclone_jax.data.sources.build import build_category_index
from experiments.cyclone_jax.data.sources.library import (
    VOLUMES, build_bookshelf, load_library,
)

BASE = np.datetime64('2020-08-01T00:00', 'ns')


def _ts(seconds):
    off = np.asarray(seconds)
    return BASE + off.astype('timedelta64[s]').astype('timedelta64[ns]')


@pytest.fixture(scope='session')
def library_root(tmp_path_factory):
    rng = np.random.default_rng(3)
    root = tmp_path_factory.mktemp('lib_v1')
    n = 500
    for name in ('land', 'marine', 'upper'):
        obs = {
            'report_timestamp': _ts(np.sort(rng.integers(0, 5 * 24 * 3600, n))),
            'lat':        rng.uniform(0, 30, n).astype(np.float32),
            'lon':        rng.uniform(-100, -30, n).astype(np.float32),
            'level':      rng.uniform(90000, 103000, n).astype(np.float32),
            'slp':        rng.uniform(99000, 103000, n).astype(np.float32),
            'air_temp':   rng.normal(300, 5, n).astype(np.float32),
            'dewpoint':   rng.normal(295, 5, n).astype(np.float32),
            'wind_speed': rng.uniform(0, 40, n).astype(np.float32),
            'wind_dir':   rng.uniform(0, 360, n).astype(np.float32),
        }
        if name == 'land':
            obs['station_pressure'] = rng.uniform(
                90000, 103000, n).astype(np.float32)
        if name == 'marine':
            obs['sst'] = rng.normal(302, 2, n).astype(np.float32)
            obs['sst'][:20] = np.nan                        # some missing
        sid = rng.choice([f'{name[:2].upper()}{i}' for i in range(8)], n)
        eids, eint, eorder, eoff = build_entity_spine(sid)
        write_volume(root / VOLUMES[name], obs, eint, eids, eorder, eoff)

    hours = np.arange(24, 96, 3)
    t = _ts(hours * 3600)
    m = len(t)
    cyc = {
        'report_timestamp': t,
        'lat': rng.uniform(10, 25, m).astype(np.float32),
        'lon': rng.uniform(-80, -50, m).astype(np.float32),
        'level': np.full(m, np.nan, np.float32),
        'sid': np.array(['AL012020'] * m),
        'usa_sshs': rng.integers(3, 9, m).astype(np.float32),
        'usa_wind': rng.uniform(35, 140, m).astype(np.float32),
        'usa_pres': rng.uniform(900, 1010, m).astype(np.float32),
        'is_subtropical': np.zeros(m, bool),
    }
    # Second storm sharing three timestamps with AL012020 -> those become
    # multi-driver times (shelf multi_times; the multistorm OOD scenario).
    mB = 3
    cyc_b = {
        'report_timestamp': _ts(np.array([48, 51, 54]) * 3600),
        'lat': rng.uniform(10, 25, mB).astype(np.float32),
        'lon': rng.uniform(-80, -50, mB).astype(np.float32),
        'level': np.full(mB, np.nan, np.float32),
        'sid': np.array(['AL022020'] * mB),
        'usa_sshs': np.array([4, 5, 6], np.float32),
        'usa_wind': rng.uniform(35, 140, mB).astype(np.float32),
        'usa_pres': rng.uniform(900, 1010, mB).astype(np.float32),
        'is_subtropical': np.zeros(mB, bool),
    }
    cyc = {k: np.concatenate([cyc[k], cyc_b[k]]) for k in cyc}
    order = np.argsort(cyc['report_timestamp'], kind='stable')
    cyc = {k: v[order] for k, v in cyc.items()}

    eids, eint, eorder, eoff = build_entity_spine(cyc['sid'])
    co, cf = build_category_index(cyc)
    write_volume(root / VOLUMES['cyclone'], cyc, eint, eids, eorder, eoff,
                 cat_order=co, cat_offsets=cf)
    build_bookshelf(root, verbose=False)
    return root


@pytest.fixture(scope='session')
def library(library_root):
    return load_library(library_root)
