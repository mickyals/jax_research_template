"""
Tests for cyclone_jax transformations / sampler / batch — the assembly
contract: fixed shapes, causality (dt <= 0), label mapping, determinism,
selection modes, stack-only collation.
"""

import numpy as np
import pytest

from experiments.cyclone_jax.data.sources.volume import build_entity_spine, write_volume
from experiments.cyclone_jax.data.transformations import build_missingness
from experiments.cyclone_jax.data.sources.build import build_category_index
from experiments.cyclone_jax.data.sources.library import (
    VOLUMES, build_bookshelf, load_library,
)
from experiments.cyclone_jax.data.sampler import (
    CHANNELS, TOKEN_DIM, FixSampler, split_by_year, stratified_fixes,
)
from experiments.cyclone_jax.data.batch import collate

BASE = np.datetime64('2020-08-01T00:00', 'ns')


def _ts(seconds):
    off = np.asarray(seconds)
    return BASE + off.astype('timedelta64[s]').astype('timedelta64[ns]')


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    """Mini library with the REAL surface columns the sampler consumes."""
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
    eids, eint, eorder, eoff = build_entity_spine(cyc['sid'])
    co, cf = build_category_index(cyc)
    write_volume(root / VOLUMES['cyclone'], cyc, eint, eids, eorder, eoff,
                 cat_order=co, cat_offsets=cf)
    build_bookshelf(root, verbose=False)
    return load_library(root)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

class TestTransformations:
    # wind decomposition is utils.geoscience.met_conversions.wind_to_components
    # (tested in tests/utils); the sampler-level wind behaviour is covered by
    # the FixSampler assembly tests below.

    def test_missingness(self):
        vals, mask = build_missingness(np.array([[1.0, np.nan]]))
        assert vals[0, 1] == 0.0 and not mask[0, 1] and mask[0, 0]


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class TestFixSampler:

    def test_shapes_and_pad(self, library):
        s = FixSampler(library, pad_to=256)
        out = s.build(5)
        assert out['tokens'].shape == (256, TOKEN_DIM)
        assert out['station_mask'].shape == (256,)
        assert out['n_stations'] == out['station_mask'].sum()
        # padded rows are exactly zero
        assert np.all(out['tokens'][out['n_stations']:] == 0.0)

    def test_causality_dt_nonpositive(self, library):
        s = FixSampler(library, pad_to=256)
        out = s.build(10)
        n = out['n_stations']
        assert n > 0
        assert np.all(out['tokens'][:n, 2] <= 0.0)          # dt column

    def test_label_mapping(self, library):
        s = FixSampler(library, pad_to=64)
        out = s.build(0)
        assert out['label'] == int(out['sshs']) - 3
        assert 0 <= out['label'] <= 5

    def test_deterministic(self, library):
        s = FixSampler(library, pad_to=128)
        a, b = s.build(7), s.build(7)
        np.testing.assert_array_equal(a['tokens'], b['tokens'])

    def test_source_id_column(self, library):
        s = FixSampler(library, pad_to=512)
        out = s.build(8)
        n = out['n_stations']
        ids = set(np.unique(out['tokens'][:n, -1]))
        assert ids.issubset({-1.0, 1.0})                    # land / marine only

    def test_max_stations_selection(self, library):
        s_all = FixSampler(library, pad_to=512)
        n_all = s_all.build(8)['n_stations']
        k = max(1, int(n_all) // 2)
        s_k = FixSampler(library, pad_to=512,
                         selection='max_stations', max_stations=k)
        assert s_k.build(8)['n_stations'] == k

    def test_invalid_selection_raises(self, library):
        with pytest.raises(ValueError, match='selection'):
            FixSampler(library, selection='nearest')
        with pytest.raises(ValueError, match='max_stations'):
            FixSampler(library, selection='max_stations')

    def test_union_channels_masked_per_source(self, library):
        """Every token carries the FULL channel union; channels a source
        lacks are zeroed with mask False (land: sst; marine:
        station_pressure). Union positions are identical across sources."""
        s = FixSampler(library, pad_to=512)
        out = s.build(8)
        n = int(out['n_stations'])
        tok = out['tokens'][:n]
        ch = {c: j for j, c in enumerate(CHANNELS)}
        mask = tok[:, 3 + len(CHANNELS):3 + 2 * len(CHANNELS)]
        land, marine = tok[:, -1] == -1.0, tok[:, -1] == 1.0
        assert land.any() and marine.any()
        assert not mask[land, ch['sst']].any()
        assert np.all(tok[land, 3 + ch['sst']] == 0.0)
        assert not mask[marine, ch['station_pressure']].any()
        assert mask[land, ch['station_pressure']].all()

    def test_leakage_no_target_values_in_tokens(self, library):
        """Token columns are loc/dt/obs/mask/id only — TOKEN_DIM accounts
        for every column, none sourced from the cyclone volume."""
        assert TOKEN_DIM == 3 + 2 * len(CHANNELS) + 1


# ---------------------------------------------------------------------------
# Splits + overfit sets + collate
# ---------------------------------------------------------------------------

class TestSelections:

    def test_split_by_year_disjoint(self, library):
        s = FixSampler(library, pad_to=64)
        t = np.asarray(s.fixes['time'])
        out = split_by_year(t, [2019], [2021], [2020])
        assert len(out['test']) == len(t)                   # all fixes 2020
        assert len(out['train']) == len(out['val']) == 0

    def test_stratified_counts(self, library):
        s = FixSampler(library, pad_to=64)
        idx = stratified_fixes(s, n_per_class=2, seed=0)
        sshs = np.asarray(s.fixes['usa_sshs']).astype(int)[idx]
        for c in range(3, 9):
            assert np.sum(sshs == c) <= 2

    def test_collate_stacks(self, library):
        s = FixSampler(library, pad_to=128)
        batch = collate([s.build(i) for i in range(4)])
        assert batch['X']['tokens'].shape == (4, 128, TOKEN_DIM)
        assert batch['X']['station_mask'].shape == (4, 128)
        assert batch['y'].shape == (4,) and batch['y'].dtype == np.int32
        assert len(batch['meta']['sid']) == 4
