"""
Tests for cyclone_jax interface — the config -> DataBundle contract: split
strategies, stream construction/coverage, seed threading, and guards.
"""

import numpy as np
import pytest

from experiments.cyclone_jax.data.interface import BatchStream, build_data


def _cfg(root, **over):
    cfg = {'root': str(root), 'sshs_min': 3, 'batch_size': 4}
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# build_data + splits
# ---------------------------------------------------------------------------

class TestBuildData:

    def test_bundle_resolves_everything(self, library_root):
        data = build_data(_cfg(library_root))
        assert data.inputs.sources == ('land', 'marine')
        assert data.targets.n_classes == 6
        assert len(data.loader) > 0
        assert set(data.splits) == {'all'}          # no split block
        assert set(data.streams) == {'all'}

    def test_year_split(self, library_root):
        data = build_data(_cfg(
            library_root,
            split={'strategy': 'year', 'exclude_multistorm': False,
                   'years': {'train': [2019], 'val': [2021],
                             'test': [2020]}}))
        assert len(data.splits['test']) == len(data.loader)  # all fixes 2020
        assert len(data.splits['train']) == 0
        assert set(data.streams) == {'test'}        # empty splits: no stream

    def test_year_split_excludes_multistorm_by_default(self, library_root):
        split = {'strategy': 'year',
                 'years': {'train': [2019], 'val': [2021], 'test': [2020]}}
        data = build_data(_cfg(library_root, split=split))
        multi = np.asarray(data.lib['shelves']['cyclone']['multi_times'])
        assert len(multi) > 0                       # fixture has 2 storms
        kept = np.asarray(data.loader.fixes['time'])[data.splits['test']]
        assert not np.isin(kept, multi).any()
        n_multi_fixes = np.isin(
            np.asarray(data.loader.fixes['time']), multi).sum()
        assert len(data.splits['test']) == len(data.loader) - n_multi_fixes

    def test_multistorm_split_is_the_excluded_complement(self, library_root):
        data = build_data(_cfg(library_root,
                               split={'strategy': 'multistorm'}))
        multi = np.asarray(data.lib['shelves']['cyclone']['multi_times'])
        times = np.asarray(data.loader.fixes['time'])[data.splits['test']]
        assert len(times) > 0 and np.isin(times, multi).all()
        # both storms' fixes at a shared timestamp are included
        sids = np.asarray(data.loader.fixes['sid'])[data.splits['test']]
        assert {'AL012020', 'AL022020'} <= set(sids)

    def test_year_split_missing_key_raises(self, library_root):
        with pytest.raises(ValueError, match='split.years'):
            build_data(_cfg(library_root,
                            split={'strategy': 'year',
                                   'years': {'train': [2019]}}))

    def test_stratified_split(self, library_root):
        data = build_data(_cfg(
            library_root, split={'strategy': 'stratified', 'n_per_class': 2}))
        np.testing.assert_array_equal(data.splits['train'],
                                      data.splits['val'])
        assert len(data.splits['train']) <= 2 * data.targets.n_classes

    def test_stratified_requires_n_per_class(self, library_root):
        with pytest.raises(ValueError, match='n_per_class'):
            build_data(_cfg(library_root, split={'strategy': 'stratified'}))

    def test_memorise_split_train_eq_val_eq_all(self, library_root):
        data = build_data(_cfg(
            library_root,
            split={'strategy': 'memorise', 'exclude_multistorm': False}))
        np.testing.assert_array_equal(data.splits['train'],
                                      np.arange(len(data.loader)))
        np.testing.assert_array_equal(data.splits['train'],
                                      data.splits['val'])
        assert set(data.streams) == {'train', 'val'}

    def test_memorise_excludes_multistorm_by_default(self, library_root):
        data = build_data(_cfg(library_root,
                               split={'strategy': 'memorise'}))
        multi = np.asarray(data.lib['shelves']['cyclone']['multi_times'])
        kept = np.asarray(data.loader.fixes['time'])[data.splits['train']]
        assert len(kept) and not np.isin(kept, multi).any()

    def test_unknown_strategy_raises(self, library_root):
        with pytest.raises(ValueError, match='split.strategy'):
            build_data(_cfg(library_root, split={'strategy': 'random'}))

    def test_no_batch_size_no_streams(self, library_root):
        data = build_data(_cfg(library_root, batch_size=None))
        assert data.streams == {} and len(data.splits['all'])


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

class TestStreams:

    def test_batches_are_device_schema(self, library_root):
        data = build_data(_cfg(library_root))
        batch = next(iter(data.streams['all']))
        pad_to = data.inputs.pad_to
        assert batch['X']['obs'].shape == (4, pad_to, data.inputs.n_channels)
        assert batch['y'].shape == (4,)
        assert len(batch['meta']['sid']) == 4

    def test_seed_threads_to_stream_order(self, library_root):
        a = build_data(_cfg(library_root), seed=1)
        b = build_data(_cfg(library_root), seed=1)
        c = build_data(_cfg(library_root), seed=2)
        ya = next(a.streams['all'].epoch(0))['meta']['time']
        yb = next(b.streams['all'].epoch(0))['meta']['time']
        yc = next(c.streams['all'].epoch(0))['meta']['time']
        np.testing.assert_array_equal(ya, yb)
        assert not np.array_equal(ya, yc)

    def test_iter_advances_epoch(self, library_root):
        data = build_data(_cfg(library_root))
        s = data.streams['all']
        first = next(iter(s))['meta']['time']       # epoch 0
        second = next(iter(s))['meta']['time']      # epoch 1
        assert not np.array_equal(first, second)
        np.testing.assert_array_equal(
            first, next(s.epoch(0))['meta']['time'])  # explicit replay

    def test_val_style_stream_covers_all_indices(self, library_root):
        data = build_data(_cfg(
            library_root, batch_size=4,
            split={'strategy': 'year',
                   'years': {'train': [2019], 'val': [2020], 'test': [2021]}}))
        s = data.streams['val']
        seen = np.concatenate(
            [b['meta']['time'] for b in s.epoch(0)])
        assert len(seen) == len(data.splits['val'])   # partial batch kept
        np.testing.assert_array_equal(
            np.sort(seen),
            np.sort(np.asarray(
                data.loader.fixes['time'])[data.splits['val']]))

    def test_len_batches_per_epoch(self, library_root):
        data = build_data(_cfg(library_root))
        n = len(data.splits['all'])
        assert len(data.streams['all']) == n // 4     # train-style drop_last
        assert isinstance(data.streams['all'], BatchStream)


# ---------------------------------------------------------------------------
# Normalisation through build_data (policy detail: test_normalise.py)
# ---------------------------------------------------------------------------

NORM = {'normalise': {'method': 'standardise', 'stats': 'auto'}}


class TestBuildDataNormalisation:

    def test_no_block_means_raw(self, library_root):
        data = build_data(_cfg(library_root))
        assert data.norms is None and data.loader.norms is None

    def test_norms_attached_and_shared(self, library_root):
        data = build_data(_cfg(library_root, **NORM))
        assert data.norms is not None
        assert data.loader.norms is data.norms

    def test_stats_come_from_the_train_split_only(self, library_root):
        """Different train splits -> different stats (they are properties
        of a training distribution, computed + logged per run)."""
        full = build_data(_cfg(library_root, **NORM))
        sub = build_data(_cfg(
            library_root, **NORM,
            split={'strategy': 'stratified', 'n_per_class': 2}))
        assert (full.norms.stats['obs']['slp']['count']
                > sub.norms.stats['obs']['slp']['count'])

    def test_batches_are_normalised(self, library_root):
        data = build_data(_cfg(library_root, **NORM))
        batch = next(iter(data.streams['all']))
        obs = batch['X']['obs'][batch['X']['missing']]
        assert abs(float(obs.mean())) < 0.5           # z-scored, not Pa
        lat = batch['X']['lat'][batch['X']['station_mask']]
        assert float(lat.min()) >= -1.0 and float(lat.max()) <= 1.0

    def test_scenario_without_train_split_raises_instructive(
            self, library_root):
        """The documented rule: a stress-test scenario must NAME its
        training distribution (inline stats or the run's saved stats)."""
        with pytest.raises(ValueError, match='norm_stats.json'):
            build_data(_cfg(library_root, **NORM,
                            split={'strategy': 'multistorm'}))
