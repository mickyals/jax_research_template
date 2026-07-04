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
            split={'strategy': 'year',
                   'years': {'train': [2019], 'val': [2021],
                             'test': [2020]}}))
        assert len(data.splits['test']) == len(data.loader)  # all fixes 2020
        assert len(data.splits['train']) == 0
        assert set(data.streams) == {'test'}        # empty splits: no stream

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
            np.sort(np.asarray(data.loader.fixes['time'])))

    def test_len_batches_per_epoch(self, library_root):
        data = build_data(_cfg(library_root))
        n = len(data.splits['all'])
        assert len(data.streams['all']) == n // 4     # train-style drop_last
        assert isinstance(data.streams['all'], BatchStream)
