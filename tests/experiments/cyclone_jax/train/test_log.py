"""
Tests for train/log.py — CALLBACKS registry, build_callbacks wiring, and
the two v1 callbacks on fakes (no arcana library needed; figure looks are
covered in tests/utils/plotting + tests/experiments/cyclone_jax/visualise).
Storm panels run basemap=False so cartopy features never render (Natural
Earth downloads at draw time).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.cyclone_jax.data.interface import DataBundle
from experiments.cyclone_jax.data.targets import TargetSpec
from experiments.cyclone_jax.train.log import CALLBACKS, build_callbacks

N_CLS = 6
DOMAIN = {'lat': [0, 30], 'lon': [-100, -30]}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeStream:
    """Explicit-epoch batch stream; iter() is forbidden on purpose —
    callbacks must never advance the shared stream's epoch counter."""

    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def epoch(self, e):
        return iter(self._batches)

    def __iter__(self):
        raise AssertionError('callbacks must use explicit .epoch(e)')


class FakeLogger:
    def __init__(self):
        self.metrics, self.figures, self.titles = [], [], []

    def log_metrics(self, m, step):
        self.metrics.append((dict(m), step))

    def log_figure(self, tag, fig, step):
        self.figures.append((tag, step))
        self.titles.append(fig.axes[0].get_title())
        plt.close(fig)


class FakeLoader:
    """loader surface the storm_panel callback touches: fixes['sid'],
    build(i) -> collatable sample, inputs.pad_to."""

    def __init__(self, sids=('S0', 'S1', 'S2', 'S3')):
        self.fixes = {'sid': np.asarray(sids)}
        self.inputs = SimpleNamespace(pad_to=8)

    def build(self, i):
        return {
            'x': {'lat': np.float32([10.0 + i, 11.0, 12.0]),
                  'lon': np.float32([-60.0, -61.0, -62.0])},
            'y': {'target': np.int32(i % N_CLS),
                  'sid':    str(self.fixes['sid'][i]),
                  'lat':    np.float32(13.0),
                  'lon':    np.float32(-58.0),
                  'time':   np.datetime64('2024-07-01T12:00')},
        }


def _targets():
    return TargetSpec(variable='usa_sshs', kind='categorical',
                      class_set=(3, 4, 5, 6, 7, 8))


def _bundle(streams, loader=None, splits=None, norms=None):
    return DataBundle(lib={}, inputs=SimpleNamespace(pad_to=8),
                      targets=_targets(), loader=loader,
                      splits=splits or {}, streams=streams, norms=norms)


def _batch(preds, labels):
    """Hand-collated batch; the fake apply just reads back X['logits']."""
    logits = np.full((len(preds), N_CLS), -100.0, np.float32)
    logits[np.arange(len(preds)), preds] = 100.0
    return {'X': {'logits': logits},
            'y': np.asarray(labels, np.int32), 'meta': {}}


def _echo_state():
    """State whose forward returns X['logits'] (per-batch control)."""
    return SimpleNamespace(params={},
                           apply_fn=lambda v, X, train=False: X['logits'])


def _const_state(pred):
    """State predicting one fixed class for any (B=1) input."""
    logits = np.full((1, N_CLS), -100.0, np.float32)
    logits[0, pred] = 100.0
    return SimpleNamespace(params={},
                           apply_fn=lambda v, X, train=False: logits)


# ---------------------------------------------------------------------------
# confusion_matrix callback
# ---------------------------------------------------------------------------

class TestConfusionMatrixCallback:

    # truth [0,0]+[1], preds [0,1]+[1]:
    # precision c0 1/1, c1 1/2 -> 0.75 ; recall c0 1/2, c1 1/1 -> 0.75
    def _run(self, run_dir):
        batches = [_batch([0, 1], [0, 0]), _batch([1], [1])]
        data = _bundle({'val': FakeStream(batches)})
        logger = FakeLogger()
        fn = CALLBACKS.get('confusion_matrix',
                           ctx={'data': data, 'logger': logger,
                                'run_dir': run_dir}, split='val')
        fn(_echo_state(), 0, 10)
        return logger

    def test_exact_macro_metrics_logged(self, tmp_path):
        logger = self._run(str(tmp_path))
        (m, step), = logger.metrics
        assert step == 10
        assert m['val/macro_precision'] == pytest.approx(0.75)
        assert m['val/macro_recall'] == pytest.approx(0.75)

    def test_figure_logged_and_stills_saved(self, tmp_path):
        logger = self._run(str(tmp_path))
        assert logger.figures == [('val/confusion_matrix', 10)]
        stills = sorted(p.name for p in (tmp_path / 'figures').iterdir())
        assert stills == ['confusion_matrix_val_step0000010.png',
                          'confusion_matrix_val_step0000010.svg']

    def test_no_run_dir_still_logs(self):
        logger = self._run(None)
        assert logger.figures == [('val/confusion_matrix', 10)]


# ---------------------------------------------------------------------------
# storm_panel callback
# ---------------------------------------------------------------------------

class TestStormPanelCallback:

    def _ctx(self, run_dir=None, storm_panels=None, domain=DOMAIN):
        data = _bundle({'val': FakeStream([])}, loader=FakeLoader(),
                       splits={'val': np.arange(4)})
        logger = FakeLogger()
        return {'data': data, 'logger': logger, 'run_dir': run_dir,
                'storm_panels': storm_panels, 'domain': domain}, logger

    def test_panel_logged_with_composed_title(self, tmp_path):
        ctx, logger = self._ctx(run_dir=str(tmp_path))
        fn = CALLBACKS.get('storm_panel', ctx=ctx, split='val',
                           basemap=False)
        fn(_const_state(pred=2), 0, 7)
        assert logger.figures == [('val/storm_panel', 7)]
        # class index 2 in class_set (3..8) = category 5 = 'Cat 2'
        title, = logger.titles
        assert 'true' in title and 'pred Cat 2' in title
        assert 'n=3' in title and 'resolvable' in title
        stills = {p.suffix for p in (tmp_path / 'figures').iterdir()}
        assert stills == {'.svg', '.png'}

    def test_no_domain_omits_resolvable(self):
        ctx, logger = self._ctx(domain=None)
        fn = CALLBACKS.get('storm_panel', ctx=ctx, split='val',
                           basemap=False)
        fn(_const_state(pred=0), 0, 1)
        assert 'resolvable' not in logger.titles[0]

    def test_pinned_sid_always_picked(self):
        ctx, logger = self._ctx(storm_panels={'val': 'S2'})
        fn = CALLBACKS.get('storm_panel', ctx=ctx, split='val',
                           basemap=False)
        for step in (1, 2, 3):
            fn(_const_state(pred=0), 0, step)
        assert all(t.startswith('S2') for t in logger.titles)

    def test_pinned_sid_absent_raises_at_build(self):
        ctx, _ = self._ctx(storm_panels={'val': 'NOPE'})
        with pytest.raises(ValueError, match='NOPE'):
            CALLBACKS.get('storm_panel', ctx=ctx, split='val',
                          basemap=False)


# ---------------------------------------------------------------------------
# build_callbacks
# ---------------------------------------------------------------------------

def _cfg(callbacks, run_dir=None, data=None):
    return {'data': data or {}, 'model': None,
            'trainer': {'callbacks': callbacks, 'run_dir': run_dir}}


class TestBuildCallbacks:

    def _data(self, n_train_batches=5):
        train = FakeStream([_batch([0], [0])] * n_train_batches)
        val = FakeStream([_batch([0], [0])])
        return _bundle({'train': train, 'val': val})

    def test_no_block_gives_empty_list(self):
        assert build_callbacks(_cfg(None), self._data(), FakeLogger()) == []

    def test_default_every_is_train_batches_per_epoch(self):
        out = build_callbacks(_cfg([{'name': 'confusion_matrix'}]),
                              self._data(n_train_batches=5), FakeLogger())
        (fn, every), = out
        assert every == 5 and callable(fn)

    def test_explicit_every_wins(self):
        out = build_callbacks(
            _cfg([{'name': 'confusion_matrix', 'every': 7}]),
            self._data(), FakeLogger())
        assert out[0][1] == 7

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match='not registered'):
            build_callbacks(_cfg([{'name': 'nope'}]), self._data(),
                            FakeLogger())

    def test_unknown_spec_key_raises(self):
        with pytest.raises(ValueError, match='evrey'):
            build_callbacks(
                _cfg([{'name': 'confusion_matrix', 'evrey': 5}]),
                self._data(), FakeLogger())

    def test_missing_split_stream_raises(self):
        with pytest.raises(ValueError, match='test'):
            build_callbacks(
                _cfg([{'name': 'confusion_matrix', 'split': 'test'}]),
                self._data(), FakeLogger())

    def test_no_train_stream_needs_explicit_every(self):
        data = _bundle({'val': FakeStream([_batch([0], [0])])})
        with pytest.raises(ValueError, match='every'):
            build_callbacks(_cfg([{'name': 'confusion_matrix'}]), data,
                            FakeLogger())

    def test_kwargs_reach_the_factory(self):
        data = _bundle({'train': FakeStream([_batch([0], [0])]),
                        'val': FakeStream([])},
                       loader=FakeLoader(), splits={'val': np.arange(4)})
        out = build_callbacks(
            _cfg([{'name': 'storm_panel', 'split': 'val',
                   'kwargs': {'basemap': False}}],
                 data={'domain': DOMAIN}),
            data, FakeLogger())
        assert len(out) == 1 and callable(out[0][0])


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_v1_callbacks_registered(self):
        assert 'CONFUSION_MATRIX' in CALLBACKS
        assert 'STORM_PANEL' in CALLBACKS
