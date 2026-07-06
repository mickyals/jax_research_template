"""
Tests for train/log.py — CALLBACKS registry, build_callbacks wiring, and
the two v1 callbacks on fakes (no arcana library needed; figure looks are
covered in tests/utils/plotting + tests/experiments/cyclone_jax/visualise).
Storm panels run basemap=False so cartopy features never render (Natural
Earth downloads at draw time). Fakes are shared:
tests/experiments/cyclone_jax/fakes.py.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.cyclone_jax.data.interface import DataBundle
from experiments.cyclone_jax.data.targets import TargetSpec
from experiments.cyclone_jax.train.log import (CALLBACKS, build_callbacks,
                                               end_of_run)
from tests.experiments.cyclone_jax.fakes import (N_CLS, FakeLoader,
                                                 FakeLogger, FakeStream)

DOMAIN = {'lat': [0, 30], 'lon': [-100, -30]}


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


def _dual_state(pred=2):
    """Echo X['logits'] on hand-collated CM batches, constant class on
    loader-built panel batches (end_of_run exercises both paths)."""
    logits = np.full((1, N_CLS), -100.0, np.float32)
    logits[0, pred] = 100.0
    return SimpleNamespace(
        params={},
        apply_fn=lambda v, X, train=False: X.get('logits', logits))


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

    def test_exact_and_per_class_accuracy_logged(self):
        logger = self._run(None)
        (m, _), = logger.metrics
        assert m['val/accuracy_exact'] == pytest.approx(2 / 3)
        names = _targets().class_names
        assert m[f'val/class_acc/{names[0]}'] == pytest.approx(0.5)
        assert m[f'val/class_acc/{names[1]}'] == pytest.approx(1.0)

    def test_count_and_pct_figures_logged_with_stills(self, tmp_path):
        logger = self._run(str(tmp_path))
        assert logger.figures == [('val/confusion_matrix', 10),
                                  ('val/confusion_matrix_pct', 10)]
        stills = sorted(p.name for p in (tmp_path / 'figures').iterdir())
        assert stills == ['confusion_matrix_pct_val_step0000010.png',
                          'confusion_matrix_pct_val_step0000010.svg',
                          'confusion_matrix_val_step0000010.png',
                          'confusion_matrix_val_step0000010.svg']

    def test_no_run_dir_still_logs(self):
        logger = self._run(None)
        assert logger.figures == [('val/confusion_matrix', 10),
                                  ('val/confusion_matrix_pct', 10)]


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
        # FakeLoader ids -1/1/0 -> one station per source
        assert 'land 1 | marine 1 | upper 1  (total 3)' in title
        assert 'resolvable' in title
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

    def test_station_ids_reach_the_panel_legend(self):
        from experiments.cyclone_jax.train.log import _render_fix
        data = _bundle({}, loader=FakeLoader(), splits={})
        fig = _render_fix(_const_state(pred=1), data, 0, DOMAIN,
                          basemap=False)
        # FakeLoader ids -1/1/0 -> one dot group per source, labelled
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels == ['land', 'upper', 'marine']
        plt.close(fig)


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
# end_of_run — test CM + storm sequence gif/stills
# ---------------------------------------------------------------------------

class TestEndOfRun:
    """Fix layout: sids B/A/B/B/A, times deliberately OUT of order for the
    B fixes (18h, 00h, 06h) — the sequence must sort them back."""

    SIDS = ('B', 'A', 'B', 'B', 'A')
    T0 = np.datetime64('2024-07-01T00')

    def _times(self):
        h = np.timedelta64(1, 'h')
        return [self.T0 + 18 * h, self.T0 + 0 * h, self.T0 + 0 * h,
                self.T0 + 6 * h, self.T0 + 12 * h]

    def _run(self, run_dir=None, storm_panels=None, n_frames=8,
             with_test_stream=True):
        streams = {'test': FakeStream([_batch([0, 1], [0, 0])])} \
            if with_test_stream else {}
        data = _bundle(streams, loader=FakeLoader(self.SIDS, self._times()),
                       splits={'test': np.arange(5)})
        logger = FakeLogger()
        cfg = {'data': {'storm_panels': storm_panels, 'domain': DOMAIN},
               'model': None, 'trainer': {'run_dir': run_dir}}
        end_of_run(cfg, data, logger, _dual_state(), global_step=99,
                   n_frames=n_frames, basemap=False)
        return logger

    def test_test_cm_logged(self):
        logger = self._run()
        m, step = logger.metrics[0]        # CM scalars logged first
        assert step == 99 and 'test/macro_precision' in m
        assert ('test/confusion_matrix', 99) in logger.figures

    def test_no_test_stream_skips_cm(self):
        logger = self._run(with_test_stream=False,
                           storm_panels={'test': 'B'})
        # no CM scalars — only the prediction-record identifiability ones
        assert not any('test/macro_precision' in m
                       for m, _ in logger.metrics)
        assert any('test/memorisation_ceiling' in m
                   for m, _ in logger.metrics)

    def test_no_storm_panels_test_key_skips_sequence(self):
        logger = self._run(storm_panels={'val': 'A'})
        assert [t for t, _ in logger.figures] == [
            'test/confusion_matrix', 'test/confusion_matrix_pct',
            'test/accuracy_hexbin',
            'test/storm_track_A', 'test/storm_track_B',
            'test/accuracy_vs_local_resolution']

    def test_sequence_gif_and_artifact(self, tmp_path):
        logger = self._run(run_dir=str(tmp_path), storm_panels={'test': 'B'})
        gif = tmp_path / 'figures' / 'storm_sequence_B.gif'
        assert gif.exists()
        assert [a for a in logger.artifacts if a[2] == 'figure']             == [('storm_sequence_B', str(gif), 'figure')]

    def test_stills_first_mid_last_time_ordered(self, tmp_path):
        logger = self._run(run_dir=str(tmp_path), storm_panels={'test': 'B'})
        seq = [t for t in logger.titles if t.startswith('B')]
        assert len(seq) == 3                       # 3 B-fixes -> 3 stills
        # title head: "B (2024)  2024-07-01T..Z" -> HH of the iso time
        hours = [t.split()[2][11:13] for t in seq]
        assert hours == ['00', '06', '18']

    def test_no_run_dir_no_gif_but_stills_logged(self):
        logger = self._run(storm_panels={'test': 'B'})
        assert logger.artifacts == []
        assert sum(t.startswith('test/storm_sequence_B') for t, _ in
                   logger.figures) == 3

    def test_n_frames_subsamples(self, tmp_path):
        logger = self._run(run_dir=str(tmp_path),
                           storm_panels={'test': 'B'}, n_frames=2)
        assert sum(t.startswith('test/storm_sequence_B') for t, _ in
                   logger.figures) == 2            # first/mid/last collapse

    def test_sid_list_renders_each(self, tmp_path):
        logger = self._run(run_dir=str(tmp_path),
                           storm_panels={'test': ['A', 'B']})
        names = [n for n, _, t in logger.artifacts if t == 'figure']
        assert names == ['storm_sequence_A', 'storm_sequence_B']

    def test_random_picks_one_test_sid(self):
        logger = self._run(storm_panels={'test': 'random'})
        seq = {t.split()[0] for t in logger.titles if not t.startswith('test')}
        assert len(seq) == 1 and seq <= {'A', 'B'}

    def test_absent_sid_raises(self):
        with pytest.raises(ValueError, match='NOPE'):
            self._run(storm_panels={'test': 'NOPE'})

    def test_all_figures_closed(self, tmp_path):
        self._run(run_dir=str(tmp_path), storm_panels={'test': 'B'})
        assert plt.get_fignums() == []

    def test_no_test_split_falls_back_to_train(self, tmp_path):
        """memorise scenarios: no test split exists, but storm_panels
        test-sids must still render sequences — from the train split."""
        data = _bundle({}, loader=FakeLoader(self.SIDS, self._times()),
                       splits={'train': np.arange(5)})
        logger = FakeLogger()
        cfg = {'data': {'storm_panels': {'test': 'B'}, 'domain': DOMAIN},
               'model': None, 'trainer': {'run_dir': str(tmp_path)}}
        end_of_run(cfg, data, logger, _dual_state(), global_step=42,
                   basemap=False)
        assert (tmp_path / 'figures' / 'storm_sequence_B.gif').exists()
        # stills tagged with the split they actually came from
        assert sum(t.startswith('train/storm_sequence_B') for t, _ in
                   logger.figures) == 3
        # still no test CM — only prediction-record identifiability scalars
        assert not any('test/macro_precision' in m
                       for m, _ in logger.metrics)


# ---------------------------------------------------------------------------
# end_of_run — per-fix prediction records / per-storm accuracy / hexbin
# ---------------------------------------------------------------------------

class TestPredictionRecords:
    """Fix layout: sids B/A/B/B/A, targets i % N_CLS = 0..4; _dual_state
    predicts class 2 everywhere -> only fix 2 (a B fix) is correct."""

    def _run(self, run_dir=None, splits=None):
        data = _bundle({}, loader=FakeLoader(('B', 'A', 'B', 'B', 'A')),
                       splits=splits if splits is not None
                       else {'test': np.arange(5)})
        logger = FakeLogger()
        cfg = {'data': {'domain': DOMAIN, 'batch_size': 2}, 'model': None,
               'trainer': {'run_dir': run_dir}}
        end_of_run(cfg, data, logger, _dual_state(pred=2), global_step=5,
                   basemap=False)
        return logger

    def _rows(self, path):
        import csv
        return list(csv.DictReader(path.read_text().splitlines()))

    def test_predictions_csv_identifies_failed_fixes(self, tmp_path):
        self._run(str(tmp_path))
        rows = self._rows(tmp_path / 'predictions_test.csv')
        assert [r['sid'] for r in rows] == ['B', 'A', 'B', 'B', 'A']
        assert [r['correct'] for r in rows] == ['0', '0', '1', '0', '0']
        assert rows[2]['true'] == 'Cat 2' and rows[2]['pred'] == 'Cat 2'
        assert rows[0]['pred'] == 'Cat 2'   # every miss names its pred
        assert rows[0]['n_stations'] == '3'
        assert rows[0]['lat'] and rows[0]['time']    # identity present
        # local-view columns: all 3 fake stations sit within ±5° of the
        # fix (13, -58) -> local count = total, finite resolvable_km
        assert rows[0]['n_stations_local'] == '3'
        assert float(rows[0]['resolvable_km_local']) > 0

    def test_local_resolution_scalars_logged(self):
        logger = self._run(None)
        (m, step), = [e for e in logger.metrics
                      if 'test/memorisation_ceiling' in e[0]]
        assert step == 5
        assert m['test/n_stations_local_mean'] == 3.0
        assert m['test/resolvable_km_local_mean'] > 0
        assert m['test/resolvable_km_global_mean'] > \
            m['test/resolvable_km_local_mean']   # FOV box >> ±5° box

    def test_per_storm_accuracy_worst_first(self, tmp_path):
        self._run(str(tmp_path))
        rows = self._rows(tmp_path / 'per_storm_accuracy_test.csv')
        assert [(r['sid'], r['n_fixes'], r['accuracy']) for r in rows] \
            == [('A', '2', '0.0000'), ('B', '3', '0.3333')]

    def test_hexbin_logged_even_without_run_dir(self):
        logger = self._run(None)
        assert ('test/accuracy_hexbin', 5) in logger.figures
        assert logger.artifacts == []          # no files -> no artifacts

    def test_records_logged_as_run_artifacts(self, tmp_path):
        logger = self._run(str(tmp_path))
        assert [(n, t) for n, _, t in logger.artifacts] == [
            ('predictions_test', 'predictions'),
            ('per_storm_accuracy_test', 'predictions'),
            ('identifiability_test', 'predictions')]

    def test_identical_splits_swept_once(self, tmp_path):
        """memorise: val is a copy of train — one sweep, one CSV pair."""
        idx = np.arange(5)
        logger = self._run(str(tmp_path),
                           splits={'train': idx, 'val': idx.copy()})
        assert (tmp_path / 'predictions_train.csv').exists()
        assert (tmp_path / 'per_storm_accuracy_train.csv').exists()
        assert not (tmp_path / 'predictions_val.csv').exists()
        assert [t for t, _ in logger.figures] == [
            'train/accuracy_hexbin',
            'train/storm_track_A', 'train/storm_track_B',
            'train/accuracy_vs_local_resolution']

    def test_storm_track_figures_for_hardest_storms(self):
        """Every imperfect storm (worst-first, capped) gets a track
        figure — the hexbin cross-read. A (0/2) before B (1/3)."""
        logger = self._run(None)
        tags = [t for t, _ in logger.figures]
        assert tags == ['test/accuracy_hexbin',
                        'test/storm_track_A', 'test/storm_track_B',
                        'test/accuracy_vs_local_resolution']
        a_title = logger.titles[tags.index('test/storm_track_A')]
        assert '0/2 correct' in a_title and '(0.00)' in a_title

    def test_fully_correct_storms_get_no_track_figure(self):
        """_dual_state(pred=2): fix targets are i % N_CLS, so with a
        single-storm split every fix of class 2 is correct."""
        data = _bundle({}, loader=FakeLoader(('A', 'A', 'A', 'A', 'A')),
                       splits={'test': np.array([2])})    # the correct fix
        logger = FakeLogger()
        cfg = {'data': {'domain': DOMAIN, 'batch_size': 2}, 'model': None,
               'trainer': {'run_dir': None}}
        end_of_run(cfg, data, logger, _dual_state(pred=2), global_step=5,
                   basemap=False)
        assert [t for t, _ in logger.figures] == [
            'test/accuracy_hexbin', 'test/accuracy_vs_local_resolution']

    def test_identifiability_ceiling_logged(self, tmp_path):
        """FakeLoader inputs are unique per fix -> ceiling 1.0, logged to
        the metrics backend and recorded as json."""
        import json
        logger = self._run(str(tmp_path))
        (m, step), = [e for e in logger.metrics
                      if 'test/memorisation_ceiling' in e[0]]
        assert step == 5
        assert m['test/memorisation_ceiling'] == 1.0
        assert m['test/n_unmemorisable'] == 0
        assert m['test/n_unique_inputs'] == 5
        rep = json.loads(
            (tmp_path / 'identifiability_test.json').read_text())
        assert rep['max_accuracy'] == 1.0 and rep['conflicts'] == []

    def test_conflicting_inputs_cap_the_ceiling(self, tmp_path):
        """All five fixes present the SAME input but different targets:
        majority group size 1 -> ceiling 1/5."""
        class CollidingLoader(FakeLoader):
            def build(self, i):
                s = super().build(0)               # identical x for all i
                s['y']['target'] = np.int32(i % N_CLS)
                s['y']['sid'] = str(self.fixes['sid'][i])
                return s

        data = _bundle({}, loader=CollidingLoader(('B', 'A', 'B', 'B', 'A')),
                       splits={'test': np.arange(5)})
        logger = FakeLogger()
        cfg = {'data': {'domain': DOMAIN, 'batch_size': 2}, 'model': None,
               'trainer': {'run_dir': str(tmp_path)}}
        end_of_run(cfg, data, logger, _dual_state(pred=2), global_step=5,
                   basemap=False)
        (m, _), = [e for e in logger.metrics
                   if 'test/memorisation_ceiling' in e[0]]
        assert m['test/memorisation_ceiling'] == pytest.approx(0.2)
        assert m['test/n_unmemorisable'] == 4
        assert m['test/n_unique_inputs'] == 1


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_v1_callbacks_registered(self):
        assert 'CONFUSION_MATRIX' in CALLBACKS
        assert 'STORM_PANEL' in CALLBACKS
