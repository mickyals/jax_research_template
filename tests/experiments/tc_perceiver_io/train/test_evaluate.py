"""
Tests for experiments/tc_perceiver_io/train/evaluate.py.

All tests use synthetic in-memory data — no disk access required.

Coverage
--------
TestConfusionMatrix       shape; diagonal counts; off-diagonal counts;
                          total count; empty prediction handled
TestPerClassMetrics       perfect class gives p/r/f1=1; zero support;
                          keys present; tp/fp/fn arithmetic
TestBinaryMetrics         all TC correct; all no-storm correct; perfect binary;
                          mixed case; all keys present; counts add up
TestCollectPredictions    output shapes; preds are argmax of logits;
                          labels match loader; no model mutation; meta passthrough
TestPerStormMetrics       groups by SID, skips background
TestPrintReport           report runs; prints scalar metrics + binary + per-class
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.tc_perceiver_io.train.evaluate import (
    binary_metrics,
    collect_class_exemplars,
    collect_predictions,
    confusion_matrix,
    per_class_metrics,
    per_storm_metrics,
    per_sample_table,
    build_prediction_outputs,
    print_report,
)
from experiments.tc_perceiver_io.train.metrics import build_metrics_fns
from experiments.tc_perceiver_io.train.model import TCPerceiverIO
from experiments.tc_perceiver_io.data.sources.ibtracs import N_CLASSES

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

B     = 6
M     = 8
F     = 5
HEADS = 2
EMBED = 32
NLAT  = 6


def _init_model() -> tuple[TCPerceiverIO, dict]:
    """Return a tiny TCPerceiverIO and its initialized variables."""
    model = TCPerceiverIO(
        embed_dim          = EMBED,
        num_heads          = HEADS,
        num_latents        = NLAT,
        num_process_layers = 1,
        fourier_dim        = 16,
        n_obs_features     = F,
        n_classes          = N_CLASSES,
    )
    rng  = np.random.default_rng(0)
    obs  = jnp.array(rng.standard_normal((B, M, F)).astype(np.float32))
    X    = {
        'station_obs':    obs,
        'station_coords': jnp.zeros((B, M, 2)),
        'station_mask':   jnp.ones((B, M), dtype=bool),
        'obs_mask':       jnp.ones((B, M, F), dtype=bool),
    }
    variables = model.init({'params': jax.random.PRNGKey(0)}, X, train=False)
    return model, variables


def _fake_batch() -> dict:
    rng  = np.random.default_rng(1)
    obs  = rng.standard_normal((B, M, F)).astype(np.float32)
    return {
        'X': {
            'station_obs':    jnp.array(obs),
            'station_coords': jnp.array(rng.uniform(-1., 1., (B, M, 2)).astype(np.float32)),
            'station_mask':   jnp.ones((B, M), dtype=bool),
            'obs_mask':       jnp.ones((B, M, F), dtype=bool),
        },
        'y': jnp.array(rng.integers(0, N_CLASSES, size=B), dtype=jnp.int32),
    }


class _FakeLoader:
    """Minimal re-iterable loader yielding fake batches."""
    def __init__(self, n: int = 2):
        self._batches = [_fake_batch() for _ in range(n)]
    def __iter__(self):
        return iter(self._batches)


# ---------------------------------------------------------------------------
# TestConfusionMatrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:

    def test_output_shape(self):
        preds  = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        labels = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        cm     = confusion_matrix(preds, labels)
        assert cm.shape == (N_CLASSES, N_CLASSES)

    def test_perfect_predictions_are_on_diagonal(self):
        preds  = np.arange(N_CLASSES, dtype=np.int32)
        labels = np.arange(N_CLASSES, dtype=np.int32)
        cm     = confusion_matrix(preds, labels)
        assert np.all(cm == np.eye(N_CLASSES, dtype=np.int64))

    def test_off_diagonal_entry(self):
        preds  = np.array([5], dtype=np.int32)
        labels = np.array([3], dtype=np.int32)
        cm     = confusion_matrix(preds, labels)
        assert cm[3, 5] == 1
        assert cm.sum() == 1

    def test_total_count_matches_n_samples(self):
        n      = 20
        rng    = np.random.default_rng(0)
        preds  = rng.integers(0, N_CLASSES, size=n).astype(np.int32)
        labels = rng.integers(0, N_CLASSES, size=n).astype(np.int32)
        assert confusion_matrix(preds, labels).sum() == n

    def test_empty_predictions_returns_zeros(self):
        cm = confusion_matrix(np.array([], dtype=np.int32),
                              np.array([], dtype=np.int32))
        assert cm.shape == (N_CLASSES, N_CLASSES)
        assert cm.sum() == 0


# ---------------------------------------------------------------------------
# TestPerClassMetrics
# ---------------------------------------------------------------------------

class TestPerClassMetrics:

    def _perfect_cm(self) -> np.ndarray:
        """Confusion matrix where every class is perfectly predicted."""
        return np.eye(N_CLASSES, dtype=np.int64) * 10

    def test_keys_present_for_every_class(self):
        pcm = per_class_metrics(self._perfect_cm())
        assert set(pcm.keys()) == set(range(N_CLASSES))
        for k in pcm:
            assert set(pcm[k].keys()) == {'precision', 'recall', 'f1', 'support'}

    def test_perfect_predictions_give_ones(self):
        pcm = per_class_metrics(self._perfect_cm())
        for k in range(N_CLASSES):
            assert pcm[k]['precision'] == pytest.approx(1.0)
            assert pcm[k]['recall']    == pytest.approx(1.0)
            assert pcm[k]['f1']        == pytest.approx(1.0)

    def test_zero_support_class_gives_zero_metrics(self):
        cm       = self._perfect_cm()
        cm[3, :] = 0   # class 3 has no actual samples
        cm[:, 3] = 0
        pcm = per_class_metrics(cm)
        assert pcm[3]['support'] == 0
        assert pcm[3]['recall']  == pytest.approx(0.0)

    def test_support_matches_row_sum(self):
        rng = np.random.default_rng(0)
        cm  = rng.integers(0, 10, size=(N_CLASSES, N_CLASSES)).astype(np.int64)
        pcm = per_class_metrics(cm)
        for k in range(N_CLASSES):
            assert pcm[k]['support'] == int(cm[k, :].sum())

    def test_f1_is_harmonic_mean_of_p_and_r(self):
        pcm = per_class_metrics(self._perfect_cm())
        for k in range(N_CLASSES):
            p, r  = pcm[k]['precision'], pcm[k]['recall']
            f1_expected = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            assert pcm[k]['f1'] == pytest.approx(f1_expected)


# ---------------------------------------------------------------------------
# TestBinaryMetrics
# ---------------------------------------------------------------------------

class TestBinaryMetrics:

    def test_all_tc_correct(self):
        preds  = np.ones(10, dtype=np.int32) * 5   # all predict TC
        labels = np.ones(10, dtype=np.int32) * 3   # all actually TC
        bm = binary_metrics(preds, labels)
        assert bm['accuracy']  == pytest.approx(1.0)
        assert bm['precision'] == pytest.approx(1.0)
        assert bm['recall']    == pytest.approx(1.0)
        assert bm['f1']        == pytest.approx(1.0)

    def test_all_no_storm_correct(self):
        preds  = np.zeros(10, dtype=np.int32)   # all predict background
        labels = np.zeros(10, dtype=np.int32)   # all actually background
        bm = binary_metrics(preds, labels)
        assert bm['accuracy'] == pytest.approx(1.0)

    def test_all_wrong_binary(self):
        preds  = np.ones(10, dtype=np.int32) * 5   # predict TC
        labels = np.zeros(10, dtype=np.int32)       # all background
        bm = binary_metrics(preds, labels)
        assert bm['accuracy'] == pytest.approx(0.0)
        assert bm['tn'] == 0
        assert bm['fp'] == 10

    def test_all_keys_present(self):
        bm = binary_metrics(np.array([0, 1]), np.array([0, 1]))
        assert set(bm.keys()) == {
            'accuracy', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn'
        }

    def test_counts_add_up_to_total(self):
        rng    = np.random.default_rng(0)
        preds  = rng.integers(0, N_CLASSES, size=40).astype(np.int32)
        labels = rng.integers(0, N_CLASSES, size=40).astype(np.int32)
        bm = binary_metrics(preds, labels)
        assert bm['tp'] + bm['fp'] + bm['fn'] + bm['tn'] == 40

    def test_balanced_50_50_gives_reasonable_accuracy(self):
        preds  = np.array([5] * 10 + [0] * 10, dtype=np.int32)   # perfect predictions
        labels = np.array([5] * 10 + [0] * 10, dtype=np.int32)
        bm = binary_metrics(preds, labels)
        assert bm['accuracy'] == pytest.approx(1.0)
        assert bm['tp'] == 10
        assert bm['tn'] == 10


# ---------------------------------------------------------------------------
# TestCollectPredictions
# ---------------------------------------------------------------------------

class TestCollectPredictions:

    @pytest.fixture
    def model_vars(self):
        return _init_model()

    def test_output_shapes(self, model_vars):
        model, variables = model_vars
        loader = _FakeLoader(n=2)
        preds, labels, logits, _ = collect_predictions(model, variables, loader)
        n_total = B * 2
        assert preds.shape  == (n_total,)
        assert labels.shape == (n_total,)
        assert logits.shape == (n_total, N_CLASSES)

    def test_preds_are_argmax_of_logits(self, model_vars):
        model, variables = model_vars
        loader = _FakeLoader(n=1)
        preds, _, logits, _ = collect_predictions(model, variables, loader)
        expected = logits.argmax(axis=-1)
        np.testing.assert_array_equal(preds, expected)

    def test_labels_match_loader(self, model_vars):
        model, variables = model_vars
        batch  = _fake_batch()
        loader = [batch]   # single-batch loader (plain list)
        _, labels, _, _ = collect_predictions(model, variables, loader)
        np.testing.assert_array_equal(labels, np.asarray(batch['y']))

    def test_preds_in_valid_class_range(self, model_vars):
        model, variables = model_vars
        preds, _, _, _ = collect_predictions(model, variables, _FakeLoader(n=2))
        assert preds.min() >= 0
        assert preds.max() < N_CLASSES

    def test_logits_are_finite(self, model_vars):
        model, variables = model_vars
        _, _, logits, _ = collect_predictions(model, variables, _FakeLoader(n=2))
        assert np.all(np.isfinite(logits))

    def test_meta_none_without_meta_in_batches(self, model_vars):
        model, variables = model_vars
        _, _, _, meta = collect_predictions(model, variables, _FakeLoader(n=2))
        assert meta is None

    def test_meta_concatenated_across_batches(self, model_vars):
        model, variables = model_vars
        batches = []
        for i in range(2):
            b = _fake_batch()
            b['meta'] = {
                'sid':         [f'SID{i}'] * (B - 1) + [None],
                'iso_time':    np.full(B, 100 + i, dtype=np.int64),
                'query_lat':   np.full(B, 15.0, dtype=np.float32),
                'query_lon':   np.full(B, -75.0, dtype=np.float32),
                'n_available': np.full(B, 7, dtype=np.int32),
                'n_used':      np.full(B, 5, dtype=np.int32),
            }
            batches.append(b)
        _, _, _, meta = collect_predictions(model, variables, batches)
        assert meta is not None
        assert len(meta['sid']) == 2 * B
        assert meta['sid'][0] == 'SID0' and meta['sid'][B] == 'SID1'
        assert meta['sid'][B - 1] is None
        assert meta['iso_time'].shape == (2 * B,)


# ---------------------------------------------------------------------------
# TestPerStormMetrics
# ---------------------------------------------------------------------------

class TestPerStormMetrics:

    def test_groups_by_sid_and_skips_background(self):
        preds  = np.array([5, 5, 0, 7, 0], dtype=np.int32)
        labels = np.array([5, 6, 0, 7, 1], dtype=np.int32)
        sids   = ['A', 'A', None, 'B', None]
        out = per_storm_metrics(preds, labels, sids)
        assert set(out.keys()) == {'A', 'B'}
        assert out['A']['n'] == 2
        assert out['A']['accuracy']  == pytest.approx(0.5)
        assert out['A']['mae_class'] == pytest.approx(0.5)
        assert out['B']['n'] == 1
        assert out['B']['accuracy'] == pytest.approx(1.0)

    def test_empty_when_all_background(self):
        out = per_storm_metrics(
            np.array([0, 0], dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
            [None, None],
        )
        assert out == {}


# ---------------------------------------------------------------------------
# TestCollectClassExemplars
# ---------------------------------------------------------------------------

class TestCollectClassExemplars:

    def test_one_example_per_class(self):
        b1 = _fake_batch(); b1['y'] = jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
        b2 = _fake_batch(); b2['y'] = jnp.array([3, 3, 3, 3, 3, 3], dtype=jnp.int32)
        ex = collect_class_exemplars([b1, b2], n_classes=N_CLASSES)
        assert ex is not None
        X, labels, metas = ex
        assert sorted(labels.tolist()) == [0, 1, 2, 3]   # one per present class
        assert X['station_obs'].shape[0] == 4            # stacked over classes
        assert len(metas) == 4 and all(m is None for m in metas)  # no 'meta' here

    def test_empty_loader_returns_none(self):
        assert collect_class_exemplars([], n_classes=N_CLASSES) is None

    def test_meta_captured_per_exemplar(self):
        b = _fake_batch(); b['y'] = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32)
        b['meta'] = {'sid': [None, 'S1', 'S2', 'S3', 'S4', 'S5'],
                     'query_lat': np.full(B, 15.0, np.float32),
                     'query_lon': np.full(B, -75.0, np.float32)}
        X, labels, metas = collect_class_exemplars([b], n_classes=N_CLASSES)
        assert len(labels) == 6
        # class 1's exemplar carries its sid
        i = labels.tolist().index(1)
        assert metas[i]['sid'] == 'S1'


# ---------------------------------------------------------------------------
# TestPrintReport
# ---------------------------------------------------------------------------

class TestPrintReport:

    def _data(self):
        rng    = np.random.default_rng(0)
        n      = 40
        labels = rng.integers(0, N_CLASSES, size=n).astype(np.int32)
        logits = rng.standard_normal((n, N_CLASSES)).astype(np.float32)
        preds  = logits.argmax(-1).astype(np.int32)
        return preds, labels, logits

    def test_report_runs_and_prints_sections(self, capsys):
        preds, labels, logits = self._data()
        print_report(preds, labels, logits, build_metrics_fns(), split='test')
        out = capsys.readouterr().out
        assert 'TEST evaluation' in out
        assert 'Binary detection' in out
        assert 'Per-class metrics' in out
        # Full-set metrics block (mAP / pr_auc) over the accumulated split.
        assert 'Full-set metrics' in out
        assert 'test/mAP' in out and 'test/pr_auc' in out

    def test_report_prints_each_scalar_metric(self, capsys):
        preds, labels, logits = self._data()
        metrics_fns = build_metrics_fns()   # {'loss','binary_accuracy','mae_class'}
        print_report(preds, labels, logits, metrics_fns, split='val')
        out = capsys.readouterr().out
        for name in metrics_fns:
            assert f'val/{name}' in out


# ---------------------------------------------------------------------------
# Per-sample classification table
# ---------------------------------------------------------------------------

class TestPerSampleTable:

    _NAMES = ['Background', 'Disturbance', 'Depression', 'Storm',
              'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']

    def _inputs(self, n=5):
        rng    = np.random.default_rng(0)
        preds  = np.array([0, 4, 2, 0, 8])[:n]
        labels = np.array([0, 4, 3, 1, 8])[:n]
        logits = rng.standard_normal((n, 9)).astype(np.float32)
        meta = {
            'sid':         ['2019A', None, '2019B', None, '2020C'][:n],
            'iso_time':    np.arange(n, dtype=np.int64) * 10_800_000_000_000
                           + 1_567_296_000_000_000_000,
            'query_lat':   np.linspace(5, 25, n).astype(np.float32),
            'query_lon':   np.linspace(-90, -55, n).astype(np.float32),
            'n_used':      np.array([3, 7, 1, 9, 4])[:n].astype(np.int32),
            'n_available': np.array([5, 9, 1, 12, 6])[:n].astype(np.int32),
        }
        return preds, labels, logits, meta

    def test_columns_and_rows(self):
        preds, labels, logits, meta = self._inputs()
        df = per_sample_table(preds, labels, logits, meta, self._NAMES)
        assert len(df) == 5
        for c in ('sid', 'iso_time', 'query_lat', 'query_lon', 'true',
                  'true_name', 'pred', 'pred_name', 'correct', 'max_prob',
                  'p_true', 'n_used', 'n_available'):
            assert c in df.columns

    def test_correct_and_names(self):
        preds, labels, logits, meta = self._inputs()
        df = per_sample_table(preds, labels, logits, meta, self._NAMES)
        assert df['correct'].tolist() == [True, True, False, False, True]
        assert df['true_name'].iloc[1] == 'Cat1'
        assert df['pred_name'].iloc[2] == 'Depression'
        # background sid (None) is filled for readability
        assert df['sid'].iloc[1] == 'background'

    def test_probs_in_range(self):
        preds, labels, logits, meta = self._inputs()
        df = per_sample_table(preds, labels, logits, meta, self._NAMES)
        assert (df['max_prob'] >= df['p_true'] - 1e-9).all()
        assert ((df['max_prob'] >= 0) & (df['max_prob'] <= 1)).all()

    def test_full_per_class_distribution(self):
        preds, labels, logits, meta = self._inputs()
        df = per_sample_table(preds, labels, logits, meta, self._NAMES)
        pcols = [f"p_{n.replace(' ', '_')}" for n in self._NAMES]
        assert all(c in df.columns for c in pcols)        # one column per class
        row_sums = df[pcols].to_numpy().sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-9)      # a proper distribution
        # max_prob equals the row-wise max of the per-class columns
        assert np.allclose(df['max_prob'].to_numpy(),
                           df[pcols].to_numpy().max(axis=1))

    def test_handles_missing_meta(self):
        preds, labels, logits, _ = self._inputs()
        df = per_sample_table(preds, labels, logits, None, self._NAMES)
        assert len(df) == 5
        assert 'true_name' in df.columns


class TestBuildPredictionOutputs:
    """One source feeds both the CSV table and the spatial maps."""

    _NAMES = ['Background', 'Disturbance', 'Depression', 'Storm',
              'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']

    def _inputs(self, n=6):
        rng    = np.random.default_rng(2)
        preds  = np.array([0, 4, 2, 0, 8, 1])[:n]
        labels = np.array([0, 4, 3, 1, 8, 1])[:n]
        logits = rng.standard_normal((n, 9)).astype(np.float32)
        meta = {
            'sid':       ['A', None, 'B', None, 'C', 'D'][:n],
            'query_lat': np.linspace(5, 25, n).astype(np.float32),
            'query_lon': np.linspace(-90, -55, n).astype(np.float32),
        }
        return preds, labels, logits, meta

    def test_returns_table_and_maps(self):
        import matplotlib
        matplotlib.use('Agg')
        import pandas as pd
        preds, labels, logits, meta = self._inputs()
        table, figs = build_prediction_outputs(
            preds, labels, logits, meta, self._NAMES,
            fov_lat=[0, 30], fov_lon=[-100, -45])
        assert isinstance(table, pd.DataFrame) and len(table) == 6
        # one map per present TRUE class (skips absent)
        assert set(figs) == {self._NAMES[c] for c in set(labels.tolist())}

    def test_maps_match_table_rows(self):
        # The maps are derived from the table, so a true-class panel's point
        # count equals that class's row count in the table.
        preds, labels, logits, meta = self._inputs()
        table, figs = build_prediction_outputs(
            preds, labels, logits, meta, self._NAMES)
        for c, name in enumerate(self._NAMES):
            if name in figs:
                n_rows = int((table['true'] == c).sum())
                # scatter collection length on the map's axes
                ax = figs[name].axes[0]
                n_pts = sum(len(col.get_offsets()) for col in ax.collections)
                assert n_pts == n_rows

    def test_make_maps_false_skips_figs(self):
        preds, labels, logits, meta = self._inputs()
        table, figs = build_prediction_outputs(
            preds, labels, logits, meta, self._NAMES, make_maps=False)
        assert figs == {}
        assert len(table) == 6

    def test_no_positions_no_maps(self):
        preds, labels, logits, _ = self._inputs()
        table, figs = build_prediction_outputs(
            preds, labels, logits, {'sid': ['A'] * 6}, self._NAMES)
        assert figs == {}
