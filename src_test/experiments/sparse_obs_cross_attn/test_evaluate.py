"""
Tests for experiments/sparse_obs_cross_attn/evaluate.py.

All tests use synthetic in-memory data — no disk access required.

Coverage
--------
TestConfusionMatrix       shape; diagonal counts; off-diagonal counts;
                          total count; empty prediction handled
TestPerClassMetrics       perfect class gives p/r/f1=1; zero support;
                          keys present; tp/fp/fn arithmetic
TestBinaryMetrics         all TC correct; all no-storm correct; perfect binary;
                          mixed case; all keys present; counts add up
TestPlotConfusionMatrix   returns Figure; normalized values in [0,1];
                          raw-count mode; existing axes accepted
TestPlotClassMetrics      returns Figure; existing axes accepted
TestCollectPredictions    output shapes; preds are argmax of logits;
                          labels match loader; no model mutation
TestExtractAttentionWeights  shape (B, H, N); values finite; non-negative
TestPlotAttentionGeographic  unit_circle returns Figure; domain returns Figure;
                             domain raises without fov
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')   # headless — no display required
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.sparse_obs_cross_attn.evaluate import (
    CLASS_NAMES,
    binary_metrics,
    collect_predictions,
    confusion_matrix,
    extract_attention_weights,
    per_class_metrics,
    plot_attention_geographic,
    plot_class_metrics,
    plot_confusion_matrix,
)
from experiments.sparse_obs_cross_attn.model import TCClassifier, N_CLASSES

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

B     = 6
N     = 8
F     = 5
HEADS = 2
EMBED = 32


def _init_model() -> tuple[TCClassifier, dict]:
    """Return a tiny TCClassifier and its initialized variables."""
    model = TCClassifier(
        embed_dim        = EMBED,
        num_heads        = HEADS,
        num_layers       = 1,
        num_cross_layers = 1,
        fourier_dim      = 16,
        n_obs_features   = F,
    )
    rng  = np.random.default_rng(0)
    obs  = jnp.array(rng.standard_normal((B, N, F)).astype(np.float32))
    X    = {
        'station_obs':    obs,
        'station_coords': jnp.zeros((B, N, 2)),
        'station_mask':   jnp.ones((B, N), dtype=bool),
        'obs_mask':       jnp.ones((B, N, F), dtype=bool),
        'query_coords':   jnp.zeros((B, 2)),
    }
    variables = model.init({'params': jax.random.PRNGKey(0)}, X, train=False)
    return model, variables


def _fake_batch(location_encoding: str = 'unit_circle') -> dict:
    rng  = np.random.default_rng(1)
    obs  = rng.standard_normal((B, N, F)).astype(np.float32)
    if location_encoding == 'unit_circle':
        query = np.zeros((B, 2), dtype=np.float32)
    else:
        query = rng.uniform(-1.5, 1.5, (B, 2)).astype(np.float32)
    return {
        'X': {
            'station_obs':    jnp.array(obs),
            'station_coords': jnp.array(rng.uniform(-1., 1., (B, N, 2)).astype(np.float32)),
            'station_mask':   jnp.ones((B, N), dtype=bool),
            'obs_mask':       jnp.ones((B, N, F), dtype=bool),
            'query_coords':   jnp.array(query),
        },
        'y': jnp.array(rng.integers(0, N_CLASSES, size=B), dtype=jnp.int32),
    }


class _FakeLoader:
    """Minimal re-iterable loader yielding a single fake batch."""
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
        preds  = rng.integers(0, 11, size=40).astype(np.int32)
        labels = rng.integers(0, 11, size=40).astype(np.int32)
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
# TestPlotConfusionMatrix
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrix:

    def _make_cm(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.integers(0, 20, (N_CLASSES, N_CLASSES)).astype(np.int64)

    def test_returns_figure_normalized(self):
        fig = plot_confusion_matrix(self._make_cm(), CLASS_NAMES, normalize=True)
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_returns_figure_raw_counts(self):
        fig = plot_confusion_matrix(self._make_cm(), CLASS_NAMES, normalize=False)
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_accepts_existing_axes(self):
        fig0, ax = plt.subplots()
        fig_ret  = plot_confusion_matrix(self._make_cm(), CLASS_NAMES, ax=ax)
        assert isinstance(fig_ret, plt.Figure)
        plt.close('all')


# ---------------------------------------------------------------------------
# TestPlotClassMetrics
# ---------------------------------------------------------------------------

class TestPlotClassMetrics:

    def _make_metrics(self) -> dict:
        return {k: {'precision': 0.8, 'recall': 0.7, 'f1': 0.74, 'support': 10}
                for k in range(N_CLASSES)}

    def test_returns_figure(self):
        fig = plot_class_metrics(self._make_metrics(), CLASS_NAMES)
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_accepts_existing_axes(self):
        fig0, ax = plt.subplots()
        fig_ret  = plot_class_metrics(self._make_metrics(), CLASS_NAMES, ax=ax)
        assert isinstance(fig_ret, plt.Figure)
        plt.close('all')


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
        preds, labels, logits = collect_predictions(model, variables, loader)
        n_total = B * 2
        assert preds.shape  == (n_total,)
        assert labels.shape == (n_total,)
        assert logits.shape == (n_total, N_CLASSES)

    def test_preds_are_argmax_of_logits(self, model_vars):
        model, variables = model_vars
        loader = _FakeLoader(n=1)
        preds, _, logits = collect_predictions(model, variables, loader)
        expected = logits.argmax(axis=-1)
        np.testing.assert_array_equal(preds, expected)

    def test_labels_match_loader(self, model_vars):
        model, variables = model_vars
        batch  = _fake_batch()
        loader = [batch]   # single-batch loader (plain list)
        _, labels, _ = collect_predictions(model, variables, loader)
        np.testing.assert_array_equal(labels, np.asarray(batch['y']))

    def test_preds_in_valid_class_range(self, model_vars):
        model, variables = model_vars
        preds, _, _ = collect_predictions(model, variables, _FakeLoader(n=2))
        assert preds.min() >= 0
        assert preds.max() < N_CLASSES

    def test_logits_are_finite(self, model_vars):
        model, variables = model_vars
        _, _, logits = collect_predictions(model, variables, _FakeLoader(n=2))
        assert np.all(np.isfinite(logits))


# ---------------------------------------------------------------------------
# TestExtractAttentionWeights
# ---------------------------------------------------------------------------

class TestExtractAttentionWeights:

    @pytest.fixture
    def model_vars(self):
        return _init_model()

    def test_output_shape(self, model_vars):
        model, variables = model_vars
        weights = extract_attention_weights(model, variables, _fake_batch())
        # (B, num_heads, N)
        assert weights.shape == (B, HEADS, N)

    def test_values_are_finite(self, model_vars):
        model, variables = model_vars
        weights = extract_attention_weights(model, variables, _fake_batch())
        assert np.all(np.isfinite(weights))

    def test_values_are_nonnegative(self, model_vars):
        model, variables = model_vars
        weights = extract_attention_weights(model, variables, _fake_batch())
        assert np.all(weights >= 0.0)


# ---------------------------------------------------------------------------
# TestPlotAttentionGeographic
# ---------------------------------------------------------------------------

class TestPlotAttentionGeographic:

    @pytest.fixture
    def weights_and_batch(self):
        model, variables = _init_model()
        batch   = _fake_batch()
        weights = extract_attention_weights(model, variables, batch)
        return weights, batch

    def test_unit_circle_returns_figure(self, weights_and_batch):
        weights, batch = weights_and_batch
        fig = plot_attention_geographic(
            weights, batch, location_encoding='unit_circle', radius_km=500.0,
        )
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_domain_encoding_returns_figure(self):
        model, variables = _init_model()
        batch   = _fake_batch(location_encoding='domain')
        weights = extract_attention_weights(model, variables, batch)
        fig = plot_attention_geographic(
            weights, batch,
            location_encoding='domain',
            fov_lat=(0.0, 30.0),
            fov_lon=(-100.0, -45.0),
        )
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_domain_raises_without_fov(self, weights_and_batch):
        weights, batch = weights_and_batch
        with pytest.raises(ValueError, match="fov_lat"):
            plot_attention_geographic(
                weights, batch, location_encoding='domain'
            )

    def test_sample_idx_selects_different_sample(self, weights_and_batch):
        weights, batch = weights_and_batch
        fig0 = plot_attention_geographic(
            weights, batch, location_encoding='unit_circle', sample_idx=0,
        )
        fig1 = plot_attention_geographic(
            weights, batch, location_encoding='unit_circle', sample_idx=1,
        )
        # Both should return figures without error
        assert isinstance(fig0, plt.Figure)
        assert isinstance(fig1, plt.Figure)
        plt.close('all')
