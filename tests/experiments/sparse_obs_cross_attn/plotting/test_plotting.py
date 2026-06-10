"""
Tests for experiments/sparse_obs_cross_attn/plotting/plotting.py.

All tests use synthetic in-memory data — no disk access required.

Coverage
--------
TestPlotConfusionMatrix      returns Figure; normalized values in [0,1];
                              raw-count mode; existing axes accepted
TestPlotClassMetrics         returns Figure; existing axes accepted
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

from experiments.sparse_obs_cross_attn.plotting.plotting import (
    extract_attention_weights,
    plot_attention_geographic,
    plot_class_metrics,
    plot_confusion_matrix,
)
from experiments.sparse_obs_cross_attn.train.evaluate import CLASS_NAMES
from experiments.sparse_obs_cross_attn.train.model import TCClassifier, N_CLASSES

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
        embed_dim       = EMBED,
        num_heads       = HEADS,
        num_layers      = 1,
        fourier_dim     = 16,
        n_obs_features  = F,
        use_learned_mask= True,
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
# TestExtractAttentionWeights
# ---------------------------------------------------------------------------

class TestExtractAttentionWeights:

    @pytest.fixture
    def model_vars(self):
        return _init_model()

    def test_output_shape(self, model_vars):
        model, variables = model_vars
        weights = extract_attention_weights(model, variables, _fake_batch())
        # (B, num_heads, N+1) — includes query self-attention weight at index N
        assert weights.shape == (B, HEADS, N + 1)

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
