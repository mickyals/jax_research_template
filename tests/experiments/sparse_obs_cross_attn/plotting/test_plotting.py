"""
Tests for experiments/sparse_obs_cross_attn/plotting/plotting.py.

All tests use synthetic in-memory data — no disk access required.

Coverage
--------
TestPlotConfusionMatrix      returns Figure; normalized values in [0,1];
                              raw-count mode
TestPlotClassMetrics         returns Figure
TestExtractAttentionWeights  shape (L, B, H, N+1, N+1); finite; non-negative;
                              rows sum to one
TestPlotAttentionGeographic  unit_circle returns Figure; domain returns Figure;
                              domain geo=True draws on a PlateCarree map
                              (skipped without cartopy); domain raises without fov
TestPlotAttentionMatrixGrid  layers×heads panel count; no per-token tick labels
TestPlotAttentionMask        Figure; rendered pattern matches
                              model.build_attention_mask exactly
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
    plot_attention_mask,
    plot_attention_matrix_grid,
    plot_class_metrics,
    plot_confusion_matrix,
)
from experiments.sparse_obs_cross_attn.data.sources.ibtracs import CLASS_NAMES
from experiments.sparse_obs_cross_attn.train.evaluate import domain_latlon_for_sample
from experiments.sparse_obs_cross_attn.train.model import TCClassifier, N_CLASSES

_FOV_LAT = (0.0, 30.0)
_FOV_LON = (-100.0, -45.0)

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
        # Full matrices from every layer (_init_model uses num_layers=1)
        assert weights.shape == (1, B, HEADS, N + 1, N + 1)

    def test_values_are_finite(self, model_vars):
        model, variables = model_vars
        weights = extract_attention_weights(model, variables, _fake_batch())
        assert np.all(np.isfinite(weights))

    def test_values_are_nonnegative(self, model_vars):
        model, variables = model_vars
        weights = extract_attention_weights(model, variables, _fake_batch())
        assert np.all(weights >= 0.0)

    def test_rows_sum_to_one(self, model_vars):
        model, variables = model_vars
        weights = extract_attention_weights(model, variables, _fake_batch())
        assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# TestPlotAttentionGeographic
# ---------------------------------------------------------------------------

class TestPlotAttentionGeographic:

    @pytest.fixture
    def weights_and_batch(self):
        model, variables = _init_model()
        batch   = _fake_batch()
        all_w   = extract_attention_weights(model, variables, batch)
        # plot_attention_geographic takes the query row of one layer
        weights = all_w[-1][:, :, 0, :]   # (B, H, N+1)
        return weights, batch

    def test_unit_circle_returns_figure(self, weights_and_batch):
        weights, batch = weights_and_batch
        fig = plot_attention_geographic(
            weights, batch, location_encoding='unit_circle', radius_km=500.0,
        )
        assert isinstance(fig, plt.Figure)
        # Local x-y map on plain Cartesian axes (d17) — not polar
        ax = fig.axes[0]
        assert ax.name == 'rectilinear'
        assert ax.get_aspect() == 1.0
        plt.close('all')

    def test_domain_encoding_returns_figure(self):
        model, variables = _init_model()
        batch   = _fake_batch(location_encoding='domain')
        weights = extract_attention_weights(model, variables, batch)[-1][:, :, 0, :]
        # Caller decodes positions (plotting no longer imports data.encoding).
        station_latlon, query_latlon = domain_latlon_for_sample(
            batch, 0, _FOV_LAT, _FOV_LON)
        fig = plot_attention_geographic(
            weights, batch,
            location_encoding='domain',
            fov_lat=_FOV_LAT,
            fov_lon=_FOV_LON,
            station_latlon=station_latlon,
            query_latlon=query_latlon,
        )
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_domain_geo_returns_map_figure(self):
        # No canvas.draw()/savefig here: Natural Earth shapefiles download
        # at render time and this test must stay network-free.
        pytest.importorskip("cartopy")
        import cartopy.crs as ccrs

        model, variables = _init_model()
        batch   = _fake_batch(location_encoding='domain')
        weights = extract_attention_weights(model, variables, batch)[-1][:, :, 0, :]
        station_latlon, query_latlon = domain_latlon_for_sample(
            batch, 0, _FOV_LAT, _FOV_LON)
        fig = plot_attention_geographic(
            weights, batch,
            location_encoding='domain',
            fov_lat=_FOV_LAT,
            fov_lon=_FOV_LON,
            geo=True,
            station_latlon=station_latlon,
            query_latlon=query_latlon,
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(fig.axes[0].projection, ccrs.PlateCarree)
        plt.close('all')

    def test_domain_requires_decoded_latlon(self):
        # Domain mode with fov but without decoded positions must error clearly.
        model, variables = _init_model()
        batch   = _fake_batch(location_encoding='domain')
        weights = extract_attention_weights(model, variables, batch)[-1][:, :, 0, :]
        with pytest.raises(ValueError, match='station_latlon'):
            plot_attention_geographic(
                weights, batch, location_encoding='domain',
                fov_lat=_FOV_LAT, fov_lon=_FOV_LON,
            )
        plt.close('all')

    def test_unit_circle_geo_returns_azimuthal_map(self, weights_and_batch):
        # No canvas.draw()/savefig: Natural Earth downloads at render time.
        pytest.importorskip("cartopy")
        import cartopy.crs as ccrs

        weights, batch = weights_and_batch
        fig = plot_attention_geographic(
            weights, batch, location_encoding='unit_circle',
            radius_km=500.0, geo=True, storm_latlon=(15.0, -75.0),
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(fig.axes[0].projection, ccrs.AzimuthalEquidistant)
        params = fig.axes[0].projection.proj4_params
        assert params['lat_0'] == pytest.approx(15.0)
        assert params['lon_0'] == pytest.approx(-75.0)
        plt.close('all')

    def test_unit_circle_geo_requires_storm_latlon(self, weights_and_batch):
        pytest.importorskip("cartopy")
        weights, batch = weights_and_batch
        with pytest.raises(ValueError, match='storm_latlon'):
            plot_attention_geographic(
                weights, batch, location_encoding='unit_circle', geo=True,
            )
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


# ---------------------------------------------------------------------------
# TestPlotAttentionMatrixGrid
# ---------------------------------------------------------------------------

class TestPlotAttentionMatrixGrid:

    def test_grid_returns_figure_with_layer_x_head_panels(self):
        # 2 layers × 2 heads synthetic weights — no model needed
        L = 2
        rng = np.random.default_rng(0)
        w = rng.uniform(0, 1, (L, B, HEADS, N + 1, N + 1)).astype(np.float32)
        fig = plot_attention_matrix_grid(w, sample_idx=0)
        assert isinstance(fig, plt.Figure)
        # L*H image panels + 1 colorbar axes
        img_axes = [ax for ax in fig.axes if ax.images]
        assert len(img_axes) == L * HEADS
        # No per-token tick labels — plain imshow
        for ax in img_axes:
            assert len(ax.get_xticks()) == 0
            assert len(ax.get_yticks()) == 0
        plt.close('all')

    def test_grid_from_real_extraction(self):
        model, variables = _init_model()
        batch = _fake_batch()
        w = extract_attention_weights(model, variables, batch)
        fig = plot_attention_matrix_grid(w, sample_idx=1)
        assert isinstance(fig, plt.Figure)
        plt.close('all')


# ---------------------------------------------------------------------------
# TestPlotAttentionMask
# ---------------------------------------------------------------------------

class TestPlotAttentionMask:

    def test_returns_figure_no_token_labels(self):
        station_mask = np.array([True] * 5 + [False] * 3)
        fig = plot_attention_mask(station_mask)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_yticks()) == 0
        plt.close('all')

    def test_rendered_mask_matches_model_pattern(self):
        station_mask = np.array([True, True, False, True])
        fig = plot_attention_mask(station_mask)
        img = fig.axes[0].images[0].get_array().data   # (1+N, 1+N)
        N_t = 4
        assert img.shape == (N_t + 1, N_t + 1)
        # CLS-first: token 0 = query, tokens 1..4 = stations (token 3 = padding).
        # stations → query column (col 0) blocked; query self-attention allowed
        assert not img[1:, 0].any()
        assert img[0, 0] == 1.0
        # padding column (token 3) blocked for everyone
        assert not img[:, 3].any()
        plt.close('all')

    def test_full_self_attention_opens_query_column(self):
        station_mask = np.array([True, True, False, True])
        fig = plot_attention_mask(station_mask, full_self_attention=True)
        img = fig.axes[0].images[0].get_array().data
        # CLS-first: real stations (tokens 1,2,4) now attend to the query col (0)
        assert img[1, 0] == 1.0 and img[2, 0] == 1.0 and img[4, 0] == 1.0
        # padding column (token 3) still blocked
        assert not img[:, 3].any()
        plt.close('all')
