"""
Tests for experiments/tc_perceiver_io/plotting/plotting.py.

All tests use synthetic in-memory data — no disk access required.

Coverage
--------
TestPlotConfusionMatrix      returns Figure; normalized + raw-count modes
TestPlotClassMetrics         returns Figure
TestPlotAttentionMatrixGrid  Processor (L,B,H,N,N): layers×heads panel count;
                             no per-latent tick labels
TestPlotDecoderQuery         Decoder (B,H,1,N): heads×latents heatmap
TestPlotAttentionGeographic  Read (B,H,N,M): unit_circle + domain return Figure;
                             domain geo=True draws a PlateCarree map and
                             unit_circle geo=True an AzimuthalEquidistant map
                             (both skipped without cartopy); domain raises
                             without fov / without decoded lat-lon
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')   # headless — no display required
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.tc_perceiver_io.plotting.plotting import (
    plot_attention_geographic,
    plot_attention_matrix_grid,
    plot_decoder_query,
    plot_class_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_pr_curves_per_class,
    plot_per_class_prediction_maps,
    plot_class_coverage_map,
)
from experiments.tc_perceiver_io.train.full_set_metrics import binary_pr_curve, per_class_pr_curves
from experiments.tc_perceiver_io.data.sources.ibtracs import CLASS_NAMES, N_CLASSES
from experiments.tc_perceiver_io.train.evaluate import domain_latlon_for_sample
from experiments.tc_perceiver_io.train.model import TCPerceiverIO

_FOV_LAT = (0.0, 30.0)
_FOV_LON = (-100.0, -45.0)

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

B      = 6     # batch
M      = 8     # stations (padded)
NLAT   = 4     # latents (N)
F      = 5     # obs features
HEADS  = 2
EMBED  = 32
LAYERS = 2


def _fake_batch(location_encoding: str = 'unit_circle') -> dict:
    rng  = np.random.default_rng(1)
    obs  = rng.standard_normal((B, M, F)).astype(np.float32)
    if location_encoding == 'unit_circle':
        query = np.zeros((B, 2), dtype=np.float32)
    else:
        query = rng.uniform(-1.5, 1.5, (B, 2)).astype(np.float32)
    return {
        'X': {
            'station_obs':    jnp.array(obs),
            'station_coords': jnp.array(rng.uniform(-1., 1., (B, M, 2)).astype(np.float32)),
            'station_mask':   jnp.ones((B, M), dtype=bool),
            'obs_mask':       jnp.ones((B, M, F), dtype=bool),
            'query_coords':   jnp.array(query),
        },
        'y': jnp.array(rng.integers(0, N_CLASSES, size=B), dtype=jnp.int32),
    }


def _model_and_attn(
    decode_mode: str = 'attention',
    location_encoding: str = 'unit_circle',
) -> tuple[dict, dict]:
    """Init a tiny TCPerceiverIO and return (softmaxed attn dict, batch).

    The returned dict mirrors the model's pre-softmax attn dict but with each
    component softmaxed over its last axis (exactly what the plotters expect):
        read      (B, H, N, M)
        processor (L, B, H, N, N)
        decoder   (B, H, 1, N)  — None for decode_mode='avgproj'
    """
    model = TCPerceiverIO(
        embed_dim          = EMBED,
        num_heads          = HEADS,
        num_latents        = NLAT,
        num_process_layers = LAYERS,
        fourier_dim        = 16,
        n_obs_features     = F,
        n_classes          = N_CLASSES,
        decode_mode        = decode_mode,
    )
    batch     = _fake_batch(location_encoding)
    variables = model.init({'params': jax.random.PRNGKey(0)}, batch['X'], train=False)
    _, attn   = model.apply(variables, batch['X'], train=False, return_weights=True)

    soft = {
        'read':      np.asarray(jax.nn.softmax(attn['read'],      axis=-1)),
        'processor': np.asarray(jax.nn.softmax(attn['processor'], axis=-1)),
        'decoder':   (np.asarray(jax.nn.softmax(attn['decoder'], axis=-1))
                      if attn.get('decoder') is not None else None),
    }
    return soft, batch


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
# TestPRCurves
# ---------------------------------------------------------------------------

class TestPRCurves:

    def _data(self):
        rng    = np.random.default_rng(0)
        logits = rng.standard_normal((50, N_CLASSES)).astype(np.float32)
        labels = rng.integers(0, N_CLASSES, 50).astype(np.int32)
        return logits, labels

    def test_binary_pr_curve_returns_figure(self):
        logits, labels = self._data()
        fig = plot_pr_curve(binary_pr_curve(logits, labels))
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Recall' and ax.get_ylabel() == 'Precision'
        plt.close('all')

    def test_per_class_pr_curves_returns_figure(self):
        logits, labels = self._data()
        fig = plot_pr_curves_per_class(
            per_class_pr_curves(logits, labels), CLASS_NAMES)
        assert isinstance(fig, plt.Figure)
        plt.close('all')


# ---------------------------------------------------------------------------
# TestAttnDictShapes — the contract the plotters rely on
# ---------------------------------------------------------------------------

class TestAttnDictShapes:

    def test_component_shapes(self):
        attn, _ = _model_and_attn()
        assert attn['read'].shape      == (B, HEADS, NLAT, M)
        assert attn['processor'].shape == (LAYERS, B, HEADS, NLAT, NLAT)
        assert attn['decoder'].shape   == (B, HEADS, 1, NLAT)

    def test_softmax_rows_sum_to_one(self):
        attn, _ = _model_and_attn()
        assert np.allclose(attn['read'].sum(axis=-1),      1.0, atol=1e-5)
        assert np.allclose(attn['processor'].sum(axis=-1), 1.0, atol=1e-5)
        assert np.allclose(attn['decoder'].sum(axis=-1),   1.0, atol=1e-5)

    def test_avgproj_has_no_decoder_weights(self):
        attn, _ = _model_and_attn(decode_mode='avgproj')
        assert attn['decoder'] is None


# ---------------------------------------------------------------------------
# TestPlotAttentionMatrixGrid (Processor)
# ---------------------------------------------------------------------------

class TestPlotAttentionMatrixGrid:

    def test_grid_returns_figure_with_layer_x_head_panels(self):
        rng = np.random.default_rng(0)
        w = rng.uniform(0, 1, (LAYERS, B, HEADS, NLAT, NLAT)).astype(np.float32)
        fig = plot_attention_matrix_grid(w, sample_idx=0)
        assert isinstance(fig, plt.Figure)
        img_axes = [ax for ax in fig.axes if ax.images]
        assert len(img_axes) == LAYERS * HEADS
        # No per-latent tick labels — plain imshow
        for ax in img_axes:
            assert len(ax.get_xticks()) == 0
            assert len(ax.get_yticks()) == 0
        plt.close('all')

    def test_grid_from_real_model(self):
        attn, _ = _model_and_attn()
        fig = plot_attention_matrix_grid(attn['processor'], sample_idx=1)
        assert isinstance(fig, plt.Figure)
        plt.close('all')


# ---------------------------------------------------------------------------
# TestPlotDecoderQuery (Decoder)
# ---------------------------------------------------------------------------

class TestPlotDecoderQuery:

    def test_returns_figure(self):
        attn, _ = _model_and_attn()
        fig = plot_decoder_query(attn['decoder'], sample_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_heatmap_is_heads_by_latents(self):
        attn, _ = _model_and_attn()
        fig = plot_decoder_query(attn['decoder'], sample_idx=0)
        img = fig.axes[0].images[0].get_array().data
        assert img.shape == (HEADS, NLAT)
        plt.close('all')

    def test_accepts_synthetic_weights(self):
        rng = np.random.default_rng(2)
        w = rng.uniform(0, 1, (B, HEADS, 1, NLAT)).astype(np.float32)
        fig = plot_decoder_query(w, sample_idx=3)
        assert isinstance(fig, plt.Figure)
        plt.close('all')


# ---------------------------------------------------------------------------
# TestPlotAttentionGeographic (Read)
# ---------------------------------------------------------------------------

class TestPlotAttentionGeographic:

    def test_unit_circle_returns_figure(self):
        attn, batch = _model_and_attn(location_encoding='unit_circle')
        fig = plot_attention_geographic(
            attn['read'], batch, location_encoding='unit_circle', radius_km=500.0,
        )
        assert isinstance(fig, plt.Figure)
        # Local x-y map on plain Cartesian axes — not polar
        ax = fig.axes[0]
        assert ax.name == 'rectilinear'
        assert ax.get_aspect() == 1.0
        plt.close('all')

    def test_domain_encoding_returns_figure(self):
        attn, batch = _model_and_attn(location_encoding='domain')
        station_latlon, query_latlon = domain_latlon_for_sample(
            batch, 0, _FOV_LAT, _FOV_LON)
        fig = plot_attention_geographic(
            attn['read'], batch, location_encoding='domain',
            fov_lat=_FOV_LAT, fov_lon=_FOV_LON,
            station_latlon=station_latlon, query_latlon=query_latlon,
        )
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_domain_geo_returns_map_figure(self):
        # No canvas.draw()/savefig here: Natural Earth shapefiles download
        # at render time and this test must stay network-free.
        pytest.importorskip("cartopy")
        import cartopy.crs as ccrs

        attn, batch = _model_and_attn(location_encoding='domain')
        station_latlon, query_latlon = domain_latlon_for_sample(
            batch, 0, _FOV_LAT, _FOV_LON)
        fig = plot_attention_geographic(
            attn['read'], batch, location_encoding='domain',
            fov_lat=_FOV_LAT, fov_lon=_FOV_LON, geo=True,
            station_latlon=station_latlon, query_latlon=query_latlon,
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(fig.axes[0].projection, ccrs.PlateCarree)
        plt.close('all')

    def test_domain_requires_decoded_latlon(self):
        attn, batch = _model_and_attn(location_encoding='domain')
        with pytest.raises(ValueError, match='station_latlon'):
            plot_attention_geographic(
                attn['read'], batch, location_encoding='domain',
                fov_lat=_FOV_LAT, fov_lon=_FOV_LON,
            )
        plt.close('all')

    def test_domain_raises_without_fov(self):
        attn, batch = _model_and_attn(location_encoding='domain')
        with pytest.raises(ValueError, match='fov_lat'):
            plot_attention_geographic(
                attn['read'], batch, location_encoding='domain',
            )
        plt.close('all')

    def test_unit_circle_geo_returns_azimuthal_map(self):
        # No canvas.draw()/savefig: Natural Earth downloads at render time.
        pytest.importorskip("cartopy")
        import cartopy.crs as ccrs

        attn, batch = _model_and_attn(location_encoding='unit_circle')
        fig = plot_attention_geographic(
            attn['read'], batch, location_encoding='unit_circle',
            radius_km=500.0, geo=True, storm_latlon=(15.0, -75.0),
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(fig.axes[0].projection, ccrs.AzimuthalEquidistant)
        params = fig.axes[0].projection.proj4_params
        assert params['lat_0'] == pytest.approx(15.0)
        assert params['lon_0'] == pytest.approx(-75.0)
        plt.close('all')

    def test_unit_circle_geo_requires_storm_latlon(self):
        pytest.importorskip("cartopy")
        attn, batch = _model_and_attn(location_encoding='unit_circle')
        with pytest.raises(ValueError, match='storm_latlon'):
            plot_attention_geographic(
                attn['read'], batch, location_encoding='unit_circle', geo=True,
            )
        plt.close('all')

    def test_title_caption_appears_in_figure(self):
        attn, batch = _model_and_attn(location_encoding='unit_circle')
        fig = plot_attention_geographic(
            attn['read'], batch, location_encoding='unit_circle',
            title='AL09 IDA — true: Cat 4, pred: Cat 3')
        assert 'IDA' in fig.axes[0].get_title()
        assert 'Read attention' in fig.axes[0].get_title()
        plt.close('all')

    def test_sample_idx_selects_different_sample(self):
        attn, batch = _model_and_attn(location_encoding='unit_circle')
        fig0 = plot_attention_geographic(
            attn['read'], batch, location_encoding='unit_circle', sample_idx=0)
        fig1 = plot_attention_geographic(
            attn['read'], batch, location_encoding='unit_circle', sample_idx=1)
        assert isinstance(fig0, plt.Figure)
        assert isinstance(fig1, plt.Figure)
        plt.close('all')


# ---------------------------------------------------------------------------
# Spatial classification / coverage maps
# ---------------------------------------------------------------------------

class TestSpatialMaps:

    _NAMES = ['Background', 'Disturbance', 'Depression', 'Storm',
              'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']

    def _data(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        lat = rng.uniform(0, 30, n)
        lon = rng.uniform(-100, -45, n)
        true = rng.integers(0, 9, n)
        pred = rng.integers(0, 9, n)
        return lat, lon, true, pred

    def test_per_class_maps_one_fig_per_present_class(self):
        lat, lon, true, pred = self._data()
        figs = plot_per_class_prediction_maps(
            lat, lon, true, pred, self._NAMES,
            fov_lat=[0, 30], fov_lon=[-100, -45], n_classes=9)
        assert set(figs) == {self._NAMES[c] for c in set(true.tolist())}
        for f in figs.values():
            assert isinstance(f, plt.Figure)

    def test_per_class_skips_absent_classes(self):
        # Only true classes {0, 4} present → exactly two maps.
        lat = np.array([5.0, 6.0, 7.0]); lon = np.array([-60.0, -61.0, -62.0])
        true = np.array([0, 4, 0]);       pred = np.array([0, 0, 4])
        figs = plot_per_class_prediction_maps(
            lat, lon, true, pred, self._NAMES, n_classes=9)
        assert set(figs) == {'Background', 'Cat1'}

    def test_coverage_map_returns_figure(self):
        lat, lon, true, _ = self._data()
        fig = plot_class_coverage_map(
            lat, lon, true, self._NAMES,
            fov_lat=[0, 30], fov_lon=[-100, -45], n_classes=9)
        assert isinstance(fig, plt.Figure)
