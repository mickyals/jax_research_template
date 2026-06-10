import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from unittest.mock import patch

from utils.plotting.fields import (
    plot_field_2d,
    plot_field_comparison_2d,
    plot_scatter_overlay,
    plot_heatmap,
    plot_mollweide,
    plot_mollweide_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def field_2d():
    rng = np.random.default_rng(0)
    return rng.standard_normal((32, 32)).astype(np.float32)


@pytest.fixture
def positive_field():
    rng = np.random.default_rng(1)
    return np.abs(rng.standard_normal((32, 32))).astype(np.float32)


@pytest.fixture
def field_pair(field_2d):
    rng = np.random.default_rng(2)
    pred = field_2d + rng.standard_normal((32, 32)).astype(np.float32) * 0.1
    return field_2d, pred


@pytest.fixture
def mollweide_grids():
    res = 30
    lons = np.linspace(-np.pi, np.pi, res)
    lats = np.linspace(-np.pi / 2, np.pi / 2, res)
    LON, LAT = np.meshgrid(lons, lats)
    field = np.random.default_rng(3).standard_normal((res, res)).astype(np.float32)
    return field, LON, LAT


@pytest.fixture
def scatter_pts():
    rng = np.random.default_rng(4)
    n = 20
    return (
        rng.uniform(-np.pi, np.pi, n),
        rng.uniform(-np.pi / 2, np.pi / 2, n),
        rng.standard_normal(n).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# plot_field_2d
# ---------------------------------------------------------------------------

class TestPlotField2d:

    @patch("matplotlib.pyplot.show")
    def test_runs(self, mock_show, field_2d):
        plot_field_2d(field_2d)

    @patch("matplotlib.pyplot.show")
    def test_with_extent(self, mock_show, field_2d):
        plot_field_2d(field_2d, extent=[-180., 180., -90., 90.])

    @patch("matplotlib.pyplot.show")
    def test_asymmetric_positive_field(self, mock_show, positive_field):
        plot_field_2d(positive_field, symmetric_cmap=False)

    @patch("matplotlib.pyplot.show")
    def test_explicit_vmin_vmax(self, mock_show, field_2d):
        plot_field_2d(field_2d, vmin=-1., vmax=1.)

    @patch("matplotlib.pyplot.show")
    def test_custom_cmap(self, mock_show, field_2d):
        plot_field_2d(field_2d, cmap="viridis")

    @patch("matplotlib.pyplot.show")
    def test_returns_figure(self, mock_show, field_2d):
        result = plot_field_2d(field_2d)
        assert isinstance(result, Figure)
        plt.close(result)


# ---------------------------------------------------------------------------
# plot_field_comparison_2d
# ---------------------------------------------------------------------------

class TestPlotFieldComparison2d:

    @patch("matplotlib.pyplot.show")
    def test_runs(self, mock_show, field_pair):
        true, pred = field_pair
        plot_field_comparison_2d(true, pred)

    @patch("matplotlib.pyplot.show")
    def test_returns_residual_and_mse(self, mock_show, field_pair):
        true, pred = field_pair
        fig, resid, mse = plot_field_comparison_2d(true, pred, verbose=False)
        assert isinstance(fig, Figure)
        assert isinstance(resid, np.ndarray)
        assert resid.shape == true.shape
        assert isinstance(mse, float)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_residual_correct(self, mock_show, field_pair):
        true, pred = field_pair
        fig, resid, _ = plot_field_comparison_2d(true, pred, verbose=False)
        assert np.allclose(resid, pred - true)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_mse_correct(self, mock_show, field_pair):
        true, pred = field_pair
        fig, _, mse = plot_field_comparison_2d(true, pred, verbose=False)
        expected = float(((pred - true) ** 2).mean())
        assert abs(mse - expected) < 1e-6
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_verbose_false_no_print(self, mock_show, field_pair, capsys):
        true, pred = field_pair
        plot_field_comparison_2d(true, pred, verbose=False)
        assert capsys.readouterr().out == ""

    @patch("matplotlib.pyplot.show")
    def test_verbose_true_prints_mse(self, mock_show, field_pair, capsys):
        true, pred = field_pair
        plot_field_comparison_2d(true, pred, verbose=True)
        assert "MSE" in capsys.readouterr().out

    @patch("matplotlib.pyplot.show")
    def test_produces_three_axes(self, mock_show, field_pair):
        true, pred = field_pair
        plot_field_comparison_2d(true, pred, verbose=False)
        fig = plt.gcf()
        assert len(fig.axes) == 6  # 3 image axes + 3 colorbars
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_scatter_overlay
# ---------------------------------------------------------------------------

class TestPlotScatterOverlay:

    @patch("matplotlib.pyplot.show")
    def test_runs_no_values(self, mock_show, field_2d):
        x = np.linspace(-1., 1., 10)
        y = np.linspace(-1., 1., 10)
        plot_scatter_overlay(field_2d, x, y)

    @patch("matplotlib.pyplot.show")
    def test_runs_with_values(self, mock_show, field_2d):
        x = np.linspace(-1., 1., 10)
        y = np.linspace(-1., 1., 10)
        v = np.random.default_rng(0).standard_normal(10)
        plot_scatter_overlay(field_2d, x, y, scatter_values=v)

    @patch("matplotlib.pyplot.show")
    def test_independent_scatter_limits(self, mock_show, field_2d):
        x = np.linspace(-1., 1., 10)
        y = np.linspace(-1., 1., 10)
        v = np.abs(np.random.default_rng(0).standard_normal(10))
        plot_scatter_overlay(field_2d, x, y, scatter_values=v,
                              scatter_vmin=0., scatter_vmax=1.)

    @patch("matplotlib.pyplot.show")
    def test_asymmetric_field(self, mock_show, positive_field):
        x = np.linspace(0., 1., 5)
        y = np.linspace(0., 1., 5)
        plot_scatter_overlay(positive_field, x, y, symmetric_cmap=False)

    @patch("matplotlib.pyplot.show")
    def test_returns_figure(self, mock_show, field_2d):
        x = np.zeros(5)
        y = np.zeros(5)
        result = plot_scatter_overlay(field_2d, x, y)
        assert isinstance(result, Figure)
        plt.close(result)

    @patch("matplotlib.pyplot.show")
    def test_no_field_no_values(self, mock_show):
        x = np.linspace(-1., 1., 10)
        y = np.linspace(-1., 1., 10)
        result = plot_scatter_overlay(None, x, y)
        assert isinstance(result, Figure)
        plt.close(result)

    @patch("matplotlib.pyplot.show")
    def test_no_field_with_values_adds_colorbar(self, mock_show):
        x = np.linspace(-1., 1., 10)
        y = np.linspace(-1., 1., 10)
        v = np.random.default_rng(0).standard_normal(10)
        fig = plot_scatter_overlay(None, x, y, scatter_values=v,
                                    colorbar_label="value")
        assert len(fig.axes) == 2  # scatter axis + colorbar
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_no_field_sets_extent_limits(self, mock_show):
        x = np.array([0., 1.])
        y = np.array([0., 1.])
        fig = plot_scatter_overlay(None, x, y, extent=[-100., -40., 0., 30.])
        ax = fig.axes[0]
        assert ax.get_xlim() == (-100., -40.)
        assert ax.get_ylim() == (0., 30.)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_size_range_scales_scatter(self, mock_show):
        x = np.linspace(-1., 1., 5)
        y = np.linspace(-1., 1., 5)
        v = np.linspace(0., 1., 5)
        fig = plot_scatter_overlay(None, x, y, scatter_values=v,
                                    scatter_size_range=(10., 100.))
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_marker_adds_reference_point(self, mock_show):
        x = np.linspace(-1., 1., 5)
        y = np.linspace(-1., 1., 5)
        fig = plot_scatter_overlay(None, x, y,
                                    marker_x=0., marker_y=0.,
                                    marker_label="Reference")
        ax = fig.axes[0]
        assert ax.get_legend() is not None
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_grid_enabled_without_field(self, mock_show):
        x = np.linspace(-1., 1., 5)
        y = np.linspace(-1., 1., 5)
        fig = plot_scatter_overlay(None, x, y, grid=True)
        ax = fig.axes[0]
        assert any(line.get_visible() for line in ax.get_xgridlines())
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------

class TestPlotHeatmap:

    @patch("matplotlib.pyplot.show")
    def test_runs(self, mock_show):
        matrix = np.random.default_rng(0).standard_normal((8, 8))
        plot_heatmap(matrix)

    @patch("matplotlib.pyplot.show")
    def test_with_labels(self, mock_show):
        matrix = np.eye(4)
        labels = ["a", "b", "c", "d"]
        plot_heatmap(matrix, row_labels=labels, col_labels=labels)

    @patch("matplotlib.pyplot.show")
    def test_annotate(self, mock_show):
        matrix = np.eye(3)
        plot_heatmap(matrix, annotate=True)

    @patch("matplotlib.pyplot.show")
    def test_annotate_fontsize(self, mock_show):
        matrix = np.eye(5)
        plot_heatmap(matrix, annotate=True, annotate_fontsize=5)

    @patch("matplotlib.pyplot.show")
    def test_large_matrix_capped_figsize(self, mock_show):
        matrix = np.random.default_rng(0).standard_normal((100, 100))
        plot_heatmap(matrix)
        fig = plt.gcf()
        w, h = fig.get_size_inches()
        assert w <= 14
        assert h <= 14
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_explicit_figsize(self, mock_show):
        matrix = np.eye(4)
        plot_heatmap(matrix, figsize=(6, 6))
        fig = plt.gcf()
        w, h = fig.get_size_inches()
        assert abs(w - 6.) < 0.1
        assert abs(h - 6.) < 0.1
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_explicit_vmin_vmax(self, mock_show):
        matrix = np.eye(4)
        plot_heatmap(matrix, vmin=-1., vmax=1.)

    @patch("matplotlib.pyplot.show")
    def test_returns_figure(self, mock_show):
        result = plot_heatmap(np.eye(3))
        assert isinstance(result, Figure)
        plt.close(result)

    @patch("matplotlib.pyplot.show")
    def test_annotate_uses_contrast_color(self, mock_show):
        matrix = np.array([[0.0, 1.0], [0.5, 0.51]])
        fig = plot_heatmap(matrix, vmin=0.0, vmax=1.0, annotate=True)
        ax = fig.axes[0]
        colors = {t.get_text(): t.get_color() for t in ax.texts}
        assert colors["0.00"] == "black"
        assert colors["1.00"] == "white"
        assert colors["0.50"] == "black"
        assert colors["0.51"] == "white"
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_annotate_contrast_uses_resolved_clim(self, mock_show):
        matrix = np.array([[0.0, 10.0], [6.0, 4.0]])
        fig = plot_heatmap(matrix, annotate=True)
        ax = fig.axes[0]
        colors = {t.get_text(): t.get_color() for t in ax.texts}
        assert colors["0.00"] == "black"
        assert colors["10.00"] == "white"
        assert colors["6.00"] == "white"
        assert colors["4.00"] == "black"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_mollweide
# ---------------------------------------------------------------------------

class TestPlotMollweide:

    @patch("matplotlib.pyplot.show")
    def test_runs(self, mock_show, mollweide_grids):
        field, LON, LAT = mollweide_grids
        plot_mollweide(field, LON, LAT)

    @patch("matplotlib.pyplot.show")
    def test_with_scatter_no_values(self, mock_show, mollweide_grids,
                                     scatter_pts):
        field, LON, LAT = mollweide_grids
        s_lon, s_lat, _ = scatter_pts
        plot_mollweide(field, LON, LAT,
                       scatter_lon=s_lon, scatter_lat=s_lat)

    @patch("matplotlib.pyplot.show")
    def test_with_scatter_values(self, mock_show, mollweide_grids,
                                  scatter_pts):
        field, LON, LAT = mollweide_grids
        s_lon, s_lat, s_val = scatter_pts
        plot_mollweide(field, LON, LAT,
                       scatter_lon=s_lon, scatter_lat=s_lat,
                       scatter_values=s_val)

    @patch("matplotlib.pyplot.show")
    def test_independent_scatter_limits(self, mock_show, mollweide_grids,
                                         scatter_pts):
        field, LON, LAT = mollweide_grids
        s_lon, s_lat, _ = scatter_pts
        probs = np.abs(np.random.default_rng(5).standard_normal(len(s_lon)))
        plot_mollweide(field, LON, LAT,
                       scatter_lon=s_lon, scatter_lat=s_lat,
                       scatter_values=probs,
                       scatter_vmin=0., scatter_vmax=1.,
                       scatter_cmap="viridis")

    @patch("matplotlib.pyplot.show")
    def test_asymmetric_field(self, mock_show, mollweide_grids):
        _, LON, LAT = mollweide_grids
        pos_field = np.abs(mollweide_grids[0])
        plot_mollweide(pos_field, LON, LAT, symmetric_cmap=False)

    @patch("matplotlib.pyplot.show")
    def test_returns_figure(self, mock_show, mollweide_grids):
        field, LON, LAT = mollweide_grids
        result = plot_mollweide(field, LON, LAT)
        assert isinstance(result, Figure)
        plt.close(result)


# ---------------------------------------------------------------------------
# plot_mollweide_comparison
# ---------------------------------------------------------------------------

class TestPlotMollweideComparison:

    @patch("matplotlib.pyplot.show")
    def test_runs(self, mock_show, mollweide_grids):
        field, LON, LAT = mollweide_grids
        pred = field + np.random.default_rng(6).standard_normal(
            field.shape
        ).astype(np.float32) * 0.1
        plot_mollweide_comparison(field, pred, LON, LAT, verbose=False)

    @patch("matplotlib.pyplot.show")
    def test_returns_residual_and_mse(self, mock_show, mollweide_grids):
        field, LON, LAT = mollweide_grids
        pred = field * 0.9
        fig, resid, mse = plot_mollweide_comparison(
            field, pred, LON, LAT, verbose=False
        )
        assert isinstance(fig, Figure)
        assert isinstance(resid, np.ndarray)
        assert resid.shape == field.shape
        assert isinstance(mse, float)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_residual_correct(self, mock_show, mollweide_grids):
        field, LON, LAT = mollweide_grids
        pred = field * 0.9
        fig, resid, _ = plot_mollweide_comparison(
            field, pred, LON, LAT, verbose=False
        )
        assert np.allclose(resid, pred - field)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_verbose_false_no_print(self, mock_show, mollweide_grids, capsys):
        field, LON, LAT = mollweide_grids
        plot_mollweide_comparison(field, field, LON, LAT, verbose=False)
        assert capsys.readouterr().out == ""

    @patch("matplotlib.pyplot.show")
    def test_verbose_true_prints_mse(self, mock_show, mollweide_grids, capsys):
        field, LON, LAT = mollweide_grids
        plot_mollweide_comparison(field, field, LON, LAT, verbose=True)
        assert "MSE" in capsys.readouterr().out

    @patch("matplotlib.pyplot.show")
    def test_with_scatter_no_values(self, mock_show, mollweide_grids,
                                     scatter_pts):
        field, LON, LAT = mollweide_grids
        s_lon, s_lat, _ = scatter_pts
        plot_mollweide_comparison(field, field, LON, LAT,
                                   scatter_lon=s_lon, scatter_lat=s_lat,
                                   verbose=False)

    @patch("matplotlib.pyplot.show")
    def test_with_scatter_independent_limits(self, mock_show,
                                              mollweide_grids, scatter_pts):
        field, LON, LAT = mollweide_grids
        s_lon, s_lat, _ = scatter_pts
        probs = np.random.default_rng(7).uniform(0., 1., len(s_lon))
        plot_mollweide_comparison(
            field, field, LON, LAT,
            scatter_lon=s_lon, scatter_lat=s_lat,
            scatter_values=probs,
            scatter_vmin=0., scatter_vmax=1.,
            scatter_cmap="viridis",
            verbose=False,
        )