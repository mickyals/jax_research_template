import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch

from utils.plotting._style import (
    _symmetric_clim,
    _resolve_clim,
    _imshow_with_colorbar,
    _comparison_stats,
    _contrast_color,
    _value_scatter,
)


# ---------------------------------------------------------------------------
# _symmetric_clim
# ---------------------------------------------------------------------------

class TestSymmetricClim:

    def test_symmetric_around_zero(self):
        data = np.array([-3., 1., 2.])
        lo, hi = _symmetric_clim(data)
        assert lo == -3.
        assert hi == 3.

    def test_all_positive(self):
        data = np.array([1., 2., 4.])
        lo, hi = _symmetric_clim(data)
        assert lo == -4.
        assert hi == 4.

    def test_all_negative(self):
        data = np.array([-5., -2., -1.])
        lo, hi = _symmetric_clim(data)
        assert lo == -5.
        assert hi == 5.

    def test_zeros(self):
        data = np.zeros((4, 4))
        lo, hi = _symmetric_clim(data)
        assert lo == 0.
        assert hi == 0.


# ---------------------------------------------------------------------------
# _resolve_clim
# ---------------------------------------------------------------------------

class TestResolveClim:

    def test_symmetric_mode(self):
        data = np.array([-2., 1.])
        lo, hi = _resolve_clim(data, symmetric=True, vmin=None, vmax=None)
        assert lo == -2.
        assert hi == 2.

    def test_asymmetric_mode(self):
        data = np.array([1., 3., 5.])
        lo, hi = _resolve_clim(data, symmetric=False, vmin=None, vmax=None)
        assert lo == 1.
        assert hi == 5.

    def test_explicit_overrides_symmetric(self):
        data = np.array([-10., 10.])
        lo, hi = _resolve_clim(data, symmetric=True, vmin=-1., vmax=1.)
        assert lo == -1.
        assert hi == 1.

    def test_partial_override_vmax_only(self):
        data = np.array([-2., 2.])
        lo, hi = _resolve_clim(data, symmetric=True, vmin=None, vmax=5.)
        assert lo == -2.
        assert hi == 5.

    def test_partial_override_vmin_only(self):
        data = np.array([-2., 2.])
        lo, hi = _resolve_clim(data, symmetric=True, vmin=-0.5, vmax=None)
        assert lo == -0.5
        assert hi == 2.


# ---------------------------------------------------------------------------
# _imshow_with_colorbar
# ---------------------------------------------------------------------------

class TestImshowWithColorbar:

    @patch("matplotlib.pyplot.show")
    def test_returns_image_artist(self, mock_show):
        fig, ax = plt.subplots()
        data = np.random.default_rng(0).standard_normal((8, 8))
        im = _imshow_with_colorbar(ax, fig, data)
        assert im is not None
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_adds_colorbar_axis(self, mock_show):
        fig, ax = plt.subplots()
        data = np.random.default_rng(0).standard_normal((8, 8))
        _imshow_with_colorbar(ax, fig, data)
        assert len(fig.axes) == 2  # image axis + colorbar
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_respects_vmin_vmax(self, mock_show):
        fig, ax = plt.subplots()
        data = np.random.default_rng(0).standard_normal((8, 8))
        im = _imshow_with_colorbar(ax, fig, data, vmin=-1.0, vmax=1.0)
        assert im.get_clim() == (-1.0, 1.0)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_origin_upper(self, mock_show):
        fig, ax = plt.subplots()
        data = np.random.default_rng(0).standard_normal((8, 8))
        im = _imshow_with_colorbar(ax, fig, data, origin="upper")
        assert im.origin == "upper"
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_colorbar_label(self, mock_show):
        fig, ax = plt.subplots()
        data = np.random.default_rng(0).standard_normal((8, 8))
        _imshow_with_colorbar(ax, fig, data, colorbar_label="value")
        cbar_ax = fig.axes[1]
        assert cbar_ax.get_ylabel() == "value"
        plt.close(fig)


# ---------------------------------------------------------------------------
# _comparison_stats
# ---------------------------------------------------------------------------

class TestComparisonStats:

    def test_residual_correct(self):
        true = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = np.array([[1.5, 1.5], [3.0, 5.0]])
        resid, vmax, rmax, mse = _comparison_stats(true, pred)
        np.testing.assert_allclose(resid, pred - true)

    def test_mse_correct(self):
        true = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = np.array([[1.5, 1.5], [3.0, 5.0]])
        _, _, _, mse = _comparison_stats(true, pred)
        expected = float(((pred - true) ** 2).mean())
        assert mse == pytest.approx(expected)

    def test_vmax_is_max_abs_of_both(self):
        true = np.array([-5.0, 1.0])
        pred = np.array([2.0, 3.0])
        _, vmax, _, _ = _comparison_stats(true, pred)
        assert vmax == pytest.approx(5.0)

    def test_rmax_is_max_abs_residual_plus_eps(self):
        true = np.array([0.0, 0.0])
        pred = np.array([1.0, -2.0])
        _, _, rmax, _ = _comparison_stats(true, pred)
        assert rmax == pytest.approx(2.0, rel=1e-6)

    def test_zero_residual_for_identical_inputs(self):
        data = np.array([1.0, 2.0, 3.0])
        resid, _, _, mse = _comparison_stats(data, data)
        np.testing.assert_allclose(resid, 0.0)
        assert mse == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _contrast_color
# ---------------------------------------------------------------------------

class TestContrastColor:

    def test_low_value_is_black(self):
        assert _contrast_color(0.0, 0.0, 1.0) == "black"

    def test_high_value_is_white(self):
        assert _contrast_color(1.0, 0.0, 1.0) == "white"

    def test_midpoint_is_black(self):
        assert _contrast_color(0.5, 0.0, 1.0) == "black"

    def test_just_above_midpoint_is_white(self):
        assert _contrast_color(0.51, 0.0, 1.0) == "white"

    def test_zero_span_is_black(self):
        assert _contrast_color(5.0, 5.0, 5.0) == "black"


# ---------------------------------------------------------------------------
# _value_scatter
# ---------------------------------------------------------------------------

class TestValueScatter:

    @patch("matplotlib.pyplot.show")
    def test_no_values_returns_none(self, mock_show):
        fig, ax = plt.subplots()
        x = np.array([0., 1., 2.])
        y = np.array([0., 1., 2.])
        result = _value_scatter(ax, x, y, values=None)
        assert result is None
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_no_values_plots_black_points(self, mock_show):
        fig, ax = plt.subplots()
        x = np.array([0., 1., 2.])
        y = np.array([0., 1., 2.])
        _value_scatter(ax, x, y, values=None)
        collection = ax.collections[0]
        np.testing.assert_allclose(collection.get_facecolor()[0][:3], [0., 0., 0.])
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_with_values_returns_collection(self, mock_show):
        fig, ax = plt.subplots()
        x = np.array([0., 1., 2.])
        y = np.array([0., 1., 2.])
        values = np.array([1., 2., 3.])
        sc = _value_scatter(ax, x, y, values=values)
        assert sc is not None
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_size_range_scales_sizes(self, mock_show):
        fig, ax = plt.subplots()
        x = np.array([0., 1., 2.])
        y = np.array([0., 1., 2.])
        values = np.array([0., 0.5, 1.])
        sc = _value_scatter(ax, x, y, values=values, size_range=(10., 110.))
        sizes = sc.get_sizes()
        assert sizes[0] == pytest.approx(10.)
        assert sizes[2] == pytest.approx(110.)
        assert sizes[1] == pytest.approx(60.)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_constant_values_with_size_range_uses_lo(self, mock_show):
        fig, ax = plt.subplots()
        x = np.array([0., 1., 2.])
        y = np.array([0., 1., 2.])
        values = np.array([5., 5., 5.])
        sc = _value_scatter(ax, x, y, values=values, size_range=(10., 110.))
        sizes = sc.get_sizes()
        assert np.allclose(sizes, 10.)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_fixed_size_without_size_range(self, mock_show):
        fig, ax = plt.subplots()
        x = np.array([0., 1., 2.])
        y = np.array([0., 1., 2.])
        values = np.array([1., 2., 3.])
        sc = _value_scatter(ax, x, y, values=values, size=42.)
        sizes = sc.get_sizes()
        assert np.allclose(sizes, 42.)
        plt.close(fig)
