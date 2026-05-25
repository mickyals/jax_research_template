import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch

from utils.plotting.plot3d import (
    plot_volume_slice,
    plot_volume_comparison,
    plot_surface_3d,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    return rng.standard_normal((32, 32, 16)).astype(np.float32)


@pytest.fixture
def positive_volume():
    rng = np.random.default_rng(1)
    return np.abs(rng.standard_normal((32, 32, 16))).astype(np.float32)


@pytest.fixture
def volume_pair(volume):
    rng = np.random.default_rng(2)
    pred = volume + rng.standard_normal(volume.shape).astype(np.float32) * 0.1
    return volume, pred


@pytest.fixture
def surface():
    rng = np.random.default_rng(3)
    return rng.standard_normal((50, 50)).astype(np.float32)


# ---------------------------------------------------------------------------
# plot_volume_slice
# ---------------------------------------------------------------------------

class TestPlotVolumeSlice:

    @patch("matplotlib.pyplot.show")
    def test_runs_default_axis(self, mock_show, volume):
        plot_volume_slice(volume, slice_index=8)

    @patch("matplotlib.pyplot.show")
    def test_runs_axis_0(self, mock_show, volume):
        plot_volume_slice(volume, slice_index=16, axis=0)

    @patch("matplotlib.pyplot.show")
    def test_runs_axis_1(self, mock_show, volume):
        plot_volume_slice(volume, slice_index=16, axis=1)

    @patch("matplotlib.pyplot.show")
    def test_returns_correct_shape_axis2(self, mock_show, volume):
        slc = plot_volume_slice(volume, slice_index=8, axis=2)
        assert slc.shape == (32, 32)

    @patch("matplotlib.pyplot.show")
    def test_returns_correct_shape_axis0(self, mock_show, volume):
        slc = plot_volume_slice(volume, slice_index=16, axis=0)
        assert slc.shape == (32, 16)

    @patch("matplotlib.pyplot.show")
    def test_returns_correct_shape_axis1(self, mock_show, volume):
        slc = plot_volume_slice(volume, slice_index=16, axis=1)
        assert slc.shape == (32, 16)

    @patch("matplotlib.pyplot.show")
    def test_returned_slice_values_correct(self, mock_show, volume):
        slc = plot_volume_slice(volume, slice_index=5, axis=2)
        assert np.allclose(slc, volume[:, :, 5])

    @patch("matplotlib.pyplot.show")
    def test_default_title_generated(self, mock_show, volume):
        plot_volume_slice(volume, slice_index=8)
        fig = plt.gcf()
        ax = fig.axes[0]
        assert "z" in ax.get_title()
        assert "8" in ax.get_title()
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_custom_title(self, mock_show, volume):
        plot_volume_slice(volume, slice_index=8, title="my slice")
        fig = plt.gcf()
        assert fig.axes[0].get_title() == "my slice"
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_asymmetric_cmap(self, mock_show, positive_volume):
        slc = plot_volume_slice(positive_volume, slice_index=8,
                                 symmetric_cmap=False)
        assert slc is not None

    @patch("matplotlib.pyplot.show")
    def test_explicit_vmin_vmax(self, mock_show, volume):
        plot_volume_slice(volume, slice_index=8, vmin=-1., vmax=1.)

    @patch("matplotlib.pyplot.show")
    def test_accepts_jax_array(self, mock_show):
        try:
            import jax.numpy as jnp
            vol = jnp.ones((16, 16, 8))
            slc = plot_volume_slice(vol, slice_index=4)
            assert isinstance(slc, np.ndarray)
        except ImportError:
            pytest.skip("JAX not available")

    @patch("matplotlib.pyplot.show")
    def test_with_extent(self, mock_show, volume):
        plot_volume_slice(volume, slice_index=8,
                           extent=[-1., 1., -1., 1.])


# ---------------------------------------------------------------------------
# plot_volume_comparison
# ---------------------------------------------------------------------------

class TestPlotVolumeComparison:

    @patch("matplotlib.pyplot.show")
    def test_runs(self, mock_show, volume_pair):
        true, pred = volume_pair
        plot_volume_comparison(true, pred, slice_index=8, verbose=False)

    @patch("matplotlib.pyplot.show")
    def test_returns_residual_and_mse(self, mock_show, volume_pair):
        true, pred = volume_pair
        resid, mse = plot_volume_comparison(true, pred, slice_index=8,
                                             verbose=False)
        assert isinstance(resid, np.ndarray)
        assert isinstance(mse, float)

    @patch("matplotlib.pyplot.show")
    def test_residual_shape_axis2(self, mock_show, volume_pair):
        true, pred = volume_pair
        resid, _ = plot_volume_comparison(true, pred, slice_index=8,
                                           axis=2, verbose=False)
        assert resid.shape == (32, 32)

    @patch("matplotlib.pyplot.show")
    def test_residual_shape_axis0(self, mock_show, volume_pair):
        true, pred = volume_pair
        resid, _ = plot_volume_comparison(true, pred, slice_index=16,
                                           axis=0, verbose=False)
        assert resid.shape == (32, 16)

    @patch("matplotlib.pyplot.show")
    def test_residual_values_correct(self, mock_show, volume_pair):
        true, pred = volume_pair
        resid, _ = plot_volume_comparison(true, pred, slice_index=8,
                                           verbose=False)
        expected = pred[:, :, 8] - true[:, :, 8]
        assert np.allclose(resid, expected)

    @patch("matplotlib.pyplot.show")
    def test_mse_correct(self, mock_show, volume_pair):
        true, pred = volume_pair
        resid, mse = plot_volume_comparison(true, pred, slice_index=8,
                                             verbose=False)
        expected_mse = float((resid ** 2).mean())
        assert abs(mse - expected_mse) < 1e-6

    @patch("matplotlib.pyplot.show")
    def test_zero_residual_for_identical(self, mock_show, volume):
        resid, mse = plot_volume_comparison(volume, volume, slice_index=8,
                                             verbose=False)
        assert np.allclose(resid, 0.)
        assert abs(mse) < 1e-10

    @patch("matplotlib.pyplot.show")
    def test_verbose_false_no_print(self, mock_show, volume_pair, capsys):
        true, pred = volume_pair
        plot_volume_comparison(true, pred, slice_index=8, verbose=False)
        assert capsys.readouterr().out == ""

    @patch("matplotlib.pyplot.show")
    def test_verbose_true_prints_mse(self, mock_show, volume_pair, capsys):
        true, pred = volume_pair
        plot_volume_comparison(true, pred, slice_index=8, verbose=True)
        assert "MSE" in capsys.readouterr().out

    @patch("matplotlib.pyplot.show")
    def test_title_contains_axis_and_index(self, mock_show, volume_pair):
        true, pred = volume_pair
        plot_volume_comparison(true, pred, slice_index=8,
                                axis=2, verbose=False)
        fig = plt.gcf()
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert any("z" in t and "8" in t for t in titles)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_produces_three_image_axes(self, mock_show, volume_pair):
        true, pred = volume_pair
        plot_volume_comparison(true, pred, slice_index=8, verbose=False)
        fig = plt.gcf()
        assert len(fig.axes) == 6  # 3 image axes + 3 colorbars
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_accepts_jax_arrays(self, mock_show):
        try:
            import jax.numpy as jnp
            true = jnp.ones((16, 16, 8))
            pred = jnp.ones((16, 16, 8)) * 1.1
            resid, mse = plot_volume_comparison(true, pred, slice_index=4,
                                                 verbose=False)
            assert isinstance(resid, np.ndarray)
        except ImportError:
            pytest.skip("JAX not available")


# ---------------------------------------------------------------------------
# plot_surface_3d
# ---------------------------------------------------------------------------

class TestPlotSurface3d:

    @patch("matplotlib.pyplot.show")
    def test_runs(self, mock_show, surface):
        plot_surface_3d(surface)

    @patch("matplotlib.pyplot.show")
    def test_with_explicit_coords(self, mock_show, surface):
        x = np.linspace(-1., 1., 50)
        y = np.linspace(-1., 1., 50)
        plot_surface_3d(surface, x=x, y=y)

    @patch("matplotlib.pyplot.show")
    def test_with_stride(self, mock_show, surface):
        plot_surface_3d(surface, stride=2)

    @patch("matplotlib.pyplot.show")
    def test_stride_1_and_stride_5_both_run(self, mock_show, surface):
        plot_surface_3d(surface, stride=1)
        plot_surface_3d(surface, stride=5)

    @patch("matplotlib.pyplot.show")
    def test_symmetric_cmap(self, mock_show, surface):
        plot_surface_3d(surface, symmetric_cmap=True)

    @patch("matplotlib.pyplot.show")
    def test_explicit_vmin_vmax(self, mock_show, surface):
        plot_surface_3d(surface, vmin=-1., vmax=1.)

    @patch("matplotlib.pyplot.show")
    def test_custom_labels(self, mock_show, surface):
        plot_surface_3d(surface, title="Loss landscape",
                         xlabel="dir 1", ylabel="dir 2", zlabel="loss")
        fig = plt.gcf()
        ax = fig.axes[0]
        assert ax.get_title() == "Loss landscape"
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_alpha(self, mock_show, surface):
        plot_surface_3d(surface, alpha=0.5)

    @patch("matplotlib.pyplot.show")
    def test_accepts_jax_array(self, mock_show):
        try:
            import jax.numpy as jnp
            z = jnp.ones((20, 20))
            plot_surface_3d(z)
        except ImportError:
            pytest.skip("JAX not available")

    @patch("matplotlib.pyplot.show")
    def test_returns_none(self, mock_show, surface):
        assert plot_surface_3d(surface) is None

    @patch("matplotlib.pyplot.show")
    def test_colour_limits_from_full_array_not_strided(self, mock_show):
        # extreme value at position that would be skipped by stride=2
        z = np.zeros((10, 10), dtype=np.float32)
        z[1, 1] = 100.
        plot_surface_3d(z, stride=2, vmax=None)