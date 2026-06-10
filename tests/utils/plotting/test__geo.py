"""
Tests for utils/plotting/_geo.py and the geo= path of plot_scatter_overlay.

IMPORTANT: never call fig.canvas.draw() or fig.savefig() in this module.
Natural Earth shapefiles download at *render* time, and these tests must
stay network-free -- assert on figure/axes structure only.

All cartopy-dependent tests are gated by ``needs_cartopy``; the
missing-cartopy error test runs in every environment (it simulates the
missing install by poisoning sys.modules).
"""

import sys

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.plotting._geo import _add_map_features, _make_geoaxes
from utils.plotting.fields import plot_scatter_overlay

try:
    import cartopy.crs as ccrs
    from cartopy.mpl.feature_artist import FeatureArtist
    from cartopy.mpl.gridliner import Gridliner
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

needs_cartopy = pytest.mark.skipif(not HAS_CARTOPY, reason="cartopy not installed")

EXTENT = [-100.0, -40.0, 0.0, 30.0]


def _count_feature_artists(ax) -> int:
    return sum(isinstance(c, FeatureArtist) for c in ax.get_children())


# ---------------------------------------------------------------------------
# Missing cartopy (runs everywhere)
# ---------------------------------------------------------------------------

class TestMissingCartopy:

    @pytest.fixture
    def no_cartopy(self, monkeypatch):
        for mod in ("cartopy", "cartopy.crs", "cartopy.feature"):
            monkeypatch.setitem(sys.modules, mod, None)

    def test_make_geoaxes_clear_error(self, no_cartopy):
        with pytest.raises(ImportError, match="pip install cartopy"):
            _make_geoaxes()

    def test_plot_scatter_overlay_clear_error(self, no_cartopy):
        x = np.linspace(-90.0, -50.0, 5)
        y = np.linspace(5.0, 25.0, 5)
        with pytest.raises(ImportError, match="pip install cartopy"):
            plot_scatter_overlay(None, x, y, geo=True)


# ---------------------------------------------------------------------------
# _make_geoaxes / _add_map_features
# ---------------------------------------------------------------------------

@needs_cartopy
class TestMakeGeoaxes:

    def test_returns_fig_geoaxes_transform(self):
        fig, ax, transform = _make_geoaxes()
        assert isinstance(fig, Figure)
        assert isinstance(ax.projection, ccrs.PlateCarree)
        assert isinstance(transform, ccrs.PlateCarree)
        plt.close(fig)

    def test_extent_applied(self):
        fig, ax, transform = _make_geoaxes(extent=EXTENT)
        assert np.allclose(ax.get_extent(crs=transform), EXTENT)
        plt.close(fig)

    def test_adds_three_features(self):
        fig, ax, _ = _make_geoaxes()
        assert _count_feature_artists(ax) == 3
        plt.close(fig)

    def test_gridlines_on_by_default(self):
        fig, ax, _ = _make_geoaxes()
        assert sum(isinstance(c, Gridliner) for c in ax.get_children()) == 1
        plt.close(fig)

    def test_gridlines_off(self):
        fig, ax, _ = _make_geoaxes(gridlines=False)
        assert not any(isinstance(c, Gridliner) for c in ax.get_children())
        plt.close(fig)

    def test_figsize(self):
        fig, _, _ = _make_geoaxes(figsize=(9, 7))
        assert np.allclose(fig.get_size_inches(), (9, 7))
        plt.close(fig)

    def test_style_overrides_run(self):
        fig, ax, _ = _make_geoaxes(scale="110m", color="gray", lw=0.2)
        assert _count_feature_artists(ax) == 3
        plt.close(fig)


@needs_cartopy
class TestAddMapFeatures:

    def test_adds_features_and_returns_ax(self):
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        result = _add_map_features(ax)
        assert result is ax
        assert _count_feature_artists(ax) == 3
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_scatter_overlay(geo=...)
# ---------------------------------------------------------------------------

@needs_cartopy
class TestPlotScatterOverlayGeo:

    @pytest.fixture
    def lonlat_points(self):
        rng = np.random.default_rng(0)
        n = 15
        return (
            rng.uniform(EXTENT[0], EXTENT[1], n),
            rng.uniform(EXTENT[2], EXTENT[3], n),
            rng.uniform(0.0, 1.0, n).astype(np.float32),
        )

    def test_geo_true_uses_geoaxes(self, lonlat_points):
        lons, lats, vals = lonlat_points
        fig = plot_scatter_overlay(None, lons, lats, scatter_values=vals,
                                   extent=EXTENT, geo=True)
        assert isinstance(fig, Figure)
        assert isinstance(fig.axes[0].projection, ccrs.PlateCarree)
        plt.close(fig)

    def test_geo_extent_applied(self, lonlat_points):
        lons, lats, vals = lonlat_points
        fig = plot_scatter_overlay(None, lons, lats, scatter_values=vals,
                                   extent=EXTENT, geo=True)
        ax = fig.axes[0]
        assert np.allclose(ax.get_extent(crs=ccrs.PlateCarree()), EXTENT)
        plt.close(fig)

    def test_geo_with_values_adds_colorbar(self, lonlat_points):
        lons, lats, vals = lonlat_points
        fig = plot_scatter_overlay(None, lons, lats, scatter_values=vals,
                                   geo=True, colorbar_label="value")
        assert len(fig.axes) == 2  # map axis + colorbar
        plt.close(fig)

    def test_geo_with_field_background(self, lonlat_points):
        lons, lats, _ = lonlat_points
        field = np.random.default_rng(1).standard_normal((16, 16))
        fig = plot_scatter_overlay(field, lons, lats, extent=EXTENT, geo=True)
        assert isinstance(fig.axes[0].projection, ccrs.PlateCarree)
        plt.close(fig)

    def test_geo_dict_options(self, lonlat_points):
        lons, lats, vals = lonlat_points
        fig = plot_scatter_overlay(None, lons, lats, scatter_values=vals,
                                   extent=EXTENT,
                                   geo={"scale": "110m", "lw": 0.3,
                                        "gridlines": False})
        assert _count_feature_artists(fig.axes[0]) == 3
        plt.close(fig)

    def test_geo_empty_dict_means_defaults(self, lonlat_points):
        lons, lats, vals = lonlat_points
        fig = plot_scatter_overlay(None, lons, lats, scatter_values=vals,
                                   geo={})
        assert isinstance(fig.axes[0].projection, ccrs.PlateCarree)
        plt.close(fig)

    def test_geo_ignores_xlabel_ylabel_grid(self, lonlat_points):
        lons, lats, vals = lonlat_points
        fig = plot_scatter_overlay(None, lons, lats, scatter_values=vals,
                                   xlabel="Longitude", ylabel="Latitude",
                                   grid=True, geo=True)
        ax = fig.axes[0]
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        plt.close(fig)

    def test_geo_marker_and_legend(self, lonlat_points):
        lons, lats, vals = lonlat_points
        fig = plot_scatter_overlay(None, lons, lats, scatter_values=vals,
                                   extent=EXTENT, geo=True,
                                   marker_x=-70.0, marker_y=15.0,
                                   marker_label="Query")
        assert fig.axes[0].get_legend() is not None
        plt.close(fig)

    def test_geo_unknown_option_raises(self, lonlat_points):
        lons, lats, vals = lonlat_points
        with pytest.raises(TypeError):
            plot_scatter_overlay(None, lons, lats, scatter_values=vals,
                                 geo={"projection": "mercator"})
        plt.close("all")

    def test_geo_false_plain_axes(self, lonlat_points):
        lons, lats, vals = lonlat_points
        fig = plot_scatter_overlay(None, lons, lats, scatter_values=vals,
                                   extent=EXTENT, geo=False)
        assert not hasattr(fig.axes[0], "projection")
        plt.close(fig)
