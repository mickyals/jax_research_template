"""
utils/plotting/_geo.py

Private cartopy canvas factory for geo-capable renderers -- not part of
the public API. Use the ``geo=`` argument on plotting functions
(e.g. ``fields.plot_scatter_overlay``) instead of importing this module.

cartopy is an optional dependency and is imported lazily inside this
module only -- nothing else in ``utils.plotting`` may import it. If
cartopy is missing, ``_import_cartopy`` raises a clear ImportError with
the install command.

Scope: PlateCarree only. Widening to other projections (e.g.
``geo="mercator"``) can happen backward-compatibly later.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt


def _import_cartopy():
    """Import cartopy lazily, with a clear error when it is not installed.

    Returns
    -------
    tuple
        ``(ccrs, cfeature)`` -- the ``cartopy.crs`` and
        ``cartopy.feature`` modules.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as e:
        raise ImportError(
            "geo plotting requires cartopy, which is not installed. "
            "Install it with: pip install cartopy"
        ) from e
    return ccrs, cfeature


def _add_map_features(
    ax,
    scale: str = "50m",
    color: str = "black",
    lw: float = 0.5,
):
    """Add coastline, country-border, and state/province linework to ``ax``.

    Linework only -- no land/ocean fill, so data colours stay readable.
    Natural Earth shapefiles download on first *render* (draw/savefig),
    not when the feature is added.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Target axes.
    scale : str
        Natural Earth resolution: '110m', '50m', or '10m'. Default '50m'
        -- sufficient for regional maps spanning tens of degrees, with a
        much smaller first-use download than '10m'.
    color : str
        Line colour for all features.
    lw : float
        Line width for all features.

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes
    """
    _, cfeature = _import_cartopy()
    for feature in (cfeature.COASTLINE, cfeature.BORDERS, cfeature.STATES):
        ax.add_feature(
            feature.with_scale(scale),
            edgecolor=color, facecolor="none", linewidth=lw,
        )
    return ax


def _make_geoaxes(
    figsize: tuple[int, int] = (10, 5),
    extent: Optional[list[float]] = None,
    scale: str = "50m",
    color: str = "black",
    lw: float = 0.5,
    gridlines: bool = True,
):
    """Create a figure with a PlateCarree GeoAxes, map features, and gridlines.

    The canvas counterpart of ``plt.subplots`` for geo-capable renderers.
    Labeled gridlines replace the plain-axes ``grid``/``xlabel``/``ylabel``
    path -- the labels already read as degrees ("20°N", "80°W").

    Parameters
    ----------
    figsize : tuple[int, int]
        Figure size in inches.
    extent : list[float], optional
        [lon_min, lon_max, lat_min, lat_max] in degrees. If None, the map
        extent follows the data.
    scale : str
        Natural Earth feature resolution ('110m', '50m', '10m').
    color : str
        Feature line colour.
    lw : float
        Feature line width.
    gridlines : bool
        If True (default), draw labeled dashed gridlines (top/right
        labels off).

    Returns
    -------
    tuple
        ``(fig, ax, transform)`` -- ``transform`` is the data CRS
        (PlateCarree) to pass to every artist call on ``ax``, so callers
        never import cartopy themselves.
    """
    ccrs, _ = _import_cartopy()
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    if extent is not None:
        ax.set_extent(extent, crs=proj)
    _add_map_features(ax, scale=scale, color=color, lw=lw)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
    return fig, ax, proj
