"""
utils/plotting/_geo.py

Private cartopy canvas factory for geo-capable renderers -- not part of
the public API. Use the ``geo=`` argument on plotting functions
(e.g. ``fields.plot_scatter_overlay``) instead of importing this module.

cartopy is an optional dependency and is imported lazily inside this
module only -- nothing else in ``utils.plotting`` may import it. If
cartopy is missing, ``_import_cartopy`` raises a clear ImportError with
the installation command.

Projections: PlateCarree (default) and AzimuthalEquidistant
(``projection='azimuthal'`` + ``center``). The azimuthal projection's
native coordinates are metres from the centre along (east, north) —
i.e. (distance·sin(bearing), distance·cos(bearing)) — which makes it the
exact geographic canvas for storm-centred local x-y data: plot in native
metres with the default transform and the coastlines land in the right
place by construction.
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
    projection: str = "platecarree",
    center: Optional[tuple[float, float]] = None,
):
    """Create a figure with a GeoAxes, map features, and gridlines.

    The canvas counterpart of ``plt.subplots`` for geo-capable renderers.
    Labeled gridlines replace the plain-axes ``grid``/``xlabel``/``ylabel``
    path -- the labels already read as degrees ("20°N", "80°W").

    Parameters
    ----------
    figsize : tuple[int, int]
        Figure size in inches.
    extent : list[float], optional
        For 'platecarree': [lon_min, lon_max, lat_min, lat_max] degrees.
        For 'azimuthal': [x_min, x_max, y_min, y_max] METRES from the
        centre (east/north offsets). If None, the extent follows the data.
    scale : str
        Natural Earth feature resolution ('110m', '50m', '10m').
    color : str
        Feature line colour.
    lw : float
        Feature line width.
    gridlines : bool
        If True (default), draw labeled dashed gridlines (top/right
        labels off).
    projection : {'platecarree', 'azimuthal'}
        'azimuthal' = AzimuthalEquidistant centred on ``center``: native
        axes coordinates are metres from the centre along (east, north),
        i.e. (distance·sin(bearing), distance·cos(bearing)) — distance
        and bearing from the centre are preserved exactly. Plot
        storm-centred local x-y data in native metres with the DEFAULT
        matplotlib transform (no ``transform=`` kwarg).
    center : (lat, lon), optional
        Projection centre in degrees. Required for 'azimuthal'.

    Returns
    -------
    tuple
        ``(fig, ax, transform)`` -- ``transform`` is the lon/lat data CRS
        (PlateCarree) to pass to artist calls whose data is in degrees,
        so callers never import cartopy themselves. Data already in the
        projection's native coordinates (azimuthal metres) should be
        plotted WITHOUT a transform kwarg.
    """
    ccrs, _ = _import_cartopy()
    lonlat = ccrs.PlateCarree()

    if projection == "platecarree":
        proj = lonlat
    elif projection == "azimuthal":
        if center is None:
            raise ValueError(
                "_make_geoaxes: projection='azimuthal' requires "
                "center=(lat, lon)."
            )
        proj = ccrs.AzimuthalEquidistant(
            central_latitude=float(center[0]),
            central_longitude=float(center[1]),
        )
    else:
        raise ValueError(
            f"_make_geoaxes: unknown projection '{projection}' "
            f"(expected 'platecarree' or 'azimuthal')."
        )

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    if extent is not None:
        ax.set_extent(extent, crs=proj)
    _add_map_features(ax, scale=scale, color=color, lw=lw)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
    return fig, ax, lonlat
