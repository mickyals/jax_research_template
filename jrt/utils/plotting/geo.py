"""
utils/plotting/geo.py

Public cartopy canvas helpers for geo-capable figures. Promoted from the
private ``_geo.py`` (PR #5 DRY ruling) so experiment figure modules build
their basemaps here instead of re-implementing cartopy setup: use
``cartopy_available()`` for an optional-basemap fallback, ``make_geoaxes``
for the canvas, and ``add_map_features`` to decorate an existing GeoAxes.
The ``geo=`` argument on plotting functions (e.g.
``fields.plot_scatter_overlay``) remains the highest-level entry point.

cartopy is an optional dependency and is imported lazily inside this
module only -- nothing else in ``utils.plotting`` may import it. If
cartopy is missing, ``import_cartopy`` raises a clear ImportError with
the installation command; ``cartopy_available`` reports instead of
raising.

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


def import_cartopy():
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


def cartopy_available() -> bool:
    """True when cartopy is importable — the optional-basemap switch.

    Figure functions with a plain-axes fallback (e.g. the cyclone_jax storm
    panel) branch on this instead of try/excepting cartopy themselves.
    """
    try:
        import_cartopy()
    except ImportError:
        return False
    return True


def add_map_features(
    ax,
    scale: str = "50m",
    color: str = "black",
    lw: float = 0.5,
    fill: bool = False,
    land_color: str = "#f2efe9",
    ocean_color: str = "#dceaf3",
    zorder: Optional[float] = None,
):
    """Add coastline, country-border, and state/province linework to ``ax``.

    Linework only by default -- no land/ocean fill, so data colours stay
    readable; ``fill=True`` adds muted land/ocean face colours underneath
    (basemap style for point/marker figures without a data field).
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
    fill : bool
        If True, also fill LAND/OCEAN polygons with ``land_color`` /
        ``ocean_color``.
    land_color, ocean_color : str
        Face colours used when ``fill=True``.
    zorder : float, optional
        Explicit zorder for the LINEWORK (fill keeps cartopy's default
        low zorder). Use to re-draw coastlines ABOVE a data layer that
        covers the map (e.g. a hexbin field).

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes
    """
    _, cfeature = import_cartopy()
    if fill:
        ax.add_feature(cfeature.LAND.with_scale(scale), facecolor=land_color)
        ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor=ocean_color)
    line_kwargs = {} if zorder is None else {"zorder": zorder}
    for feature in (cfeature.COASTLINE, cfeature.BORDERS, cfeature.STATES):
        ax.add_feature(
            feature.with_scale(scale),
            edgecolor=color, facecolor="none", linewidth=lw, **line_kwargs,
        )
    return ax


def make_geoaxes(
    figsize: tuple[int, int] = (10, 5),
    extent: Optional[list[float]] = None,
    scale: str = "50m",
    color: str = "black",
    lw: float = 0.5,
    gridlines: bool = True,
    projection: str = "platecarree",
    center: Optional[tuple[float, float]] = None,
    fill: bool = False,
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
    fill : bool
        Forwarded to ``add_map_features``: fill land/ocean underneath the
        linework (basemap style).

    Returns
    -------
    tuple
        ``(fig, ax, transform)`` -- ``transform`` is the lon/lat data CRS
        (PlateCarree) to pass to artist calls whose data is in degrees,
        so callers never import cartopy themselves. Data already in the
        projection's native coordinates (azimuthal metres) should be
        plotted WITHOUT a transform kwarg.
    """
    ccrs, _ = import_cartopy()
    lonlat = ccrs.PlateCarree()

    if projection == "platecarree":
        proj = lonlat
    elif projection == "azimuthal":
        if center is None:
            raise ValueError(
                "make_geoaxes: projection='azimuthal' requires "
                "center=(lat, lon)."
            )
        proj = ccrs.AzimuthalEquidistant(
            central_latitude=float(center[0]),
            central_longitude=float(center[1]),
        )
    else:
        raise ValueError(
            f"make_geoaxes: unknown projection '{projection}' "
            f"(expected 'platecarree' or 'azimuthal')."
        )

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    if extent is not None:
        ax.set_extent(extent, crs=proj)
    add_map_features(ax, scale=scale, color=color, lw=lw, fill=fill)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
    return fig, ax, lonlat
