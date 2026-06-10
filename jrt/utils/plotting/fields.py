import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from utils.plotting._style import (
    DEFAULT_CMAP,
    _resolve_clim,
    _imshow_with_colorbar,
    _comparison_stats,
    _contrast_color,
    _value_scatter,
)


# ---------------------------------------------------------------------------
# fields functions
# ---------------------------------------------------------------------------

def plot_field_2d(
    field: np.ndarray,
    extent: Optional[list[float]] = None,
    cmap: str = DEFAULT_CMAP,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    colorbar_label: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric_cmap: bool = True,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot a 2D scalar field as an image.

    Parameters
    ----------
    field : np.ndarray
        2D array of shape (rows, cols).
    extent : list[float], optional
        [xmin, xmax, ymin, ymax] for axis labels.
        If None the axes show pixel indices.
    cmap : str
        Matplotlib colormap. Default "RdBu_r".
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    colorbar_label : str
        Label for the colorbar.
    vmin : float, optional
        Colormap minimum. Overrides symmetric scaling when provided.
    vmax : float, optional
        Colormap maximum. Overrides symmetric scaling when provided.
    symmetric_cmap : bool
        If True (default), scale colormap symmetrically around zero
        using max(|field|). Set to False for all-positive or all-negative
        fields where symmetric scaling would waste half the colorbar range.
    figsize : tuple[int, int]
        Figure size in inches.

    Returns
    -------
    plt.Figure

    Example
    -------
    >>> field = np.random.randn(64, 64)
    >>> fig = plot_field_2d(field, extent=[-180, 180, -90, 90], title="Example")
    >>> fig = plot_field_2d(field ** 2, symmetric_cmap=False)
    """
    lo, hi = _resolve_clim(field, symmetric_cmap, vmin, vmax)

    fig, ax = plt.subplots(figsize=figsize)
    _imshow_with_colorbar(
        ax, fig, field, extent=extent, cmap=cmap,
        vmin=lo, vmax=hi, colorbar_label=colorbar_label,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_field_comparison_2d(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    extent: Optional[list[float]] = None,
    cmap: str = DEFAULT_CMAP,
    title_prefix: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: tuple[int, int] = (16, 4),
    verbose: bool = True,
) -> tuple[plt.Figure, np.ndarray, float]:
    """Plot target, prediction, and residual side by side.

    All three panels share a symmetric colormap scaled to the maximum
    absolute value across target and prediction. The residual panel uses
    its own symmetric scale.

    Parameters
    ----------
    true_field : np.ndarray
        Ground truth 2D array of shape (rows, cols).
    pred_field : np.ndarray
        Model prediction 2D array of shape (rows, cols).
    extent : list[float], optional
        [xmin, xmax, ymin, ymax] for axis labels.
    cmap : str
        Matplotlib colormap.
    title_prefix : str
        String prepended to each panel title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : tuple[int, int]
        Figure size in inches.
    verbose : bool
        If True (default), print the grid MSE after plotting.

    Returns
    -------
    tuple[plt.Figure, np.ndarray, float]
        (figure, residual array of shape (rows, cols), grid MSE scalar).

    Example
    -------
    >>> fig, resid, mse = plot_field_comparison_2d(true, pred,
    ...                                             extent=[-100, -40, 0, 30])
    """
    resid, vmax, rmax, mse = _comparison_stats(true_field, pred_field)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        (true_field,  f"{title_prefix}Target",     vmax),
        (pred_field,  f"{title_prefix}Prediction", vmax),
        (resid,       f"{title_prefix}Residual",   rmax),
    ]
    for ax, (data, title, clim) in zip(axes, panels):
        _imshow_with_colorbar(ax, fig, data, extent=extent, cmap=cmap, vmin=-clim, vmax=clim)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.tight_layout()

    if verbose:
        print(f"Grid MSE: {mse:.5f}")

    return fig, resid, mse


def plot_scatter_overlay(
    field: Optional[np.ndarray],
    scatter_x: np.ndarray,
    scatter_y: np.ndarray,
    scatter_values: Optional[np.ndarray] = None,
    extent: Optional[list[float]] = None,
    cmap: str = DEFAULT_CMAP,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    colorbar_label: str = "",
    scatter_size: float = 30,
    scatter_size_range: Optional[tuple[float, float]] = None,
    scatter_vmin: Optional[float] = None,
    scatter_vmax: Optional[float] = None,
    symmetric_cmap: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    marker_x: Optional[float] = None,
    marker_y: Optional[float] = None,
    marker_label: Optional[str] = None,
    marker_kwargs: Optional[dict] = None,
    grid: bool = False,
    geo: bool | dict = False,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot scattered points, optionally over a 2D field background.

    Scatter points can use independent colour scaling from the field
    via ``scatter_vmin`` / ``scatter_vmax``. If those are not provided
    and ``scatter_values`` is given, the scatter inherits the field's
    colormap limits. If ``field`` is None, the scatter values get their
    own colorbar (resolved the same way the field's would be).

    With ``geo`` set, x/y are longitude/latitude in degrees and the plot
    is drawn on a PlateCarree map with coastlines, borders, and
    state/province lines (requires cartopy, an optional dependency).

    Parameters
    ----------
    field : np.ndarray or None
        2D array of shape (rows, cols), or None to plot scatter points
        on bare axes (e.g. station positions with no background field).
    scatter_x : np.ndarray
        X coordinates of scatter points, shape (n,).
    scatter_y : np.ndarray
        Y coordinates of scatter points, shape (n,).
    scatter_values : np.ndarray, optional
        Values used to colour scatter points.
        If None, points are plotted in black.
    extent : list[float], optional
        [xmin, xmax, ymin, ymax] for axis labels. Also sets the axis
        limits when ``field`` is None.
    cmap : str
        Matplotlib colormap applied to both field and scatter.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    colorbar_label : str
        Label for the scatter colorbar, used only when ``field`` is None
        and ``scatter_values`` is given.
    scatter_size : float
        Scatter marker size, used when ``scatter_size_range`` is None.
    scatter_size_range : tuple[float, float], optional
        If given, point sizes are scaled by ``scatter_values`` normalised
        to ``[0, 1]`` and mapped onto ``(lo, hi)``.
    scatter_vmin : float, optional
        Colormap minimum for scatter points. Defaults to field vmin.
    scatter_vmax : float, optional
        Colormap maximum for scatter points. Defaults to field vmax.
    symmetric_cmap : bool
        If True (default), scale field colormap symmetrically around zero.
        Set to False for all-positive or all-negative fields.
    vmin : float, optional
        Field colormap minimum override.
    vmax : float, optional
        Field colormap maximum override.
    marker_x, marker_y : float, optional
        Position of a single highlighted reference point (e.g. a query
        or storm centre), drawn as a star marker.
    marker_label : str, optional
        Legend label for the reference point.
    marker_kwargs : dict, optional
        Extra kwargs for the reference-point scatter, merged over the
        default ``{"marker": "*", "s": 200, "color": "royalblue", "zorder": 5}``.
    grid : bool
        If True, draw a dashed grid. Only applied when ``field`` is None
        (a grid over an image is usually unwanted). Default False.
    geo : bool or dict
        If truthy, draw on a PlateCarree map instead of plain axes
        (coordinates must be lon/lat degrees). ``True`` uses the default
        map styling; a dict is forwarded to the canvas factory as
        overrides (``scale`` ('110m'/'50m'/'10m'), ``color``, ``lw``,
        ``gridlines``). Labeled map gridlines replace ``grid``,
        ``xlabel``, and ``ylabel``, which are ignored. Requires cartopy.
    figsize : tuple[int, int]
        Figure size in inches.

    Returns
    -------
    plt.Figure

    Example
    -------
    >>> fig = plot_scatter_overlay(field, lons, lats, values,
    ...                            extent=[-100, -40, 0, 30])
    >>> fig = plot_scatter_overlay(None, lons, lats, values,
    ...                            extent=[-100, -40, 0, 30],
    ...                            marker_x=q_lon, marker_y=q_lat)
    >>> fig = plot_scatter_overlay(None, lons, lats, values,
    ...                            extent=[-100, -40, 0, 30],
    ...                            geo={"scale": "10m"})
    """
    geo_opts = {} if geo is True else dict(geo) if isinstance(geo, dict) else None

    if geo_opts is not None:
        from utils.plotting._geo import _make_geoaxes
        fig, ax, transform = _make_geoaxes(
            figsize=figsize, extent=extent, **geo_opts,
        )
        tkw = {"transform": transform}
    else:
        fig, ax = plt.subplots(figsize=figsize)
        transform = None
        tkw = {}

    if field is not None:
        field_vmin, field_vmax = _resolve_clim(field, symmetric_cmap, vmin, vmax)
        _imshow_with_colorbar(
            ax, fig, field, extent=extent, cmap=cmap,
            vmin=field_vmin, vmax=field_vmax, transform=transform,
        )
        s_vmin = scatter_vmin if scatter_vmin is not None else field_vmin
        s_vmax = scatter_vmax if scatter_vmax is not None else field_vmax
    elif scatter_values is not None:
        s_vmin, s_vmax = _resolve_clim(scatter_values, symmetric_cmap,
                                        scatter_vmin, scatter_vmax)
    else:
        s_vmin = s_vmax = None

    if scatter_values is not None:
        sc = _value_scatter(
            ax, scatter_x, scatter_y, values=scatter_values,
            cmap=cmap, vmin=s_vmin, vmax=s_vmax,
            size=scatter_size, size_range=scatter_size_range,
            edgecolor="black", linewidth=0.3, **tkw,
        )
        if field is None:
            fig.colorbar(sc, ax=ax, label=colorbar_label)
    else:
        _value_scatter(ax, scatter_x, scatter_y, values=None,
                       size=scatter_size, alpha=0.6, **tkw)

    if marker_x is not None and marker_y is not None:
        mk = {"marker": "*", "s": 200, "color": "royalblue", "zorder": 5}
        if marker_kwargs:
            mk.update(marker_kwargs)
        ax.scatter([marker_x], [marker_y], label=marker_label, **mk, **tkw)
        if marker_label:
            ax.legend()

    ax.set_title(title)
    if geo_opts is None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if field is None:
            if extent is not None:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
            if grid:
                ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: Optional[list[str]] = None,
    col_labels: Optional[list[str]] = None,
    xlabel: str = "",
    ylabel: str = "",
    cmap: str = DEFAULT_CMAP,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
    colorbar_label: str = "",
    annotate: bool = False,
    fmt: str = ".2f",
    annotate_fontsize: int = 8,
    figsize: Optional[tuple[int, int]] = None,
    max_figsize: tuple[int, int] = (14, 14),
) -> plt.Figure:
    """Plot a 2D matrix as a heatmap with optional annotations.

    General purpose -- used for cosine similarity matrices, confusion
    matrices, correlation matrices, or any 2D array where explicit
    tick labels add meaning.

    Parameters
    ----------
    matrix : np.ndarray
        2D array of shape (rows, cols).
    row_labels : list[str], optional
        Tick labels for the y-axis.
    col_labels : list[str], optional
        Tick labels for the x-axis.
    xlabel : str
        X-axis label. Default ``""``.
    ylabel : str
        Y-axis label. Default ``""``.
    cmap : str
        Matplotlib colormap.
    vmin : float, optional
        Colormap minimum.
    vmax : float, optional
        Colormap maximum.
    title : str
        Plot title.
    colorbar_label : str
        Label for the colorbar.
    annotate : bool
        If True, write the numeric value of each cell in the plot, with
        text colour (white/black) chosen for contrast against the cell.
    fmt : str
        Format string for annotations. Default ".2f".
    annotate_fontsize : int
        Font size for cell annotations. Default 8. Reduce for large matrices.
    figsize : tuple[int, int], optional
        Figure size. Auto-computed from matrix shape if not provided,
        capped by ``max_figsize``.
    max_figsize : tuple[int, int]
        Upper bound on auto-computed figure size. Default (14, 14).

    Returns
    -------
    plt.Figure

    Example
    -------
    >>> sim = enc_norm @ enc_norm.T
    >>> fig = plot_heatmap(sim, row_labels=labels, col_labels=labels,
    ...                    xlabel="Predicted", ylabel="True",
    ...                    title="Cosine similarity", annotate=True)
    >>> fig.savefig("sim.png", dpi=150)
    """
    n_rows, n_cols = matrix.shape
    if figsize is None:
        w = min(max_figsize[0], max(4, n_cols * 0.8 + 1))
        h = min(max_figsize[1], max(4, n_rows * 0.8 + 1))
        figsize = (w, h)

    lo = vmin if vmin is not None else float(matrix.min())
    hi = vmax if vmax is not None else float(matrix.max())

    fig, ax = plt.subplots(figsize=figsize)
    _imshow_with_colorbar(
        ax, fig, matrix, cmap=cmap, vmin=lo, vmax=hi,
        origin="upper", colorbar_label=colorbar_label,
    )

    if row_labels is not None:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels)
    if col_labels is not None:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(j, i, format(matrix[i, j], fmt),
                        ha="center", va="center",
                        fontsize=annotate_fontsize,
                        color=_contrast_color(matrix[i, j], lo, hi))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_mollweide(
    field: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    cmap: str = DEFAULT_CMAP,
    title: str = "",
    colorbar_label: str = "",
    symmetric_cmap: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    scatter_lon: Optional[np.ndarray] = None,
    scatter_lat: Optional[np.ndarray] = None,
    scatter_values: Optional[np.ndarray] = None,
    scatter_cmap: Optional[str] = None,
    scatter_vmin: Optional[float] = None,
    scatter_vmax: Optional[float] = None,
    scatter_size: int = 8,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot a scalar field on a Mollweide projection.

    Generic matplotlib Mollweide projection -- works for any sphere (Earth,
    all-sky, black-hole imaging), not just geographic data. For Earth maps
    with coastlines/borders, see ``geographic.py``.

    Scatter points can be coloured independently of the field via
    ``scatter_vmin`` / ``scatter_vmax`` and ``scatter_cmap``.
    If those are not provided, scatter inherits the field's colormap
    and colour limits.

    Parameters
    ----------
    field : np.ndarray
        2D array of shape (rows, cols) matching lon_grid / lat_grid.
    lon_grid : np.ndarray
        2D array of longitudes in radians, shape (rows, cols).
    lat_grid : np.ndarray
        2D array of latitudes in radians, shape (rows, cols).
    cmap : str
        Matplotlib colormap for the field.
    title : str
        Plot title.
    colorbar_label : str
        Colorbar label.
    symmetric_cmap : bool
        If True (default), scale field colormap symmetrically around zero.
        Set to False for all-positive or all-negative fields.
    vmin : float, optional
        Field colormap minimum override.
    vmax : float, optional
        Field colormap maximum override.
    scatter_lon : np.ndarray, optional
        Longitudes of scatter points in radians.
    scatter_lat : np.ndarray, optional
        Latitudes of scatter points in radians.
    scatter_values : np.ndarray, optional
        Values used to colour scatter points. If None, points are black.
    scatter_cmap : str, optional
        Colormap for scatter points. Defaults to field cmap.
    scatter_vmin : float, optional
        Colormap minimum for scatter. Defaults to field vmin.
    scatter_vmax : float, optional
        Colormap maximum for scatter. Defaults to field vmax.
    scatter_size : int
        Scatter marker size.
    figsize : tuple[int, int]
        Figure size in inches.

    Returns
    -------
    plt.Figure

    Example
    -------
    >>> fig = plot_mollweide(field, LON, LAT, title="Global field")
    >>> fig.savefig("global.png", dpi=150)
    """
    field_vmin, field_vmax = _resolve_clim(field, symmetric_cmap, vmin, vmax)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="mollweide")
    im = ax.pcolormesh(
        lon_grid, lat_grid, field,
        cmap=cmap, vmin=field_vmin, vmax=field_vmax, shading="auto",
    )
    if scatter_lon is not None:
        if scatter_values is not None:
            s_vmin = scatter_vmin if scatter_vmin is not None else field_vmin
            s_vmax = scatter_vmax if scatter_vmax is not None else field_vmax
            s_cmap = scatter_cmap if scatter_cmap is not None else cmap
            ax.scatter(
                scatter_lon, scatter_lat,
                c=scatter_values, cmap=s_cmap,
                vmin=s_vmin, vmax=s_vmax,
                s=scatter_size, edgecolor="none", alpha=0.7,
            )
        else:
            ax.scatter(scatter_lon, scatter_lat,
                       color="black", s=scatter_size, alpha=0.6)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.colorbar(im, ax=ax, orientation="horizontal",
                 pad=0.05, shrink=0.7, label=colorbar_label)
    fig.tight_layout()
    return fig


def plot_mollweide_comparison(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    cmap: str = DEFAULT_CMAP,
    title_prefix: str = "",
    scatter_lon: Optional[np.ndarray] = None,
    scatter_lat: Optional[np.ndarray] = None,
    scatter_values: Optional[np.ndarray] = None,
    scatter_cmap: Optional[str] = None,
    scatter_vmin: Optional[float] = None,
    scatter_vmax: Optional[float] = None,
    figsize: tuple[int, int] = (18, 4),
    verbose: bool = True,
) -> tuple[plt.Figure, np.ndarray, float]:
    """Plot target, prediction, and residual on three Mollweide panels.

    Parameters
    ----------
    true_field : np.ndarray
        Ground truth 2D array of shape (rows, cols).
    pred_field : np.ndarray
        Model prediction 2D array of shape (rows, cols).
    lon_grid : np.ndarray
        2D array of longitudes in radians, shape (rows, cols).
    lat_grid : np.ndarray
        2D array of latitudes in radians, shape (rows, cols).
    cmap : str
        Matplotlib colormap.
    title_prefix : str
        String prepended to each panel title.
    scatter_lon : np.ndarray, optional
        Longitudes of scatter points in radians.
    scatter_lat : np.ndarray, optional
        Latitudes of scatter points in radians.
    scatter_values : np.ndarray, optional
        Values used to colour scatter points.
    scatter_cmap : str, optional
        Colormap for scatter points. Defaults to field cmap.
    scatter_vmin : float, optional
        Colormap minimum for scatter. Defaults to field vmin.
    scatter_vmax : float, optional
        Colormap maximum for scatter. Defaults to field vmax.
    figsize : tuple[int, int]
        Figure size in inches.
    verbose : bool
        If True (default), print the grid MSE.

    Returns
    -------
    tuple[plt.Figure, np.ndarray, float]
        (figure, residual array of shape (rows, cols), grid MSE scalar).

    Example
    -------
    >>> fig, resid, mse = plot_mollweide_comparison(true, pred, LON, LAT)
    """
    resid, vmax, rmax, mse = _comparison_stats(true_field, pred_field)

    fig = plt.figure(figsize=figsize)
    panels = [
        (true_field,  f"{title_prefix}Target",     vmax),
        (pred_field,  f"{title_prefix}Prediction", vmax),
        (resid,       f"{title_prefix}Residual",   rmax),
    ]
    for idx, (data, title, clim) in enumerate(panels):
        ax = fig.add_subplot(1, 3, idx + 1, projection="mollweide")
        im = ax.pcolormesh(
            lon_grid, lat_grid, data,
            cmap=cmap, vmin=-clim, vmax=clim, shading="auto",
        )
        if scatter_lon is not None:
            if scatter_values is not None:
                s_vmin = scatter_vmin if scatter_vmin is not None else -vmax
                s_vmax = scatter_vmax if scatter_vmax is not None else vmax
                s_cmap = scatter_cmap if scatter_cmap is not None else cmap
                ax.scatter(
                    scatter_lon, scatter_lat,
                    c=scatter_values, cmap=s_cmap,
                    vmin=s_vmin, vmax=s_vmax,
                    s=4, edgecolor="none", alpha=0.7,
                )
            else:
                ax.scatter(scatter_lon, scatter_lat,
                           color="black", s=4, alpha=0.4)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)

    fig.tight_layout()

    if verbose:
        print(f"Grid MSE: {mse:.5f}")

    return fig, resid, mse
