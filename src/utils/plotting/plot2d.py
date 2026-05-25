import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _symmetric_clim(data: np.ndarray) -> tuple[float, float]:
    """Return (-vmax, vmax) where vmax = max(|data|)."""
    vmax = float(np.abs(data).max())
    return -vmax, vmax


def _resolve_clim(
    data: np.ndarray,
    symmetric: bool,
    vmin: Optional[float],
    vmax: Optional[float],
) -> tuple[float, float]:
    """Resolve colormap limits from data, symmetric flag, and explicit overrides.

    Explicit vmin/vmax always win. Otherwise symmetric or data-range scaling.
    """
    if vmin is not None and vmax is not None:
        return vmin, vmax
    if symmetric:
        lo, hi = _symmetric_clim(data)
    else:
        lo = float(data.min())
        hi = float(data.max())
    if vmin is not None:
        lo = vmin
    if vmax is not None:
        hi = vmax
    return lo, hi


# ---------------------------------------------------------------------------
# plot2d functions
# ---------------------------------------------------------------------------

def plot_field_2d(
    field: np.ndarray,
    extent: Optional[list[float]] = None,
    cmap: str = "RdBu_r",
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    colorbar_label: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric_cmap: bool = True,
    figsize: tuple[int, int] = (10, 5),
) -> None:
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

    Example
    -------
    >>> field = np.random.randn(64, 64)
    >>> plot_field_2d(field, extent=[-180, 180, -90, 90], title="Example")
    >>> plot_field_2d(field ** 2, symmetric_cmap=False)
    """
    lo, hi = _resolve_clim(field, symmetric_cmap, vmin, vmax)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        field, origin="lower", cmap=cmap,
        vmin=lo, vmax=hi, aspect="auto", extent=extent,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, label=colorbar_label)
    plt.tight_layout()
    plt.show()


def plot_field_comparison_2d(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    extent: Optional[list[float]] = None,
    cmap: str = "RdBu_r",
    title_prefix: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: tuple[int, int] = (16, 4),
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
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
    tuple[np.ndarray, float]
        (residual array of shape (rows, cols), grid MSE scalar).

    Example
    -------
    >>> resid, mse = plot_field_comparison_2d(true, pred,
    ...                                        extent=[-100, -40, 0, 30])
    >>> print(f"MSE: {mse:.5f}")
    """
    resid = pred_field - true_field
    mse = float((resid ** 2).mean())
    vmax = float(max(np.abs(true_field).max(), np.abs(pred_field).max()))
    rmax = float(np.abs(resid).max()) + 1e-12

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        (true_field,  f"{title_prefix}Target",     vmax),
        (pred_field,  f"{title_prefix}Prediction", vmax),
        (resid,       f"{title_prefix}Residual",   rmax),
    ]
    for ax, (data, title, clim) in zip(axes, panels):
        im = ax.imshow(
            data, origin="lower", extent=extent,
            cmap=cmap, vmin=-clim, vmax=clim, aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

    if verbose:
        print(f"Grid MSE: {mse:.5f}")

    return resid, mse


def plot_scatter_overlay(
    field: np.ndarray,
    scatter_x: np.ndarray,
    scatter_y: np.ndarray,
    scatter_values: Optional[np.ndarray] = None,
    extent: Optional[list[float]] = None,
    cmap: str = "RdBu_r",
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    scatter_size: int = 30,
    scatter_vmin: Optional[float] = None,
    scatter_vmax: Optional[float] = None,
    symmetric_cmap: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """Plot a 2D field with a scatter overlay.

    Scatter points can use independent colour scaling from the field
    via ``scatter_vmin`` / ``scatter_vmax``. If those are not provided
    and ``scatter_values`` is given, the scatter inherits the field's
    colormap limits.

    Parameters
    ----------
    field : np.ndarray
        2D array of shape (rows, cols).
    scatter_x : np.ndarray
        X coordinates of scatter points, shape (n,).
    scatter_y : np.ndarray
        Y coordinates of scatter points, shape (n,).
    scatter_values : np.ndarray, optional
        Values used to colour scatter points.
        If None, points are plotted in black.
    extent : list[float], optional
        [xmin, xmax, ymin, ymax] for axis labels.
    cmap : str
        Matplotlib colormap applied to both field and scatter.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    scatter_size : int
        Scatter marker size.
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
    figsize : tuple[int, int]
        Figure size in inches.

    Example
    -------
    >>> plot_scatter_overlay(field, lons, lats, values,
    ...                       extent=[-100, -40, 0, 30])
    >>> plot_scatter_overlay(field, lons, lats, probs,
    ...                       scatter_vmin=0., scatter_vmax=1.,
    ...                       symmetric_cmap=False)
    """
    field_vmin, field_vmax = _resolve_clim(field, symmetric_cmap, vmin, vmax)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        field, origin="lower", cmap=cmap,
        vmin=field_vmin, vmax=field_vmax,
        aspect="auto", extent=extent,
    )
    if scatter_values is not None:
        s_vmin = scatter_vmin if scatter_vmin is not None else field_vmin
        s_vmax = scatter_vmax if scatter_vmax is not None else field_vmax
        ax.scatter(
            scatter_x, scatter_y,
            c=scatter_values, cmap=cmap,
            vmin=s_vmin, vmax=s_vmax,
            s=scatter_size, edgecolor="black", linewidth=0.3,
        )
    else:
        ax.scatter(scatter_x, scatter_y, color="black",
                   s=scatter_size, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: Optional[list[str]] = None,
    col_labels: Optional[list[str]] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
    colorbar_label: str = "",
    annotate: bool = False,
    fmt: str = ".2f",
    annotate_fontsize: int = 8,
    figsize: Optional[tuple[int, int]] = None,
    max_figsize: tuple[int, int] = (14, 14),
) -> None:
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
        If True, write the numeric value of each cell in the plot.
    fmt : str
        Format string for annotations. Default ".2f".
    annotate_fontsize : int
        Font size for cell annotations. Default 8. Reduce for large matrices.
    figsize : tuple[int, int], optional
        Figure size. Auto-computed from matrix shape if not provided,
        capped by ``max_figsize``.
    max_figsize : tuple[int, int]
        Upper bound on auto-computed figure size. Default (14, 14).

    Example
    -------
    >>> sim = enc_norm @ enc_norm.T
    >>> plot_heatmap(sim, row_labels=labels, col_labels=labels,
    ...              title="Cosine similarity", annotate=True)
    >>> plot_heatmap(large_matrix, annotate=True, annotate_fontsize=5)
    """
    n_rows, n_cols = matrix.shape
    if figsize is None:
        w = min(max_figsize[0], max(4, n_cols * 0.8 + 1))
        h = min(max_figsize[1], max(4, n_rows * 0.8 + 1))
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

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
                        fontsize=annotate_fontsize)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=colorbar_label)
    plt.tight_layout()
    plt.show()


def plot_mollweide(
    field: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    cmap: str = "RdBu_r",
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
) -> None:
    """Plot a scalar field on a Mollweide projection.

    Scatter points can be coloured independently from the field via
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

    Example
    -------
    >>> plot_mollweide(field, LON, LAT, title="Global field")
    >>> plot_mollweide(field, LON, LAT,
    ...                scatter_lon=obs_lon, scatter_lat=obs_lat,
    ...                scatter_values=obs_prob,
    ...                scatter_vmin=0., scatter_vmax=1.,
    ...                scatter_cmap="viridis")
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
    plt.colorbar(im, ax=ax, orientation="horizontal",
                 pad=0.05, shrink=0.7, label=colorbar_label)
    plt.tight_layout()
    plt.show()


def plot_mollweide_comparison(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    cmap: str = "RdBu_r",
    title_prefix: str = "",
    scatter_lon: Optional[np.ndarray] = None,
    scatter_lat: Optional[np.ndarray] = None,
    scatter_values: Optional[np.ndarray] = None,
    scatter_cmap: Optional[str] = None,
    scatter_vmin: Optional[float] = None,
    scatter_vmax: Optional[float] = None,
    figsize: tuple[int, int] = (18, 4),
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """Plot target, prediction, and residual on three Mollweide panels.

    Scatter points can be coloured independently from the field via
    ``scatter_vmin`` / ``scatter_vmax``. If not provided, scatter
    inherits the field's symmetric colour limits. This is documented
    explicitly to avoid confusion when scatter values have a different
    range to the field (e.g. probabilities in [0, 1]).

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
        Values used to colour scatter points. If None, points are black.
        By default shares the field's symmetric colour limits -- pass
        ``scatter_vmin`` / ``scatter_vmax`` to decouple.
    scatter_cmap : str, optional
        Colormap for scatter points. Defaults to field cmap.
    scatter_vmin : float, optional
        Colormap minimum for scatter. Defaults to field vmin.
    scatter_vmax : float, optional
        Colormap maximum for scatter. Defaults to field vmax.
    figsize : tuple[int, int]
        Figure size in inches.
    verbose : bool
        If True (default), print the grid MSE after plotting.

    Returns
    -------
    tuple[np.ndarray, float]
        (residual array of shape (rows, cols), grid MSE scalar).

    Example
    -------
    >>> resid, mse = plot_mollweide_comparison(true, pred, LON, LAT)
    >>> resid, mse = plot_mollweide_comparison(
    ...     true, pred, LON, LAT,
    ...     scatter_lon=obs_lon, scatter_lat=obs_lat,
    ...     scatter_values=obs_prob,
    ...     scatter_vmin=0., scatter_vmax=1.,
    ...     scatter_cmap="viridis",
    ... )
    """
    resid = pred_field - true_field
    mse = float((resid ** 2).mean())
    vmax = float(max(np.abs(true_field).max(), np.abs(pred_field).max()))
    rmax = float(np.abs(resid).max()) + 1e-12

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
        plt.colorbar(im, ax=ax, orientation="horizontal",
                     pad=0.05, shrink=0.7)

    plt.tight_layout()
    plt.show()

    if verbose:
        print(f"Grid MSE: {mse:.5f}")

    return resid, mse