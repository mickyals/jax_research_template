import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from utils.plotting._style import DEFAULT_CMAP, _resolve_clim, _imshow_with_colorbar, _comparison_stats
from utils.plotting.aggregation import take_slice_np


# ---------------------------------------------------------------------------
# volumes functions
# ---------------------------------------------------------------------------

def plot_volume_comparison(
    true_volume: np.ndarray,
    pred_volume: np.ndarray,
    slice_index: int,
    axis: int = 2,
    extent: Optional[list[float]] = None,
    cmap: str = DEFAULT_CMAP,
    title_prefix: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (16, 4),
    verbose: bool = True,
) -> tuple[plt.Figure, np.ndarray, float]:
    """Plot target, prediction, and residual slices side by side.

    Extracts the same slice from both volumes (via
    ``aggregation.take_slice_np``), computes the residual, and plots all
    three as a three-panel figure. The target and prediction panels share a
    symmetric colormap scaled to the maximum absolute value across both. The
    residual panel uses its own symmetric scale.

    Parameters
    ----------
    true_volume : np.ndarray
        Ground truth 3D array of shape (nx, ny, nz).
    pred_volume : np.ndarray
        Model prediction 3D array, same shape as ``true_volume``.
    slice_index : int
        Index along ``axis`` to extract.
    axis : int
        Axis to slice along. Default 2 (z / altitude).
    extent : list[float], optional
        [xmin, xmax, ymin, ymax] for the two axes not sliced.
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
        If True (default), print the slice MSE.

    Returns
    -------
    tuple[plt.Figure, np.ndarray, float]
        (figure, residual 2D slice, MSE scalar for this slice).

    Example
    -------
    >>> fig, resid, mse = plot_volume_comparison(true_vol, pred_vol,
    ...                                           slice_index=16)
    """
    true_slc = take_slice_np(true_volume, axis=axis, index=slice_index)
    pred_slc = take_slice_np(pred_volume, axis=axis, index=slice_index)
    resid, vmax, rmax, mse = _comparison_stats(true_slc, pred_slc)

    axis_name = {0: "x", 1: "y", 2: "z"}.get(axis, str(axis))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        (true_slc, f"{title_prefix}Target",     vmax),
        (pred_slc, f"{title_prefix}Prediction", vmax),
        (resid,    f"{title_prefix}Residual",   rmax),
    ]
    for ax, (data, title, clim) in zip(axes, panels):
        _imshow_with_colorbar(ax, fig, data, extent=extent, cmap=cmap, vmin=-clim, vmax=clim)
        ax.set_title(f"{title} ({axis_name}={slice_index})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.tight_layout()

    if verbose:
        print(f"Slice MSE ({axis_name}={slice_index}): {mse:.5f}")

    return fig, resid, mse


def plot_surface_3d(
    z: np.ndarray,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    alpha: float = 1.0,
    stride: int = 1,
    symmetric_cmap: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple[int, int] = (10, 7),
) -> plt.Figure:
    """Plot a 2D array as a 3D surface.

    Parameters
    ----------
    z : np.ndarray
        2D array of shape (rows, cols) representing surface heights.
    x : np.ndarray, optional
        1D array of x coordinates, shape (cols,). If None, uses column indices.
    y : np.ndarray, optional
        1D array of y coordinates, shape (rows,). If None, uses row indices.
    cmap : str
        Matplotlib colormap. Default "viridis".
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    zlabel : str
        Z-axis label.
    alpha : float
        Surface transparency. 1.0 is fully opaque.
    stride : int
        Subsampling stride for rendering. Default 1 (no subsampling).
        Increase for large grids -- ``stride=2`` reduces vertex count by 4x.
    symmetric_cmap : bool
        If True, scale colormap symmetrically around zero. Default False.
    vmin : float, optional
        Colormap minimum override.
    vmax : float, optional
        Colormap maximum override.
    figsize : tuple[int, int]
        Figure size in inches.

    Returns
    -------
    plt.Figure

    Example
    -------
    >>> fig = plot_surface_3d(z, title="Random surface")
    >>> fig = plot_surface_3d(loss_grid, x=x, y=y,
    ...                       title="Loss landscape",
    ...                       xlabel="direction 1",
    ...                       ylabel="direction 2",
    ...                       zlabel="loss",
    ...                       stride=2)
    >>> fig.savefig("surface.png", dpi=150)
    """
    z = np.asarray(z)
    rows, cols = z.shape

    if x is None:
        x = np.arange(cols)
    if y is None:
        y = np.arange(rows)

    X, Y = np.meshgrid(x, y)
    lo, hi = _resolve_clim(z, symmetric_cmap, vmin, vmax)

    X_s = X[::stride, ::stride]
    Y_s = Y[::stride, ::stride]
    Z_s = z[::stride, ::stride]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X_s, Y_s, Z_s,
        cmap=cmap, alpha=alpha,
        vmin=lo, vmax=hi,
        linewidth=0, antialiased=True,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.tight_layout()
    return fig
