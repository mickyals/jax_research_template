import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from utils.plotting.plot2d import _resolve_clim


# ---------------------------------------------------------------------------
# plot3d functions
# ---------------------------------------------------------------------------

def plot_volume_slice(
    volume: np.ndarray,
    slice_index: int,
    axis: int = 2,
    extent: Optional[list[float]] = None,
    cmap: str = "RdBu_r",
    title: str = "",
    colorbar_label: str = "",
    symmetric_cmap: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple[int, int] = (8, 5),
) -> np.ndarray:
    """Plot a single 2D slice of a 3D volume.

    Extracts one slice along the specified axis at the given index
    and plots it as a 2D image. Call in a loop to visualise multiple
    levels.

    Input should be a NumPy array or convertible via ``np.array()``.
    JAX arrays are accepted and converted automatically.

    Parameters
    ----------
    volume : np.ndarray
        3D array of shape (nx, ny, nz), or any array-like convertible
        via ``np.array()``.
    slice_index : int
        Index along ``axis`` to extract.
    axis : int
        Axis to slice along. Default 2 (z / altitude).
        0 = x, 1 = y, 2 = z.
    extent : list[float], optional
        [xmin, xmax, ymin, ymax] for the two axes not sliced.
        If None the axes show pixel indices.
    cmap : str
        Matplotlib colormap.
    title : str
        Plot title. If empty, defaults to "axis=index" e.g. "z = 16".
    colorbar_label : str
        Colorbar label.
    symmetric_cmap : bool
        If True (default), scale colormap symmetrically around zero.
        Set to False for all-positive or all-negative data.
    vmin : float, optional
        Colormap minimum override.
    vmax : float, optional
        Colormap maximum override.
    figsize : tuple[int, int]
        Figure size in inches.

    Returns
    -------
    np.ndarray
        The extracted 2D slice. Shape depends on which axis is sliced:
        axis=0 -> (ny, nz), axis=1 -> (nx, nz), axis=2 -> (nx, ny).

    Example
    -------
    >>> vol = np.random.randn(64, 64, 32)
    >>> plot_volume_slice(vol, slice_index=16)
    >>> plot_volume_slice(vol, slice_index=32, axis=0)
    >>> for i in [8, 16, 24]:
    ...     plot_volume_slice(vol, slice_index=i, title=f"z={i}")
    """
    volume = np.asarray(volume)
    slc = np.take(volume, slice_index, axis=axis)
    lo, hi = _resolve_clim(slc, symmetric_cmap, vmin, vmax)

    if not title:
        axis_name = {0: "x", 1: "y", 2: "z"}.get(axis, str(axis))
        title = f"{axis_name} = {slice_index}"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        slc, origin="lower", cmap=cmap,
        vmin=lo, vmax=hi, aspect="auto", extent=extent,
    )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=colorbar_label)
    plt.tight_layout()
    plt.show()

    return slc


def plot_volume_comparison(
    true_volume: np.ndarray,
    pred_volume: np.ndarray,
    slice_index: int,
    axis: int = 2,
    extent: Optional[list[float]] = None,
    cmap: str = "RdBu_r",
    title_prefix: str = "",
    figsize: tuple[int, int] = (16, 4),
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """Plot target, prediction, and residual slices side by side.

    Extracts the same slice from both volumes, computes the residual,
    and plots all three as a three-panel figure. The target and
    prediction panels share a symmetric colormap scaled to the maximum
    absolute value across both. The residual panel uses its own
    symmetric scale. Residuals are always plotted symmetrically since
    they are naturally centred around zero.

    Input arrays should be NumPy arrays or convertible via ``np.array()``.

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
    figsize : tuple[int, int]
        Figure size in inches.
    verbose : bool
        If True (default), print the slice MSE after plotting.

    Returns
    -------
    tuple[np.ndarray, float]
        (residual 2D slice, MSE scalar for this slice).

    Example
    -------
    >>> resid, mse = plot_volume_comparison(true_vol, pred_vol,
    ...                                      slice_index=16)
    >>> resid, mse = plot_volume_comparison(true_vol, pred_vol,
    ...                                      slice_index=32, axis=0,
    ...                                      verbose=False)
    """
    true_volume = np.asarray(true_volume)
    pred_volume = np.asarray(pred_volume)

    true_slc = np.take(true_volume, slice_index, axis=axis)
    pred_slc = np.take(pred_volume, slice_index, axis=axis)
    resid = pred_slc - true_slc
    mse = float((resid ** 2).mean())

    axis_name = {0: "x", 1: "y", 2: "z"}.get(axis, str(axis))
    vmax = float(max(np.abs(true_slc).max(), np.abs(pred_slc).max()))
    rmax = float(np.abs(resid).max()) + 1e-12

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        (true_slc, f"{title_prefix}Target",     vmax),
        (pred_slc, f"{title_prefix}Prediction", vmax),
        (resid,    f"{title_prefix}Residual",   rmax),
    ]
    for ax, (data, title, clim) in zip(axes, panels):
        im = ax.imshow(
            data, origin="lower", extent=extent,
            cmap=cmap, vmin=-clim, vmax=clim, aspect="auto",
        )
        ax.set_title(f"{title} ({axis_name}={slice_index})")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

    if verbose:
        print(f"Slice MSE ({axis_name}={slice_index}): {mse:.5f}")

    return resid, mse


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
) -> None:
    """Plot a 2D array as a 3D surface.

    Parameters
    ----------
    z : np.ndarray
        2D array of shape (rows, cols) representing surface heights.
        Should be a NumPy array or convertible via ``np.array()``.
    x : np.ndarray, optional
        1D array of x coordinates, shape (cols,).
        If None, uses column indices.
    y : np.ndarray, optional
        1D array of y coordinates, shape (rows,).
        If None, uses row indices.
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
        Increase for large grids where rendering is slow -- e.g.
        ``stride=2`` renders every other point in each dimension,
        reducing the vertex count by 4x.
    symmetric_cmap : bool
        If True, scale colormap symmetrically around zero via
        ``_resolve_clim``. Default False since surfaces like loss
        landscapes are typically all-positive.
    vmin : float, optional
        Colormap minimum override.
    vmax : float, optional
        Colormap maximum override.
    figsize : tuple[int, int]
        Figure size in inches.

    Example
    -------
    >>> z = np.random.randn(50, 50)
    >>> plot_surface_3d(z, title="Random surface")

    >>> x = np.linspace(-1., 1., 50)
    >>> y = np.linspace(-1., 1., 50)
    >>> plot_surface_3d(loss_grid, x=x, y=y,
    ...                 title="Loss landscape",
    ...                 xlabel="direction 1",
    ...                 ylabel="direction 2",
    ...                 zlabel="loss",
    ...                 stride=2)
    """
    z = np.asarray(z)
    rows, cols = z.shape

    if x is None:
        x = np.arange(cols)
    if y is None:
        y = np.arange(rows)

    X, Y = np.meshgrid(x, y)

    lo, hi = _resolve_clim(z, symmetric_cmap, vmin, vmax)

    # subsampling via stride
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
    plt.tight_layout()
    plt.show()