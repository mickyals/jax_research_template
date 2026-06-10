"""
utils/plotting/_style.py

Private rendering helpers shared across curves.py, fields.py, and volumes.py.
Not part of the public API -- import from those modules instead.

Contents
--------
- DEFAULT_CMAP: shared default colormap.
- _symmetric_clim / _resolve_clim: colormap limit resolution.
- _imshow_with_colorbar: imshow + colorbar, the core of every 2D image panel.
- _comparison_stats: residual + shared clims + MSE for target/prediction/
  residual three-panel comparisons (field, volume, mollweide).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_CMAP = "RdBu_r"


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

    Explicit vmin/vmax always win. Otherwise, symmetric or data-range scaling.
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


def _imshow_with_colorbar(
    ax: plt.Axes,
    fig: plt.Figure,
    data: np.ndarray,
    extent: Optional[list[float]] = None,
    cmap: str = DEFAULT_CMAP,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    aspect: str = "auto",
    origin: str = "lower",
    colorbar_label: str = "",
    transform=None,
    **colorbar_kwargs,
):
    """Draw ``data`` as an image on ``ax`` with a colorbar.

    The shared core of every 2D image panel in fields.py/volumes.py --
    callers set titles and axis labels themselves. ``transform`` is the
    data CRS when drawing on a cartopy GeoAxes (omitted otherwise --
    passing transform=None to imshow would override the default).

    Returns
    -------
    matplotlib.image.AxesImage
    """
    tkw = {} if transform is None else {"transform": transform}
    im = ax.imshow(
        data, origin=origin, cmap=cmap,
        vmin=vmin, vmax=vmax, aspect=aspect, extent=extent, **tkw,
    )
    fig.colorbar(im, ax=ax, label=colorbar_label, **colorbar_kwargs)
    return im


def _comparison_stats(
    true_field: np.ndarray,
    pred_field: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Residual and shared colour limits for a target/prediction/residual comparison.

    Returns
    -------
    tuple[np.ndarray, float, float, float]
        ``(resid, vmax, rmax, mse)`` -- target and prediction share a
        symmetric clim of +/-``vmax``; the residual panel uses its own
        symmetric clim of +/-``rmax``.
    """
    resid = pred_field - true_field
    mse = float((resid ** 2).mean())
    vmax = float(max(np.abs(true_field).max(), np.abs(pred_field).max()))
    rmax = float(np.abs(resid).max()) + 1e-12
    return resid, vmax, rmax, mse


def _contrast_color(value: float, vmin: float, vmax: float) -> str:
    """White text on dark cells, black text on light cells.

    ``value`` is normalized to ``[vmin, vmax]``; values in the upper half
    of the range get white text. Intended for sequential colormaps
    (e.g. annotated heatmap cells).
    """
    span = vmax - vmin
    frac = (value - vmin) / span if span > 0 else 0.0
    return "white" if frac > 0.5 else "black"


def _value_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: Optional[np.ndarray] = None,
    cmap: str = DEFAULT_CMAP,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    size: float = 30,
    size_range: Optional[tuple[float, float]] = None,
    **scatter_kwargs,
):
    """Scatter points, optionally coloured (and sized) by ``values``.

    If ``values`` is None, points are drawn in flat black at ``size`` and
    this returns None (nothing to put on a colorbar). Otherwise points are
    coloured by ``values`` using ``cmap``/``vmin``/``vmax``. If
    ``size_range=(lo, hi)`` is given, point sizes are linearly scaled by
    ``values`` normalised to ``[0, 1]``; otherwise all points use ``size``.

    Returns
    -------
    matplotlib.collections.PathCollection or None
        The scatter artist, or None if ``values`` is None.
    """
    if values is None:
        ax.scatter(x, y, color="black", s=size, **scatter_kwargs)
        return None

    if size_range is not None:
        lo, hi = size_range
        span = float(values.max() - values.min())
        norm = (values - values.min()) / span if span > 0 else np.zeros_like(values)
        s = lo + norm * (hi - lo)
    else:
        s = size

    return ax.scatter(
        x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=s, **scatter_kwargs,
    )
