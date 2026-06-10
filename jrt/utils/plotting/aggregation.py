"""
utils/plotting/aggregation.py

Array preparation for plotting: masked reductions, point binning, smoothing,
and volume slicing. No matplotlib imports — every function here returns a
plain array (or array + extent), ready to hand to a renderer in
curves.py / fields.py / volumes.py.

Design
------
A function lives here only if it encodes a decision that would otherwise be
duplicated at call sites: mask semantics, the NaN/empty-bin convention, bin
edge placement, or window centering. Trivial reductions (np.mean, jnp.sum,
...) are not wrapped — call them directly.

_np / _jax twins
-----------------
``masked_mean`` and ``bin_to_grid`` have NumPy and JAX variants: the JAX
variants are for use inside jitted training/eval loops, the NumPy variants
for post-hoc plotting from large in-memory arrays. ``rolling_mean`` and
``take_slice`` are NumPy-only — always called on small, already-materialised
arrays for plotting.

bin_to_grid empty-bin / out-of-range convention
------------------------------------------------
Empty cells are NaN for "mean"/"max" and 0 for "count".

_np (scipy.stats.binned_statistic_2d) drops points outside ``extent``.
_jax (static-shape scatter) clips out-of-range points into the nearest edge
bin, since JIT requires a fixed output size and cannot drop a variable
number of points. Pre-filter points before calling the JAX variant if
out-of-range handling matters.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import binned_statistic_2d


# ---------------------------------------------------------------------------
# Masked reductions
# ---------------------------------------------------------------------------

def masked_mean_np(
    x: np.ndarray,
    mask: np.ndarray,
    axis: Optional[int | tuple[int, ...]] = None,
) -> np.ndarray:
    """Mean of ``x`` over positions where ``mask`` is True.

    Returns 0.0 (not NaN) where no positions are valid, so the result is
    always safe to plot. Mirrors the masked-loss convention in
    ``training/losses.py``.

    Parameters
    ----------
    x : np.ndarray
    mask : np.ndarray
        Boolean, broadcastable to ``x``. True = include.
    axis : int or tuple[int, ...], optional
        Axis/axes to reduce over. Default None reduces over all axes.

    Returns
    -------
    np.ndarray

    Example
    -------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> mask = np.array([True, False, True])
    >>> masked_mean_np(x, mask)
    2.0
    """
    x = np.asarray(x)
    mask = np.asarray(mask, dtype=bool)
    n_valid = np.sum(mask, axis=axis)
    total = np.sum(np.where(mask, x, 0.0), axis=axis)
    return np.where(n_valid > 0, total / np.maximum(n_valid, 1), 0.0)


@partial(jax.jit, static_argnames=("axis",))
def masked_mean_jax(
    x: jax.Array,
    mask: jax.Array,
    axis: Optional[int | tuple[int, ...]] = None,
) -> jax.Array:
    """JAX twin of :func:`masked_mean_np`. ``axis`` must be static.

    Example
    -------
    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> mask = jnp.array([True, False, True])
    >>> masked_mean_jax(x, mask)
    Array(2., dtype=float32)
    """
    n_valid = jnp.sum(mask, axis=axis)
    total = jnp.sum(jnp.where(mask, x, 0.0), axis=axis)
    return jnp.where(n_valid > 0, total / n_valid, 0.0)


# ---------------------------------------------------------------------------
# Binning scattered points to a grid
# ---------------------------------------------------------------------------

def bin_to_grid_np(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    extent: list[float],
    shape: tuple[int, int],
    reduce: str = "mean",
) -> tuple[np.ndarray, list[float]]:
    """Bin scattered ``(x, y, values)`` points into a 2D grid.

    Parameters
    ----------
    x, y : np.ndarray
        Point coordinates, same shape.
    values : np.ndarray
        Values to reduce, same shape as ``x``. Ignored if ``reduce="count"``.
    extent : list[float]
        ``[xmin, xmax, ymin, ymax]`` bounds of the grid.
    shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the output grid (rows = y, cols = x, matching
        ``imshow`` with ``origin="lower"``).
    reduce : str
        One of "mean", "count", "max". Default "mean".

    Returns
    -------
    tuple[np.ndarray, list[float]]
        ``(grid, extent)`` — grid has shape ``shape``, with empty cells NaN
        for "mean"/"max" and 0 for "count". Points outside ``extent`` are
        dropped.

    Example
    -------
    >>> x = np.array([0.1, 0.1, 0.9])
    >>> y = np.array([0.1, 0.1, 0.9])
    >>> v = np.array([1.0, 3.0, 5.0])
    >>> grid, extent = bin_to_grid_np(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2))
    >>> grid
    array([[2., nan],
           [nan, 5.]])
    """
    if reduce not in ("mean", "count", "max"):
        raise ValueError(f"reduce must be one of 'mean'/'count'/'max', got {reduce!r}")

    xmin, xmax, ymin, ymax = extent
    n_rows, n_cols = shape

    x = np.asarray(x)
    y = np.asarray(y)
    stat_values = np.ones_like(x, dtype=float) if reduce == "count" else np.asarray(values, dtype=float)

    statistic, _, _, _ = binned_statistic_2d(
        x, y, stat_values, statistic=reduce,
        bins=[n_cols, n_rows], range=[[xmin, xmax], [ymin, ymax]],
    )
    grid = statistic.T  # statistic is (n_cols, n_rows); transpose to (n_rows, n_cols)
    return grid, extent


@partial(jax.jit, static_argnames=("shape", "reduce"))
def bin_to_grid_jax(
    x: jax.Array,
    y: jax.Array,
    values: jax.Array,
    extent: list[float],
    shape: tuple[int, int],
    reduce: str = "mean",
) -> tuple[jax.Array, list[float]]:
    """JAX twin of :func:`bin_to_grid_np`. ``shape`` and ``reduce`` must be static.

    Points outside ``extent`` are clipped into the nearest edge bin (see
    module docstring) rather than dropped, since JIT requires a fixed output
    size.

    Example
    -------
    >>> x = jnp.array([0.1, 0.1, 0.9])
    >>> y = jnp.array([0.1, 0.1, 0.9])
    >>> v = jnp.array([1.0, 3.0, 5.0])
    >>> grid, extent = bin_to_grid_jax(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2))
    """
    if reduce not in ("mean", "count", "max"):
        raise ValueError(f"reduce must be one of 'mean'/'count'/'max', got {reduce!r}")

    n_rows, n_cols = shape
    xmin, xmax, ymin, ymax = extent

    x_idx = jnp.clip(((x - xmin) / (xmax - xmin) * n_cols).astype(jnp.int32), 0, n_cols - 1)
    y_idx = jnp.clip(((y - ymin) / (ymax - ymin) * n_rows).astype(jnp.int32), 0, n_rows - 1)
    flat_idx = y_idx * n_cols + x_idx
    n_bins = n_rows * n_cols

    counts = jnp.zeros(n_bins).at[flat_idx].add(1.0)
    if reduce == "count":
        grid = counts
    elif reduce == "mean":
        sums = jnp.zeros(n_bins).at[flat_idx].add(values)
        grid = jnp.where(counts > 0, sums / jnp.maximum(counts, 1.0), jnp.nan)
    else:  # max
        maxima = jnp.full(n_bins, -jnp.inf).at[flat_idx].max(values)
        grid = jnp.where(counts > 0, maxima, jnp.nan)

    return grid.reshape(n_rows, n_cols), extent


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def rolling_mean_np(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, output aligned to input indices.

    Edge points are averaged over however much of the window overlaps the
    array (i.e. the window shrinks near the boundaries rather than the
    output shrinking).

    Parameters
    ----------
    values : np.ndarray
        1D array.
    window : int
        Window size. Must be >= 1.

    Returns
    -------
    np.ndarray
        Same shape as ``values``.

    Example
    -------
    >>> rolling_mean_np(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), window=3)
    array([1.5, 2. , 3. , 4. , 4.5])
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    values = np.asarray(values, dtype=float)
    kernel = np.ones(window)
    sums = np.convolve(values, kernel, mode="same")
    counts = np.convolve(np.ones_like(values), kernel, mode="same")
    return sums / counts


# ---------------------------------------------------------------------------
# Volume slicing
# ---------------------------------------------------------------------------

def take_slice_np(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract a 2D slice from a 3D (or N-D) volume, with bounds checking.

    Parameters
    ----------
    volume : np.ndarray
    axis : int
        Axis to slice along.
    index : int
        Index along ``axis``. Negative indices are allowed (NumPy convention).

    Returns
    -------
    np.ndarray
        ``volume`` with ``axis`` removed.

    Example
    -------
    >>> vol = np.random.randn(64, 64, 32)
    >>> slc = take_slice_np(vol, axis=2, index=16)
    >>> slc.shape
    (64, 64)
    """
    volume = np.asarray(volume)
    if not (0 <= axis < volume.ndim):
        raise ValueError(f"axis must be in [0, {volume.ndim}), got {axis}")
    if not (-volume.shape[axis] <= index < volume.shape[axis]):
        raise IndexError(
            f"index {index} out of bounds for axis {axis} with size {volume.shape[axis]}"
        )
    return np.take(volume, index, axis=axis)
