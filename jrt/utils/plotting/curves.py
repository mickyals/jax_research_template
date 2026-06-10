from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from utils.plotting.aggregation import rolling_mean_np


def plot_losses(
    losses: dict[str, list[float]],
    title:  str = "Training loss",
    window: int = 20,
    xlabel: str = "step",
    ylabel: str = "loss",
) -> plt.Figure:
    """Plot train and optional validation/test loss curves with a smoothed overlay.

    Produces two side-by-side panels:
    - Left:  raw log-scale loss curves.
    - Right: moving-average smoothed log-scale curves.

    Parameters
    ----------
    losses : dict[str, list[float]]
        Dictionary with key ``"train"`` and optionally ``"test"`` or ``"val"``,
        each mapping to a list of scalar loss values, one per step.
    title : str
        Base title used for both panel headings.
    window : int
        Moving average window size for the smoothed panel, passed to
        ``aggregation.rolling_mean_np``. The window shrinks near the edges
        of the series rather than dropping points. Default 20.
    xlabel : str
        X-axis label for both panels. Default ``"step"``.
    ylabel : str
        Y-axis label for both panels. Default ``"loss"``.

    Returns
    -------
    plt.Figure

    Example
    -------
    >>> fig = plot_losses({"train": [1.0, 0.8, 0.6], "test": [1.1, 0.9, 0.7]})
    >>> fig.savefig("losses.png")
    >>> fig = plot_losses(losses, title="Sphere INR", window=50, ylabel="MSE")
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- raw ---
    axes[0].plot(losses["train"], label="train", alpha=0.7)
    for key in ("val", "test"):
        if losses.get(key):
            axes[0].plot(losses[key], label=key, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f"{title} (log)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")

    # --- smoothed ---
    tr_arr = np.array(losses["train"])
    smoothed = rolling_mean_np(tr_arr, window)
    axes[1].plot(smoothed, label="train (smoothed)", linewidth=2)
    for key in ("val", "test"):
        if losses.get(key):
            arr = np.array(losses[key])
            sm  = rolling_mean_np(arr, window)
            axes[1].plot(sm, label=f"{key} (smoothed)", linewidth=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f"{title} (smoothed log)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    return fig


def plot_grouped_bars(
    values: dict[str, Sequence[float]],
    categories: list[str],
    ylabel: str = "",
    title: str = "",
    ylim: Optional[tuple[float, float]] = None,
    colors: Optional[list[str]] = None,
    bar_width: float = 0.25,
    figsize: tuple[int, int] = (13, 4),
) -> plt.Figure:
    """Grouped bar chart -- one group of bars per category, one bar per series.

    Parameters
    ----------
    values : dict[str, Sequence[float]]
        Series name -> one value per category, e.g.
        ``{"Precision": [...], "Recall": [...], "F1": [...]}``.
        All sequences must have length ``len(categories)``.
    categories : list[str]
        Category labels along the x-axis.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    ylim : tuple[float, float], optional
        Y-axis limits.
    colors : list[str], optional
        One colour per series, in ``values`` order. Falls back to
        matplotlib's default colour cycle if not given.
    bar_width : float
        Width of each bar. Default 0.25.
    figsize : tuple[int, int]
        Figure size in inches.

    Returns
    -------
    plt.Figure

    Example
    -------
    >>> fig = plot_grouped_bars(
    ...     {"Precision": prec, "Recall": rec, "F1": f1}, class_names,
    ...     ylabel="Score", title="Per-class metrics", ylim=(0, 1.05))
    """
    n_cat = len(categories)
    n_series = len(values)
    x = np.arange(n_cat)
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, series) in enumerate(values.items()):
        color = colors[i] if colors is not None else None
        ax.bar(x + offsets[i], series, bar_width, label=name,
               color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
