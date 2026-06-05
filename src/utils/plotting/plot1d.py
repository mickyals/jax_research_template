import numpy as np
import matplotlib.pyplot as plt


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
        Moving average window size for the smoothed panel.
        Steps before ``window`` are omitted from the smoothed curve.
        Default 20.
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
    if len(tr_arr) > window:
        smoothed = np.convolve(tr_arr, np.ones(window) / window, mode="valid")
        axes[1].plot(
            np.arange(len(smoothed)) + window // 2,
            smoothed,
            label="train (smoothed)",
            linewidth=2,
        )
    for key in ("val", "test"):
        if losses.get(key) and len(losses[key]) > window:
            arr = np.array(losses[key])
            sm  = np.convolve(arr, np.ones(window) / window, mode="valid")
            axes[1].plot(
                np.arange(len(sm)) + window // 2,
                sm,
                label=f"{key} (smoothed)",
                linewidth=2,
            )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f"{title} (smoothed log)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    return fig
