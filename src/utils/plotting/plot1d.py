import numpy as np
import matplotlib.pyplot as plt


def plot_losses(
    losses: dict[str, list[float]],
    title: str = "Training loss",
    window: int = 20,
) -> None:
    """Plot train and optional test loss curves with a smoothed overlay.

    Produces two side-by-side panels:
    - Left: raw log-scale loss curves.
    - Right: moving-average smoothed log-scale curves.

    Parameters
    ----------
    losses : dict[str, list[float]]
        Dictionary with keys ``"train"`` and optionally ``"test"``,
        each mapping to a list of scalar loss values, one per step.
        Returned directly by ``train()``.
    title : str
        Base title used for both panel headings.
    window : int
        Moving average window size for the smoothed panel.
        Steps before ``window`` are omitted from the smoothed curve.
        Default 20.

    Example
    -------
    >>> plot_losses({"train": [1.0, 0.8, 0.6], "test": [1.1, 0.9, 0.7]})
    >>> plot_losses(losses, title="Sphere INR", window=50)
    """
    has_test = bool(losses.get("test"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- raw ---
    axes[0].plot(losses["train"], label="train", alpha=0.7)
    if has_test:
        axes[0].plot(losses["test"], label="test", alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("MSE")
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
    if has_test and len(losses["test"]) > window:
        te_arr = np.array(losses["test"])
        smoothed_te = np.convolve(te_arr, np.ones(window) / window, mode="valid")
        axes[1].plot(
            np.arange(len(smoothed_te)) + window // 2,
            smoothed_te,
            label="test (smoothed)",
            linewidth=2,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("MSE")
    axes[1].set_title(f"{title} (smoothed log)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.show()