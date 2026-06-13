"""
training/logger.py

Experiment logging backends for JAX/Flax research.

Provides a shared interface over Weights & Biases (primary) and
TensorBoard (cluster fallback), plus a NullLogger for unit tests and
dry runs. The trainer only ever calls the shared interface, so switching
backends requires changing one argument at logger construction time.

Shared interface
----------------
    log_metrics(metrics, step)          -- scalar dict
    log_hyperparams(hparams)            -- experiment config
    log_figure(tag, figure, step)       -- matplotlib Figure
    log_image(tag, image, step)         -- numpy/jax HWC array
    log_histogram(tag, values, step)    -- 1-D array -> distribution
    log_artifact(name, path, type)      -- file/dir artifact (backend-specific)
    finalize(status)                    -- 'success' | 'failed'
    log_dir                             -- pathlib.Path to run directory

All backends accept matplotlib Figures for log_figure (no conversion
required by the caller). log_image accepts HWC arrays in [0, 255] uint8
or [0, 1] float32 — the backend handles any needed transposing.

Dependencies
------------
WandB and TensorBoard are soft dependencies — ImportError is raised only
when the corresponding logger is instantiated, not at module import time.
NullLogger has no external dependencies.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseLogger(ABC):
    """Abstract base for all experiment loggers."""

    @property
    @abstractmethod
    def log_dir(self) -> Path:
        """Path to the run's logging directory."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log a dictionary of scalar metrics.

        Parameters
        ----------
        metrics : dict[str, scalar]
            Keys are metric names (e.g. 'train/loss').
        step : int
            Training step or epoch index.
        """

    @abstractmethod
    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        """Log experiment hyperparameters (called once at training start).

        Parameters
        ----------
        hparams : dict
            Full config dictionary.
        """

    @abstractmethod
    def log_figure(self, tag: str, figure, step: int) -> None:
        """Log a matplotlib Figure.

        The logger closes the figure after logging — the caller should not
        call plt.close() separately.

        Parameters
        ----------
        tag : str
            Name for this figure (e.g. 'val/wind_field').
        figure : matplotlib.figure.Figure
        step : int
        """

    @abstractmethod
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """Log an image array.

        Parameters
        ----------
        tag : str
        image : np.ndarray
            Shape (H, W) or (H, W, C). Values in [0, 255] uint8 or
            [0.0, 1.0] float32 — both accepted.
        step : int
        """

    @abstractmethod
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a distribution of values as a histogram.

        Parameters
        ----------
        tag : str
            Name (e.g. 'params/layer0_weights').
        values : array-like
            Values to histogram. Any shape — will be flattened.
        step : int
        """

    def log_artifact(
        self,
        name: str,
        path: str | Path,
        artifact_type: str = "profile",
    ) -> None:
        """Attach a file or directory to the run as an artifact.

        Backend-specific: WandB uploads it as a run artifact (the point of
        WandB is to store run outputs); TensorBoard and Null leave it on
        disk and print where it lives. Not abstract — backends that have
        no artifact store inherit the default, which prints the path.

        Parameters
        ----------
        name : str
            Artifact name (e.g. 'profile-trace').
        path : str or Path
            File or directory to attach.
        artifact_type : str
            Artifact category (e.g. 'profile', 'manifest').
        """
        print(f"[logger] artifact '{name}' ({artifact_type}) at: {path}")

    @abstractmethod
    def finalize(self, status: str = "success") -> None:
        """Close the logger and mark the run complete.

        Parameters
        ----------
        status : str
            Run outcome. Only the explicit failure words
            ('failed'/'failure'/'crashed'/'error', case-insensitive) mark the
            run as failed; any other value (e.g. 'success', 'completed') is
            treated as a clean finish. See ``_is_failure_status``.
        """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FAILURE_STATUSES = frozenset({"failed", "failure", "crashed", "error"})


def _is_failure_status(status: str) -> bool:
    """True only for explicit failure words (case-insensitive).

    Failure is the special case, so any positive/neutral status — 'success',
    'completed', 'done', … — is treated as a clean finish. This avoids
    silently marking a successful run as failed just because the caller
    passed a synonym of 'success'.
    """
    return str(status).strip().lower() in _FAILURE_STATUSES


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce an image array to HWC uint8, handling greyscale and float32."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.dtype != np.uint8:
        lo, hi = float(image.min()), float(image.max())
        if hi <= 1.0 and lo >= 0.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _save_hparams_json(log_dir: Path, hparams: dict) -> None:
    """Write hparams to log_dir/hparams.json (once — does not overwrite)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "hparams.json"
    if not path.exists():
        with open(path, "w") as f:
            json.dump(hparams, f, indent=4, default=str)


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------

class WandbLogger(BaseLogger):
    """
    Weights & Biases logger (primary backend).

    Supports scalar metrics, matplotlib figures, image arrays, histograms,
    and hyperparameter logging. Run artefacts are stored both remotely and
    locally under base_log_dir.

    Parameters
    ----------
    project : str
        WandB project name.
    name : str, optional
        Run display name. Auto-generated by WandB if not given.
    base_log_dir : str or Path
        Local root directory for wandb run files. Default ``'logs/'``.
    config : dict, optional
        Hyperparameter config passed to wandb at init.
    tags : list[str], optional
        WandB run tags for filtering in the UI.
    notes : str, optional
        Free-text run description.
    offline : bool
        Run in offline mode (sync manually with ``wandb sync`` later).
    **wandb_kwargs
        Any additional kwargs forwarded verbatim to ``wandb.init``.

    Example
    -------
    >>> logger = WandbLogger(project='tc-reconstruction', name='siren-v1',
    ...                      config={'lr': 1e-3, 'n_layers': 5})
    >>> logger.log_metrics({'train/loss': 0.03, 'val/loss': 0.04}, step=10)
    >>> logger.log_figure('val/wind_field', fig, step=10)
    >>> logger.log_histogram('params/layer0', weights, step=10)
    >>> logger.finalize('success')
    """

    def __init__(
        self,
        project:      str,
        name:         str | None = None,
        base_log_dir: str | Path = "logs/",
        config:       dict | None = None,
        tags:         list[str] | None = None,
        notes:        str | None = None,
        offline:      bool = False,
        api_key:      str | None = None, # for the love of god do not push your api keys to GitHub!!!!!!
        **wandb_kwargs,
    ) -> None:
        try:
            import wandb as _wandb
        except ImportError as e:
            raise ImportError(
                "wandb is required for WandbLogger. "
                "Install it with: pip install wandb"
            ) from e

        self._wandb = _wandb

        # api_key is an escape hatch for environments where the WANDB_API_KEY
        # env var cannot be set.  Prefer the env var — never commit a key to a
        # config file that lives in version control.
        if api_key is not None:
            _wandb.login(key=api_key)

        self._run = _wandb.init(
            project=project,
            name=name,
            dir=str(base_log_dir),
            config=config or {},
            tags=tags,
            notes=notes,
            mode="offline" if offline else "online",
            **wandb_kwargs,
        )
        self._log_dir = Path(self._run.dir)
        if config:
            _save_hparams_json(self._log_dir, config)

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        self._wandb.log({k: float(v) for k, v in metrics.items()}, step=step)

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        self._run.config.update(hparams)
        _save_hparams_json(self._log_dir, hparams)

    def log_figure(self, tag: str, figure, step: int) -> None:
        self._wandb.log({tag: self._wandb.Image(figure)}, step=step)
        import matplotlib.pyplot as plt
        plt.close(figure)

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        self._wandb.log(
            {tag: self._wandb.Image(_to_hwc_uint8(image))}, step=step
        )

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        self._wandb.log(
            {tag: self._wandb.Histogram(np.asarray(values).ravel())},
            step=step,
        )

    def log_artifact(
        self,
        name: str,
        path: str | Path,
        artifact_type: str = "profile",
    ) -> None:
        artifact = self._wandb.Artifact(name, type=artifact_type)
        path = Path(path)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self._run.log_artifact(artifact)
        print(
            f"[logger] '{name}' uploaded to WandB as a '{artifact_type}' "
            f"artifact. (XLA traces need TensorBoard's Profile plugin to "
            f"view — download the artifact and run tensorboard --logdir "
            f"on it.)"
        )

    def finalize(self, status: str = "success") -> None:
        self._run.finish(exit_code=1 if _is_failure_status(status) else 0)


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------

class TensorBoardLogger(BaseLogger):
    """
    TensorBoard logger (cluster / offline fallback).

    Uses ``torch.utils.tensorboard.SummaryWriter``, the most stable
    TensorBoard writer available without TensorFlow. PyTorch does not need
    to be used for training — only the writer is imported here.

    Parameters
    ----------
    log_dir : str or Path
        Directory for this specific run. Created if it does not exist.
    config : dict, optional
        Hyperparameter config written to hparams.json and the HParams plugin.
    flush_secs : int
        How often TensorBoard flushes pending events to disk. Default 30.

    Example
    -------
    >>> logger = TensorBoardLogger('logs/siren-v1', config={'lr': 1e-3})
    >>> logger.log_metrics({'train/loss': 0.03}, step=10)
    >>> logger.log_figure('val/wind_field', fig, step=10)
    >>> logger.finalize('success')
    """

    def __init__(
        self,
        log_dir:    str | Path,
        config:     dict | None = None,
        flush_secs: int = 30,
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for TensorBoardLogger. "
                "Install it with: pip install torch"
            ) from e

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(
            log_dir=str(self._log_dir),
            flush_secs=flush_secs,
        )
        if config:
            _save_hparams_json(self._log_dir, config)
            self._log_hparams_plugin(config)

    def _log_hparams_plugin(self, hparams: dict) -> None:
        """Push flat scalar hparams to TensorBoard's HParams plugin."""
        flat = {
            k: v for k, v in hparams.items()
            if isinstance(v, (int, float, str, bool))
        }
        self._writer.add_hparams(flat, metric_dict={})

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, float(value), global_step=step)

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        _save_hparams_json(self._log_dir, hparams)
        self._log_hparams_plugin(hparams)

    def log_figure(self, tag: str, figure, step: int) -> None:
        # add_figure closes the figure when close=True
        self._writer.add_figure(tag, figure, global_step=step, close=True)

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        # TensorBoard expects CHW layout
        chw = np.moveaxis(_to_hwc_uint8(image), -1, 0)
        self._writer.add_image(tag, chw, global_step=step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        self._writer.add_histogram(
            tag, np.asarray(values).ravel(), global_step=step
        )

    def log_artifact(
        self,
        name: str,
        path: str | Path,
        artifact_type: str = "profile",
    ) -> None:
        # TensorBoard has no artifact store — the files stay on disk.
        # XLA profile traces are viewable directly in this backend's UI.
        print(
            f"[logger] artifact '{name}' ({artifact_type}) at: {path}"
            + (f" — view with: tensorboard --logdir {path}"
               if artifact_type == "profile" else "")
        )

    def finalize(self, status: str = "success") -> None:
        self._writer.flush()
        self._writer.close()


# ---------------------------------------------------------------------------
# Null logger — no external dependencies
# ---------------------------------------------------------------------------

class NullLogger(BaseLogger):
    """
    No-op logger that prints to stdout and saves figures/images to disk.

    No WandB or TensorBoard dependency. Intended for unit tests, dry runs,
    and local debugging where remote logging is not needed. Metrics are
    printed; figures and images are saved as PNGs under log_dir.

    Parameters
    ----------
    log_dir : str or Path
        Directory for hparams.json, figures/, and images/. Created if
        it does not exist. Default ``'logs/debug'``.
    config : dict, optional
        Hyperparameter config written to hparams.json.
    verbose : bool
        If True (default), print metrics and histogram stats to stdout.

    Example
    -------
    >>> logger = NullLogger('logs/debug', verbose=True)
    >>> logger.log_metrics({'train/loss': 0.1}, step=1)
    [step=1] train/loss=0.1000
    >>> logger.log_histogram('params/w0', weights, step=1)
    [step=1] histogram params/w0: mean=0.0012  std=0.2310  min=-0.9 max=0.8
    """

    def __init__(
        self,
        log_dir: str | Path = "logs/debug",
        config:  dict | None = None,
        verbose: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._verbose = verbose
        if config:
            _save_hparams_json(self._log_dir, config)

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        if self._verbose:
            parts = "  ".join(
                f"{k}={float(v):.4f}" for k, v in metrics.items()
            )
            print(f"[step={step:>6}] {parts}")

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        _save_hparams_json(self._log_dir, hparams)

    def log_figure(self, tag: str, figure, step: int) -> None:
        import matplotlib.pyplot as plt
        out_dir = self._log_dir / "figures"
        out_dir.mkdir(exist_ok=True)
        safe_tag = tag.replace("/", "_")
        path = out_dir / f"{safe_tag}_step{step:06d}.png"
        figure.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(figure)
        if self._verbose:
            print(f"[step={step:>6}] figure -> {path}")

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        image = _to_hwc_uint8(image)
        out_dir = self._log_dir / "images"
        out_dir.mkdir(exist_ok=True)
        safe_tag = tag.replace("/", "_")
        path = out_dir / f"{safe_tag}_step{step:06d}.png"
        try:
            from PIL import Image as _PILImage
            arr = image[:, :, 0] if image.shape[-1] == 1 else image
            _PILImage.fromarray(arr).save(path)
        except ImportError:
            # Pillow not available — skip file write, still print
            pass
        if self._verbose:
            print(f"[step={step:>6}] image  -> {path}  shape={image.shape}")

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        if self._verbose:
            v = np.asarray(values).ravel()
            print(
                f"[step={step:>6}] histogram {tag}: "
                f"mean={v.mean():.4f}  std={v.std():.4f}  "
                f"min={v.min():.4f}  max={v.max():.4f}"
            )

    def log_artifact(
        self,
        name: str,
        path: str | Path,
        artifact_type: str = "profile",
    ) -> None:
        if self._verbose:
            print(f"[logger] artifact '{name}' ({artifact_type}) at: {path}")

    def finalize(self, status: str = "success") -> None:
        if self._verbose:
            print(f"[logger] finalized - status: {status}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_logger(
    backend: str,
    log_dir: str | Path | None = None,
    config:  dict | None = None,
    **kwargs,
) -> BaseLogger:
    """
    Instantiate a logger by backend name.

    Parameters
    ----------
    backend : str
        One of ``'wandb'``, ``'tensorboard'``, ``'null'``.
    log_dir : str or Path, optional
        Run directory. Required for ``'tensorboard'`` and ``'null'``.
        Ignored for ``'wandb'`` (WandB manages its own directory tree).
    config : dict, optional
        Hyperparameter config forwarded to the logger.
    **kwargs
        Forwarded to the logger constructor.
        wandb:       ``project``, ``name``, ``tags``, ``offline``, …
        tensorboard: ``flush_secs``
        null:        ``verbose``

    Returns
    -------
    BaseLogger

    Raises
    ------
    ValueError
        If backend is not one of the three accepted values.

    Example
    -------
    >>> logger = create_logger('wandb', project='tc-recon', config=hparams)
    >>> logger = create_logger('tensorboard', log_dir='logs/run_01', config=hparams)
    >>> logger = create_logger('null', log_dir='logs/debug', verbose=False)
    """
    backend = backend.lower().strip()
    if backend == "wandb":
        # Honour log_dir as base_log_dir so WandB writes under the run directory
        # when run_dir is set in the trainer config.  Explicit base_log_dir in
        # log_kwargs always wins.
        if log_dir is not None and "base_log_dir" not in kwargs:
            kwargs = {"base_log_dir": log_dir, **kwargs}
        return WandbLogger(config=config, **kwargs)
    elif backend == "tensorboard":
        if log_dir is None:
            raise ValueError("log_dir is required for TensorBoardLogger.")
        return TensorBoardLogger(log_dir=log_dir, config=config, **kwargs)
    elif backend in ("null", "none"):
        return NullLogger(log_dir=log_dir or "logs/debug", config=config, **kwargs)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Choose from 'wandb', 'tensorboard', 'null'."
        )
