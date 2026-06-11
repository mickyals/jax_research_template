"""
experiments/sparse_obs_cross_attn/train/train.py

Entry point for the sparse_obs_cross_attn experiment.

Usage
-----
    python -m experiments.sparse_obs_cross_attn.train.train \
        jrt/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml

    # Resume an interrupted run
    python -m experiments.sparse_obs_cross_attn.train.train \
        jrt/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
        --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import yaml

from experiments.sparse_obs_cross_attn.data.datamodule import TCDataModule
from experiments.sparse_obs_cross_attn.train.evaluate import (
    CLASS_NAMES,
    collect_predictions,
    confusion_matrix,
    per_class_metrics,
)
from experiments.sparse_obs_cross_attn.plotting.plotting import (
    plot_attention_geographic,
    plot_class_metrics,
    plot_confusion_matrix,
)
from experiments.sparse_obs_cross_attn.train.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.train.model import TCClassifier
from training.trainer import Trainer, TrainState
from utils.jax_core.diagnostics import model_tabulate


# ---------------------------------------------------------------------------
# Attention entropy callback
# ---------------------------------------------------------------------------

def _make_attn_entropy_callback(
    model:       TCClassifier,
    probe_batch: dict,
    logger,
) -> Callable[[TrainState, int, int], None]:
    """Return a **step-level** callback that logs attention entropy.

    Intended for use with ``Trainer.fit(step_callbacks=[(fn, every_n_steps)])``.
    Uses ``step=global_step`` so the entropy curve is plotted on the same
    x-axis as the step-level training loss in WandB.

    A falling curve means the model is concentrating on fewer stations rather
    than spreading attention uniformly — a useful proxy for learning progress.

    Parameters
    ----------
    model : TCClassifier
    probe_batch : dict
        A fixed validation batch held in memory for the run duration.
    logger : experiment logger
        Must expose ``log_metrics``.
    """
    probe_X = probe_batch['X']

    @jax.jit
    def _attn(params):
        _, weights = model.apply({'params': params}, probe_X,
                                 train=False, return_weights=True)
        return weights  # (B, H, N+1)

    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        weights = np.asarray(_attn(state.params))          # (B, H, N+1)
        # Padding positions carry ~0 weight after the masked softmax —
        # their contribution to entropy is negligible.
        entropy = float(
            -np.sum(weights * np.log(weights + 1e-12), axis=-1).mean()
        )
        # Use global_step so the curve aligns with step-level train/loss.
        logger.log_metrics({'val/attn_entropy': entropy}, step=global_step)

    return callback


def _make_attn_figure_callback(
    model:       TCClassifier,
    probe_batch: dict,
    logger,
    data_config: dict,
    fig_every:   int = 5,
) -> Callable[[TrainState, int, int], None]:
    """Return an **epoch-level** callback that logs geographic attention maps.

    Intended for use with ``Trainer.fit(epoch_callbacks=[fn])``.
    Uses ``step=epoch`` to align figures with the epoch-level val metrics.

    Parameters
    ----------
    model : TCClassifier
    probe_batch : dict
        A fixed validation batch held in memory for the run duration.
    logger : experiment logger
        Must expose ``log_figure``.
    data_config : dict
        The ``data:`` block from the YAML config — supplies location_encoding,
        fov_lat, fov_lon, radius_km for geographic rendering.
    fig_every : int
        Log a figure every this many epochs. 0 = never. Default 5.
    """
    probe_X   = probe_batch['X']
    loc_enc   = data_config.get('location_encoding', 'unit_circle')
    fov_lat   = data_config.get('fov_lat')
    fov_lon   = data_config.get('fov_lon')
    radius_km = float(data_config.get('radius_km', 500.0))

    @jax.jit
    def _attn(params):
        _, weights = model.apply({'params': params}, probe_X,
                                 train=False, return_weights=True)
        return weights  # (B, H, N+1)

    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        if fig_every <= 0 or epoch % fig_every != 0:
            return
        weights = np.asarray(_attn(state.params))
        fig = plot_attention_geographic(
            weights, probe_batch,
            location_encoding=loc_enc,
            fov_lat=fov_lat,
            fov_lon=fov_lon,
            radius_km=radius_km,
            sample_idx=0,
        )
        # Use global_step so the map aligns with all other metrics in WandB.
        logger.log_figure('val/attn_map', fig, step=global_step)

    return callback


def _make_eval_plots_callback(
    model:       TCClassifier,
    val_loader,
    logger,
    every_n_epochs: int = 1,
) -> Callable[[TrainState, int, int], None]:
    """Return an epoch-level callback that logs confusion matrix and per-class F1.

    Runs a full forward pass over the val loader to collect predictions, then
    plots and uploads to WandB as images. Appears under Media → Images.

    Parameters
    ----------
    model : TCClassifier
    val_loader : TCLoader
        Re-iterable val loader — iterated fresh on each callback invocation.
    logger : experiment logger
        Must expose ``log_figure``.
    every_n_epochs : int
        How often to run. 0 = disabled. Default 1 (every epoch).
    """
    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        if every_n_epochs <= 0 or epoch % every_n_epochs != 0:
            return

        variables = {'params': state.params}
        preds, labels, _ = collect_predictions(model, variables, val_loader)

        cm  = confusion_matrix(preds, labels)
        pcm = per_class_metrics(cm)

        fig_norm = plot_confusion_matrix(
            cm, CLASS_NAMES, normalize=True,
            title=f'Val confusion matrix — epoch {epoch} (recall per class)',
        )
        fig_raw = plot_confusion_matrix(
            cm, CLASS_NAMES, normalize=False,
            title=f'Val confusion matrix — epoch {epoch} (counts)',
        )
        fig_cls = plot_class_metrics(
            pcm, CLASS_NAMES,
        )

        logger.log_figure('val/confusion_norm',    fig_norm, step=global_step)
        logger.log_figure('val/confusion_counts',  fig_raw,  step=global_step)
        logger.log_figure('val/per_class_metrics', fig_cls,  step=global_step)

    return callback


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def _validate_config(config: dict) -> None:
    """Catch config inconsistencies that the JSON schema cannot enforce."""
    n_obs_cfg = config['model'].get('n_obs_features')
    obs_vars  = config['data'].get('obs_vars', [])
    if n_obs_cfg is not None and obs_vars and n_obs_cfg != len(obs_vars):
        raise ValueError(
            f"model.n_obs_features={n_obs_cfg} does not match "
            f"len(data.obs_vars)={len(obs_vars)}."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(config_path: str | Path, resume: bool = False) -> None:
    """Run training from a YAML config file.

    Parameters
    ----------
    config_path : str or Path
        Path to tc_classifier.yaml (or any config following the same schema).
    resume : bool
        If True, resume from the latest checkpoint in trainer.checkpoint_dir.
    """
    # Resolve config path immediately so relative paths inside the config
    # can be anchored to the config file's own directory.
    config_path = Path(config_path).resolve()
    config      = _load_config(config_path)
    _validate_config(config)

    # Single top-level seed propagated to all components
    seed = int(config.get('seed', 42))

    # Inject seed into trainer block (Trainer reads config['seed'])
    trainer_cfg = config['trainer']
    trainer_cfg.setdefault('seed', seed)

    # ------------------------------------------------------------------
    # Top-level shared values — single source of truth, propagated down.
    # ------------------------------------------------------------------

    # batch_size lives in trainer: (training hyperparam); data loader reads it here.
    config['data']['batch_size'] = trainer_cfg['batch_size']

    # location_encoding is an experiment-design choice that affects both data
    # preprocessing (coordinate encoding) and model architecture (query token
    # construction).  Defined once at the top level; injected into both blocks.
    loc_enc = config.get('location_encoding', 'unit_circle')
    config['data']['location_encoding']  = loc_enc
    config['model']['location_encoding'] = loc_enc

    # ------------------------------------------------------------------
    # Resolve run_dir relative to the experiment root (two levels up from
    # this script, which lives in train/), NOT the config file directory
    # (configs/).  This ensures that runs/tc_classifier/run_01 always
    # expands to
    #   <experiment_dir>/runs/tc_classifier/run_01
    # regardless of where the CLI is invoked from.  Absolute paths are used
    # as-is.
    # ------------------------------------------------------------------
    _experiment_dir = Path(__file__).resolve().parent.parent
    if 'run_dir' in trainer_cfg:
        rd = Path(trainer_cfg['run_dir'])
        if not rd.is_absolute():
            trainer_cfg['run_dir'] = str(_experiment_dir / rd)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dm = TCDataModule.from_config(config['data'])

    steps_per_epoch = trainer_cfg.get('steps_per_epoch')
    dm.summary(steps_per_epoch=steps_per_epoch)

    train_loader = dm.train_loader(
        seed            = seed,
        shuffle         = True,
        steps_per_epoch = steps_per_epoch,
    )
    val_loader   = dm.val_loader()
    test_loader  = dm.test_loader()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TCClassifier(**config['model'])

    # Print a per-layer shape + parameter count table before training.
    # Peek a small batch (4 samples) so Flax can trace all tensor shapes.
    _peek  = next(iter(train_loader))
    _exmp  = {k: v[:4] for k, v in _peek['X'].items()}
    print()
    print("─" * 58)
    print("Model  (TCClassifier)")
    model_tabulate(model, _exmp, False)   # args: X dict, train=False
    del _peek, _exmp

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    metrics_fns = build_metrics_fns()
    trainer     = Trainer(model, metrics_fns, trainer_cfg)

    # Log full config so every run is reproducible from its artifact
    trainer.log_hyperparams(config)

    # Persist the resolved data split (manifest.json next to checkpoints +
    # logger copy) — the durable answer to "what did this run train on"
    trainer.write_manifest(dm.manifest())

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    probe_batch = next(iter(val_loader))

    # Attention entropy — logged every N steps inside the training loop.
    # Frequency is read from trainer.attn_log_every_n_steps; falls back to
    # trainer.log_every_n_steps so it matches the loss logging cadence by
    # default without requiring a separate config key.
    attn_step_every = trainer_cfg.get(
        'attn_log_every_n_steps',
        trainer_cfg.get('log_every_n_steps', 50),
    )
    step_callbacks = [
        (_make_attn_entropy_callback(model, probe_batch, trainer.logger),
         attn_step_every),
    ]

    # Geographic attention map — logged once per epoch (expensive render).
    # fig_every controls how many epochs between maps; 0 = disabled.
    fig_every       = trainer_cfg.get('attn_fig_every_n_epochs', 5)

    # Confusion matrix + per-class F1 over the full val set.
    # Runs a full val forward pass so keep infrequent for long runs.
    eval_plots_every = trainer_cfg.get('eval_plots_every_n_epochs', 1)

    epoch_callbacks = [
        _make_attn_figure_callback(model, probe_batch, trainer.logger,
                                   data_config=config['data'],
                                   fig_every=fig_every),
        _make_eval_plots_callback(model, val_loader, trainer.logger,
                                  every_n_epochs=eval_plots_every),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    best_state = trainer.fit(
        train_loader, val_loader,
        resume          = resume,
        epoch_callbacks = epoch_callbacks,
        step_callbacks  = step_callbacks,
    )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    test_metrics = trainer.test(test_loader)
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.5f}")

    # Finalize logger here, after test(), so test metrics are logged before
    # the WandB run is closed.  fit() no longer calls finalize() internally.
    trainer.logger.finalize("completed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train TCClassifier for sparse_obs_cross_attn experiment."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train(args.config, resume=args.resume)
