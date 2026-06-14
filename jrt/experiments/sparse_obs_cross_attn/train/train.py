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
    plot_attention_mask,
    plot_attention_matrix_grid,
    plot_class_metrics,
    plot_confusion_matrix,
)
from experiments.sparse_obs_cross_attn.train.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.train.model import TCClassifier
from training.metrics import (
    cross_entropy,
    expected_calibration_error,
    quadratic_weighted_kappa,
)
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
        # Query row of the LAST layer — preserves the metric's original
        # definition now that weights cover all layers.
        return weights[-1][:, :, -1, :]  # (B, H, N+1)

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
    """Return an **epoch-level** callback that logs attention figures.

    Two figures per invocation, both from the fixed VAL probe batch:
    ``val/attn_map`` (geographic query-row attention, last layer) and
    ``val/attn_grid`` (layers × heads grid of full attention matrices).

    Intended for use with ``Trainer.fit(epoch_callbacks=[fn])``.

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
        Log figures every this many epochs. 0 = never. Default 5.
    """
    probe_X   = probe_batch['X']
    loc_enc   = data_config.get('location_encoding', 'unit_circle')
    fov_lat   = data_config.get('fov_lat')
    fov_lon   = data_config.get('fov_lon')
    radius_km = float(data_config.get('radius_km', 500.0))

    @jax.jit
    def _attn(params):
        logits, weights = model.apply({'params': params}, probe_X,
                                      train=False, return_weights=True)
        return logits, weights  # (B, C), (L, B, H, N+1, N+1)

    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        if fig_every <= 0 or epoch % fig_every != 0:
            return
        logits, weights = _attn(state.params)
        weights = np.asarray(weights)               # (L, B, H, N+1, N+1)
        true_c  = CLASS_NAMES[int(probe_batch['y'][0])]
        pred_c  = CLASS_NAMES[int(np.asarray(logits)[0].argmax())]
        title   = f'true: {true_c}, pred: {pred_c}'

        fig = plot_attention_geographic(
            weights[-1][:, :, -1, :], probe_batch,   # last layer, query row
            location_encoding=loc_enc,
            fov_lat=fov_lat,
            fov_lon=fov_lon,
            radius_km=radius_km,
            sample_idx=0,
        )
        # Keep the caption inside the figure (wandb.Image renders at the
        # figure's own bounds, so a y>1.0 suptitle would be clipped) and
        # reserve top margin so it clears the axes title.
        fig.suptitle(title, y=0.99, fontsize=10)
        fig.subplots_adjust(top=0.86)
        # Use global_step so the map aligns with all other metrics in WandB.
        logger.log_figure('val/attn_map', fig, step=global_step)

        fig_grid = plot_attention_matrix_grid(
            weights, sample_idx=0,
            title=f'Attention matrices — {title}',
        )
        logger.log_figure('val/attn_grid', fig_grid, step=global_step)

    return callback


# ---------------------------------------------------------------------------
# Gradient-flow callback (TRAIN probe only)
# ---------------------------------------------------------------------------

def _make_grad_flow_callback(
    model:       TCClassifier,
    probe_batch: dict,
    logger,
    every_n_epochs: int = 5,
) -> Callable[[TrainState, int, int], None]:
    """Return an epoch-level callback that logs per-layer gradient histograms.

    Computes jax.grad of the cross-entropy loss on a fixed TRAIN probe
    batch and pushes one histogram per parameter leaf, named by its tree
    path (e.g. ``grad_flow/encoder/blocks_0/attn/query/kernel``), via
    ``logger.log_histogram``. Vanishing/exploding layers show up as
    histograms collapsing to 0 or blowing up across depth.

    Call once manually with the freshly initialised state for the init
    snapshot, then register as an epoch callback.

    Parameters
    ----------
    model : TCClassifier
    probe_batch : dict
        A fixed TRAINING batch held in memory for the run duration
        (gradient flow is a training diagnostic — never wired to val/test).
    logger : experiment logger
        Must expose ``log_histogram``.
    every_n_epochs : int
        Log every this many epochs. 0 = never. Default 5.
    """
    probe_X = probe_batch['X']
    probe_y = probe_batch['y']

    @jax.jit
    def _grads(params):
        def loss_fn(p):
            logits = model.apply({'params': p}, probe_X, train=False)
            return cross_entropy(logits, probe_y)
        return jax.grad(loss_fn)(params)

    def _log(params, step: int) -> None:
        grads = _grads(params)
        leaves = jax.tree_util.tree_flatten_with_path(grads)[0]
        for path, leaf in leaves:
            name = '/'.join(
                getattr(k, 'key', getattr(k, 'name', str(k))) for k in path
            )
            logger.log_histogram(
                f'grad_flow/{name}', np.asarray(leaf).ravel(), step=step,
            )

    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        if every_n_epochs <= 0 or epoch % every_n_epochs != 0:
            return
        _log(state.params, global_step)

    callback.log_now = _log   # exposed for the init-time snapshot
    return callback


def _make_eval_plots_callback(
    model:       TCClassifier,
    val_loader,
    logger,
    every_n_epochs: int = 1,
) -> Callable[[TrainState, int, int], None]:
    """Return an epoch-level callback that logs confusion matrix and per-class F1.

    Runs a full forward pass over the val loader to collect predictions, then
    plots and uploads to WandB as images (appears under Media → Images) and
    logs full-set scalars ``val/qwk`` (quadratic-weighted kappa — ordinal
    agreement) and ``val/ece`` (expected calibration error), both from
    metrics.py over the accumulated predictions/logits.

    Parameters
    ----------
    model : TCClassifier
    val_loader : TCLoader
        Re-iterable val loader — iterated fresh on each callback invocation.
    logger : experiment logger
        Must expose ``log_figure`` and ``log_metrics``.
    every_n_epochs : int
        How often to run. 0 = disabled. Default 1 (every epoch).
    """
    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        if every_n_epochs <= 0 or epoch % every_n_epochs != 0:
            return

        variables = {'params': state.params}
        preds, labels, logits, _ = collect_predictions(model, variables, val_loader)

        cm  = confusion_matrix(preds, labels)
        pcm = per_class_metrics(cm)

        probs = np.asarray(jax.nn.softmax(jnp.array(logits), axis=-1))
        qwk   = quadratic_weighted_kappa(cm)
        ece   = expected_calibration_error(probs, labels)
        logger.log_metrics({'val/qwk': qwk, 'val/ece': ece}, step=global_step)

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

    # location_encoding picks the coordinate convention for the datamodule's
    # encoder. The model is coordinate-agnostic (Senseiver single projection of
    # whatever coords it is handed), so it is injected into the data block only.
    loc_enc = config.get('location_encoding', 'unit_circle')
    config['data']['location_encoding']  = loc_enc

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
    # train_probe_batch doubles as the fixed probe for the gradient-flow
    # callback below (gradient flow is a TRAIN diagnostic).
    train_probe_batch = next(iter(train_loader))
    _exmp  = {k: v[:4] for k, v in train_probe_batch['X'].items()}
    print()
    print("─" * 58)
    print("Model  (TCClassifier)")
    model_tabulate(model, _exmp, False)   # args: X dict, train=False
    del _exmp

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    metrics_fns = build_metrics_fns(
        loss        = trainer_cfg.get('loss', 'cross_entropy'),
        loss_kwargs = trainer_cfg.get('loss_kwargs'),
    )
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

    # Gradient-flow histograms — TRAIN probe batch only, logged at init
    # and every grad_hist_every_n_epochs epochs.
    grad_hist_every = trainer_cfg.get('grad_hist_every_n_epochs', 5)
    grad_flow_cb    = _make_grad_flow_callback(
        model, train_probe_batch, trainer.logger,
        every_n_epochs=grad_hist_every,
    )

    epoch_callbacks = [
        _make_attn_figure_callback(model, probe_batch, trainer.logger,
                                   data_config=config['data'],
                                   fig_every=fig_every),
        _make_eval_plots_callback(model, val_loader, trainer.logger,
                                  every_n_epochs=eval_plots_every),
        grad_flow_cb,
    ]

    # One-off static figures + init-time gradient snapshot.
    # The mask figure documents the attention pattern for this run's probe
    # sample — it never changes during training.
    mask_fig = plot_attention_mask(
        np.asarray(probe_batch['X']['station_mask'][0]),
        full_self_attention=config['model'].get('full_self_attention', False),
    )
    trainer.logger.log_figure('val/attn_mask', mask_fig, step=0)

    if grad_hist_every > 0:
        # fit() re-creates the state with the same seed, so this initial
        # state's gradients are the true step-0 snapshot.
        init_state = trainer.init_state(train_probe_batch)
        grad_flow_cb.log_now(init_state.params, step=0)
        del init_state

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
