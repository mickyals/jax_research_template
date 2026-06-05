"""
experiments/sparse_obs_cross_attn/train.py

Entry point for the sparse_obs_cross_attn experiment.

Usage
-----
    python -m experiments.sparse_obs_cross_attn.train \
        src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml

    # Resume an interrupted run
    python -m experiments.sparse_obs_cross_attn.train \
        src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
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

from experiments.sparse_obs_cross_attn.datamodule import TCDataModule
from experiments.sparse_obs_cross_attn.evaluate import plot_attention_geographic
from experiments.sparse_obs_cross_attn.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.model import TCClassifier
from training.trainer import Trainer, TrainState


# ---------------------------------------------------------------------------
# Attention entropy callback
# ---------------------------------------------------------------------------

def _make_attn_entropy_callback(
    model:       TCClassifier,
    probe_batch: dict,
    logger,
    data_config: dict,
    fig_every:   int = 5,
) -> Callable[[TrainState, int, int], None]:
    """Return an epoch callback that logs cross-attention entropy and maps.

    Both metrics are keyed under ``val/`` because the probe batch comes from
    the validation set.  All metrics use ``step=epoch`` so they share the
    same x-axis as the epoch-level train/val loss curves.

    A falling entropy curve means the model is learning to concentrate on
    specific stations rather than distributing attention uniformly.

    Parameters
    ----------
    model : TCClassifier
    probe_batch : dict
        A fixed validation batch held constant across all epochs.
    logger : experiment logger
        Must expose ``log_metrics`` and ``log_figure``.
    data_config : dict
        The ``data:`` block from the YAML config — used to recover
        location_encoding, fov_lat, fov_lon, radius_km for figure rendering.
    fig_every : int
        Log an attention geographic figure every this many epochs.
        0 = never. Default 5.
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
        return weights  # (B, H, N)

    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        weights = np.asarray(_attn(state.params))          # (B, H, N)
        # Padding positions have w ≈ 0 after masked softmax;
        # their contribution to entropy is negligible.
        entropy = float(
            -np.sum(weights * np.log(weights + 1e-12), axis=-1).mean()
        )
        logger.log_metrics({'val/attn_entropy': entropy}, step=epoch)

        if fig_every > 0 and epoch % fig_every == 0:
            fig = plot_attention_geographic(
                weights, probe_batch,
                location_encoding=loc_enc,
                fov_lat=fov_lat,
                fov_lon=fov_lon,
                radius_km=radius_km,
                sample_idx=0,
            )
            logger.log_figure('val/attn_map', fig, step=epoch)

    return callback


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _validate_config(config: dict) -> None:
    """Catch cross-block inconsistencies the JSON schema cannot enforce."""
    data_enc  = config['data'].get('location_encoding', 'unit_circle')
    model_enc = config['model'].get('location_encoding', 'unit_circle')
    if data_enc != model_enc:
        raise ValueError(
            f"data.location_encoding='{data_enc}' and "
            f"model.location_encoding='{model_enc}' must match."
        )

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
    config = _load_config(config_path)
    _validate_config(config)

    # Single top-level seed propagated to all components
    seed = int(config.get('seed', 42))

    # Inject seed into trainer block (Trainer reads config['seed'])
    trainer_cfg = config['trainer']
    trainer_cfg.setdefault('seed', seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dm           = TCDataModule.from_config(config['data'])
    train_loader = dm.train_loader(seed=seed, shuffle=True)
    val_loader   = dm.val_loader()
    test_loader  = dm.test_loader()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TCClassifier(**config['model'])

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    metrics_fns = build_metrics_fns()
    trainer     = Trainer(model, metrics_fns, trainer_cfg)

    # Log full config so every run is reproducible from its artifact
    trainer.log_hyperparams(config)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    probe_batch = next(iter(val_loader))
    callbacks   = [
        _make_attn_entropy_callback(model, probe_batch, trainer.logger,
                                    data_config=config['data']),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    best_state = trainer.fit(train_loader, val_loader, resume=resume,
                             epoch_callbacks=callbacks)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    test_metrics = trainer.test(test_loader)
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.5f}")


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
