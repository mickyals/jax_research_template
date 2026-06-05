"""
experiments/sparse_obs_cross_attn/evaluate.py

Post-training evaluation and visualisation for the sparse-obs
cross-attention TC intensity classifier.

Functions
---------
load_model(checkpoint_dir, config)
    Restore TCClassifier from an orbax checkpoint.

attention_map(sample, attn_weights, ax, title)
    Plot one sample: storm centre + station dots coloured by
    mean attention weight on a lat/lon scatter axes.

confusion_matrix_fig(true_labels, pred_labels, n_classes)
    Return a matplotlib Figure of the ordinal confusion matrix.

run_attention_maps(config_path, checkpoint_dir, split, n, out_dir)
    Main evaluation loop: loads model, draws n samples from split,
    produces and saves one attention map per sample.

run_confusion_matrix(config_path, checkpoint_dir, split, out_dir)
    Evaluates the full split and saves a confusion matrix figure.

Usage
-----
    python src/experiments/sparse_obs_cross_attn/evaluate.py \
        --config  src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml \
        --ckpt    checkpoints/tc_classifier/best \
        --split   val \
        --n       8 \
        --out_dir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import orbax.checkpoint as ocp
import yaml

from datasets.joint.dataset import N_CLASSES, SSHS_TO_CLASS, OBS_DIM
from experiments.sparse_obs_cross_attn.datamodule import JointDataModule
from experiments.sparse_obs_cross_attn.model import TCClassifier
from training.ordinal_loss import ordinal_predict, ordinal_probs
from utils.jax_core.helpers import create_rng, create_rng_dict


# ---------------------------------------------------------------------------
# Class labels for plots
# ---------------------------------------------------------------------------

_CLASS_LABELS = [
    'No storm', 'Disturbance', 'Subtropical',
    'TD', 'TS', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5',
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_dir: str | Path,
    config:         dict,
    dummy_max_stations: int = 64,
) -> tuple[TCClassifier, dict]:
    """Restore TCClassifier parameters from an orbax checkpoint.

    Parameters
    ----------
    checkpoint_dir : path to the orbax checkpoint directory (e.g. best/).
    config : full experiment config dict.
    dummy_max_stations : used only to build the dummy batch for init shapes.

    Returns
    -------
    (model, params)
    """
    model = TCClassifier(**config.get('model', {}))

    # Build dummy batch to initialise parameter shapes
    max_s = int(config['data'].get('max_stations', dummy_max_stations))
    dummy_x = {
        'query_coords': jnp.ones((1, 3)),
        'station_obs':  jnp.ones((1, max_s, OBS_DIM)),
        'station_mask': jnp.ones((1, max_s), dtype=bool),
        'obs_mask':     jnp.ones((1, max_s, OBS_DIM), dtype=bool),
    }

    variables = model.init(
        create_rng_dict(0, keys=['params']),
        dummy_x,
        train=False,
    )

    # Restore
    abstract = {'params': variables['params']}
    restored = ocp.PyTreeCheckpointer().restore(str(checkpoint_dir), item=abstract)
    return model, restored['params']


# ---------------------------------------------------------------------------
# Attention map
# ---------------------------------------------------------------------------

def attention_map(
    sample:       dict,
    attn_weights: list,
    ax:           plt.Axes,
    title:        Optional[str] = None,
) -> None:
    """Plot station attention weights for one sample on geographic axes.

    Uses the final cross-attention layer's weights, averaged over heads,
    to colour each real station dot.  Padding stations are hidden.

    Parameters
    ----------
    sample : dict  output of JointDataModule.get_eval_samples()
             Must contain 'storm_lat', 'storm_lon', 'station_mask',
             and InsituLand lat/lon queried during sample assembly.
             Since we don't store station coords in the sample dict,
             this function uses the station positions retrieved from
             the insitu dataset attached to the JointTCDataset.
    attn_weights : list[jax.Array]
        Per-layer attention weights from model.forward_with_weights().
        Each element: (1, num_heads, 1, N).
    ax : matplotlib Axes
    title : str, optional
    """
    # Final layer, average over heads, squeeze batch and query dims
    # attn_weights[-1]: (1, num_heads, 1, N) → (N,)
    w = np.array(jnp.mean(attn_weights[-1][0, :, 0, :], axis=0))  # (N,)

    n_real = int(sample['n_stations'])
    mask   = np.array(sample['station_mask'])  # (max_stations,) bool

    station_lats = sample.get('station_lats')
    station_lons = sample.get('station_lons')

    storm_lat = sample['storm_lat']
    storm_lon = sample['storm_lon']

    ax.set_xlim(storm_lon - 12, storm_lon + 12)
    ax.set_ylim(storm_lat - 12, storm_lat + 12)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.axhline(storm_lat, color='grey', lw=0.5, ls='--')
    ax.axvline(storm_lon, color='grey', lw=0.5, ls='--')

    # Storm centre
    ax.plot(storm_lon, storm_lat, marker='*', color='red',
            markersize=14, zorder=5, label='Storm centre')

    # Stations coloured by attention weight
    if station_lats is not None and station_lons is not None:
        lats = np.array(station_lats)[:n_real]
        lons = np.array(station_lons)[:n_real]
        sc = ax.scatter(
            lons, lats,
            c=w[:n_real], cmap='YlOrRd',
            vmin=0.0, vmax=w[:n_real].max() + 1e-8,
            s=60, edgecolors='k', linewidths=0.4, zorder=4,
        )
        plt.colorbar(sc, ax=ax, label='Attention weight')
    else:
        # No coordinates available — annotate only
        ax.text(
            0.5, 0.5, f'{n_real} stations\n(coords not stored)',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color='grey',
        )

    true_cls = int(sample['label'])
    if title is None:
        sid = sample.get('sid', '?')
        title = f"{sid}  true={_CLASS_LABELS[true_cls]}"
    ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_fig(
    true_labels:  np.ndarray,
    pred_labels:  np.ndarray,
    n_classes:    int = N_CLASSES,
) -> plt.Figure:
    """Return a matplotlib Figure of the ordinal confusion matrix.

    Parameters
    ----------
    true_labels, pred_labels : 1-D integer arrays.
    n_classes : int

    Returns
    -------
    plt.Figure
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(true_labels, pred_labels):
        cm[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(_CLASS_LABELS[:n_classes], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(_CLASS_LABELS[:n_classes], fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Ordinal class confusion matrix')

    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,
                    color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main evaluation runners
# ---------------------------------------------------------------------------

def run_attention_maps(
    config_path:    str | Path,
    checkpoint_dir: str | Path,
    split:          str = 'val',
    n:              int = 8,
    out_dir:        str | Path = 'figures',
) -> None:
    """Load model + data, produce attention map figures for n samples."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data module…")
    dm = JointDataModule.from_config(config['data'])

    print("Loading model checkpoint…")
    model, params = load_model(checkpoint_dir, config)

    print(f"Drawing {n} samples from {split} split…")
    samples = dm.get_eval_samples(split=split, n=n, seed=0)

    for i, sample in enumerate(samples):
        # Build X dict for the model
        x = {
            k: jnp.array(sample[k][None])   # add batch dim
            for k in ('query_coords', 'station_obs', 'station_mask', 'obs_mask')
        }
        logits, attn_weights = model.apply({'params': params}, x, method=model.forward_with_weights)

        pred_cls = int(ordinal_predict(logits)[0])
        true_cls = int(sample['label'])

        fig, ax = plt.subplots(figsize=(6, 5))
        attention_map(sample, attn_weights, ax)
        ax.set_title(
            f"SID={sample.get('sid', '?')}  "
            f"True={_CLASS_LABELS[true_cls]}  "
            f"Pred={_CLASS_LABELS[pred_cls]}",
            fontsize=8,
        )
        path = out_dir / f"attn_{split}_{i:03d}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")


def run_confusion_matrix(
    config_path:    str | Path,
    checkpoint_dir: str | Path,
    split:          str = 'val',
    out_dir:        str | Path = 'figures',
) -> None:
    """Evaluate the full split and save a confusion matrix figure."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data module…")
    dm = JointDataModule.from_config(config['data'])

    print("Loading model checkpoint…")
    model, params = load_model(checkpoint_dir, config)

    eval_fn = jax.jit(
        lambda x: model.apply({'params': params}, x, train=False)
    )

    all_true, all_pred = [], []
    loader = getattr(dm, f'{split}_loader')()
    for batch in loader:
        logits = eval_fn(batch['X'])
        preds  = np.array(ordinal_predict(logits))
        trues  = np.array(batch['y'])
        all_pred.extend(preds.tolist())
        all_true.extend(trues.tolist())

    fig  = confusion_matrix_fig(np.array(all_true), np.array(all_pred))
    path = out_dir / f"confusion_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrix saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate sparse-obs cross-attention TC classifier.'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt',   required=True, help='Path to orbax checkpoint dir.')
    parser.add_argument('--split',  default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--n',      type=int, default=8, help='Number of attention map examples.')
    parser.add_argument('--out_dir', default='figures')
    parser.add_argument('--mode', default='all', choices=['attention', 'confusion', 'all'])
    args = parser.parse_args()

    if args.mode in ('attention', 'all'):
        run_attention_maps(args.config, args.ckpt, args.split, args.n, args.out_dir)
    if args.mode in ('confusion', 'all'):
        run_confusion_matrix(args.config, args.ckpt, args.split, args.out_dir)


if __name__ == '__main__':
    main()
