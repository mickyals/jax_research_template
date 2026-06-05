"""
experiments/sparse_obs_cross_attn/evaluate.py

Post-training evaluation for TCClassifier.

Produces
--------
- Scalar metrics (cross-entropy, accuracy, binary_accuracy, mae_class)
- 11×11 confusion matrix — row-normalised (recall per class) + raw counts
- Per-class precision, recall, F1 bar chart
- Binary detection summary (TC vs. no-storm)

CLI
---
    python -m experiments.sparse_obs_cross_attn.evaluate \\
        src/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml

    # Override checkpoint directory
    python -m experiments.sparse_obs_cross_attn.evaluate \\
        src/.../tc_classifier.yaml --checkpoint_dir runs/exp01/checkpoints

    # Save plots to disk instead of displaying
    python -m experiments.sparse_obs_cross_attn.evaluate \\
        src/.../tc_classifier.yaml --output_dir runs/exp01/eval --no_show

    # Evaluate on validation split instead of test
    python -m experiments.sparse_obs_cross_attn.evaluate \\
        src/.../tc_classifier.yaml --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import yaml

from experiments.sparse_obs_cross_attn.datamodule import TCDataModule
from experiments.sparse_obs_cross_attn.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.model import TCClassifier, N_CLASSES
from training.trainer import Trainer


# ---------------------------------------------------------------------------
# Class label names  (label k → SSHS k-5)
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = [
    'No Storm',     # 0  — background
    'SSHS -4',      # 1
    'SSHS -3',      # 2
    'SSHS -2',      # 3
    'SSHS -1 (TD)', # 4  — tropical depression
    'SSHS  0 (TS)', # 5  — tropical storm
    'Cat 1',        # 6
    'Cat 2',        # 7
    'Cat 3',        # 8
    'Cat 4',        # 9
    'Cat 5',        # 10
]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_predictions(
    model:     TCClassifier,
    variables: dict,
    loader:    Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run all batches through the model.

    Parameters
    ----------
    model : TCClassifier
    variables : dict
        Flax variables dict, e.g. ``{'params': state.params}``.
    loader : iterable
        Yields ``{'X': ..., 'y': ...}`` dicts.

    Returns
    -------
    preds  : np.ndarray int32 (N,)  — argmax predictions
    labels : np.ndarray int32 (N,)
    logits : np.ndarray float32 (N, n_classes)
    """
    apply_fn = jax.jit(lambda X: model.apply(variables, X, train=False))

    all_preds  = []
    all_labels = []
    all_logits = []
    for batch in loader:
        logits = np.asarray(apply_fn(batch['X']), dtype=np.float32)
        preds  = logits.argmax(axis=-1).astype(np.int32)
        labels = np.asarray(batch['y'], dtype=np.int32)
        all_preds.append(preds)
        all_labels.append(labels)
        all_logits.append(logits)

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_logits),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(
    preds:     np.ndarray,
    labels:    np.ndarray,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    """Compute confusion matrix.

    Returns
    -------
    np.ndarray int64 (n_classes, n_classes)
        ``cm[i, j]`` = number of samples with true class i predicted as j.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


def per_class_metrics(cm: np.ndarray) -> dict[int, dict[str, float]]:
    """Per-class precision, recall, F1, and support from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray (n_classes, n_classes)

    Returns
    -------
    dict[int, dict]
        Keys: class indices. Values: dicts with
        'precision', 'recall', 'f1', 'support'.
    """
    n   = cm.shape[0]
    out = {}
    for k in range(n):
        tp  = int(cm[k, k])
        fp  = int(cm[:, k].sum()) - tp
        fn  = int(cm[k, :].sum()) - tp
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1  = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
        out[k] = {
            'precision': pre,
            'recall':    rec,
            'f1':        f1,
            'support':   int(cm[k, :].sum()),
        }
    return out


def binary_metrics(
    preds:  np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """TC vs. no-storm detection metrics (class 0 = negative, 1–10 = positive).

    Returns
    -------
    dict with 'accuracy', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn'.
    """
    pred_tc = preds  > 0
    true_tc = labels > 0

    tp = int(( pred_tc &  true_tc).sum())
    fp = int(( pred_tc & ~true_tc).sum())
    fn = int((~pred_tc &  true_tc).sum())
    tn = int((~pred_tc & ~true_tc).sum())
    total = tp + fp + fn + tn

    acc = (tp + tn) / total if total > 0 else 0.0
    pre = tp / (tp + fp)    if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn)    if (tp + fn) > 0 else 0.0
    f1  = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0

    return {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: list[str],
    normalize:   bool = True,
    title:       str  = 'Confusion Matrix',
    ax:          Optional[Any] = None,
) -> plt.Figure:
    """Heatmap of confusion matrix.

    Parameters
    ----------
    normalize : bool
        True  → row-normalised (each row sums to 1; value = recall per class).
        False → raw counts.
    ax : matplotlib Axes, optional
        If None a new figure is created.

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 9))
    else:
        fig = ax.figure

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display  = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
        vmax     = 1.0
        clabel   = 'Recall (fraction of true class)'
        cell_fmt = '{:.2f}'
    else:
        display  = cm.astype(float)
        vmax     = float(cm.max()) if cm.max() > 0 else 1.0
        clabel   = 'Count'
        cell_fmt = '{:d}'

    im   = ax.imshow(display, interpolation='nearest', cmap='Blues',
                     vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(clabel, fontsize=9)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted class', fontsize=9)
    ax.set_ylabel('True class', fontsize=9)
    ax.set_title(title, fontsize=11)

    thresh = display.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = display[i, j]
            txt = cell_fmt.format(val if normalize else int(cm[i, j]))
            ax.text(j, i, txt, ha='center', va='center', fontsize=6,
                    color='white' if val > thresh else 'black')

    fig.tight_layout()
    return fig


def plot_class_metrics(
    metrics:     dict[int, dict[str, float]],
    class_names: list[str],
    ax:          Optional[Any] = None,
) -> plt.Figure:
    """Grouped bar chart: per-class precision, recall, F1.

    Parameters
    ----------
    metrics : dict returned by per_class_metrics()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    else:
        fig = ax.figure

    n     = len(class_names)
    x     = np.arange(n)
    w     = 0.25

    prec = [metrics[k]['precision'] for k in range(n)]
    rec  = [metrics[k]['recall']    for k in range(n)]
    f1   = [metrics[k]['f1']        for k in range(n)]

    ax.bar(x - w, prec, w, label='Precision', color='steelblue',  alpha=0.85)
    ax.bar(x,     rec,  w, label='Recall',    color='darkorange', alpha=0.85)
    ax.bar(x + w, f1,   w, label='F1',        color='seagreen',   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-class Precision / Recall / F1')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Attention visualization
# ---------------------------------------------------------------------------

def extract_attention_weights(
    model:     TCClassifier,
    variables: dict,
    batch:     dict,
) -> np.ndarray:
    """Run batch through model with return_weights=True.

    Returns
    -------
    np.ndarray float32 (B, num_heads, N)
        Cross-attention weights from the last cross-attention block.
        N = max_stations; padding positions have weight ≈ 0.
    """
    apply_fn = jax.jit(
        lambda X: model.apply(variables, X, train=False, return_weights=True)
    )
    _, weights = apply_fn(batch['X'])
    return np.asarray(weights, dtype=np.float32)


def plot_attention_geographic(
    weights:           np.ndarray,
    batch:             dict,
    location_encoding: str,
    fov_lat:           Optional[tuple[float, float]] = None,
    fov_lon:           Optional[tuple[float, float]] = None,
    radius_km:         float = 500.0,
    sample_idx:        int   = 0,
    head_agg:          str   = 'mean',
    ax:                Optional[Any] = None,
) -> plt.Figure:
    """Plot per-station attention weight for one sample.

    For ``unit_circle`` encoding: polar axes — radius = normalised distance
    from storm centre, angle = bearing.  The storm is at the origin.

    For ``domain`` encoding: Cartesian axes — decoded lat/lon from the
    normalised coord representation.  Query position marked with a star.

    Parameters
    ----------
    weights : np.ndarray (B, H, N)
        From extract_attention_weights().
    batch : dict
        Raw batch dict (contains 'X' with station_coords, station_mask,
        query_coords).
    location_encoding : {'unit_circle', 'domain'}
    fov_lat, fov_lon : required for domain mode.
    radius_km : float
        Search radius used when building samples (unit_circle mode label).
    sample_idx : int
        Which sample in the batch to visualise.
    head_agg : {'mean', 'max'}
        How to collapse the head dimension before plotting.
    """
    X            = batch['X']
    coords       = np.asarray(X['station_coords'][sample_idx])   # (N, 2)
    mask         = np.asarray(X['station_mask'][sample_idx])     # (N,) bool
    query_coords = np.asarray(X['query_coords'][sample_idx])     # (2,)

    # Aggregate attention over heads: (H, N) → (N,)
    w = weights[sample_idx]                                       # (H, N)
    w_station = w.mean(axis=0) if head_agg == 'mean' else w.max(axis=0)
    w_real = w_station[mask]                                      # (n_real,)

    # Normalise to [0, 1] for sizing/colouring
    w_norm = (w_real - w_real.min()) / (w_real.max() - w_real.min() + 1e-12)

    if location_encoding == 'unit_circle':
        norm_dist   = coords[mask, 0]          # [0, 1]
        bearing_rad = coords[mask, 1]          # [0, 2π)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7),
                                   subplot_kw={'projection': 'polar'})
        else:
            fig = ax.figure

        sc = ax.scatter(
            bearing_rad, norm_dist,
            s=30 + w_norm * 250,
            c=w_real,
            cmap='YlOrRd',
            alpha=0.85,
            edgecolors='k',
            linewidths=0.4,
            zorder=3,
        )
        # Storm centre
        ax.scatter([0], [0], marker='*', s=200, color='royalblue',
                   zorder=5, label='Storm centre')

        ax.set_theta_zero_location('N')   # North at top
        ax.set_theta_direction(-1)        # Clockwise (compass convention)
        ax.set_rlim(0, 1)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_rlabel_position(45)
        ax.yaxis.set_tick_params(labelsize=7)
        tick_labels = [f'{r * radius_km:.0f} km' for r in [0.25, 0.5, 0.75, 1.0]]
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_title('Cross-attention weights\n(polar: distance × bearing from storm)',
                     pad=15, fontsize=10)
        fig.colorbar(sc, ax=ax, label='Attention weight', shrink=0.7, pad=0.1)

    else:  # domain
        if fov_lat is None or fov_lon is None:
            raise ValueError("fov_lat and fov_lon required for domain encoding.")

        half_pi  = float(np.pi / 2)
        lat_min, lat_max = fov_lat
        lon_min, lon_max = fov_lon
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min

        # Decode station positions
        lats = (coords[mask, 0] / half_pi + 1) / 2 * lat_span + lat_min
        lons = (coords[mask, 1] / half_pi + 1) / 2 * lon_span + lon_min

        # Decode query position
        q_lat = (query_coords[0] / half_pi + 1) / 2 * lat_span + lat_min
        q_lon = (query_coords[1] / half_pi + 1) / 2 * lon_span + lon_min

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 7))
        else:
            fig = ax.figure

        sc = ax.scatter(
            lons, lats,
            s=30 + w_norm * 250,
            c=w_real,
            cmap='YlOrRd',
            alpha=0.85,
            edgecolors='k',
            linewidths=0.4,
            zorder=3,
        )
        ax.scatter([q_lon], [q_lat], marker='*', s=250, color='royalblue',
                   zorder=5, label='Query (storm centre)')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Cross-attention weights (domain encoding)', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.colorbar(sc, ax=ax, label='Attention weight')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(
    preds:       np.ndarray,
    labels:      np.ndarray,
    logits:      np.ndarray,
    metrics_fns: dict,
    split:       str       = 'test',
    class_names: list[str] = CLASS_NAMES,
    n_classes:   int       = N_CLASSES,
) -> None:
    """Print scalar metrics, binary summary, and per-class table to stdout."""
    cm    = confusion_matrix(preds, labels, n_classes)
    pcm   = per_class_metrics(cm)
    bin_m = binary_metrics(preds, labels)

    logits_j = jnp.array(logits)
    labels_j = jnp.array(labels)

    w = 62
    print(f"\n{'='*w}")
    print(f"  TCClassifier — {split.upper()} evaluation")
    print(f"{'='*w}")
    print(f"  Samples : {len(preds)}")

    print(f"\n  Scalar metrics:")
    for name, fn in metrics_fns.items():
        val = float(fn(logits_j, labels_j))
        print(f"    {split}/{name}: {val:.5f}")

    print(f"\n  Binary detection (TC vs. No Storm):")
    print(f"    Accuracy : {bin_m['accuracy']:.4f}")
    print(f"    Precision: {bin_m['precision']:.4f}")
    print(f"    Recall   : {bin_m['recall']:.4f}")
    print(f"    F1       : {bin_m['f1']:.4f}")
    print(f"    TP={bin_m['tp']}  FP={bin_m['fp']}  "
          f"FN={bin_m['fn']}  TN={bin_m['tn']}")

    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<15}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}")
    print(f"  {'-'*50}")
    for k, name in enumerate(class_names):
        m = pcm[k]
        print(f"  {name:<15}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  "
              f"{m['f1']:>6.3f}  {m['support']:>8}")
    print(f"{'='*w}\n")


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(
    config_path:    str | Path,
    checkpoint_dir: Optional[str | Path] = None,
    output_dir:     Optional[str | Path] = None,
    split:          str  = 'test',
    n_attn_samples: int  = 4,
    show_plots:     bool = True,
) -> None:
    """Full evaluation pipeline: load checkpoint → inference → report + plots.

    Parameters
    ----------
    config_path : str or Path
        Path to tc_classifier.yaml.
    checkpoint_dir : str or Path, optional
        Override ``trainer.checkpoint_dir`` in config.
    output_dir : str or Path, optional
        Save plots here as PNG files. None = no saving.
    split : {'test', 'val'}
        Data split to evaluate.
    n_attn_samples : int
        Number of samples for which to produce attention geographic plots.
        Each sample uses the first batch; set 0 to skip attention plots.
    show_plots : bool
        Call ``plt.show()`` after plotting.
    """
    config = _load_config(config_path)
    if checkpoint_dir is not None:
        config['trainer']['checkpoint_dir'] = str(checkpoint_dir)

    dm          = TCDataModule.from_config(config['data'])
    model       = TCClassifier(**config['model'])
    metrics_fns = build_metrics_fns()
    trainer     = Trainer(model, metrics_fns, config['trainer'])

    loader = dm.test_loader() if split == 'test' else dm.val_loader()

    # Initialise model pytree structure, then restore best weights
    exmp_batch     = next(iter(loader))
    abstract_state = trainer.init_state(exmp_batch)
    best_state     = trainer.load_checkpoint(abstract_state)
    variables      = {'params': best_state.params}

    preds, labels, logits = collect_predictions(model, variables, loader)

    print_report(preds, labels, logits, metrics_fns,
                 split=split, class_names=CLASS_NAMES, n_classes=N_CLASSES)

    cm  = confusion_matrix(preds, labels)
    pcm = per_class_metrics(cm)

    fig_norm = plot_confusion_matrix(
        cm, CLASS_NAMES, normalize=True,
        title=f'Confusion Matrix — {split} (row-normalised)',
    )
    fig_raw = plot_confusion_matrix(
        cm, CLASS_NAMES, normalize=False,
        title=f'Confusion Matrix — {split} (counts)',
    )
    fig_cls = plot_class_metrics(pcm, CLASS_NAMES)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig_norm.savefig(out / f'{split}_confusion_norm.png',    dpi=150, bbox_inches='tight')
        fig_raw.savefig( out / f'{split}_confusion_counts.png',  dpi=150, bbox_inches='tight')
        fig_cls.savefig( out / f'{split}_per_class_metrics.png', dpi=150, bbox_inches='tight')
        print(f"Confusion / metrics plots saved to {out}/")

    # ------------------------------------------------------------------
    # Attention geographic plots
    # ------------------------------------------------------------------
    if n_attn_samples > 0:
        loc_enc  = config['model'].get('location_encoding', 'unit_circle')
        fov_lat  = config['data'].get('fov_lat')
        fov_lon  = config['data'].get('fov_lon')
        rad_km   = config['data'].get('radius_km', 500.0)

        attn_batch   = exmp_batch  # first batch reused
        attn_weights = extract_attention_weights(model, variables, attn_batch)
        n_plot       = min(n_attn_samples, attn_weights.shape[0])

        for i in range(n_plot):
            label_i = int(attn_batch['y'][i])
            title_i = CLASS_NAMES[label_i] if label_i < len(CLASS_NAMES) else str(label_i)
            fig_a   = plot_attention_geographic(
                attn_weights, attn_batch,
                location_encoding=loc_enc,
                fov_lat=fov_lat,
                fov_lon=fov_lon,
                radius_km=rad_km,
                sample_idx=i,
            )
            fig_a.suptitle(f'Sample {i} — true label: {title_i}', y=1.01,
                           fontsize=10)
            if output_dir is not None:
                fig_a.savefig(out / f'{split}_attn_sample{i}.png',
                              dpi=150, bbox_inches='tight')

        if output_dir is not None:
            print(f"Attention plots saved to {out}/")

    if show_plots:
        plt.show()
    else:
        plt.close('all')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TCClassifier from its best checkpoint."
    )
    parser.add_argument(
        'config', type=str,
        help="Path to tc_classifier.yaml.",
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, default=None,
        help="Override trainer.checkpoint_dir from config.",
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help="Directory to save plots as PNG files.",
    )
    parser.add_argument(
        '--split', choices=['test', 'val'], default='test',
        help="Data split to evaluate on (default: test).",
    )
    parser.add_argument(
        '--n_attn_samples', type=int, default=4,
        help="Number of attention geographic plots to produce (default: 4, 0 = skip).",
    )
    parser.add_argument(
        '--no_show', action='store_true',
        help="Do not display plots interactively.",
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args()
    evaluate(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        split=args.split,
        n_attn_samples=args.n_attn_samples,
        show_plots=not args.no_show,
    )
