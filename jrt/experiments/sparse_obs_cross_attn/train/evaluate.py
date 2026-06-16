"""
experiments/sparse_obs_cross_attn/train/evaluate.py

Post-training evaluation for TCClassifier.

Produces
--------
- Scalar metrics (loss, cross-entropy, accuracy, binary_accuracy, mae_class)
- Quadratic-weighted kappa (ordinal agreement) and ECE (calibration gap),
  computed over the full accumulated split — see metrics.py
- 9×9 confusion matrix — row-normalised (recall per class) + raw counts
- Per-class precision, recall, F1 bar chart
- Binary detection summary (TC vs. no-storm)

CLI
---
    python -m experiments.sparse_obs_cross_attn.train.evaluate \\
        jrt/experiments/sparse_obs_cross_attn/configs/tc_classifier.yaml

    # Override checkpoint directory
    python -m experiments.sparse_obs_cross_attn.train.evaluate \\
        jrt/.../tc_classifier.yaml --checkpoint_dir runs/exp01/checkpoints

    # Save plots to disk instead of displaying
    python -m experiments.sparse_obs_cross_attn.train.evaluate \\
        jrt/.../tc_classifier.yaml --output_dir runs/exp01/eval --no_show

    # Evaluate on validation split instead of test
    python -m experiments.sparse_obs_cross_attn.train.evaluate \\
        jrt/.../tc_classifier.yaml --split val
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

from experiments.sparse_obs_cross_attn.data.datamodule import TCDataModule
from experiments.sparse_obs_cross_attn.data.sources.ibtracs import CLASS_NAMES, N_CLASSES
from experiments.sparse_obs_cross_attn.train.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.train.model import TCClassifier
from training.metrics import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    maximum_calibration_error,
    quadratic_weighted_kappa,
)
from experiments.sparse_obs_cross_attn.plotting.plotting import (
    plot_confusion_matrix,
    plot_class_metrics,
    extract_attention_weights,
    plot_attention_geographic,
    plot_attention_mask,
    plot_attention_matrix_grid,
)
from training.trainer import Trainer


# CLASS_NAMES / N_CLASSES are the canonical label space, imported from
# data/sources/ibtracs.py (the experiment's single source of truth).


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_predictions(
    model:     TCClassifier,
    variables: dict,
    loader:    Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
    """Run all batches through the model.

    Parameters
    ----------
    model : TCClassifier
    variables : dict
        Flax variables dict, e.g. ``{'params': state.params}``.
    loader : iterable
        Yields ``{'X': ..., 'y': ...}`` dicts, optionally with a 'meta'
        entry (see data.datamodule._collate).

    Returns
    -------
    preds  : np.ndarray int32 (N,)  — argmax predictions
    labels : np.ndarray int32 (N,)
    logits : np.ndarray float32 (N, n_classes)
    meta   : dict | None — concatenated batch metadata aligned with preds
             (sid: list[str | None], iso_time / query_lat / query_lon /
             n_available / n_used arrays), or None if the loader yields
             no 'meta' entry.
    """
    apply_fn = jax.jit(lambda X: model.apply(variables, X, train=False))

    all_preds  = []
    all_labels = []
    all_logits = []
    all_meta: dict[str, list] = {}
    for batch in loader:
        logits = np.asarray(apply_fn(batch['X']), dtype=np.float32)
        preds  = logits.argmax(axis=-1).astype(np.int32)
        labels = np.asarray(batch['y'], dtype=np.int32)
        all_preds.append(preds)
        all_labels.append(labels)
        all_logits.append(logits)
        if 'meta' in batch:
            for k, v in batch['meta'].items():
                all_meta.setdefault(k, []).append(v)

    meta: Optional[dict] = None
    if all_meta:
        meta = {
            k: (sum(chunks, []) if isinstance(chunks[0], list)
                else np.concatenate(chunks))
            for k, chunks in all_meta.items()
        }

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_logits),
        meta,
    )


def per_storm_metrics(
    preds:  np.ndarray,
    labels: np.ndarray,
    sids:   list,
) -> dict[str, dict[str, float]]:
    """Group predictions by storm so every prediction is attributable.

    Background samples (sid is None) are skipped.

    Returns
    -------
    dict[sid, dict] with 'n' (samples), 'accuracy' (exact class match)
    and 'mae_class' (mean |pred − label|), per named storm.
    """
    by_sid: dict[str, list[int]] = {}
    for i, sid in enumerate(sids):
        if sid is not None:
            by_sid.setdefault(sid, []).append(i)

    out: dict[str, dict[str, float]] = {}
    for sid, idx in by_sid.items():
        p = preds[idx]
        l = labels[idx]
        out[sid] = {
            'n':         len(idx),
            'accuracy':  float((p == l).mean()),
            'mae_class': float(np.abs(p - l).mean()),
        }
    return out


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
    temperature: float     = 1.0,
) -> None:
    """Print scalar metrics, binary summary, and per-class table to stdout.

    ``temperature`` (fit on the val split via temperature scaling, Guo et al.
    2017) recalibrates the reported ECE: when != 1.0 the report prints ECE
    both before and after scaling. T does not change the argmax, so accuracy,
    QWK and the per-class table are identical either way.
    """
    cm    = confusion_matrix(preds, labels, n_classes)
    pcm   = per_class_metrics(cm)
    bin_m = binary_metrics(preds, labels)
    qwk   = quadratic_weighted_kappa(cm)
    probs = np.asarray(jax.nn.softmax(jnp.array(logits), axis=-1))
    ece   = expected_calibration_error(probs, labels)
    mce   = maximum_calibration_error(probs, labels)
    if temperature != 1.0:
        probs_ts = np.asarray(
            jax.nn.softmax(jnp.array(apply_temperature(logits, temperature)), axis=-1)
        )
        ece_ts = expected_calibration_error(probs_ts, labels)

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
    print(f"    {split}/qwk: {qwk:.5f}  (ordinal agreement; 1=perfect, 0=chance)")
    print(f"    {split}/ece: {ece:.5f}  (mean calibration gap; 0=perfectly calibrated)")
    print(f"    {split}/mce: {mce:.5f}  (worst-bin calibration gap)")
    if temperature != 1.0:
        print(f"    {split}/ece_tempscaled: {ece_ts:.5f}  "
              f"(T={temperature:.3f} fit on val; lower = better calibrated)")

    print(f"\n  Binary detection (TC vs. Background):")
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
    geo:            bool = False,
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
    geo : bool
        Draw the attention maps on cartopy map canvases (coastlines/
        borders): azimuthal storm-centred for unit_circle, PlateCarree
        for domain. Requires cartopy (optional dependency).
    """
    config = _load_config(config_path)
    if checkpoint_dir is not None:
        config['trainer']['checkpoint_dir'] = str(checkpoint_dir)

    # Coordinate convention (top-level) drives the datamodule encoding and the
    # CLS position handling — must match training for the checkpoint to load.
    loc_enc = config.get('location_encoding',
                         config['data'].get('location_encoding', 'unit_circle'))
    config['data']['location_encoding'] = loc_enc
    config['model']['learnable_query_pos'] = (loc_enc == 'unit_circle')

    dm          = TCDataModule.from_config(config['data'])
    target_spec = dm.target_spec
    class_names = target_spec.class_names
    n_classes   = target_spec.n_classes
    config['model']['n_classes'] = n_classes
    model       = TCClassifier(**config['model'])
    metrics_fns = build_metrics_fns(metrics=config['trainer'].get('metrics'))
    trainer     = Trainer(model, metrics_fns, config['trainer'])

    loader = dm.test_loader() if split == 'test' else dm.val_loader()

    # Initialise model pytree structure, then restore best weights
    exmp_batch     = next(iter(loader))
    abstract_state = trainer.init_state(exmp_batch)
    best_state     = trainer.load_checkpoint(abstract_state)
    variables      = {'params': best_state.params}

    preds, labels, logits, meta = collect_predictions(model, variables, loader)

    # Temperature scaling (Guo et al. 2017): fit a single T on the VAL split
    # and report calibrated ECE. For split='test' this is the proper val->test
    # transfer; for split='val' it is fit and applied on the same split (an
    # optimistic in-sample calibration check).
    val_loader = dm.val_loader()
    _, val_labels, val_logits, _ = collect_predictions(model, variables, val_loader)
    temperature = fit_temperature(val_logits, val_labels)

    print_report(preds, labels, logits, metrics_fns,
                 split=split, class_names=class_names, n_classes=n_classes,
                 temperature=temperature)

    if meta is not None:
        storm_m = per_storm_metrics(preds, labels, meta['sid'])
        print(f"  Per-storm attribution: {len(storm_m)} named storms")
        worst = sorted(storm_m.items(), key=lambda kv: kv[1]['accuracy'])[:5]
        print(f"  Lowest-accuracy storms:")
        for sid, m in worst:
            print(f"    {sid:<16}  n={m['n']:>4}  acc={m['accuracy']:.3f}  "
                  f"mae={m['mae_class']:.2f}")
        print()

    cm  = confusion_matrix(preds, labels, n_classes)
    pcm = per_class_metrics(cm)

    fig_norm = plot_confusion_matrix(
        cm, class_names, normalize=True,
        title=f'Confusion Matrix — {split} (row-normalised)',
    )
    fig_raw = plot_confusion_matrix(
        cm, class_names, normalize=False,
        title=f'Confusion Matrix — {split} (counts)',
    )
    fig_cls = plot_class_metrics(pcm, class_names)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig_norm.savefig(out / f'{split}_confusion_norm.png',    dpi=150, bbox_inches='tight')
        fig_raw.savefig( out / f'{split}_confusion_counts.png',  dpi=150, bbox_inches='tight')
        fig_cls.savefig( out / f'{split}_per_class_metrics.png', dpi=150, bbox_inches='tight')
        print(f"Confusion / metrics plots saved to {out}/")

    # ------------------------------------------------------------------
    # Attention plots — geographic maps + layers×heads matrix grids + the
    # static asymmetric-mask figure, all from the first batch
    # ------------------------------------------------------------------
    if n_attn_samples > 0:
        loc_enc  = config['data'].get('location_encoding', 'unit_circle')
        fov_lat  = config['data'].get('fov_lat')
        fov_lon  = config['data'].get('fov_lon')
        rad_km   = config['data'].get('radius_km', 500.0)

        attn_batch   = exmp_batch  # first batch reused
        attn_weights = extract_attention_weights(model, variables, attn_batch)
        # attn_weights: (num_layers, B, num_heads, N+1, N+1)
        n_plot       = min(n_attn_samples, attn_weights.shape[1])

        # Per-batch predictions + storm attribution for figure titles.
        batch_preds = np.asarray(
            model.apply(variables, attn_batch['X'], train=False)
        ).argmax(axis=-1)
        batch_meta = attn_batch.get('meta')
        ds         = dm._test_ds if split == 'test' else dm._val_ds
        sid_to_name = dict(zip(
            np.asarray(ds.ibtracs['SID']).tolist(),
            np.asarray(ds.ibtracs['NAME']).tolist(),
        ))

        def _sample_title(i: int) -> str:
            true_c = class_names[int(attn_batch['y'][i])]
            pred_c = class_names[int(batch_preds[i])]
            sid    = batch_meta['sid'][i] if batch_meta is not None else None
            who    = (f"{sid} {sid_to_name.get(sid, '')}".strip()
                      if sid is not None else 'background')
            return f"{who} — true: {true_c}, pred: {pred_c}"

        fig_mask = plot_attention_mask(
            np.asarray(attn_batch['X']['station_mask'][0]),
            full_self_attention=config['model'].get('full_self_attention', False),
        )
        if output_dir is not None:
            fig_mask.savefig(out / f'{split}_attn_mask.png',
                             dpi=150, bbox_inches='tight')

        for i in range(n_plot):
            title_i = _sample_title(i)
            storm_latlon = (
                (float(batch_meta['query_lat'][i]),
                 float(batch_meta['query_lon'][i]))
                if batch_meta is not None else None
            )
            fig_a   = plot_attention_geographic(
                attn_weights[-1][:, :, 0, :], attn_batch,  # last layer, query row (CLS = token 0)
                location_encoding=loc_enc,
                fov_lat=fov_lat,
                fov_lon=fov_lon,
                radius_km=rad_km,
                sample_idx=i,
                geo=geo,
                storm_latlon=storm_latlon,
            )
            # Keep the caption inside the figure (a y>1.0 suptitle is clipped
            # by wandb.Image / non-tight saves) with room above the axes title.
            fig_a.suptitle(title_i, y=0.99, fontsize=10)
            fig_a.subplots_adjust(top=0.86)
            fig_g = plot_attention_matrix_grid(
                attn_weights, sample_idx=i,
                title=f'Attention matrices — {title_i}',
            )
            if output_dir is not None:
                fig_a.savefig(out / f'{split}_attn_sample{i}.png',
                              dpi=150, bbox_inches='tight')
                fig_g.savefig(out / f'{split}_attn_grid_sample{i}.png',
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
    parser.add_argument(
        '--geo', action='store_true',
        help="Draw attention maps on cartopy map canvases (coastlines/"
             "borders; azimuthal storm-centred for unit_circle). "
             "Requires cartopy.",
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
        geo=args.geo,
    )
