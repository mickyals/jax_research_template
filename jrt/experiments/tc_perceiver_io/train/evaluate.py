"""
experiments/tc_perceiver_io/train/evaluate.py

Post-training evaluation for TCPerceiverIO.

Produces
--------
- Scalar metrics (loss, cross-entropy, accuracy, binary_accuracy, mae_class)
- Full-set metrics — mAP (macro one-vs-rest average precision) and pr_auc
  (binary TC-vs-background detection AP), computed over the full accumulated
  split — see training/metrics.py FULL_SET_METRICS
- 9×9 confusion matrix — row-normalised (recall per class) + raw counts
- Per-class precision, recall, F1 bar chart
- Binary detection summary (TC vs. no-storm)

CLI
---
    python -m experiments.tc_perceiver_io.train.evaluate \\
        jrt/experiments/tc_perceiver_io/configs/train.yaml

    # Override checkpoint directory
    python -m experiments.tc_perceiver_io.train.evaluate \\
        jrt/.../train.yaml --checkpoint_dir runs/exp01/checkpoints

    # Save plots to disk instead of displaying
    python -m experiments.tc_perceiver_io.train.evaluate \\
        jrt/.../train.yaml --output_dir runs/exp01/eval --no_show

    # Evaluate on validation split instead of test
    python -m experiments.tc_perceiver_io.train.evaluate \\
        jrt/.../train.yaml --split val
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

from experiments.tc_perceiver_io.data.datamodule import TCDataModule
from experiments.tc_perceiver_io.data.transforms.encoding import decode_domain
from experiments.tc_perceiver_io.data.sources.ibtracs import CLASS_NAMES, N_CLASSES
from experiments.tc_perceiver_io.train.metrics import build_metrics_fns
from training.metrics import (
    binary_pr_curve,
    compute_full_set_metrics,
    per_class_pr_curves,
)
from experiments.tc_perceiver_io.train.model import TCPerceiverIO
from experiments.tc_perceiver_io.plotting.plotting import (
    plot_confusion_matrix,
    plot_class_metrics,
    plot_attention_geographic,
    plot_attention_matrix_grid,
    plot_decoder_query,
    plot_pr_curve,
    plot_pr_curves_per_class,
)
from training.trainer import Trainer


# CLASS_NAMES / N_CLASSES are the canonical label space, imported from
# data/sources/ibtracs.py (the experiment's single source of truth).


def domain_latlon_for_sample(batch, sample_idx, fov_lat, fov_lon):
    """Decode a domain sample's station/query coords to lat/lon for plotting.

    Returns ``((station_lats, station_lons), (query_lat, query_lon))`` for the
    REAL (masked) stations of ``batch`` sample ``sample_idx``. The attention
    plotter takes these pre-decoded positions so the plotting layer stays free
    of the experiment's coordinate encoding (decode_domain lives in
    data/transforms/encoding.py) — see plot_attention_geographic (plan r9/r17).
    """
    X      = batch['X']
    coords = np.asarray(X['station_coords'][sample_idx])   # (N, 2)
    mask   = np.asarray(X['station_mask'][sample_idx])     # (N,) bool
    qc     = np.asarray(X['query_coords'][sample_idx])     # (2,)
    lats, lons   = decode_domain(coords[mask, 0], coords[mask, 1], fov_lat, fov_lon)
    q_lat, q_lon = decode_domain(qc[0], qc[1], fov_lat, fov_lon)
    return (lats, lons), (float(q_lat), float(q_lon))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_predictions(
    model:     TCPerceiverIO,
    variables: dict,
    loader:    Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
    """Run all batches through the model.

    Parameters
    ----------
    model : TCPerceiverIO
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


def collect_class_exemplars(loader, n_classes: int):
    """Pull ONE example of each true class from the loader (first seen).

    Used to show a concrete prediction per class (Background → Cat 5) on the
    attention Read map. Returns ``(X, labels, metas)`` stacked over the classes
    actually present (class order), or None if the loader is empty:
        X      : dict of (C', ...) arrays — the model input for the exemplars
        labels : (C',) int true classes
        metas  : list of per-sample meta dicts (sid / query_lat / query_lon),
                 or None entries when the loader carries no 'meta'.
    """
    found: dict[int, tuple[dict, Optional[dict]]] = {}
    for batch in loader:
        y    = np.asarray(batch['y'])
        meta = batch.get('meta')
        for j in range(len(y)):
            c = int(y[j])
            if c in found:
                continue
            xj = {k: np.asarray(v[j]) for k, v in batch['X'].items()}
            mj = ({k: v[j] for k, v in meta.items()} if meta is not None else None)
            found[c] = (xj, mj)
        if len(found) >= n_classes:
            break

    if not found:
        return None
    classes = sorted(found)
    X = {k: np.stack([found[c][0][k] for c in classes])
         for k in found[classes[0]][0]}
    labels = np.array(classes, dtype=np.int32)
    metas  = [found[c][1] for c in classes]
    return X, labels, metas


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
) -> None:
    """Print scalar metrics, binary summary, and per-class table to stdout."""
    cm    = confusion_matrix(preds, labels, n_classes)
    pcm   = per_class_metrics(cm)
    bin_m = binary_metrics(preds, labels)

    logits_j = jnp.array(logits)
    labels_j = jnp.array(labels)

    w = 62
    print(f"\n{'='*w}")
    print(f"  TCPerceiverIO — {split.upper()} evaluation")
    print(f"{'='*w}")
    print(f"  Samples : {len(preds)}")

    print(f"\n  Scalar metrics:")
    for name, fn in metrics_fns.items():
        val = float(fn(logits_j, labels_j))
        print(f"    {split}/{name}: {val:.5f}")

    # Full-set metrics — integrate a PR curve over the whole split, so they are
    # computed here over the accumulated logits/labels, not per batch.
    print(f"\n  Full-set metrics:")
    for name, val in compute_full_set_metrics(logits, labels).items():
        print(f"    {split}/{name}: {val:.5f}")

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
    class_examples: bool = True,
) -> None:
    """Full evaluation pipeline: load checkpoint → inference → report + plots.

    Parameters
    ----------
    config_path : str or Path
        Path to train.yaml.
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
    # The model is coordinate-agnostic (the learned latent array is the query),
    # so location_encoding configures the datamodule only — nothing is injected
    # into config['model'].
    radius_km = float(config['data'].get('radius_km', 500.0))
    fov_lat   = config['data'].get('fov_lat')
    fov_lon   = config['data'].get('fov_lon')

    dm          = TCDataModule.from_config(config['data'])
    target_spec = dm.target_spec
    class_names = target_spec.class_names
    n_classes   = target_spec.n_classes
    config['model']['n_classes'] = n_classes
    model       = TCPerceiverIO(**config['model'])
    metrics_fns = build_metrics_fns(metrics=config['trainer'].get('metrics'))
    trainer     = Trainer(model, metrics_fns, config['trainer'])

    loader = dm.test_loader() if split == 'test' else dm.val_loader()

    # Initialise model pytree structure, then restore best weights
    exmp_batch     = next(iter(loader))
    abstract_state = trainer.init_state(exmp_batch)
    best_state     = trainer.load_checkpoint(abstract_state)
    variables      = {'params': best_state.params}

    preds, labels, logits, meta = collect_predictions(model, variables, loader)

    print_report(preds, labels, logits, metrics_fns,
                 split=split, class_names=class_names, n_classes=n_classes)

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

    # Precision-recall curves (the shape behind pr_auc / mAP).
    fig_pr     = plot_pr_curve(
        binary_pr_curve(logits, labels),
        title=f'PR — TC vs. background detection ({split})')
    fig_pr_cls = plot_pr_curves_per_class(
        per_class_pr_curves(logits, labels), class_names,
        title=f'Per-class PR — one-vs-rest ({split})')

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig_norm.savefig(out / f'{split}_confusion_norm.png',    dpi=150, bbox_inches='tight')
        fig_raw.savefig( out / f'{split}_confusion_counts.png',  dpi=150, bbox_inches='tight')
        fig_cls.savefig( out / f'{split}_per_class_metrics.png', dpi=150, bbox_inches='tight')
        fig_pr.savefig(  out / f'{split}_pr_curve.png',          dpi=150, bbox_inches='tight')
        fig_pr_cls.savefig(out / f'{split}_pr_curves_per_class.png', dpi=150, bbox_inches='tight')
        print(f"Confusion / metrics / PR-curve plots saved to {out}/")

    # ------------------------------------------------------------------
    # Attention plots — the three Perceiver-IO components (pre-softmax scores,
    # softmaxed for display) from the first batch:
    #   Read map      — per-station attention on the station geometry,
    #   Processor grid — layers × heads of the N×N latent self-attention,
    #   Decoder query  — heads × latents output-query attention (decode_mode
    #                    'attention' only; None for 'avgproj').
    # ------------------------------------------------------------------
    if n_attn_samples > 0:
        attn_batch   = exmp_batch  # first batch reused
        logits, attn = model.apply(variables, attn_batch['X'],
                                   train=False, return_weights=True)
        read   = np.asarray(jax.nn.softmax(attn['read'],      axis=-1))  # (B,H,N,M)
        proc   = np.asarray(jax.nn.softmax(attn['processor'], axis=-1))  # (L,B,H,N,N)
        dec    = attn.get('decoder')
        dec    = np.asarray(jax.nn.softmax(dec, axis=-1)) if dec is not None else None
        n_plot = min(n_attn_samples, proc.shape[1])

        # Per-batch predictions + storm attribution for figure titles.
        batch_preds = np.asarray(logits).argmax(axis=-1)
        batch_meta  = attn_batch.get('meta')
        ds          = dm._test_ds if split == 'test' else dm._val_ds
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

        for i in range(n_plot):
            # Geographic Read map. For domain encoding decode this sample's
            # positions (plotting does not import the coordinate encoding).
            station_latlon = query_latlon = None
            if loc_enc == 'domain':
                station_latlon, query_latlon = domain_latlon_for_sample(
                    attn_batch, i, fov_lat, fov_lon)
            fig_r = plot_attention_geographic(
                read, attn_batch, location_encoding=loc_enc,
                fov_lat=fov_lat, fov_lon=fov_lon, radius_km=radius_km,
                sample_idx=i, geo=geo,
                storm_latlon=(
                    (float(batch_meta['query_lat'][i]),
                     float(batch_meta['query_lon'][i]))
                    if (geo and loc_enc == 'unit_circle'
                        and batch_meta is not None) else None),
                station_latlon=station_latlon, query_latlon=query_latlon,
                title=_sample_title(i),
            )

            fig_g = plot_attention_matrix_grid(
                proc, sample_idx=i,
                title=f'Processor self-attention — {_sample_title(i)}',
            )

            fig_d = None
            if dec is not None:
                fig_d = plot_decoder_query(
                    dec, sample_idx=i,
                    title=f'Decoder output-query attention — {_sample_title(i)}',
                )

            if output_dir is not None:
                fig_r.savefig(out / f'{split}_attn_read_map_sample{i}.png',
                              dpi=150, bbox_inches='tight')
                fig_g.savefig(out / f'{split}_attn_processor_grid_sample{i}.png',
                              dpi=150, bbox_inches='tight')
                if fig_d is not None:
                    fig_d.savefig(out / f'{split}_attn_decoder_query_sample{i}.png',
                                  dpi=150, bbox_inches='tight')

        if output_dir is not None:
            print(f"Attention plots saved to {out}/")

    # ------------------------------------------------------------------
    # Per-class exemplars — one sample of each true class (Background → Cat 5)
    # with its prediction, shown on the Read attention map. A quick "how does
    # the model behave on each class" panel.
    # ------------------------------------------------------------------
    if class_examples:
        ex = collect_class_exemplars(
            dm.test_loader() if split == 'test' else dm.val_loader(), n_classes)
        if ex is not None:
            Xex, yex, metas = ex
            logits_ex, attn_ex = model.apply(variables, Xex, train=False,
                                             return_weights=True)
            read_ex  = np.asarray(jax.nn.softmax(attn_ex['read'], axis=-1))
            preds_ex = np.asarray(logits_ex).argmax(-1)
            ds          = dm._test_ds if split == 'test' else dm._val_ds
            sid_to_name = dict(zip(
                np.asarray(ds.ibtracs['SID']).tolist(),
                np.asarray(ds.ibtracs['NAME']).tolist(),
            ))

            print(f"\n  Per-class exemplars (true → pred):")
            for k in range(len(yex)):
                true_c = class_names[int(yex[k])]
                pred_c = class_names[int(preds_ex[k])]
                sid    = metas[k]['sid'] if metas[k] is not None else None
                who    = (f"{sid} {sid_to_name.get(sid, '')}".strip()
                          if sid is not None else 'background')
                mark   = '✓' if int(yex[k]) == int(preds_ex[k]) else '✗'
                print(f"    {true_c:<14} → {pred_c:<14} {mark}  ({who})")

                station_latlon = query_latlon = None
                if loc_enc == 'domain':
                    station_latlon, query_latlon = domain_latlon_for_sample(
                        {'X': Xex}, k, fov_lat, fov_lon)
                storm_latlon = None
                if (geo and loc_enc == 'unit_circle' and metas[k] is not None
                        and 'query_lat' in metas[k]):
                    storm_latlon = (float(metas[k]['query_lat']),
                                    float(metas[k]['query_lon']))
                fig_ex = plot_attention_geographic(
                    read_ex, {'X': Xex}, location_encoding=loc_enc,
                    fov_lat=fov_lat, fov_lon=fov_lon, radius_km=radius_km,
                    sample_idx=k, geo=geo, storm_latlon=storm_latlon,
                    station_latlon=station_latlon, query_latlon=query_latlon,
                    title=f"true: {true_c}, pred: {pred_c} ({who})",
                )
                if output_dir is not None:
                    fig_ex.savefig(
                        out / f'{split}_classex_{int(yex[k])}_read_map.png',
                        dpi=150, bbox_inches='tight')
            if output_dir is not None:
                print(f"Per-class exemplar maps saved to {out}/")

    if show_plots:
        plt.show()
    else:
        plt.close('all')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TCPerceiverIO from its best checkpoint."
    )
    parser.add_argument(
        'config', type=str,
        help="Path to train.yaml.",
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
    parser.add_argument(
        '--no_class_examples', action='store_true',
        help="Skip the per-class exemplar Read maps (one sample per class with "
             "its prediction).",
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
        class_examples=not args.no_class_examples,
    )
