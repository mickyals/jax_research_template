"""
experiments/cyclone_jax/train/log.py

WHEN/WHAT of during-training figures; visualise/figures.py and
utils.plotting decide HOW they look (layering ruling). Callbacks are
FACTORIES registered in CALLBACKS: each closes over the data bundle and
logger (via ``ctx``) and returns the ``fn(state, epoch, global_step)``
callable the jrt Trainer's step_callbacks contract expects;
``build_callbacks`` turns the trainer.callbacks yaml block into the
``[(fn, every_n_steps), ...]`` list ``fit()`` consumes.

Cadence is STEP-based (training budgets are num_steps; an "epoch" is just
an estimated dataset/batch_size number of steps): ``every`` defaults to
the train stream's batches-per-epoch. A val confusion matrix at step
cadence runs a full val pass — the scenarios here are small, that is
deliberate.

yaml surface (key sets validated in config.py):

    trainer:
      callbacks:
        - {name: confusion_matrix, split: val}          # every ~ one epoch
        - {name: storm_panel, split: val, every: 200}

    # storm selection is a DATA property (data yaml):
    storm_panels: {train: random, val: random, test: <sid>}

v1 callbacks:

    confusion_matrix  accumulate training.metrics.update_cm over the whole
                      split stream -> exact macro precision/recall scalars
                      (the metrics deregistered from per-batch METRICS) +
                      TWO annotated figures (counts + row-% — same matrix,
                      two readings)
    storm_panel       one fix (random | pinned sid per the data yaml knob)
                      -> truth-star/pred-ring map via visualise.figures

end_of_run additionally writes the per-fix prediction record for every
distinct split (predictions_<split>.csv: which fixes the model got right
and wrong), aggregates it per storm (per_storm_accuracy_<split>.csv,
worst-first: which storms are memorised vs hard), and emits the spatial
accuracy hexbin (where over the FOV the model is right/wrong) plus a
track-correctness figure per hardest storm — the hexbin/track cross-read
separates storm problems from location/sensing problems.

Figures go to logger.log_figure (all backends) AND run_dir/figures/ as
svg + png (editable-vector workflow). The logger closes each figure.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from training.logger import emit_figure
from training.metrics import compute_final_metrics, update_cm
from utils.jax_core.helpers import eval_forward
from utils.plotting.fields import confusion_matrix_figure
from utils.registry import Registry

from experiments.cyclone_jax.data.batching import collate
from experiments.cyclone_jax.data.identifiability import input_collisions
from experiments.cyclone_jax.data.sparsity import network_sparsity
from experiments.cyclone_jax.models.features import TOKEN_FIELDS
from experiments.cyclone_jax.visualise.figures import (
    SOURCE_STYLE, accuracy_hexbin_figure, accuracy_vs_resolution_figure,
    save_gif, storm_panel_figure, storm_track_correctness_figure,
)

CALLBACKS = Registry('Callback')

# The run's text record — train.py configures handlers (stdout + run.log)
# before anything trains; unconfigured (tests, notebooks) it propagates.
RUN_LOG = logging.getLogger('cyclone_jax.run')

_CALLBACK_SPEC_KEYS = {'name', 'every', 'split', 'kwargs'}


# ---------------------------------------------------------------------------
# Shared mechanics
# ---------------------------------------------------------------------------

def _predict(state, batch):
    """Model forward on one collated batch (eval mode; meta never traced).

    Thin over utils.jax_core.helpers.eval_forward (jitted, apply_fn
    static): full-split CM sweeps reuse one compiled forward instead of
    running op-by-op; pad_to keeps shapes fixed, so it traces once per
    batch shape (stream batch + the B=1 panel batch).
    """
    return eval_forward(state.apply_fn, state.params, batch['X'],
                        getattr(state, 'batch_stats', None))


def _emit(fig, name, split, global_step, logger, run_dir):
    """jrt emit_figure with this experiment's naming convention:
    tag {split}/{name}, still stem {name}_{split}."""
    emit_figure(logger, fig, f'{split}/{name}', global_step,
                run_dir=run_dir, stem=f'{name}_{split}')


def _domain_from_norms(norms):
    """Coordinate-scaling bounds double as the effective FOV when the data
    yaml has no explicit domain block (both feed the same stats record)."""
    if norms is None:
        return None
    return {'lat': [norms.stats['lat']['min'], norms.stats['lat']['max']],
            'lon': [norms.stats['lon']['min'], norms.stats['lon']['max']]}


def _storm_track(fixes, sid, until):
    """The storm's fixes up to and incl. `until`, time-ordered -> (lon,
    lat) trail arrays, or (None, None) when the fix table lacks coords
    (fakes)."""
    if 'lat' not in fixes or 'lon' not in fixes:
        return None, None
    sel = ((np.asarray(fixes['sid']) == sid)
           & (np.asarray(fixes['time']) <= until))
    order = np.argsort(np.asarray(fixes['time'])[sel])
    return (np.asarray(fixes['lon'])[sel][order],
            np.asarray(fixes['lat'])[sel][order])


def _render_fix(state, data, i, domain, basemap):
    """One fix index -> titled truth-star/pred-ring panel (with the track
    so far and the per-source station breakdown). Shared by the
    storm_panel callback and the end-of-run storm sequence."""
    loader, targets = data.loader, data.targets
    batch = collate([loader.build(i)], loader.inputs.pad_to)
    pred = int(np.argmax(np.asarray(_predict(state, batch))[0]))
    true = int(batch['y'][0])
    meta, mask = batch['meta'], batch['X']['station_mask'][0]
    lat = np.asarray(batch['X']['lat'][0])[mask]
    lon = np.asarray(batch['X']['lon'][0])[mask]
    ids = np.asarray(batch['X']['id'][0])[mask]   # untouched by normalise
    if data.norms is not None:
        lat, lon = data.norms.invert_coords(lat, lon)

    sid, t = str(meta['sid'][0]), meta['time'][0]
    track_lon, track_lat = _storm_track(loader.fixes, sid, t)
    name = (str(loader.fixes['name'][i]).strip()
            if 'name' in loader.fixes else '')
    codes = np.rint(ids).astype(int)
    counts = {label: int((codes == code).sum())
              for code, (label, _) in SOURCE_STYLE.items()}
    n = int(meta['n_stations'][0])
    title = (f"{sid}{' ' + name if name else ''} ({str(t)[:4]})  "
             f"{str(t)[:16]}Z\n"
             f"true {targets.class_names[true]} vs "
             f"pred {targets.class_names[pred]}   "
             f"land {counts['land']} | marine {counts['marine']} | "
             f"upper {counts['upper']}  (total {n})")
    if domain:
        r_km = network_sparsity(n, domain)['resolvable_km']
        title += f"   resolvable {r_km:.0f} km"
    return storm_panel_figure(
        lon, lat, float(meta['lon'][0]), float(meta['lat'][0]),
        true, pred, targets.n_classes, title=title,
        domain=domain, basemap=basemap, station_id=ids,
        class_names=targets.class_names,
        track_lon=track_lon, track_lat=track_lat)


# ---------------------------------------------------------------------------
# Callback factories — factory(ctx, split, **kwargs) -> fn(state, epoch, step)
# ---------------------------------------------------------------------------

@CALLBACKS.register(
    'confusion_matrix',
    description=(
        "Accumulate the split's confusion matrix (training.metrics."
        "update_cm) -> exact macro precision/recall scalars + annotated "
        "figure."
    ),
)
def _confusion_matrix(ctx, split='val'):
    data, logger = ctx['data'], ctx['logger']
    stream  = data.streams[split]
    targets = data.targets
    n_cls   = targets.n_classes

    def fn(state, epoch, global_step):
        cm = jnp.zeros((n_cls, n_cls), jnp.float32)
        # explicit-epoch iteration: never advances the stream's own epoch
        # counter (the Trainer is iterating the same object)
        for batch in stream.epoch(int(epoch)):
            cm = update_cm(cm, _predict(state, batch),
                           jnp.asarray(batch['y']))
        m = compute_final_metrics(cm)
        scalars = {f'{split}/macro_precision': float(m['macro_precision']),
                   f'{split}/macro_recall':    float(m['macro_recall']),
                   f'{split}/accuracy_exact':  float(m['accuracy'])}
        # per-class accuracy = that class's recall (diag / row sum)
        for name, r in zip(targets.class_names, np.asarray(m['recall'])):
            scalars[f'{split}/class_acc/{name}'] = float(r)
        logger.log_metrics(scalars, step=global_step)
        fig = confusion_matrix_figure(
            np.asarray(cm), class_names=targets.class_names,
            title=f'{split} confusion — step {global_step}')
        _emit(fig, 'confusion_matrix', split, global_step,
              logger, ctx.get('run_dir'))
        fig = confusion_matrix_figure(
            np.asarray(cm), class_names=targets.class_names,
            title=f'{split} confusion (row %) — step {global_step}',
            normalise=True)
        _emit(fig, 'confusion_matrix_pct', split, global_step,
              logger, ctx.get('run_dir'))

    return fn


@CALLBACKS.register(
    'storm_panel',
    description=(
        "One fix of the split (random, or the sid pinned in the data "
        "yaml's storm_panels block) -> station map with truth-star inside "
        "a predicted-class ring."
    ),
)
def _storm_panel(ctx, split='val', basemap=True):
    data, logger = ctx['data'], ctx['logger']
    loader = data.loader
    split_idx = np.asarray(data.splits[split])

    sel = (ctx.get('storm_panels') or {}).get(split, 'random')
    if sel and str(sel) != 'random':
        sids = np.asarray(loader.fixes['sid'])[split_idx]
        pool = split_idx[sids == str(sel)]
        if not len(pool):
            raise ValueError(
                f"storm_panels[{split!r}] = {sel!r} matches no fix in the "
                f"{split!r} split — check the sid against the library.")
    else:
        pool = split_idx
    domain = ctx.get('domain') or _domain_from_norms(data.norms)

    def fn(state, epoch, global_step):
        rng = np.random.default_rng(global_step)      # reproducible pick
        i = int(rng.choice(pool))
        fig = _render_fix(state, data, i, domain, basemap)
        _emit(fig, 'storm_panel', split, global_step,
              logger, ctx.get('run_dir'))

    return fn


# ---------------------------------------------------------------------------
# build_callbacks — trainer.callbacks yaml -> Trainer step_callbacks list
# ---------------------------------------------------------------------------

def build_callbacks(cfg, data, logger) -> list:
    """trainer.callbacks -> [(fn, every_n_steps), ...] for Trainer.fit.

    ``cfg`` is the merged {data, model, trainer} config; ``data`` the
    DataBundle; ``logger`` the run's jrt logger. ``every`` defaults to the
    train stream's batches-per-epoch (step-based "once per epoch");
    scenarios without a train stream must set it explicitly. Spec keys are
    re-checked here so direct (non-config) callers get the same typo guard.
    """
    specs = cfg['trainer'].get('callbacks') or []
    if not specs:
        return []

    ctx = {'data': data, 'logger': logger,
           'run_dir': cfg['trainer'].get('run_dir'),
           'storm_panels': cfg['data'].get('storm_panels'),
           'domain': cfg['data'].get('domain')}
    default_every = (len(data.streams['train'])
                     if 'train' in data.streams else None)

    out = []
    for spec in specs:
        if not isinstance(spec, dict) or not spec.get('name'):
            raise ValueError(f"trainer.callbacks items must be dicts with a "
                             f"'name' (keys: {sorted(_CALLBACK_SPEC_KEYS)}), "
                             f"got {spec!r}")
        unknown = set(spec) - _CALLBACK_SPEC_KEYS
        if unknown:
            raise ValueError(f"unknown key(s) {sorted(unknown)} in callback "
                             f"{spec['name']!r} — allowed: "
                             f"{sorted(_CALLBACK_SPEC_KEYS)}")
        split = spec.get('split', 'val')
        if split not in data.streams:
            raise ValueError(f"callback {spec['name']!r}: split {split!r} "
                             f"has no stream — available: "
                             f"{sorted(data.streams)}")
        every = spec.get('every', default_every)
        if not every or int(every) < 1:
            raise ValueError(f"callback {spec['name']!r} needs every >= 1 "
                             f"(no train stream to estimate an epoch from).")
        fn = CALLBACKS.get(spec['name'], ctx=ctx, split=split,
                           **(spec.get('kwargs') or {}))
        out.append((fn, int(every)))
    return out


# ---------------------------------------------------------------------------
# Prediction records — which fixes/storms/places the model gets right
# ---------------------------------------------------------------------------

# Local-view half-width: ±5° around the fix (the supervisor's
# local-vs-global resolution question) — the box the per-fix local
# station count / resolvable_km columns use.
LOCAL_HALF_WIDTH_DEG = 5.0


def _local_resolution(batch, norms):
    """Per-sample station count + resolvable_km within
    ±LOCAL_HALF_WIDTH_DEG of the fix. Station coords come out of the
    loader normalised (invert them); fix (meta) coords stay raw.
    resolvable_km is inf when the local box holds no stations."""
    lat = np.asarray(batch['X']['lat'], np.float32)
    lon = np.asarray(batch['X']['lon'], np.float32)
    mask = np.asarray(batch['X']['station_mask']).astype(bool)
    if norms is not None:
        lat, lon = norms.invert_coords(lat, lon)
    fix_lat = np.asarray(batch['meta']['lat'], float)
    fix_lon = np.asarray(batch['meta']['lon'], float)
    h = LOCAL_HALF_WIDTH_DEG
    n_local = np.empty(len(fix_lat), np.int64)
    r_local = np.empty(len(fix_lat), np.float64)
    for b in range(len(fix_lat)):
        near = (mask[b]
                & (np.abs(lat[b] - fix_lat[b]) <= h)
                & (np.abs(lon[b] - fix_lon[b]) <= h))
        n_local[b] = int(near.sum())
        box = {'lat': [max(fix_lat[b] - h, -90.0),
                       min(fix_lat[b] + h, 90.0)],
               'lon': [fix_lon[b] - h, fix_lon[b] + h]}
        r_local[b] = network_sparsity(n_local[b], box)['resolvable_km']
    return n_local, r_local


def _sweep_split(state, data, idx, batch_size):
    """Deterministic full sweep over fix indices -> (pred, true,
    n_stations, n_stations_local, resolvable_km_local) arrays. Chunked
    at the training batch size so the compiled forward is reused (one
    extra trace for the tail chunk). The local-resolution pair rides
    the sweep because the stations are already in hand per batch."""
    loader = data.loader
    pad_to = loader.inputs.pad_to
    preds = np.empty(len(idx), np.int64)
    trues = np.empty(len(idx), np.int64)
    n_stations = np.empty(len(idx), np.int64)
    n_local = np.empty(len(idx), np.int64)
    r_local = np.empty(len(idx), np.float64)
    for s in range(0, len(idx), batch_size):
        chunk = idx[s:s + batch_size]
        batch = collate([loader.build(int(i)) for i in chunk], pad_to)
        preds[s:s + len(chunk)] = np.argmax(
            np.asarray(_predict(state, batch)), axis=-1)
        trues[s:s + len(chunk)] = np.asarray(batch['y'])
        n_stations[s:s + len(chunk)] = np.asarray(
            batch['meta']['n_stations'])
        n_local[s:s + len(chunk)], r_local[s:s + len(chunk)] = \
            _local_resolution(batch, data.norms)
    return preds, trues, n_stations, n_local, r_local


def _per_storm(sids, correct):
    """[(sid, accuracy, n_fixes), ...] worst-first (ties: bigger storm
    first — more evidence of difficulty)."""
    out = []
    for sid in np.unique(sids):
        m = sids == sid
        out.append((str(sid), float(correct[m].mean()), int(m.sum())))
    return sorted(out, key=lambda r: (r[1], -r[2]))


def _write_predictions_csv(path, fixes, idx, preds, trues, n_stations,
                           n_local, r_local, class_names):
    """One row per fix: identity + true/pred/correct — THE record for
    finding exactly which fixes the model fails on — plus the local
    (±LOCAL_HALF_WIDTH_DEG) station count and resolvable_km: the
    covariate for 'does the model fail where the network is locally
    coarse?' (blank when the local box is empty)."""
    storm_names = fixes.get('name')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sid', 'name', 'time', 'lat', 'lon', 'n_stations',
                    'n_stations_local', 'resolvable_km_local',
                    'true', 'pred', 'correct'])
        for k, i in enumerate(idx):
            w.writerow([
                fixes['sid'][i],
                (str(storm_names[i]).strip()
                 if storm_names is not None else ''),
                str(fixes['time'][i]),
                f"{float(fixes['lat'][i]):.4f}",
                f"{float(fixes['lon'][i]):.4f}",
                int(n_stations[k]),
                int(n_local[k]),
                f'{r_local[k]:.1f}' if np.isfinite(r_local[k]) else '',
                class_names[trues[k]], class_names[preds[k]],
                int(preds[k] == trues[k]),
            ])


def _write_per_storm_csv(path, sids, correct, storm_names=None):
    """Per-storm accuracy, worst-first — which storms are memorised and
    which resist (comparable across runs at fixed split)."""
    storm_names = storm_names or {}
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sid', 'name', 'n_fixes', 'n_correct', 'accuracy'])
        for sid, acc, n in _per_storm(sids, correct):
            w.writerow([sid, storm_names.get(sid, ''), n,
                        int(round(acc * n)), f"{acc:.4f}"])


def _prediction_records(cfg, data, logger, state, global_step, basemap):
    """Per-fix predictions for every DISTINCT split (memorise: val ==
    train — swept once): predictions_<split>.csv +
    per_storm_accuracy_<split>.csv under run_dir, the spatial accuracy
    hexbin via the usual emit path, the identifiability ceiling
    (input_collisions — a second sample-build pass, deliberate: the
    ceiling belongs in the run record next to the accuracy it bounds;
    scalars to the logger, conflict groups to
    identifiability_<split>.json), and a summary line on the run log."""
    fixes = data.loader.fixes
    if 'lat' not in fixes or 'lon' not in fixes:
        return                               # fix table without coords
    batch_size = int(cfg['data'].get('batch_size') or 256)
    domain = cfg['data'].get('domain') or _domain_from_norms(data.norms)
    run_dir = cfg['trainer'].get('run_dir')
    class_names = data.targets.class_names
    # the ceiling hashes ONLY what the model consumes: its encoding.fields
    # (default: every token field) + position — an x field the model never
    # sees must not make two fixes count as distinguishable
    encoding = (cfg.get('model') or {}).get('encoding') or {}
    consumed = tuple(encoding.get('fields') or TOKEN_FIELDS) + ('lat', 'lon')
    seen = {}
    for split, idx in data.splits.items():
        idx = np.asarray(idx)
        if not len(idx):
            continue
        key = idx.tobytes()
        if key in seen:
            RUN_LOG.info(f"  [predictions] {split}: same fixes as "
                         f"{seen[key]!r} — one sweep covers both")
            continue
        seen[key] = split
        preds, trues, n_st, n_local, r_local = _sweep_split(
            state, data, idx, batch_size)
        correct = preds == trues
        sids = np.asarray(fixes['sid'])[idx]
        report = input_collisions(data.loader, idx, fields=consumed)

        if run_dir:
            out = Path(run_dir)
            out.mkdir(parents=True, exist_ok=True)
            storm_names = (dict(zip(sids.tolist(),
                                    np.asarray(fixes['name'])[idx]))
                           if 'name' in fixes else None)
            _write_predictions_csv(out / f'predictions_{split}.csv',
                                   fixes, idx, preds, trues, n_st,
                                   n_local, r_local, class_names)
            _write_per_storm_csv(out / f'per_storm_accuracy_{split}.csv',
                                 sids, correct, storm_names)
            (out / f'identifiability_{split}.json').write_text(json.dumps(
                {k: report[k] for k in ('n_fixes', 'n_unique_inputs',
                                        'n_unmemorisable', 'max_accuracy',
                                        'conflicts')}, indent=2))
            # the records belong ON the run page, not just the run_dir
            for stem, kind in ((f'predictions_{split}', 'csv'),
                               (f'per_storm_accuracy_{split}', 'csv'),
                               (f'identifiability_{split}', 'json')):
                logger.log_artifact(stem, out / f'{stem}.{kind}',
                                    artifact_type='predictions')
        acc = float(correct.mean())
        scalars = {
            f'{split}/memorisation_ceiling': float(report['max_accuracy']),
            f'{split}/n_unmemorisable': int(report['n_unmemorisable']),
            f'{split}/n_unique_inputs': int(report['n_unique_inputs']),
            f'{split}/n_stations_local_mean': float(n_local.mean()),
        }
        # local vs global resolution scalars (the ±5° box vs the FOV);
        # empty local boxes (inf) drop out of the local mean
        finite = np.isfinite(r_local)
        if finite.any():
            scalars[f'{split}/resolvable_km_local_mean'] = \
                float(r_local[finite].mean())
        if domain:
            r_global = np.asarray(
                [network_sparsity(int(n), domain)['resolvable_km']
                 for n in n_st])
            g_ok = np.isfinite(r_global)
            if g_ok.any():
                scalars[f'{split}/resolvable_km_global_mean'] = \
                    float(r_global[g_ok].mean())
        logger.log_metrics(scalars, step=global_step)
        if finite.any():
            RUN_LOG.info(
                f"  [resolution] {split}: local (±{LOCAL_HALF_WIDTH_DEG:g}°)"
                f" mean {r_local[finite].mean():.0f} km over"
                f" {n_local.mean():.1f} stations"
                + (f"; global mean {r_global[g_ok].mean():.0f} km"
                   if domain and g_ok.any() else ''))
        per_storm = _per_storm(sids, correct)
        hardest = ', '.join(f'{s} {a:.2f}' for s, a, _ in per_storm[:5])
        RUN_LOG.info(
            f"  [predictions] {split}: {int(correct.sum())}/{len(idx)} "
            f"correct ({acc:.4f})  ceiling {report['max_accuracy']:.4f} "
            f"({report['n_unmemorisable']} unmemorisable)  "
            f"storms fully memorised "
            f"{sum(a == 1.0 for _, a, _ in per_storm)}/{len(per_storm)}  "
            f"hardest: {hardest}")
        lons = np.asarray(fixes['lon'])[idx]
        lats = np.asarray(fixes['lat'])[idx]
        fig = accuracy_hexbin_figure(
            lons, lats, correct, domain=domain, basemap=basemap,
            title=f'{split} prediction correctness — step {global_step}  '
                  f'accuracy {acc:.4f} (ceiling '
                  f'{report["max_accuracy"]:.4f})')
        _emit(fig, 'accuracy_hexbin', split, global_step, logger, run_dir)

        # storm-track misclassification: the hardest storms as tracks,
        # each fix coloured correct/wrong. Cross-read with the hexbin:
        # a red cell traced by ONE storm's track = storm problem; red
        # shared by many storms' fixes = location/sensing problem.
        times = np.asarray(fixes['time'])[idx]
        for sid, storm_acc, n_fixes in [r for r in per_storm
                                        if r[1] < 1.0][:5]:
            m = sids == sid
            order = np.argsort(times[m])
            fig = storm_track_correctness_figure(
                lons[m][order], lats[m][order], correct[m][order],
                domain=domain, basemap=basemap,
                title=f'{split} {sid} track — '
                      f'{int(correct[m].sum())}/{n_fixes} correct '
                      f'({storm_acc:.2f})  step {global_step}')
            _emit(fig, f'storm_track_{sid}', split, global_step,
                  logger, run_dir)

        # local-vs-global interaction: does the model fail where the
        # network is locally coarse? (binned accuracy vs the per-fix
        # ±5° resolvable_km already computed in the sweep)
        if finite.any():
            fig = accuracy_vs_resolution_figure(
                r_local, correct,
                title=f'{split} accuracy vs local (±{LOCAL_HALF_WIDTH_DEG:g}°)'
                      f' resolution — step {global_step}')
            _emit(fig, 'accuracy_vs_local_resolution', split,
                  global_step, logger, run_dir)


# ---------------------------------------------------------------------------
# end_of_run — post-trainer.test() figures (train.py calls this once)
# ---------------------------------------------------------------------------

def end_of_run(cfg, data, logger, state, global_step,
               n_frames=8, basemap=True):
    """Test confusion matrix + prediction records + storm sequences for
    the best state.

    Runs after trainer.test(): (1) the confusion_matrix callback on the
    test split, if a test stream exists; (1b) the per-fix prediction
    records (_prediction_records: CSVs + per-storm accuracy + spatial
    accuracy hexbin) for every distinct split; (2) for each sid named in the
    data yaml's ``storm_panels: {test: ...}`` knob (sid | [sids] |
    'random' -> one random sid), the sid's fixes — from the test split,
    or the train split when no test split exists (memorise) — are
    time-ordered, evenly subsampled to ``n_frames``, rendered as panels,
    and saved as ONE gif (run_dir/figures/storm_sequence_<sid>.gif,
    logged as an artifact — gif only when there is a run_dir) plus
    first/mid/last stills via the usual svg+png/log_figure path. The
    remaining frames are closed here. ``state`` should be the BEST state
    returned by fit; ``global_step`` the trainer's final step so these
    land on the shared x-axis.
    """
    ctx = {'data': data, 'logger': logger,
           'run_dir': cfg['trainer'].get('run_dir'),
           'storm_panels': cfg['data'].get('storm_panels'),
           'domain': cfg['data'].get('domain')}
    if 'test' in data.streams:
        CALLBACKS.get('confusion_matrix', ctx=ctx, split='test')(
            state, 0, global_step)
    if data.loader is not None:
        _prediction_records(cfg, data, logger, state, global_step, basemap)

    sel = (ctx['storm_panels'] or {}).get('test')
    if sel is None or data.loader is None:
        return
    # sequences come from the test split when one exists; scenarios
    # without one (memorise: train == val == all fixes) fall back to
    # train so the showpiece storms still render
    src = 'test' if 'test' in data.splits else 'train'
    if src not in data.splits:
        return
    split_idx = np.asarray(data.splits[src])
    split_sids = np.asarray(data.loader.fixes['sid'])[split_idx]
    if isinstance(sel, (list, tuple)):
        sids = [str(s) for s in sel]
    elif str(sel) == 'random':
        rng = np.random.default_rng(global_step)
        sids = [str(rng.choice(np.unique(split_sids)))]
    else:
        sids = [str(sel)]
    domain = ctx['domain'] or _domain_from_norms(data.norms)
    run_dir = ctx['run_dir']

    for sid in sids:
        pool = split_idx[split_sids == sid]
        if not len(pool):
            raise ValueError(
                f"storm_panels['test'] = {sid!r} matches no fix in the "
                f"{src!r} split — check the sid against the library.")
        pool = pool[np.argsort(np.asarray(data.loader.fixes['time'])[pool])]
        if len(pool) > n_frames:
            keep = np.linspace(0, len(pool) - 1, n_frames).round().astype(int)
            pool = pool[keep]

        figs = [_render_fix(state, data, int(i), domain, basemap)
                for i in pool]
        if run_dir:
            fig_dir = Path(run_dir) / 'figures'
            fig_dir.mkdir(parents=True, exist_ok=True)
            gif = save_gif(figs, str(fig_dir / f'storm_sequence_{sid}.gif'))
            logger.log_artifact(f'storm_sequence_{sid}', gif,
                                artifact_type='figure')
        stills = sorted({0, len(figs) // 2, len(figs) - 1})
        for k, fig in enumerate(figs):
            if k in stills:      # _emit hands to the logger, which closes
                _emit(fig, f'storm_sequence_{sid}_f{k}', src,
                      global_step, logger, run_dir)
            else:
                plt.close(fig)
