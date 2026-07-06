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
                      the annotated figure
    storm_panel       one fix (random | pinned sid per the data yaml knob)
                      -> truth-star/pred-ring map via visualise.figures

Figures go to logger.log_figure (all backends) AND run_dir/figures/ as
svg + png (editable-vector workflow). The logger closes each figure.
"""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from training.metrics import compute_final_metrics, update_cm
from utils.plotting.fields import confusion_matrix_figure
from utils.registry import Registry

from experiments.cyclone_jax.data.batching import collate
from experiments.cyclone_jax.data.sparsity import network_sparsity
from experiments.cyclone_jax.visualise.figures import (save_gif,
                                                       storm_panel_figure)

CALLBACKS = Registry('Callback')

_CALLBACK_SPEC_KEYS = {'name', 'every', 'split', 'kwargs'}


# ---------------------------------------------------------------------------
# Shared mechanics
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=0)
def _jit_apply(apply_fn, variables, X):
    return apply_fn(variables, X, train=False)


def _predict(state, batch):
    """Model forward on one collated batch (eval mode; meta never traced).

    Jitted (apply_fn static): full-split CM sweeps reuse one compiled
    forward instead of running op-by-op; pad_to keeps shapes fixed, so it
    traces once per batch shape (stream batch + the B=1 panel batch).
    """
    variables = {'params': state.params}
    if getattr(state, 'batch_stats', None) is not None:
        variables['batch_stats'] = state.batch_stats
    return _jit_apply(state.apply_fn, variables, batch['X'])


def _emit(fig, name, split, global_step, logger, run_dir):
    """Save svg + png stills under run_dir/figures/, then hand the figure
    to the logger (which closes it — save first)."""
    if run_dir:
        fig_dir = Path(run_dir) / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        stem = str(fig_dir / f'{name}_{split}_step{global_step:07d}')
        fig.savefig(stem + '.svg')
        fig.savefig(stem + '.png', dpi=150)
    logger.log_figure(f'{split}/{name}', fig, global_step)


def _domain_from_norms(norms):
    """Coordinate-scaling bounds double as the effective FOV when the data
    yaml has no explicit domain block (both feed the same stats record)."""
    if norms is None:
        return None
    return {'lat': [norms.stats['lat']['min'], norms.stats['lat']['max']],
            'lon': [norms.stats['lon']['min'], norms.stats['lon']['max']]}


def _render_fix(state, data, i, domain, basemap):
    """One fix index -> titled truth-star/pred-ring panel. Shared by the
    storm_panel callback and the end-of-run storm sequence."""
    loader, targets = data.loader, data.targets
    batch = collate([loader.build(i)], loader.inputs.pad_to)
    pred = int(np.argmax(np.asarray(_predict(state, batch))[0]))
    true = int(batch['y'][0])
    meta, mask = batch['meta'], batch['X']['station_mask'][0]
    lat = np.asarray(batch['X']['lat'][0])[mask]
    lon = np.asarray(batch['X']['lon'][0])[mask]
    if data.norms is not None:
        lat, lon = data.norms.invert_coords(lat, lon)

    n = int(meta['n_stations'][0])
    title = (f"{meta['sid'][0]}  {str(meta['time'][0])[:16]}  "
             f"true {targets.class_names[true]} vs "
             f"pred {targets.class_names[pred]}  n={n}")
    if domain:
        r_km = network_sparsity(n, domain)['resolvable_km']
        title += f"  resolvable {r_km:.0f} km"
    return storm_panel_figure(
        lon, lat, float(meta['lon'][0]), float(meta['lat'][0]),
        true, pred, targets.n_classes, title=title,
        domain=domain, basemap=basemap)


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
# end_of_run — post-trainer.test() figures (train.py calls this once)
# ---------------------------------------------------------------------------

def end_of_run(cfg, data, logger, state, global_step,
               n_frames=8, basemap=True):
    """Test confusion matrix + storm sequence for the best state.

    Runs after trainer.test(): (1) the confusion_matrix callback on the
    test split, if a test stream exists; (2) for each sid named in the
    data yaml's ``storm_panels: {test: ...}`` knob (sid | [sids] |
    'random' -> one random test sid), the sid's test-split fixes are
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

    sel = (ctx['storm_panels'] or {}).get('test')
    if sel is None or data.loader is None or 'test' not in data.splits:
        return
    split_idx = np.asarray(data.splits['test'])
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
                f"test split — check the sid against the library.")
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
                _emit(fig, f'storm_sequence_{sid}_f{k}', 'test',
                      global_step, logger, run_dir)
            else:
                plt.close(fig)
