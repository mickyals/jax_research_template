"""
experiments/tc_perceiver_io/train/train.py

Entry point for the tc_perceiver_io experiment.

Usage
-----
    python -m experiments.tc_perceiver_io.train.train \
        jrt/experiments/tc_perceiver_io/configs/train.yaml

    # Resume an interrupted run
    python -m experiments.tc_perceiver_io.train.train \
        jrt/experiments/tc_perceiver_io/configs/train.yaml \
        --resume
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
import yaml

from experiments.tc_perceiver_io.data.datamodule import TCDataModule
from experiments.tc_perceiver_io.train.evaluate import (
    collect_predictions,
    build_prediction_outputs,
    confusion_matrix,
    domain_latlon_for_sample,
    per_class_metrics,
)
from experiments.tc_perceiver_io.plotting.plotting import (
    plot_attention_geographic,
    plot_attention_matrix_grid,
    plot_decoder_query,
    plot_class_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_pr_curves_per_class,
)
from experiments.tc_perceiver_io.train.metrics import build_metrics_fns
from experiments.tc_perceiver_io.train.model import TCPerceiverIO
from datasets.class_weights import class_weights_from_counts
from training.metrics import (
    binary_pr_curve,
    compute_full_set_metrics,
    cross_entropy,
    per_class_pr_curves,
)
from training.trainer import Trainer, TrainState
from utils.jax_core.diagnostics import (
    model_tabulate,
    visualize_weight_distribution,
    visualize_gradients,
    visualize_activations,
    plot_loss_landscape,
)


def _should_run(epoch: int, every_n_epochs: int) -> bool:
    """Whether an epoch-cadence callback fires this epoch.

    Shared gate for the epoch-level callbacks (attention figures, gradient
    histograms, eval plots): True when ``every_n_epochs`` is positive and the
    epoch is a multiple of it. ``0`` (or negative) disables the callback.
    """
    return every_n_epochs > 0 and epoch % every_n_epochs == 0


# ---------------------------------------------------------------------------
# Attention entropy callback
# ---------------------------------------------------------------------------

def _make_attn_entropy_callback(
    model:       TCPerceiverIO,
    probe_batch: dict,
    logger,
) -> Callable[[TrainState, int, int], None]:
    """Return a **step-level** callback logging Read cross-attention entropy.

    Mean entropy (over batch, heads, latents) of each latent's attention over
    the M stations — softmax of the pre-softmax Read scores. A falling curve
    means latents are concentrating on fewer stations rather than attending
    uniformly, a useful proxy for learning progress. Logged at step=global_step
    so it aligns with the step-level training loss.

    Parameters
    ----------
    model : TCPerceiverIO
    probe_batch : dict
        A fixed validation batch held in memory for the run duration.
    logger : experiment logger
        Must expose ``log_metrics``.
    """
    probe_X = probe_batch['X']

    @jax.jit
    def _entropy(params):
        _, attn = model.apply({'params': params}, probe_X,
                              train=False, return_weights=True)
        p   = jax.nn.softmax(attn['read'], axis=-1)        # (B, H, N, M)
        ent = -jnp.sum(p * jnp.log(p + 1e-12), axis=-1)    # (B, H, N)
        return jnp.mean(ent)

    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        logger.log_metrics(
            {'val/attn_entropy': float(_entropy(state.params))},
            step=global_step,
        )

    return callback


def _make_attn_figure_callback(
    model:             TCPerceiverIO,
    probe_batch:       dict,
    logger,
    class_names:       list[str],
    location_encoding: str = 'unit_circle',
    radius_km:         float = 500.0,
    fov_lat:           tuple[float, float] | None = None,
    fov_lon:           tuple[float, float] | None = None,
    fig_every:         int = 5,
) -> Callable[[TrainState, int, int], None]:
    """Return an **epoch-level** callback logging the per-component attention maps.

    All three Perceiver-IO components are rendered from the fixed VAL probe
    sample (softmax of the pre-softmax scores) every ``fig_every`` epochs:

    ``val/attn_read_map`` — geographic Read map: per-station attention (mean
        over latents + heads) on the station geometry (unit_circle local x-y
        with km rings, or domain lat/lon).
    ``val/attn_processor_grid`` — layers × heads grid of the N×N latent
        self-attention matrices.
    ``val/attn_decoder_query`` — heads × latents heatmap of the Decoder output
        query's attention (only for decode_mode='attention'; absent for
        'avgproj').

    Parameters
    ----------
    model : TCPerceiverIO
    probe_batch : dict
        A fixed validation batch held in memory for the run duration.
    logger : experiment logger
        Must expose ``log_figure``.
    class_names : list[str]
        For the figure caption (true/pred of the probe sample).
    location_encoding : {'unit_circle', 'domain'}
        Coordinate convention for the Read map. For 'domain' the probe's
        station/query positions are decoded once here (the probe is fixed).
    radius_km : float
        Search radius (unit_circle Read-map km-ring labels).
    fov_lat, fov_lon : tuple, optional
        Domain field-of-view (required when location_encoding='domain').
    fig_every : int
        Log figures every this many epochs. 0 = never. Default 5.
    """
    probe_X = probe_batch['X']

    # The probe batch is fixed, so for domain mode decode its sample-0
    # station/query positions once (plotting does not import the encoding).
    station_latlon = query_latlon = None
    if location_encoding == 'domain':
        station_latlon, query_latlon = domain_latlon_for_sample(
            probe_batch, 0, fov_lat, fov_lon)

    @jax.jit
    def _attn(params):
        logits, attn = model.apply({'params': params}, probe_X,
                                   train=False, return_weights=True)
        # softmax the pre-softmax scores (over the LAST axis) so each plotter
        # renders proper attention distributions.
        read = jax.nn.softmax(attn['read'], axis=-1)          # (B, H, N, M)
        proc = jax.nn.softmax(attn['processor'], axis=-1)     # (L, B, H, N, N)
        dec  = attn.get('decoder')                            # (B, H, 1, N) | None
        if dec is not None:
            dec = jax.nn.softmax(dec, axis=-1)
        return logits, read, proc, dec

    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        if not _should_run(epoch, fig_every):
            return
        logits, read, proc, dec = _attn(state.params)
        true_c = class_names[int(probe_batch['y'][0])]
        pred_c = class_names[int(np.asarray(logits)[0].argmax())]
        caption = f'true: {true_c}, pred: {pred_c}'

        fig_read = plot_attention_geographic(
            np.asarray(read), probe_batch,
            location_encoding=location_encoding,
            fov_lat=fov_lat, fov_lon=fov_lon, radius_km=radius_km,
            sample_idx=0,
            station_latlon=station_latlon, query_latlon=query_latlon,
            title=caption,
        )
        logger.log_figure('val/attn_read_map', fig_read, step=global_step)

        fig_grid = plot_attention_matrix_grid(
            np.asarray(proc), sample_idx=0,
            title=f'Processor self-attention — {caption}',
        )
        logger.log_figure('val/attn_processor_grid', fig_grid, step=global_step)

        if dec is not None:
            fig_dec = plot_decoder_query(
                np.asarray(dec), sample_idx=0,
                title=f'Decoder output-query attention — {caption}',
            )
            logger.log_figure('val/attn_decoder_query', fig_dec, step=global_step)

    return callback


# ---------------------------------------------------------------------------
# Gradient-flow callback (TRAIN probe only)
# ---------------------------------------------------------------------------

def _grad_flow_keep(name: str, last_block: int) -> bool:
    """Keep only the OUTPUT projection of each Perceiver stage for grad-flow.

    A full per-leaf dump (~70 histograms) is too much to review. The three
    seams that matter for vanishing/exploding diagnosis are the layer that
    produces each stage's output: Read's FFN output, the LAST Processor block's
    FFN output, and the Decoder's classifier head. ``last_block`` is the index
    of the deepest Processor block (so this tracks num_process_layers).
    """
    return (
        name.startswith('read/mlp/output_layer')
        or name.startswith(f'processor/blocks_{last_block}/mlp/output_layer')
        or name.startswith('decoder/head')
    )


def _make_grad_flow_callback(
    model:       TCPerceiverIO,
    probe_batch: dict,
    logger,
    every_n_epochs:   int  = 5,
    final_layers_only: bool = True,
) -> Callable[[TrainState, int, int], None]:
    """Return an epoch-level callback that logs gradient histograms.

    Computes jax.grad of the cross-entropy loss on a fixed TRAIN probe batch and
    pushes gradient histograms named by tree path (e.g.
    ``grad_flow/processor/blocks_1/mlp/output_layer/kernel``) via
    ``logger.log_histogram``. Vanishing/exploding stages show up as histograms
    collapsing to 0 or blowing up.

    By default (``final_layers_only=True``) only the OUTPUT projection of each
    stage is logged — Read's FFN output, the last Processor block's FFN output,
    and the Decoder head — which is enough to read gradient flow across the
    Read→Process→Decode path without dumping every leaf. Set False to log every
    parameter leaf.

    Call once manually with the freshly initialised state for the init
    snapshot, then register as an epoch callback.

    Parameters
    ----------
    model : TCPerceiverIO
    probe_batch : dict
        A fixed TRAINING batch held in memory for the run duration
        (gradient flow is a training diagnostic — never wired to val/test).
    logger : experiment logger
        Must expose ``log_histogram``.
    every_n_epochs : int
        Log every this many epochs. 0 = never. Default 5.
    final_layers_only : bool
        Restrict to each stage's output layer (default True). False = all leaves.
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
        names = [
            '/'.join(getattr(k, 'key', getattr(k, 'name', str(k))) for k in path)
            for path, _ in leaves
        ]
        # Deepest Processor block index, so the filter tracks num_process_layers.
        last_block = max(
            (int(n.split('/')[1].removeprefix('blocks_'))
             for n in names if n.startswith('processor/blocks_')),
            default=0,
        )
        for name, (_, leaf) in zip(names, leaves):
            if final_layers_only and not _grad_flow_keep(name, last_block):
                continue
            logger.log_histogram(
                f'grad_flow/{name}', np.asarray(leaf).ravel(), step=step,
            )

    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        if not _should_run(epoch, every_n_epochs):
            return
        _log(state.params, global_step)

    callback.log_now = _log   # exposed for the init-time snapshot
    return callback


def _make_eval_plots_callback(
    model:       TCPerceiverIO,
    val_loader,
    logger,
    class_names: list[str],
    every_n_epochs: int = 1,
    prefix:      str = 'val',
    spatial_maps: bool = False,
    fov_lat=None,
    fov_lon=None,
    geo: bool = False,
    csv_dir=None,
) -> Callable[[TrainState, int, int], None]:
    """Return an epoch-level callback that logs confusion / per-class figures
    and full-set scalar metrics under ``<prefix>/...``.

    Runs a full forward pass over the loader to collect predictions, then plots
    and uploads the confusion matrix + per-class F1 to WandB as images (Media →
    Images), and logs the full-set scalars ``<prefix>/mAP`` (macro one-vs-rest
    average precision) and ``<prefix>/pr_auc`` (binary TC-vs-background PR-AUC)
    from training/metrics.py over the accumulated logits/labels — these
    integrate a PR curve over the whole split, so they cannot be per-batch.

    Parameters
    ----------
    model : TCPerceiverIO
    val_loader : TCLoader
        Re-iterable loader — iterated fresh on each callback invocation.
    logger : experiment logger
        Must expose ``log_figure`` and ``log_metrics``.
    every_n_epochs : int
        How often to run. 0 = disabled. Default 1 (every epoch).
    prefix : str
        Metric/figure namespace ('val' during training; 'test' for the one-shot
        end-of-training pass).
    """
    def callback(state: TrainState, epoch: int, global_step: int) -> None:
        if not _should_run(epoch, every_n_epochs):
            return

        variables = {'params': state.params}
        preds, labels, logits, meta = collect_predictions(model, variables, val_loader)

        # Full-set scalars (PR-curve based — not per-batch averageable).
        logger.log_metrics(
            {f'{prefix}/{k}': v
             for k, v in compute_full_set_metrics(logits, labels).items()},
            step=global_step,
        )

        cm  = confusion_matrix(preds, labels)
        pcm = per_class_metrics(cm)

        fig_norm = plot_confusion_matrix(
            cm, class_names, normalize=True,
            title=f'{prefix.capitalize()} confusion matrix (recall per class)',
        )
        fig_raw = plot_confusion_matrix(
            cm, class_names, normalize=False,
            title=f'{prefix.capitalize()} confusion matrix (counts)',
        )
        fig_cls = plot_class_metrics(
            pcm, class_names,
        )

        logger.log_figure(f'{prefix}/confusion_norm',    fig_norm, step=global_step)
        logger.log_figure(f'{prefix}/confusion_counts',  fig_raw,  step=global_step)
        logger.log_figure(f'{prefix}/per_class_metrics', fig_cls,  step=global_step)

        # Precision-recall CURVES (the shape behind the pr_auc / mAP scalars):
        # binary TC-vs-background detection + per-class one-vs-rest overlay.
        fig_pr = plot_pr_curve(
            binary_pr_curve(logits, labels),
            title=f'{prefix.capitalize()} PR — TC vs. background detection',
        )
        fig_pr_cls = plot_pr_curves_per_class(
            per_class_pr_curves(logits, labels), class_names,
            title=f'{prefix.capitalize()} per-class PR (one-vs-rest)',
        )
        logger.log_figure(f'{prefix}/pr_curve',           fig_pr,     step=global_step)
        logger.log_figure(f'{prefix}/pr_curves_per_class', fig_pr_cls, step=global_step)

        # Per-sample table + per-class spatial maps from ONE source (the maps
        # derive from the table's own columns — see build_prediction_outputs),
        # so the logged CSV and the maps describe the same rows. Both are opt-in
        # and used for the end-of-train TEST pass, giving the wandb test/ section
        # the full study table + WHERE the model classifies each class well.
        if spatial_maps or csv_dir is not None:
            table, spatial_figs = build_prediction_outputs(
                preds, labels, logits, meta, class_names,
                fov_lat=fov_lat, fov_lon=fov_lon, geo=geo, make_maps=spatial_maps)
            for name, fig_sp in spatial_figs.items():
                logger.log_figure(f'{prefix}/spatial_pred/{name}', fig_sp,
                                  step=global_step)
            if csv_dir is not None:
                csv_path = Path(csv_dir) / f'{prefix}_per_sample.csv'
                table.to_csv(csv_path, index=False)
                logger.log_artifact(f'{prefix}_per_sample', csv_path,
                                    artifact_type='predictions')

    return callback


# ---------------------------------------------------------------------------
# Diagnostics row (weight / gradient / activation distributions + loss landscape)
# ---------------------------------------------------------------------------

def _log_diagnostics(
    model:       TCPerceiverIO,
    params:      dict,
    train_probe: dict,
    val_probe:   dict,
    logger,
    step:        int,
    loss_landscape_grid: int = 0,
) -> None:
    """Log the 'diagnostics/' figure row (weight/grad/activation, + optional
    loss landscape) for one parameter snapshot.

    Called twice from train(): once at init (step 0) for the start-of-training
    reference, and once on the best params after fit() for the end-of-training
    picture. Gradients use the TRAIN probe (training signal); activations use
    the VAL probe at train=False. The loss landscape (grid_size² forward passes)
    is rendered only when loss_landscape_grid > 0 — keep it for the end snapshot.
    """
    def _loss_fn(p):
        logits = model.apply({'params': p}, train_probe['X'], train=False)
        return cross_entropy(logits, train_probe['y'])

    logger.log_figure(
        'diagnostics/weight_dist',
        visualize_weight_distribution(params), step=step)
    logger.log_figure(
        'diagnostics/gradients',
        visualize_gradients(params, _loss_fn), step=step)
    logger.log_figure(
        'diagnostics/activations',
        visualize_activations(model, {'params': params}, val_probe['X'], train=False),
        step=step)
    if loss_landscape_grid and loss_landscape_grid > 0:
        logger.log_figure(
            'diagnostics/loss_landscape',
            plot_loss_landscape(params, _loss_fn, grid_size=loss_landscape_grid),
            step=step)


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


def _print_token_summary(config: dict) -> None:
    """Print the per-station token width so it is never a mystery.

    A station token is one linear projection of [obs; (mask); Fourier(coords)].
    The raw concatenation width is ``F (+ F mask) + fourier_dim``; ``token_proj``
    maps it to ``embed_dim`` (D). The sequence length is ``max_stations``.
    """
    m       = config['model']
    F       = int(m['n_obs_features'])
    fdim    = int(m.get('fourier_dim', 64))
    miss    = bool(m.get('missingness_indicator', True))
    embed   = int(m['embed_dim'])
    max_st  = config.get('data', {}).get('max_stations')
    obs_part = f"obs {F}" + (f" + mask {F}" if miss else "")
    raw      = (2 * F if miss else F) + fdim
    print(f"  token input : {raw}d   [{obs_part} + Fourier(coords) {fdim}]")
    print(f"  token_proj  : {raw}d -> {embed}d   (embed_dim D)")
    if max_st:
        print(f"  sequence    : up to {max_st} stations (padded)")


def _background_count(data_cfg: dict, batch_size: int,
                      steps_per_epoch: Optional[int], n_tc_total: int) -> int:
    """Effective class-0 (background) sample count for class weighting.

    Background is synthesised (random FOV point × pool timestamp), so unlike the
    TC classes it has no fixed dataset count — its count for the effective-number
    weighting is a hyperparameter. Resolution:
      * explicit ``data.n_background`` wins;
      * else the realized per-epoch sampling count ``steps_per_epoch × bg_half``
        (random mode), which is what the model actually sees;
      * else (sequential mode) the ratio-consistent count
        ``n_tc_total × bg_half / tc_half`` so class 0 sits at the same TC:bg
        ratio the loader produces.
    """
    n = data_cfg.get('n_background')
    if n is not None:
        return int(n)
    tc_frac = data_cfg.get('tc_fraction', 0.5)
    if isinstance(tc_frac, dict):
        tc_frac = tc_frac.get('train', 0.5)
    tc_half = max(1, round(batch_size * tc_frac))
    bg_half = batch_size - tc_half
    if steps_per_epoch:
        return int(steps_per_epoch * bg_half)
    return int(round(n_tc_total * bg_half / tc_half))


def _resolve_run_dir(trainer_cfg: dict, experiment_dir: Path,
                     name: Optional[str] = None) -> Path:
    """Resolve the run directory, auto-incrementing under a run_group.

    Precedence:
      * an explicit ``trainer.run_dir`` is used as-is (pin a fixed directory;
        relative paths anchor to the experiment dir) — back-compat.
      * otherwise ``trainer.run_group`` is treated as the parent and the next
        ``run_NN`` (max existing + 1) is created under it, so forgetting to bump
        a number can never clobber an earlier run. ``name`` (the run's purpose
        slug, from --name) is appended as ``run_NN-<name>`` for legibility.

    Raises if neither key is set.
    """
    explicit = trainer_cfg.get('run_dir')
    if explicit:
        rd = Path(explicit)
        return rd if rd.is_absolute() else experiment_dir / rd

    group = trainer_cfg.get('run_group')
    if not group:
        raise ValueError(
            "Set trainer.run_dir (a fixed directory) or trainer.run_group "
            "(parent under which run_NN is auto-created)."
        )
    group_dir = Path(group)
    group_dir = group_dir if group_dir.is_absolute() else experiment_dir / group_dir
    group_dir.mkdir(parents=True, exist_ok=True)

    nums = [
        int(mt.group(1))
        for p in group_dir.iterdir() if p.is_dir()
        if (mt := re.match(r'run_(\d+)', p.name))
    ]
    leaf = f"run_{max(nums, default=0) + 1:02d}"
    if name:
        leaf += f"-{name}"
    return group_dir / leaf


def _resolve_schedule_steps(trainer_cfg: dict) -> None:
    """Fill a cosine schedule's ``decay_steps`` to span the run when omitted.

    The Trainer is schedule-agnostic — it forwards ``scheduler_kwargs`` verbatim
    to optax and never relates ``decay_steps`` to the run length. For the cosine
    schedules (``warmup_cosine`` / ``cosine``) ``decay_steps`` is the TOTAL number
    of steps over which the LR anneals to ``end_value`` (warmup included, per the
    optax convention); set shorter than the run, the LR floors at ``end_value``
    for every remaining step. So when ``decay_steps`` is omitted (or null) we
    derive it here, in the experiment glue, as ``num_epochs * steps_per_epoch``
    so the anneal covers all of training. An explicit ``decay_steps`` is always
    respected (this only fills the gap).

    No-op for non-cosine schedules and for sequential mode (``steps_per_epoch``
    unset), where the total step count is not known until the loader is built —
    set ``decay_steps`` explicitly in that case.
    """
    if trainer_cfg.get('scheduler') not in ('warmup_cosine', 'cosine'):
        return
    sk = trainer_cfg.setdefault('scheduler_kwargs', {})
    if sk.get('decay_steps') is not None:
        return                                   # explicit — respect it
    spe = trainer_cfg.get('steps_per_epoch')
    if not spe:
        print("  [schedule] decay_steps omitted but steps_per_epoch is unset "
              "(sequential mode) — set scheduler_kwargs.decay_steps explicitly "
              "so the cosine anneal spans the run.")
        return
    total = int(trainer_cfg.get('num_epochs', 1)) * int(spe)
    sk['decay_steps'] = total
    print(f"  [schedule] decay_steps auto-set to num_epochs * steps_per_epoch "
          f"= {trainer_cfg.get('num_epochs', 1)} * {int(spe)} = {total}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(config_path: str | Path, resume: bool = False,
          name: Optional[str] = None) -> None:
    """Run training from a YAML config file.

    Parameters
    ----------
    config_path : str or Path
        Path to train.yaml (or any config following the same schema).
    resume : bool
        If True, resume from the latest checkpoint in trainer.checkpoint_dir.
    name : str, optional
        Short purpose/hypothesis slug for this run (e.g. ``tcfrac0.3-cw-effnum``).
        Appended to the auto-incremented run dir (``run_NN-<name>`` under
        trainer.run_group) and used as the WandB run name. Ignored when a fixed
        trainer.run_dir is set (the dir is pinned), though it still names WandB.
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

    # Cosine LR schedules: fill decay_steps to span the whole run when the
    # config leaves it open (omitted/null). The Trainer forwards scheduler_kwargs
    # verbatim to optax, so without this the anneal would complete in decay_steps
    # and then sit at end_value for the rest of training.
    _resolve_schedule_steps(trainer_cfg)

    # location_encoding picks the coordinate convention for the datamodule's
    # encoder. The model is coordinate-agnostic (a single projection of
    # whatever coords it is handed), so it is injected into the data block only.
    loc_enc = config.get('location_encoding', 'unit_circle')
    config['data']['location_encoding']  = loc_enc
    # The model is coordinate-agnostic (the learned latent array is the query;
    # there is no query token to position), so location_encoding configures the
    # datamodule only — nothing is injected into config['model'].

    # ------------------------------------------------------------------
    # Resolve run_dir relative to the experiment root (two levels up from this
    # script, which lives in train/), NOT the config file directory (configs/),
    # so a relative run path always expands under <experiment_dir>/runs/...
    # regardless of where the CLI is invoked from. Either pin trainer.run_dir or
    # let trainer.run_group auto-increment run_NN (so a forgotten number never
    # clobbers an earlier run); --name appends the run's purpose slug.
    # ------------------------------------------------------------------
    _experiment_dir = Path(__file__).resolve().parent.parent
    run_dir = _resolve_run_dir(trainer_cfg, _experiment_dir, name=name)
    trainer_cfg['run_dir'] = str(run_dir)
    print(f"  run_dir     : {run_dir}")

    # --name sets the WandB run NAME (the run's purpose/hypothesis); tags stay as
    # facets and the full config is in hparams.json. Overrides log_kwargs.name.
    if name:
        trainer_cfg.setdefault('log_kwargs', {})['name'] = name

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dm = TCDataModule.from_config(config['data'])

    # TargetSpec (data.target) is the single source of truth for the head size,
    # class names, and default loss. Sync the model's n_classes to it so the
    # head always matches the chosen target.
    target_spec = dm.target_spec
    class_names = target_spec.class_names
    config['model']['n_classes'] = target_spec.n_classes

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
    model = TCPerceiverIO(**config['model'])

    # Print a per-layer shape + parameter count table before training.
    # Peek a small batch (4 samples) so Flax can trace all tensor shapes.
    # train_probe_batch doubles as the fixed probe for the gradient-flow
    # callback below (gradient flow is a TRAIN diagnostic).
    train_probe_batch = next(iter(train_loader))
    _exmp  = {k: v[:4] for k, v in train_probe_batch['X'].items()}
    print()
    print("─" * 58)
    print("Model  (TCPerceiverIO)")
    _print_token_summary(config)
    model_tabulate(model, _exmp, False)   # args: X dict, train=False
    del _exmp

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    # Optional class weighting: derive per-class weights from the train split's
    # realized class counts (computed once here, not per batch) and feed them to
    # the loss via loss_kwargs['class_weights']. The realized vector is recorded
    # in the manifest so the run is reproducible from its artifact.
    manifest    = dm.manifest()
    loss_kwargs = dict(trainer_cfg.get('loss_kwargs') or {})
    cw_scheme   = config['data'].get('class_weight_scheme', 'none')
    # Precedence: an explicit loss_kwargs.class_weights wins; otherwise a
    # data.class_weight_scheme is computed from the train split's class counts.
    if 'class_weights' not in loss_kwargs and cw_scheme != 'none':
        n_classes = target_spec.n_classes
        counts = [int(manifest['train']['class_counts'].get(str(c), 0))
                  for c in range(n_classes)]
        # Fold background (class 0) in as a real class with a count (its count is
        # a hyperparameter — see _background_count) instead of pinning it at 1.0.
        counts[0] = _background_count(
            config['data'], trainer_cfg['batch_size'], steps_per_epoch,
            n_tc_total=sum(counts))
        cw = class_weights_from_counts(
            counts,
            scheme    = cw_scheme,
            beta      = config['data'].get('class_weight_beta', 0.999),
            normalize = config['data'].get('class_weight_normalize', True),
        )
        loss_kwargs['class_weights'] = cw.tolist()
        manifest['train']['class_weights'] = {
            'scheme':           cw_scheme,
            'n_background':     counts[0],
            'effective_counts': counts,        # all 9, incl. folded-in background
            'weights':          cw.tolist(),
        }
        print(f"  class counts (incl. background) : {counts}")
        print(f"  class weighting [{cw_scheme}]   : {np.round(cw, 3).tolist()}")

    metrics_fns = build_metrics_fns(
        loss        = trainer_cfg.get('loss', target_spec.loss),
        loss_kwargs = loss_kwargs,
        metrics     = trainer_cfg.get('metrics'),
    )

    # Fail fast if early stopping watches a metric that isn't being reported
    # (only 'loss' is guaranteed; everything else must be listed in
    # trainer.metrics). The patience metric is logged as '<split>/<name>'.
    patience_metric = trainer_cfg.get('patience_metric', 'val/loss')
    pm_name = patience_metric.split('/')[-1]
    if pm_name not in metrics_fns:
        raise ValueError(
            f"trainer.patience_metric={patience_metric!r} watches '{pm_name}', "
            f"which is not a reported metric. Add it to trainer.metrics "
            f"(have: {sorted(metrics_fns)}) or set patience_metric to val/loss."
        )

    trainer     = Trainer(model, metrics_fns, trainer_cfg)

    # Log full config so every run is reproducible from its artifact
    trainer.log_hyperparams(config)

    # Persist the resolved data split (manifest.json next to checkpoints +
    # logger copy) — the durable answer to "what did this run train on"
    trainer.write_manifest(manifest)

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
                                   class_names=class_names,
                                   location_encoding=loc_enc,
                                   radius_km=config['data'].get('radius_km', 500.0),
                                   fov_lat=config['data'].get('fov_lat'),
                                   fov_lon=config['data'].get('fov_lon'),
                                   fig_every=fig_every),
        _make_eval_plots_callback(model, val_loader, trainer.logger,
                                  class_names=class_names,
                                  every_n_epochs=eval_plots_every),
        grad_flow_cb,
    ]

    # Diagnostics row: weight/grad/activation distributions (and optionally a
    # loss landscape). Logged at init (start-of-training reference) and again on
    # the best params after fit(). Loss landscape is opt-in via its grid size.
    diagnostics    = trainer_cfg.get('diagnostics', True)
    ll_grid        = int(trainer_cfg.get('diagnostics_loss_landscape_grid', 0))

    if grad_hist_every > 0 or diagnostics:
        # fit() re-creates the state with the same seed, so this initial
        # state is the true step-0 snapshot.
        init_state = trainer.init_state(train_probe_batch)
        if grad_hist_every > 0:
            grad_flow_cb.log_now(init_state.params, step=0)
        if diagnostics:
            # No loss landscape at init — it characterises the trained minimum.
            _log_diagnostics(model, init_state.params, train_probe_batch,
                             probe_batch, trainer.logger, step=0,
                             loss_landscape_grid=0)
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

    # End-of-training diagnostics on the best params (+ loss landscape).
    if diagnostics:
        _log_diagnostics(model, best_state.params, train_probe_batch,
                         probe_batch, trainer.logger, step=int(best_state.step),
                         loss_landscape_grid=ll_grid)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    test_metrics = trainer.test(test_loader)
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.5f}")

    # One-shot eval-plots on the TEST split (best params): test/confusion_*,
    # test/per_class_metrics figures + test/mAP + test/pr_auc full-set scalars —
    # trainer.test() only reports the per-batch scalars, so this fills in the
    # per-class picture without a separate evaluate.py run.
    _make_eval_plots_callback(
        model, test_loader, trainer.logger, class_names=class_names,
        every_n_epochs=1, prefix='test',
        spatial_maps=True,
        fov_lat=config['data'].get('fov_lat'),
        fov_lon=config['data'].get('fov_lon'),
        geo=bool(config['data'].get('eval_geo_maps', False)),
        csv_dir=run_dir,
    )(best_state, epoch=0, global_step=int(best_state.step))

    # Finalize logger here, after test(), so test metrics are logged before
    # the WandB run is closed.  fit() no longer calls finalize() internally.
    trainer.logger.finalize("completed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pin_gpu(cli_gpu: Optional[str], config_path: str | Path) -> None:
    """Pin training to a single GPU via CUDA_VISIBLE_DEVICES.

    JAX otherwise claims EVERY visible GPU and preallocates memory on each — not
    what you want for single-device training on a shared multi-GPU box. The
    device is resolved from ``--gpu``, else the config's top-level ``gpu`` field.
    A ``CUDA_VISIBLE_DEVICES`` already set in the shell always wins.

    Must run before the first JAX device op. JAX's GPU backend initialises
    lazily (not at ``import jax``), so calling this at the top of ``__main__``,
    before ``train()``, is sufficient. For an absolute guarantee, export
    ``CUDA_VISIBLE_DEVICES`` in the shell instead.
    """
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        return                                   # shell setting wins
    gpu = cli_gpu
    if gpu is None:
        try:
            with open(config_path, encoding='utf-8') as f:
                gpu = yaml.safe_load(f).get('gpu')
        except (OSError, yaml.YAMLError):
            gpu = None
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        print(f"  [device] CUDA_VISIBLE_DEVICES={gpu} (single-GPU pin)")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train TCPerceiverIO for tc_perceiver_io experiment."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU index to pin to (sets CUDA_VISIBLE_DEVICES; overrides the "
             "config's top-level `gpu`). On a multi-GPU box this stops JAX from "
             "grabbing and preallocating memory on every device.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Short purpose/hypothesis slug for this run (e.g. "
             "'tcfrac0.3-cw-effnum'). Appended to the auto-incremented "
             "run_NN dir under trainer.run_group and used as the WandB run name.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    _pin_gpu(args.gpu, args.config)      # before any JAX device op
    train(args.config, resume=args.resume, name=args.name)
