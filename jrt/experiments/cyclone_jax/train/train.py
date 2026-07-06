"""
experiments/cyclone_jax/train/train.py

THE training entry point. One call, one config:

    PYTHONPATH=jrt python -m experiments.cyclone_jax.train.train \
        jrt/experiments/cyclone_jax/configs/train/train.yaml

Composition (all pieces independently tested):

    load_config -> build_data(cfg['data'], seed) -> build_model(cfg['model'],
    data.targets) -> jrt Trainer -> fit(train, val) [-> test if split exists]

trainer.seed is THE seed: it orders the data (build_data -> Sampler) and
initialises the model (Trainer -> create_rng_dict). DataBundle streams
are jrt-Trainer-compatible as-is (re-iterable, yield {'X','y','meta'};
the Trainer drops 'meta' before tracing).

The logger is built HERE, not inside the Trainer, so the model config's
wandb `tags` reach the run; trainer.logger_kwargs (project, offline, ...)
passes through. patience_metric train/loss is supported by the Trainer
(memorisation gate: watch the train loss go to ~0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.logger import create_logger
from training.trainer import Trainer

from experiments.cyclone_jax.config import load_config
from experiments.cyclone_jax.data.interface import build_data
from experiments.cyclone_jax.models import build_model
from experiments.cyclone_jax.train.log import build_callbacks, end_of_run
from experiments.cyclone_jax.train.metrics import build_metrics_fns


def build_trainer_config(cfg: dict) -> dict:
    """Translate the experiment trainer block to the jrt Trainer schema.

    Renames: gradient_clip -> max_grad_norm. batch_size comes from the
    DATA scenario (one source of truth — the streams are already batched
    with it). The logger is pre-built by build_logger, so no log_backend
    here.
    """
    t = cfg['trainer']
    return {
        'batch_size':       cfg['data']['batch_size'],
        'optimizer':        t.get('optimizer', 'adamw'),
        'optimizer_kwargs': t.get('optimizer_kwargs') or {},
        'scheduler':        t.get('scheduler', 'constant'),
        'scheduler_kwargs': t.get('scheduler_kwargs') or {},
        'num_epochs':       t.get('num_epochs', 100),
        'check_val_every_n_epoch': t.get('check_val_every_n_epoch', 1),
        'patience':         t.get('patience', 10),
        'patience_metric':  t.get('patience_metric', 'train/loss'),
        'max_grad_norm':    t.get('gradient_clip'),
        'seed':             t.get('seed', 0),
        'run_dir':          t.get('run_dir'),
    }


def build_logger(cfg: dict, tags: tuple, norms=None):
    """trainer.logger/logger_kwargs + model/data tags -> a jrt logger.

    wandb run tags = model tags + data tags + logger_kwargs tags (the
    other backends have no tag concept); run name defaults to
    {model}-{data}-s{seed} from the config pointer names. Norm stats join
    the logged config (they are properties of the training distribution).
    log_dir mirrors the Trainer's run_dir/logs convention.
    """
    t = cfg['trainer']
    backend = str(t.get('logger') or 'null')
    kwargs = dict(t.get('logger_kwargs') or {})
    run_dir = t.get('run_dir')
    log_dir = str(Path(run_dir).resolve() / 'logs') if run_dir else None
    if backend == 'wandb':
        kwargs.setdefault('project', 'cyclone_jax')
        data_tags = (cfg.get('data') or {}).get('tags') or []
        kwargs['tags'] = (list(tags) + list(data_tags)
                          + list(kwargs.get('tags') or []))
        names = cfg.get('names') or {}
        if names.get('model') and names.get('data'):
            kwargs.setdefault('name', f"{names['model']}-{names['data']}"
                                      f"-s{t.get('seed', 0)}")
    config = dict(cfg)
    if norms is not None:
        config['norm_stats'] = norms.to_json()
    return create_logger(backend, log_dir=log_dir, config=config, **kwargs)


def _split_summary(data) -> dict:
    """Per-split sizes + class counts (manifest AND startup banner)."""
    sshs = np.rint(np.asarray(data.loader.fixes['usa_sshs'])).astype(int)
    names = data.targets.class_names
    splits = {}
    for split, idx in data.splits.items():
        vals = sshs[np.asarray(idx)]
        splits[split] = {
            'size': int(len(idx)),
            'class_counts': {names[pos]: int((vals == c).sum())
                             for pos, c in enumerate(data.targets.class_set)},
        }
    return splits


def write_run_records(cfg: dict, data, run_dir) -> None:
    """run_dir/norm_stats.json + run_dir/data_manifest.json.

    norm_stats.json is the record evaluation REUSES (stats are properties
    of the training distribution — see data/usage_doc.md). The manifest
    records what the run actually trained on: per-split sizes and class
    counts, plus the merged config. Printing is the banner's job.
    """
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if data.norms is not None:
        (run / 'norm_stats.json').write_text(
            json.dumps(data.norms.to_json(), indent=2))
    (run / 'data_manifest.json').write_text(
        json.dumps({'splits': _split_summary(data), 'config': cfg},
                   indent=2))


def print_startup_banner(cfg, data, model, seed) -> None:
    """Pre-tqdm run summary — the experiment side; the jrt Trainer prints
    its own block (_print_startup_summary) after this.

    Lines: run name, scenario + sources, per-split size/class counts,
    normalisation method + coord bounds, model name + param count
    (jax.eval_shape — no FLOPs) + the nn.tabulate architecture table.
    """
    import jax
    import flax.linen as nn
    from experiments.cyclone_jax.data.batching import collate

    names = cfg.get('names') or {}
    print(f"  [run] {names.get('model')}-{names.get('data')}-s{seed}")
    srcs = '+'.join(cfg['data'].get('sources') or ('land', 'marine'))
    print(f"  [data] scenario {names.get('data')!r}  sources {srcs}")
    for name, info in _split_summary(data).items():
        print(f"  [data] {name}: {info['size']} fixes  "
              f"{info['class_counts']}")
    if data.norms is not None:
        s = data.norms.stats
        print(f"  [norm] {data.norms.method}  "
              f"lat [{s['lat']['min']:g}, {s['lat']['max']:g}]  "
              f"lon [{s['lon']['min']:g}, {s['lon']['max']:g}]")
    else:
        print("  [norm] raw (no normalise block)")

    idx = next((v for v in data.splits.values() if len(v)), None)
    if idx is None:      # all splits empty — main()'s stream check reports
        return
    X = collate([data.loader.build(int(idx[0]))], data.inputs.pad_to)['X']
    rng = jax.random.PRNGKey(seed)
    shapes = jax.eval_shape(lambda x: model.init(rng, x, train=False), X)
    n_params = sum(int(np.prod(p.shape))
                   for p in jax.tree_util.tree_leaves(shapes))
    print(f"  [model] {cfg['model'].get('name')}  {n_params:,} params")
    import warnings
    with warnings.catch_warnings():
        # flax summary.py calls jnp.shape on non-array module attributes
        # (activation callables) — its noise, not ours
        warnings.simplefilter('ignore', DeprecationWarning)
        print(nn.tabulate(model, rng)(X, train=False))


def main(train_yaml, config_dir=None):
    """Train per the config; returns (trainer, test_metrics)."""
    cfg = load_config(train_yaml, config_dir=config_dir)
    if not cfg['model']:
        raise ValueError(f"{train_yaml}: training needs a 'model' pointer "
                         f"(configs/models/<name>.yaml).")
    seed = cfg['trainer'].get('seed', 0)

    data = build_data(cfg['data'], seed=seed)
    model, tags = build_model(cfg['model'], data.targets, seed=seed)
    print_startup_banner(cfg, data, model, seed)

    if cfg['trainer'].get('run_dir'):
        write_run_records(cfg, data, cfg['trainer']['run_dir'])
    logger = build_logger(cfg, tags, norms=data.norms)
    trainer = Trainer(model, build_metrics_fns(cfg['trainer']),
                      build_trainer_config(cfg), logger=logger)

    missing = {'train', 'val'} - set(data.streams)
    if missing:
        raise ValueError(f"data scenario provides no {sorted(missing)} "
                         f"stream(s) — check split/batch_size.")

    callbacks = build_callbacks(cfg, data, logger)
    best_state = trainer.fit(data.streams['train'], data.streams['val'],
                             step_callbacks=callbacks or None)

    test_metrics = {}
    if 'test' in data.streams:
        test_metrics = trainer.test(data.streams['test'])
    end_of_run(cfg, data, logger, best_state,
               global_step=trainer.global_step)

    trainer.logger.finalize('success')
    return trainer, test_metrics


def _pin_gpu(cli_gpu, config_path) -> None:
    """Pin to one GPU via CUDA_VISIBLE_DEVICES (multi-GPU boxes: JAX
    otherwise claims and preallocates EVERY visible device).

    Resolution: shell CUDA_VISIBLE_DEVICES wins > --gpu > the train yaml's
    top-level ``gpu:`` field. Must run before the first JAX device op —
    JAX's GPU backend initialises lazily, so top of __main__ suffices.
    """
    import os
    import yaml
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        return
    gpu = cli_gpu
    if gpu is None:
        try:
            with open(config_path, encoding='utf-8') as f:
                gpu = (yaml.safe_load(f) or {}).get('gpu')
        except (OSError, yaml.YAMLError):
            gpu = None
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        print(f"  [device] CUDA_VISIBLE_DEVICES={gpu} (single-GPU pin)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train a cyclone_jax model from a train config.')
    parser.add_argument('config', type=Path,
                        help='configs/train/<name>.yaml entry point')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU index to pin (sets CUDA_VISIBLE_DEVICES; '
                             'overrides the yaml top-level `gpu`).')
    args = parser.parse_args()
    _pin_gpu(args.gpu, args.config)
    main(args.config)
