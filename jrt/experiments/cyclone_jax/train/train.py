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

from pathlib import Path

from training.logger import create_logger
from training.metrics import get_metric
from training.trainer import Trainer

from experiments.cyclone_jax.config import load_config
from experiments.cyclone_jax.data.interface import build_data
from experiments.cyclone_jax.models import build_model
from experiments.cyclone_jax.train.losses import build_loss


def build_metrics_fns(trainer_cfg: dict) -> dict:
    """trainer yaml block -> Trainer metrics_fns dict.

    'loss' is the FIRST key (the Trainer differentiates the first entry),
    so the logged/patience name is train/loss regardless of which
    registered loss — or weighted term list (train/losses.py) — the
    config picked; trainer.metrics adds report-only metrics after it.
    """
    fns = {'loss': build_loss(trainer_cfg)}
    for name in trainer_cfg.get('metrics') or ():
        if name not in fns:
            fns[name] = get_metric(name)
    return fns


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
        'patience':         t.get('patience', 10),
        'patience_metric':  t.get('patience_metric', 'train/loss'),
        'max_grad_norm':    t.get('gradient_clip'),
        'seed':             t.get('seed', 0),
        'run_dir':          t.get('run_dir'),
    }


def build_logger(cfg: dict, tags: tuple):
    """trainer.logger/logger_kwargs + model tags -> a jrt logger.

    Model tags prepend any config-level tags (wandb only — the other
    backends have no tag concept). log_dir mirrors the Trainer's
    run_dir/logs convention.
    """
    t = cfg['trainer']
    backend = str(t.get('logger') or 'null')
    kwargs = dict(t.get('logger_kwargs') or {})
    run_dir = t.get('run_dir')
    log_dir = str(Path(run_dir).resolve() / 'logs') if run_dir else None
    if backend == 'wandb':
        kwargs.setdefault('project', 'cyclone_jax')
        kwargs['tags'] = list(tags) + list(kwargs.get('tags') or [])
    return create_logger(backend, log_dir=log_dir, config=cfg, **kwargs)


def main(train_yaml, config_dir=None):
    """Train per the config; returns (trainer, test_metrics)."""
    cfg = load_config(train_yaml, config_dir=config_dir)
    if not cfg['model']:
        raise ValueError(f"{train_yaml}: training needs a 'model' pointer "
                         f"(configs/models/<name>.yaml).")
    seed = cfg['trainer'].get('seed', 0)

    data = build_data(cfg['data'], seed=seed)
    model, tags = build_model(cfg['model'], data.targets)

    logger = build_logger(cfg, tags)
    trainer = Trainer(model, build_metrics_fns(cfg['trainer']),
                      build_trainer_config(cfg), logger=logger)

    missing = {'train', 'val'} - set(data.streams)
    if missing:
        raise ValueError(f"data scenario provides no {sorted(missing)} "
                         f"stream(s) — check split/batch_size.")

    trainer.fit(data.streams['train'], data.streams['val'])

    test_metrics = {}
    if 'test' in data.streams:
        test_metrics = trainer.test(data.streams['test'])

    trainer.logger.finalize('success')
    return trainer, test_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train a cyclone_jax model from a train config.')
    parser.add_argument('config', type=Path,
                        help='configs/train/<name>.yaml entry point')
    args = parser.parse_args()
    main(args.config)
