"""
experiments/cyclone_jax/train/tune.py

Hyperparameter search entry point — thin over jrt training/tuner.Tuner
(Optuna + MedianPruner + per-trial run_dir isolation):

    PYTHONPATH=jrt python -m experiments.cyclone_jax.train.tune \
        jrt/experiments/cyclone_jax/configs/train/tune_memorise_mlp.yaml

The tune yaml (surface validated by config.load_tune_config) points at a
BASE train config and maps DOTTED config paths to search specs — the
overrides land anywhere in the merged {data, model, trainer} config, so
architecture, data and optimisation HPs tune through one mechanism:

    base: memorise_mlp
    search:
      trainer.scheduler_kwargs.value: {low: 1.0e-4, high: 1.0e-2, log: true}
      model.hidden_features: {choices: [64, 256]}

Rulings on record (2026-07-05):
  - NO sqlite. The study record = trials.csv in the study run_dir
    (appended after every trial) + one wandb run PER TRIAL (group =
    study name, run {study}-t{N}, tag trial_N) when the base config
    logs to wandb. In-memory optuna per invocation — no cross-session
    resume, the csv is the record.
  - The study DIRECTION derives from the base trainer's
    patience_direction (Trainer.is_better already encodes better/worse
    — one source of truth, no separate key to contradict it).
  - retrain_best: best.yaml = the MERGED resolved config with the
    winning overrides applied (data/model/trainer blocks INLINED — a
    self-contained record, not a pointer file), retrained via the
    factored train.main(cfg dict), wandb-tagged {study}-best. best.yaml
    is written whether or not the retrain runs.

The library is loaded ONCE and cached across trials (build_data lib=);
DataBundles are cached by the trial's resolved data block, so only
data-path overrides pay a rebuild.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import yaml

from training.logger import create_logger
from training.tuner import Tuner

from experiments.cyclone_jax.config import (CONFIG_DIR, load_config,
                                            load_tune_config)
from experiments.cyclone_jax.data.interface import build_data
from experiments.cyclone_jax.models import build_model
from experiments.cyclone_jax.train.metrics import build_metrics_fns
from experiments.cyclone_jax.train.train import (build_trainer_config,
                                                 main as train_main)


def apply_overrides(cfg: dict, params: dict) -> dict:
    """Set {'trainer.scheduler_kwargs.value': 3e-4, ...} into the merged
    config (in place; returns it). Missing intermediate blocks are
    created — a search may introduce a kwargs dict the base omits.
    Path roots are validated at load_tune_config time."""
    for path, value in params.items():
        keys = str(path).split('.')
        node = cfg
        for k in keys[:-1]:
            nxt = node.get(k)
            if nxt is None:
                nxt = node[k] = {}
            node = nxt
        node[keys[-1]] = value
    return cfg


def study_direction(trainer_cfg: dict) -> str:
    """Optuna direction from the base trainer's patience_direction —
    the objective IS the patience metric, so better/worse must agree."""
    d = trainer_cfg.get('patience_direction', 'lower_is_better')
    return 'minimize' if d == 'lower_is_better' else 'maximize'


def _suggest(trial, path, spec):
    """One search spec -> sampled value (the dotted path is the optuna
    param name, so trial.params maps straight back into the config)."""
    if 'choices' in spec:
        return trial.suggest_categorical(path, list(spec['choices']))
    low, high = spec['low'], spec['high']
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(path, low, high,
                                 step=spec.get('step', 1),
                                 log=spec.get('log', False))
    return trial.suggest_float(path, float(low), float(high),
                               step=spec.get('step'),
                               log=spec.get('log', False))


def write_best_yaml(study_dir: Path, base: dict, study: str,
                    best_trial) -> dict:
    """The winning MERGED config -> study_dir/best.yaml; returns the dict
    (what retrain runs). run_dir moves to study_dir/best; the wandb tag
    {study}-best marks the retrain's provenance."""
    cfg = apply_overrides(copy.deepcopy(base), dict(best_trial.params))
    cfg['trainer']['run_dir'] = str(study_dir / 'best')
    if str(cfg['trainer'].get('logger')) == 'wandb':
        # tags are a wandb concept — the other backends reject the kwarg
        kwargs = cfg['trainer'].setdefault('logger_kwargs', {}) or {}
        kwargs['tags'] = list(kwargs.get('tags') or []) + [f'{study}-best']
        cfg['trainer']['logger_kwargs'] = kwargs
    (study_dir / 'best.yaml').write_text(yaml.safe_dump(cfg,
                                                        sort_keys=False))
    return cfg


def tune(tune_yaml, config_dir=None):
    """Run the search per the tune yaml.

    Returns (tuner, best_cfg, retrain_result): the jrt Tuner (study,
    best_params, summary()), the merged best config (= best.yaml), and
    train.main's (trainer, test_metrics) when retrain_best ran else None.
    """
    tcfg = load_tune_config(tune_yaml)
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    base = load_config(cdir / 'train' / f"{tcfg['base']}.yaml",
                       config_dir=cdir)
    study = str(tcfg.get('study') or f"{tcfg['base']}_tune")
    seed = base['trainer'].get('seed', 0)
    study_dir = Path(base['trainer'].get('run_dir') or 'runs') / study
    study_dir.mkdir(parents=True, exist_ok=True)
    search = tcfg['search']

    # caches shared across trials: the library loads once; DataBundles
    # key on the trial's resolved data block (only data-path overrides
    # pay a rebuild). ``cell`` carries the CURRENT trial's cfg + data
    # from suggest_fn to the model/metrics/loader/logger callables.
    lib_cache, data_cache, cell = {}, {}, {}

    def _data_for(cfg):
        key = json.dumps(cfg['data'], sort_keys=True, default=str)
        if key not in data_cache:
            sources = tuple(cfg['data'].get('sources') or ('land', 'marine'))
            bundle = build_data(cfg['data'], seed=seed,
                                lib=lib_cache.get(sources))
            lib_cache[sources] = bundle.lib
            data_cache[key] = bundle
        return data_cache[key]

    def suggest_fn(trial, base_cfg):
        params = {p: _suggest(trial, p, s) for p, s in search.items()}
        cfg = apply_overrides(base_cfg, params)      # base_cfg = deepcopy
        cell['cfg'] = cfg
        cell['data'] = _data_for(cfg)
        jrt_cfg = build_trainer_config(cfg)
        jrt_cfg['run_dir'] = str(study_dir)          # Tuner adds trial_N
        return {'trainer': jrt_cfg, 'experiment': cfg}

    def model_fn(config):
        model, _ = build_model(config['experiment']['model'],
                               cell['data'].targets, seed=seed)
        return model

    def metrics_fn(config):
        return build_metrics_fns(config['experiment']['trainer'])

    def logger_fn(trial):
        cfg = cell['cfg']
        backend = str(cfg['trainer'].get('logger') or 'null')
        kwargs = dict(cfg['trainer'].get('logger_kwargs') or {})
        if backend == 'wandb':
            kwargs.setdefault('project', 'cyclone_jax')
            kwargs['group'] = study
            kwargs['name'] = f'{study}-t{trial.number}'
            kwargs['tags'] = (list(kwargs.get('tags') or [])
                              + [study, f'trial_{trial.number}'])
        if backend == 'null':
            kwargs.setdefault('verbose', False)
        logged = dict(cfg)
        logged['trial_params'] = dict(trial.params)
        return create_logger(backend,
                             log_dir=str(study_dir / f'trial_{trial.number}'
                                         / 'logs'),
                             config=logged, **kwargs)

    def append_trial(study_obj, trial):
        """After every trial: one trials.csv row — THE study record
        (no sqlite by ruling; survives crashes trial-by-trial)."""
        path = study_dir / 'trials.csv'
        fresh = not path.exists()
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if fresh:
                w.writerow(['trial', 'state', 'value', 'params'])
            w.writerow([trial.number, trial.state.name,
                        '' if trial.value is None else f'{trial.value:.6g}',
                        json.dumps(trial.params)])

    tuner = Tuner(
        suggest_fn       = suggest_fn,
        base_config      = base,          # Tuner deep-copies per trial
        model_fn         = model_fn,
        metrics_fns      = metrics_fn,    # rebuilt per trial (loss HPs)
        train_loader_fn  = lambda: cell['data'].streams['train'],
        val_loader_fn    = lambda: cell['data'].streams['val'],
        study_name       = study,
        direction        = study_direction(base['trainer']),
        n_startup_trials = int(tcfg.get('n_startup_trials', 4)),
        n_warmup_steps   = int(tcfg.get('n_warmup_steps', 3)),
        logger_fn        = logger_fn,
    )
    study_obj = tuner.run(int(tcfg.get('n_trials', 25)),
                          callbacks=[append_trial])
    tuner.summary()

    best_cfg = write_best_yaml(study_dir, base, study,
                               study_obj.best_trial)
    result = train_main(best_cfg) if tcfg.get('retrain_best') else None
    return tuner, best_cfg, result


if __name__ == '__main__':
    import argparse
    from experiments.cyclone_jax.train.train import _pin_gpu
    parser = argparse.ArgumentParser(
        description='Optuna HP search over a cyclone_jax train config.')
    parser.add_argument('config', type=Path,
                        help='configs/train/tune_*.yaml entry point')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU index to pin (sets CUDA_VISIBLE_DEVICES; '
                             'overrides the yaml top-level `gpu`).')
    args = parser.parse_args()
    _pin_gpu(args.gpu, args.config)
    tune(args.config)
