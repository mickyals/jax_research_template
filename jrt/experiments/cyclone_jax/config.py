"""
experiments/cyclone_jax/config.py

Config composition + validation. Train configs are the entry point and
POINT at one data scenario and one model config:

    configs/train/train.yaml:   data: overfit      # configs/data/<name>.yaml
                                model: mlp | null  # configs/models/<name>.yaml
                                trainer: {...}     # inline

    cfg = load_config('.../configs/train/train.yaml')
    data = build_data(cfg['data'], seed=cfg['trainer']['seed'])

load_config resolves the pointers and validates every block against its
known-key set — an unknown key is an ERROR (typo guard), a missing pointer
file fails with the resolved path. Value-level validation lives where
values are consumed (resolve_input / resolve_target / build_data /
Trainer); this module only guards the config surface. Model blocks are
not key-checked yet — each model defines its own keys when it lands.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / 'configs'

TOP_KEYS = {'data', 'model', 'trainer'}

DATA_KEYS = {'root', 'sources', 'selection', 'max_stations', 'pad_to',
             'target', 'sshs_min', 'class_set', 'drop_subtropical',
             'source_id', 'timesteps', 'batch_size', 'split'}

SPLIT_KEYS = {'strategy', 'years', 'n_per_class', 'exclude_multistorm'}

TRAINER_KEYS = {'seed', 'loss', 'loss_kwargs', 'optimizer',
                'optimizer_kwargs', 'scheduler', 'scheduler_kwargs',
                'num_epochs', 'check_val_every_n_epoch', 'gradient_clip',
                'patience', 'patience_metric', 'metrics', 'logger',
                'run_dir'}


def _load_yaml(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _check_keys(block, allowed, where):
    unknown = set(block) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in {where} — "
                         f"allowed: {sorted(allowed)}")


def load_config(train_yaml, config_dir=None):
    """Train yaml -> {'data': dict, 'model': dict | None, 'trainer': dict}
    with pointers resolved and key sets validated.

    config_dir overrides the pointer root (tests); default = configs/
    beside this module.
    """
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR

    raw = _load_yaml(train_yaml)
    _check_keys(raw, TOP_KEYS, str(train_yaml))
    if not raw.get('data'):
        raise ValueError(f"{train_yaml}: a 'data' scenario pointer is "
                         f"required (configs/data/<name>.yaml).")

    data = _load_yaml(config_dir / 'data' / f"{raw['data']}.yaml")
    _check_keys(data, DATA_KEYS, f"data scenario {raw['data']!r}")
    if data.get('split'):
        _check_keys(data['split'], SPLIT_KEYS,
                    f"data scenario {raw['data']!r} split block")

    model = None
    if raw.get('model'):
        model = _load_yaml(config_dir / 'models' / f"{raw['model']}.yaml")

    trainer = raw.get('trainer') or {}
    _check_keys(trainer, TRAINER_KEYS, 'trainer block')

    return {'data': data, 'model': model, 'trainer': trainer}
