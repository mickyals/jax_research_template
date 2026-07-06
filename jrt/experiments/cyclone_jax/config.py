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
build_model / Trainer); this module only guards the config surface.
Model blocks are keyed per model name (MODEL_KEYS) — yaml-only here, no
model imports, so config loading stays jax-free.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / 'configs'

# 'gpu' pins CUDA_VISIBLE_DEVICES (train.py _pin_gpu, CLI --gpu overrides;
# a shell CUDA_VISIBLE_DEVICES always wins) — device policy, not config.
TOP_KEYS = {'data', 'model', 'trainer', 'gpu'}

DATA_KEYS = {'root', 'sources', 'channels', 'selection', 'max_stations',
             'pad_to', 'target', 'sshs_min', 'class_set',
             'drop_subtropical', 'source_id', 'timesteps', 'batch_size',
             'split', 'normalise', 'domain', 'tags', 'storm_panels',
             'num_workers', 'prefetch_factor'}

# storm_panels: per-split storm selection for the storm_panel callback /
# end-of-run sequence (a DATA property — which storm to watch belongs to
# the scenario). Values ('random' | sid | list of sids) are consumed in
# train/log.py.
STORM_PANEL_KEYS = {'train', 'val', 'test'}

SPLIT_KEYS = {'strategy', 'years', 'n_per_class', 'exclude_multistorm'}

NORMALISE_KEYS = {'method', 'stats'}

DOMAIN_KEYS = {'lat', 'lon'}

TRAINER_KEYS = {'seed', 'loss', 'loss_kwargs', 'optimizer',
                'optimizer_kwargs', 'scheduler', 'scheduler_kwargs',
                'num_epochs', 'check_val_every_n_epoch', 'gradient_clip',
                'patience', 'patience_metric', 'metrics', 'logger',
                'logger_kwargs', 'run_dir', 'callbacks'}

# trainer.callbacks list items (name from train/log.py CALLBACKS; 'every'
# in steps, default = est. steps per epoch; kwargs go to the factory).
CALLBACK_KEYS = {'name', 'every', 'split', 'kwargs'}

# trainer.loss may be a bare string or a weighted term list; list items
# are key-checked here (registry membership is checked where the names
# are consumed — train/losses.py build_loss — keeping config jax-free).
LOSS_TERM_KEYS = {'name', 'weight', 'kwargs'}

# Per-model key sets (mirror the build_model factories — keep in sync;
# names literal here so config loading never imports jax).
_MODEL_KEYS_COMMON = {'name', 'tags', 'n_classes', 'station_features',
                      'hidden_features', 'n_layers'}
MODEL_KEYS = {
    'mlp':   _MODEL_KEYS_COMMON | {'activation', 'dropout_rate', 'encoding'},
    'siren': _MODEL_KEYS_COMMON | {'first_omega', 'hidden_omega'},
    'finer': _MODEL_KEYS_COMMON | {'first_omega', 'hidden_omega', 'bias_k'},
}

ENCODING_KEYS = {'mode', 'embedding', 'embedding_kwargs', 'fields'}


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
    if data.get('normalise'):
        _check_keys(data['normalise'], NORMALISE_KEYS,
                    f"data scenario {raw['data']!r} normalise block")
    if data.get('domain'):
        _check_keys(data['domain'], DOMAIN_KEYS,
                    f"data scenario {raw['data']!r} domain block")
    if data.get('storm_panels'):
        _check_keys(data['storm_panels'], STORM_PANEL_KEYS,
                    f"data scenario {raw['data']!r} storm_panels block")

    model = None
    if raw.get('model'):
        model = _load_yaml(config_dir / 'models' / f"{raw['model']}.yaml")
        name = model.get('name')
        if name not in MODEL_KEYS:
            raise ValueError(f"model config {raw['model']!r} has unknown "
                             f"name {name!r} — known: {sorted(MODEL_KEYS)}")
        _check_keys(model, MODEL_KEYS[name],
                    f"model config {raw['model']!r}")
        if model.get('encoding'):
            _check_keys(model['encoding'], ENCODING_KEYS,
                        f"model config {raw['model']!r} encoding block")

    trainer = raw.get('trainer') or {}
    _check_keys(trainer, TRAINER_KEYS, 'trainer block')
    if isinstance(trainer.get('loss'), list):
        for term in trainer['loss']:
            if not isinstance(term, dict) or not term.get('name'):
                raise ValueError(
                    f"trainer.loss list items must be dicts with a 'name' "
                    f"(keys: {sorted(LOSS_TERM_KEYS)}), got {term!r}")
            _check_keys(term, LOSS_TERM_KEYS,
                        f"trainer.loss term {term['name']!r}")
    if trainer.get('callbacks'):
        for cb in trainer['callbacks']:
            if not isinstance(cb, dict) or not cb.get('name'):
                raise ValueError(
                    f"trainer.callbacks items must be dicts with a 'name' "
                    f"(keys: {sorted(CALLBACK_KEYS)}), got {cb!r}")
            _check_keys(cb, CALLBACK_KEYS,
                        f"trainer.callbacks entry {cb['name']!r}")

    # the pointer names survive for run naming ({model}-{data}-s{seed})
    return {'data': data, 'model': model, 'trainer': trainer,
            'names': {'data': raw['data'], 'model': raw.get('model')}}
