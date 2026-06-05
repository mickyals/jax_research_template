"""
experiments/sparse_obs_cross_attn/train.py

Entry point for the sparse-obs cross-attention TC intensity classifier.

Usage
-----
    python src/experiments/sparse_obs_cross_attn/train.py
    python src/experiments/sparse_obs_cross_attn/train.py --config path/to/other.yaml
    python src/experiments/sparse_obs_cross_attn/train.py --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.sparse_obs_cross_attn.datamodule import JointDataModule
from experiments.sparse_obs_cross_attn.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.model import TCClassifier
from training.trainer import Trainer

_DEFAULT_CONFIG = Path(__file__).parent / 'configs' / 'tc_classifier.yaml'
_SCHEMA_PATH    = Path(__file__).parent / 'configs' / 'schema.json'


def validate_config(config: dict) -> None:
    """Validate config against schema.json using jsonschema.

    Soft dependency — if jsonschema is not installed, prints a warning
    and skips validation rather than crashing.
    """
    try:
        import json
        import jsonschema
        with open(_SCHEMA_PATH) as f:
            schema = json.load(f)
        jsonschema.validate(instance=config, schema=schema)
    except ImportError:
        print(
            "Warning: jsonschema not installed — config validation skipped. "
            "Install with: pip install jsonschema"
        )
    except jsonschema.ValidationError as e:
        raise ValueError(
            f"Config validation failed:\n  {e.message}\n"
            f"  Path: {' -> '.join(str(p) for p in e.absolute_path)}"
        ) from None


def train(config: dict, resume: bool = False) -> None:
    """Run training from a loaded config dict."""
    validate_config(config)

    print("Setting up data module…")
    dm = JointDataModule.from_config(config['data'])
    dm.summary()

    model        = TCClassifier(**config.get('model', {}))
    metrics_fns  = build_metrics_fns()
    trainer      = Trainer(model, metrics_fns, config['trainer'])

    trainer.fit(
        dm.train_loader(seed=config['trainer'].get('seed', 42)),
        dm.val_loader(),
        resume=resume,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train the sparse-obs cross-attention TC classifier.'
    )
    parser.add_argument(
        '--config', type=str, default=str(_DEFAULT_CONFIG),
        help='Path to YAML config (default: experiment configs/tc_classifier.yaml).',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from the latest checkpoint in checkpoint_dir.',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, resume=args.resume)


if __name__ == '__main__':
    main()
