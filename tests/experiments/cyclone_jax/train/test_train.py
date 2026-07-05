"""
Tests for train/train.py — config translation helpers and an end-to-end
fit on the synthetic library fixture (2 epochs, tiny model, null logger).
"""

import numpy as np
import pytest
import yaml

from experiments.cyclone_jax.train.train import (
    build_logger, build_trainer_config, main,
)

TINY_MODEL = {
    'name': 'mlp', 'tags': ['mlp', 'test'], 'n_classes': None,
    'station_features': 4, 'hidden_features': 8, 'n_layers': 1,
    'activation': 'relu',
}


def _write_configs(config_dir, library_root, model=TINY_MODEL,
                   trainer=None, data_over=None):
    """A minimal self-contained config tree over the fixture library."""
    (config_dir / 'data').mkdir(parents=True)
    (config_dir / 'models').mkdir()
    data = {'root': str(library_root), 'sshs_min': 3, 'batch_size': 4,
            'pad_to': 64, 'max_stations': 32, 'selection': 'max_stations',
            'split': {'strategy': 'stratified', 'n_per_class': 2},
            **(data_over or {})}
    (config_dir / 'data' / 'tiny.yaml').write_text(yaml.safe_dump(data))
    (config_dir / 'models' / 'tiny.yaml').write_text(yaml.safe_dump(model))
    trainer = {'seed': 0, 'loss': 'cross_entropy', 'optimizer': 'adam',
               'scheduler': 'constant', 'scheduler_kwargs': {'value': 1e-3},
               'num_epochs': 2, 'patience': 5,
               'patience_metric': 'train/loss',
               'metrics': ['accuracy'], 'logger': 'null',
               'run_dir': str(config_dir / 'run'),
               **(trainer or {})}
    entry = config_dir / 'entry.yaml'
    entry.write_text(yaml.safe_dump({'data': 'tiny', 'model': 'tiny',
                                     'trainer': trainer}))
    return entry


# ---------------------------------------------------------------------------
# Config translation helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_trainer_schema_translation(self):
        cfg = {'data': {'batch_size': 16},
               'trainer': {'gradient_clip': 1.0, 'num_epochs': 3,
                           'seed': 7, 'patience_metric': 'train/loss'}}
        out = build_trainer_config(cfg)
        assert out['batch_size'] == 16          # from the DATA scenario
        assert out['max_grad_norm'] == 1.0      # renamed
        assert out['seed'] == 7
        assert out['patience_metric'] == 'train/loss'

    def test_null_logger_built_under_run_dir(self, tmp_path):
        cfg = {'trainer': {'logger': 'null', 'run_dir': str(tmp_path)}}
        logger = build_logger(cfg, tags=('mlp',))
        assert str(logger.log_dir).startswith(str(tmp_path))

    def test_wandb_kwargs_gain_model_tags(self, monkeypatch, tmp_path):
        """Model tags must reach wandb.init — capture the create call."""
        import experiments.cyclone_jax.train.train as tr
        seen = {}

        def fake_create_logger(backend, log_dir=None, config=None, **kw):
            seen.update(kw, backend=backend)
            return object()

        monkeypatch.setattr(tr, 'create_logger', fake_create_logger)
        cfg = {'trainer': {'logger': 'wandb', 'run_dir': str(tmp_path),
                           'logger_kwargs': {'project': 'p',
                                             'tags': ['extra']}}}
        tr.build_logger(cfg, tags=('mlp', 'baseline'))
        assert seen['backend'] == 'wandb'
        assert seen['tags'] == ['mlp', 'baseline', 'extra']
        assert seen['project'] == 'p'


# ---------------------------------------------------------------------------
# End-to-end on the fixture library
# ---------------------------------------------------------------------------

class TestMain:

    def test_fit_runs_and_checkpoints(self, library_root, tmp_path):
        entry = _write_configs(tmp_path / 'configs', library_root)
        trainer, test_metrics = main(entry, config_dir=tmp_path / 'configs')
        assert np.isfinite(trainer._best_metric_value)
        assert (trainer._checkpoint_dir / 'best').exists()
        assert test_metrics == {}               # stratified split: no test

    def test_fit_runs_with_term_list_loss(self, library_root, tmp_path):
        """End-to-end: model term (l1_params) through the real train path."""
        entry = _write_configs(
            tmp_path / 'configs', library_root,
            trainer={'loss': [{'name': 'cross_entropy'},
                              {'name': 'l1_params', 'weight': 1.0e-4}]})
        trainer, _ = main(entry, config_dir=tmp_path / 'configs')
        assert np.isfinite(trainer._best_metric_value)

    def test_missing_model_pointer_raises(self, library_root, tmp_path):
        entry = _write_configs(tmp_path / 'configs', library_root)
        raw = yaml.safe_load(entry.read_text())
        raw['model'] = None
        entry.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match='model'):
            main(entry, config_dir=tmp_path / 'configs')
