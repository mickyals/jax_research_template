"""
Tests for train/train.py — config translation helpers and an end-to-end
fit on the synthetic library fixture (2 epochs, tiny model, null logger).
"""

import json

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

    def test_patience_direction_passes_through(self):
        """tune.py derives the study direction from this key — it must
        reach the jrt Trainer (which defaults lower_is_better)."""
        base = {'data': {'batch_size': 4}, 'trainer': {}}
        assert (build_trainer_config(base)['patience_direction']
                == 'lower_is_better')
        base['trainer']['patience_direction'] = 'higher_is_better'
        assert (build_trainer_config(base)['patience_direction']
                == 'higher_is_better')

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

    def test_wandb_gains_data_tags_and_run_name(self, monkeypatch):
        """Run tags = model + data + kwargs tags; run name from the config
        pointer names ({model}-{data}-s{seed})."""
        import experiments.cyclone_jax.train.train as tr
        seen = {}

        def fake_create_logger(backend, log_dir=None, config=None, **kw):
            seen.update(kw, config=config)
            return object()

        monkeypatch.setattr(tr, 'create_logger', fake_create_logger)
        cfg = {'data': {'tags': ['memorise', 'identifiability']},
               'names': {'model': 'mlp', 'data': 'memorise'},
               'trainer': {'logger': 'wandb', 'seed': 3,
                           'logger_kwargs': {'tags': ['extra']}}}
        tr.build_logger(cfg, tags=('mlp',))
        assert seen['tags'] == ['mlp', 'memorise', 'identifiability', 'extra']
        assert seen['name'] == 'mlp-memorise-s3'

    # _pin_gpu tests run against a patched os.environ dict — fully
    # isolated, nothing leaks into the real process environment.
    def _env(self, monkeypatch, initial):
        import os
        env = dict(initial)
        monkeypatch.setattr(os, 'environ', env)
        return env

    def test_pin_gpu_shell_env_wins(self, monkeypatch, tmp_path):
        from experiments.cyclone_jax.train.train import _pin_gpu
        env = self._env(monkeypatch, {'CUDA_VISIBLE_DEVICES': '7'})
        _pin_gpu('1', tmp_path / 'nope.yaml')
        assert env['CUDA_VISIBLE_DEVICES'] == '7'

    def test_pin_gpu_cli_over_yaml(self, monkeypatch, tmp_path):
        from experiments.cyclone_jax.train.train import _pin_gpu
        env = self._env(monkeypatch, {})
        entry = tmp_path / 'e.yaml'
        entry.write_text(yaml.safe_dump({'gpu': 3}))
        _pin_gpu('2', entry)
        assert env['CUDA_VISIBLE_DEVICES'] == '2'

    def test_pin_gpu_yaml_fallback(self, monkeypatch, tmp_path):
        from experiments.cyclone_jax.train.train import _pin_gpu
        env = self._env(monkeypatch, {})
        entry = tmp_path / 'e.yaml'
        entry.write_text(yaml.safe_dump({'gpu': 3}))
        _pin_gpu(None, entry)
        assert env['CUDA_VISIBLE_DEVICES'] == '3'

    def test_pin_gpu_noop_without_any_source(self, monkeypatch, tmp_path):
        from experiments.cyclone_jax.train.train import _pin_gpu
        env = self._env(monkeypatch, {})
        _pin_gpu(None, tmp_path / 'missing.yaml')
        assert 'CUDA_VISIBLE_DEVICES' not in env

    def test_norm_stats_join_logged_config(self, monkeypatch):
        import experiments.cyclone_jax.train.train as tr
        seen = {}

        def fake_create_logger(backend, log_dir=None, config=None, **kw):
            seen['config'] = config
            return object()

        class FakeNorms:
            def to_json(self):
                return {'method': 'standardise'}

        monkeypatch.setattr(tr, 'create_logger', fake_create_logger)
        tr.build_logger({'trainer': {'logger': 'null'}}, tags=(),
                        norms=FakeNorms())
        assert seen['config']['norm_stats'] == {'method': 'standardise'}


# ---------------------------------------------------------------------------
# End-to-end on the fixture library
# ---------------------------------------------------------------------------

class TestMain:

    @pytest.fixture(autouse=True)
    def _no_basemap(self, monkeypatch):
        # end_of_run renders the accuracy hexbin: force plain axes so
        # cartopy never draws (Natural Earth downloads at render time)
        import experiments.cyclone_jax.visualise.figures as figs
        monkeypatch.setattr(figs, 'cartopy_available', lambda: False)

    def test_fit_runs_and_checkpoints(self, library_root, tmp_path):
        entry = _write_configs(tmp_path / 'configs', library_root)
        trainer, test_metrics = main(entry, config_dir=tmp_path / 'configs')
        assert np.isfinite(trainer._best_metric_value)
        assert (trainer._checkpoint_dir / 'best').exists()
        assert test_metrics == {}               # stratified split: no test

    def test_main_accepts_merged_dict(self, library_root, tmp_path):
        """tune.py retrain-best passes the merged best.yaml config
        straight in — no pointer files involved."""
        from experiments.cyclone_jax.config import load_config
        entry = _write_configs(tmp_path / 'configs', library_root)
        cfg = load_config(entry, config_dir=tmp_path / 'configs')
        trainer, _ = main(cfg)
        assert np.isfinite(trainer._best_metric_value)

    def test_prediction_records_written(self, library_root, tmp_path):
        """end_of_run: per-fix predictions + per-storm accuracy CSVs for
        the distinct splits (stratified: val == train, swept once)."""
        entry = _write_configs(tmp_path / 'configs', library_root)
        main(entry, config_dir=tmp_path / 'configs')
        run = tmp_path / 'configs' / 'run'
        header, *rows = (run / 'predictions_train.csv').read_text() \
            .strip().splitlines()
        assert header == ('sid,name,time,lat,lon,n_stations,'
                          'n_stations_local,resolvable_km_local,'
                          'true,pred,correct')
        assert len(rows) > 0
        assert not (run / 'predictions_val.csv').exists()
        assert (run / 'per_storm_accuracy_train.csv').exists()

    def test_run_log_written_before_tables(self, library_root, tmp_path):
        """Logging begins first: every banner line lands in
        run_dir/logs/run.log, not just the terminal."""
        entry = _write_configs(tmp_path / 'configs', library_root)
        main(entry, config_dir=tmp_path / 'configs')
        text = (tmp_path / 'configs' / 'run' / 'logs' / 'run.log') \
            .read_text(encoding='utf-8')
        assert '[run] tiny-tiny-s0' in text
        assert '[data] train:' in text
        assert '[model] X per sample' in text
        assert 'params' in text

    def test_fit_runs_with_term_list_loss(self, library_root, tmp_path):
        """End-to-end: model term (l1_params) through the real train path."""
        entry = _write_configs(
            tmp_path / 'configs', library_root,
            trainer={'loss': [{'name': 'cross_entropy'},
                              {'name': 'l1_params', 'weight': 1.0e-4}]})
        trainer, _ = main(entry, config_dir=tmp_path / 'configs')
        assert np.isfinite(trainer._best_metric_value)

    def test_run_records_written(self, library_root, tmp_path):
        """norm_stats.json (evaluate reuses) + data_manifest.json (what the
        run trained on) land in run_dir before training starts."""
        entry = _write_configs(
            tmp_path / 'configs', library_root,
            data_over={'normalise': {'method': 'standardise',
                                     'stats': 'auto'}})
        main(entry, config_dir=tmp_path / 'configs')
        run = tmp_path / 'configs' / 'run'
        stats = json.loads((run / 'norm_stats.json').read_text())
        assert stats['method'] == 'standardise'
        man = json.loads((run / 'data_manifest.json').read_text())
        tr = man['splits']['train']
        assert tr['size'] > 0
        assert sum(tr['class_counts'].values()) == tr['size']
        assert man['config']['names'] == {'data': 'tiny', 'model': 'tiny'}

    def test_startup_banner_prints_data_norm_model(self, library_root,
                                                   tmp_path, capsys):
        """Pre-tqdm banner: split sizes/class counts, norm line, model
        name + param count + nn.tabulate architecture table."""
        entry = _write_configs(
            tmp_path / 'configs', library_root,
            data_over={'normalise': {'method': 'standardise',
                                     'stats': 'auto'}})
        main(entry, config_dir=tmp_path / 'configs')
        out = capsys.readouterr().out
        assert '[data] train:' in out
        assert '[data] channels (' in out      # the resolved union, named
        assert '[norm] standardise' in out
        assert '[model] X per sample' in out    # what enters the model
        assert '[model] mlp' in out and 'params' in out
        assert 'flatten_mlp' in out.lower() or 'Dense' in out  # tabulate

    def test_missing_model_pointer_raises(self, library_root, tmp_path):
        entry = _write_configs(tmp_path / 'configs', library_root)
        raw = yaml.safe_load(entry.read_text())
        raw['model'] = None
        entry.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match='model'):
            main(entry, config_dir=tmp_path / 'configs')
