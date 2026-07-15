"""
Tests for train/tune.py — dotted-override mechanics, direction
derivation, and 2-trial end-to-end searches on the fixture library
(trials.csv record, per-trial run dirs, merged best.yaml, retrain-best).
"""

import numpy as np
import pytest
import yaml

from experiments.cyclone_jax.train.tune import (apply_overrides,
                                                study_direction, tune,
                                                write_best_yaml)
from tests.experiments.cyclone_jax.train.test_train import _write_configs


def _tune_setup(config_dir, library_root, search=None, trainer=None,
                **over):
    """Config tree from test_train's helper + a train/base.yaml pointer
    copy + the tune yaml itself."""
    entry = _write_configs(config_dir, library_root, trainer=trainer)
    (config_dir / 'train').mkdir()
    (config_dir / 'train' / 'base.yaml').write_text(entry.read_text())
    block = {'base': 'base', 'study': 'st', 'n_trials': 2,
             'n_startup_trials': 1, 'n_warmup_steps': 0,
             'retrain_best': False,
             'search': search or {
                 'trainer.scheduler_kwargs.value':
                     {'low': 1.0e-4, 'high': 1.0e-2, 'log': True}}}
    block.update(over)
    tune_yaml = config_dir / 'tune.yaml'
    tune_yaml.write_text(yaml.safe_dump(block))
    return tune_yaml


# ---------------------------------------------------------------------------
# Unit: overrides + direction
# ---------------------------------------------------------------------------

class TestOverrides:

    def test_sets_nested_value(self):
        cfg = {'trainer': {'scheduler_kwargs': {'value': 1e-3}}}
        apply_overrides(cfg, {'trainer.scheduler_kwargs.value': 5e-4})
        assert cfg['trainer']['scheduler_kwargs']['value'] == 5e-4

    def test_creates_missing_intermediate_block(self):
        cfg = {'trainer': {'optimizer': 'adamw'}}
        apply_overrides(cfg, {'trainer.optimizer_kwargs.weight_decay': 0.01})
        assert cfg['trainer']['optimizer_kwargs'] == {'weight_decay': 0.01}

    def test_multiple_paths_land_in_their_blocks(self):
        cfg = {'data': {'batch_size': 32}, 'model': {'n_layers': 3},
               'trainer': {}}
        apply_overrides(cfg, {'data.batch_size': 64, 'model.n_layers': 1})
        assert cfg['data']['batch_size'] == 64
        assert cfg['model']['n_layers'] == 1

    def test_direction_defaults_minimize(self):
        assert study_direction({}) == 'minimize'
        assert (study_direction({'patience_direction': 'lower_is_better'})
                == 'minimize')

    def test_direction_maximize(self):
        assert (study_direction({'patience_direction': 'higher_is_better'})
                == 'maximize')


class TestWriteBestYaml:

    class _BestTrial:
        params = {'trainer.scheduler_kwargs.value': 5e-4}

    def test_wandb_base_gains_study_best_tag(self, tmp_path):
        base = {'data': {'root': 'x'}, 'model': {'name': 'mlp'},
                'trainer': {'logger': 'wandb',
                            'scheduler_kwargs': {'value': 1e-3},
                            'logger_kwargs': {'tags': ['memorise']}},
                'names': {'data': 'd', 'model': 'm'}}
        cfg = write_best_yaml(tmp_path, base, 'st', self._BestTrial())
        assert cfg['trainer']['logger_kwargs']['tags'] == ['memorise',
                                                           'st-best']
        assert cfg['trainer']['scheduler_kwargs']['value'] == 5e-4
        assert cfg['trainer']['run_dir'] == str(tmp_path / 'best')
        # the record round-trips
        again = yaml.safe_load((tmp_path / 'best.yaml').read_text())
        assert again == cfg
        # the base was NOT mutated
        assert base['trainer']['scheduler_kwargs']['value'] == 1e-3


# ---------------------------------------------------------------------------
# End-to-end on the fixture library
# ---------------------------------------------------------------------------

class TestTuneEndToEnd:

    @pytest.fixture(autouse=True)
    def _no_basemap(self, monkeypatch):
        # retrain-best runs end_of_run figures: force plain axes so
        # cartopy never draws (Natural Earth downloads at render time)
        import experiments.cyclone_jax.visualise.figures as figs
        monkeypatch.setattr(figs, 'cartopy_available', lambda: False)

    def test_two_trials_record_and_best_yaml(self, library_root, tmp_path):
        ty = _tune_setup(tmp_path / 'c', library_root)
        tuner, best_cfg, result = tune(ty, config_dir=tmp_path / 'c')
        study_dir = tmp_path / 'c' / 'run' / 'st'

        assert len(tuner.study.trials) == 2
        assert result is None                    # retrain_best false

        # trials.csv = the study record, one row per trial
        header, *rows = (study_dir / 'trials.csv').read_text() \
            .strip().splitlines()
        assert header == 'trial,state,value,params'
        assert len(rows) == 2
        assert 'trainer.scheduler_kwargs.value' in rows[0]

        # per-trial isolated run dirs (jrt Tuner appends trial_N)
        assert (study_dir / 'trial_0').exists()
        assert (study_dir / 'trial_1').exists()

        # best.yaml = merged self-contained config, override applied
        best = yaml.safe_load((study_dir / 'best.yaml').read_text())
        assert best == best_cfg
        v = best['trainer']['scheduler_kwargs']['value']
        assert 1e-4 <= v <= 1e-2
        assert best['trainer']['run_dir'].endswith('best')
        # null backend: no wandb-only tags kwarg injected
        assert 'tags' not in (best['trainer'].get('logger_kwargs') or {})
        assert best['data']['root']              # data block INLINED

    def test_model_and_data_paths_searchable(self, library_root, tmp_path):
        search = {'model.hidden_features': {'choices': [4, 8]},
                  'data.batch_size': {'choices': [4]}}
        ty = _tune_setup(tmp_path / 'c', library_root, search=search)
        tuner, best_cfg, _ = tune(ty, config_dir=tmp_path / 'c')
        assert len(tuner.study.trials) == 2
        assert best_cfg['model']['hidden_features'] in (4, 8)
        assert best_cfg['data']['batch_size'] == 4

    def test_retrain_best_runs_and_records(self, library_root, tmp_path):
        ty = _tune_setup(tmp_path / 'c', library_root, n_trials=1,
                         retrain_best=True)
        tuner, best_cfg, result = tune(ty, config_dir=tmp_path / 'c')
        assert result is not None
        trainer, _ = result
        assert np.isfinite(trainer._best_metric_value)
        # the retrain trained under study_dir/best with full records
        best_dir = tmp_path / 'c' / 'run' / 'st' / 'best'
        assert (best_dir / 'predictions_train.csv').exists()
