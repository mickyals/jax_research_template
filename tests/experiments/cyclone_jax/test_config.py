"""
Tests for cyclone_jax config composition — pointer resolution, key-set
validation (unknown key = error), and a round-trip guard that every
SHIPPED config stays consumable by the specs it feeds.
"""

import pytest
import yaml

from experiments.cyclone_jax.config import (
    CONFIG_DIR, DATA_KEYS, MODEL_KEYS, load_config,
)
from experiments.cyclone_jax.data.inputs import resolve_input
from experiments.cyclone_jax.data.targets import resolve_target

SHIPPED_SCENARIOS = sorted(p.stem for p in (CONFIG_DIR / 'data').glob('*.yaml'))
SHIPPED_MODELS = sorted(p.stem for p in (CONFIG_DIR / 'models').glob('*.yaml'))


# ---------------------------------------------------------------------------
# Shipped configs stay valid (drift guard)
# ---------------------------------------------------------------------------

class TestShippedConfigs:

    def test_train_entry_point_loads(self):
        cfg = load_config(CONFIG_DIR / 'train' / 'train.yaml')
        assert set(cfg) == {'data', 'model', 'trainer'}
        assert cfg['model']['name'] == 'mlp'         # the gate baseline
        assert cfg['trainer']['seed'] == 0
        assert cfg['data']['split']['strategy'] == 'stratified'

    @pytest.mark.parametrize('scenario', SHIPPED_SCENARIOS)
    def test_every_scenario_resolves_through_the_specs(self, scenario,
                                                       tmp_path):
        entry = tmp_path / 'entry.yaml'
        entry.write_text(yaml.safe_dump({'data': scenario}))
        cfg = load_config(entry)
        spec_in = resolve_input(cfg['data'])         # raises on bad values
        spec_t = resolve_target(cfg['data'])
        assert spec_in.pad_to == 1536
        assert spec_t.n_classes == 6

    def test_expected_scenarios_shipped(self):
        assert SHIPPED_SCENARIOS == ['multistorm', 'overfit', 'test',
                                     'train']

    @pytest.mark.parametrize('name', SHIPPED_MODELS)
    def test_every_shipped_model_config_validates(self, name, tmp_path):
        """Key-set validation passes for every shipped model yaml (the
        build/run round-trip lives in models/test_registry.py)."""
        entry = tmp_path / 'entry.yaml'
        entry.write_text(yaml.safe_dump({'data': 'overfit', 'model': name}))
        cfg = load_config(entry, config_dir=CONFIG_DIR)
        assert cfg['model']['name'] == name

    def test_shipped_model_names_all_key_checked(self):
        assert set(SHIPPED_MODELS) == set(MODEL_KEYS)


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def _write(config_dir, scenario_name, block, entry=None):
    (config_dir / 'data').mkdir(parents=True, exist_ok=True)
    (config_dir / 'data' / f'{scenario_name}.yaml').write_text(
        yaml.safe_dump(block))
    entry_file = config_dir / 'entry.yaml'
    entry_file.write_text(yaml.safe_dump(entry or {'data': scenario_name}))
    return entry_file


class TestGuards:

    def test_unknown_top_key_raises(self, tmp_path):
        entry = _write(tmp_path, 's', {'root': 'x'},
                       entry={'data': 's', 'trianer': {}})
        with pytest.raises(ValueError, match='trianer'):
            load_config(entry, config_dir=tmp_path)

    def test_missing_data_pointer_raises(self, tmp_path):
        entry = tmp_path / 'entry.yaml'
        entry.write_text(yaml.safe_dump({'trainer': {'seed': 0}}))
        with pytest.raises(ValueError, match="'data' scenario pointer"):
            load_config(entry, config_dir=tmp_path)

    def test_missing_scenario_file_raises_with_path(self, tmp_path):
        entry = tmp_path / 'entry.yaml'
        entry.write_text(yaml.safe_dump({'data': 'nope'}))
        with pytest.raises(FileNotFoundError, match='nope.yaml'):
            load_config(entry, config_dir=tmp_path)

    def test_unknown_data_key_raises(self, tmp_path):
        entry = _write(tmp_path, 's', {'root': 'x', 'pad_too': 8})
        with pytest.raises(ValueError, match='pad_too'):
            load_config(entry, config_dir=tmp_path)

    def test_unknown_split_key_raises(self, tmp_path):
        entry = _write(tmp_path, 's',
                       {'root': 'x', 'split': {'strategy': 'year',
                                               'yeras': {}}})
        with pytest.raises(ValueError, match='yeras'):
            load_config(entry, config_dir=tmp_path)

    def test_unknown_trainer_key_raises(self, tmp_path):
        entry = _write(tmp_path, 's', {'root': 'x'},
                       entry={'data': 's',
                              'trainer': {'seed': 0, 'lr': 1e-3}})
        with pytest.raises(ValueError, match='lr'):
            load_config(entry, config_dir=tmp_path)

    def test_unknown_model_key_raises(self, tmp_path):
        entry = _write(tmp_path, 's', {'root': 'x'},
                       entry={'data': 's', 'model': 'm'})
        (tmp_path / 'models').mkdir()
        (tmp_path / 'models' / 'm.yaml').write_text(yaml.safe_dump(
            {'name': 'mlp', 'hidden_featuers': 8}))
        with pytest.raises(ValueError, match='hidden_featuers'):
            load_config(entry, config_dir=tmp_path)

    def test_unknown_model_name_raises(self, tmp_path):
        entry = _write(tmp_path, 's', {'root': 'x'},
                       entry={'data': 's', 'model': 'm'})
        (tmp_path / 'models').mkdir()
        (tmp_path / 'models' / 'm.yaml').write_text(yaml.safe_dump(
            {'name': 'perceiver'}))
        with pytest.raises(ValueError, match='perceiver'):
            load_config(entry, config_dir=tmp_path)

    def test_unknown_encoding_key_raises(self, tmp_path):
        entry = _write(tmp_path, 's', {'root': 'x'},
                       entry={'data': 's', 'model': 'm'})
        (tmp_path / 'models').mkdir()
        (tmp_path / 'models' / 'm.yaml').write_text(yaml.safe_dump(
            {'name': 'mlp', 'encoding': {'mdoe': 'concat'}}))
        with pytest.raises(ValueError, match='mdoe'):
            load_config(entry, config_dir=tmp_path)

    def test_siren_rejects_mlp_only_keys(self, tmp_path):
        entry = _write(tmp_path, 's', {'root': 'x'},
                       entry={'data': 's', 'model': 'm'})
        (tmp_path / 'models').mkdir()
        (tmp_path / 'models' / 'm.yaml').write_text(yaml.safe_dump(
            {'name': 'siren', 'encoding': {'mode': 'concat'}}))
        with pytest.raises(ValueError, match='encoding'):
            load_config(entry, config_dir=tmp_path)

    def test_data_keys_cover_interface_reads(self):
        """Keys build_data/interface reads must be declared known."""
        assert {'root', 'sshs_min', 'drop_subtropical', 'split',
                'batch_size'} <= DATA_KEYS
