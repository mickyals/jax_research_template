"""
Tests for models/__init__.py — MODELS registry, build_model, and the
shipped-model-yaml drift guard (every configs/models/*.yaml must build
and run against a real TargetSpec).
"""

import jax
import pytest
import yaml

from experiments.cyclone_jax.config import CONFIG_DIR
from experiments.cyclone_jax.data.targets import TargetSpec
from experiments.cyclone_jax.models import MODELS, build_model, list_models

from .conftest import B

SHIPPED_MODELS = sorted(p.stem for p in (CONFIG_DIR / 'models').glob('*.yaml'))

SPEC = TargetSpec(variable='usa_sshs', kind='categorical',
                  class_set=(3, 4, 5, 6, 7, 8))


def _shipped(name):
    with open(CONFIG_DIR / 'models' / f'{name}.yaml') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Shipped configs stay buildable (drift guard)
# ---------------------------------------------------------------------------

class TestShippedModels:

    def test_expected_models_shipped(self):
        assert SHIPPED_MODELS == ['finer', 'mlp', 'siren']

    def test_every_shipped_yaml_is_registered(self):
        registered = {n.lower() for n in MODELS.names()}
        assert set(SHIPPED_MODELS) <= registered

    @pytest.mark.parametrize('name', SHIPPED_MODELS)
    def test_every_shipped_yaml_builds_and_runs(self, name, X):
        module, tags = build_model(_shipped(name), SPEC)
        assert tags                                    # wandb needs them
        variables = module.init(jax.random.PRNGKey(0), X, train=False)
        out = module.apply(variables, X, train=False)
        assert out.shape == (B, SPEC.n_classes)

    @pytest.mark.parametrize('name', SHIPPED_MODELS)
    def test_shipped_n_classes_is_null(self, name):
        assert _shipped(name)['n_classes'] is None


# ---------------------------------------------------------------------------
# build_model guards
# ---------------------------------------------------------------------------

class TestBuildModel:

    def test_n_classes_comes_from_target_spec(self, X):
        module, _ = build_model(_shipped('mlp'), SPEC)
        assert module.n_classes == SPEC.n_classes == 6

    def test_hand_set_n_classes_raises(self):
        cfg = _shipped('mlp')
        cfg['n_classes'] = 4
        with pytest.raises(ValueError, match='n_classes'):
            build_model(cfg, SPEC)

    def test_missing_name_raises(self):
        cfg = _shipped('mlp')
        del cfg['name']
        with pytest.raises(ValueError, match='name'):
            build_model(cfg, SPEC)

    def test_unknown_name_raises_with_available(self):
        with pytest.raises(ValueError, match='not registered'):
            build_model({'name': 'perceiver'}, SPEC)

    def test_missing_tags_default_to_empty(self):
        cfg = _shipped('siren')
        del cfg['tags']
        _, tags = build_model(cfg, SPEC)
        assert tags == ()

    def test_list_models_describes_all(self):
        assert set(list_models()) == {'MLP', 'SIREN', 'FINER'}
