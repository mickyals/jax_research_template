"""
experiments/cyclone_jax/models/__init__.py

MODELS registry + build_model — THE way train/notebooks construct a model:

    module, tags = build_model(cfg['model'], data.targets)

Each configs/models/<name>.yaml is one registered entry; `tags` flow to
the wandb run. n_classes comes from the TargetSpec at build time, NEVER
the yaml (keep `n_classes: null` — a hand-set value is an error, so the
label space cannot silently diverge from the data config's class_set).
"""

from __future__ import annotations

from utils.registry import Registry

from experiments.cyclone_jax.models.features import build_encoder
from experiments.cyclone_jax.models.mlp import StationMLP
from experiments.cyclone_jax.models.siren import StationFINER, StationSIREN

MODELS = Registry('Model')
register_model = MODELS.register
get_model = MODELS.get


def list_models() -> dict[str, str]:
    """Sorted ``{name: description}`` of all registered entries."""
    return dict(sorted(MODELS.describe().items()))


@register_model('mlp', description='StationMLP baseline (activation ladder '
                                   'relu|gelu|silu|leaky_relu)')
def _mlp(n_classes, station_features, hidden_features, n_layers,
         activation='relu', dropout_rate=0.0, encoding=None):
    return StationMLP(n_classes=n_classes,
                      station_features=station_features,
                      hidden_features=hidden_features,
                      n_layers=n_layers,
                      activation=activation,
                      dropout_rate=dropout_rate,
                      encoder=build_encoder(encoding))


@register_model('siren', description='StationSIREN (Sitzmann et al. 2020; '
                                     'raw coords, no PE)')
def _siren(n_classes, station_features, hidden_features, n_layers,
           first_omega=30.0, hidden_omega=30.0):
    return StationSIREN(n_classes=n_classes,
                        station_features=station_features,
                        hidden_features=hidden_features,
                        n_layers=n_layers,
                        first_omega=first_omega,
                        hidden_omega=hidden_omega)


@register_model('finer', description='StationFINER (Liu et al. 2024; '
                                     'FINER-on-SIREN, U(-k,k) bias)')
def _finer(n_classes, station_features, hidden_features, n_layers,
           first_omega=30.0, hidden_omega=30.0, bias_k=1.0):
    return StationFINER(n_classes=n_classes,
                        station_features=station_features,
                        hidden_features=hidden_features,
                        n_layers=n_layers,
                        first_omega=first_omega,
                        hidden_omega=hidden_omega,
                        bias_k=bias_k)


def build_model(model_cfg: dict, targets):
    """cfg['model'] block + TargetSpec -> (flax module, wandb tags).

    Pops the non-arch keys (name/tags/n_classes), injects
    targets.n_classes, and instantiates through the registry.
    """
    cfg = dict(model_cfg)
    name = cfg.pop('name', None)
    if not name:
        raise ValueError("model config needs a 'name' (registry key) — "
                         f"registered: {MODELS.names()}")
    tags = tuple(cfg.pop('tags', None) or ())
    if cfg.pop('n_classes', None) is not None:
        raise ValueError("n_classes is injected from the TargetSpec at "
                         "build — keep 'n_classes: null' in the model yaml.")
    module = MODELS.get(name, n_classes=targets.n_classes, **cfg)
    return module, tags
