"""
experiments/cyclone_jax/models/siren.py

SIREN and FINER-on-SIREN heads, same flat-input skeleton as the MLP:

    X -> FeatureEncoder (concat + RAW coords — sine IS the encoding,
         no Fourier gamma)
      -> shared per-station sine layer (the paper FIRST layer:
         U(-1/fan_in, 1/fan_in) kernel; FINER adds the U(-k, k) bias)
      -> padding slots zeroed via station_mask
      -> flat concat -> core SIRENet/FINERNet body -> logits

The bodies are the core nets with ONE override: their first kernel init
uses the hidden scheme, because the paper first layer already happened in
the station perceptron (sine outputs land in [-1, 1], the hidden regime).
Everything else — activations, hidden inits, FINER bias — is inherited.
"""

from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn

from core import get_activation, get_initializer
from core.initializations import inr_first_init, inr_hidden_init
from core.nets.mlp import FINERNet, SIRENet

from experiments.cyclone_jax.models.features import FeatureEncoder, flatten


class _HiddenBodySIREN(SIRENet):
    """SIRENet used as a body AFTER a paper first layer — its own first
    layer therefore initialises with the hidden scheme."""
    def _make_first_kernel_init(self):
        return inr_hidden_init(self.first_omega)


class _HiddenBodyFINER(FINERNet):
    """FINERNet body, same first-layer override as _HiddenBodySIREN."""
    def _make_first_kernel_init(self):
        return inr_hidden_init(self.first_omega)


class _StationINR(nn.Module):
    """Shared SIREN/FINER skeleton; variants supply the perceptron
    activation/bias-init and the body class."""
    n_classes:        int
    station_features: int
    hidden_features:  int
    n_layers:         int
    first_omega:      float = 30.0
    hidden_omega:     float = 30.0

    def _perceptron_act(self):
        raise NotImplementedError

    def _perceptron_bias_init(self):
        return get_initializer('zeros')

    def _body(self):
        raise NotImplementedError

    @nn.compact
    def __call__(self, X, train: bool = True) -> jnp.ndarray:
        tokens = FeatureEncoder()(X)                     # raw coords, no PE
        h = nn.Dense(self.station_features,
                     kernel_init=inr_first_init,
                     bias_init=self._perceptron_bias_init(),
                     name='station_perceptron')(tokens)
        h = self._perceptron_act()(h)
        h = h * jnp.asarray(X['station_mask'], h.dtype)[..., None]
        h = flatten(h)
        return self._body()(h, train=train)


class StationSIREN(_StationINR):
    """SIREN head (Sitzmann et al. 2020) over the flat station vector."""

    def _perceptron_act(self):
        return get_activation('SINE', omega=self.first_omega)

    def _body(self):
        return _HiddenBodySIREN(out_features=self.n_classes,
                                hidden_features=self.hidden_features,
                                n_layers=self.n_layers,
                                first_omega=self.hidden_omega,
                                hidden_omega=self.hidden_omega,
                                name='body')


class StationFINER(_StationINR):
    """FINER-on-SIREN head (Liu et al. 2024): FINER activation everywhere,
    U(-bias_k, bias_k) bias in the perceptron and body hidden layers."""
    bias_k: float = 1.0

    def _perceptron_act(self):
        return get_activation('FINER', omega=self.first_omega)

    def _perceptron_bias_init(self):
        return get_initializer('FINER_BIAS', k=self.bias_k)

    def _body(self):
        return _HiddenBodyFINER(out_features=self.n_classes,
                                hidden_features=self.hidden_features,
                                n_layers=self.n_layers,
                                first_omega=self.hidden_omega,
                                hidden_omega=self.hidden_omega,
                                bias_k=self.bias_k,
                                name='body')
