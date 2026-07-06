"""
experiments/cyclone_jax/models/mlp.py

The MLP baseline ladder: all stations as input, no set logic.

    X -> FeatureEncoder -> (B, N, F') tokens
      -> one SHARED per-station perceptron (Dense over the last axis);
         station_features: null SKIPS it — RAW encoded tokens flatten
         directly ("can raw stations, positionally encoded, detect the
         storm?" probe; pair with encoding concat)
      -> padding slots zeroed via station_mask (they carry no signal, not
         even the perceptron bias)
      -> flat concat (B, pad_to * width)              [slot-sensitive]
      -> core MLP body -> (B, n_classes) logits

The flat vector is deliberately order/padding-sensitive — this is the
CLEAN memorisation-gate baseline, not a permutation-invariant model.
Remember that when reading generalisation runs.

activation is the config ladder key (relu | gelu | silu | leaky_relu),
applied to the perceptron and the body alike.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import flax.linen as nn

from core import get_activation
from core.nets.mlp import MLP

from experiments.cyclone_jax.models.features import FeatureEncoder, flatten

ACTIVATIONS = ('relu', 'gelu', 'silu', 'leaky_relu')


class StationMLP(nn.Module):
    """Shared per-station perceptron -> flat concat -> MLP -> logits.

    Parameters
    ----------
    n_classes : int
        Logit count — ALWAYS TargetSpec.n_classes (build_model injects it).
    station_features : int, optional
        Width of the shared per-station perceptron; None = no perceptron,
        the encoded tokens flatten raw (masked all the same).
    hidden_features, n_layers : int
        core MLP body size.
    activation : str
        One of ACTIVATIONS; perceptron + body.
    dropout_rate : float
        Body dropout (keep 0.0 for the memorisation gate).
    encoder : FeatureEncoder, optional
        Positional-encoding front end; default concat + raw coords.
    """
    n_classes:        int
    station_features: Optional[int]
    hidden_features:  int
    n_layers:         int
    activation:       str = 'relu'
    dropout_rate:     float = 0.0
    encoder:          Optional[FeatureEncoder] = None

    @nn.compact
    def __call__(self, X, train: bool = True) -> jnp.ndarray:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, "
                             f"got {self.activation!r}")
        encoder = (self.encoder if self.encoder is not None
                   else FeatureEncoder())
        tokens = encoder(X)                                    # (B, N, F')
        if self.station_features is None:      # raw-token probe: no
            h = tokens                         # per-station embedding
        else:
            h = nn.Dense(self.station_features,
                         name='station_perceptron')(tokens)    # shared
            h = get_activation(self.activation)(h)
        h = h * jnp.asarray(X['station_mask'], h.dtype)[..., None]
        h = flatten(h)                                         # (B, N*d)
        return MLP(out_features=self.n_classes,
                   hidden_features=self.hidden_features,
                   n_layers=self.n_layers,
                   activation=self.activation,
                   dropout_rate=self.dropout_rate,
                   name='body')(h, train=train)
