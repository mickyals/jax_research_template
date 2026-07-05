"""
Tests for models/mlp.py — StationMLP shapes, activation ladder, masking.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.cyclone_jax.models.features import build_encoder
from experiments.cyclone_jax.models.mlp import ACTIVATIONS, StationMLP

from .conftest import B

N_CLASSES = 6


def _make(**kw):
    kw = {'n_classes': N_CLASSES, 'station_features': 8,
          'hidden_features': 16, 'n_layers': 2, **kw}
    return StationMLP(**kw)


def _logits(model, X, seed=0):
    variables = model.init(jax.random.PRNGKey(seed), X, train=False)
    return variables, model.apply(variables, X, train=False)


class TestStationMLP:

    def test_logit_shape(self, X):
        _, out = _logits(_make(), X)
        assert out.shape == (B, N_CLASSES)

    def test_param_structure(self, X):
        variables, _ = _logits(_make(), X)
        assert set(variables['params']) >= {'station_perceptron', 'body'}

    @pytest.mark.parametrize('act', ACTIVATIONS)
    def test_every_ladder_activation_runs(self, X, act):
        _, out = _logits(_make(activation=act), X)
        assert np.isfinite(np.asarray(out)).all()

    def test_activation_outside_ladder_raises(self, X):
        with pytest.raises(ValueError, match='activation'):
            _make(activation='tanh').init(jax.random.PRNGKey(0), X,
                                          train=False)

    def test_padded_slot_is_inert(self, X):
        """Values in masked station slots must not reach the logits."""
        model = _make()
        variables, base = _logits(model, X)
        X2 = {k: np.array(v) for k, v in X.items()}
        X2['obs'][0, -1] = 999.0            # slot (0, -1) is mask=False
        X2['lat'][0, -1] = 89.0
        out = model.apply(variables, X2, train=False)
        np.testing.assert_array_equal(np.asarray(base), np.asarray(out))

    def test_real_slot_is_not_inert(self, X):
        model = _make()
        variables, base = _logits(model, X)
        X2 = {k: np.array(v) for k, v in X.items()}
        X2['obs'][0, 0] = 999.0             # unmasked slot
        out = model.apply(variables, X2, train=False)
        assert not np.allclose(np.asarray(base[0]), np.asarray(out[0]))

    def test_gradients_flow(self, X):
        model = _make()
        variables, _ = _logits(model, X)

        def loss(params):
            out = model.apply({'params': params}, X, train=False)
            return jnp.sum(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        flat = jax.tree_util.tree_leaves(grads)
        assert any(float(jnp.abs(g).max()) > 0 for g in flat)

    def test_dropout_needs_rng_only_in_train(self, X):
        model = _make(dropout_rate=0.5)
        variables = model.init(jax.random.PRNGKey(0), X, train=False)
        out = model.apply(variables, X, train=True,
                          rngs={'dropout': jax.random.PRNGKey(1)})
        assert out.shape == (B, N_CLASSES)

    def test_custom_encoder_composes(self, X):
        enc = build_encoder({'mode': 'additive',
                             'embedding': 'GAUSSIAN_POSITIONAL',
                             'embedding_kwargs': {'input_dim': 2,
                                                  'mapping_dim': 8,
                                                  'scale': 1.0}})
        _, out = _logits(_make(encoder=enc), X)
        assert out.shape == (B, N_CLASSES)
