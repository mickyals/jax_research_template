"""
Tests for models/siren.py — SIREN/FINER shapes, paper inits, masking.
"""

import jax
import numpy as np
import pytest

from core.initializations import inr_first_bound, inr_hidden_bound
from experiments.cyclone_jax.models.siren import StationFINER, StationSIREN

from .conftest import B, C

N_CLASSES = 6
D = 8                                   # station_features
F_IN = 3 + 2 * C + 2                    # tokens + raw coords


def _make(cls, **kw):
    kw = {'n_classes': N_CLASSES, 'station_features': D,
          'hidden_features': 16, 'n_layers': 2, **kw}
    return cls(**kw)


def _init(model, X):
    variables = model.init(jax.random.PRNGKey(0), X, train=False)
    return variables, model.apply(variables, X, train=False)


@pytest.mark.parametrize('cls', [StationSIREN, StationFINER])
class TestSharedSkeleton:

    def test_logit_shape(self, X, cls):
        _, out = _init(_make(cls), X)
        assert out.shape == (B, N_CLASSES)
        assert np.isfinite(np.asarray(out)).all()

    def test_padded_slot_is_inert(self, X, cls):
        model = _make(cls)
        variables, base = _init(model, X)
        X2 = {k: np.array(v) for k, v in X.items()}
        X2['obs'][0, -1] = 999.0
        X2['lon'][0, -1] = 179.0
        out = model.apply(variables, X2, train=False)
        np.testing.assert_array_equal(np.asarray(base), np.asarray(out))

    def test_perceptron_kernel_uses_first_layer_init(self, X, cls):
        """Station perceptron kernel ~ U(-1/fan_in, 1/fan_in)."""
        variables, _ = _init(_make(cls), X)
        k = np.asarray(variables['params']['station_perceptron']['kernel'])
        assert k.shape == (F_IN, D)
        bound = inr_first_bound(F_IN)
        assert np.abs(k).max() <= bound
        assert np.abs(k).max() > 0.5 * bound      # actually spread, not zeros

    def test_body_first_layer_uses_hidden_init(self, X, cls):
        """The body's first kernel is HIDDEN-initialised (the paper first
        layer already happened in the perceptron)."""
        variables, _ = _init(_make(cls, hidden_omega=30.0), X)
        k = np.asarray(variables['params']['body']['first_layer']['kernel'])
        fan_in = k.shape[0]
        hidden = inr_hidden_bound(fan_in, 30.0)
        # at this fan_in the hidden bound sits well BELOW the first-layer
        # bound, so a kernel inside (and filling) it proves the override
        assert hidden < inr_first_bound(fan_in)
        assert np.abs(k).max() <= hidden
        assert np.abs(k).max() > 0.5 * hidden


class TestVariants:

    def test_siren_perceptron_bias_is_zero(self, X):
        variables, _ = _init(_make(StationSIREN), X)
        b = np.asarray(variables['params']['station_perceptron']['bias'])
        assert (b == 0).all()

    def test_finer_perceptron_bias_is_uniform_k(self, X):
        variables, _ = _init(_make(StationFINER, bias_k=2.0), X)
        b = np.asarray(variables['params']['station_perceptron']['bias'])
        assert np.abs(b).max() <= 2.0
        assert np.abs(b).max() > 0.5            # spread — the FINER point

    def test_finer_body_bias_also_uniform(self, X):
        variables, _ = _init(_make(StationFINER, bias_k=1.0), X)
        b = np.asarray(variables['params']['body']['first_layer']['bias'])
        assert np.abs(b).max() <= 1.0 and np.abs(b).max() > 0.0

    def test_omegas_are_configurable(self, X):
        model = _make(StationSIREN, first_omega=1.0, hidden_omega=5.0)
        variables, out = _init(model, X)
        k = np.asarray(variables['params']['body']['first_layer']['kernel'])
        assert np.abs(k).max() <= inr_hidden_bound(k.shape[0], 5.0)
        assert np.isfinite(np.asarray(out)).all()
