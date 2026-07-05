"""
Tests for train/losses.py — the trainer-yaml -> LossStack glue. Stack
mechanics (folding, per-term values, model-term contract) are jrt
territory, tested in tests/training/test_losses.py; here only the two
yaml forms and their conflicts.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from training.losses import LossStack, get_loss

from experiments.cyclone_jax.train.losses import build_loss


def _logits_labels(seed=0):
    rng = np.random.default_rng(seed)
    logits = jnp.array(rng.standard_normal((4, 6)).astype(np.float32))
    labels = jnp.array(rng.integers(0, 6, size=4), dtype=jnp.int32)
    return logits, labels


class TestBuildLoss:

    def test_default_is_cross_entropy(self):
        stack = build_loss({})
        assert isinstance(stack, LossStack)
        assert stack.term_names == ('cross_entropy',)
        assert stack.needs_model is False
        logits, labels = _logits_labels()
        assert jnp.allclose(stack(logits, labels),
                            get_loss('cross_entropy')(logits, labels))

    def test_bare_string_with_loss_kwargs(self):
        stack = build_loss({'loss': 'cross_entropy',
                            'loss_kwargs': {'focal_gamma': 2.0}})
        logits, labels = _logits_labels()
        assert jnp.allclose(
            stack(logits, labels),
            get_loss('cross_entropy', focal_gamma=2.0)(logits, labels))

    def test_term_list_form(self):
        stack = build_loss({'loss': [
            {'name': 'cross_entropy'},
            {'name': 'l1_params', 'weight': 1.0e-4},
        ]})
        assert stack.term_names == ('cross_entropy', 'l1_params')
        assert stack.needs_model is True

    def test_term_list_with_loss_kwargs_raises(self):
        with pytest.raises(ValueError, match='loss_kwargs'):
            build_loss({'loss': [{'name': 'cross_entropy'}],
                        'loss_kwargs': {'focal_gamma': 2.0}})

    def test_unknown_term_name_raises(self):
        with pytest.raises(ValueError, match='not_a_term'):
            build_loss({'loss': [{'name': 'not_a_term'}]})
