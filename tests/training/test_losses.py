"""
Tests for training/losses.py.

Coverage
--------
TestMse                 plain + masked (mask=None/True/array) MSE: correctness,
                        NaN exclusion, all-NaN -> 0, shape mismatch, grad
                        (zero at NaN), jit
TestReExports           the three optax functions re-exported from training.losses
TestCrossEntropyLoss    compositional CE: defaults match plain CE; focal
                        down-weights easy examples; class weights amplify rare
                        classes; EMD regulariser penalises far mass; all pieces
                        compose; shape/jit/grad
TestLossRegistry        register_loss/get_loss/list_losses; mse + cross_entropy
                        (focal_gamma/class_weights/emd kwargs); filtering/warnings
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax.losses as optax_losses
import pytest

from training.losses import (
    # regression
    mse,
    # optax re-exports
    squared_error,
    softmax_cross_entropy_with_integer_labels,
    # classification
    cross_entropy_loss,
    # loss registry
    LOSSES, get_loss, list_losses, register_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean():
    """Pred and target with no NaN; error = 0.5 everywhere."""
    pred   = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, 2.5, 3.5])
    return pred, target


@pytest.fixture
def with_nan():
    """Middle value NaN — positions 0 and 2 are valid."""
    pred   = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, jnp.nan, 3.5])
    return pred, target


@pytest.fixture
def all_nan():
    pred   = jnp.array([1.0, 2.0, 3.0])
    target = jnp.full(3, jnp.nan)
    return pred, target


# ---------------------------------------------------------------------------
# TestMse — plain + masked (NaN-safe) mean squared error
# ---------------------------------------------------------------------------

class TestMse:

    def test_known_value(self, clean):
        pred, target = clean
        assert float(mse(pred, target)) == pytest.approx(0.25)

    def test_matches_squared_error(self, clean):
        pred, target = clean
        assert jnp.allclose(mse(pred, target), jnp.mean(squared_error(pred, target)))

    def test_perfect_prediction(self, clean):
        pred, _ = clean
        assert float(mse(pred, pred)) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            mse(jnp.zeros(3), jnp.zeros(4))

    def test_scalar_output(self, clean):
        assert mse(*clean).shape == ()

    # --- masked (mask=True derives from finite targets) ---
    def test_mask_true_excludes_nan(self, with_nan):
        pred, target = with_nan          # positions 0,2 valid, error 0.5 each
        assert float(mse(pred, target, mask=True)) == pytest.approx(0.25)

    def test_mask_true_all_nan_returns_zero(self, all_nan):
        assert float(mse(*all_nan, mask=True)) == pytest.approx(0.0)

    def test_explicit_array_mask(self, clean):
        pred, target = clean
        mask = jnp.array([True, False, True])
        assert float(mse(pred, target, mask=mask)) == pytest.approx(0.25)

    def test_mask_none_propagates_nan(self, with_nan):
        pred, target = with_nan
        assert not jnp.isfinite(mse(pred, target))   # plain mean over a NaN -> NaN

    # --- grad / jit ---
    def test_grad_shape_finite(self, clean):
        pred, target = clean
        grad = jax.grad(lambda p: mse(p, target))(pred)
        assert grad.shape == pred.shape and jnp.all(jnp.isfinite(grad))

    def test_grad_zero_at_nan_position(self, with_nan):
        pred, target = with_nan          # position 1 is NaN
        grad = jax.grad(lambda p: mse(p, target, mask=True))(pred)
        assert float(grad[1]) == 0.0
        assert float(grad[0]) != 0.0 and float(grad[2]) != 0.0

    def test_jit_plain_and_masked(self, clean, with_nan):
        assert jnp.isfinite(jax.jit(mse)(*clean))
        assert jnp.isfinite(jax.jit(lambda p, t: mse(p, t, mask=True))(*with_nan))


# ---------------------------------------------------------------------------
# TestReExports — optax losses must be importable from training.losses
# ---------------------------------------------------------------------------

class TestReExports:

    def test_squared_error_importable(self):
        assert callable(squared_error)

    def test_softmax_cross_entropy_with_integer_labels_importable(self):
        assert callable(softmax_cross_entropy_with_integer_labels)

    def test_squared_error_value(self):
        pred   = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 2.5, 3.5])
        assert jnp.allclose(squared_error(pred, target), jnp.array([0.25, 0.25, 0.25]))


# ---------------------------------------------------------------------------
# TestCrossEntropyLoss (compositional: basic / focal / class-weighted)
# ---------------------------------------------------------------------------

class TestCrossEntropyLoss:

    B_CLS = 4
    N_CLS = 11

    def _rand_logits_labels(self, seed=0):
        rng = np.random.default_rng(seed)
        logits = jnp.array(rng.standard_normal((self.B_CLS, self.N_CLS)).astype(np.float32))
        labels = jnp.array(rng.integers(0, self.N_CLS, size=self.B_CLS), dtype=jnp.int32)
        return logits, labels

    # --- basic ---
    def test_defaults_match_plain_cross_entropy(self):
        logits, labels = self._rand_logits_labels()
        out      = cross_entropy_loss(logits, labels)
        expected = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, labels))
        assert jnp.allclose(out, expected)

    def test_scalar_shape(self):
        logits, labels = self._rand_logits_labels()
        assert cross_entropy_loss(logits, labels).shape == ()

    def test_nonnegative(self):
        logits, labels = self._rand_logits_labels()
        assert float(cross_entropy_loss(logits, labels)) >= 0.0

    def test_grad_flows(self):
        logits, labels = self._rand_logits_labels()
        grad = jax.grad(lambda l: cross_entropy_loss(l, labels))(logits)
        assert grad.shape == logits.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit(self):
        logits, labels = self._rand_logits_labels()
        out = jax.jit(lambda l, y: cross_entropy_loss(l, y))(logits, labels)
        assert jnp.isfinite(out)

    # --- focal ---
    def test_focal_gamma_zero_matches_basic(self):
        logits, labels = self._rand_logits_labels()
        assert jnp.allclose(
            cross_entropy_loss(logits, labels, focal_gamma=0.0),
            cross_entropy_loss(logits, labels),
        )

    def test_focal_downweights_easy_examples(self):
        # One confidently-correct (easy) and one wrong (hard) sample.
        labels = jnp.array([0, 1], dtype=jnp.int32)
        logits = jnp.zeros((2, self.N_CLS)).at[0, 0].set(100.0)  # sample 0 easy/correct
        basic = float(cross_entropy_loss(logits, labels))
        focal = float(cross_entropy_loss(logits, labels, focal_gamma=2.0))
        # focal shrinks the (already tiny) easy term; the mean loss drops.
        assert focal < basic

    # --- class-weighted ---
    def test_uniform_weights_match_basic(self):
        logits, labels = self._rand_logits_labels()
        weights = jnp.ones(self.N_CLS)
        assert jnp.allclose(
            cross_entropy_loss(logits, labels, class_weights=weights),
            cross_entropy_loss(logits, labels),
        )

    def test_rare_class_weight_amplifies_its_contribution(self):
        labels = jnp.array([0, 1], dtype=jnp.int32)
        logits = jnp.zeros((2, self.N_CLS)).at[1, 1].set(100.0)   # sample 1 correct
        uniform         = jnp.ones(self.N_CLS)
        upweight_class0 = jnp.ones(self.N_CLS).at[0].set(10.0)
        out_uniform  = float(cross_entropy_loss(logits, labels, class_weights=uniform))
        out_weighted = float(cross_entropy_loss(logits, labels, class_weights=upweight_class0))
        assert out_weighted > out_uniform   # up-weighting the high-loss class raises loss

    # --- EMD regulariser ---
    def test_emd_lambda_zero_matches_basic(self):
        logits, labels = self._rand_logits_labels()
        assert jnp.allclose(
            cross_entropy_loss(logits, labels, emd_lambda=0.0),
            cross_entropy_loss(logits, labels),
        )

    def test_emd_regulariser_changes_loss(self):
        logits, labels = self._rand_logits_labels()
        base = cross_entropy_loss(logits, labels)
        reg  = cross_entropy_loss(logits, labels, emd_lambda=1.0)
        assert not jnp.allclose(base, reg)

    def test_emd_penalises_far_mass_more(self):
        # Same total off-class mass, placed near vs far from the true class.
        # True class 5; compare prob mass on class 4 (near) vs class 0 (far).
        labels = jnp.array([5], dtype=jnp.int32)
        near = jnp.full((1, self.N_CLS), -100.0).at[0, 5].set(0.0).at[0, 4].set(0.0)
        far  = jnp.full((1, self.N_CLS), -100.0).at[0, 5].set(0.0).at[0, 0].set(0.0)
        # positive omega>1, mu=0 -> far mass penalised more
        l_near = float(cross_entropy_loss(near, labels, emd_lambda=1.0, emd_omega=2.0))
        l_far  = float(cross_entropy_loss(far,  labels, emd_lambda=1.0, emd_omega=2.0))
        assert l_far > l_near

    def test_emd_grad_flows(self):
        logits, labels = self._rand_logits_labels()
        grad = jax.grad(
            lambda l: cross_entropy_loss(l, labels, emd_lambda=0.5, emd_mu=-1.0)
        )(logits)
        assert grad.shape == logits.shape
        assert jnp.all(jnp.isfinite(grad))

    # --- composition ---
    def test_all_pieces_compose(self):
        logits, labels = self._rand_logits_labels()
        weights = jnp.ones(self.N_CLS).at[0].set(3.0)
        out = cross_entropy_loss(
            logits, labels, class_weights=weights, focal_gamma=2.0,
            emd_lambda=0.5, emd_omega=2.0, emd_mu=-1.0,
        )
        assert out.shape == ()
        assert bool(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# TestLossRegistry
# ---------------------------------------------------------------------------

class TestLossRegistry:

    B_CLS = 4
    N_CLS = 11

    def _rand_logits_labels(self):
        rng = np.random.default_rng(0)
        logits = jnp.array(rng.standard_normal((self.B_CLS, self.N_CLS)).astype(np.float32))
        labels = jnp.array(rng.integers(0, self.N_CLS, size=self.B_CLS), dtype=jnp.int32)
        return logits, labels

    def test_cross_entropy_registered(self):
        assert 'CROSS_ENTROPY' in LOSSES

    def test_get_loss_cross_entropy_with_emd(self):
        logits, labels = self._rand_logits_labels()
        loss_fn = get_loss('cross_entropy', emd_lambda=0.5, emd_omega=2.0, emd_mu=-1.0)
        out = loss_fn(logits, labels)
        assert out.shape == ()
        assert bool(jnp.isfinite(out))

    def test_get_loss_cross_entropy(self):
        logits, labels = self._rand_logits_labels()
        loss_fn = get_loss('cross_entropy')
        out = loss_fn(logits, labels)
        expected = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, labels))
        assert jnp.allclose(out, expected)

    def test_get_loss_cross_entropy_with_class_weights(self):
        logits, labels = self._rand_logits_labels()
        weights = [1.0] * self.N_CLS
        loss_fn = get_loss('cross_entropy', class_weights=weights)
        out = loss_fn(logits, labels)
        # uniform weights -> identical to plain CE
        expected = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, labels))
        assert jnp.allclose(out, expected)

    def test_get_loss_cross_entropy_with_focal(self):
        logits, labels = self._rand_logits_labels()
        loss_fn = get_loss('cross_entropy', focal_gamma=2.0)
        out = loss_fn(logits, labels)
        assert out.shape == ()
        assert bool(jnp.isfinite(out))
        # focal differs from plain CE in general
        plain = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, labels))
        assert not jnp.allclose(out, plain)

    def test_get_loss_class_balanced_focal_composes(self):
        logits, labels = self._rand_logits_labels()
        weights = [1.0] * self.N_CLS
        loss_fn = get_loss('cross_entropy', class_weights=weights, focal_gamma=2.0)
        out = loss_fn(logits, labels)
        assert out.shape == ()
        assert bool(jnp.isfinite(out))

    def test_get_loss_case_insensitive(self):
        assert get_loss('Cross_Entropy') is not None
        assert get_loss('CROSS_ENTROPY') is not None

    def test_get_loss_unknown_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            get_loss('not_a_real_loss')

    def test_get_loss_unknown_kwargs_warns_and_drops(self):
        logits, labels = self._rand_logits_labels()
        with pytest.warns(UserWarning, match="unknown kwargs"):
            loss_fn = get_loss('cross_entropy', bogus_kwarg=123)
        expected = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, labels))
        assert jnp.allclose(loss_fn(logits, labels), expected)

    def test_list_losses_returns_descriptions(self):
        names = list_losses()
        assert 'CROSS_ENTROPY' in names
        assert all(isinstance(v, str) for v in names.values())

    def test_register_loss_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_loss('cross_entropy')
            def _dup():
                ...
