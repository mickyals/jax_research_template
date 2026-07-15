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
TestModelTerms          MODEL_TERMS registry + l1_params: manual value, ignores
                        pred/batch, grad w.r.t. params
TestLossStack           build_loss_stack: single/weighted/multi-term folds match
                        manual sums; needs_model/term_names; detailed() unweighted
                        per-term values; weight-0 monitor; model-kwarg guard;
                        repeated-name suffixing; ambiguous/unknown names;
                        jit + grad through a mixed stack
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
    # loss stack (weighted term list) + model-term registry
    MODEL_TERMS, register_model_term, LossStack, build_loss_stack,
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


# ---------------------------------------------------------------------------
# TestModelTerms — MODEL_TERMS registry + the l1_params example term
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_params():
    """Params pytree with known L1: leaves [1,-2,3] and [[−4],[0]] ->
    sum(abs)=10 over 5 elements -> mean 2.0."""
    return {
        'dense': {'kernel': jnp.array([1.0, -2.0, 3.0]),
                  'bias':   jnp.array([[-4.0], [0.0]])},
    }


class TestModelTerms:

    def test_l1_params_registered(self):
        assert 'L1_PARAMS' in MODEL_TERMS

    def test_l1_params_manual_value(self, toy_params):
        term = MODEL_TERMS.get('l1_params')
        out  = term(toy_params, None, None, None)
        assert float(out) == pytest.approx(2.0)
        assert out.shape == ()

    def test_l1_params_ignores_pred_and_batch(self, toy_params):
        term = MODEL_TERMS.get('l1_params')
        a = term(toy_params, None, None, None)
        b = term(toy_params, lambda *_: None, {'X': jnp.zeros(3)}, jnp.ones(2))
        assert jnp.allclose(a, b)

    def test_l1_params_grad_wrt_params(self, toy_params):
        term = MODEL_TERMS.get('l1_params')
        grads = jax.grad(lambda p: term(p, None, None, None))(toy_params)
        # d(mean |w|)/dw = sign(w)/n, n=5
        assert jnp.allclose(grads['dense']['kernel'],
                            jnp.array([0.2, -0.2, 0.2]))

    def test_l2_params_registered(self):
        assert 'L2_PARAMS' in MODEL_TERMS

    def test_l2_params_manual_value(self, toy_params):
        # sum(w^2) = 1+4+9+16+0 = 30 over 5 elements -> mean 6.0
        term = MODEL_TERMS.get('l2_params')
        out  = term(toy_params, None, None, None)
        assert float(out) == pytest.approx(6.0)
        assert out.shape == ()

    def test_l2_params_grad_wrt_params(self, toy_params):
        term = MODEL_TERMS.get('l2_params')
        grads = jax.grad(lambda p: term(p, None, None, None))(toy_params)
        # d(mean w^2)/dw = 2w/n, n=5
        assert jnp.allclose(grads['dense']['kernel'],
                            jnp.array([0.4, -0.8, 1.2]))

    def test_register_model_term_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_model_term('l1_params')
            def _dup():
                ...


# ---------------------------------------------------------------------------
# TestLossStack — build_loss_stack folds a term list into one callable
# ---------------------------------------------------------------------------

class TestLossStack:

    B_CLS = 4
    N_CLS = 11

    def _rand_logits_labels(self, seed=0):
        rng = np.random.default_rng(seed)
        logits = jnp.array(rng.standard_normal((self.B_CLS, self.N_CLS)).astype(np.float32))
        labels = jnp.array(rng.integers(0, self.N_CLS, size=self.B_CLS), dtype=jnp.int32)
        return logits, labels

    # --- single prediction term: parity with get_loss ---
    def test_single_term_matches_get_loss(self):
        logits, labels = self._rand_logits_labels()
        stack = build_loss_stack([{'name': 'cross_entropy'}])
        assert jnp.allclose(stack(logits, labels),
                            get_loss('cross_entropy')(logits, labels))

    def test_single_term_attrs(self):
        stack = build_loss_stack([{'name': 'cross_entropy'}])
        assert isinstance(stack, LossStack)
        assert stack.needs_model is False
        assert stack.term_names == ('cross_entropy',)

    def test_term_kwargs_passed_through(self):
        logits, labels = self._rand_logits_labels()
        stack = build_loss_stack(
            [{'name': 'cross_entropy', 'kwargs': {'focal_gamma': 2.0}}])
        assert jnp.allclose(
            stack(logits, labels),
            get_loss('cross_entropy', focal_gamma=2.0)(logits, labels))

    # --- weights ---
    def test_weight_scales_term(self):
        logits, labels = self._rand_logits_labels()
        base   = build_loss_stack([{'name': 'cross_entropy'}])
        scaled = build_loss_stack([{'name': 'cross_entropy', 'weight': 2.0}])
        assert jnp.allclose(scaled(logits, labels), 2.0 * base(logits, labels))

    def test_two_prediction_terms_weighted_sum(self):
        logits, labels = self._rand_logits_labels()
        stack = build_loss_stack([
            {'name': 'cross_entropy', 'weight': 1.0},
            {'name': 'cross_entropy', 'weight': 0.5,
             'kwargs': {'focal_gamma': 2.0}},
        ])
        ce    = get_loss('cross_entropy')(logits, labels)
        focal = get_loss('cross_entropy', focal_gamma=2.0)(logits, labels)
        assert jnp.allclose(stack(logits, labels), ce + 0.5 * focal)
        # repeated names get suffixed labels so curves stay distinct
        assert stack.term_names == ('cross_entropy', 'cross_entropy_2')
        _, values = stack.detailed(logits, labels)
        assert jnp.allclose(values['cross_entropy'], ce)
        assert jnp.allclose(values['cross_entropy_2'], focal)

    # --- model terms ---
    def test_model_term_needs_model(self):
        stack = build_loss_stack([{'name': 'l1_params'}])
        assert stack.needs_model is True

    def test_mixed_stack_matches_manual(self, toy_params):
        logits, labels = self._rand_logits_labels()
        stack = build_loss_stack([
            {'name': 'cross_entropy'},
            {'name': 'l1_params', 'weight': 0.1},
        ])
        ce = get_loss('cross_entropy')(logits, labels)
        l1 = MODEL_TERMS.get('l1_params')(toy_params, None, None, None)
        out = stack(logits, labels, params=toy_params)
        assert jnp.allclose(out, ce + 0.1 * l1)
        assert stack.needs_model is True
        assert stack.term_names == ('cross_entropy', 'l1_params')

    def test_needs_model_without_params_raises(self):
        logits, labels = self._rand_logits_labels()
        stack = build_loss_stack([{'name': 'l1_params'}])
        with pytest.raises(ValueError, match="params"):
            stack(logits, labels)

    # --- detailed(): per-term unweighted values ---
    def test_detailed_returns_total_and_unweighted_values(self, toy_params):
        logits, labels = self._rand_logits_labels()
        stack = build_loss_stack([
            {'name': 'cross_entropy'},
            {'name': 'l1_params', 'weight': 0.1},
        ])
        total, values = stack.detailed(logits, labels, params=toy_params)
        assert jnp.allclose(total, stack(logits, labels, params=toy_params))
        assert set(values) == {'cross_entropy', 'l1_params'}
        # values are UNWEIGHTED so curves stay comparable across weight sweeps
        assert jnp.allclose(values['l1_params'],
                            MODEL_TERMS.get('l1_params')(toy_params, None, None, None))
        assert jnp.allclose(values['cross_entropy'],
                            get_loss('cross_entropy')(logits, labels))

    def test_weight_zero_is_a_monitor_term(self, toy_params):
        logits, labels = self._rand_logits_labels()
        stack = build_loss_stack([
            {'name': 'cross_entropy'},
            {'name': 'l1_params', 'weight': 0.0},
        ])
        total, values = stack.detailed(logits, labels, params=toy_params)
        assert jnp.allclose(total, get_loss('cross_entropy')(logits, labels))
        assert float(values['l1_params']) == pytest.approx(2.0)

    # --- validation ---
    def test_unknown_name_lists_both_registries(self):
        with pytest.raises(ValueError, match="(?s)prediction.*model term"):
            build_loss_stack([{'name': 'not_a_real_term'}])

    def test_repeated_name_labels_are_suffixed(self):
        stack = build_loss_stack([{'name': 'cross_entropy'},
                                  {'name': 'cross_entropy'},
                                  {'name': 'cross_entropy'}])
        assert stack.term_names == ('cross_entropy', 'cross_entropy_2',
                                    'cross_entropy_3')

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            build_loss_stack([{'weight': 1.0}])

    def test_unknown_term_key_raises(self):
        with pytest.raises(ValueError, match="wieght"):
            build_loss_stack([{'name': 'cross_entropy', 'wieght': 1.0}])

    def test_empty_terms_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            build_loss_stack([])

    def test_name_in_both_registries_is_ambiguous(self):
        # Register a throwaway name in BOTH registries (session-global —
        # unique name so no other test collides).
        @register_loss('_ambig_stack_test')
        def _p():
            return lambda pred, y: jnp.asarray(0.0)

        @register_model_term('_ambig_stack_test')
        def _m():
            return lambda params, apply_fn, batch, pred: jnp.asarray(0.0)

        with pytest.raises(ValueError, match="ambiguous"):
            build_loss_stack([{'name': '_ambig_stack_test'}])

    # --- jit / grad ---
    def test_jit_prediction_stack(self):
        logits, labels = self._rand_logits_labels()
        stack = build_loss_stack([{'name': 'cross_entropy'}])
        assert jnp.isfinite(jax.jit(stack)(logits, labels))

    def test_grad_through_mixed_stack(self, toy_params):
        logits, labels = self._rand_logits_labels()
        stack = build_loss_stack([
            {'name': 'cross_entropy'},
            {'name': 'l1_params', 'weight': 0.1},
        ])
        grads = jax.grad(
            lambda p: stack(logits, labels, params=p))(toy_params)
        assert jnp.all(jnp.isfinite(grads['dense']['kernel']))
        # 0.1 * sign(w)/n with n=5 -> magnitude 0.02
        assert jnp.allclose(jnp.abs(grads['dense']['kernel']), 0.02)
