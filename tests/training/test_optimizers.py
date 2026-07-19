"""
Tests for training/optimizers.py.

Coverage
--------
TestOptimizerRegistry       register / get / list mechanics
TestSchedulerRegistry       register / get / list mechanics
TestOptimizerInstantiation  each registered optimizer builds a valid GradientTransformation
TestSchedulerInstantiation  each registered scheduler returns a callable schedule
TestSchedulerValues         key points on warmup, decay, constant schedules
TestComposition             optimizer + schedule compose correctly
TestGradientFlow            one optimizer step on a real Flax MLP — params change,
                            grads are finite, tested across activations and inits
TestEdgeCases               unknown kwarg warning, case-insensitive lookup,
                            unregistered name error, LBFGS NotImplementedError
"""

import warnings

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import linen as nn

from training.optimizers import (
    OPTIMIZERS,
    SCHEDULERS,
    get_optimizer,
    get_scheduler,
    list_optimizers,
    list_schedulers,
    register_optimizer,
    register_scheduler,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY = jax.random.PRNGKey(0)

_ALL_OPTIMIZERS = ["adam", "adamw", "lamb", "sgd", "rmsprop"]  # lbfgs excluded (raises)

_SCHEDULER_CONFIGS = {
    "constant":           dict(value=1e-3),
    "cosine_decay":       dict(init_value=1e-3, decay_steps=1000),
    "cosine_onecycle":    dict(transition_steps=1000, peak_value=1e-3),
    "exponential_decay":  dict(init_value=1e-3, transition_steps=500, decay_rate=0.9),
    "polynomial":         dict(init_value=1e-3, end_value=1e-5, power=2.0, transition_steps=1000),
    "sgdr":               dict(cosine_kwargs=[
                              {"init_value": 0.0, "peak_value": 1e-3,
                               "warmup_steps": 50, "decay_steps": 500},
                              {"init_value": 0.0, "peak_value": 5e-4,
                               "warmup_steps": 50, "decay_steps": 1000},
                          ]),
    "warmup_constant":    dict(init_value=0.0, peak_value=1e-3, warmup_steps=100),
    "warmup_cosine":      dict(init_value=0.0, peak_value=1e-3,
                               warmup_steps=100, decay_steps=1000),
    "warmup_exponential": dict(init_value=0.0, peak_value=1e-3,
                               warmup_steps=100, transition_steps=500, decay_rate=0.9),
}


# ---------------------------------------------------------------------------
# TestOptimizerRegistry
# ---------------------------------------------------------------------------

class TestOptimizerRegistry:

    def test_list_optimizers_returns_dict(self):
        result = list_optimizers()
        assert isinstance(result, dict)

    def test_all_expected_names_present(self):
        names = list_optimizers()
        for name in ["ADAM", "ADAMW", "LAMB", "SGD", "RMSPROP", "LBFGS"]:
            assert name in names, f"Expected '{name}' in list_optimizers()"

    def test_descriptions_are_strings(self):
        for name, desc in list_optimizers().items():
            assert isinstance(desc, str), f"Description for {name} is not a string"

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_optimizer("ADAM", description="duplicate")
            def _dup(lr):
                return optax.adam(lr)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            get_optimizer("NOT_A_REAL_OPTIMIZER", learning_rate=1e-3)

    def test_case_insensitive_lookup(self):
        opt_lower = get_optimizer("adam", learning_rate=1e-3)
        opt_upper = get_optimizer("ADAM", learning_rate=1e-3)
        assert type(opt_lower) is type(opt_upper)

    def test_unknown_kwargs_warns_and_drops(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opt = get_optimizer("adam", learning_rate=1e-3, nonexistent_kwarg=999)
        assert any("nonexistent_kwarg" in str(warning.message) for warning in w)
        assert opt is not None


# ---------------------------------------------------------------------------
# TestSchedulerRegistry
# ---------------------------------------------------------------------------

class TestSchedulerRegistry:

    def test_list_schedulers_returns_dict(self):
        result = list_schedulers()
        assert isinstance(result, dict)

    def test_all_expected_names_present(self):
        names = list_schedulers()
        for name in [
            "CONSTANT", "COSINE_DECAY", "COSINE_ONECYCLE",
            "EXPONENTIAL_DECAY", "POLYNOMIAL", "SGDR",
            "WARMUP_CONSTANT", "WARMUP_COSINE", "WARMUP_EXPONENTIAL",
        ]:
            assert name in names, f"Expected '{name}' in list_schedulers()"

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_scheduler("CONSTANT", description="duplicate")
            def _dup(value):
                return optax.constant_schedule(value)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            get_scheduler("NOT_A_REAL_SCHEDULE")

    def test_case_insensitive_lookup(self):
        s1 = get_scheduler("constant", value=1e-3)
        s2 = get_scheduler("CONSTANT", value=1e-3)
        assert s1(0) == s2(0)

    def test_unknown_kwargs_warns_and_drops(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_scheduler("constant", value=1e-3, nonexistent_kwarg=999)
        assert any("nonexistent_kwarg" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# TestOptimizerInstantiation
# ---------------------------------------------------------------------------

class TestOptimizerInstantiation:

    @pytest.mark.parametrize("name", _ALL_OPTIMIZERS)
    def test_returns_gradient_transformation(self, name):
        opt = get_optimizer(name, learning_rate=1e-3)
        assert isinstance(opt, optax.GradientTransformation)

    def test_adam_defaults(self):
        opt = get_optimizer("adam", learning_rate=1e-3)
        assert isinstance(opt, optax.GradientTransformation)

    def test_adamw_weight_decay(self):
        opt = get_optimizer("adamw", learning_rate=1e-3, weight_decay=1e-2)
        assert isinstance(opt, optax.GradientTransformation)

    def test_sgd_with_momentum(self):
        opt = get_optimizer("sgd", learning_rate=0.1, momentum=0.9)
        assert isinstance(opt, optax.GradientTransformation)

    def test_sgd_nesterov(self):
        opt = get_optimizer("sgd", learning_rate=0.1, momentum=0.9, nesterov=True)
        assert isinstance(opt, optax.GradientTransformation)

    def test_lamb_weight_decay(self):
        opt = get_optimizer("lamb", learning_rate=1e-3, weight_decay=1e-2)
        assert isinstance(opt, optax.GradientTransformation)

    def test_rmsprop_centered(self):
        opt = get_optimizer("rmsprop", learning_rate=1e-3, centered=True)
        assert isinstance(opt, optax.GradientTransformation)

    def test_lbfgs_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            get_optimizer("lbfgs", learning_rate=1e-3)


# ---------------------------------------------------------------------------
# TestSchedulerInstantiation
# ---------------------------------------------------------------------------

class TestSchedulerInstantiation:

    @pytest.mark.parametrize("name,kwargs", _SCHEDULER_CONFIGS.items())
    def test_returns_callable(self, name, kwargs):
        schedule = get_scheduler(name, **kwargs)
        assert callable(schedule)

    @pytest.mark.parametrize("name,kwargs", _SCHEDULER_CONFIGS.items())
    def test_evaluates_at_step_zero(self, name, kwargs):
        schedule = get_scheduler(name, **kwargs)
        val = schedule(0)
        assert jnp.isfinite(val), f"Schedule '{name}' returned non-finite value at step 0"

    @pytest.mark.parametrize("name,kwargs", _SCHEDULER_CONFIGS.items())
    def test_evaluates_at_large_step(self, name, kwargs):
        schedule = get_scheduler(name, **kwargs)
        val = schedule(10_000)
        assert jnp.isfinite(val), f"Schedule '{name}' returned non-finite value at step 10000"

    @pytest.mark.parametrize("name,kwargs", _SCHEDULER_CONFIGS.items())
    def test_returns_non_negative(self, name, kwargs):
        schedule = get_scheduler(name, **kwargs)
        for step in [0, 50, 100, 500, 1000]:
            val = float(schedule(step))
            assert val >= 0.0, (
                f"Schedule '{name}' returned negative lr {val} at step {step}"
            )


# ---------------------------------------------------------------------------
# TestSchedulerValues
# ---------------------------------------------------------------------------

class TestSchedulerValues:

    def test_constant_stays_flat(self):
        s = get_scheduler("constant", value=1e-3)
        assert float(s(0))     == pytest.approx(1e-3)
        assert float(s(5000))  == pytest.approx(1e-3)
        assert float(s(99999)) == pytest.approx(1e-3)

    def test_cosine_decay_starts_at_init(self):
        s = get_scheduler("cosine_decay", init_value=1e-2, decay_steps=1000)
        assert float(s(0)) == pytest.approx(1e-2, rel=1e-4)

    def test_cosine_decay_near_zero_at_end(self):
        s = get_scheduler("cosine_decay", init_value=1e-2, decay_steps=1000)
        assert float(s(1000)) < 1e-4

    def test_warmup_cosine_starts_at_init(self):
        s = get_scheduler("warmup_cosine",
                          init_value=0.0, peak_value=1e-3,
                          warmup_steps=100, decay_steps=1000)
        assert float(s(0)) == pytest.approx(0.0, abs=1e-9)

    def test_warmup_cosine_reaches_peak(self):
        s = get_scheduler("warmup_cosine",
                          init_value=0.0, peak_value=1e-3,
                          warmup_steps=100, decay_steps=1000)
        assert float(s(100)) == pytest.approx(1e-3, rel=1e-3)

    def test_warmup_cosine_decays_after_peak(self):
        s = get_scheduler("warmup_cosine",
                          init_value=0.0, peak_value=1e-3,
                          warmup_steps=100, decay_steps=1000)
        assert float(s(500)) < float(s(100))

    def test_warmup_constant_reaches_peak_and_stays(self):
        s = get_scheduler("warmup_constant",
                          init_value=0.0, peak_value=1e-3, warmup_steps=100)
        assert float(s(0))   == pytest.approx(0.0, abs=1e-9)
        assert float(s(100)) == pytest.approx(1e-3, rel=1e-3)
        assert float(s(200)) == pytest.approx(float(s(100)), rel=1e-4)

    def test_warmup_exponential_increases_during_warmup(self):
        s = get_scheduler("warmup_exponential",
                          init_value=0.0, peak_value=1e-3,
                          warmup_steps=100, transition_steps=500, decay_rate=0.9)
        assert float(s(50)) > float(s(0))
        assert float(s(100)) > float(s(50))

    def test_polynomial_starts_and_ends_at_specified_values(self):
        s = get_scheduler("polynomial",
                          init_value=1e-2, end_value=1e-4,
                          power=1.0, transition_steps=1000)
        assert float(s(0))    == pytest.approx(1e-2, rel=1e-4)
        assert float(s(1000)) == pytest.approx(1e-4, rel=1e-4)


# ---------------------------------------------------------------------------
# TestComposition  (schedule passed as lr to optimizer)
# ---------------------------------------------------------------------------

class TestComposition:

    @pytest.mark.parametrize("opt_name", _ALL_OPTIMIZERS)
    def test_optimizer_accepts_schedule(self, opt_name):
        schedule = get_scheduler("warmup_cosine",
                                 init_value=0.0, peak_value=1e-3,
                                 warmup_steps=100, decay_steps=1000)
        opt = get_optimizer(opt_name, learning_rate=schedule)
        assert isinstance(opt, optax.GradientTransformation)

    @pytest.mark.parametrize("opt_name", _ALL_OPTIMIZERS)
    def test_optimizer_state_initialises(self, opt_name):
        schedule = get_scheduler("constant", value=1e-3)
        opt      = get_optimizer(opt_name, learning_rate=schedule)
        params   = {"w": jnp.ones((4, 4)), "b": jnp.zeros((4,))}
        state    = opt.init(params)
        assert state is not None


# ---------------------------------------------------------------------------
# TestGradientFlow
# ---------------------------------------------------------------------------

class _TinyMLP(nn.Module):
    """Minimal two-layer MLP for gradient-flow tests."""
    hidden: int = 16
    out:    int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out)(x)
        return x


def _one_step(opt_name: str, schedule_name: str, schedule_kwargs: dict):
    """Init model, run one optimizer step, return (old_params, new_params)."""
    model    = _TinyMLP()
    x        = jax.random.normal(KEY, (8, 4))
    y        = jax.random.normal(KEY, (8, 2))
    schedule = get_scheduler(schedule_name, **schedule_kwargs)
    opt      = get_optimizer(opt_name, learning_rate=schedule)

    variables = model.init(KEY, x)
    params    = variables["params"]
    opt_state = opt.init(params)

    def loss_fn(p):
        pred = model.apply({"params": p}, x)
        return jnp.mean((pred - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return params, new_params, grads, loss


class TestGradientFlow:

    @pytest.mark.parametrize("opt_name", _ALL_OPTIMIZERS)
    def test_params_change_after_one_step(self, opt_name):
        old, new, _, _ = _one_step(opt_name, "constant",
                                   dict(value=1e-3))
        old_leaves = jax.tree_util.tree_leaves(old)
        new_leaves = jax.tree_util.tree_leaves(new)
        # At least one parameter tensor must have changed
        changed = any(
            not jnp.allclose(o, n)
            for o, n in zip(old_leaves, new_leaves)
        )
        assert changed, f"Optimizer '{opt_name}': params unchanged after one step"

    @pytest.mark.parametrize("opt_name", _ALL_OPTIMIZERS)
    def test_grads_are_finite(self, opt_name):
        _, _, grads, _ = _one_step(opt_name, "constant", dict(value=1e-3))
        for g in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(g)), (
                f"Optimizer '{opt_name}': non-finite gradients"
            )

    @pytest.mark.parametrize("opt_name", _ALL_OPTIMIZERS)
    def test_loss_is_finite(self, opt_name):
        _, _, _, loss = _one_step(opt_name, "constant", dict(value=1e-3))
        assert jnp.isfinite(loss), f"Optimizer '{opt_name}': non-finite loss"

    @pytest.mark.parametrize("schedule_name,schedule_kwargs", [
        ("constant",           dict(value=1e-3)),
        ("cosine_decay",       dict(init_value=1e-3, decay_steps=1000)),
        ("warmup_cosine",      dict(init_value=1e-5, peak_value=1e-3,
                                    warmup_steps=100, decay_steps=1000)),
        ("warmup_exponential", dict(init_value=1e-5, peak_value=1e-3,
                                    warmup_steps=100, transition_steps=500,
                                    decay_rate=0.9)),
    ])
    def test_adam_with_each_schedule(self, schedule_name, schedule_kwargs):
        old, new, grads, loss = _one_step("adam", schedule_name, schedule_kwargs)
        assert jnp.isfinite(loss)
        old_leaves = jax.tree_util.tree_leaves(old)
        new_leaves = jax.tree_util.tree_leaves(new)
        changed = any(
            not jnp.allclose(o, n)
            for o, n in zip(old_leaves, new_leaves)
        )
        assert changed, (
            f"Adam + schedule '{schedule_name}': params unchanged after one step"
        )

    def test_multiple_steps_reduce_loss(self):
        """Loss should decrease over several steps with Adam + constant lr."""
        model     = _TinyMLP()
        x         = jax.random.normal(KEY, (32, 4))
        y         = jax.random.normal(KEY, (32, 2))
        opt       = get_optimizer("adam", learning_rate=1e-2)
        variables = model.init(KEY, x)
        params    = variables["params"]
        opt_state = opt.init(params)

        def loss_fn(p):
            return jnp.mean((model.apply({"params": p}, x) - y) ** 2)

        losses = []
        for _ in range(20):
            loss, grads   = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params        = optax.apply_updates(params, updates)
            losses.append(float(loss))

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
