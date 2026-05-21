import pytest
import jax
import jax.numpy as jnp

from utils.jax_core.helpers import (
    check_environment,
    create_rng,
    create_rng_dict,
    split_rng,
    show_jaxpr,
    grad_fn,
)


# ---------------------------------------------------------------------------
# check_environment
# ---------------------------------------------------------------------------

class TestCheckEnvironment:

    def test_prints_version(self, capsys):
        check_environment()
        captured = capsys.readouterr()
        assert "JAX version:" in captured.out

    def test_prints_backend(self, capsys):
        check_environment()
        captured = capsys.readouterr()
        assert "Backend:" in captured.out

    def test_prints_gpu_or_cpu(self, capsys):
        check_environment()
        captured = capsys.readouterr()
        assert "GPU" in captured.out or "CPU" in captured.out or "cpu" in captured.out


# ---------------------------------------------------------------------------
# create_rng
# ---------------------------------------------------------------------------

class TestCreateRng:

    def test_returns_array(self):
        rng = create_rng(0)
        assert isinstance(rng, jax.Array)

    def test_shape(self):
        rng = create_rng(0)
        assert rng.shape == (2,)

    def test_default_seed(self):
        rng = create_rng()
        assert rng is not None

    def test_deterministic(self):
        rng1 = create_rng(123)
        rng2 = create_rng(123)
        assert jnp.array_equal(rng1, rng2)

    def test_different_seeds_differ(self):
        rng1 = create_rng(0)
        rng2 = create_rng(1)
        assert not jnp.array_equal(rng1, rng2)


# ---------------------------------------------------------------------------
# create_rng_dict
# ---------------------------------------------------------------------------

class TestCreateRngDict:

    def test_default_keys(self):
        rngs = create_rng_dict(0)
        assert "params" in rngs
        assert "dropout" in rngs
        assert len(rngs) == 2

    def test_custom_keys(self):
        rngs = create_rng_dict(0, keys=["a", "b", "c"])
        assert list(rngs.keys()) == ["a", "b", "c"]

    def test_values_are_arrays(self):
        rngs = create_rng_dict(0)
        for v in rngs.values():
            assert isinstance(v, jax.Array)
            assert v.shape == (2,)

    def test_keys_are_distinct(self):
        rngs = create_rng_dict(0, keys=["params", "dropout"])
        assert not jnp.array_equal(rngs["params"], rngs["dropout"])

    def test_deterministic(self):
        rngs1 = create_rng_dict(42)
        rngs2 = create_rng_dict(42)
        for k in rngs1:
            assert jnp.array_equal(rngs1[k], rngs2[k])

    def test_single_key(self):
        rngs = create_rng_dict(0, keys=["only"])
        assert len(rngs) == 1
        assert "only" in rngs


# ---------------------------------------------------------------------------
# split_rng
# ---------------------------------------------------------------------------

class TestSplitRng:

    def test_returns_two_arrays(self):
        rng = create_rng(0)
        result = split_rng(rng)
        assert len(result) == 2

    def test_both_are_arrays(self):
        rng = create_rng(0)
        new_rng, subkey = split_rng(rng)
        assert isinstance(new_rng, jax.Array)
        assert isinstance(subkey, jax.Array)

    def test_shapes(self):
        rng = create_rng(0)
        new_rng, subkey = split_rng(rng)
        assert new_rng.shape == (2,)
        assert subkey.shape == (2,)

    def test_outputs_differ(self):
        rng = create_rng(0)
        new_rng, subkey = split_rng(rng)
        assert not jnp.array_equal(new_rng, subkey)

    def test_differs_from_original(self):
        rng = create_rng(0)
        new_rng, subkey = split_rng(rng)
        assert not jnp.array_equal(rng, new_rng)
        assert not jnp.array_equal(rng, subkey)

    def test_deterministic(self):
        rng1_a, key1_a = split_rng(create_rng(99))
        rng1_b, key1_b = split_rng(create_rng(99))
        assert jnp.array_equal(rng1_a, rng1_b)
        assert jnp.array_equal(key1_a, key1_b)

    def test_chaining(self):
        rng = create_rng(0)
        rng, key1 = split_rng(rng)
        rng, key2 = split_rng(rng)
        assert not jnp.array_equal(key1, key2)


# ---------------------------------------------------------------------------
# show_jaxpr
# ---------------------------------------------------------------------------

class TestShowJaxpr:

    def test_prints_output(self, capsys):
        show_jaxpr(lambda x: x + 1, jnp.array(1.0))
        captured = capsys.readouterr()
        assert "lambda" in captured.out or "let" in captured.out

    def test_quadratic(self, capsys):
        show_jaxpr(lambda x: x ** 2, jnp.array(3.0))
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_multi_input(self, capsys):
        show_jaxpr(lambda x, y: x + y, jnp.array(1.0), jnp.array(2.0))
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_static_argnums(self, capsys):
        def f(x, n):
            return x ** n
        show_jaxpr(f, jnp.array(2.0), 3, static_argnums=1)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_none_static_argnums(self, capsys):
        show_jaxpr(lambda x: x, jnp.array(1.0), static_argnums=None)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ---------------------------------------------------------------------------
# grad_fn
# ---------------------------------------------------------------------------

class TestGradFn:

    def test_returns_callable(self):
        f = grad_fn(lambda x: x ** 2)
        assert callable(f)

    def test_value_and_grad_cubic(self):
        f = grad_fn(lambda x: x ** 3)
        val, grad = f(2.0)
        assert jnp.isclose(val, 8.0)
        assert jnp.isclose(grad, 12.0)

    def test_value_and_grad_quadratic(self):
        f = grad_fn(lambda x: x ** 2)
        val, grad = f(3.0)
        assert jnp.isclose(val, 9.0)
        assert jnp.isclose(grad, 6.0)

    def test_linear(self):
        f = grad_fn(lambda x: 5.0 * x)
        val, grad = f(1.0)
        assert jnp.isclose(val, 5.0)
        assert jnp.isclose(grad, 5.0)

    def test_argnums(self):
        def f(x, y):
            return x * y
        vg = grad_fn(f, argnums=1)
        val, grad = vg(3.0, 4.0)
        assert jnp.isclose(val, 12.0)
        assert jnp.isclose(grad, 3.0)  # d(x*y)/dy = x

    def test_has_aux(self):
        def f(x):
            return x ** 2, {"aux": x}
        vg = grad_fn(f, has_aux=True)
        (val, aux), grad = vg(3.0)
        assert jnp.isclose(val, 9.0)
        assert jnp.isclose(aux["aux"], 3.0)
        assert jnp.isclose(grad, 6.0)

    def test_deterministic(self):
        f = grad_fn(lambda x: x ** 2)
        val1, grad1 = f(4.0)
        val2, grad2 = f(4.0)
        assert jnp.isclose(val1, val2)
        assert jnp.isclose(grad1, grad2)