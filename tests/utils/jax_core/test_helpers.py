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
    degrees_to_radians,
    radians_to_degrees,
    latlon_deg_to_rad,
    latlon_rad_to_deg,
    spherical_to_cartesian,
    cartesian_to_spherical,
    minmax_norm,
    standardise
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



# ---------------------------------------------------------------------------
# degrees_to_radians / radians_to_degrees
# ---------------------------------------------------------------------------

class TestAngleConversion:

    def test_degrees_to_radians_known_values(self):
        x = jnp.array([0., 90., 180., 360.])
        out = degrees_to_radians(x)
        expected = jnp.array([0., jnp.pi / 2, jnp.pi, 2 * jnp.pi])
        assert jnp.allclose(out, expected, atol=1e-6)

    def test_radians_to_degrees_known_values(self):
        x = jnp.array([0., jnp.pi / 2, jnp.pi, 2 * jnp.pi])
        out = radians_to_degrees(x)
        expected = jnp.array([0., 90., 180., 360.])
        assert jnp.allclose(out, expected, atol=1e-4)

    def test_round_trip(self):
        x = jnp.linspace(-360., 360., 50)
        assert jnp.allclose(radians_to_degrees(degrees_to_radians(x)), x, atol=1e-5)

    def test_output_shape_preserved(self):
        x = jnp.ones((3, 4))
        assert degrees_to_radians(x).shape == (3, 4)
        assert radians_to_degrees(x).shape == (3, 4)

    def test_scalar_input(self):
        assert jnp.allclose(degrees_to_radians(jnp.array(180.)), jnp.pi, atol=1e-6)


# ---------------------------------------------------------------------------
# latlon_deg_to_rad / latlon_rad_to_deg
# ---------------------------------------------------------------------------

class TestLatlonConversion:

    def test_deg_to_rad_known_values(self):
        lat_r, lon_r = latlon_deg_to_rad(jnp.array([0., 45.]),
                                          jnp.array([90., 180.]))
        assert jnp.allclose(lat_r, jnp.array([0., jnp.pi / 4]), atol=1e-6)
        assert jnp.allclose(lon_r, jnp.array([jnp.pi / 2, jnp.pi]), atol=1e-6)

    def test_rad_to_deg_known_values(self):
        lat_d, lon_d = latlon_rad_to_deg(jnp.array([0., jnp.pi / 4]),
                                          jnp.array([jnp.pi / 2, jnp.pi]))
        assert jnp.allclose(lat_d, jnp.array([0., 45.]), atol=1e-4)
        assert jnp.allclose(lon_d, jnp.array([90., 180.]), atol=1e-4)

    def test_round_trip(self):
        lat = jnp.linspace(-90., 90., 20)
        lon = jnp.linspace(-180., 180., 20)
        lat_r, lon_r = latlon_deg_to_rad(lat, lon)
        lat_d, lon_d = latlon_rad_to_deg(lat_r, lon_r)
        assert jnp.allclose(lat_d, lat, atol=1e-4)
        assert jnp.allclose(lon_d, lon, atol=1e-4)

    def test_returns_tuple_of_two(self):
        result = latlon_deg_to_rad(jnp.array([10.]), jnp.array([20.]))
        assert len(result) == 2

    def test_output_shapes_match_input(self):
        lat = jnp.ones((5,))
        lon = jnp.ones((5,))
        lat_r, lon_r = latlon_deg_to_rad(lat, lon)
        assert lat_r.shape == lat.shape
        assert lon_r.shape == lon.shape


# ---------------------------------------------------------------------------
# spherical_to_cartesian / cartesian_to_spherical
# ---------------------------------------------------------------------------

class TestSphericalCartesian:

    def test_equator_prime_meridian(self):
        xyz = spherical_to_cartesian(jnp.array([0.]), jnp.array([0.]))
        assert jnp.allclose(xyz, jnp.array([[1., 0., 0.]]), atol=1e-6)

    def test_north_pole(self):
        xyz = spherical_to_cartesian(
            jnp.array([jnp.pi / 2]), jnp.array([0.])
        )
        assert jnp.allclose(xyz, jnp.array([[0., 0., 1.]]), atol=1e-6)

    def test_south_pole(self):
        xyz = spherical_to_cartesian(
            jnp.array([-jnp.pi / 2]), jnp.array([0.])
        )
        assert jnp.allclose(xyz, jnp.array([[0., 0., -1.]]), atol=1e-6)

    def test_output_is_unit_vector(self):
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 10)
        lon = jnp.linspace(-jnp.pi, jnp.pi, 10)
        xyz = spherical_to_cartesian(lat, lon)
        norms = jnp.linalg.norm(xyz, axis=-1)
        assert jnp.allclose(norms, jnp.ones(10), atol=1e-6)

    def test_output_shape(self):
        lat = jnp.zeros((8,))
        lon = jnp.zeros((8,))
        assert spherical_to_cartesian(lat, lon).shape == (8, 3)

    def test_round_trip(self):
        lat = jnp.array([0.1, 0.5, -0.3, -1.2, 1.0])
        lon = jnp.array([1.2, -0.7, 2.5, 0.0, -1.5])
        xyz = spherical_to_cartesian(lat, lon)
        lat2, lon2 = cartesian_to_spherical(xyz)
        assert jnp.allclose(lat, lat2, atol=1e-6)
        assert jnp.allclose(lon, lon2, atol=1e-6)

    def test_cartesian_to_spherical_ellipsis_indexing(self):
        # shape (2, 3, 3) -- arbitrary leading dims
        xyz = jnp.ones((2, 3, 3))
        xyz = xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
        lat, lon = cartesian_to_spherical(xyz)
        assert lat.shape == (2, 3)
        assert lon.shape == (2, 3)

    def test_cartesian_to_spherical_clips_z(self):
        # z slightly outside [-1, 1] due to floating point
        xyz = jnp.array([[0., 0., 1.0000001]])
        lat, _ = cartesian_to_spherical(xyz)
        assert jnp.isfinite(lat).all()

    def test_lon_range(self):
        lat = jnp.zeros((100,))
        lon = jnp.linspace(-jnp.pi, jnp.pi, 100)
        xyz = spherical_to_cartesian(lat, lon)
        _, lon2 = cartesian_to_spherical(xyz)
        assert jnp.all(lon2 >= -jnp.pi - 1e-5)
        assert jnp.all(lon2 <= jnp.pi + 1e-5)


# ---------------------------------------------------------------------------
# minmax_norm
# ---------------------------------------------------------------------------

class TestMinmaxNorm:

    def test_01_known_values(self):
        x = jnp.array([0., 5., 10.])
        out = minmax_norm(x, 0., 10., mode="01")
        assert jnp.allclose(out, jnp.array([0., 0.5, 1.]), atol=1e-6)

    def test_neg11_known_values(self):
        x = jnp.array([0., 5., 10.])
        out = minmax_norm(x, 0., 10., mode="-11")
        assert jnp.allclose(out, jnp.array([-1., 0., 1.]), atol=1e-6)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            minmax_norm(jnp.array([1., 2.]), 0., 10., mode="bad")

    def test_per_column_bounds(self):
        x = jnp.array([[0., 0.], [5., 10.], [10., 20.]])
        x_min = jnp.array([0., 0.])
        x_max = jnp.array([10., 20.])
        out = minmax_norm(x, x_min, x_max, mode="01")
        assert jnp.allclose(out[:, 0], jnp.array([0., 0.5, 1.]), atol=1e-5)
        assert jnp.allclose(out[:, 1], jnp.array([0., 0.5, 1.]), atol=1e-5)

    def test_equal_min_max_no_nan(self):
        x = jnp.array([5., 5., 5.])
        out = minmax_norm(x, 5., 5., mode="01")
        assert jnp.isfinite(out).all()

    def test_output_shape_preserved(self):
        x = jnp.ones((4, 3))
        out = minmax_norm(x, 0., 2.)
        assert out.shape == (4, 3)

    def test_nd_array(self):
        x = jnp.ones((2, 3, 4))
        out = minmax_norm(x, 0., 2., mode="-11")
        assert out.shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# standardise
# ---------------------------------------------------------------------------

class TestStandardise:

    def test_zero_mean_unit_std(self):
        x = jnp.array([1., 2., 3., 4., 5.])
        out = standardise(x)
        assert jnp.abs(out.mean()) < 1e-5
        assert jnp.abs(out.std() - 1.0) < 1e-4

    def test_precomputed_mean_std(self):
        x = jnp.array([1., 2., 3., 4., 5.])
        out = standardise(x, mean=3.0, std=1.0)
        assert jnp.allclose(out, jnp.array([-2., -1., 0., 1., 2.]), atol=1e-5)

    def test_per_column_axis0(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (100, 4))
        out = standardise(x, axis=0)
        assert out.shape == (100, 4)
        col_means = out.mean(axis=0)
        col_stds = out.std(axis=0)
        assert jnp.allclose(col_means, jnp.zeros(4), atol=1e-5)
        assert jnp.allclose(col_stds, jnp.ones(4), atol=1e-4)

    def test_train_test_consistency(self):
        key = jax.random.PRNGKey(1)
        train = jax.random.normal(key, (50, 3)) * 2 + 5
        test = jax.random.normal(key, (20, 3)) * 2 + 5
        train_mean = train.mean(axis=0, keepdims=True)
        train_std = train.std(axis=0, keepdims=True)
        train_norm = standardise(train, train_mean, train_std)
        test_norm = standardise(test, train_mean, train_std)
        assert train_norm.shape == train.shape
        assert test_norm.shape == test.shape

    def test_constant_array_no_nan(self):
        x = jnp.ones((10,)) * 3.0
        out = standardise(x)
        assert jnp.isfinite(out).all()

    def test_output_shape_preserved(self):
        x = jnp.ones((5, 6))
        assert standardise(x).shape == (5, 6)

    def test_nd_array_axis1(self):
        x = jnp.ones((4, 8, 3))
        out = standardise(x, axis=1)
        assert out.shape == (4, 8, 3)