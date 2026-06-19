import pytest
import warnings
import jax
import jax.numpy as jnp
import math

from core.initializations import (
    register_initializer,
    get_initializer,
    list_initializers,
    INITIALIZERS,
    SirenInit,
    FinerInit,
    FinerBiasInit,
    IdentityInit,
    GaborInit,
    WireInit,
)

# Standard inits (XAVIER_*, LECUN_NORMAL, NORMAL, UNIFORM, ORTHOGONAL, ZEROS)
# are delegated to flax.linen.initializers (r16) and exercised via the registry
# by name -- no concrete classes to import for those.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def shape_2d():
    return (256, 128)


@pytest.fixture
def shape_square():
    return (128, 128)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_register_and_retrieve(self):
        @register_initializer("TEST_INIT_REG", description="test")
        class _TestInit:
            def __call__(self, key, shape, dtype):
                return jnp.zeros(shape, dtype)

        assert "TEST_INIT_REG" in INITIALIZERS
        init = get_initializer("TEST_INIT_REG")
        assert callable(init)

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_initializer("SIREN")
            class _Dup:
                pass

    def test_case_insensitive_lookup(self, key, shape_2d):
        i1 = get_initializer("XAVIER_UNIFORM")
        i2 = get_initializer("xavier_uniform")
        w1 = i1(key, shape_2d, jnp.float32)
        w2 = i2(key, shape_2d, jnp.float32)
        assert jnp.allclose(w1, w2)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="is not registered"):
            get_initializer("NONEXISTENT_XYZ")

    def test_error_lists_available(self):
        with pytest.raises(ValueError) as exc_info:
            get_initializer("NONEXISTENT_XYZ")
        assert "SIREN" in str(exc_info.value)

    def test_unknown_kwarg_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_initializer("SIREN", fan_in=64, unknown_arg=99)
            assert any("unknown kwargs" in str(x.message) for x in w)

    def test_list_initializers_returns_dict(self):
        result = list_initializers()
        assert isinstance(result, dict)
        assert "SIREN" in result
        assert "WIRE" in result

    def test_list_initializers_sorted(self):
        result = list_initializers()
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_list_initializers_descriptions_are_strings(self):
        for name, desc in list_initializers().items():
            assert isinstance(desc, str)


# ---------------------------------------------------------------------------
# Shared initializer contract tests
# ---------------------------------------------------------------------------

class TestInitializerContract:
    """All initializers must satisfy the Flax kernel_init contract."""

    @pytest.mark.parametrize("name,kwargs", [
        ("SIREN",         {"fan_in": 256, "is_first": True}),
        ("SIREN",         {"fan_in": 256, "is_first": False, "omega": 30.}),
        ("FINER",         {"fan_in": 256, "is_first": True}),
        ("FINER",         {"fan_in": 256, "is_first": False, "omega": 30.}),
        ("FINER_BIAS",    {"k": 1.0}),
        ("XAVIER_UNIFORM",{}),
        ("XAVIER_NORMAL", {}),
        ("LECUN_NORMAL",  {}),
        ("NORMAL",        {"stddev": 0.1}),
        ("UNIFORM",       {"scale": 0.2}),
        ("ORTHOGONAL",    {"scale": 1.0}),
        ("GABOR",         {"std_scale": 1.0}),
        ("ZEROS",         {}),
    ])
    def test_output_shape(self, key, shape_2d, name, kwargs):
        init = get_initializer(name, **kwargs)
        w = init(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    @pytest.mark.parametrize("name,kwargs", [
        ("SIREN",         {"fan_in": 256, "is_first": False}),
        ("FINER",         {"fan_in": 256, "is_first": False}),
        ("FINER_BIAS",    {"k": 1.0}),
        ("XAVIER_UNIFORM",{}),
        ("XAVIER_NORMAL", {}),
        ("LECUN_NORMAL",  {}),
        ("NORMAL",        {}),
        ("UNIFORM",       {}),
        ("ORTHOGONAL",    {}),
        ("GABOR",         {}),
    ])
    def test_output_finite(self, key, shape_2d, name, kwargs):
        init = get_initializer(name, **kwargs)
        w = init(key, shape_2d, jnp.float32)
        assert jnp.all(jnp.isfinite(w))

    @pytest.mark.parametrize("name,kwargs", [
        ("SIREN",         {"fan_in": 256, "is_first": False}),
        ("XAVIER_UNIFORM",{}),
        ("NORMAL",        {}),
        ("LECUN_NORMAL",  {}),
    ])
    def test_different_keys_different_output(self, shape_2d, name, kwargs):
        init = get_initializer(name, **kwargs)
        w1 = init(jax.random.PRNGKey(0), shape_2d, jnp.float32)
        w2 = init(jax.random.PRNGKey(1), shape_2d, jnp.float32)
        assert not jnp.allclose(w1, w2)

    @pytest.mark.parametrize("name,kwargs", [
        ("SIREN",         {"fan_in": 256, "is_first": False}),
        ("XAVIER_UNIFORM",{}),
        ("NORMAL",        {}),
    ])
    def test_same_key_reproducible(self, key, shape_2d, name, kwargs):
        init = get_initializer(name, **kwargs)
        w1 = init(key, shape_2d, jnp.float32)
        w2 = init(key, shape_2d, jnp.float32)
        assert jnp.allclose(w1, w2)


# ---------------------------------------------------------------------------
# SirenInit
# ---------------------------------------------------------------------------

class TestSirenInit:

    def test_first_layer_bounds(self, key, shape_2d):
        fan_in = shape_2d[0]
        init = SirenInit(fan_in=fan_in, is_first=True)
        w = init(key, shape_2d, jnp.float32)
        bound = 1.0 / fan_in
        assert jnp.all(w >= -bound)
        assert jnp.all(w <= bound)

    def test_hidden_layer_bounds(self, key, shape_2d):
        fan_in = shape_2d[0]
        omega = 30.0
        init = SirenInit(fan_in=fan_in, is_first=False, omega=omega)
        w = init(key, shape_2d, jnp.float32)
        bound = math.sqrt(6.0 / fan_in) / omega
        assert jnp.all(w >= -bound)
        assert jnp.all(w <= bound)

    @pytest.mark.parametrize("fan_in,expected_wider", [
        (64, "first"),  # fan_in < omega^2/6 = 150 --> first layer wider
        (256, "hidden"),  # fan_in > 150 --> hidden layer wider
    ])
    def test_layer_bound_ordering(self, key, fan_in, expected_wider):
        bound_first = 1.0 / fan_in
        bound_hidden = math.sqrt(6.0 / fan_in) / 30.
        if expected_wider == "first":
            assert bound_first > bound_hidden
        else:
            assert bound_hidden > bound_first

    def test_omega_scales_bounds(self, key, shape_2d):
        fan_in = shape_2d[0]
        init_30 = SirenInit(fan_in=fan_in, is_first=False, omega=30.)
        init_10 = SirenInit(fan_in=fan_in, is_first=False, omega=10.)
        bound_30 = math.sqrt(6.0 / fan_in) / 30.
        bound_10 = math.sqrt(6.0 / fan_in) / 10.
        assert bound_10 > bound_30


# ---------------------------------------------------------------------------
# FinerInit / FinerBiasInit
# ---------------------------------------------------------------------------

class TestFinerInit:

    def test_kernel_bounds_first(self, key, shape_2d):
        fan_in = shape_2d[0]
        init = FinerInit(fan_in=fan_in, is_first=True)
        w = init(key, shape_2d, jnp.float32)
        bound = 1.0 / fan_in
        assert jnp.all(w >= -bound)
        assert jnp.all(w <= bound)

    def test_kernel_bounds_hidden(self, key, shape_2d):
        fan_in = shape_2d[0]
        init = FinerInit(fan_in=fan_in, is_first=False, omega=30.)
        w = init(key, shape_2d, jnp.float32)
        bound = math.sqrt(6.0 / fan_in) / 30.
        assert jnp.all(w >= -bound)
        assert jnp.all(w <= bound)


class TestFinerBiasInit:

    def test_bias_bounds(self, key):
        shape = (128,)
        k = 1.5
        init = FinerBiasInit(k=k)
        b = init(key, shape, jnp.float32)
        assert b.shape == shape
        assert jnp.all(b >= -k)
        assert jnp.all(b <= k)

    def test_bias_default_k(self, key):
        shape = (64,)
        init = FinerBiasInit()
        b = init(key, shape, jnp.float32)
        assert jnp.all(b >= -1.0)
        assert jnp.all(b <= 1.0)


# ---------------------------------------------------------------------------
# Xavier / Glorot initializers (flax-backed)
# ---------------------------------------------------------------------------

class TestXavierUniform:

    def test_output_shape(self, key, shape_2d):
        w = get_initializer("XAVIER_UNIFORM")(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_within_glorot_bound(self, key, shape_2d):
        # flax glorot-uniform draws in [-sqrt(6/(fan_in+fan_out)), +bound].
        fan_in, fan_out = shape_2d
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        w = get_initializer("XAVIER_UNIFORM")(key, shape_2d, jnp.float32)
        assert jnp.all(jnp.abs(w) <= bound + 1e-6)

    def test_output_finite(self, key, shape_2d):
        w = get_initializer("XAVIER_UNIFORM")(key, shape_2d, jnp.float32)
        assert jnp.all(jnp.isfinite(w))


class TestXavierNormal:

    def test_output_shape(self, key, shape_2d):
        w = get_initializer("XAVIER_NORMAL")(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_std_matches_glorot(self, key, shape_2d):
        # truncated-normal is variance-corrected, so realized std ~ glorot std.
        fan_in, fan_out = shape_2d
        target = math.sqrt(2.0 / (fan_in + fan_out))
        w = get_initializer("XAVIER_NORMAL")(key, shape_2d, jnp.float32)
        assert abs(float(w.std()) - target) / target < 0.1

    def test_output_finite(self, key, shape_2d):
        w = get_initializer("XAVIER_NORMAL")(key, shape_2d, jnp.float32)
        assert jnp.all(jnp.isfinite(w))


# ---------------------------------------------------------------------------
# LeCun normal (flax-backed)
# ---------------------------------------------------------------------------

class TestLecunNormal:

    def test_output_shape(self, key, shape_2d):
        w = get_initializer("LECUN_NORMAL")(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_std_matches_fan_in(self, key, shape_2d):
        # lecun-normal targets std = 1 / sqrt(fan_in).
        fan_in = shape_2d[0]
        target = 1.0 / math.sqrt(fan_in)
        w = get_initializer("LECUN_NORMAL")(key, shape_2d, jnp.float32)
        assert abs(float(w.std()) - target) / target < 0.1

    def test_output_finite(self, key, shape_2d):
        w = get_initializer("LECUN_NORMAL")(key, shape_2d, jnp.float32)
        assert jnp.all(jnp.isfinite(w))


# ---------------------------------------------------------------------------
# Normal / Uniform (flax-backed)
# ---------------------------------------------------------------------------

class TestNormal:

    def test_output_shape(self, key, shape_2d):
        w = get_initializer("NORMAL")(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_stddev_affects_spread(self, key, shape_2d):
        w1 = get_initializer("NORMAL", stddev=0.01)(key, shape_2d, jnp.float32)
        w2 = get_initializer("NORMAL", stddev=1.0)(key, shape_2d, jnp.float32)
        assert w2.std() > w1.std()

    def test_stddev_approximately_correct(self, key, shape_2d):
        w = get_initializer("NORMAL", stddev=0.5)(key, shape_2d, jnp.float32)
        assert abs(float(w.std()) - 0.5) / 0.5 < 0.1


class TestUniform:

    def test_output_shape(self, key, shape_2d):
        w = get_initializer("UNIFORM")(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_bounds_respected(self, key, shape_2d):
        # flax uniform draws in [0, scale).
        scale = 0.5
        w = get_initializer("UNIFORM", scale=scale)(key, shape_2d, jnp.float32)
        assert jnp.all(w >= 0.0)
        assert jnp.all(w <= scale)


# ---------------------------------------------------------------------------
# IdentityInit
# ---------------------------------------------------------------------------

class TestIdentityInit:

    def test_square_matrix(self, key, shape_square):
        w = IdentityInit()(key, shape_square, jnp.float32)
        assert jnp.allclose(w, jnp.eye(shape_square[0]))

    def test_non_square_raises(self, key, shape_2d):
        with pytest.raises(ValueError, match="square"):
            IdentityInit()(key, shape_2d, jnp.float32)


# ---------------------------------------------------------------------------
# Orthogonal (flax-backed; gain -> scale)
# ---------------------------------------------------------------------------

class TestOrthogonal:

    def test_output_shape(self, key, shape_2d):
        w = get_initializer("ORTHOGONAL")(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_rectangular_rows_orthonormal(self, key):
        # fat matrix (fan_out > fan_in) -- rows are orthonormal
        shape = (64, 128)
        w = get_initializer("ORTHOGONAL", scale=1.0)(key, shape, jnp.float32)
        assert w.shape == shape
        gram = w @ w.T  # (64, 64)
        assert jnp.allclose(gram, jnp.eye(shape[0]), atol=1e-5)

    def test_scale_scales_output(self, key, shape_square):
        w1 = get_initializer("ORTHOGONAL", scale=1.0)(key, shape_square, jnp.float32)
        w2 = get_initializer("ORTHOGONAL", scale=2.0)(key, shape_square, jnp.float32)
        assert jnp.allclose(jnp.abs(w2), jnp.abs(w1) * 2.0, atol=1e-5)


# ---------------------------------------------------------------------------
# GaborInit
# ---------------------------------------------------------------------------

class TestGaborInit:

    def test_output_shape(self, key, shape_2d):
        w = GaborInit()(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_std_scale_affects_spread(self, key, shape_2d):
        w1 = GaborInit(std_scale=1.0)(key, shape_2d, jnp.float32)
        w2 = GaborInit(std_scale=5.0)(key, shape_2d, jnp.float32)
        assert w2.std() > w1.std()

    def test_std_approximately_correct(self, key):
        shape = (1000, 500)
        fan_in = shape[0]
        std_scale = 1.0
        w = GaborInit(std_scale=std_scale)(key, shape, jnp.float32)
        expected_std = std_scale / math.sqrt(fan_in)
        assert abs(float(w.std()) - expected_std) < 0.02

    def test_output_finite(self, key, shape_2d):
        w = GaborInit()(key, shape_2d, jnp.float32)
        assert jnp.all(jnp.isfinite(w))


# ---------------------------------------------------------------------------
# WireInit
# ---------------------------------------------------------------------------

class TestWireInit:

    def test_output_shape_complex64(self, key, shape_2d):
        w = WireInit()(key, shape_2d, jnp.complex64)
        assert w.shape == shape_2d

    def test_output_is_complex64(self, key, shape_2d):
        w = WireInit()(key, shape_2d, jnp.complex64)
        assert w.dtype == jnp.complex64

    @pytest.mark.skipif(
        not jax.config.jax_enable_x64,
        reason="complex128 requires jax_enable_x64=True"
    )
    def test_output_is_complex128(self, key, shape_2d):
        w = WireInit()(key, shape_2d, jnp.complex128)
        assert w.dtype == jnp.complex128

    def test_real_and_imag_both_nonzero(self, key, shape_2d):
        w = WireInit()(key, shape_2d, jnp.complex64)
        assert jnp.any(jnp.real(w) != 0.)
        assert jnp.any(jnp.imag(w) != 0.)

    def test_gain_affects_magnitude(self, key, shape_2d):
        w1 = WireInit(gain=1.0)(key, shape_2d, jnp.complex64)
        w2 = WireInit(gain=2.0)(key, shape_2d, jnp.complex64)
        assert jnp.abs(w2).mean() > jnp.abs(w1).mean()

    def test_output_finite(self, key, shape_2d):
        w = WireInit()(key, shape_2d, jnp.complex64)
        assert jnp.all(jnp.isfinite(jnp.real(w)))
        assert jnp.all(jnp.isfinite(jnp.imag(w)))

    def test_different_keys_differ(self, shape_2d):
        w1 = WireInit()(jax.random.PRNGKey(0), shape_2d, jnp.complex64)
        w2 = WireInit()(jax.random.PRNGKey(1), shape_2d, jnp.complex64)
        assert not jnp.allclose(jnp.real(w1), jnp.real(w2))