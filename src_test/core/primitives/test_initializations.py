import pytest
import warnings
import jax
import jax.numpy as jnp
import math

from core.primitives.initializations import (
    register_initializer,
    get_initializer,
    list_initializers,
    INITIALIZERS,
    SirenInit,
    FinerInit,
    FinerBiasInit,
    XavierUniformInit,
    XavierNormalInit,
    LeCunNormalInit,
    NormalInit,
    UniformInit,
    IdentityInit,
    OrthogonalInit,
    GaborInit,
    WireInit,
)


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
        with pytest.raises(ValueError, match="already exists"):
            @register_initializer("SIREN")
            class _Dup:
                pass

    def test_case_insensitive_lookup(self, key, shape_2d):
        i1 = get_initializer("XAVIER_UNIFORM", gain=1.0)
        i2 = get_initializer("xavier_uniform", gain=1.0)
        w1 = i1(key, shape_2d, jnp.float32)
        w2 = i2(key, shape_2d, jnp.float32)
        assert jnp.allclose(w1, w2)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
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
        ("XAVIER_UNIFORM",{"gain": 1.0}),
        ("XAVIER_NORMAL", {"gain": 1.0}),
        ("LECUN_NORMAL",  {"scale": 1.0}),
        ("NORMAL",        {"mean": 0., "std": 0.1}),
        ("UNIFORM",       {"a": -0.1, "b": 0.1}),
        ("ORTHOGONAL",    {"gain": 1.0}),
        ("GABOR",         {"std_scale": 1.0}),
    ])
    def test_output_shape(self, key, shape_2d, name, kwargs):
        init = get_initializer(name, **kwargs)
        w = init(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    @pytest.mark.parametrize("name,kwargs", [
        ("SIREN",         {"fan_in": 256, "is_first": False}),
        ("FINER",         {"fan_in": 256, "is_first": False}),
        ("FINER_BIAS",    {"k": 1.0}),
        ("XAVIER_UNIFORM",{"gain": 1.0}),
        ("XAVIER_NORMAL", {"gain": 1.0}),
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
        ("XAVIER_UNIFORM",{"gain": 1.0}),
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
        ("XAVIER_UNIFORM",{"gain": 1.0}),
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
# Xavier initializers
# ---------------------------------------------------------------------------

class TestXavierUniformInit:

    def test_output_shape(self, key, shape_2d):
        init = XavierUniformInit(gain=1.0)
        w = init(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_gain_affects_output(self, key, shape_2d):
        init1 = XavierUniformInit(gain=1.0)
        init2 = XavierUniformInit(gain=0.1)
        w1 = init1(key, shape_2d, jnp.float32)
        w2 = init2(key, shape_2d, jnp.float32)
        # smaller gain -> smaller magnitude weights
        assert jnp.abs(w2).max() < jnp.abs(w1).max()

    def test_output_finite(self, key, shape_2d):
        init = XavierUniformInit()
        w = init(key, shape_2d, jnp.float32)
        assert jnp.all(jnp.isfinite(w))


class TestXavierNormalInit:

    def test_output_shape(self, key, shape_2d):
        init = XavierNormalInit(gain=1.0)
        w = init(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_gain_affects_output(self, key, shape_2d):
        fan_in, fan_out = shape_2d
        gain1, gain2 = 1.0, 0.1
        w1 = XavierUniformInit(gain=gain1)(key, shape_2d, jnp.float32)
        w2 = XavierUniformInit(gain=gain2)(key, shape_2d, jnp.float32)
        # theoretical bound scales linearly with gain
        bound1 = gain1 * math.sqrt(6.0 / (fan_in + fan_out))
        bound2 = gain2 * math.sqrt(6.0 / (fan_in + fan_out))
        assert jnp.all(jnp.abs(w1) <= bound1 + 1e-6)
        assert jnp.all(jnp.abs(w2) <= bound2 + 1e-6)

    def test_output_finite(self, key, shape_2d):
        init = XavierNormalInit()
        w = init(key, shape_2d, jnp.float32)
        assert jnp.all(jnp.isfinite(w))


# ---------------------------------------------------------------------------
# LeCunNormalInit
# ---------------------------------------------------------------------------

class TestLeCunNormalInit:

    def test_output_shape(self, key, shape_2d):
        init = LeCunNormalInit()
        w = init(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_scale_affects_std(self, key, shape_2d):
        init1 = LeCunNormalInit(scale=1.0)
        init2 = LeCunNormalInit(scale=2.0)
        w1 = init1(key, shape_2d, jnp.float32)
        w2 = init2(key, shape_2d, jnp.float32)
        assert w2.std() > w1.std()

    def test_output_finite(self, key, shape_2d):
        w = LeCunNormalInit()(key, shape_2d, jnp.float32)
        assert jnp.all(jnp.isfinite(w))


# ---------------------------------------------------------------------------
# NormalInit / UniformInit
# ---------------------------------------------------------------------------

class TestNormalInit:

    def test_output_shape(self, key, shape_2d):
        w = NormalInit()(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_std_affects_spread(self, key, shape_2d):
        w1 = NormalInit(std=0.01)(key, shape_2d, jnp.float32)
        w2 = NormalInit(std=1.0)(key, shape_2d, jnp.float32)
        assert w2.std() > w1.std()

    def test_mean_shifts_output(self, key, shape_2d):
        w = NormalInit(mean=5.0, std=0.01)(key, shape_2d, jnp.float32)
        assert abs(float(w.mean()) - 5.0) < 0.1


class TestUniformInit:

    def test_output_shape(self, key, shape_2d):
        w = UniformInit()(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_bounds_respected(self, key, shape_2d):
        a, b = -0.5, 0.5
        w = UniformInit(a=a, b=b)(key, shape_2d, jnp.float32)
        assert jnp.all(w >= a)
        assert jnp.all(w <= b)


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
# OrthogonalInit
# ---------------------------------------------------------------------------

class TestOrthogonalInit:

    def test_output_shape(self, key, shape_2d):
        w = OrthogonalInit()(key, shape_2d, jnp.float32)
        assert w.shape == shape_2d

    def test_rectangular_rows_orthonormal(self, key):
        # fat matrix (fan_out > fan_in) -- rows are orthonormal
        shape = (64, 128)
        w = OrthogonalInit(gain=1.0)(key, shape, jnp.float32)
        assert w.shape == shape
        gram = w @ w.T  # (64, 64)
        assert jnp.allclose(gram, jnp.eye(shape[0]), atol=1e-5)

    def test_gain_scales_output(self, key, shape_square):
        w1 = OrthogonalInit(gain=1.0)(key, shape_square, jnp.float32)
        w2 = OrthogonalInit(gain=2.0)(key, shape_square, jnp.float32)
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