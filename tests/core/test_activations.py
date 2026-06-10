import pytest
import warnings
import jax
import jax.numpy as jnp
import numpy as np

from core.activations import (
    register_activation,
    get_activation,
    list_activations,
    ACTIVATIONS,
    ReLU,
    LeakyReLU,
    SiLU,
    Sigmoid,
    Tanh,
    GELU,
    ELU,
    SELU,
    Softplus,
    Identity,
    SineActivation,
    SineFinerActivation,
    GaussianActivation,
    GaussianFinerActivation,
    WireActivation,
    WireRealActivation,
    WireFinerActivation,
    WireFinerRealActivation,
    HoscActivation,
    HoscFinerActivation,
    SincActivation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def x_real():
    return jnp.linspace(-2., 2., 20)


@pytest.fixture
def x_scalar():
    return jnp.array(1.0)


@pytest.fixture
def x_2d():
    return jnp.ones((4, 8))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_register_and_retrieve(self):
        @register_activation("TEST_REGISTRY_ACT", description="test")
        class _TestAct:
            def __call__(self, x: jax.Array) -> jax.Array:
                return x
        assert "TEST_REGISTRY_ACT" in ACTIVATIONS
        act = get_activation("TEST_REGISTRY_ACT")
        assert callable(act)

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            @register_activation("RELU")
            class _Dup:
                def __call__(self, x):
                    return x

    def test_case_insensitive_lookup(self, x_real):
        act_upper = get_activation("SINE", omega=10.)
        act_lower = get_activation("sine", omega=10.)
        assert jnp.allclose(act_upper(x_real), act_lower(x_real))

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            get_activation("NONEXISTENT_XYZ")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError) as exc_info:
            get_activation("NONEXISTENT_XYZ")
        assert "RELU" in str(exc_info.value)

    def test_unknown_kwarg_warns(self):
        with pytest.warns(UserWarning, match="unknown kwargs"):
            get_activation("RELU", omega=30)

    def test_valid_kwarg_no_warning(self, x_real):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            act = get_activation("SINE", omega=10.)
            act(x_real)

    def test_list_activations_returns_dict(self):
        result = list_activations()
        assert isinstance(result, dict)
        assert "RELU" in result
        assert "SINE" in result

    def test_list_activations_sorted(self):
        result = list_activations()
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_list_activations_descriptions_are_strings(self):
        for name, desc in list_activations().items():
            assert isinstance(desc, str)


# ---------------------------------------------------------------------------
# Built-in wrappers -- output shape and dtype
# ---------------------------------------------------------------------------

class TestBuiltinActivations:

    @pytest.mark.parametrize("cls,kwargs", [
        (ReLU, {}),
        (LeakyReLU, {"negative_slope": 0.01}),
        (SiLU, {}),
        (Sigmoid, {}),
        (Tanh, {}),
        (GELU, {"approximate": True}),
        (ELU, {"alpha": 1.0}),
        (SELU, {}),
        (Softplus, {}),
        (Identity, {}),
    ])
    def test_output_shape_matches_input(self, cls, kwargs, x_real):
        act = cls(**kwargs)
        out = act(x_real)
        assert out.shape == x_real.shape

    @pytest.mark.parametrize("cls,kwargs", [
        (ReLU, {}),
        (LeakyReLU, {}),
        (SiLU, {}),
        (Sigmoid, {}),
        (Tanh, {}),
        (GELU, {}),
        (ELU, {}),
        (SELU, {}),
        (Softplus, {}),
        (Identity, {}),
    ])
    def test_output_is_real(self, cls, kwargs, x_real):
        act = cls(**kwargs)
        out = act(x_real)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_relu_clips_negative(self, x_real):
        out = ReLU()(x_real)
        assert jnp.all(out >= 0.)

    def test_sigmoid_range(self, x_real):
        out = Sigmoid()(x_real)
        assert jnp.all(out > 0.) and jnp.all(out < 1.)

    def test_tanh_range(self, x_real):
        out = Tanh()(x_real)
        assert jnp.all(out >= -1.) and jnp.all(out <= 1.)

    def test_identity_passthrough(self, x_real):
        assert jnp.allclose(Identity()(x_real), x_real)

    def test_2d_input(self, x_2d):
        for cls in [ReLU, SiLU, Sigmoid, Tanh, Identity]:
            out = cls()(x_2d)
            assert out.shape == x_2d.shape


# ---------------------------------------------------------------------------
# Sine / FINER
# ---------------------------------------------------------------------------

class TestSineActivations:

    def test_sine_output_shape(self, x_real):
        assert SineActivation()(x_real).shape == x_real.shape

    def test_sine_known_value(self):
        x = jnp.array([0.])
        assert jnp.allclose(SineActivation(omega=1.)(x), jnp.array([0.]))

    def test_sine_omega_scales_frequency(self, x_real):
        out1 = SineActivation(omega=1.)(x_real)
        out30 = SineActivation(omega=30.)(x_real)
        assert not jnp.allclose(out1, out30)

    def test_sine_is_real(self, x_real):
        out = SineActivation()(x_real)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_finer_output_shape(self, x_real):
        assert SineFinerActivation()(x_real).shape == x_real.shape

    def test_finer_is_real(self, x_real):
        out = SineFinerActivation()(x_real)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_finer_differs_from_sine(self, x_real):
        sine = SineActivation(omega=30.)(x_real)
        finer = SineFinerActivation(omega=30.)(x_real)
        # FINER adapts frequency so outputs should differ away from x=0
        assert not jnp.allclose(sine, finer)

    def test_finer_alpha_effect(self):
        # at x=0, alpha=1 so FINER == SINE; away from 0 they diverge
        x_zero = jnp.array([0.])
        x_nonzero = jnp.array([1.])
        sine = SineActivation(omega=30.)
        finer = SineFinerActivation(omega=30.)
        assert jnp.allclose(sine(x_zero), finer(x_zero), atol=1e-5)
        assert not jnp.allclose(sine(x_nonzero), finer(x_nonzero))

    def test_sine_grad_real_input_real_grad(self):
        def f(x):
            return SineActivation(omega=30.)(x).sum()
        grad = jax.grad(f)(jnp.array(1.0))
        assert jnp.isfinite(grad)
        assert not jnp.issubdtype(grad.dtype, jnp.complexfloating)

    def test_finer_grad_real_input_real_grad(self):
        def f(x):
            return SineFinerActivation(omega=30.)(x).sum()
        grad = jax.grad(f)(jnp.array(1.0))
        assert jnp.isfinite(grad)
        assert not jnp.issubdtype(grad.dtype, jnp.complexfloating)


# ---------------------------------------------------------------------------
# Gaussian / Gaussian FINER
# ---------------------------------------------------------------------------

class TestGaussianActivations:

    def test_gaussian_output_shape(self, x_real):
        assert GaussianActivation()(x_real).shape == x_real.shape

    def test_gaussian_peak_at_zero(self):
        x = jnp.array([0.])
        assert jnp.allclose(GaussianActivation(sigma=10.)(x), jnp.array([1.]))

    def test_gaussian_is_real_positive(self, x_real):
        out = GaussianActivation()(x_real)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)
        assert jnp.all(out >= 0.)

    def test_gaussian_finer_output_shape(self, x_real):
        assert GaussianFinerActivation()(x_real).shape == x_real.shape

    def test_gaussian_finer_is_real_positive(self, x_real):
        out = GaussianFinerActivation()(x_real)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)
        assert jnp.all(out >= 0.)


# ---------------------------------------------------------------------------
# WIRE
# ---------------------------------------------------------------------------

class TestWireActivations:

    # --- WIRE (complex output) ---

    def test_wire_output_is_complex(self, x_real):
        out = WireActivation()(x_real)
        assert jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_wire_output_shape(self, x_real):
        out = WireActivation()(x_real)
        assert out.shape == x_real.shape

    def test_wire_real_input_complex_output_backprop(self):
        """
        Real input -> complex output -> take real part -> backprop to real input.
        Verifies that gradients flow through the complex intermediate correctly.
        """
        def f(x):
            out = WireActivation(omega_0=20., sigma_0=10.)(x)
            return jnp.real(out).sum()

        x = jnp.linspace(-1., 1., 10)
        grad = jax.grad(f)(x)
        assert grad.shape == x.shape
        assert jnp.all(jnp.isfinite(grad))
        assert not jnp.issubdtype(grad.dtype, jnp.complexfloating)

    def test_wire_complex_internally_real_grad_via_abs(self):
        """
        Real input -> complex WIRE -> abs (real output) -> backprop.
        Confirms grad flows through |complex| to real input.
        """
        def f(x):
            out = WireActivation(omega_0=20., sigma_0=10.)(x)
            return jnp.abs(out).sum()

        x = jnp.linspace(-1., 1., 10)
        grad = jax.grad(f)(x)
        assert grad.shape == x.shape
        assert jnp.all(jnp.isfinite(grad))
        assert not jnp.issubdtype(grad.dtype, jnp.complexfloating)

    def test_wire_complex_input_complex_output(self):
        """
        Complex input -> complex WIRE -> complex output.
        Verifies the activation works end-to-end with complex dtype input.
        """
        x_complex = jnp.array([1.+0.j, 0.+1.j, 1.+1.j])
        out = WireActivation(omega_0=20., sigma_0=10.)(x_complex)
        assert jnp.issubdtype(out.dtype, jnp.complexfloating)
        assert out.shape == x_complex.shape
        assert jnp.all(jnp.isfinite(jnp.real(out)))
        assert jnp.all(jnp.isfinite(jnp.imag(out)))

    # --- WIRE_REAL (real output via abs) ---

    def test_wire_real_output_is_real(self, x_real):
        out = WireRealActivation()(x_real)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_wire_real_output_is_nonnegative(self, x_real):
        out = WireRealActivation()(x_real)
        assert jnp.all(out >= 0.)

    def test_wire_real_output_shape(self, x_real):
        assert WireRealActivation()(x_real).shape == x_real.shape

    def test_wire_real_grad_real_input_real_grad(self):
        """Real input -> WIRE_REAL (abs internally) -> real output -> backprop."""
        def f(x):
            return WireRealActivation(omega_0=20., sigma_0=10.)(x).sum()

        x = jnp.linspace(-1., 1., 10)
        grad = jax.grad(f)(x)
        assert grad.shape == x.shape
        assert jnp.all(jnp.isfinite(grad))
        assert not jnp.issubdtype(grad.dtype, jnp.complexfloating)

    def test_wire_real_matches_abs_of_wire(self, x_real):
        """WIRE_REAL should equal |WIRE| pointwise."""
        wire_abs = jnp.abs(WireActivation()(x_real))
        wire_real = WireRealActivation()(x_real)
        assert jnp.allclose(wire_abs, wire_real, atol=1e-6)

    # --- WIRE_FINER (complex output) ---

    def test_wire_finer_output_is_complex(self, x_real):
        out = WireFinerActivation()(x_real)
        assert jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_wire_finer_output_shape(self, x_real):
        assert WireFinerActivation()(x_real).shape == x_real.shape

    def test_wire_finer_real_input_backprop_via_real_part(self):
        """Real input -> WIRE_FINER (complex) -> real part -> backprop."""
        def f(x):
            out = WireFinerActivation(omega_0=20., sigma_0=10., omega_finer=5.)(x)
            return jnp.real(out).sum()

        x = jnp.linspace(-1., 1., 10)
        grad = jax.grad(f)(x)
        assert grad.shape == x.shape
        assert jnp.all(jnp.isfinite(grad))
        assert not jnp.issubdtype(grad.dtype, jnp.complexfloating)

    def test_wire_finer_differs_from_wire(self, x_real):
        """FINER adaptive scaling should produce different outputs to standard WIRE."""
        wire = WireActivation(omega_0=20., sigma_0=10.)(x_real)
        finer = WireFinerActivation(omega_0=20., sigma_0=10.)(x_real)
        assert not jnp.allclose(jnp.abs(wire), jnp.abs(finer))

    def test_wire_finer_alpha_effect_at_zero(self):
        """At x=0, alpha=1 so WIRE_FINER and WIRE should agree."""
        x_zero = jnp.array([0.])
        wire = WireActivation(omega_0=20., sigma_0=10.)(x_zero)
        finer = WireFinerActivation(omega_0=20., sigma_0=10., omega_finer=20.)(x_zero)
        assert jnp.allclose(jnp.abs(wire), jnp.abs(finer), atol=1e-5)

    # --- WIRE_FINER_REAL ---

    def test_wire_finer_real_output_is_real(self, x_real):
        out = WireFinerRealActivation()(x_real)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_wire_finer_real_output_is_nonnegative(self, x_real):
        out = WireFinerRealActivation()(x_real)
        assert jnp.all(out >= 0.)

    def test_wire_finer_real_grad(self):
        def f(x):
            return WireFinerRealActivation()(x).sum()
        x = jnp.linspace(-1., 1., 10)
        grad = jax.grad(f)(x)
        assert jnp.all(jnp.isfinite(grad))
        assert not jnp.issubdtype(grad.dtype, jnp.complexfloating)

    def test_wire_finer_real_matches_abs_of_wire_finer(self, x_real):
        finer_abs = jnp.abs(WireFinerActivation()(x_real))
        finer_real = WireFinerRealActivation()(x_real)
        assert jnp.allclose(finer_abs, finer_real, atol=1e-6)


# ---------------------------------------------------------------------------
# HOSC / HOSC_FINER
# ---------------------------------------------------------------------------

class TestHoscActivations:

    def test_hosc_output_shape(self, x_real):
        assert HoscActivation()(x_real).shape == x_real.shape

    def test_hosc_is_real(self, x_real):
        out = HoscActivation()(x_real)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_hosc_range(self, x_real):
        out = HoscActivation()(x_real)
        assert jnp.all(out >= -1.) and jnp.all(out <= 1.)

    def test_hosc_finer_output_shape(self, x_real):
        assert HoscFinerActivation()(x_real).shape == x_real.shape

    def test_hosc_finer_is_real(self, x_real):
        out = HoscFinerActivation()(x_real)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_hosc_grad(self):
        def f(x):
            return HoscActivation(beta=10.)(x).sum()
        grad = jax.grad(f)(jnp.array(1.0))
        assert jnp.isfinite(grad)

    def test_hosc_finer_grad(self):
        def f(x):
            return HoscFinerActivation(beta=10., omega=30.)(x).sum()
        grad = jax.grad(f)(jnp.array(1.0))
        assert jnp.isfinite(grad)


# ---------------------------------------------------------------------------
# Sinc
# ---------------------------------------------------------------------------

class TestSincActivation:

    def test_sinc_output_shape(self, x_real):
        assert SincActivation()(x_real).shape == x_real.shape

    def test_sinc_is_real(self, x_real):
        out = SincActivation()(x_real)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_sinc_at_zero(self):
        # sinc(0) = 1 by definition (jnp.sinc uses normalised sinc)
        x = jnp.array([0.])
        out = SincActivation(omega=1.)(x)
        assert jnp.allclose(out, jnp.array([1.]), atol=1e-6)

    def test_sinc_grad(self):
        def f(x):
            return SincActivation(omega=30.)(x).sum()
        grad = jax.grad(f)(jnp.array(0.5))
        assert jnp.isfinite(grad)


# ---------------------------------------------------------------------------
# JIT compatibility across all activations
# ---------------------------------------------------------------------------

class TestJitCompatibility:

    @pytest.mark.parametrize("name,kwargs", [
        ("RELU", {}),
        ("LEAKY_RELU", {"negative_slope": 0.01}),
        ("SILU", {}),
        ("SIGMOID", {}),
        ("TANH", {}),
        ("GELU", {}),
        ("ELU", {}),
        ("SELU", {}),
        ("SOFTPLUS", {}),
        ("IDENTITY", {}),
        ("SINE", {"omega": 30.}),
        ("FINER", {"omega": 30.}),
        ("GAUSSIAN", {"sigma": 10.}),
        ("GAUSSIAN_FINER", {"sigma": 10., "omega": 30.}),
        ("WIRE_REAL", {"omega_0": 20., "sigma_0": 10.}),
        ("WIRE_FINER_REAL", {"omega_0": 20., "sigma_0": 10., "omega_finer": 5.}),
        ("HOSC", {"beta": 10.}),
        ("HOSC_FINER", {"beta": 10., "omega": 30.}),
        ("SINC", {"omega": 30.}),
    ])
    def test_jit_real_activations(self, name, kwargs, x_real):
        act = get_activation(name, **kwargs)
        jit_act = jax.jit(act)
        out = jit_act(x_real)
        assert out.shape == x_real.shape
        assert jnp.all(jnp.isfinite(jnp.real(out)))

    @pytest.mark.parametrize("name,kwargs", [
        ("WIRE", {"omega_0": 20., "sigma_0": 10.}),
        ("WIRE_FINER", {"omega_0": 20., "sigma_0": 10., "omega_finer": 5.}),
    ])
    def test_jit_complex_activations(self, name, kwargs, x_real):
        act = get_activation(name, **kwargs)
        jit_act = jax.jit(act)
        out = jit_act(x_real)
        assert out.shape == x_real.shape
        assert jnp.issubdtype(out.dtype, jnp.complexfloating)
        assert jnp.all(jnp.isfinite(jnp.real(out)))