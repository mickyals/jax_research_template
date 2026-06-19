import jax
import jax.numpy as jnp
import jax.nn as nn

from utils.registry import Registry

# Activation registry on the shared utils.registry.Registry (r16). Module-level
# register_activation / get_activation / list_activations are kept as thin
# aliases so existing call sites are unchanged. Registry.get filters kwargs via
# inspect.signature(cls), which for a plain class introspects __init__ (a class
# with no __init__ exposes no params, so unknown kwargs warn + drop as before).
ACTIVATIONS = Registry("Activation")
register_activation = ACTIVATIONS.register
get_activation      = ACTIVATIONS.get


def list_activations() -> dict[str, str]:
    """Sorted ``{name: description}`` of all registered activations."""
    return dict(sorted(ACTIVATIONS.describe().items()))


def _generate_alpha(x: jax.Array) -> jax.Array:
    """Compute |x| + 1 as a positive scaling factor for FINER-style activations.

    Always >= 1, so it adaptively increases the effective frequency for
    inputs with larger magnitude.
    """
    return jnp.abs(x) + 1


# ===================================================================
# Flax/JAX built-in wrappers
# ===================================================================

@register_activation("RELU", description="ReLU activation")
class ReLU:
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.relu(x)


@register_activation("LEAKY_RELU", description="Leaky ReLU activation")
class LeakyReLU:
    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.leaky_relu(x, self.negative_slope)


@register_activation("SILU", description="SiLU (swish) activation")
class SiLU:
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.silu(x)


@register_activation("SIGMOID", description="Sigmoid activation")
class Sigmoid:
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.sigmoid(x)


@register_activation("TANH", description="Tanh activation")
class Tanh:
    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.tanh(x)


@register_activation("GELU", description="Gaussian error linear unit activation")
class GELU:
    def __init__(self, approximate: bool = True):
        self.approximate = approximate

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.gelu(x, approximate=self.approximate)


@register_activation("ELU", description="Exponential linear unit activation")
class ELU:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.elu(x, alpha=self.alpha)


@register_activation("SELU", description="Scaled exponential linear unit activation")
class SELU:
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.selu(x)


@register_activation("SOFTPLUS", description="Softplus activation")
class Softplus:
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.softplus(x)


@register_activation("IDENTITY", description="Identity (no-op) activation")
class Identity:
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


# ===================================================================
# Sine activation (SIREN)
# ===================================================================

@register_activation("SINE", description="Sine activation (SIREN)")
class SineActivation:
    """Applies sin(omega * x).

    Parameters
    ----------
    omega : float
        Frequency parameter. Default 30.
    """
    def __init__(self, omega: float = 30.0):
        self.omega = omega

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.sin(self.omega * x)


@register_activation("FINER", description="FINER sine activation")
class SineFinerActivation:
    """Applies sin(omega * alpha(x) * x) where alpha(x) = |x| + 1.

    The adaptive scaling factor alpha(x) increases effective frequency
    for inputs with larger magnitude, improving representational capacity
    over standard SIREN.

    Parameters
    ----------
    omega : float
        Frequency parameter. Default 30.
    """
    def __init__(self, omega: float = 30.0):
        self.omega = omega

    def __call__(self, x: jax.Array) -> jax.Array:
        alpha = _generate_alpha(x)
        return jnp.sin(self.omega * alpha * x)


# ===================================================================
# Gaussian activation
# ===================================================================

@register_activation("GAUSSIAN", description="Gaussian activation")
class GaussianActivation:
    """Applies exp(-(sigma * x)^2).

    Parameters
    ----------
    sigma : float
        Width parameter. Default 10.
    """
    def __init__(self, sigma: float = 10.0):
        self.sigma = sigma

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.exp(-(self.sigma * x) ** 2)


@register_activation("GAUSSIAN_FINER", description="FINER Gaussian activation")
class GaussianFinerActivation:
    """Applies exp(-((sigma/omega) * sin(omega * alpha(x) * x))^2).

    Parameters
    ----------
    sigma : float
        Width parameter. Default 10.
    omega : float
        Frequency parameter. Default 30.
    """
    def __init__(self, sigma: float = 10.0, omega: float = 30.0):
        self.sigma = sigma
        self.omega = omega

    def __call__(self, x: jax.Array) -> jax.Array:
        alpha = _generate_alpha(x)
        finer = jnp.sin(self.omega * alpha * x)
        scaler = self.sigma / self.omega
        return jnp.exp(-(scaler * finer) ** 2)


# ===================================================================
# WIRE activation
# ===================================================================

@register_activation("WIRE", description="WIRE activation (complex Gabor wavelet)")
class WireActivation:
    """Applies exp(j * omega_0 * x) * exp(-(sigma_0 * |x|)^2).

    Returns a complex-valued array. This activation is intended for use
    in networks explicitly designed for complex arithmetic. Passing the
    output to a standard real-valued Dense layer will raise a dtype error.
    Use ``WIRE_REAL`` for a real-valued alternative that takes the
    magnitude of the complex output.

    Parameters
    ----------
    omega_0 : float
        Frequency parameter. Default 20.
    sigma_0 : float
        Width parameter. Default 10.
    """
    def __init__(self, omega_0: float = 20.0, sigma_0: float = 10.0):
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0

    def __call__(self, x: jax.Array) -> jax.Array:
        complex_exp = jnp.exp(1j * self.omega_0 * x)
        real_exp = jnp.exp(-(jnp.abs(self.sigma_0 * x)) ** 2)
        return complex_exp * real_exp


@register_activation("WIRE_REAL",
                     description="Real-valued WIRE (imaginary part of the complex Gabor)")
class WireRealActivation:
    """Applies sin(omega_0 * x) * exp(-(sigma_0 * x)^2).

    The real-valued instantiation of WIRE — the imaginary part of the complex
    Gabor wavelet (Saragadam et al. 2023), for networks that cannot use complex
    weights. Safe with standard real-valued Dense layers. Note: taking the
    *magnitude* of the complex Gabor instead would cancel the oscillation and
    leave only the Gaussian envelope; the imaginary part keeps the wavelet's
    oscillation. Limiting cases: sigma_0 = 0 reduces to SIREN's sine,
    omega_0 = 0 to a Gaussian.

    Parameters
    ----------
    omega_0 : float
        Frequency parameter. Default 20.
    sigma_0 : float
        Width parameter. Default 10.
    """
    def __init__(self, omega_0: float = 20.0, sigma_0: float = 10.0):
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.sin(self.omega_0 * x) * jnp.exp(-(self.sigma_0 * x) ** 2)


@register_activation("WIRE_FINER", description="FINER WIRE activation")
class WireFinerActivation:
    """WIRE with FINER-style adaptive frequency scaling.

    Returns a complex-valued array. See ``WireActivation`` for notes on
    complex output and downstream dtype compatibility.

    Parameters
    ----------
    omega_0 : float
        Frequency parameter. Default 20.
    sigma_0 : float
        Width parameter. Default 10.
    omega_finer : float
        FINER frequency parameter. Default 5.
    """
    def __init__(self, omega_0: float = 20.0, sigma_0: float = 10.0,
                 omega_finer: float = 5.0):
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        self.omega_finer = omega_finer

    def __call__(self, x: jax.Array) -> jax.Array:
        alpha = _generate_alpha(x)
        z = alpha * x
        y = jnp.sin(self.omega_finer * z)
        scaler_omega = self.omega_0 / self.omega_finer
        scaler_sigma = self.sigma_0 / self.omega_finer
        complex_exp = jnp.exp(1j * scaler_omega * y)
        real_exp = jnp.exp(-(scaler_sigma * jnp.abs(y)) ** 2)
        return complex_exp * real_exp


@register_activation("WIRE_FINER_REAL",
                     description="Real-valued FINER-WIRE (variable-frequency real Gabor)")
class WireFinerRealActivation:
    """Applies sin(w_f (|x|+1) x) * exp(-((sigma_0/w_f) * sin(w_f (|x|+1) x))^2).

    The real-valued FINER variant of WIRE: a variable-frequency wavelet whose
    instantaneous frequency grows with |x| (FINER-style scaling, alpha = |x|+1)
    and whose Gaussian envelope is taken over the oscillation itself. Real and
    safe with standard Dense layers (unlike the complex WIRE_FINER). The
    complex form's carrier frequency omega_0 is not used here — the FINER factor
    omega_finer sets the scale.

    Parameters
    ----------
    sigma_0 : float
        Width parameter. Default 10.
    omega_finer : float
        FINER frequency parameter. Default 5.
    """
    def __init__(self, sigma_0: float = 10.0, omega_finer: float = 5.0):
        self.sigma_0 = sigma_0
        self.omega_finer = omega_finer

    def __call__(self, x: jax.Array) -> jax.Array:
        y = jnp.sin(self.omega_finer * _generate_alpha(x) * x)   # sin(w_f (|x|+1) x)
        scaler_sigma = self.sigma_0 / self.omega_finer
        return y * jnp.exp(-(scaler_sigma * y) ** 2)


# ===================================================================
# HOSC activation (hyperbolic sine composition)
# ===================================================================

@register_activation("HOSC", description="Hyperbolic sine composition activation")
class HoscActivation:
    """Applies tanh(beta * sin(x)).

    Input x should be scaled appropriately (e.g. to [-pi, pi]) since
    sin(x) is periodic and gradients become highly oscillatory for large x.

    Parameters
    ----------
    beta : float
        Scaling parameter. Default 10.
    """
    def __init__(self, beta: float = 10.0):
        self.beta = beta

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.tanh(self.beta * jnp.sin(x))


@register_activation("HOSC_FINER", description="FINER HOSC activation")
class HoscFinerActivation:
    """Applies tanh((beta/omega) * sin(omega * alpha(x) * x)).

    Input x should be scaled appropriately (e.g. to [-pi, pi]) since
    sin is periodic and gradients become highly oscillatory for large x.

    Parameters
    ----------
    beta : float
        Scaling parameter. Default 10.
    omega : float
        Frequency parameter. Default 30.
    """
    def __init__(self, beta: float = 10.0, omega: float = 30.0):
        self.beta = beta
        self.omega = omega

    def __call__(self, x: jax.Array) -> jax.Array:
        beta_scaler = self.beta / self.omega
        alpha = _generate_alpha(x)
        return jnp.tanh(beta_scaler * jnp.sin(self.omega * alpha * x))


# ===================================================================
# Sinc activation
# ===================================================================

@register_activation("SINC", description="Sinc activation")
class SincActivation:
    """Applies sinc(omega * x) = sin(pi * omega * x) / (pi * omega * x).

    Uses jnp.sinc which computes the normalised sinc: sinc(t) = sin(pi*t) / (pi*t).

    Parameters
    ----------
    omega : float
        Frequency parameter. Default 30.
    """
    def __init__(self, omega: float = 30.0):
        self.omega = omega

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.sinc(self.omega * x)