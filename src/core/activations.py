import jax
import jax.numpy as jnp
import flax.nnx as nnx

ACTIVATIONS: dict[str, dict] = {}


def register_activation(name: str, description: str = ""):
    """Register an activation function by name.

    Parameters
    ----------
    name : str
        Uppercase name used for lookup.
    description : str, optional
        Short description of the activation.

    Returns
    -------
    callable
        Class decorator.

    Raises
    ------
    ValueError
        If an activation with the same name is already registered.

    Example
    -------
    >>> @register_activation("MY_ACT", description="Custom activation")
    ... class MyActivation:
    ...     def __call__(self, x):
    ...         return x
    """
    name = name.upper()

    def decorator(cls):
        if name in ACTIVATIONS:
            raise ValueError(f"Activation with name {name} already exists.")
        ACTIVATIONS[name] = {"cls": cls, "description": description}
        return cls

    return decorator


def get_activation(name: str, **kwargs):
    """Get an activation function by name.

    Parameters
    ----------
    name : str
        Name of the registered activation (case-insensitive).
    **kwargs
        Arguments forwarded to the activation constructor.

    Returns
    -------
    callable
        An instantiated activation function.

    Raises
    ------
    ValueError
        If no activation with the given name exists.

    Example
    -------
    >>> act = get_activation("SINE", omega=30)
    >>> act(jnp.array([0.0, 1.0]))
    """
    name = name.upper()
    if name not in ACTIVATIONS:
        available = ", ".join(sorted(ACTIVATIONS.keys()))
        raise ValueError(f"Activation '{name}' does not exist. Available: {available}")
    return ACTIVATIONS[name]["cls"](**kwargs)


def list_activations() -> dict[str, str]:
    """Return a dict of all registered activation names and descriptions.

    Returns
    -------
    dict[str, str]

    Example
    -------
    >>> list_activations()
    {'RELU': 'ReLU activation', 'SINE': 'Sine activation', ...}
    """
    return {name: info["description"] for name, info in sorted(ACTIVATIONS.items())}


def _generate_alpha(x: jax.Array) -> jax.Array:
    """Compute abs(x) + 1 as a scaling factor for FINER-style activations."""
    return jnp.abs(x) + 1


# ===================================================================
# Flax/JAX built-in wrappers
# ===================================================================

@register_activation("RELU", description="ReLU activation")
class ReLU:
    def __call__(self, x):
        return nnx.relu(x)


@register_activation("LEAKY_RELU", description="Leaky ReLU activation")
class LeakyReLU:
    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def __call__(self, x):
        return nnx.leaky_relu(x, self.negative_slope)


@register_activation("SILU", description="SiLU (swish) activation")
class SiLU:
    def __call__(self, x):
        return nnx.silu(x)


@register_activation("SIGMOID", description="Sigmoid activation")
class Sigmoid:
    def __call__(self, x):
        return nnx.sigmoid(x)


@register_activation("TANH", description="Tanh activation")
class Tanh:
    def __call__(self, x):
        return jnp.tanh(x)


@register_activation("GELU", description="Gaussian error linear unit activation")
class GELU:
    def __init__(self, approximate: bool = True):
        self.approximate = approximate

    def __call__(self, x):
        return nnx.gelu(x, approximate=self.approximate)


@register_activation("ELU", description="Exponential linear unit activation")
class ELU:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x):
        return nnx.elu(x, alpha=self.alpha)


@register_activation("SELU", description="Scaled exponential linear unit activation")
class SELU:
    def __call__(self, x):
        return nnx.selu(x)


@register_activation("SOFTPLUS", description="Softplus activation")
class Softplus:
    def __call__(self, x):
        return nnx.softplus(x)


@register_activation("IDENTITY", description="Identity (no-op) activation")
class Identity:
    def __call__(self, x):
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

    def __call__(self, x):
        return jnp.sin(self.omega * x)


@register_activation("FINER", description="FINER sine activation")
class SineFinerActivation:
    """Applies sin(omega * alpha(x) * x) where alpha(x) = |x| + 1.

    Parameters
    ----------
    omega : float
        Frequency parameter. Default 30.
    """
    def __init__(self, omega: float = 30.0):
        self.omega = omega

    def __call__(self, x):
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

    def __call__(self, x):
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

    def __call__(self, x):
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

    def __call__(self, x):
        complex_exp = jnp.exp(1j * self.omega_0 * x)
        real_exp = jnp.exp(-(jnp.abs(self.sigma_0 * x)) ** 2)
        return complex_exp * real_exp


@register_activation("WIRE_FINER", description="FINER WIRE activation")
class WireFinerActivation:
    """WIRE with FINER-style adaptive frequency scaling.

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

    def __call__(self, x):
        alpha = _generate_alpha(x)
        z = alpha * x
        y = jnp.sin(self.omega_finer * z)
        scaler_omega = self.omega_0 / self.omega_finer
        scaler_sigma = self.sigma_0 / self.omega_finer
        complex_exp = jnp.exp(1j * scaler_omega * y)
        real_exp = jnp.exp(-(scaler_sigma * jnp.abs(y)) ** 2)
        return complex_exp * real_exp


# ===================================================================
# HOSC activation (hyperbolic sine composition)
# ===================================================================

@register_activation("HOSC", description="Hyperbolic sine composition activation")
class HoscActivation:
    """Applies tanh(beta * sin(x)).

    Parameters
    ----------
    beta : float
        Scaling parameter. Default 10.
    """
    def __init__(self, beta: float = 10.0):
        self.beta = beta

    def __call__(self, x):
        return jnp.tanh(self.beta * jnp.sin(x))


@register_activation("HOSC_FINER", description="FINER HOSC activation")
class HoscFinerActivation:
    """Applies tanh((beta/omega) * sin(omega * alpha(x) * x)).

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

    def __call__(self, x):
        beta_scaler = self.beta / self.omega
        alpha = _generate_alpha(x)
        return jnp.tanh(beta_scaler * jnp.sin(self.omega * alpha * x))


# ===================================================================
# Sinc activation
# ===================================================================

@register_activation("SINC", description="Sinc activation")
class SincActivation:
    """Applies sinc(omega * x) = sin(pi * omega * x) / (pi * omega * x).

    Parameters
    ----------
    omega : float
        Frequency parameter. Default 30.
    """
    def __init__(self, omega: float = 30.0):
        self.omega = omega

    def __call__(self, x):
        return jnp.sinc(self.omega * x)