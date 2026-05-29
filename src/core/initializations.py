import inspect
import warnings
import math

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers as flax_init

INITIALIZERS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def register_initializer(name: str, description: str = ""):
    """Register an initializer by name.

    Parameters
    ----------
    name : str
        Name used for lookup. Stored uppercase.
    description : str, optional
        Short description shown in ``list_initializers()``.

    Returns
    -------
    callable
        Class decorator.

    Raises
    ------
    ValueError
        If an initializer with the same name is already registered.

    Example
    -------
    >>> @register_initializer("MY_INIT", description="Custom initializer")
    ... class MyInit:
    ...     def __call__(self, key, shape, dtype):
    ...         return jax.random.normal(key, shape, dtype)
    """
    name_upper = name.upper()

    def decorator(cls):
        if name_upper in INITIALIZERS:
            raise ValueError(
                f"Initializer with name '{name_upper}' already exists."
            )
        INITIALIZERS[name_upper] = {"cls": cls, "description": description}
        return cls

    return decorator


def get_initializer(name: str, **kwargs):
    """Retrieve and instantiate a registered initializer by name.

    Inspects the constructor signature and emits a UserWarning for any
    kwargs not accepted by the initializer class. Unknown kwargs are
    dropped rather than forwarded to prevent a TypeError at instantiation.

    The returned object is callable with signature
    ``(key: jax.Array, shape: tuple, dtype) -> jax.Array`` and can be
    passed directly to ``nn.Dense`` as ``kernel_init`` or ``bias_init``.

    Parameters
    ----------
    name : str
        Name of the registered initializer (case-insensitive).
    **kwargs
        Arguments forwarded to the initializer constructor.

    Returns
    -------
    callable
        An instantiated initializer with signature
        ``(key, shape, dtype) -> jax.Array``.

    Raises
    ------
    ValueError
        If no initializer with the given name exists.

    Example
    -------
    >>> init = get_initializer("SIREN", fan_in=256, is_first=False, omega=30.)
    >>> layer = nn.Dense(256, kernel_init=init)

    >>> init = get_initializer("XAVIER_UNIFORM", gain=0.5)
    >>> layer = nn.Dense(256, kernel_init=init)
    """
    name = name.upper()
    if name not in INITIALIZERS:
        available = ", ".join(sorted(INITIALIZERS.keys()))
        raise ValueError(
            f"Initializer '{name}' does not exist. Available: {available}"
        )

    cls = INITIALIZERS[name]["cls"]

    if kwargs:
        try:
            sig = inspect.signature(cls.__init__)
            valid = {
                k for k, p in sig.parameters.items()
                if k != "self"
                and p.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            }
            unknown = set(kwargs.keys()) - valid
            if unknown:
                warnings.warn(
                    f"get_initializer('{name}'): unknown kwargs {unknown} "
                    f"will be ignored. Valid kwargs: {valid or 'none'}.",
                    UserWarning,
                    stacklevel=2,
                )
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        except (ValueError, TypeError):
            pass

    return cls(**kwargs)


def list_initializers() -> dict[str, str]:
    """Return a sorted dictionary of all registered initializer names and descriptions.

    Returns
    -------
    dict[str, str]

    Example
    -------
    >>> list_initializers()
    {'FINER': 'FINER-specific initialization', 'SIREN': '...', ...}
    """
    return {
        name: info["description"]
        for name, info in sorted(INITIALIZERS.items())
    }


# ---------------------------------------------------------------------------
# SIREN initializers
# ---------------------------------------------------------------------------

@register_initializer("SIREN", description="SIREN-specific initialization")
class SirenInit:
    """SIREN weight initializer (Sitzmann et al. 2020).

    First layer:   U(-1/fan_in, 1/fan_in)
    Hidden layers: U(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)

    Parameters
    ----------
    fan_in : int
        Number of input features to the layer.
    is_first : bool
        If True, use first-layer bounds. Default False.
    omega : float
        Frequency parameter. Default 30.

    Example
    -------
    >>> init = get_initializer("SIREN", fan_in=256, is_first=True)
    >>> layer = nn.Dense(256, kernel_init=init)
    """
    def __init__(self, fan_in: int, is_first: bool = False,
                 omega: float = 30.0):
        self.fan_in = fan_in
        self.is_first = is_first
        self.omega = omega

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        if self.is_first:
            bound = 1.0 / self.fan_in
        else:
            bound = math.sqrt(6.0 / self.fan_in) / self.omega
        return jax.random.uniform(key, shape, dtype,
                                   minval=-bound, maxval=bound)


# ---------------------------------------------------------------------------
# FINER initializers
# ---------------------------------------------------------------------------

@register_initializer("FINER", description="FINER-specific kernel initialization")
class FinerInit:
    """FINER kernel initializer (Liu et al. 2024).

    Same weight bounds as SIREN. Use ``FINER_BIAS`` for bias init.

    First layer:   U(-1/fan_in, 1/fan_in)
    Hidden layers: U(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)

    Parameters
    ----------
    fan_in : int
        Number of input features to the layer.
    is_first : bool
        If True, use first-layer bounds. Default False.
    omega : float
        Frequency parameter. Default 30.

    Example
    -------
    >>> kernel_init = get_initializer("FINER", fan_in=256, is_first=False)
    >>> bias_init = get_initializer("FINER_BIAS", k=1.0)
    >>> layer = nn.Dense(256, kernel_init=kernel_init, bias_init=bias_init)
    """
    def __init__(self, fan_in: int, is_first: bool = False,
                 omega: float = 30.0):
        self.fan_in = fan_in
        self.is_first = is_first
        self.omega = omega

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        if self.is_first:
            bound = 1.0 / self.fan_in
        else:
            bound = math.sqrt(6.0 / self.fan_in) / self.omega
        return jax.random.uniform(key, shape, dtype,
                                   minval=-bound, maxval=bound)


@register_initializer("FINER_BIAS", description="FINER bias initialization U(-k, k)")
class FinerBiasInit:
    """FINER bias initializer -- U(-k, k).

    In Flax, kernel and bias inits are passed separately to ``nn.Dense``.
    This provides the bias component of the FINER init scheme.

    Parameters
    ----------
    k : float
        Half-range of the uniform distribution. Default 1.0.

    Example
    -------
    >>> bias_init = get_initializer("FINER_BIAS", k=1.0)
    >>> layer = nn.Dense(256, bias_init=bias_init)
    """
    def __init__(self, k: float = 1.0):
        self.k = k

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        return jax.random.uniform(key, shape, dtype,
                                   minval=-self.k, maxval=self.k)


# ---------------------------------------------------------------------------
# Xavier initializers
# -- implemented directly from formulas to avoid Flax version differences
# ---------------------------------------------------------------------------

@register_initializer("XAVIER_UNIFORM", description="Xavier uniform initialization")
class XavierUniformInit:
    """Xavier uniform initialization.

    Draws from U(-bound, bound) where
    bound = gain * sqrt(6 / (fan_in + fan_out)).

    Parameters
    ----------
    gain : float
        Scaling factor applied to the standard bound. Default 1.0.

    Example
    -------
    >>> init = get_initializer("XAVIER_UNIFORM", gain=0.5)
    >>> layer = nn.Dense(256, kernel_init=init)
    """
    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        bound = self.gain * math.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(key, shape, dtype,
                                   minval=-bound, maxval=bound)


@register_initializer("XAVIER_NORMAL", description="Xavier normal initialization")
class XavierNormalInit:
    """Xavier normal initialization.

    Draws from N(0, std^2) where
    std = gain * sqrt(2 / (fan_in + fan_out)).

    Parameters
    ----------
    gain : float
        Scaling factor applied to the standard deviation. Default 1.0.

    Example
    -------
    >>> init = get_initializer("XAVIER_NORMAL", gain=1.0)
    >>> layer = nn.Dense(256, kernel_init=init)
    """
    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        std = self.gain * math.sqrt(2.0 / (fan_in + fan_out))
        return jax.random.normal(key, shape, dtype) * std


# ---------------------------------------------------------------------------
# Standard initializers
# ---------------------------------------------------------------------------

@register_initializer("LECUN_NORMAL", description="LeCun normal initialization")
class LeCunNormalInit:
    """LeCun normal initialization.

    Draws from N(0, std^2) where std = scale / sqrt(fan_in).
    Default for most JAX/Flax MLPs.

    Parameters
    ----------
    scale : float
        Scaling factor applied to std. Default 1.0.

    Example
    -------
    >>> init = get_initializer("LECUN_NORMAL")
    >>> layer = nn.Dense(256, kernel_init=init)
    """
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        fan_in = shape[0]
        std = self.scale / math.sqrt(fan_in)
        return jax.random.normal(key, shape, dtype) * std


@register_initializer("NORMAL", description="Normal initialization")
class NormalInit:
    """Normal (Gaussian) initialization.

    Parameters
    ----------
    mean : float
        Mean of the distribution. Default 0.0.
    std : float
        Standard deviation. Default 0.1.

    Example
    -------
    >>> init = get_initializer("NORMAL", mean=0.0, std=0.01)
    >>> layer = nn.Dense(256, kernel_init=init)
    """
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        return self.mean + jax.random.normal(key, shape, dtype) * self.std


@register_initializer("UNIFORM", description="Uniform initialization")
class UniformInit:
    """Uniform initialization over [a, b].

    Parameters
    ----------
    a : float
        Lower bound. Default -0.1.
    b : float
        Upper bound. Default 0.1.

    Example
    -------
    >>> init = get_initializer("UNIFORM", a=-0.1, b=0.1)
    >>> layer = nn.Dense(256, kernel_init=init)
    """
    def __init__(self, a: float = -0.1, b: float = 0.1):
        self.a = a
        self.b = b

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        return jax.random.uniform(key, shape, dtype,
                                   minval=self.a, maxval=self.b)


@register_initializer("IDENTITY", description="Identity initialization")
class IdentityInit:
    """Identity matrix initialization.

    Only valid for square weight matrices. Raises ValueError otherwise.

    Example
    -------
    >>> init = get_initializer("IDENTITY")
    >>> layer = nn.Dense(256, kernel_init=init)   # only valid if in==out
    """
    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        if shape[0] != shape[1]:
            raise ValueError(
                f"Identity initialization requires a square matrix, "
                f"got shape {shape}."
            )
        return jnp.eye(shape[0], dtype=dtype)


@register_initializer("ORTHOGONAL", description="Orthogonal initialization")
class OrthogonalInit:
    """Orthogonal matrix initialization.

    Parameters
    ----------
    gain : float
        Scaling factor applied to the orthogonal matrix. Default 1.0.

    Example
    -------
    >>> init = get_initializer("ORTHOGONAL", gain=1.0)
    >>> layer = nn.Dense(256, kernel_init=init)
    """
    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        return flax_init.orthogonal()(key, shape, dtype) * self.gain


# ---------------------------------------------------------------------------
# MFN / WIRE initializers
# ---------------------------------------------------------------------------

@register_initializer(
    "GABOR",
    description="Gabor filter initialization for MFN (Fathony et al. 2021)",
)
class GaborInit:
    """Gabor filter weight initializer for Multiplicative Filter Networks.

    Draws from N(0, std^2) where std = std_scale / sqrt(fan_in).

    Parameters
    ----------
    std_scale : float
        Scales the standard deviation relative to 1/sqrt(fan_in).
        Default 1.0.

    Example
    -------
    >>> init = get_initializer("GABOR", std_scale=1.0)
    >>> layer = nn.Dense(256, kernel_init=init)
    """
    def __init__(self, std_scale: float = 1.0):
        self.std_scale = std_scale

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.float32) -> jax.Array:
        fan_in = shape[0]
        std = self.std_scale / math.sqrt(fan_in)
        return jax.random.normal(key, shape, dtype) * std


@register_initializer(
    "WIRE",
    description="WIRE complex weight initialization",
)
class WireInit:
    """Complex weight initializer for WIRE networks.

    Initializes complex-dtype weight matrices by drawing real and
    imaginary parts independently from N(0, std^2) where
    std = gain * sqrt(2 / (fan_in + fan_out)).

    The output dtype matches the requested dtype, supporting both
    complex64 and complex128.

    Parameters
    ----------
    gain : float
        Scaling factor applied to std. Default 1.0.

    Example
    -------
    >>> init = get_initializer("WIRE", gain=1.0)
    >>> layer = nn.Dense(256, kernel_init=init,
    ...                   param_dtype=jnp.complex64)
    """
    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, key: jax.Array, shape: tuple,
                 dtype=jnp.complex64) -> jax.Array:
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        std = self.gain * math.sqrt(2.0 / (fan_in + fan_out))
        float_dtype = (
            jnp.float64 if dtype in (jnp.complex128,) else jnp.float32
        )
        key_r, key_i = jax.random.split(key)
        real = jax.random.normal(key_r, shape, float_dtype) * std
        imag = jax.random.normal(key_i, shape, float_dtype) * std
        return (real + 1j * imag).astype(dtype)


@register_initializer("ZEROS", description="Zero initialization")
class ZerosInit:
    def __call__(self, key, shape, dtype=jnp.float32) -> jnp.ndarray:
        return jnp.zeros(shape, dtype=dtype)