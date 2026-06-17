import math
from typing import Callable

import jax
import jax.numpy as jnp
from flax.linen import initializers as flax_init

from utils.registry import Registry


INITIALIZERS = Registry("Initializer")
register_initializer = INITIALIZERS.register
get_initializer = INITIALIZERS.get


def list_initializers() -> dict[str, str]:
    """Sorted ``{name: description}`` of all registered entries (r16)."""
    return dict(sorted(INITIALIZERS.describe().items()))


# ---------------------------------------------------------------------------
# INR weight-init helpers (SIREN/FINER)
#
# The canonical SIREN/FINER uniform bounds. Shared by the SirenInit/FinerInit
# registry classes (fan_in from construction) and the SIREN/FINER MLPs in
# core.nets.mlp, which call ``inr_first_init`` / ``inr_hidden_init`` to derive
# fan_in from the weight shape at init time. The formula lives here only.
# ---------------------------------------------------------------------------

def inr_first_bound(fan_in: int) -> float:
    """First-layer INR uniform bound: ``1 / fan_in``."""
    return 1.0 / fan_in


def inr_hidden_bound(fan_in: int, omega: float) -> float:
    """Hidden-layer INR uniform bound: ``sqrt(6 / fan_in) / omega``."""
    return math.sqrt(6.0 / fan_in) / omega


def inr_first_init(key: jax.Array, shape: tuple, dtype=jnp.float32) -> jax.Array:
    """First-layer INR weight init U(-1/fan_in, 1/fan_in); fan_in = ``shape[0]``."""
    bound = inr_first_bound(shape[0])
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def inr_hidden_init(omega: float) -> Callable:
    """Hidden-layer INR weight init; fan_in derived from ``shape[0]`` at call."""
    def init(key: jax.Array, shape: tuple, dtype=jnp.float32) -> jax.Array:
        bound = inr_hidden_bound(shape[0], omega)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    return init


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
        bound = (inr_first_bound(self.fan_in) if self.is_first
                 else inr_hidden_bound(self.fan_in, self.omega))
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
        bound = (inr_first_bound(self.fan_in) if self.is_first
                 else inr_hidden_bound(self.fan_in, self.omega))
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


@register_initializer("IDENTITY", description="Identity initialization")
class IdentityInit:
    """Identity matrix initialization.

    Only valid for square weight matrices. Raises ValueError otherwise.
    Kept hand-rolled -- flax.linen.initializers has no identity init.

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


# ---------------------------------------------------------------------------
# Standard initializers -- delegated to flax.linen.initializers (r16)
#
# These were hand-rolled; flax provides identical-purpose factories, so we
# register the flax factories directly (each ``factory(**kwargs)`` returns the
# ``(key, shape, dtype) -> array`` init the Registry hands back). Two caveats
# vs the old hand-rolled forms:
#   * flax lecun/xavier-normal draw from a TRUNCATED normal (clipped at 2 std,
#     rescaled), not a plain normal -- exact values differ.
#   * the old gain/scale/mean/(a,b) knobs are dropped; flax exposes only
#     stddev (NORMAL), scale (UNIFORM -> U(0, scale)), scale (ORTHOGONAL).
# Hand-rolled forms are kept only for the INR/filter family (SIREN, FINER,
# GABOR, WIRE) and IDENTITY (no flax equivalent).
# ---------------------------------------------------------------------------

_FLAX_INITIALIZERS = {
    "XAVIER_UNIFORM": (flax_init.xavier_uniform,
                       "Xavier/Glorot uniform initialization (flax)"),
    "XAVIER_NORMAL":  (flax_init.xavier_normal,
                       "Xavier/Glorot normal initialization (flax, truncated)"),
    "LECUN_NORMAL":   (flax_init.lecun_normal,
                       "LeCun normal initialization (flax, truncated)"),
    "NORMAL":         (flax_init.normal,
                       "Normal initialization N(0, stddev^2) (flax)"),
    "UNIFORM":        (flax_init.uniform,
                       "Uniform initialization U(0, scale) (flax)"),
    "ORTHOGONAL":     (flax_init.orthogonal,
                       "Orthogonal initialization (flax)"),
    "ZEROS":          (flax_init.zeros_init,
                       "Zero initialization (flax)"),
}

for _name, (_factory, _desc) in _FLAX_INITIALIZERS.items():
    register_initializer(_name, description=_desc)(_factory)


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