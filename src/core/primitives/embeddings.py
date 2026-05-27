import inspect
import warnings
import math

import jax
import jax.numpy as jnp
import flax.linen as nn

EMBEDDINGS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def register_embedding(name: str, description: str = ""):
    """Register an embedding module by name.

    Parameters
    ----------
    name : str
        Name used for lookup. Stored uppercase.
    description : str, optional
        Short description shown in ``list_embeddings()``.

    Returns
    -------
    callable
        Class decorator.

    Raises
    ------
    ValueError
        If an embedding with the same name is already registered.

    Example
    -------
    >>> @register_embedding("MY_EMBED", description="Custom embedding")
    ... class MyEmbedding(nn.Module):
    ...     @nn.compact
    ...     def __call__(self, x: jax.Array) -> jax.Array:
    ...         return x
    """
    name_upper = name.upper()

    def decorator(cls):
        if name_upper in EMBEDDINGS:
            raise ValueError(
                f"Embedding with name '{name_upper}' already exists."
            )
        EMBEDDINGS[name_upper] = {"cls": cls, "description": description}
        return cls

    return decorator


def get_embedding(name: str, **kwargs):
    """Retrieve and instantiate a registered embedding by name.

    Inspects the constructor signature and emits a UserWarning for any
    kwargs not accepted by the embedding class. Unknown kwargs are dropped
    rather than forwarded to prevent a TypeError at instantiation.

    In Flax Linen, instantiating a module does not run any computation --
    call ``module.init(key, *inputs)`` to initialise parameters and
    ``module.apply(variables, *inputs)`` to run a forward pass.

    Parameters
    ----------
    name : str
        Name of the registered embedding (case-insensitive).
    **kwargs
        Arguments forwarded to the embedding constructor (hyperparameters
        only). Unknown kwargs trigger a UserWarning and are dropped.

    Returns
    -------
    nn.Module
        An instantiated Flax Linen embedding module.

    Raises
    ------
    ValueError
        If no embedding with the given name exists.

    Example
    -------
    >>> embed = get_embedding("GAUSSIAN_POSITIONAL",
    ...                        input_dim=2, mapping_dim=64, scale=10.0)
    >>> variables = embed.init(jax.random.PRNGKey(0), jnp.ones((8, 2)))
    >>> out = embed.apply(variables, jnp.ones((8, 2)))
    >>> out.shape
    (8, 64)
    """
    name = name.upper()
    if name not in EMBEDDINGS:
        available = ", ".join(sorted(EMBEDDINGS.keys()))
        raise ValueError(
            f"Embedding '{name}' does not exist. Available: {available}"
        )

    cls = EMBEDDINGS[name]["cls"]

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
                    f"get_embedding('{name}'): unknown kwargs {unknown} "
                    f"will be ignored. Valid kwargs: {valid or 'none'}.",
                    UserWarning,
                    stacklevel=2,
                )
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        except (ValueError, TypeError):
            pass

    return cls(**kwargs)


def list_embeddings() -> dict[str, str]:
    """Return a sorted dictionary of all registered embedding names and descriptions.

    Returns
    -------
    dict[str, str]

    Example
    -------
    >>> list_embeddings()
    {'DFS': 'Double Fourier Sphere embedding', ...}
    """
    return {
        name: info["description"]
        for name, info in sorted(EMBEDDINGS.items())
    }


# ---------------------------------------------------------------------------
# Shared beta schedule
# ---------------------------------------------------------------------------

def _make_beta(scale: int, r_min: float, r_max: float) -> jax.Array:
    """Geometric frequency schedule from r_min to r_max over scale steps.

    Parameters
    ----------
    scale : int
        Number of frequency bins.
    r_min : float
        Minimum frequency.
    r_max : float
        Maximum frequency.

    Returns
    -------
    jax.Array
        Shape (scale,).
    """
    g = r_max / r_min
    s = jnp.arange(scale, dtype=jnp.float32)
    scale_minus_one = max(scale - 1, 1)
    return r_min * g ** (s / scale_minus_one)


# ---------------------------------------------------------------------------
# Gaussian Fourier embedding
# ---------------------------------------------------------------------------

@register_embedding(
    "GAUSSIAN_POSITIONAL",
    description="Random Gaussian Fourier features (Tancik et al. 2020)",
)
class GaussianFourierEmbedding(nn.Module):
    """Random Fourier features with a Gaussian frequency matrix.

    Samples a fixed random projection matrix B ~ N(0, scale^2) at init
    and computes [cos(2*pi*x*B), sin(2*pi*x*B)].

    B is stored as a plain Python attribute in setup() -- it is never
    placed in params or constants and is therefore invisible to the
    optimizer. This matches the eqx.static_field() pattern.

    To get different projections across runs pass different seeds via
    the seed parameter.

    Following Tancik et al. 2020 (https://arxiv.org/abs/2006.10739).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input coordinates.
    mapping_dim : int
        Output dimensionality. Must be even.
    scale : float
        Standard deviation of the Gaussian used to sample frequencies.
    seed : int
        Seed used to sample the frequency matrix. Default 0.

    Attributes
    ----------
    out_features : int
        Output feature dimension, equal to mapping_dim.

    Raises
    ------
    ValueError
        If mapping_dim is not even.
    """
    input_dim: int
    mapping_dim: int
    scale: float
    seed: int = 0

    def setup(self):
        if self.mapping_dim % 2 != 0:
            raise ValueError(
                f"mapping_dim must be even, got {self.mapping_dim}."
            )
        # stored as plain attribute -- never seen by optimizer
        # equivalent to eqx.static_field()
        self.B = jax.random.normal(
            jax.random.PRNGKey(self.seed),
            (self.input_dim, self.mapping_dim // 2),
        ) * self.scale

    @property
    def out_features(self) -> int:
        return self.mapping_dim

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (N, input_dim).

        Returns
        -------
        jax.Array
            Shape (N, mapping_dim).
        """
        x_proj = 2.0 * jnp.pi * x @ self.B
        return jnp.concatenate([jnp.cos(x_proj), jnp.sin(x_proj)], axis=-1)

# ---------------------------------------------------------------------------
# Deterministic positional embedding
# ---------------------------------------------------------------------------

@register_embedding(
    "GENERAL_POSITIONAL",
    description="Deterministic geometric frequency bank (Tancik et al. 2020)",
)
class PositionalEmbedding(nn.Module):
    """Deterministic positional encoding with geometrically spaced frequencies.

    Constructs a fixed frequency matrix where frequencies are spaced as
    scale^(j / mapping_dim) for j = 0 ... mapping_dim-1, broadcast across
    all input dimensions. With scale=1 this reduces to standard sinusoidal
    positional encoding.

    Following Tancik et al. 2020 (https://arxiv.org/abs/2006.10739).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input coordinates.
    mapping_dim : int
        Number of frequencies per input dimension.
    scale : float
        Frequency growth base. scale=1 gives uniform frequencies.

    Attributes
    ----------
    out_features : int
        Output feature dimension, equal to 2 * mapping_dim.
    """
    input_dim: int
    mapping_dim: int
    scale: float

    @property
    def out_features(self) -> int:
        return 2 * self.mapping_dim

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array
            Shape (N, input_dim).

        Returns
        -------
        jax.Array
            Shape (N, 2 * mapping_dim).
        """
        j = jnp.arange(self.mapping_dim, dtype=jnp.float32)
        beta_row = self.scale ** (j / self.mapping_dim)
        beta = jnp.tile(beta_row[None, :], (self.input_dim, 1))
        x_proj = 2.0 * jnp.pi * x @ beta
        return jnp.concatenate([jnp.cos(x_proj), jnp.sin(x_proj)], axis=-1)


# ---------------------------------------------------------------------------
# SPHERE_GRID
# ---------------------------------------------------------------------------

@register_embedding(
    "SPHERE_GRID",
    description="Sphere2Vec grid embedding (Mai et al. 2023)",
)
class SphericalGridEmbedding(nn.Module):
    """Independent multi-scale sinusoidal encoding of lat and lon.

    Applies a geometric frequency bank independently to lat and lon and
    concatenates sin and cos for each. No cross-terms between lat and lon.

    output_dim = 4 * scale

    Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

    Parameters
    ----------
    scale : int
        Number of frequency bins per coordinate.
    r_min : float
        Minimum frequency.
    r_max : float
        Maximum frequency. Default 1.0.

    Attributes
    ----------
    out_features : int
        Output feature dimension, equal to 4 * scale.
    """
    scale: int
    r_min: float
    r_max: float = 1.0

    @property
    def out_features(self) -> int:
        return 4 * self.scale

    @nn.compact
    def __call__(self, lat: jax.Array, lon: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        lat : jax.Array
            Latitudes in radians, shape (N,).
        lon : jax.Array
            Longitudes in radians, shape (N,).

        Returns
        -------
        jax.Array
            Shape (N, 4 * scale).
        """
        beta = _make_beta(self.scale, self.r_min, self.r_max)
        lat_t = lat[:, None] * beta
        lon_t = lon[:, None] * beta
        return jnp.concatenate([
            jnp.sin(lat_t),
            jnp.cos(lat_t),
            jnp.sin(lon_t),
            jnp.cos(lon_t),
        ], axis=-1)


# ---------------------------------------------------------------------------
# SPHERE_C
# ---------------------------------------------------------------------------

@register_embedding(
    "SPHERE_C",
    description="Sphere2Vec Cartesian embedding (Mai et al. 2023)",
)
class SphericalCartesianEmbedding(nn.Module):
    """Multi-scale encoding of the 3D unit Cartesian vector.

    Converts (lat, lon) to unit sphere Cartesian coordinates
    (cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)) and applies a geometric
    frequency bank to each Cartesian component via sin.

    output_dim = 3 * scale

    Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

    Parameters
    ----------
    scale : int
        Number of frequency bins per Cartesian component.
    r_min : float
        Minimum frequency.
    r_max : float
        Maximum frequency. Default 1.0.

    Attributes
    ----------
    out_features : int
        Output feature dimension, equal to 3 * scale.
    """
    scale: int
    r_min: float
    r_max: float = 1.0

    @property
    def out_features(self) -> int:
        return 3 * self.scale

    @nn.compact
    def __call__(self, lat: jax.Array, lon: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        lat : jax.Array
            Latitudes in radians, shape (N,).
        lon : jax.Array
            Longitudes in radians, shape (N,).

        Returns
        -------
        jax.Array
            Shape (N, 3 * scale).
        """
        beta = _make_beta(self.scale, self.r_min, self.r_max)
        x = jnp.cos(lat) * jnp.cos(lon)
        y = jnp.cos(lat) * jnp.sin(lon)
        z = jnp.sin(lat)
        return jnp.concatenate([
            jnp.sin(x[:, None] * beta),
            jnp.sin(y[:, None] * beta),
            jnp.sin(z[:, None] * beta),
        ], axis=-1)


# ---------------------------------------------------------------------------
# SPHERE_M
# ---------------------------------------------------------------------------

@register_embedding(
    "SPHERE_M",
    description="Sphere2Vec multi-scale embedding (Mai et al. 2023)",
)
class SphericalMultiScaleEmbedding(nn.Module):
    """Multi-scale encoding mixing transformed and raw spherical coordinates.

    Combines multi-scale transformed lat terms with raw lon (and vice versa)
    to capture interactions between the two coordinates at different scales.

    output_dim = 5 * scale

    Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

    Parameters
    ----------
    scale : int
        Number of frequency bins.
    r_min : float
        Minimum frequency.
    r_max : float
        Maximum frequency. Default 1.0.

    Attributes
    ----------
    out_features : int
        Output feature dimension, equal to 5 * scale.
    """
    scale: int
    r_min: float
    r_max: float = 1.0

    @property
    def out_features(self) -> int:
        return 5 * self.scale

    @nn.compact
    def __call__(self, lat: jax.Array, lon: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        lat : jax.Array
            Latitudes in radians, shape (N,).
        lon : jax.Array
            Longitudes in radians, shape (N,).

        Returns
        -------
        jax.Array
            Shape (N, 5 * scale).
        """
        beta = _make_beta(self.scale, self.r_min, self.r_max)
        lat_t = lat[:, None] * beta
        lon_t = lon[:, None] * beta
        return jnp.concatenate([
            jnp.sin(lat_t),
            jnp.cos(lat_t) * jnp.cos(lon[:, None]),
            jnp.cos(lat[:, None]) * jnp.cos(lon_t),
            jnp.cos(lat_t) * jnp.sin(lon[:, None]),
            jnp.sin(lat[:, None]) * jnp.cos(lon_t),
        ], axis=-1)


# ---------------------------------------------------------------------------
# DFS
# ---------------------------------------------------------------------------

@register_embedding(
    "DFS",
    description="Double Fourier Sphere embedding (Mai et al. 2023)",
)
class DoubleFourierSphericalEmbedding(nn.Module):
    """Double Fourier Sphere encoding with cross-frequency interaction terms.

    Computes base sin/cos terms for lat and lon independently, then computes
    all pairwise products (cos*cos, cos*sin, sin*cos, sin*sin) across the
    scale dimension.

    output_dim = 4 * scale + 4 * scale^2

    Note: output grows quadratically with scale. Use small values
    (scale <= 16 is recommended).

    Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

    Parameters
    ----------
    scale : int
        Number of frequency bins per coordinate.
    r_lat_min : float
        Minimum frequency for latitude.
    r_lon_min : float
        Minimum frequency for longitude.
    r_max : float
        Maximum frequency. Default 1.0.

    Attributes
    ----------
    out_features : int
        Output feature dimension, equal to 4 * scale + 4 * scale^2.
    """
    scale: int
    r_lat_min: float
    r_lon_min: float
    r_max: float = 1.0

    @property
    def out_features(self) -> int:
        return 4 * self.scale + 4 * self.scale ** 2

    @nn.compact
    def __call__(self, lat: jax.Array, lon: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        lat : jax.Array
            Latitudes in radians, shape (N,).
        lon : jax.Array
            Longitudes in radians, shape (N,).

        Returns
        -------
        jax.Array
            Shape (N, 4 * scale + 4 * scale^2).
        """
        beta_lat = _make_beta(self.scale, self.r_lat_min, self.r_max)
        beta_lon = _make_beta(self.scale, self.r_lon_min, self.r_max)

        lat_s = lat[:, None] * beta_lat
        lon_s = lon[:, None] * beta_lon

        lat_cos = jnp.cos(lat_s)
        lat_sin = jnp.sin(lat_s)
        lon_cos = jnp.cos(lon_s)
        lon_sin = jnp.sin(lon_s)

        base = jnp.concatenate([lat_sin, lat_cos, lon_sin, lon_cos], axis=-1)

        N = lat.shape[0]
        cc = (lat_cos[:, :, None] * lon_cos[:, None, :]).reshape(N, -1)
        cs = (lat_cos[:, :, None] * lon_sin[:, None, :]).reshape(N, -1)
        sc = (lat_sin[:, :, None] * lon_cos[:, None, :]).reshape(N, -1)
        ss = (lat_sin[:, :, None] * lon_sin[:, None, :]).reshape(N, -1)

        return jnp.concatenate([base, cc, cs, sc, ss], axis=-1)


# ---------------------------------------------------------------------------
# SPHERE_C+
# ---------------------------------------------------------------------------

@register_embedding(
    "SPHERE_C+",
    description="Sphere2Vec Cartesian-plus embedding (Mai et al. 2023)",
)
class SphericalCartesianPlusEmbedding(nn.Module):
    """Sphere-C augmented with independent lat/lon sinusoidal terms.

    Extends SPHERE_C by adding sin/cos of the transformed lat and lon
    directly alongside the Cartesian component encoding.

    output_dim = 6 * scale

    Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

    Parameters
    ----------
    scale : int
        Number of frequency bins.
    r_min : float
        Minimum frequency.
    r_max : float
        Maximum frequency. Default 1.0.

    Attributes
    ----------
    out_features : int
        Output feature dimension, equal to 6 * scale.
    """
    scale: int
    r_min: float
    r_max: float = 1.0

    @property
    def out_features(self) -> int:
        return 6 * self.scale

    @nn.compact
    def __call__(self, lat: jax.Array, lon: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        lat : jax.Array
            Latitudes in radians, shape (N,).
        lon : jax.Array
            Longitudes in radians, shape (N,).

        Returns
        -------
        jax.Array
            Shape (N, 6 * scale).
        """
        beta = _make_beta(self.scale, self.r_min, self.r_max)
        x = jnp.cos(lat) * jnp.cos(lon)
        y = jnp.cos(lat) * jnp.sin(lon)
        z = jnp.sin(lat)
        lat_t = lat[:, None] * beta
        lon_t = lon[:, None] * beta
        return jnp.concatenate([
            jnp.sin(x[:, None] * beta),
            jnp.sin(y[:, None] * beta),
            jnp.sin(z[:, None] * beta),
            jnp.sin(lat_t),
            jnp.sin(lon_t),
            jnp.cos(lon_t),
        ], axis=-1)


# ---------------------------------------------------------------------------
# SPHERE_M+
# ---------------------------------------------------------------------------

@register_embedding(
    "SPHERE_M+",
    description="Sphere2Vec multi-scale-plus embedding (Mai et al. 2023)",
)
class SphericalMultiScalePlusEmbedding(nn.Module):
    """Sphere-M augmented with independent transformed lat/lon sin/cos terms.

    Extends SPHERE_M by adding sin(lat_t), sin(lon_t), cos(lon_t) terms.

    output_dim = 8 * scale

    Following Mai et al. 2023 (https://arxiv.org/abs/2306.17624).

    Parameters
    ----------
    scale : int
        Number of frequency bins.
    r_min : float
        Minimum frequency.
    r_max : float
        Maximum frequency. Default 1.0.

    Attributes
    ----------
    out_features : int
        Output feature dimension, equal to 8 * scale.
    """
    scale: int
    r_min: float
    r_max: float = 1.0

    @property
    def out_features(self) -> int:
        return 8 * self.scale

    @nn.compact
    def __call__(self, lat: jax.Array, lon: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        lat : jax.Array
            Latitudes in radians, shape (N,).
        lon : jax.Array
            Longitudes in radians, shape (N,).

        Returns
        -------
        jax.Array
            Shape (N, 8 * scale).
        """
        beta = _make_beta(self.scale, self.r_min, self.r_max)
        lat_t = lat[:, None] * beta
        lon_t = lon[:, None] * beta
        return jnp.concatenate([
            jnp.sin(lat_t),
            jnp.cos(lat_t) * jnp.cos(lon[:, None]),
            jnp.cos(lat[:, None]) * jnp.cos(lon_t),
            jnp.cos(lat_t) * jnp.sin(lon[:, None]),
            jnp.sin(lat[:, None]) * jnp.cos(lon_t),
            jnp.cos(lat_t),
            jnp.sin(lon_t),
            jnp.cos(lon_t),
        ], axis=-1)


@register_embedding(
    "SPHERICAL_HARMONICS",
    description="Spherical harmonic location encoding (Russwurm et al. 2024)",
)
class SphericalHarmonicsEmbedding(nn.Module):
    """Spherical harmonic basis functions as positional encoding.

    Evaluates real spherical harmonics Y_l^m(lat, lon) up to degree L.
    Unlike DFS and Sphere2Vec variants, spherical harmonics are natively
    defined on the sphere and produce no pole artifacts.

    For degree L the total number of basis functions is L^2, since for
    each l in 0..L-1 there are 2l+1 values of m in [-l, l].

    output_dim = legendre_polys^2

    Following Russwurm et al. 2024 (https://arxiv.org/abs/2310.06743).

    Parameters
    ----------
    legendre_polys : int
    Number of degrees to evaluate, covering degrees 0 to legendre_polys-1.
    The highest degree used is legendre_polys - 1. output_dim = legendre_polys^2.
    Default 10. Recommended range 5-20. For values above 20 numerical
    errors in the Legendre recursion may accumulate.

    Attributes
    ----------
    out_features : int
        Output feature dimension, equal to legendre_polys^2.

    Notes
    -----
    Input lat must be in radians in [-pi/2, pi/2].
    Input lon must be in radians in [-pi, pi].
    Internally converts to colatitude theta in [0, pi] and
    longitude phi in [0, 2*pi] as required by the SH convention.

    Normalisation constants are precomputed in setup() using plain Python
    math and stored as a static jnp.array. This avoids math.factorial
    being called inside JAX-traced code which would cause a TypeError.
    """
    legendre_polys: int = 10

    @property
    def out_features(self) -> int:
        return self.legendre_polys ** 2

    def setup(self):
        """Precompute normalisation constants K_l^m for all (l, m) pairs.

        Uses plain Python math.factorial which is safe here because setup()
        runs at module construction time, not inside a JAX-traced call.
        Stored as a static jnp.array of shape (L^2,).
        """
        norms = []
        for l in range(self.legendre_polys):
            for m in range(-l, l + 1):
                abs_m = abs(m)
                num = (2 * l + 1) * math.factorial(l - abs_m)
                den = 4.0 * math.pi * math.factorial(l + abs_m)
                norms.append(math.sqrt(num / den))
        self.norm_const = jnp.array(norms)   # (L^2,) -- static, never traced

    @staticmethod
    def _associated_legendre(l: int, m: int, x: jax.Array) -> jax.Array:
        """Compute associated Legendre polynomial P_l^m(x) recursively.

        Parameters
        ----------
        l : int
            Degree.
        m : int
            Order (>= 0).
        x : jax.Array
            Argument, shape (N,). For spherical harmonics x = cos(theta).

        Returns
        -------
        jax.Array
            Shape (N,).
        """
        pmm = jnp.ones_like(x)
        if m > 0:
            somx2 = jnp.sqrt(jnp.clip((1.0 - x) * (1.0 + x), 0.0))
            fact = 1.0
            for _ in range(1, m + 1):
                pmm = pmm * (-fact) * somx2
                fact += 2.0

        if l == m:
            return pmm

        pmmp1 = x * (2.0 * m + 1.0) * pmm
        if l == m + 1:
            return pmmp1

        pll = pmmp1
        for ll in range(m + 2, l + 1):
            pll = (
                (2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm
            ) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll

        return pll

    def __call__(self, lat: jax.Array, lon: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        lat : jax.Array
            Latitudes in radians, shape (N,). Range [-pi/2, pi/2].
        lon : jax.Array
            Longitudes in radians, shape (N,). Range [-pi, pi].

        Returns
        -------
        jax.Array
            Shape (N, legendre_polys^2).
        """
        # convert to SH convention
        theta = jnp.pi / 2.0 - lat    # colatitude in [0, pi]
        phi = lon + jnp.pi             # longitude in [0, 2*pi]
        cos_theta = jnp.cos(theta)

        basis = []
        idx = 0
        for l in range(self.legendre_polys):
            for m in range(-l, l + 1):
                K = self.norm_const[idx]
                P = self._associated_legendre(l, abs(m), cos_theta)
                if m < 0:
                    y = jnp.sqrt(2.0) * K * jnp.sin(-m * phi) * P
                elif m == 0:
                    y = K * P
                else:
                    y = jnp.sqrt(2.0) * K * jnp.cos(m * phi) * P
                basis.append(y)
                idx += 1

        return jnp.stack(basis, axis=-1)