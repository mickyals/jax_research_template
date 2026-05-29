# core/nets/mlp.py
import math
import inspect
from typing import Optional, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn

from core import get_activation
from core import get_initializer


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NETS: dict[str, dict] = {}


def register_mlp(name: str, description: str = ""):
    """Register an MLP net class by name.

    Parameters
    ----------
    name : str
        Name used for lookup. Stored uppercase.
    description : str, optional
        Short description shown in ``list_mlps()``.

    Returns
    -------
    callable
        Class decorator.

    Raises
    ------
    ValueError
        If a net with the same name is already registered.

    Example
    -------
    >>> @register_mlp("MY_NET", description="Custom net")
    ... class MyNet(_BaseMLP):
    ...     pass
    """
    name_upper = name.upper()

    def decorator(cls):
        if name_upper in NETS:
            raise ValueError(f"Net with name '{name_upper}' already exists.")
        NETS[name_upper] = {"cls": cls, "description": description}
        return cls

    return decorator


def get_mlp(name: str, **kwargs):
    """Retrieve and instantiate a registered net by name.

    Uses ``__dataclass_fields__`` for reliable kwarg inspection since
    Flax modules are dataclasses.

    Parameters
    ----------
    name : str
        Name of the registered net (case-insensitive).
    **kwargs
        Arguments forwarded to the net constructor. Unknown kwargs
        trigger a UserWarning and are dropped.

    Returns
    -------
    nn.Module
        An instantiated Flax Linen net module.

    Raises
    ------
    ValueError
        If no net with the given name exists.

    Example
    -------
    >>> net = get_mlp("SIREN", out_features=2, hidden_features=64, n_layers=3)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 2)))
    """
    import warnings

    name = name.upper()
    if name not in NETS:
        available = ", ".join(sorted(NETS.keys()))
        raise ValueError(f"Net '{name}' does not exist. Available: {available}")

    cls = NETS[name]["cls"]

    if kwargs:
        try:
            valid = set(cls.__dataclass_fields__.keys())
            unknown = set(kwargs.keys()) - valid
            if unknown:
                warnings.warn(
                    f"get_mlp('{name}'): unknown kwargs {unknown} will be "
                    f"ignored. Valid kwargs: {valid or 'none'}.",
                    UserWarning,
                    stacklevel=2,
                )
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        except AttributeError:
            pass

    return cls(**kwargs)


def list_mlps() -> dict[str, str]:
    """Return a sorted dictionary of all registered net names and descriptions.

    Returns
    -------
    dict[str, str]

    Example
    -------
    >>> list_mlps()
    {'FINER': 'FINER adaptive frequency SIREN', 'MLP': 'General MLP', ...}
    """
    return {name: info["description"] for name, info in sorted(NETS.items())}


# ---------------------------------------------------------------------------
# INR init helpers
# ---------------------------------------------------------------------------

def _uniform_first_init(key: jax.Array, shape: tuple, dtype=jnp.float32) -> jax.Array:
    """First-layer init for SIREN and FINER: U(-1/fan_in, 1/fan_in)."""
    fan_in = shape[0]
    bound = 1.0 / fan_in
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _uniform_inr_hidden_init(omega: float) -> Callable:
    """Hidden-layer init shared by SIREN and FINER.

    U(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega).
    """
    def init(key: jax.Array, shape: tuple, dtype=jnp.float32) -> jax.Array:
        fan_in = shape[0]
        bound = math.sqrt(6.0 / fan_in) / omega
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    return init


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _call_embedding(embedding: nn.Module, x: jnp.ndarray,
                    train: bool) -> jnp.ndarray:
    """Call an embedding module, forwarding train only if it accepts it.

    Deterministic embeddings (all current core.embeddings) define
    __call__(self, x) and do not receive train. Future trainable or
    stochastic embeddings that define __call__(self, x, train) will
    receive it automatically.

    Parameters
    ----------
    embedding : nn.Module
        Embedding module to call.
    x : jnp.ndarray
        Input array.
    train : bool
        Training flag forwarded only if the embedding accepts it.

    Returns
    -------
    jnp.ndarray
        Embedding output.
    """
    sig = inspect.signature(embedding.__call__)
    if 'train' in sig.parameters:
        return embedding(x, train=train)
    return embedding(x)


# ---------------------------------------------------------------------------
# Embedding wrappers
# ---------------------------------------------------------------------------

class LatLonEmbeddingWrapper(nn.Module):
    """Wraps a spherical embedding that takes (lat, lon) into a single x input.

    Adapts embeddings from core.embeddings that expect separate lat and lon
    arrays (e.g. SphericalGridEmbedding, DFS, SphericalHarmonicsEmbedding)
    to accept a single concatenated input array, making them composable
    with CombinedEmbedding and the embedding field on _BaseMLP.

    Parameters
    ----------
    embedding : nn.Module
        Spherical embedding with signature (lat, lon) -> jnp.ndarray.

    Notes
    -----
    Expects x of shape (N, 2) where x[:, 0] is lat and x[:, 1] is lon,
    both in radians.

    Example
    -------
    >>> from core.embeddings import get_embedding
    >>> embed = LatLonEmbeddingWrapper(
    ...     embedding=get_embedding("SPHERE_GRID", scale=8, r_min=0.01)
    ... )
    >>> net = SIREN(out_features=1, hidden_features=256, n_layers=5,
    ...             embedding=embed)
    """
    embedding: nn.Module

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.embedding(x[:, 0], x[:, 1])


class CombinedEmbedding(nn.Module):
    """Splits input into spatial and temporal components, embeds each,
    and concatenates the results before the first dense layer.

    The input x is expected to have spatial coords in columns
    0..spatial_dim-1 and temporal/other coords in the remaining columns.
    Either or both embeddings may be None, in which case those coords
    are passed through raw.

    Parameters
    ----------
    spatial_dim : int
        Number of spatial input dimensions used to split x.
        x must have at least spatial_dim columns.
    spatial_embedding : nn.Module, optional
        Applied to x[:, :spatial_dim]. Use LatLonEmbeddingWrapper
        for spherical embeddings that take (lat, lon) separately.
        If None, spatial coords are concatenated raw.
    time_embedding : nn.Module, optional
        Applied to x[:, spatial_dim:]. If None, temporal coords
        are concatenated raw. If the auxiliary values are already
        normalised scalars, leave this as None and they will be
        concatenated directly.

    Notes
    -----
    Output dimension is inferred automatically by Flax at first call.
    The first dense layer sees spatial_out + time_out features, where
    each is the embedding output dim or the raw input dim if no embedding.

    x must satisfy x.shape[1] >= spatial_dim. If x.shape[1] == spatial_dim,
    the time slice is empty and time_embedding is ignored.

    If a sub-embedding accepts a train argument (e.g. a future trainable
    embedding), it will receive the train flag automatically via
    _call_embedding. Deterministic embeddings are unaffected.

    Example
    -------
    >>> from core.embeddings import get_embedding
    >>>
    >>> # lat/lon through spherical embedding, time through Fourier features
    >>> net = SIREN(
    ...     out_features=1,
    ...     hidden_features=256,
    ...     n_layers=5,
    ...     embedding=CombinedEmbedding(
    ...         spatial_dim=2,
    ...         spatial_embedding=LatLonEmbeddingWrapper(
    ...             embedding=get_embedding("SPHERE_GRID", scale=8, r_min=0.01)
    ...         ),
    ...         time_embedding=get_embedding(
    ...             "GENERAL_POSITIONAL", input_dim=1, mapping_dim=32, scale=2.0
    ...         ),
    ...     ),
    ... )
    >>>
    >>> # lat/lon embedded, normalised pressure passed raw
    >>> net = SIREN(
    ...     out_features=1,
    ...     hidden_features=256,
    ...     n_layers=5,
    ...     embedding=CombinedEmbedding(
    ...         spatial_dim=2,
    ...         spatial_embedding=LatLonEmbeddingWrapper(
    ...             embedding=get_embedding("SPHERE_GRID", scale=8, r_min=0.01)
    ...         ),
    ...         # time_embedding=None -- normalised pressure concatenated raw
    ...     ),
    ... )
    """
    spatial_dim: int
    spatial_embedding: Optional[nn.Module] = None
    time_embedding: Optional[nn.Module] = None

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        if x.shape[1] < self.spatial_dim:
            raise ValueError(
                f"Input has {x.shape[1]} features but spatial_dim={self.spatial_dim}."
            )
        x_spatial = x[:, :self.spatial_dim]
        x_time    = x[:, self.spatial_dim:]

        if self.spatial_embedding is not None:
            x_spatial = _call_embedding(self.spatial_embedding, x_spatial, train)

        if self.time_embedding is not None and x_time.shape[1] > 0:
            x_time = _call_embedding(self.time_embedding, x_time, train)

        return jnp.concatenate([x_spatial, x_time], axis=-1)


# ---------------------------------------------------------------------------
# Base MLP
# ---------------------------------------------------------------------------

class _BaseMLP(nn.Module):
    """Internal base class for all MLP variants.

    All subclasses inherit:
        - Optional embedding layer applied before the first dense layer.
          When embedding is set, its output dimension drives the first
          layer input. When None, raw input dim is used (inferred by Flax).
        - Configurable bias and output bias initializers.
        - Dropout support via dropout_rate (global) or dropout_rates
          (per-layer list of length n_layers).

    Subclasses implement:
        _make_act()                 -> callable, hidden activation
        _make_kernel_init()         -> kernel init for hidden + output layers
        _make_first_kernel_init()   -> kernel init for first layer
                                       (default: same as _make_kernel_init)
        _make_first_act()           -> activation for first layer
                                       (default: same as _make_act)
        _make_param_dtype()         -> param dtype (default float32)

    Notes
    -----
    bias_initializer defaults to 'zeros'. Subclasses with paper-specified
    bias schemes (e.g. FINER) override _make_bias_init() directly.
    output_bias_initializer defaults to 'zeros'.

    Embedding protocol: the embedding field accepts any nn.Module.
    Deterministic embeddings define __call__(self, x) and do not receive
    the train flag. Future trainable or stochastic embeddings that define
    __call__(self, x, train: bool) will receive it automatically via
    _call_embedding. No changes needed to existing embeddings.

    Dropout: when dropout_rate > 0.0 or dropout_rates is set, a 'dropout'
    PRNG key must be passed at call time:
        model.apply(variables, x, train=True,
                    rngs={'dropout': jax.random.PRNGKey(0)})
    At eval time pass train=False -- dropout becomes a no-op and no PRNG
    key is needed.
    """
    out_features: int
    hidden_features: int
    n_layers: int
    use_bias: bool = True
    bias_initializer: str = "zeros"
    bias_initializer_kwargs: Optional[dict] = None
    output_bias_initializer: str = "zeros"
    output_bias_initializer_kwargs: Optional[dict] = None
    embedding: Optional[nn.Module] = None
    dropout_rate: float = 0.0
    dropout_rates: Optional[list] = None

    def _make_act(self) -> Callable:
        raise NotImplementedError

    def _make_kernel_init(self) -> Callable:
        raise NotImplementedError

    def _make_bias_init(self) -> Callable:
        return get_initializer(self.bias_initializer,
                               **(self.bias_initializer_kwargs or {}))

    def _make_first_kernel_init(self) -> Callable:
        return self._make_kernel_init()

    def _make_first_act(self) -> Callable:
        return self._make_act()

    def _make_output_bias_init(self) -> Callable:
        return get_initializer(self.output_bias_initializer,
                               **(self.output_bias_initializer_kwargs or {}))

    def _make_param_dtype(self):
        return jnp.float32

    def setup(self):
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")

        # resolve per-layer dropout rates
        if self.dropout_rates is not None:
            if len(self.dropout_rates) != self.n_layers:
                raise ValueError(
                    f"dropout_rates length {len(self.dropout_rates)} "
                    f"must equal n_layers {self.n_layers}"
                )
            rates = self.dropout_rates
        else:
            rates = [self.dropout_rate] * self.n_layers

        param_dtype   = self._make_param_dtype()
        kernel_init   = self._make_kernel_init()
        bias_init     = self._make_bias_init()
        first_k_init  = self._make_first_kernel_init()
        out_bias_init = self._make_output_bias_init()

        self._first_act  = self._make_first_act()
        self._hidden_act = self._make_act()

        self.first_layer = nn.Dense(
            self.hidden_features,
            use_bias=self.use_bias,
            kernel_init=first_k_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
        )
        self.hidden_layers = [
            nn.Dense(
                self.hidden_features,
                use_bias=self.use_bias,
                kernel_init=kernel_init,
                bias_init=bias_init,
                param_dtype=param_dtype,
            )
            for _ in range(self.n_layers - 1)
        ]
        self.output_layer = nn.Dense(
            self.out_features,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
            bias_init=out_bias_init,
            param_dtype=param_dtype,
        )
        self.dropouts = [nn.Dropout(rate=r) for r in rates]

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        if self.embedding is not None:
            x = _call_embedding(self.embedding, x, train)
        x = self.dropouts[0](
            self._first_act(self.first_layer(x)),
            deterministic=not train,
        )
        for layer, drop in zip(self.hidden_layers, self.dropouts[1:]):
            x = drop(self._hidden_act(layer(x)), deterministic=not train)
        return self.output_layer(x)


# ---------------------------------------------------------------------------
# Complex base
# ---------------------------------------------------------------------------

class _BaseComplexMLP(_BaseMLP):
    """Base for complex-valued networks.

    Hidden layers operate in complex arithmetic throughout.
    Output layer takes the real part of the final hidden state.
    Embedding is applied before the complex cast.

    Notes
    -----
    Dropout is applied to the magnitude of complex activations. The
    nn.Dropout module in Flax handles complex tensors by zeroing both
    real and imaginary parts of dropped units.
    """

    def _make_param_dtype(self):
        return jnp.complex64

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        if self.embedding is not None:
            x = _call_embedding(self.embedding, x, train)
        x = x.astype(jnp.complex64, copy=False)
        x = self.dropouts[0](
            self._first_act(self.first_layer(x)),
            deterministic=not train,
        )
        for layer, drop in zip(self.hidden_layers, self.dropouts[1:]):
            x = drop(self._hidden_act(layer(x)), deterministic=not train)
        return self.output_layer(x).real


# ---------------------------------------------------------------------------
# General MLP
# ---------------------------------------------------------------------------

@register_mlp("MLP", description="General MLP with configurable activation and initializer")
class MLP(_BaseMLP):
    """General MLP with fully configurable activation, initializer, and bias.

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
        Number of hidden layers. Must be >= 1.
    activation : str
        Registered activation name. Default 'relu'.
    initializer : str
        Kernel initializer for all layers. Default 'xavier_uniform'.
    initializer_kwargs : dict, optional
    activation_kwargs : dict, optional
    bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    bias_initializer_kwargs : dict, optional
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    use_bias : bool
        Default True.
    dropout_rate : float
        Global dropout rate applied after every hidden activation.
        Default 0.0 (no dropout). Inherited from _BaseMLP.
    dropout_rates : list, optional
        Per-layer dropout rates of length n_layers. Overrides dropout_rate
        when set. Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied to x before the first dense layer. Default None.

    Example
    -------
    >>> # plain MLP
    >>> net = MLP(out_features=1, hidden_features=128, n_layers=4)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    >>>
    >>> # MLP with global dropout
    >>> net = MLP(
    ...     out_features=1, hidden_features=128, n_layers=4,
    ...     dropout_rate=0.1,
    ... )
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=True,
    ...                 rngs={'dropout': jax.random.PRNGKey(1)})
    >>>
    >>> # MLP with targeted per-layer dropout
    >>> net = MLP(
    ...     out_features=1, hidden_features=128, n_layers=4,
    ...     dropout_rates=[0.0, 0.1, 0.1, 0.2],
    ... )
    >>>
    >>> # MLP with Gaussian Fourier embedding
    >>> from core.embeddings import get_embedding
    >>> net = MLP(
    ...     out_features=1, hidden_features=128, n_layers=4,
    ...     embedding=get_embedding(
    ...         "GAUSSIAN_POSITIONAL", input_dim=3, mapping_dim=64, scale=10.0
    ...     ),
    ... )
    """
    activation: str = "relu"
    initializer: str = "xavier_uniform"
    initializer_kwargs: Optional[dict] = None
    activation_kwargs: Optional[dict] = None

    def _make_act(self):
        return get_activation(self.activation, **(self.activation_kwargs or {}))

    def _make_kernel_init(self):
        return get_initializer(self.initializer, **(self.initializer_kwargs or {}))


# ---------------------------------------------------------------------------
# SIREN
# ---------------------------------------------------------------------------

@register_mlp("SIREN", description="Sinusoidal representation network (Sitzmann et al. 2020)")
class SIREN(_BaseMLP):
    """Sinusoidal representation network (Sitzmann et al. 2020).

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
        Total hidden layers including the first sine layer. Must be >= 1.
        n_layers=1 produces a single sine layer followed by the output.
    first_omega : float
        Default 30.
    hidden_omega : float
        Default 30.
    use_bias : bool
        Default True. Bias zero-initialized per paper.
    bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    bias_initializer_kwargs : dict, optional
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    dropout_rate : float
        Default 0.0. Inherited from _BaseMLP.
    dropout_rates : list, optional
        Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied to x before the first dense layer. Default None.

    Example
    -------
    >>> net = SIREN(out_features=1, hidden_features=256, n_layers=5)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    """
    first_omega: float = 30.0
    hidden_omega: float = 30.0

    def _make_act(self):
        return get_activation("SINE", omega=self.hidden_omega)

    def _make_first_act(self):
        return get_activation("SINE", omega=self.first_omega)

    def _make_kernel_init(self):
        return _uniform_inr_hidden_init(self.hidden_omega)

    def _make_first_kernel_init(self):
        return _uniform_first_init


# ---------------------------------------------------------------------------
# FINER
# ---------------------------------------------------------------------------

@register_mlp("FINER", description="Adaptive frequency SIREN (Liu et al. 2024)")
class FINERNet(_BaseMLP):
    """FINER: adaptive frequency SIREN (Liu et al. 2024).

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
    first_omega : float
        Default 30.
    hidden_omega : float
        Default 30.
    bias_k : float
        Half-range for FINER bias init U(-k, k). Default 1.0.
        Used when bias_initializer is 'finer_bias' (the default).
        Ignored if bias_initializer is overridden to another scheme.
    use_bias : bool
        Default True.
    bias_initializer : str
        Default 'finer_bias'. Override to use a different scheme.
    bias_initializer_kwargs : dict, optional
        Used only when bias_initializer is overridden from 'finer_bias'.
        When using 'finer_bias', control the range via bias_k instead.
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    dropout_rate : float
        Default 0.0. Inherited from _BaseMLP.
    dropout_rates : list, optional
        Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied to x before the first dense layer. Default None.

    Example
    -------
    >>> net = FINERNet(out_features=1, hidden_features=256, n_layers=5)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    """
    first_omega: float = 30.0
    hidden_omega: float = 30.0
    bias_k: float = 1.0
    bias_initializer: str = "finer_bias"

    def _make_act(self):
        return get_activation("FINER", omega=self.hidden_omega)

    def _make_first_act(self):
        return get_activation("FINER", omega=self.first_omega)

    def _make_kernel_init(self):
        return _uniform_inr_hidden_init(self.hidden_omega)

    def _make_first_kernel_init(self):
        return _uniform_first_init

    def _make_bias_init(self):
        if self.bias_initializer == "finer_bias":
            return get_initializer("FINER_BIAS", k=self.bias_k)
        return get_initializer(self.bias_initializer,
                               **(self.bias_initializer_kwargs or {}))


# ---------------------------------------------------------------------------
# Gaussian variants
# ---------------------------------------------------------------------------

@register_mlp("GAUSSIAN", description="MLP with Gaussian activation")
class GaussianNet(_BaseMLP):
    """MLP with Gaussian activation exp(-(sigma*x)^2).

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
    sigma : float
        Default 10.
    initializer : str
        Default 'xavier_uniform'.
    initializer_kwargs : dict, optional
    use_bias : bool
        Default True.
    bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    bias_initializer_kwargs : dict, optional
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    dropout_rate : float
        Default 0.0. Inherited from _BaseMLP.
    dropout_rates : list, optional
        Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied to x before the first dense layer. Default None.

    Example
    -------
    >>> net = GaussianNet(out_features=1, hidden_features=256, n_layers=5)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    """
    sigma: float = 10.0
    initializer: str = "xavier_uniform"
    initializer_kwargs: Optional[dict] = None

    def _make_act(self):
        return get_activation("GAUSSIAN", sigma=self.sigma)

    def _make_kernel_init(self):
        return get_initializer(self.initializer, **(self.initializer_kwargs or {}))


@register_mlp("GAUSSIAN_FINER", description="MLP with FINER Gaussian activation")
class GaussianFINERNet(_BaseMLP):
    """MLP with FINER Gaussian activation.

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
    sigma : float
        Default 10.
    omega : float
        Default 30.
    initializer : str
        Default 'xavier_uniform'.
    initializer_kwargs : dict, optional
    use_bias : bool
        Default True.
    bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    bias_initializer_kwargs : dict, optional
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    dropout_rate : float
        Default 0.0. Inherited from _BaseMLP.
    dropout_rates : list, optional
        Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied to x before the first dense layer. Default None.

    Example
    -------
    >>> net = GaussianFINERNet(out_features=1, hidden_features=256, n_layers=5)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    """
    sigma: float = 10.0
    omega: float = 30.0
    initializer: str = "xavier_uniform"
    initializer_kwargs: Optional[dict] = None

    def _make_act(self):
        return get_activation("GAUSSIAN_FINER", sigma=self.sigma, omega=self.omega)

    def _make_kernel_init(self):
        return get_initializer(self.initializer, **(self.initializer_kwargs or {}))


# ---------------------------------------------------------------------------
# WIRE variants -- real
# ---------------------------------------------------------------------------

@register_mlp("WIRE", description="MLP with real-valued WIRE activation")
class WireNet(_BaseMLP):
    """MLP with real-valued WIRE activation (magnitude of complex Gabor).

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
    omega_0 : float
        Default 20.
    sigma_0 : float
        Default 10.
    use_bias : bool
        Default True.
    bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    bias_initializer_kwargs : dict, optional
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    dropout_rate : float
        Default 0.0. Inherited from _BaseMLP.
    dropout_rates : list, optional
        Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied to x before the first dense layer. Default None.

    Example
    -------
    >>> net = WireNet(out_features=1, hidden_features=256, n_layers=5)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    """
    omega_0: float = 20.0
    sigma_0: float = 10.0

    def _make_act(self):
        return get_activation("WIRE_REAL", omega_0=self.omega_0, sigma_0=self.sigma_0)

    def _make_kernel_init(self):
        return get_initializer("WIRE")


@register_mlp("WIRE_FINER", description="MLP with real-valued FINER WIRE activation")
class WireFINERNet(_BaseMLP):
    """MLP with real-valued FINER WIRE activation.

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
    omega_0 : float
        Default 20.
    sigma_0 : float
        Default 10.
    omega_finer : float
        Default 5.
    use_bias : bool
        Default True.
    bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    bias_initializer_kwargs : dict, optional
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    dropout_rate : float
        Default 0.0. Inherited from _BaseMLP.
    dropout_rates : list, optional
        Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied to x before the first dense layer. Default None.

    Example
    -------
    >>> net = WireFINERNet(out_features=1, hidden_features=256, n_layers=5)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    """
    omega_0: float = 20.0
    sigma_0: float = 10.0
    omega_finer: float = 5.0

    def _make_act(self):
        return get_activation("WIRE_FINER_REAL", omega_0=self.omega_0,
                              sigma_0=self.sigma_0, omega_finer=self.omega_finer)

    def _make_kernel_init(self):
        return get_initializer("WIRE")


# ---------------------------------------------------------------------------
# WIRE complex
# ---------------------------------------------------------------------------

@register_mlp("WIRE_COMPLEX", description="MLP with complex Gabor wavelet activation (Saragadam et al. 2023)")
class WireComplexNet(_BaseComplexMLP):
    """MLP with complex Gabor wavelet activation throughout.

    Hidden representations are complex-valued. Output takes real part.
    Embedding is applied before the complex cast.
    Dropout zeros both real and imaginary parts of dropped units.

    WIRE_FINER complex variant is not supported -- complex sin applied to
    complex hidden states causes sinh explosion via cosh(im) overflow.

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
    omega_0 : float
        Default 20.
    sigma_0 : float
        Default 10.
    use_bias : bool
        Default True.
    bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    bias_initializer_kwargs : dict, optional
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    dropout_rate : float
        Default 0.0. Inherited from _BaseMLP.
    dropout_rates : list, optional
        Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied before the complex cast and first dense layer. Default None.

    Example
    -------
    >>> net = WireComplexNet(out_features=1, hidden_features=256, n_layers=5)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    """
    omega_0: float = 20.0
    sigma_0: float = 10.0

    def _make_act(self):
        return get_activation("WIRE", omega_0=self.omega_0, sigma_0=self.sigma_0)

    def _make_kernel_init(self):
        return get_initializer("WIRE")


# ---------------------------------------------------------------------------
# HOSC variants
# ---------------------------------------------------------------------------

@register_mlp("HOSC", description="MLP with HOSC activation tanh(beta*sin(x))")
class HOSCNet(_BaseMLP):
    """MLP with hyperbolic sine composition activation tanh(beta*sin(x)).

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
    beta : float
        Default 10.
    initializer : str
        Default 'xavier_uniform'.
    initializer_kwargs : dict, optional
    use_bias : bool
        Default True.
    bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    bias_initializer_kwargs : dict, optional
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    dropout_rate : float
        Default 0.0. Inherited from _BaseMLP.
    dropout_rates : list, optional
        Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied to x before the first dense layer. Default None.

    Example
    -------
    >>> net = HOSCNet(out_features=1, hidden_features=256, n_layers=5)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    """
    beta: float = 10.0
    initializer: str = "xavier_uniform"
    initializer_kwargs: Optional[dict] = None

    def _make_act(self):
        return get_activation("HOSC", beta=self.beta)

    def _make_kernel_init(self):
        return get_initializer(self.initializer, **(self.initializer_kwargs or {}))


@register_mlp("HOSC_FINER", description="MLP with FINER HOSC activation")
class HOSCFINERNet(_BaseMLP):
    """MLP with FINER HOSC activation.

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
    beta : float
        Default 10.
    omega : float
        Default 30.
    initializer : str
        Default 'xavier_uniform'.
    initializer_kwargs : dict, optional
    use_bias : bool
        Default True.
    bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    bias_initializer_kwargs : dict, optional
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    dropout_rate : float
        Default 0.0. Inherited from _BaseMLP.
    dropout_rates : list, optional
        Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied to x before the first dense layer. Default None.

    Example
    -------
    >>> net = HOSCFINERNet(out_features=1, hidden_features=256, n_layers=5)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    """
    beta: float = 10.0
    omega: float = 30.0
    initializer: str = "xavier_uniform"
    initializer_kwargs: Optional[dict] = None

    def _make_act(self):
        return get_activation("HOSC_FINER", beta=self.beta, omega=self.omega)

    def _make_kernel_init(self):
        return get_initializer(self.initializer, **(self.initializer_kwargs or {}))


# ---------------------------------------------------------------------------
# Sinc
# ---------------------------------------------------------------------------

@register_mlp("SINC", description="MLP with sinc activation")
class SincNet(_BaseMLP):
    """MLP with sinc activation sinc(omega*x).

    Parameters
    ----------
    out_features : int
    hidden_features : int
    n_layers : int
    omega : float
        Default 30.
    initializer : str
        Default 'xavier_uniform'.
    initializer_kwargs : dict, optional
    use_bias : bool
        Default True.
    bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    bias_initializer_kwargs : dict, optional
    output_bias_initializer : str
        Default 'zeros'. Inherited from _BaseMLP.
    output_bias_initializer_kwargs : dict, optional
    dropout_rate : float
        Default 0.0. Inherited from _BaseMLP.
    dropout_rates : list, optional
        Inherited from _BaseMLP.
    embedding : nn.Module, optional
        Applied to x before the first dense layer. Default None.

    Example
    -------
    >>> net = SincNet(out_features=1, hidden_features=256, n_layers=5)
    >>> variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
    >>> out = net.apply(variables, jnp.ones((8, 3)), train=False)
    """
    omega: float = 30.0
    initializer: str = "xavier_uniform"
    initializer_kwargs: Optional[dict] = None

    def _make_act(self):
        return get_activation("SINC", omega=self.omega)

    def _make_kernel_init(self):
        return get_initializer(self.initializer, **(self.initializer_kwargs or {}))