import warnings
#from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

NORMS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def register_norm(name: str, description: str = ""):
    """Register a normalisation module by name.

    Parameters
    ----------
    name : str
        Name used for lookup. Stored uppercase.
    description : str, optional
        Short description shown in ``list_norms()``.

    Returns
    -------
    callable
        Class decorator.

    Raises
    ------
    ValueError
        If a norm with the same name is already registered.

    Example
    -------
    >>> @register_norm("MY_NORM", description="Custom norm")
    ... class MyNorm(nn.Module):
    ...     def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
    ...         return x
    """
    name_upper = name.upper()

    def decorator(cls):
        if name_upper in NORMS:
            raise ValueError(f"Norm with name '{name_upper}' already exists.")
        NORMS[name_upper] = {"cls": cls, "description": description}
        return cls

    return decorator


def get_norm(name: str, **kwargs):
    """Retrieve and instantiate a registered normalisation module by name.

    Uses ``__dataclass_fields__`` for reliable kwarg inspection since
    Flax modules are dataclasses.

    Parameters
    ----------
    name : str
        Name of the registered norm (case-insensitive).
    **kwargs
        Arguments forwarded to the norm constructor. Unknown kwargs
        trigger a UserWarning and are dropped.

    Returns
    -------
    nn.Module
        An instantiated Flax Linen normalisation module.

    Raises
    ------
    ValueError
        If no norm with the given name exists.

    Example
    -------
    >>> norm = get_norm("BATCH_NORM")
    >>> norm = get_norm("GROUP_NORM", num_groups=8)
    >>> norm = get_norm("LAYER_NORM", use_bias=False)
    """
    name = name.upper()
    if name not in NORMS:
        available = ", ".join(sorted(NORMS.keys()))
        raise ValueError(
            f"Norm '{name}' does not exist. Available: {available}"
        )

    cls = NORMS[name]["cls"]

    if kwargs:
        try:
            valid = set(cls.__dataclass_fields__.keys())
            unknown = set(kwargs.keys()) - valid
            if unknown:
                warnings.warn(
                    f"get_norm('{name}'): unknown kwargs {unknown} "
                    f"will be ignored. Valid kwargs: {valid or 'none'}.",
                    UserWarning,
                    stacklevel=2,
                )
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        except AttributeError:
            pass

    return cls(**kwargs)


def list_norms() -> dict[str, str]:
    """Return a sorted dictionary of all registered norm names and descriptions.

    Returns
    -------
    dict[str, str]

    Example
    -------
    >>> list_norms()
    {'BATCH_NORM': 'Batch normalisation', 'GROUP_NORM': 'Group normalisation', ...}
    """
    return {name: info["description"] for name, info in sorted(NORMS.items())}


# ---------------------------------------------------------------------------
# Normalisations
# ---------------------------------------------------------------------------

@register_norm("BATCH_NORM", description="Batch normalisation (Ioffe & Szegedy 2015)")
class BatchNorm(nn.Module):
    """Batch normalisation.

    Normalises over the batch dimension. Maintains running statistics
    during training for use at eval time. Requires train=True during
    training and train=False during evaluation.

    Parameters
    ----------
    use_scale : bool
        Whether to learn a scale parameter (gamma). Default True.
    use_bias : bool
        Whether to learn a bias parameter (beta). Default True.
    momentum : float
        Momentum for running statistics update. Default 0.1.
    epsilon : float
        Small constant for numerical stability. Default 1e-5.

    Notes
    -----
    BatchNorm requires the train flag at call time to switch between
    batch statistics (train=True) and running statistics (train=False).
    The layer or net using this norm is responsible for passing train
    correctly.

    BatchNorm also requires mutable batch_stats in the variable
    collection during training:

        model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x, train=True,
            mutable=['batch_stats'],
        )

    Example
    -------
    >>> norm = get_norm("BATCH_NORM")
    >>> norm = get_norm("BATCH_NORM", momentum=0.01)
    """
    use_scale: bool = True
    use_bias: bool = True
    momentum: float = 0.1
    epsilon: float = 1e-5

    def setup(self):
        self.bn = nn.BatchNorm(
            use_running_average=None,
            momentum=self.momentum,
            epsilon=self.epsilon,
            use_scale=self.use_scale,
            use_bias=self.use_bias,
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        return self.bn(x, use_running_average=not train)


@register_norm("LAYER_NORM", description="Layer normalisation (Ba et al. 2016)")
class LayerNorm(nn.Module):
    """Layer normalisation.

    Normalises over the last dimension (feature dimension). Does not
    depend on batch size -- behaviour is identical at train and eval time.

    Parameters
    ----------
    use_scale : bool
        Whether to learn a scale parameter (gamma). Default True.
    use_bias : bool
        Whether to learn a bias parameter (beta). Default True.
    epsilon : float
        Small constant for numerical stability. Default 1e-6.
    train : bool
        Ignored -- included for API consistency with BatchNorm.

    Example
    -------
    >>> norm = get_norm("LAYER_NORM")
    >>> norm = get_norm("LAYER_NORM", use_bias=False)
    """
    use_scale: bool = True
    use_bias: bool = True
    epsilon: float = 1e-6

    def setup(self):
        self.ln = nn.LayerNorm(
            epsilon=self.epsilon,
            use_scale=self.use_scale,
            use_bias=self.use_bias,
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        return self.ln(x)


@register_norm("GROUP_NORM", description="Group normalisation (Wu & He 2018)")
class GroupNorm(nn.Module):
    """Group normalisation.

    Divides channels into groups and normalises within each group.
    Does not depend on batch size -- recommended over BatchNorm for
    small batches and geospatial data.

    Parameters
    ----------
    num_groups : int
        Number of groups to divide channels into. Must divide the
        channel dimension evenly. Default 8.
    use_scale : bool
        Whether to learn a scale parameter (gamma). Default True.
    use_bias : bool
        Whether to learn a bias parameter (beta). Default True.
    epsilon : float
        Small constant for numerical stability. Default 1e-6.
    train : bool
        Ignored -- included for API consistency with BatchNorm.

    Example
    -------
    >>> norm = get_norm("GROUP_NORM", num_groups=8)
    >>> norm = get_norm("GROUP_NORM", num_groups=32, use_bias=False)
    """
    num_groups: int = 8
    use_scale: bool = True
    use_bias: bool = True
    epsilon: float = 1e-6

    def setup(self):
        self.gn = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            use_scale=self.use_scale,
            use_bias=self.use_bias,
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        return self.gn(x)


@register_norm("INSTANCE_NORM", description="Instance normalisation (Ulyanov et al. 2016)")
class InstanceNorm(nn.Module):
    """Instance normalisation.

    Normalises each sample and each channel independently. Equivalent
    to GroupNorm with num_groups equal to the number of channels.
    Does not depend on batch size -- behaviour is identical at train
    and eval time.

    Parameters
    ----------
    use_scale : bool
        Whether to learn a scale parameter (gamma). Default True.
    use_bias : bool
        Whether to learn a bias parameter (beta). Default True.
    epsilon : float
        Small constant for numerical stability. Default 1e-6.
    train : bool
        Ignored -- included for API consistency with BatchNorm.

    Notes
    -----
    Implemented via nn.GroupNorm with num_groups=None and group_size=1,
    which is the idiomatic Flax way to achieve per-channel normalisation.

    Example
    -------
    >>> norm = get_norm("INSTANCE_NORM")
    >>> norm = get_norm("INSTANCE_NORM", use_bias=False)
    """
    use_scale: bool = True
    use_bias: bool = True
    epsilon: float = 1e-6

    def setup(self):
        self.gn = nn.GroupNorm(
            num_groups=None,
            group_size=1,
            epsilon=self.epsilon,
            use_scale=self.use_scale,
            use_bias=self.use_bias,
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        return self.gn(x)


@register_norm("RMS_NORM", description="RMS normalisation (no mean centering)")
class RMSNorm(nn.Module):
    """RMS normalisation.

    Normalises by the root mean square of the activations with no mean
    centering. Used in modern transformer variants (LLaMA, Gemma etc.)
    as a cheaper alternative to LayerNorm.

    Not natively available in Flax linen as of the current version.
    This is a minimal manual implementation sufficient for standard use.

    Parameters
    ----------
    use_scale : bool
        Whether to learn a scale parameter. Default True.
    epsilon : float
        Small constant for numerical stability. Default 1e-6.
    train : bool
        Ignored -- included for API consistency with BatchNorm.

    Notes
    -----
    RMSNorm has no bias term by design -- the absence of mean centering
    makes a bias redundant. use_bias is not supported.

    The feature dimension (last axis of x) is inferred at first call
    and fixed thereafter. Do not reuse this module with inputs of
    different feature dimensions.

    Example
    -------
    >>> norm = get_norm("RMS_NORM")
    >>> norm = get_norm("RMS_NORM", epsilon=1e-8)
    """
    use_scale: bool = True
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        x_norm = x / rms
        if self.use_scale:
            scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
            x_norm = x_norm * scale
        return x_norm