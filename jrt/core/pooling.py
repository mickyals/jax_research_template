# core/pooling.py
import inspect
import warnings
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import flax.linen as nn

POOLING: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def register_pooling(name: str, description: str = ""):
    """Register a pooling operation by name.

    Parameters
    ----------
    name : str
        Name used for lookup. Stored uppercase.
    description : str, optional
        Short description shown in ``list_pooling()``.

    Returns
    -------
    callable
        Class decorator.

    Raises
    ------
    ValueError
        If a pooling operation with the same name is already registered.

    Example
    -------
    >>> @register_pooling("MY_POOL", description="Custom pooling")
    ... class MyPool:
    ...     def __call__(self, x: jax.Array, axis) -> jax.Array:
    ...         return x.mean(axis=axis)
    """
    name_upper = name.upper()

    def decorator(cls):
        if name_upper in POOLING:
            raise ValueError(f"Pooling with name '{name_upper}' already exists.")
        POOLING[name_upper] = {"cls": cls, "description": description}
        return cls

    return decorator


def get_pooling(name: str, **kwargs):
    """Retrieve and instantiate a registered pooling operation by name.

    Inspects the constructor signature and emits a UserWarning for any
    kwargs not accepted by the pooling class. Unknown kwargs are dropped.

    Parameters
    ----------
    name : str
        Name of the registered pooling operation (case-insensitive).
    **kwargs
        Arguments forwarded to the pooling constructor.

    Returns
    -------
    callable
        An instantiated pooling operation.

    Raises
    ------
    ValueError
        If no pooling operation with the given name exists.

    Example
    -------
    >>> pool = get_pooling("MEAN")
    >>> pool = get_pooling("MAX")
    >>> pool = get_pooling("SPATIAL_MAX", kernel_size=(2, 2), strides=(2, 2))
    """
    name = name.upper()
    if name not in POOLING:
        available = ", ".join(sorted(POOLING.keys()))
        raise ValueError(
            f"Pooling '{name}' does not exist. Available: {available}"
        )

    cls = POOLING[name]["cls"]

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
                    f"get_pooling('{name}'): unknown kwargs {unknown} "
                    f"will be ignored. Valid kwargs: {valid or 'none'}.",
                    UserWarning,
                    stacklevel=2,
                )
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        except (ValueError, TypeError):
            pass

    return cls(**kwargs)


def list_pooling() -> dict[str, str]:
    """Return a sorted dictionary of all registered pooling names and descriptions.

    Returns
    -------
    dict[str, str]

    Example
    -------
    >>> list_pooling()
    {'MAX': 'Max pooling over axis', 'MEAN': 'Mean pooling over axis', ...}
    """
    return {name: info["description"] for name, info in sorted(POOLING.items())}


# ---------------------------------------------------------------------------
# Global reductions
# Axis-parameterised -- used for both spatial pooling and set aggregation.
#
# Spatial pooling (conv nets):   axis=(1, 2)  -- reduce over H, W
# Set aggregation (encoder):     axis=1        -- reduce over N_obs
# ---------------------------------------------------------------------------

@register_pooling("MEAN", description="Mean pooling over axis")
class MeanPooling:
    """Computes the mean over the specified axis.

    Parameters
    ----------
    keepdims : bool
        Whether to keep the reduced dimensions. Default False.

    Example
    -------
    >>> pool = get_pooling("MEAN")
    >>> out = pool(x, axis=1)            # set aggregation: (B, N, D) -> (B, D)
    >>> out = pool(x, axis=(1, 2))       # spatial: (B, H, W, C) -> (B, C)
    """
    def __init__(self, keepdims: bool = False):
        self.keepdims = keepdims

    def __call__(self, x: jax.Array,
                 axis: Union[int, Sequence[int]] = 1) -> jax.Array:
        return jnp.mean(x, axis=axis, keepdims=self.keepdims)


@register_pooling("MAX", description="Max pooling over axis")
class MaxPooling:
    """Computes the max over the specified axis.

    Parameters
    ----------
    keepdims : bool
        Whether to keep the reduced dimensions. Default False.

    Example
    -------
    >>> pool = get_pooling("MAX")
    >>> out = pool(x, axis=1)            # set aggregation: (B, N, D) -> (B, D)
    >>> out = pool(x, axis=(1, 2))       # spatial: (B, H, W, C) -> (B, C)
    """
    def __init__(self, keepdims: bool = False):
        self.keepdims = keepdims

    def __call__(self, x: jax.Array,
                 axis: Union[int, Sequence[int]] = 1) -> jax.Array:
        return jnp.max(x, axis=axis, keepdims=self.keepdims)


@register_pooling("MIN", description="Min pooling over axis")
class MinPooling:
    """Computes the min over the specified axis.

    Parameters
    ----------
    keepdims : bool
        Whether to keep the reduced dimensions. Default False.

    Example
    -------
    >>> pool = get_pooling("MIN")
    >>> out = pool(x, axis=1)
    """
    def __init__(self, keepdims: bool = False):
        self.keepdims = keepdims

    def __call__(self, x: jax.Array,
                 axis: Union[int, Sequence[int]] = 1) -> jax.Array:
        return jnp.min(x, axis=axis, keepdims=self.keepdims)


@register_pooling("SUM", description="Sum pooling over axis")
class SumPooling:
    """Computes the sum over the specified axis.

    Parameters
    ----------
    keepdims : bool
        Whether to keep the reduced dimensions. Default False.

    Example
    -------
    >>> pool = get_pooling("SUM")
    >>> out = pool(x, axis=1)
    """
    def __init__(self, keepdims: bool = False):
        self.keepdims = keepdims

    def __call__(self, x: jax.Array,
                 axis: Union[int, Sequence[int]] = 1) -> jax.Array:
        return jnp.sum(x, axis=axis, keepdims=self.keepdims)


@register_pooling("STD", description="Standard deviation pooling over axis")
class StdPooling:
    """Computes the standard deviation over the specified axis.

    Useful as a second-order statistic alongside mean pooling for
    richer set representations.

    Parameters
    ----------
    keepdims : bool
        Whether to keep the reduced dimensions. Default False.

    Example
    -------
    >>> pool = get_pooling("STD")
    >>> out = pool(x, axis=1)
    """
    def __init__(self, keepdims: bool = False):
        self.keepdims = keepdims

    def __call__(self, x: jax.Array,
                 axis: Union[int, Sequence[int]] = 1) -> jax.Array:
        return jnp.std(x, axis=axis, keepdims=self.keepdims)


@register_pooling("MEAN_MAX", description="Concatenation of mean and max pooling over axis")
class MeanMaxPooling:
    """Concatenates mean and max pooling along the feature dimension.

    Produces a richer representation than either alone by capturing
    both the average activation and the peak activation across the
    reduced axis.

    Parameters
    ----------
    keepdims : bool
        Whether to keep the reduced dimensions before concatenation.
        Default False.

    Notes
    -----
    Output feature dimension is 2x the input feature dimension.

    Example
    -------
    >>> pool = get_pooling("MEAN_MAX")
    >>> out = pool(x, axis=1)   # (B, N, D) -> (B, 2D)
    """
    def __init__(self, keepdims: bool = False):
        self.keepdims = keepdims

    def __call__(self, x: jax.Array,
                 axis: Union[int, Sequence[int]] = 1) -> jax.Array:
        mean = jnp.mean(x, axis=axis, keepdims=self.keepdims)
        max_ = jnp.max(x, axis=axis, keepdims=self.keepdims)
        return jnp.concatenate([mean, max_], axis=-1)


# ---------------------------------------------------------------------------
# Spatial pooling (conv nets)
# Fixed 2D window operations over H and W.
# These wrap Flax's functional pooling and are nn.Module subclasses
# since they may carry state (stride, padding) and plug into conv nets.
# ---------------------------------------------------------------------------

@register_pooling("SPATIAL_MAX", description="2D max pooling with kernel and stride")
class SpatialMaxPool(nn.Module):
    """2D max pooling over a spatial window.

    Reduces height and width by taking the max within each kernel window.
    Used in conv nets between conv blocks for downsampling.

    Parameters
    ----------
    kernel_size : tuple of int
        Size of the pooling window. Default (2, 2).
    strides : tuple of int
        Stride of the pooling window. Default (2, 2).
    padding : str
        Padding mode, 'VALID' or 'SAME'. Default 'VALID'.

    Example
    -------
    >>> pool = get_pooling("SPATIAL_MAX", kernel_size=(2, 2), strides=(2, 2))
    >>> out = pool(x)   # (B, H, W, C) -> (B, H//2, W//2, C)
    """
    kernel_size: tuple = (2, 2)
    strides: tuple = (2, 2)
    padding: str = "VALID"

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.max_pool(x, self.kernel_size, self.strides, self.padding)


@register_pooling("SPATIAL_AVG", description="2D average pooling with kernel and stride")
class SpatialAvgPool(nn.Module):
    """2D average pooling over a spatial window.

    Reduces height and width by averaging within each kernel window.
    Used in conv nets between conv blocks for downsampling.

    Parameters
    ----------
    kernel_size : tuple of int
        Size of the pooling window. Default (2, 2).
    strides : tuple of int
        Stride of the pooling window. Default (2, 2).
    padding : str
        Padding mode, 'VALID' or 'SAME'. Default 'VALID'.

    Example
    -------
    >>> pool = get_pooling("SPATIAL_AVG", kernel_size=(2, 2), strides=(2, 2))
    >>> out = pool(x)   # (B, H, W, C) -> (B, H//2, W//2, C)
    """
    kernel_size: tuple = (2, 2)
    strides: tuple = (2, 2)
    padding: str = "VALID"

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.avg_pool(x, self.kernel_size, self.strides, self.padding)


@register_pooling("GLOBAL_AVG", description="Global average pooling over spatial dimensions")
class GlobalAvgPool:
    """Global average pooling over specified spatial dimensions.

    Parameters
    ----------
    spatial_axes : tuple of int
        Axes to reduce over. Default (1, 2) for channels-last 2D spatial
        input (B, H, W, C). Use (1,) for sequence input (B, T, C).

    Example
    -------
    >>> pool = get_pooling("GLOBAL_AVG")
    >>> out = pool(x)                          # (B, H, W, C) -> (B, C)
    >>> pool = get_pooling("GLOBAL_AVG", spatial_axes=(1,))
    >>> out = pool(x)                          # (B, T, C) -> (B, C)
    """
    def __init__(self, spatial_axes: tuple = (1, 2)):
        self.spatial_axes = spatial_axes

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.mean(x, axis=self.spatial_axes)


@register_pooling("GLOBAL_MAX", description="Global max pooling over spatial dimensions")
class GlobalMaxPool:
    """Global max pooling over specified spatial dimensions.

    Parameters
    ----------
    spatial_axes : tuple of int
        Axes to reduce over. Default (1, 2) for channels-last 2D spatial
        input (B, H, W, C). Use (1,) for sequence input (B, T, C).

    Example
    -------
    >>> pool = get_pooling("GLOBAL_MAX")
    >>> out = pool(x)                          # (B, H, W, C) -> (B, C)
    """
    def __init__(self, spatial_axes: tuple = (1, 2)):
        self.spatial_axes = spatial_axes

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.max(x, axis=self.spatial_axes)