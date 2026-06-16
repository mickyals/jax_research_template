# core/pooling.py
import inspect
import warnings
from typing import Sequence, Union

import jax
import jax.numpy as jnp
import flax.linen as nn

from utils.registry import Registry


POOLING = Registry("Pooling")
register_pooling = POOLING.register
get_pooling = POOLING.get


def list_pooling() -> dict[str, str]:
    """Sorted ``{name: description}`` of all registered entries (r16)."""
    return dict(sorted(POOLING.describe().items()))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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