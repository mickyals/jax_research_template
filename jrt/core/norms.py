import jax
import flax.linen as nn

from utils.registry import Registry

# Normalisation-module registry on the shared utils.registry.Registry (r16).
# The module-level register_norm / get_norm / list_norms names are kept as thin
# aliases so existing call sites (core.__init__, nets.conv, tests) are
# unchanged. Registry.get filters kwargs via inspect.signature, which for a
# Flax nn.Module (a dataclass) resolves to its fields — verified equivalent to
# the old __dataclass_fields__ check.
NORMS = Registry("Norm")
register_norm = NORMS.register
get_norm      = NORMS.get


def list_norms() -> dict[str, str]:
    """Sorted ``{name: description}`` of all registered norms."""
    return dict(sorted(NORMS.describe().items()))


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
    Thin wrapper over flax.linen.InstanceNorm (r16; previously hand-rolled via
    GroupNorm(group_size=1)). Same per-sample, per-channel normalisation.

    Example
    -------
    >>> norm = get_norm("INSTANCE_NORM")
    >>> norm = get_norm("INSTANCE_NORM", use_bias=False)
    """
    use_scale: bool = True
    use_bias: bool = True
    epsilon: float = 1e-6

    def setup(self):
        self.norm = nn.InstanceNorm(
            epsilon=self.epsilon,
            use_scale=self.use_scale,
            use_bias=self.use_bias,
        )

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        return self.norm(x)


@register_norm("RMS_NORM", description="RMS normalisation (no mean centering)")
class RMSNorm(nn.Module):
    """RMS normalization.

    Normalizes by the root mean square of the activations with no mean
    centering. Used in modern transformer variants (LLaMA, Gemma etc.)
    as a cheaper alternative to LayerNorm.

    Thin wrapper over flax.linen.RMSNorm (r16; previously hand-rolled). The
    learnable scale lives under the ``norm`` submodule (``params['norm']['scale']``).

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

    Example
    -------
    >>> norm = get_norm("RMS_NORM")
    >>> norm = get_norm("RMS_NORM", epsilon=1e-8)
    """
    use_scale: bool = True
    epsilon: float = 1e-6

    def setup(self):
        self.norm = nn.RMSNorm(epsilon=self.epsilon, use_scale=self.use_scale)

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        return self.norm(x)