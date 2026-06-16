# noinspection SpellCheckingInspection
"""
training/optimizers.py

Registry of optax optimizers and learning-rate schedules.

Follows the same register / get / list pattern used throughout the template
(activations, norms, initializations, etc.) so every component is visible
and swappable from a YAML config.

Usage
-----
    from training.optimizers import get_optimizer, get_scheduler

    schedule  = get_scheduler("warmup_cosine",
                               init_value=0.0, peak_value=1e-3,
                               warmup_steps=500, decay_steps=10_000)

    optimizer = get_optimizer("adamw",
                               learning_rate=schedule,
                               weight_decay=1e-4)

    # or with a fixed float lr
    optimizer = get_optimizer("adam", learning_rate=1e-3)

Optimizers
----------
ADAM            Adam (Kingma & Ba 2015)
ADAMW           Adam with decoupled weight decay (Loshchilov & Hutter 2019)
SGD             SGD with optional momentum and Nesterov
RMSPROP         RMSProp
LBFGS           L-BFGS — quasi-Newton (registered; requires specialised
                train_step; see LBFGS note below)

Schedules
---------
CONSTANT            Fixed value — no decay
COSINE_DECAY        Cosine annealing to near-zero
COSINE_ONECYCLE     One-cycle cosine (Smith 2019)
EXPONENTIAL_DECAY   Continuous or staircase exponential decay
POLYNOMIAL          Polynomial interpolation from init to end value
SGDR                SGD with warm restarts (Loshchilov & Hutter 2017)
WARMUP_CONSTANT     Linear warmup then constant (no decay)
WARMUP_COSINE       Linear warmup then cosine decay
WARMUP_EXPONENTIAL  Linear warmup then exponential decay

LBFGS note
----------
optax.lbfgs is a quasi-Newton method. Its optimizer.update() requires the
loss function itself (for line search), so the standard apply_gradients()
path in the trainer does not work with it. A dedicated train_step branch
is needed and will be added to trainer.py when quasi-Newton support lands.
get_optimizer("LBFGS", ...) raises NotImplementedError until then.
"""

from __future__ import annotations

from typing import Union

import optax

from utils.registry import Registry

# Schedule type alias: int -> float
Schedule = Union[float, optax.Schedule]


# ---------------------------------------------------------------------------
# Optimizer registry
# ---------------------------------------------------------------------------

OPTIMIZERS = Registry("Optimizer")
register_optimizer = OPTIMIZERS.register


def get_optimizer(
    name: str,
    learning_rate: Schedule,
    **kwargs,
) -> optax.GradientTransformation:
    """Instantiate a registered optimizer.

    Case-insensitive; ``learning_rate`` is forwarded to the factory and other
    unknown kwargs are dropped with a UserWarning (see utils.registry).

    Raises
    ------
    ValueError
        If the name is not registered.
    NotImplementedError
        If the optimizer requires a specialised train_step (e.g. LBFGS).

    Example
    -------
    >>> # noinspection SpellCheckingInspection
opt = get_optimizer("adamw", learning_rate=1e-3, weight_decay=1e-4)
    >>> opt = get_optimizer("adam", learning_rate=get_scheduler("cosine_decay",
    ...                             init_value=1e-3, decay_steps=10_000))
    """
    return OPTIMIZERS.get(name, learning_rate=learning_rate, **kwargs)


def list_optimizers() -> dict[str, str]:
    """Map of registered optimizer name -> description."""
    return OPTIMIZERS.describe()


# ---------------------------------------------------------------------------
# Scheduler registry
# ---------------------------------------------------------------------------

SCHEDULERS = Registry("Scheduler")
register_scheduler = SCHEDULERS.register


def get_scheduler(name: str, **kwargs) -> optax.Schedule:
    """Instantiate a registered learning-rate schedule (case-insensitive).

    Unknown kwargs are dropped with a UserWarning (see utils.registry); raises
    ValueError for an unknown name.

    Example
    -------
    >>> schedule = get_scheduler("warmup_cosine",
    ...                           init_value=0.0, peak_value=1e-3,
    ...                           warmup_steps=500, decay_steps=10_000)
    >>> schedule(0)    # lr at step 0
    0.0
    >>> schedule(500)  # lr at peak
    0.001
    """
    return SCHEDULERS.get(name, **kwargs)


def list_schedulers() -> dict[str, str]:
    """Map of registered scheduler name -> description."""
    return SCHEDULERS.describe()


# ---------------------------------------------------------------------------
# Registered optimizers
# ---------------------------------------------------------------------------

@register_optimizer(
    "ADAM",
    description="Adam (Kingma & Ba 2015)",
)
def _adam(
    learning_rate: Schedule,
    b1:  float = 0.9,
    b2:  float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    return optax.adam(learning_rate, b1=b1, b2=b2, eps=eps)


@register_optimizer(
    "ADAMW",
    description="Adam with decoupled weight decay (Loshchilov & Hutter 2019)",
)
def _adamw(
    learning_rate:  Schedule,
    b1:             float = 0.9,
    b2:             float = 0.999,
    eps:            float = 1e-8,
    weight_decay:   float = 1e-4,
) -> optax.GradientTransformation:
    return optax.adamw(
        learning_rate,
        b1=b1, b2=b2, eps=eps,
        weight_decay=weight_decay,
    )


@register_optimizer(
    "SGD",
    description="SGD with optional momentum and Nesterov",
)
def _sgd(
    learning_rate: Schedule,
    momentum:      float = 0.0,
    nesterov:      bool  = False,
) -> optax.GradientTransformation:
    return optax.sgd(learning_rate, momentum=momentum, nesterov=nesterov)


@register_optimizer(
    "RMSPROP",
    description="RMSProp",
)
def _rmsprop(
    learning_rate: Schedule,
    decay:         float = 0.9,
    eps:           float = 1e-8,
    momentum:      float = 0.0,
    centered:      bool  = False,
) -> optax.GradientTransformation:
    return optax.rmsprop(
        learning_rate,
        decay=decay, eps=eps,
        momentum=momentum, centered=centered,
    )


@register_optimizer(
    "LBFGS",
    description=(
        "L-BFGS — quasi-Newton; requires specialised train_step "
        "(see LBFGS note in module docstring)"
    ),
)
def _lbfgs(
    learning_rate: Schedule,  # noqa: ARG001 — kept for registry interface consistency
    memory_size:        int  = 10,
    scale_init_precond: bool = True,
) -> optax.GradientTransformation:
    raise NotImplementedError(
        "LBFGS requires optimizer.update() to receive value_fn for line "
        "search and cannot use the standard apply_gradients() path. "
        "A dedicated quasi-Newton train_step will be added to trainer.py."
    )


# ---------------------------------------------------------------------------
# Registered schedules
# ---------------------------------------------------------------------------

@register_scheduler(
    "CONSTANT",
    description="Fixed learning rate — no decay",
)
def _constant(value: float) -> optax.Schedule:
    return optax.constant_schedule(value)


@register_scheduler(
    "COSINE_DECAY",
    description="Cosine annealing to near-zero (Loshchilov & Hutter 2017)",
)
def _cosine_decay(
    init_value:  float,
    decay_steps: int,
    alpha:       float = 0.0,
) -> optax.Schedule:
    return optax.cosine_decay_schedule(
        init_value, decay_steps, alpha=alpha
    )


@register_scheduler(
    "COSINE_ONECYCLE",
    description="One-cycle cosine lr schedule (Smith 2019)",
)
def _cosine_onecycle(
    transition_steps:  int,
    peak_value:        float,
    pct_start:         float = 0.3,
    div_factor:        float = 25.0,
    final_div_factor:  float = 1e4,
) -> optax.Schedule:
    return optax.cosine_onecycle_schedule(
        transition_steps=transition_steps,
        peak_value=peak_value,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )


@register_scheduler(
    "EXPONENTIAL_DECAY",
    description="Continuous or staircase exponential decay",
)
def _exponential_decay(
    init_value:        float,
    transition_steps:  int,
    decay_rate:        float,
    transition_begin:  int   = 0,
    staircase:         bool  = False,
    end_value:         float | None = None,
) -> optax.Schedule:
    kwargs = dict(
        init_value=init_value,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        transition_begin=transition_begin,
        staircase=staircase,
    )
    if end_value is not None:
        kwargs["end_value"] = end_value
    return optax.exponential_decay(**kwargs)


@register_scheduler(
    "POLYNOMIAL",
    description="Polynomial interpolation from init_value to end_value",
)
def _polynomial(
    init_value:       float,
    end_value:        float,
    power:            float,
    transition_steps: int,
) -> optax.Schedule:
    return optax.polynomial_schedule(
        init_value=init_value,
        end_value=end_value,
        power=power,
        transition_steps=transition_steps,
    )


@register_scheduler(
    "SGDR",
    description=(
        "SGD with warm restarts (Loshchilov & Hutter 2017). "
        "cosine_kwargs is a list of dicts, one per restart cycle — "
        "each dict passed directly to cosine_decay_schedule."
    ),
)
def _sgdr(cosine_kwargs: list[dict]) -> optax.Schedule:
    return optax.sgdr_schedule(cosine_kwargs)


@register_scheduler(
    "WARMUP_CONSTANT",
    description="Linear warmup then constant lr (no decay)",
)
def _warmup_constant(
    init_value:   float,
    peak_value:   float,
    warmup_steps: int,
) -> optax.Schedule:
    return optax.warmup_constant_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
    )


@register_scheduler(
    "WARMUP_COSINE",
    description="Linear warmup then cosine decay",
)
def _warmup_cosine(
    init_value:   float,
    peak_value:   float,
    warmup_steps: int,
    decay_steps:  int,
    end_value:    float = 0.0,
) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_value,
    )


@register_scheduler(
    "WARMUP_EXPONENTIAL",
    description="Linear warmup then exponential decay",
)
def _warmup_exponential(
    init_value:        float,
    peak_value:        float,
    warmup_steps:      int,
    transition_steps:  int,
    decay_rate:        float,
    transition_begin:  int   = 0,
    staircase:         bool  = False,
    end_value:         float | None = None,
) -> optax.Schedule:
    kwargs = dict(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        transition_begin=transition_begin,
        staircase=staircase,
    )
    if end_value is not None:
        kwargs["end_value"] = end_value
    return optax.warmup_exponential_decay_schedule(**kwargs)
