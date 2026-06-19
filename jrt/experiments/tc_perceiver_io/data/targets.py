"""
experiments/tc_perceiver_io/data/targets.py

TargetSpec — declarative description of a prediction target.

A TargetSpec maps the ``data.target`` config key to everything that depends on
the target: the per-sample label builder, the head size + class names, the
default loss, and whether background samples participate. Switching targets — or
attaching a fresh probe head to a frozen encoder — is therefore a config change,
not a code change. The encoder stays target-agnostic (plan-encoder-probing-
rescope r3/r4).

Targets are declared EXPLICITLY in ``TARGET_SCHEMA``, never inferred from a
column's numpy dtype: USA_SSHS is integer-coded but ordinal, USA_WIND is
continuous, BASIN is nominal-string — dtype cannot tell these apart.

Status
------
``kind='nominal'`` (single-label classification) is wired end to end.
``kind='continuous'`` (regression) is reserved: wiring it means a Dense(1) head,
an mse loss branch, regression metrics, target normalisation, and background
exclusion in the sampler (continuous intensity is undefined for a no-storm
sample). Declaring a continuous TargetSpec raises until that pass lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from experiments.tc_perceiver_io.data.sources.ibtracs import (
    CLASS_NAMES, N_CLASSES, status_sshs_to_class,
)

NOMINAL    = 'nominal'
CONTINUOUS = 'continuous'


@dataclass(frozen=True)
class TargetSpec:
    """One prediction target — see module docstring.

    Parameters
    ----------
    name : str
        Target key (matches the data.target config value).
    kind : {'nominal', 'continuous'}
        'nominal' = single-label classification (wired). 'continuous' =
        regression (reserved; construction raises until wired).
    n_classes : int, optional
        Number of classes (nominal only).
    class_names : list[str], optional
        Display names, index = class id (nominal only).
    labeller : callable, optional
        ``(ibtracs, idx) -> int | None`` — the per-row label, or None to drop
        the row. Required for nominal targets.
    include_background : bool
        Whether no-storm background samples participate (class 0 for nominal).
    column, bounds, units :
        Continuous-target metadata (reserved, unused while continuous is
        deferred).
    """
    name:               str
    kind:               str = NOMINAL
    # --- nominal ---
    n_classes:          Optional[int] = None
    class_names:        Optional[list[str]] = None
    labeller:           Optional[Callable[..., Optional[int]]] = None
    include_background: bool = True
    # --- continuous (reserved) ---
    column:             Optional[str] = None
    bounds:             Optional[tuple[float, float]] = None
    units:              Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind == NOMINAL:
            if self.n_classes is None or self.labeller is None:
                raise ValueError(
                    f"nominal target {self.name!r} needs n_classes and labeller."
                )
        elif self.kind == CONTINUOUS:
            raise NotImplementedError(
                f"continuous target {self.name!r}: regression head/loss/metrics/"
                "sampling are not wired yet (plan-encoder-probing-rescope r3 — "
                "continuous deferred). Add the regression branch before declaring it."
            )
        else:
            raise ValueError(
                f"unknown target kind {self.kind!r} for {self.name!r} "
                f"(expected {NOMINAL!r} or {CONTINUOUS!r})."
            )

    @property
    def loss(self) -> str:
        """Default training loss for this target's kind."""
        return 'cross_entropy' if self.kind == NOMINAL else 'mse'


# ---------------------------------------------------------------------------
# Label builders
# ---------------------------------------------------------------------------

def _organisation_label(ibtracs, idx) -> Optional[int]:
    """9-class ordinal organisation label (STATUS-driven, see ibtracs.py)."""
    return status_sshs_to_class(
        str(ibtracs['USA_STATUS'][idx]),
        float(ibtracs['USA_SSHS'][idx]),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TARGET_SCHEMA: dict[str, TargetSpec] = {
    'organisation': TargetSpec(
        name               = 'organisation',
        kind               = NOMINAL,
        n_classes          = N_CLASSES,
        class_names        = CLASS_NAMES,
        labeller           = _organisation_label,
        include_background = True,
    ),
}

DEFAULT_TARGET = 'organisation'


def resolve_target(name: Optional[str] = None) -> TargetSpec:
    """Resolve a ``data.target`` name to its TargetSpec. None → DEFAULT_TARGET."""
    key = name or DEFAULT_TARGET
    if key not in TARGET_SCHEMA:
        raise ValueError(
            f"unknown target {key!r}. Available: {sorted(TARGET_SCHEMA)}. "
            "Declare new targets in data/targets.py TARGET_SCHEMA."
        )
    return TARGET_SCHEMA[key]
