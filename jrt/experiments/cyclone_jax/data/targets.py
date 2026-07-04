"""
experiments/cyclone_jax/data/targets.py

TargetSpec — the declarative TARGET side of a sample (mirror of inputs.py).

y is a named dict, never a positional block:

    y = {'target',                  # what the loss consumes
         'sid', 'lat', 'lon', 'time'}   # fix identity, for eval/plots

Identity fields are the fix's OWN coordinates — evaluation metadata, never
model input (the leakage allowlist: the model sees x only). Batching maps
y['target'] -> batch['y'] and identity -> batch['meta'].

The selected variable comes from the data config. v1 target: 'usa_sshs'
categorical over class_set (remapped 0..8 scheme, see build.remap_sshs).
Continuous targets (usa_wind etc.) are reserved — resolve raises until the
regression head exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.cyclone_jax.data.sources.library import (
    CYC_SSHS, CYC_TARGETS, SSHS_MAX, SSHS_MIN,
)

# Remapped 0..8 scheme (build.remap_sshs), human names for reports/plots.
SSHS_NAMES = ('Post-Tropical', 'Disturbance', 'Depression', 'Tropical Storm',
              'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5')

DEFAULT_CLASS_SET = (3, 4, 5, 6, 7, 8)      # TS+; no background class


@dataclass(frozen=True)
class TargetSpec:
    """Resolved target policy: the supervised variable and its label space.

    Built by resolve_target from the data config — construct directly only
    in tests.
    """
    variable:  str                       # CYC_SSHS or a CYC_TARGETS column
    kind:      str                       # 'categorical' | 'continuous'
    class_set: tuple[int, ...] | None    # remapped categories, ascending

    def __post_init__(self):
        if self.kind not in ('categorical', 'continuous'):
            raise ValueError(f"kind must be 'categorical' or 'continuous', "
                             f"got {self.kind!r}")
        if self.kind == 'continuous':
            raise NotImplementedError(
                f"continuous target {self.variable!r} is reserved — the "
                f"regression head is not built yet.")
        if not self.class_set:
            raise ValueError("categorical target requires a class_set.")
        cs = self.class_set
        if list(cs) != sorted(set(cs)):
            raise ValueError(f"class_set must be strictly ascending, got {cs}.")
        if cs[0] < SSHS_MIN or cs[-1] > SSHS_MAX:
            raise ValueError(f"class_set {cs} outside the remapped scheme "
                             f"[{SSHS_MIN}, {SSHS_MAX}].")

    @property
    def n_classes(self) -> int:
        return len(self.class_set)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(SSHS_NAMES[c - SSHS_MIN] for c in self.class_set)

    # ------------------------------------------------------------------

    def label(self, value) -> int:
        """One remapped category -> class index (position in class_set)."""
        c = int(value)
        try:
            return self.class_set.index(c)
        except ValueError:
            raise ValueError(f"category {c} not in class_set "
                             f"{self.class_set}.") from None

    def labels(self, values) -> np.ndarray:
        """Vectorised label mapping (int32) — for splits/stratification."""
        cs = np.asarray(self.class_set)
        v = np.rint(np.asarray(values)).astype(np.int64)
        idx = np.searchsorted(cs, v)
        bad = (idx >= len(cs)) | (cs[np.minimum(idx, len(cs) - 1)] != v)
        if bad.any():
            raise ValueError(f"categories {sorted(set(v[bad].tolist()))} not "
                             f"in class_set {self.class_set}.")
        return idx.astype(np.int32)

    def build_y(self, fixes, i) -> dict:
        """y dict for fix i of a get_fixes table (see module docstring)."""
        return {
            'target': np.int32(self.label(fixes[CYC_SSHS][i])),
            'sid':    str(fixes['sid'][i]),
            'lat':    np.float32(fixes['lat'][i]),
            'lon':    np.float32(fixes['lon'][i]),
            'time':   fixes['time'][i],
        }


def resolve_target(config: dict) -> TargetSpec:
    """Build the TargetSpec from the data config block.

    Keys read: target (default usa_sshs), class_set (default TS+ 3..8).
    Variables other than usa_sshs must come from the CYC_TARGETS allowlist
    and are continuous (reserved).
    """
    variable = config.get('target', CYC_SSHS)
    if variable == CYC_SSHS:
        kind = 'categorical'
        class_set = tuple(int(c) for c in
                          config.get('class_set', DEFAULT_CLASS_SET))
    elif variable in CYC_TARGETS:
        kind, class_set = 'continuous', None
    else:
        raise ValueError(f"unknown target {variable!r} — must be "
                         f"{CYC_SSHS!r} or one of CYC_TARGETS.")
    return TargetSpec(variable=variable, kind=kind, class_set=class_set)
