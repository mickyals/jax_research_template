"""
experiments/tc_perceiver_io/data/inputs.py

InputSpec — declarative description of the model inputs (the input-side mirror
of data/targets.py TargetSpec).

An InputSpec bundles everything that defines what enters the encoder: the
observation variables, their normalisation, the coordinate encoding, and the
field-of-view bounds. The swappable pieces (normalisation, coordinate
encoding, derived variables) live in data/transforms/ as small registries; the
spec selects them by name and exposes the resolved callables. The dataset
orchestrates assembly, delegating each swappable step to the spec.

Built from config (``resolve_input``), not from a preset registry: unlike a
target (pure logic), an input's bounds and FOV are data-derived numbers that
belong in the YAML ``data:`` block. The encoder stays input-agnostic — its
hyperparameters do not change with the InputSpec.

Missingness (the isfinite mask, NaN→0, re-zero after normalisation) is NOT
part of the spec — it is invariant assembly logic and stays in dataset.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from experiments.tc_perceiver_io.data.sources.insitu_land import DEFAULT_OBS_VARS
from utils.normalise import (
    NORMALISERS, get_normaliser,
)
from experiments.tc_perceiver_io.data.transforms.encoding import (
    COORD_ENCODERS, get_coord_encoder, get_coord_decoder,
)
from experiments.tc_perceiver_io.data.transforms.derived import (
    resolve_fetch_vars,
)


@dataclass(frozen=True)
class InputSpec:
    """One input configuration — see module docstring.

    Parameters
    ----------
    name : str
        Descriptive label (diagnostics / repr only).
    obs_vars : tuple[str, ...]
        Model-facing observation variables, in order. May include derived
        names (see data/transforms/derived.py); their source columns are
        fetched automatically. F = len(obs_vars).
    normalisation : str
        Normaliser name (utils/normalise.py registry). Applied to
        station_obs when obs_bounds is provided.
    obs_bounds : dict[str, (a, b)] or None
        Per-variable bounds keyed by obs_vars: (min, max) for the minmax
        modes, (mean, std) for 'standardise'. None → obs values are not
        normalised.
    location_encoding : str
        Coordinate encoder name (data/transforms/encoding.py).
    fov_lat, fov_lon : (min, max)
        Field-of-view bounds in degrees (used by 'domain' encoding and by the
        datamodule's background sampling).
    """
    name:              str = 'land_insitu'
    obs_vars:          tuple[str, ...] = tuple(DEFAULT_OBS_VARS)
    normalisation:     str = 'minmax_01'
    obs_bounds:        Optional[dict[str, tuple[float, float]]] = None
    location_encoding: str = 'unit_circle'
    fov_lat:           tuple[float, float] = (0.0, 30.0)
    fov_lon:           tuple[float, float] = (-100.0, -45.0)

    def __post_init__(self) -> None:
        if not self.obs_vars:
            raise ValueError(f"input spec {self.name!r}: obs_vars must be non-empty.")
        if self.normalisation not in NORMALISERS:
            raise ValueError(
                f"obs_normalisation {self.normalisation!r} is not a registered "
                f"normaliser. Available: {', '.join(NORMALISERS.names())}."
            )
        if self.location_encoding not in COORD_ENCODERS:
            raise ValueError(
                f"location_encoding {self.location_encoding!r} is not a registered "
                f"coordinate encoder. Available: {', '.join(COORD_ENCODERS.names())}."
            )

    # -- resolved transforms (input-side analogue of TargetSpec.labeller) ---

    @property
    def feature_dim(self) -> int:
        """Number of model-facing observation variables (F)."""
        return len(self.obs_vars)

    @property
    def fetch_vars(self) -> list[str]:
        """Source columns to fetch (derived names → their source columns)."""
        return resolve_fetch_vars(self.obs_vars)

    @property
    def normaliser(self):
        """The resolved normaliser callable ``fn(obs_safe, lo, hi)``."""
        return get_normaliser(self.normalisation)

    @property
    def coord_encoder(self):
        """The resolved sample-level coordinate encoder callable."""
        return get_coord_encoder(self.location_encoding)

    @property
    def coord_decoder(self):
        """The resolved coordinate decoder callable (plotting/diagnostics)."""
        return get_coord_decoder(self.location_encoding)

    def bounds_arrays(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Per-variable (lo, hi) float32 arrays aligned to obs_vars.

        Returns (None, None) when no obs_bounds are set. For minmax modes the
        arrays are (min, max); for 'standardize' they are (mean, std).
        """
        if self.obs_bounds is None:
            return None, None
        lo = np.array([self.obs_bounds[v][0] for v in self.obs_vars], dtype=np.float32)
        hi = np.array([self.obs_bounds[v][1] for v in self.obs_vars], dtype=np.float32)
        return lo, hi


def resolve_input(config: dict) -> InputSpec:
    """Build the InputSpec from the ``data:`` config block.

    Reads the same keys the datamodule already exposes: ``obs_vars``
    (None → DEFAULT_OBS_VARS), ``obs_normalisation``, ``obs_bounds``,
    ``location_encoding``, ``fov_lat``, ``fov_lon``.
    """
    obs_vars   = config.get('obs_vars') or list(DEFAULT_OBS_VARS)
    bounds_raw = config.get('obs_bounds')
    obs_bounds = (
        {k: tuple(v) for k, v in bounds_raw.items()}
        if bounds_raw is not None else None
    )
    return InputSpec(
        name              = config.get('input_name', 'land_insitu'),
        obs_vars          = tuple(obs_vars),
        normalisation     = config.get('obs_normalisation', 'minmax_01'),
        obs_bounds        = obs_bounds,
        location_encoding = config.get('location_encoding', 'unit_circle'),
        fov_lat           = tuple(config.get('fov_lat', (0.0, 30.0))),
        fov_lon           = tuple(config.get('fov_lon', (-100.0, -45.0))),
    )
