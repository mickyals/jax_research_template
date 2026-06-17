"""
experiments/sparse_obs_encoder/data/transforms/normalise.py

Observation normalisers — one of the swappable input transforms (see
data/inputs.py InputSpec). The ``obs_normalisation`` config key selects one
of these by name; the per-variable bounds come from ``obs_bounds``.

Each normaliser is a pure scaling ``fn(obs_safe, lo, hi) -> normalised`` over
the (N, F) station-observation matrix. ``lo``/``hi`` are the per-variable
(F,) bound arrays: (min, max) for the minmax modes, (mean, std) for
'standardise'. Missingness handling (re-zeroing positions that were missing
before scaling) stays structural in dataset.py — these functions do not see
the mask.
"""

from __future__ import annotations

from utils.registry import Registry

NORMALISERS = Registry("normaliser")


@NORMALISERS.register("minmax_01", "scale to [0, 1] using (min, max) bounds")
def _minmax_01(obs_safe, lo, hi):
    span = hi - lo
    return (obs_safe - lo) / (span + 1e-12)


@NORMALISERS.register("minmax_11", "scale to [-1, 1] using (min, max) bounds")
def _minmax_11(obs_safe, lo, hi):
    span = hi - lo
    return (obs_safe - lo) / (span + 1e-12) * 2.0 - 1.0


@NORMALISERS.register("standardise", "z-score using (mean, std) bounds")
def _standardise(obs_safe, lo, hi):
    # bounds are interpreted as (mean, std)
    return (obs_safe - lo) / (hi + 1e-8)


def get_normaliser(name: str):
    """Return the normaliser callable registered under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not a registered normaliser.
    """
    if name not in NORMALISERS:
        raise ValueError(
            f"obs_normalisation {name!r} is not a registered normaliser. "
            f"Available: {', '.join(NORMALISERS.names())}."
        )
    return NORMALISERS[name]
