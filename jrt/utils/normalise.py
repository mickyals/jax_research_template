"""
utils/normalise.py

Numpy normalisation mechanics, shared across experiments (promoted from
experiments/tc_perceiver_io/data/transforms/normalise.py, 2026-07-04 —
policy stays experiment-side, e.g. cyclone_jax data/normalise.py NormSpec).

Two pieces:

- NORMALISERS — pure scalings ``fn(values, lo, hi) -> normalised`` over an
  (N,) or (N, C) array with per-column (C,) bounds: (min, max) for the
  minmax modes, (mean, std) for 'standardise'. NaN-propagating — missing
  values stay NaN, so missingness masks built downstream are unaffected.
- StatsAccumulator / compute_stats — nan-aware per-column
  mean/std/min/max/count, single-shot or accumulated over chunks (float64
  internally), for computing normalisation stats from a training split.

This module is deliberately JAX-FREE (data loaders / multiprocess workers
import it). Device-side jax twins live in utils/jax_core/helpers.py
(standardise, minmax_norm — consumed by datasets/datamodule.py, which also
owns target denormalisation via DataModule.denormalise_targets).
"""

from __future__ import annotations

import numpy as np

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


# ---------------------------------------------------------------------------
# Nan-aware statistics (for computing bounds from a training split)
# ---------------------------------------------------------------------------

class StatsAccumulator:
    """Accumulate nan-aware per-column stats over chunks of samples.

    update() takes (n,) or (n, C) arrays; result() returns
    ``{'mean', 'std', 'min', 'max', 'count'}`` float64/int64 arrays of
    shape (C,) ((1,) for 1-D input). Columns never observed (count 0)
    yield NaN stats — callers decide the substitution policy.

    Accumulates count/sum/sumsq/min/max in float64, so chunked accumulation
    equals the single-shot computation over the concatenated data.
    """

    def __init__(self):
        self._count = None

    def update(self, values) -> None:
        v = np.asarray(values, np.float64)
        if v.ndim == 1:
            v = v[:, None]
        if v.ndim != 2:
            raise ValueError(f"expected (n,) or (n, C) values, "
                             f"got shape {v.shape}.")
        finite = np.isfinite(v)
        vz = np.where(finite, v, 0.0)
        if self._count is None:
            c = v.shape[1]
            self._count = np.zeros(c, np.int64)
            self._sum   = np.zeros(c, np.float64)
            self._sumsq = np.zeros(c, np.float64)
            self._min   = np.full(c, np.inf)
            self._max   = np.full(c, -np.inf)
        elif v.shape[1] != len(self._count):
            raise ValueError(f"chunk has {v.shape[1]} columns, "
                             f"accumulator has {len(self._count)}.")
        self._count += finite.sum(axis=0)
        self._sum   += vz.sum(axis=0)
        self._sumsq += (vz * vz).sum(axis=0)
        self._min = np.minimum(self._min, np.where(finite, v, np.inf).min(axis=0))
        self._max = np.maximum(self._max, np.where(finite, v, -np.inf).max(axis=0))

    def result(self) -> dict:
        if self._count is None:
            raise ValueError("no data accumulated — call update() first.")
        seen = self._count > 0
        n = np.maximum(self._count, 1)
        mean = np.where(seen, self._sum / n, np.nan)
        var = np.where(seen, self._sumsq / n - mean ** 2, np.nan)
        return {
            'mean':  mean,
            'std':   np.sqrt(np.maximum(var, 0.0)),
            'min':   np.where(seen, self._min, np.nan),
            'max':   np.where(seen, self._max, np.nan),
            'count': self._count.copy(),
        }


def compute_stats(values) -> dict:
    """Single-shot nan-aware per-column stats (see StatsAccumulator)."""
    acc = StatsAccumulator()
    acc.update(values)
    return acc.result()
