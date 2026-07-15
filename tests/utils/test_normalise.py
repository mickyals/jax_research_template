"""
Tests for utils/normalise.py — the promoted numpy normaliser registry
(ex tc_perceiver_io transforms) + nan-aware stats accumulation.
"""

import numpy as np
import pytest

from utils.normalise import (
    NORMALISERS, StatsAccumulator, compute_stats, get_normaliser,
)


# ---------------------------------------------------------------------------
# Registry (moved verbatim — behaviour pinned)
# ---------------------------------------------------------------------------

class TestNormalisers:

    def test_registered_names(self):
        assert {'MINMAX_01', 'MINMAX_11', 'STANDARDISE'} <= set(
            NORMALISERS.names())

    def test_minmax_01(self):
        fn = get_normaliser('minmax_01')
        out = fn(np.array([0.0, 5.0, 10.0]), 0.0, 10.0)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-9)

    def test_minmax_11(self):
        fn = get_normaliser('minmax_11')
        out = fn(np.array([0.0, 5.0, 10.0]), 0.0, 10.0)
        np.testing.assert_allclose(out, [-1.0, 0.0, 1.0], atol=1e-9)

    def test_standardise_bounds_are_mean_std(self):
        fn = get_normaliser('standardise')
        out = fn(np.array([1.0, 3.0]), 2.0, 1.0)     # (mean, std)
        np.testing.assert_allclose(out, [-1.0, 1.0], atol=1e-6)

    def test_per_column_bounds(self):
        fn = get_normaliser('standardise')
        vals = np.array([[10.0, 200.0], [20.0, 400.0]])
        out = fn(vals, np.array([15.0, 300.0]), np.array([5.0, 100.0]))
        np.testing.assert_allclose(out, [[-1.0, -1.0], [1.0, 1.0]],
                                   atol=1e-5)

    def test_nan_propagates(self):
        fn = get_normaliser('standardise')
        out = fn(np.array([np.nan, 1.0]), 0.0, 1.0)
        assert np.isnan(out[0]) and np.isfinite(out[1])

    def test_unknown_name_raises_with_available(self):
        with pytest.raises(ValueError, match='not a registered'):
            get_normaliser('zscore')


# ---------------------------------------------------------------------------
# Stats accumulation
# ---------------------------------------------------------------------------

class TestStats:

    def test_matches_numpy_nan_functions(self):
        rng = np.random.default_rng(0)
        v = rng.normal(100.0, 5.0, (200, 3))
        v[rng.random((200, 3)) < 0.2] = np.nan
        s = compute_stats(v)
        np.testing.assert_allclose(s['mean'], np.nanmean(v, 0), rtol=1e-12)
        np.testing.assert_allclose(s['std'], np.nanstd(v, 0), rtol=1e-9)
        np.testing.assert_allclose(s['min'], np.nanmin(v, 0))
        np.testing.assert_allclose(s['max'], np.nanmax(v, 0))
        np.testing.assert_array_equal(s['count'],
                                      np.isfinite(v).sum(0))

    def test_chunked_equals_single_shot(self):
        rng = np.random.default_rng(1)
        v = rng.normal(0.0, 1.0, (90, 2))
        v[::7] = np.nan
        acc = StatsAccumulator()
        for chunk in np.array_split(v, 4):
            acc.update(chunk)
        chunked, single = acc.result(), compute_stats(v)
        for k in ('mean', 'std', 'min', 'max', 'count'):
            np.testing.assert_allclose(chunked[k], single[k], rtol=1e-12)

    def test_1d_input_gives_single_column(self):
        s = compute_stats(np.array([1.0, 2.0, 3.0]))
        assert s['mean'].shape == (1,)
        np.testing.assert_allclose(s['mean'], [2.0])

    def test_never_observed_column_is_nan_with_zero_count(self):
        v = np.column_stack([np.arange(5.0), np.full(5, np.nan)])
        s = compute_stats(v)
        assert s['count'][1] == 0
        assert np.isnan(s['mean'][1]) and np.isnan(s['min'][1])

    def test_empty_accumulator_raises(self):
        with pytest.raises(ValueError, match='no data'):
            StatsAccumulator().result()

    def test_column_mismatch_raises(self):
        acc = StatsAccumulator()
        acc.update(np.ones((3, 2)))
        with pytest.raises(ValueError, match='columns'):
            acc.update(np.ones((3, 4)))
