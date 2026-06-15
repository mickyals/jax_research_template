"""
Tests for training/class_weights.py.

Coverage
--------
TestClassWeightsFromCounts
    none -> all ones; inverse/sqrt/effective/median rank rarer classes higher;
    sqrt compresses spread vs inverse; zero-count classes stay 1.0 and are
    excluded from normalization; normalize gives mean 1 over present classes;
    unknown scheme raises; effective_number beta monotonicity.
"""

import numpy as np
import pytest

from datasets.class_weights import SCHEMES, class_weights_from_counts


COUNTS = np.array([0, 72, 1356, 96, 808, 2935, 738, 337, 288, 271, 54], dtype=float)
#                  ^class 0 (background) absent from the TC count table


class TestClassWeightsFromCounts:

    def test_none_all_ones(self):
        w = class_weights_from_counts(COUNTS, scheme="none")
        assert np.allclose(w, 1.0)

    def test_zero_count_class_stays_one(self):
        for scheme in SCHEMES:
            w = class_weights_from_counts(COUNTS, scheme=scheme)
            assert w[0] == pytest.approx(1.0), scheme

    @pytest.mark.parametrize("scheme", ["inverse_freq", "sqrt_inverse_freq",
                                        "effective_number", "median_freq"])
    def test_rarer_class_weighted_higher(self, scheme):
        w = class_weights_from_counts(COUNTS, scheme=scheme)
        # class 10 (count 54) is rarer than class 5 (count 2935)
        assert w[10] > w[5]

    def test_normalize_mean_one_over_present(self):
        w = class_weights_from_counts(COUNTS, scheme="inverse_freq", normalize=True)
        present = COUNTS > 0
        assert w[present].mean() == pytest.approx(1.0)

    def test_sqrt_compresses_spread(self):
        inv  = class_weights_from_counts(COUNTS, scheme="inverse_freq")
        sqrt = class_weights_from_counts(COUNTS, scheme="sqrt_inverse_freq")
        present = COUNTS > 0
        spread_inv  = inv[present].max()  / inv[present].min()
        spread_sqrt = sqrt[present].max() / sqrt[present].min()
        assert spread_sqrt < spread_inv

    def test_median_freq_matches_formula(self):
        w = class_weights_from_counts(COUNTS, scheme="median_freq", normalize=False)
        present = COUNTS > 0
        med = np.median(COUNTS[present])
        assert np.allclose(w[present], med / COUNTS[present])

    def test_effective_number_higher_beta_more_aggressive(self):
        w_lo = class_weights_from_counts(COUNTS, scheme="effective_number", beta=0.99)
        w_hi = class_weights_from_counts(COUNTS, scheme="effective_number", beta=0.9999)
        present = COUNTS > 0
        spread_lo = w_lo[present].max() / w_lo[present].min()
        spread_hi = w_hi[present].max() / w_hi[present].min()
        assert spread_hi > spread_lo

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown class_weight scheme"):
            class_weights_from_counts(COUNTS, scheme="bogus")

    def test_all_present_no_special_case(self):
        counts = np.array([10, 20, 40], dtype=float)
        w = class_weights_from_counts(counts, scheme="inverse_freq")
        assert w.shape == (3,)
        assert w.mean() == pytest.approx(1.0)
        assert w[0] > w[2]   # rarer -> higher
