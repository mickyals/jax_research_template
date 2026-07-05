"""
Tests for experiments/cyclone_jax/data/sources/sources.py — build-time
transforms that must be exactly right: the SSHS remap table, the
subtropical-by-wind folding, the hypsometric cross-fill (direction +
no-chaining), and the category index.
"""

import numpy as np
import pytest

from utils.geoscience.met_conversions import R34_MS_THRESHOLD, kt_to_ms

from experiments.cyclone_jax.data.sources.build import (
    G,
    N_CAT,
    RD,
    STORM_KT_COLS,
    STORM_MB_COLS,
    STORM_NMILE_COLS,
    _hypsometric_fill,
    build_category_index,
    convert_storm_units,
    remap_sshs,
)


# ---------------------------------------------------------------------------
# SSHS remap
# ---------------------------------------------------------------------------

class TestRemapSSHS:

    def test_direct_mapping_table(self):
        old = np.array([-4, -3, -1, 0, 1, 2, 3, 4, 5], np.float32)
        wind = np.full(9, np.nan, np.float32)          # wind unused off -2
        new = remap_sshs(old, wind)
        np.testing.assert_array_equal(new, [0, 1, 2, 3, 4, 5, 6, 7, 8])

    @pytest.mark.parametrize('wind, expected', [
        (20.0, 2),    # < 34 kt  -> depression
        (34.0, 3),    # TS band
        (63.9, 3),
        (64.0, 4),    # cat 1
        (83.0, 5),    # cat 2
        (96.0, 6),    # cat 3
        (113.0, 7),   # cat 4
        (137.0, 8),   # cat 5
        (150.0, 8),
    ])
    def test_subtropical_folded_by_wind(self, wind, expected):
        new = remap_sshs(np.array([-2.0]), np.array([wind]))
        assert new[0] == expected

    def test_subtropical_nan_wind_stays_nan(self):
        new = remap_sshs(np.array([-2.0]), np.array([np.nan]))
        assert np.isnan(new[0])

    def test_unmapped_code_stays_nan(self):
        new = remap_sshs(np.array([-5.0, np.nan]), np.array([50.0, 50.0]))
        assert np.all(np.isnan(new))


# ---------------------------------------------------------------------------
# Storm unit conversion (kt/mb/nmile -> SI, strictly after the remap)
# ---------------------------------------------------------------------------

class TestConvertStormUnits:

    def _cols(self):
        out = {c: np.float32([10.0, np.nan]) for c in
               STORM_KT_COLS + STORM_MB_COLS + STORM_NMILE_COLS}
        out['usa_sshs'] = np.float32([3.0, 3.0])       # untouched control
        return out

    def test_factors(self):
        out = convert_storm_units(self._cols())
        assert out['usa_wind'][0] == pytest.approx(10 * 0.514444)
        assert out['usa_pres'][0] == pytest.approx(1000.0)     # mb -> Pa
        assert out['level'][0] == pytest.approx(1000.0)
        assert out['usa_rmw'][0] == pytest.approx(18520.0)     # nmi -> m
        assert out['usa_r34_NE'][0] == pytest.approx(18520.0)

    def test_nan_passthrough_and_dtype(self):
        out = convert_storm_units(self._cols())
        for c in STORM_KT_COLS + STORM_MB_COLS + STORM_NMILE_COLS:
            assert np.isnan(out[c][1]) and out[c].dtype == np.float32

    def test_category_column_untouched(self):
        out = convert_storm_units(self._cols())
        np.testing.assert_array_equal(out['usa_sshs'], [3.0, 3.0])

    def test_remap_before_convert_order(self):
        """The build-order contract: remap sees kt, conversion follows.
        A 40-kt subtropical fix is a tropical storm (class 3) under the
        kt thresholds; converted first (20.6 m/s) it would land in the
        depression band — the wrong class."""
        wind_kt = np.float32([40.0])
        assert remap_sshs(np.float32([-2.0]), wind_kt)[0] == 3
        assert kt_to_ms(wind_kt)[0] < 34            # would misclassify

    def test_threshold_equivalence(self):
        """34 kt converts to exactly the R34 m/s threshold constant —
        remapped classes and converted winds stay consistent."""
        assert kt_to_ms(34.0) == pytest.approx(R34_MS_THRESHOLD)


# ---------------------------------------------------------------------------
# Hypsometric cross-fill
# ---------------------------------------------------------------------------

class TestHypsometricFill:

    def test_slp_derived_from_station_p(self):
        # station at 100 m, 300 K: p_sl = p_sfc * exp(z / H), H = Rd*T/g
        sp = np.array([100000.0], np.float32)
        sl = np.array([np.nan], np.float32)
        T = np.array([300.0], np.float32)
        z = np.array([100.0], np.float32)
        sp2, sl2, n_sp, n_sl = _hypsometric_fill(sp, sl, T, z)
        H = RD * 300.0 / G
        assert n_sl == 1 and n_sp == 0
        np.testing.assert_allclose(sl2[0], 100000.0 * np.exp(100.0 / H),
                                   rtol=1e-6)

    def test_station_p_derived_from_slp(self):
        sp = np.array([np.nan], np.float32)
        sl = np.array([101325.0], np.float32)
        T = np.array([288.0], np.float32)
        z = np.array([500.0], np.float32)
        sp2, sl2, n_sp, n_sl = _hypsometric_fill(sp, sl, T, z)
        H = RD * 288.0 / G
        assert n_sp == 1 and n_sl == 0
        np.testing.assert_allclose(sp2[0], 101325.0 * np.exp(-500.0 / H),
                                   rtol=1e-6)
        # sea level above station for z > 0
        assert sl2[0] > sp2[0]

    def test_no_chaining_when_both_missing(self):
        """A derived value must never seed the complementary derivation."""
        sp = np.array([np.nan], np.float32)
        sl = np.array([np.nan], np.float32)
        T = np.array([290.0], np.float32)
        z = np.array([50.0], np.float32)
        sp2, sl2, n_sp, n_sl = _hypsometric_fill(sp, sl, T, z)
        assert n_sp == 0 and n_sl == 0
        assert np.isnan(sp2[0]) and np.isnan(sl2[0])

    def test_requires_temp_and_elevation(self):
        sp = np.array([np.nan, np.nan], np.float32)
        sl = np.array([101000.0, 101000.0], np.float32)
        T = np.array([np.nan, 290.0], np.float32)      # first lacks temp
        z = np.array([100.0, np.nan], np.float32)      # second lacks elev
        sp2, _, n_sp, _ = _hypsometric_fill(sp, sl, T, z)
        assert n_sp == 0
        assert np.all(np.isnan(sp2))

    def test_originals_untouched(self):
        sp = np.array([99000.0], np.float32)
        sl = np.array([100500.0], np.float32)
        T = np.array([295.0], np.float32)
        z = np.array([120.0], np.float32)
        sp2, sl2, n_sp, n_sl = _hypsometric_fill(sp, sl, T, z)
        assert n_sp == n_sl == 0
        assert sp2[0] == sp[0] and sl2[0] == sl[0]


# ---------------------------------------------------------------------------
# Category index
# ---------------------------------------------------------------------------

class TestCategoryIndex:

    def _obs(self):
        return {'usa_sshs': np.array(
            [3, 3, 8, np.nan, 0, 3, 4, 8], np.float32)}

    def test_counts_and_offsets(self):
        cat_order, cat_offsets = build_category_index(self._obs())
        counts = np.diff(cat_offsets)
        assert len(cat_offsets) == N_CAT + 1
        assert counts[0] == 1 and counts[3] == 3
        assert counts[4] == 1 and counts[8] == 2
        assert counts.sum() == 7                       # NaN row excluded

    def test_time_order_within_bucket(self):
        cat_order, cat_offsets = build_category_index(self._obs())
        b3 = cat_order[cat_offsets[3]:cat_offsets[4]]
        assert np.all(np.diff(b3) > 0)                 # ascending row = time

    def test_exclude_mask(self):
        obs = self._obs()
        exclude = np.zeros(8, bool)
        exclude[0] = True                              # drop one cat-3 row
        cat_order, cat_offsets = build_category_index(obs, exclude=exclude)
        counts = np.diff(cat_offsets)
        assert counts[3] == 2
        assert 0 not in cat_order
