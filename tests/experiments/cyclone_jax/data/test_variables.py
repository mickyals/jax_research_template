"""
Tests for data/variables.py — the catalogue must cover every column the
schemas actually store (units ruling 2026-07-05), and the display helper
must apply the one canonical mapping (m -> km, Pa -> hPa, else identity).
"""

import numpy as np
import pytest

from experiments.cyclone_jax.data.inputs import CHANNEL_ORDER, SOURCE_SCHEMAS
from experiments.cyclone_jax.data.sources.build import (
    LAND_VARS,
    MARINE_VARS,
    STORM_KT_COLS,
    STORM_MB_COLS,
    STORM_NMILE_COLS,
    UPPER_VARS,
)
from experiments.cyclone_jax.data.sources.library import CYC_TARGETS
from experiments.cyclone_jax.data.variables import (
    VARIABLES,
    VarInfo,
    column_meta,
    to_display,
)

# every column each builder stores beyond its VARS map (build.py out dicts)
COORD_COLS = {'report_timestamp', 'launch_timestamp', 'lat', 'lon', 'level',
              'elevation', 'platform_type'}
STORM_META_COLS = {'sid', 'name', 'season', 'basin', 'subbasin', 'iflag',
                   'usa_status', 'usa_sshs', 'usa_sshs_raw',
                   'is_subtropical', 'storm_dir'}


class TestCatalogueCoverage:

    def test_cdm_aliases_catalogued(self):
        for vars_map in (LAND_VARS, MARINE_VARS, UPPER_VARS):
            assert set(vars_map.values()) <= set(VARIABLES)

    def test_coordinate_columns_catalogued(self):
        assert COORD_COLS <= set(VARIABLES)

    def test_storm_columns_catalogued(self):
        stored = (set(STORM_KT_COLS) | set(STORM_MB_COLS)
                  | set(STORM_NMILE_COLS) | STORM_META_COLS)
        assert stored <= set(VARIABLES)

    def test_cyc_targets_catalogued(self):
        assert set(CYC_TARGETS) <= set(VARIABLES)

    def test_input_channels_catalogued(self):
        assert set(CHANNEL_ORDER) <= set(VARIABLES)
        for schema in SOURCE_SCHEMAS.values():
            assert set(schema.direct) <= set(VARIABLES)

    def test_converted_columns_have_si_units(self):
        for c in STORM_KT_COLS:
            assert VARIABLES[c].units == 'm s-1'
        for c in STORM_MB_COLS:
            assert VARIABLES[c].units == 'Pa'
        for c in STORM_NMILE_COLS:
            assert VARIABLES[c].units == 'm'

    def test_every_entry_documented(self):
        for name, info in VARIABLES.items():
            assert isinstance(info, VarInfo) and info.units and \
                info.description, name


class TestColumnMeta:

    def test_known_columns(self):
        meta = column_meta(['usa_wind', 'slp'])
        assert meta['usa_wind']['units'] == 'm s-1'
        assert meta['slp'] == {'units': 'Pa',
                               'description': VARIABLES['slp'].description}

    def test_unknown_column_gets_explicit_entry(self):
        meta = column_meta(['made_up'])
        assert meta['made_up']['units'] == 'unknown'


class TestToDisplay:

    def test_m_to_km(self):
        assert to_display(18520.0, 'm') == (pytest.approx(18.52), 'km')

    def test_pa_to_hpa(self):
        assert to_display(100000.0, 'Pa') == (pytest.approx(1000.0), 'hPa')

    def test_identity_otherwise(self):
        assert to_display(17.5, 'm s-1') == (17.5, 'm s-1')

    def test_arrays_pass_through(self):
        v, u = to_display(np.float32([1000.0, np.nan]), 'm')
        assert u == 'km' and v[0] == pytest.approx(1.0) and np.isnan(v[1])
