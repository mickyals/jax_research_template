"""
experiments/cyclone_jax/data/variables.py

Variable catalogue: every column stored in the four volumes, with its
CANONICAL units and a one-line description. Plain python — importable by
the build-time layer (build.py, for meta.json sidecars) and the
train-time layer (figures, eval tables) alike.

Canonical storage units are SI (units ruling 2026-07-05): m, Pa, m/s, K,
degrees for angles. LISO/MISO/CUON deliver SI from the CDS CDM;
storm-arcanum (IBTrACS) is converted at build time (kt -> m/s, mb -> Pa,
nmile -> m; see build.convert_storm_units). Plots/tables that want
conventional units (km, hPa) go through `to_display` — storage never
deviates from SI.

Angle conventions are kept as-is: wind_dir/storm_dir are 0..360 bearings
(wind FROM / storm heading TOWARD); lat/lon are symmetric degrees
(-90..90 / -180..180). Any remap is an experiment-specific transformation
that declares its from-unit AND to-unit explicitly.

Only columns present in the codebase and the data files are catalogued.
"""

from __future__ import annotations

from typing import NamedTuple

from utils.geoscience.met_conversions import m_to_km, pa_to_hpa


class VarInfo(NamedTuple):
    units: str
    description: str


_QUADS = ('NE', 'SE', 'SW', 'NW')

VARIABLES: dict[str, VarInfo] = {
    # -- shared coordinates / spine -------------------------------------
    'report_timestamp': VarInfo('datetime64[ns]',
        'observation / fix time (UTC); the volume sort key'),
    'launch_timestamp': VarInfo('datetime64[ns]',
        'upper air only: actual radiosonde ascent second (provenance; '
        'report_timestamp carries the canonical record time)'),
    'lat': VarInfo('degrees_north', 'latitude, -90..90'),
    'lon': VarInfo('degrees_east', 'longitude, -180..180 symmetric'),
    'level': VarInfo('Pa',
        'vertical coordinate — pressure everywhere: land station '
        'pressure, marine SLP, upper z_coordinate, storm usa_pres '
        '(NaN allowed on the driver volume only)'),
    'elevation': VarInfo('m', 'land station height above sea level'),
    'platform_type': VarInfo('code',
        'CDM platform type code (marine / upper)'),

    # -- surface + upper observations (CDM, SI as delivered) ------------
    'station_pressure': VarInfo('Pa', 'air pressure at station elevation'),
    'slp': VarInfo('Pa', 'air pressure at sea level'),
    'air_temp': VarInfo('K', 'air temperature'),
    'dewpoint': VarInfo('K', 'dew point temperature'),
    'sst': VarInfo('K', 'water temperature (marine)'),
    'dpd': VarInfo('K', 'dew point depression (upper)'),
    'wind_dir': VarInfo('degrees',
        'wind FROM direction, 0..360 bearing (meteorological convention)'),
    'wind_speed': VarInfo('m s-1', 'wind speed'),
    'u_wind': VarInfo('m s-1',
        'eastward wind component (upper: stored; surface: derived from '
        'wind_speed + wind_dir)'),
    'v_wind': VarInfo('m s-1',
        'northward wind component (see u_wind)'),
    'geopot': VarInfo('m', 'geopotential height (upper)'),
    'rh': VarInfo('%', 'relative humidity (upper)'),
    'q': VarInfo('kg kg-1', 'specific humidity (upper)'),

    # -- storm-arcanum (IBTrACS v04r01, converted to SI at build) --------
    'sid': VarInfo('str', 'IBTrACS storm serial id'),
    'name': VarInfo('str', 'storm name'),
    'season': VarInfo('year', 'hurricane season'),
    'basin': VarInfo('str', 'IBTrACS basin code'),
    'subbasin': VarInfo('str', 'IBTrACS subbasin code'),
    'usa_wind': VarInfo('m s-1',
        'maximum sustained wind, US agencies (1-MINUTE average — not '
        'directly comparable to WMO 10-min winds; converted from kt)'),
    'usa_pres': VarInfo('Pa',
        'minimum central pressure (converted from mb)'),
    'usa_poci': VarInfo('Pa',
        'pressure of the outermost closed isobar (converted from mb)'),
    'usa_roci': VarInfo('m',
        'radius of the outermost closed isobar (converted from nmile)'),
    'usa_rmw': VarInfo('m',
        'radius of maximum winds (converted from nmile)'),
    'storm_speed': VarInfo('m s-1',
        'storm translation speed (converted from kt)'),
    'storm_dir': VarInfo('degrees',
        'storm translation heading TOWARD, 0..360 bearing'),
    'iflag': VarInfo('str', 'IBTrACS interpolation flags'),
    'usa_status': VarInfo('str', 'US agency storm status code'),
    'usa_sshs': VarInfo('category',
        'project SSHS scheme 0..8 (remapped from USA_SSHS; subtropical '
        'folded by wind — see build.remap_sshs)'),
    'usa_sshs_raw': VarInfo('category',
        'raw IBTrACS USA_SSHS code, -5..5'),
    'is_subtropical': VarInfo('bool',
        'raw USA_SSHS == -2 (subtropical) before the remap'),
}

# usa_r{34,50,64}_{NE..NW}: wind radii per quadrant, nmile -> m at build.
for _r in (34, 50, 64):
    for _q in _QUADS:
        VARIABLES[f'usa_r{_r}_{_q}'] = VarInfo('m',
            f'radius of {_r}-kt winds, {_q} quadrant '
            f'(converted from nmile)')


def column_meta(columns) -> dict:
    """{column: {'units', 'description'}} for a volume's meta.json sidecar.

    Columns missing from the catalogue get an explicit 'unknown' entry
    rather than an error — write_volume stays usable for scratch volumes;
    the schema tests assert the REAL schemas are fully catalogued.
    """
    unknown = VarInfo('unknown', 'not in the data/variables.py catalogue')
    return {c: VARIABLES.get(c, unknown)._asdict() for c in columns}


# ---------------------------------------------------------------------------
# Display units — storage is SI; plots/tables convert at the boundary
# ---------------------------------------------------------------------------

# canonical unit -> (display unit, converter). One consistent mapping for
# every figure/table (units ruling): m -> km, Pa -> hPa, everything else
# displays as stored.
DISPLAY_UNITS = {
    'm':  ('km',  m_to_km),
    'Pa': ('hPa', pa_to_hpa),
}


def to_display(value, units):
    """(value, canonical units) -> (value, display units).

    Identity for units without a DISPLAY_UNITS entry (m/s, K, degrees...).
    """
    display, convert = DISPLAY_UNITS.get(units, (units, lambda x: x))
    return convert(value), display
