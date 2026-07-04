"""
experiments/cyclone_jax/data/sources/build.py

BUILD-TIME module: converts the four raw sources into volume_v1 column
stores (see sources/volume.py). Run where the raw NetCDF lives; the
training path never imports this — train-time access goes through
library.py (numpy-only).

Sources (CDS version pins are provenance — record them with the data):
    LISO  land surface   (insitu-observations-surface-land,   v3_0_0) -> earth-arcanum
    MISO  marine surface (insitu-observations-surface-marine, v2_0_0) -> ocean-arcanum
    CUON  upper air      (insitu-comprehensive-upper-air-...,  v1_1_0) -> sky-arcanum
    IBTrACS v04r01 NA    (NOAA NCEI best track)                        -> storm-arcanum

Shared build skeleton (per source):
    fused compute -> QC/var/value mask -> factorize composite key ->
    reverse-scatter long->wide (first occurrence wins) -> vertical gate ->
    stable time sort -> entity spine.

Per-source differences: the VARS map, the composite-key columns, the gate
column, and post-hooks (hypsometric cross-fill = land; SSHS remap +
category spine = cyclone; launch timestamp + z-in-key = upper).

The surface vertical gate (require finite pressure) is DELIBERATE:
pressure is fundamental to the weather state and doubles as the vertical
coordinate — a report without it is not a usable measurement here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.cyclone_jax.data.sources.volume import build_entity_spine, write_volume  # noqa: F401 (write_volume re-exported for build scripts)


# ===========================================================================
# Channel schemas: CDM observed_variable byte-strings -> short column aliases
# ===========================================================================

LAND_VARS = {
    b'air_pressure':              'station_pressure',
    b'air_pressure_at_sea_level': 'slp',
    b'air_temperature':           'air_temp',
    b'dew_point_temperature':     'dewpoint',
    b'wind_from_direction':       'wind_dir',
    b'wind_speed':                'wind_speed',
}

MARINE_VARS = {
    b'air_pressure_at_sea_level': 'slp',          # -> level (vertical coord)
    b'air_temperature':           'air_temp',
    b'dew_point_temperature':     'dewpoint',
    b'water_temperature':         'sst',
    b'wind_from_direction':       'wind_dir',
    b'wind_speed':                'wind_speed',
}

UPPER_VARS = {
    b'air_temperature':      'air_temp',
    b'air_dewpoint':         'dewpoint',
    b'dew_point_depression': 'dpd',
    b'eastward_wind_speed':  'u_wind',
    b'northward_wind_speed': 'v_wind',
    b'geopotential_height':  'geopot',
    b'relative_humidity':    'rh',
    b'specific_humidity':    'q',
}

# CDM quality_flag values: 0,2,4,5 usable; 1,3,6 suspect/erroneous (dropped)
QC_BAD = (1, 3, 6)

# Physical constants for the hypsometric equation
RD = 287.052874   # J/(kg K), specific gas constant for dry air
G  = 9.80665      # m/s^2,    standard gravitational acceleration

QUADS = ['NE', 'SE', 'SW', 'NW']

# Project SSHS scheme (0..8, no negatives, subtropical folded by wind)
SSHS_MIN, SSHS_MAX = 0, 8
N_CAT = SSHS_MAX - SSHS_MIN + 1
RAW_SUBTROPICAL = -2                     # IBTrACS USA_SSHS subtropical code


# ===========================================================================
# Shared extraction helpers
# ===========================================================================

def _extract(sub, var_map, use_qc=True):
    """QC/var/value mask shared by the CDM builders.

    Returns (keep_mask, var_code[keep] int8, val[keep] float32).
    """
    var_col = {k: i for i, k in enumerate(var_map)}
    if use_qc:
        qc_bad = np.isin(sub['quality_flag'].values, QC_BAD)
    else:
        qc_bad = np.zeros(len(sub['observation_value']), dtype=bool)
    var_code = pd.Series(sub['observed_variable'].values).map(var_col).to_numpy()
    val      = sub['observation_value'].values.astype(np.float32)
    keep = np.isfinite(val) & ~np.isnan(var_code) & ~qc_bad
    print(f"  qc/var kept: {keep.sum():,} / {len(keep):,} ({100 * keep.mean():.1f}%)")
    return keep, var_code[keep].astype(np.int8), val[keep]


def _reverse_scatter(row_code, var_code, val, nrow, nvar):
    """Wide (nrow x nvar) fill where the FIRST occurrence wins per cell
    (equivalent to aggfunc='first' in a pivot_table)."""
    wide = np.full((nrow, nvar), np.nan, np.float32)
    wide[row_code[::-1], var_code[::-1]] = val[::-1]
    return wide


def _time_sort(out, sid):
    """Stable time sort of the column dict + entity spine over the result."""
    order = np.argsort(out['report_timestamp'], kind='stable')
    out = {k: v[order] for k, v in out.items()}
    sid = np.asarray(sid)[order]
    entity_ids, entity_int, entity_order, entity_offsets = build_entity_spine(sid)
    return out, entity_int, entity_ids, entity_order, entity_offsets


# ===========================================================================
# LISO — land surface -> earth-arcanum
# ===========================================================================

LAND_NEEDED = [
    'primary_station_id', 'report_timestamp', 'latitude', 'longitude',
    'height_of_station_above_sea_level', 'observed_variable',
    'observation_value', 'quality_flag',
]


def _hypsometric_fill(station_p, slp, temp_K, elev):
    """Cross-fill missing station pressure or SLP via the hypsometric equation.

    p_sl = p_sfc * exp(elev / H) with H = Rd*T/g. Each side is derived only
    from the ORIGINAL other side — no chaining. Runs BEFORE the vertical gate
    so SLP-only stations with temp+elev are not dropped.

    Returns (station_p, slp, n_station_p_derived, n_slp_derived).
    """
    sp, sl = station_p.copy(), slp.copy()

    good_TZ = np.isfinite(temp_K) & np.isfinite(elev)
    H = np.where(good_TZ, RD * temp_K / G, np.nan)

    need_sp = ~np.isfinite(station_p) & np.isfinite(slp) & good_TZ
    sp[need_sp] = slp[need_sp] * np.exp(-elev[need_sp] / H[need_sp])

    need_sl = ~np.isfinite(slp) & np.isfinite(station_p) & good_TZ
    sl[need_sl] = station_p[need_sl] * np.exp(elev[need_sl] / H[need_sl])

    return sp, sl, int(need_sp.sum()), int(need_sl.sum())


def build_land_volume(ds):
    """Long-format CDM land dataset -> time-sorted volume columns + spine.

    Stages: fused compute -> QC mask -> factorize (station, time, lat, lon,
    elev) -> reverse scatter -> hypsometric cross-fill -> vertical gate on
    station pressure (= level) -> time sort -> entity spine.
    """
    sub = ds[LAND_NEEDED].compute()
    keep, var_code, val = _extract(sub, LAND_VARS)

    sid  = sub['primary_station_id'].values[keep]
    t    = sub['report_timestamp'].values[keep]
    lat  = sub['latitude'].values[keep].astype(np.float32)
    lon  = sub['longitude'].values[keep].astype(np.float32)
    elev = sub['height_of_station_above_sea_level'].values[keep].astype(np.float32)

    key = pd.MultiIndex.from_arrays([sid, t, lat, lon, elev])
    row_code, row_keys = key.factorize(sort=False)
    wide = _reverse_scatter(row_code, var_code, val, len(row_keys), len(LAND_VARS))

    g = row_keys.get_level_values
    elev_k = g(4).to_numpy().astype(np.float32)
    var_col = {k: i for i, k in enumerate(LAND_VARS)}

    station_p = wide[:, var_col[b'air_pressure']]
    slp_col   = wide[:, var_col[b'air_pressure_at_sea_level']]
    temp_K    = wide[:, var_col[b'air_temperature']]
    station_p, slp_col, n_sp, n_sl = _hypsometric_fill(
        station_p, slp_col, temp_K, elev_k)
    wide[:, var_col[b'air_pressure']]              = station_p
    wide[:, var_col[b'air_pressure_at_sea_level']] = slp_col
    print(f"  hypsometric: +{n_sp:,} station_p, +{n_sl:,} slp derived")

    has_level = np.isfinite(station_p)
    print(f"  with-pressure: {has_level.sum():,} / {len(has_level):,} "
          f"({100 * has_level.mean():.1f}%)")

    out = {
        'report_timestamp': g(1).to_numpy()[has_level].astype('datetime64[ns]'),
        'lat':              g(2).to_numpy()[has_level].astype(np.float32),
        'lon':              g(3).to_numpy()[has_level].astype(np.float32),
        'elevation':        elev_k[has_level],
        'level':            station_p[has_level].copy(),
    }
    for j, name in enumerate(LAND_VARS.values()):
        out[name] = wide[has_level, j]
    sid_u = g(0).to_numpy()[has_level].astype('U11')

    return _time_sort(out, sid_u)


# ===========================================================================
# MISO — marine surface -> ocean-arcanum
# ===========================================================================

MARINE_NEEDED = ['primary_station_id', 'report_timestamp', 'latitude',
                 'longitude', 'platform_type', 'observed_variable',
                 'observation_value', 'quality_flag']


def build_marine_volume(ds):
    """As land, with platform_type in the key, no elevation/hypsometric fill,
    and the vertical gate on SLP (= level)."""
    sub = ds[MARINE_NEEDED].compute()
    keep, var_code, val = _extract(sub, MARINE_VARS)

    sid  = sub['primary_station_id'].values[keep]
    t    = sub['report_timestamp'].values[keep]
    lat  = sub['latitude'].values[keep].astype(np.float32)
    lon  = sub['longitude'].values[keep].astype(np.float32)
    plat = sub['platform_type'].values[keep].astype(np.int32)

    key = pd.MultiIndex.from_arrays([sid, t, lat, lon, plat])
    row_code, row_keys = key.factorize(sort=False)
    wide = _reverse_scatter(row_code, var_code, val, len(row_keys), len(MARINE_VARS))

    var_col = {k: i for i, k in enumerate(MARINE_VARS)}
    slp = wide[:, var_col[b'air_pressure_at_sea_level']]
    has_level = np.isfinite(slp)
    print(f"  with-pressure: {has_level.sum():,} / {len(has_level):,} "
          f"({100 * has_level.mean():.1f}%)")

    g = row_keys.get_level_values
    out = {
        'report_timestamp' : g(1).to_numpy()[has_level].astype('datetime64[ns]'),
        'lat'              : g(2).to_numpy()[has_level].astype(np.float32),
        'lon'              : g(3).to_numpy()[has_level].astype(np.float32),
        'level'            : slp[has_level].copy(),
        'platform_type'    : g(4).to_numpy()[has_level].astype(np.int32),
    }
    for j, name in enumerate(MARINE_VARS.values()):
        out[name] = wide[has_level, j]
    sid_u = g(0).to_numpy()[has_level].astype('U10')

    return _time_sort(out, sid_u)


# ===========================================================================
# CUON — upper air -> sky-arcanum
# ===========================================================================

UPPER_NEEDED = ['primary_station_id', 'record_timestamp', 'report_timestamp',
                'z_coordinate', 'latitude', 'longitude', 'platform_type',
                'observed_variable', 'observation_value']


def build_upper_volume(ds):
    """CUON differences: no quality_flag; record_timestamp is the canonical
    spine (report_timestamp = actual launch second, kept as
    launch_timestamp for provenance/leak-checking); z_coordinate (pressure)
    rides in the composite key, so one row == one (station, time, level)."""
    sub = ds[UPPER_NEEDED].compute()
    keep, var_code, val = _extract(sub, UPPER_VARS, use_qc=False)

    sid    = sub['primary_station_id'].values[keep]
    t      = sub['record_timestamp'].values[keep]     # canonical spine
    launch = sub['report_timestamp'].values[keep]     # actual ascent second
    z      = sub['z_coordinate'].values[keep].astype(np.float32)
    lat    = sub['latitude'].values[keep].astype(np.float32)
    lon    = sub['longitude'].values[keep].astype(np.float32)
    plat   = sub['platform_type'].values[keep].astype(np.int32)

    key = pd.MultiIndex.from_arrays([sid, t, launch, z, lat, lon, plat])
    row_code, row_keys = key.factorize(sort=False)
    wide = _reverse_scatter(row_code, var_code, val, len(row_keys), len(UPPER_VARS))

    # level is a KEY component here, so this only catches genuine NaN
    # pressures in source — cheap insurance, should remove ~nothing.
    g = row_keys.get_level_values
    level = g(3).to_numpy().astype(np.float32)
    has_level = np.isfinite(level)
    if not has_level.all():
        print(f"  with-pressure: {has_level.sum():,} / {len(has_level):,} "
              f"({100 * has_level.mean():.1f}%)")

    out = {
        'report_timestamp' : g(1).to_numpy()[has_level].astype('datetime64[ns]'),
        'launch_timestamp' : g(2).to_numpy()[has_level].astype('datetime64[ns]'),
        'lat'              : g(4).to_numpy()[has_level].astype(np.float32),
        'lon'              : g(5).to_numpy()[has_level].astype(np.float32),
        'platform_type'    : g(6).to_numpy()[has_level].astype(np.int32),
        'level'            : level[has_level],
    }
    for j, name in enumerate(UPPER_VARS.values()):
        out[name] = wide[has_level, j]
    sid_u = g(0).to_numpy()[has_level].astype('U18')

    return _time_sort(out, sid_u)


# ===========================================================================
# IBTrACS -> storm-arcanum (driver volume: SSHS remap + category spine)
# ===========================================================================

def remap_sshs(usa_sshs, usa_wind):
    """IBTrACS USA_SSHS (-5..5) -> project scheme (0..8).

        old             new
        -4 post-trop  -> 0
        -3 misc       -> 1
        -1 trop dep   -> 2
         0 trop storm -> 3
         1..5 cat1-5  -> 4..8
        -2 subtropical-> by wind (Saffir-Simpson kt thresholds)

    Subtropical with NaN wind stays NaN (cannot classify); unmapped rows
    stay NaN and are excluded from the category index.
    """
    old  = np.asarray(usa_sshs)
    wind = np.asarray(usa_wind)
    new  = np.full(old.shape, np.nan, np.float32)

    for o, n in [(-4, 0), (-3, 1), (-1, 2), (0, 3),
                 (1, 4), (2, 5), (3, 6), (4, 7), (5, 8)]:
        new[old == o] = n

    sub = old == RAW_SUBTROPICAL
    new[sub & (wind <  34)]                  = 2
    new[sub & (wind >= 34)  & (wind <  64)]  = 3
    new[sub & (wind >= 64)  & (wind <  83)]  = 4   # rare, kept consistent
    new[sub & (wind >= 83)  & (wind <  96)]  = 5
    new[sub & (wind >= 96)  & (wind < 113)]  = 6
    new[sub & (wind >= 113) & (wind < 137)]  = 7
    new[sub & (wind >= 137)]                 = 8
    # subtropical with NaN wind stays NaN (cannot classify)

    return new


def build_category_index(obs, exclude=None):
    """Fix-level CSR over remapped usa_sshs (0..8 buckets, time order kept).

    exclude: optional bool mask of rows omitted from the index (e.g.
    subtropical); excluded rows remain in the volume, just not in cat_order.
    """
    sshs = np.asarray(obs['usa_sshs'])
    finite = np.isfinite(sshs)
    cat = np.empty(sshs.shape, np.int64)
    cat[finite] = np.rint(sshs[finite]).astype(np.int64)

    in_range = finite & (cat >= SSHS_MIN) & (cat <= SSHS_MAX)
    if exclude is not None:
        in_range &= ~np.asarray(exclude, bool)
    rows = np.nonzero(in_range)[0]              # ascending row = ascending time
    bucket = cat[rows] - SSHS_MIN

    local = np.argsort(bucket, kind='stable')   # group, keep time order
    cat_order = rows[local].astype(np.int64)
    counts = np.bincount(bucket, minlength=N_CAT)
    cat_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return cat_order, cat_offsets


def build_storm_volume(ds, keep_subtropical=True):
    """IBTrACS (storms x date_time) cube -> flat fix-level volume.

    No vertical gate: the cyclone volume is the DRIVER (query points are
    lat/lon/time); usa_pres is stored as level but allowed NaN. Raw SSHS is
    preserved (usa_sshs_raw + is_subtropical) alongside the remapped 0..8.

    keep_subtropical=False removes subtropical fixes from the CATEGORY index
    only — the rows stay in the volume and every other index.
    """
    valid = ~np.isnat(ds['time'].values)              # (S, D) time-only mask

    def col(name):
        v = ds[name].values
        if v.ndim == 1:                               # per-storm (S,) -> (S, D)
            v = np.broadcast_to(v[:, None], valid.shape)
        return v[valid]

    out = {
        'sid'              : col('sid').astype('U13'),
        'name'             : col('name').astype('U16'),
        'season'           : col('season').astype(np.float32),
        'basin'            : col('basin').astype('U2'),
        'subbasin'         : col('subbasin').astype('U2'),
        'report_timestamp' : ds['time'].values[valid].astype('datetime64[ns]'),
        'lat'              : col('lat').astype(np.float32),
        'lon'              : col('lon').astype(np.float32),
        # SLP as vertical coord — allowed NaN (driver role, no gate).
        'level'            : col('usa_pres').astype(np.float32),
        'usa_wind'         : col('usa_wind').astype(np.float32),
        'usa_pres'         : col('usa_pres').astype(np.float32),
        'iflag'            : col('iflag').astype('U16'),
        'usa_status'       : col('usa_status').astype('U2'),
        'usa_sshs'         : col('usa_sshs').astype(np.float32),   # remapped below
        'usa_poci'         : col('usa_poci').astype(np.float32),
        'usa_roci'         : col('usa_roci').astype(np.float32),
        'usa_rmw'          : col('usa_rmw').astype(np.float32),
        'storm_speed'      : col('storm_speed').astype(np.float32),
        'storm_dir'        : col('storm_dir').astype(np.float32),
    }

    for src in ['usa_r34', 'usa_r50', 'usa_r64']:
        arr = ds[src].values                           # (S, D, 4)
        for q, qn in enumerate(QUADS):
            out[f'{src}_{qn}'] = arr[:, :, q][valid].astype(np.float32)

    # keep BOTH mappings: raw IBTrACS scale and the project scheme
    raw = out['usa_sshs'].copy()
    out['usa_sshs_raw']   = raw
    out['is_subtropical'] = (raw == RAW_SUBTROPICAL)
    out['usa_sshs']       = remap_sshs(raw, out['usa_wind'])

    out, entity_int, entity_ids, entity_order, entity_offsets = \
        _time_sort(out, out['sid'])

    exclude = (None if keep_subtropical else out['is_subtropical'])
    cat_order, cat_offsets = build_category_index(out, exclude=exclude)

    return (out, entity_int, entity_ids, entity_order, entity_offsets,
            cat_order, cat_offsets)
