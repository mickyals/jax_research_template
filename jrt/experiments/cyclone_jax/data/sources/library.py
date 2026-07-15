"""
experiments/cyclone_jax/data/sources/library.py

TRAIN-TIME access to the arcana library (numpy-only — never imports
xarray/dask; those live in build.py, the build-time module).

The library ("Caribbean-Obs") is four volume_v1 directories plus one
_BOOKSHELF of cross-volume indices:

    earth-arcanum  (land)    ocean-arcanum (marine)
    sky-arcanum    (upper)   storm-arcanum (cyclone driver)

This module owns the cyclone-specific vocabulary (SSHS constants, target
columns, category-spine access, driver fixes) and the canonical per-volume
lookback schedules; the generic volume/shelf mechanics live in
sources/volume.py and sources/shelf.py.

Paths come from configs/data.yaml — nothing here hardcodes the library
root. The deltas actually baked into a bookshelf are recorded in each
volume's shelf meta (lookback_deltas_s); LOOKBACK_DELTAS below is what a
(re)build uses, and load_library verifies the two agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.cyclone_jax.data.sources.volume import load_volume, get_entity, time_slice, rows_at  # noqa: F401
from experiments.cyclone_jax.data.sources.shelf import (  # noqa: F401
    SHELF_DIR, write_shelf, load_shelf, load_all_shelves, check_shelf_fresh,
    build_time_index, rows_at_shelf, build_lookback_pointers, write_lookback,
    load_lookback, window_obs, window_temporal_encoding, report_occupancy,
)

# ---------------------------------------------------------------------------
# Library layout + cyclone vocabulary
# ---------------------------------------------------------------------------

VOLUMES = {'land':   'earth-arcanum',  'marine': 'ocean-arcanum',
           'upper':  'sky-arcanum',    'cyclone': 'storm-arcanum'}

OBS_VOLUMES = ('land', 'marine', 'upper')

CYC_SSHS = 'usa_sshs'        # remapped 0..8 category column on the cyclone volume

TROPICAL_STORM = 3           # remapped scheme thresholds
HURRICANE      = 4
MAJOR_HURR     = 6
SSHS_MIN, SSHS_MAX = 0, 8

# IBTrACS intensity/structure fields: TARGET/METADATA ONLY — never model
# input (the leakage allowlist: model inputs are obs channels + query
# position/time, nothing from this tuple).
CYC_TARGETS = ('usa_wind', 'usa_pres', 'usa_rmw', 'usa_poci', 'usa_roci',
               'usa_r34_NE', 'usa_r34_SE', 'usa_r34_SW', 'usa_r34_NW',
               'usa_r50_NE', 'usa_r50_SE', 'usa_r50_SW', 'usa_r50_NW',
               'usa_r64_NE', 'usa_r64_SE', 'usa_r64_SW', 'usa_r64_NW')

# Per-volume lookback windows, DESCENDING (deltas[0] = outer reach), sized
# to each source's cadence: land/marine report ~hourly (3 h reach); upper
# soundings are ~12-hourly, so a 12 h reach guarantees the previous synoptic
# cycle is captured (empties drop to the 2025 coverage hole only).
LOOKBACK_DELTAS = {
    'land':   [np.timedelta64(3, 'h'),  np.timedelta64(2, 'h'),
               np.timedelta64(1, 'h'),  np.timedelta64(30, 'm'),
               np.timedelta64(10, 'm')],
    'marine': [np.timedelta64(3, 'h'),  np.timedelta64(2, 'h'),
               np.timedelta64(1, 'h'),  np.timedelta64(30, 'm'),
               np.timedelta64(10, 'm')],
    'upper':  [np.timedelta64(12, 'h'), np.timedelta64(6, 'h'),
               np.timedelta64(4, 'h'),  np.timedelta64(3, 'h'),
               np.timedelta64(2, 'h'),  np.timedelta64(1, 'h'),
               np.timedelta64(30, 'm'), np.timedelta64(10, 'm')],
}


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def load_library(root, names=None, check_fresh=True, check_deltas=True):
    """Load volumes + shelves from a library root, with staleness guards.

    Parameters
    ----------
    root : path
        Library root (e.g. E:/Caribbean-Obs) — from configs/data.yaml.
    names : sequence of volume keys, optional
        Subset of VOLUMES to load (default: all four).
    check_fresh : bool
        Raise RuntimeError when a shelf's fingerprint mismatches its volume.
    check_deltas : bool
        Raise RuntimeError when a shelf's baked lookback deltas differ from
        LOOKBACK_DELTAS (the schedule the code expects).

    Returns
    -------
    {'root': Path, 'volumes': {name: vol_dict}, 'shelves': {name: shelf_dict}}
    """
    root = Path(root)
    names = tuple(names) if names is not None else tuple(VOLUMES)

    volumes = {n: load_volume(root / VOLUMES[n]) for n in names}
    shelves = load_all_shelves(root, names)

    if check_fresh:
        for n in names:
            fresh, why = check_shelf_fresh(volumes[n], shelves[n])
            if not fresh:
                raise RuntimeError(
                    f"stale shelf for {n!r}: {why}. Rebuild with build_bookshelf.")

    if check_deltas:
        for n in names:
            if n not in LOOKBACK_DELTAS:
                continue
            _, baked = load_lookback(shelves[n])
            if baked is None:
                continue
            expect = [int(d / np.timedelta64(1, 's')) for d in LOOKBACK_DELTAS[n]]
            got = [int(d / np.timedelta64(1, 's')) for d in baked]
            if got != expect:
                raise RuntimeError(
                    f"lookback deltas for {n!r} differ from LOOKBACK_DELTAS: "
                    f"baked={got} expected={expect}. Rebuild with build_bookshelf.")

    return {'root': root, 'volumes': volumes, 'shelves': shelves}


def build_bookshelf(root, sshs_min=TROPICAL_STORM, drop_subtropical=False,
                    deltas=None, verbose=True):
    """(Re)build the whole _BOOKSHELF: time spines, cyclone driver manifest,
    and per-volume lookback edges. Idempotent; run after any volume rebuild.
    """
    root = Path(root)
    deltas = deltas if deltas is not None else LOOKBACK_DELTAS

    for name in OBS_VOLUMES:
        vol = load_volume(root / VOLUMES[name])
        write_shelf(root, name, vol['obs'])
        if verbose:
            print(f"  shelf {name}: "
                  f"{vol['obs']['report_timestamp'].shape[0]:,} rows indexed")

    cyc = load_volume(root / VOLUMES['cyclone'])
    write_shelf(root, 'cyclone', cyc['obs'],
                driver_col=CYC_SSHS, driver_min=sshs_min,
                exclude_col=('is_subtropical' if drop_subtropical else None))
    scyc = load_shelf(root, 'cyclone')
    if verbose:
        m = scyc['meta']
        print(f"  shelf cyclone: driver manifest at sshs>={sshs_min} | "
              f"single={m['n_single']:,} multi={m['n_multi']:,}")

    storm_times = np.asarray(scyc['storm_times'])
    for name in OBS_VOLUMES:
        vol = load_volume(root / VOLUMES[name])
        edges = write_lookback(root, name, storm_times, vol['obs'], deltas[name])
        if verbose:
            n_empty = int(np.sum(edges[:, 0] == edges[:, -1]))
            reach_h = deltas[name][0] / np.timedelta64(1, 'h')
            print(f"  lookback {name}: {edges.shape} | reach {reach_h:.0f}h | "
                  f"{len(deltas[name])} windows | {n_empty} empty full-window")
    return root


# ---------------------------------------------------------------------------
# Driver fixes (cyclone volume)
# ---------------------------------------------------------------------------

def get_fixes(cyc_vol, sshs_min=TROPICAL_STORM, drop_subtropical=False):
    """Flat driver-fix table at a threshold: time/position/identity/label
    plus the CYC_TARGETS columns (target/metadata — never model input)."""
    obs = cyc_vol['obs']
    m = np.asarray(obs[CYC_SSHS]) >= sshs_min
    if drop_subtropical and 'is_subtropical' in obs:
        m &= ~np.asarray(obs['is_subtropical'])
    fixes = {
        'time'       : np.asarray(obs['report_timestamp'])[m],
        'lat'        : np.asarray(obs['lat'])[m],
        'lon'        : np.asarray(obs['lon'])[m],
        'entity_int' : np.asarray(cyc_vol['entity_int'])[m],
        'sid'        : np.asarray(obs['sid'])[m],
        CYC_SSHS     : np.asarray(obs[CYC_SSHS])[m],
    }
    if 'name' in obs:        # human storm name (v2 volumes) — plot titles
        fixes['name'] = np.asarray(obs['name'])[m]
    for k in CYC_TARGETS:
        if k in obs:
            fixes[k] = np.asarray(obs[k])[m]
    return fixes


# ---------------------------------------------------------------------------
# Category spine (cyclone volume)
# ---------------------------------------------------------------------------

def get_category(vol, sshs_cat):
    """All fixes of exactly one remapped category, time-ordered."""
    co, cf = vol['cat_order'], vol['cat_offsets']
    b = int(sshs_cat) - SSHS_MIN
    rows = co[cf[b]:cf[b + 1]]
    return {k: v[rows] for k, v in vol['obs'].items()}


def get_category_atleast(vol, sshs_min):
    """All fixes with category >= sshs_min (grouped by category, time-ordered
    within each — NOT globally time-sorted; re-sort on report_timestamp if
    that matters)."""
    co, cf = vol['cat_order'], vol['cat_offsets']
    rows = co[cf[int(sshs_min) - SSHS_MIN]:]
    return {k: v[rows] for k, v in vol['obs'].items()}


def get_hurricanes(vol):        # remapped cat 4-8
    return get_category_atleast(vol, HURRICANE)


def get_major_hurricanes(vol):  # remapped cat 6-8
    return get_category_atleast(vol, MAJOR_HURR)


def category_counts(vol):
    """Per-bucket fix counts from the category spine — {category: count}."""
    counts = np.diff(np.asarray(vol['cat_offsets']))
    return {int(SSHS_MIN + i): int(c) for i, c in enumerate(counts)}
