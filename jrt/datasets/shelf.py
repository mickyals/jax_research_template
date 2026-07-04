"""
datasets/shelf.py

The Shelf: one `_BOOKSHELF/` directory beside the volumes holding every
volume's cross-volume indices, keyed by volume name:

    <root>/_BOOKSHELF/<name>_uniq_t.npy          distinct timestamps (sorted)
    <root>/_BOOKSHELF/<name>_time_offsets.npy    time-group CSR offsets
    <root>/_BOOKSHELF/<name>_lookback_edges.npy  causal lookback pointers
    <root>/_BOOKSHELF/<name>_meta.json           freshness fingerprint + deltas
    driver volumes additionally:
    <root>/_BOOKSHELF/<name>_storm_times.npy     driver-threshold times
    <root>/_BOOKSHELF/<name>_n_storms.npy        drivers per timestamp
    <root>/_BOOKSHELF/<name>_single_times.npy    exactly 1 qualifying driver
    <root>/_BOOKSHELF/<name>_multi_times.npy     >1 (the OOD test set)

Causality is the shelf's core invariant: lookback windows reach strictly
backward from a driver time T ([T - reach, T]); observations after T are
never indexed for T. The freshness fingerprint (n_rows / t_first / t_last)
ties each shelf entry to the exact volume build it describes, so a stale
shelf is caught at load instead of corrupting downstream samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from datasets.volume import build_volume_time_index, rows_at

SHELF_DIR = '_BOOKSHELF'


# ---------------------------------------------------------------------------
# Write / load / freshness
# ---------------------------------------------------------------------------

def write_shelf(root, name, obs, driver_col=None, driver_min=None,
                exclude_col=None, shelf_dir=SHELF_DIR):
    """Write one volume's shelf entry (time spine + optional driver manifest).

    Parameters
    ----------
    root : path
        Library root; the shelf lives at <root>/<shelf_dir>/.
    name : str
        Volume name used as the file-name key (e.g. 'land', 'cyclone').
    obs : dict
        The volume's column dict (time-sorted).
    driver_col : str, optional
        Column holding the driver label (e.g. remapped 'usa_sshs'). When
        given with driver_min, a driver manifest is written: the unique
        timestamps with at least one row >= driver_min, split into
        single_times / multi_times by the per-timestamp count.
    driver_min : numeric, optional
        Threshold on driver_col for a row to qualify.
    exclude_col : str, optional
        Boolean column; True rows are excluded from the driver manifest
        (they stay in the volume and every other index).
    """
    sd = Path(root) / shelf_dir
    sd.mkdir(parents=True, exist_ok=True)

    uniq_t, time_offsets = build_volume_time_index(obs)
    np.save(sd / f'{name}_uniq_t.npy',       uniq_t)
    np.save(sd / f'{name}_time_offsets.npy', time_offsets.astype(np.int64))

    # Freshness fingerprint: ties this shelf to the exact volume build it
    # describes; any rebuild/patch mismatches n_rows/t_first/t_last and is
    # caught by check_shelf_fresh instead of silently corrupting samples.
    t = np.asarray(obs['report_timestamp'], dtype='datetime64[ns]')
    meta = {
        'n_times' : int(len(uniq_t)),
        'n_rows'  : int(len(t)),
        't_first' : int(t[0].astype('int64')),
        't_last'  : int(t[-1].astype('int64')),
        'dtype'   : str(t.dtype),
    }

    if driver_col is not None and driver_min is not None and driver_col in obs:
        m = np.asarray(obs[driver_col]) >= driver_min
        if exclude_col is not None and exclude_col in obs:
            m &= ~np.asarray(obs[exclude_col])

        storm_times, counts = np.unique(t[m], return_counts=True)
        single_times = storm_times[counts == 1]
        multi_times  = storm_times[counts > 1]

        np.save(sd / f'{name}_storm_times.npy',  storm_times)
        np.save(sd / f'{name}_n_storms.npy',     counts.astype(np.int32))
        np.save(sd / f'{name}_single_times.npy', single_times)
        np.save(sd / f'{name}_multi_times.npy',  multi_times)
        meta.update(driver_col=str(driver_col), driver_min=float(driver_min),
                    exclude_col=(str(exclude_col) if exclude_col else None),
                    n_storm_times=int(len(storm_times)),
                    n_single=int(len(single_times)),
                    n_multi=int(len(multi_times)))
    (sd / f'{name}_meta.json').write_text(json.dumps(meta))
    return sd


def load_shelf(root, name, shelf_dir=SHELF_DIR):
    """Load one volume's shelf entry. Returns {} when never written."""
    sd = Path(root) / shelf_dir
    up = sd / f'{name}_uniq_t.npy'
    if not up.exists():
        return {}
    shelf = {
        'uniq_t'       : np.load(up, mmap_mode='r'),
        'time_offsets' : np.load(sd / f'{name}_time_offsets.npy'),
    }
    for extra in ('storm_times', 'n_storms', 'single_times', 'multi_times',
                  'lookback_edges'):
        ep = sd / f'{name}_{extra}.npy'
        if ep.exists():
            shelf[extra] = np.load(ep, mmap_mode='r')
    mp = sd / f'{name}_meta.json'
    if mp.exists():
        shelf['meta'] = json.loads(mp.read_text())
    return shelf


def load_all_shelves(root, names, shelf_dir=SHELF_DIR):
    """All volumes' shelf entries as {name: shelf_dict}."""
    return {n: load_shelf(root, n, shelf_dir=shelf_dir) for n in names}


def check_shelf_fresh(vol, shelf):
    """(fresh, reason) — does the shelf fingerprint match the loaded volume?

    Old shelves without a fingerprint are treated as fresh (skipped).
    Callers decide whether a stale shelf is fatal (training: raise).
    """
    meta = shelf.get('meta', {})
    if 'n_rows' not in meta:
        return True, "no fingerprint (old shelf, skipped)"
    t = np.asarray(vol['obs']['report_timestamp'], dtype='datetime64[ns]')
    live = {'n_rows': int(len(t)),
            't_first': int(t[0].astype('int64')),
            't_last':  int(t[-1].astype('int64'))}
    for k, v in live.items():
        if meta.get(k) != v:
            return False, f"{k}: shelf={meta.get(k)} live={v}"
    return True, "fresh"


# ---------------------------------------------------------------------------
# Union time index + shelf-backed time queries
# ---------------------------------------------------------------------------

def build_time_index(*obs_dicts):
    """Sorted, deduped datetime64[ns] axis over one or more volumes."""
    cat = np.concatenate(
        [np.asarray(v['report_timestamp'], dtype='datetime64[ns]')
         for v in obs_dicts])
    return np.unique(cat)


def time_to_idx(times, t):
    """Index of exact timestamp t in a sorted axis, or -1 when absent."""
    if len(times) == 0:
        return -1
    dt = np.datetime64(t)
    i = int(np.searchsorted(times, dt))
    if i < len(times) and times[i] == dt:
        return i
    return -1


def rows_at_shelf(obs, shelf, t):
    """rows_at using a loaded shelf's time spine (built on the fly if empty)."""
    if 'uniq_t' in shelf:
        return rows_at(obs, shelf['uniq_t'], shelf['time_offsets'], t)
    ut, off = build_volume_time_index(obs)
    return rows_at(obs, ut, off, t)


# ---------------------------------------------------------------------------
# Lookback pointers: causal windows relative to driver (cyclone fix) times
# ---------------------------------------------------------------------------

def build_lookback_pointers(storm_times, obs, deltas):
    """Row-slice edges for contiguous backward windows before each driver time.

    Windows are defined by `deltas` (DESCENDING — deltas[0] is the outer
    reach): the first window spans [T - deltas[0], T - deltas[1]), the next
    [T - deltas[1], T - deltas[2]), and so on, with the final window covering
    [T - deltas[-1], T]. Nothing after T is ever included (causality).

    Both storm_times and the volume's report_timestamp are sorted, so all
    boundaries are computed with np.searchsorted in O(S log N) — no per-row
    scanning.

    Parameters
    ----------
    storm_times : ndarray, datetime64[ns], sorted
        Driver times from the shelf (e.g. shelf['storm_times']).
    obs : dict
        A volume's column dict with sorted report_timestamp.
    deltas : sequence of np.timedelta64, descending
        Lookback boundaries, e.g. [3h, 2h, 1h, 30m, 10m].

    Returns
    -------
    edges : ndarray int64, shape (len(storm_times), len(deltas) + 1)
        Window w of driver i spans obs[k][edges[i, w] : edges[i, w + 1]].
        Empty windows have edges[i, w] == edges[i, w + 1] — an empty window
        is itself informative to the model.
    """
    deltas = list(deltas)
    if any(deltas[i] <= deltas[i + 1] for i in range(len(deltas) - 1)):
        raise ValueError("deltas must be strictly descending (largest reach first).")

    t = obs['report_timestamp']
    S = len(storm_times)
    W = len(deltas)
    edges = np.empty((S, W + 1), np.int64)

    for w, d in enumerate(deltas):
        edges[:, w] = np.searchsorted(t, storm_times - d, side='left')
    edges[:, W] = np.searchsorted(t, storm_times, side='right')

    return edges


def write_lookback(root, vol_name, storm_times, obs, deltas,
                   shelf_dir=SHELF_DIR):
    """Build + persist lookback edges; record the deltas in the shelf meta."""
    sd = Path(root) / shelf_dir
    sd.mkdir(parents=True, exist_ok=True)
    edges = build_lookback_pointers(storm_times, obs, deltas)
    np.save(sd / f'{vol_name}_lookback_edges.npy', edges)

    meta_path = sd / f'{vol_name}_meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta['lookback_deltas_s'] = [int(d / np.timedelta64(1, 's')) for d in deltas]
    meta_path.write_text(json.dumps(meta))

    return edges


def load_lookback(shelf):
    """(edges, deltas) from a loaded shelf dict — either may be None."""
    edges = shelf.get('lookback_edges')
    meta = shelf.get('meta', {})
    deltas = None
    if 'lookback_deltas_s' in meta:
        deltas = [np.timedelta64(s, 's') for s in meta['lookback_deltas_s']]
    return edges, deltas


def window_obs(obs, edges, i, w):
    """Slice volume obs for driver time i, lookback window w."""
    lo, hi = edges[i, w], edges[i, w + 1]
    return {k: v[lo:hi] for k, v in obs.items()}


def window_temporal_encoding(obs, edges, storm_times, i, w, delta_max):
    """Relative temporal encoding for observations in window w of driver i.

    Values lie in [-1, 0]: -1 is delta_max ago, 0 is the driver time.
    `delta_max` is the normalising reach (that volume's deltas[0], or a
    global scale if uniform cross-source semantics are wanted).
    """
    lo, hi = edges[i, w], edges[i, w + 1]
    if lo == hi:
        return np.array([], dtype=np.float32)
    t_obs = obs['report_timestamp'][lo:hi].astype('int64')
    t_ref = storm_times[i].astype('int64')
    d_max = delta_max / np.timedelta64(1, 'ns')
    return ((t_obs - t_ref) / d_max).astype(np.float32)


# ---------------------------------------------------------------------------
# Occupancy diagnostics
# ---------------------------------------------------------------------------

def _dlabel(d):
    """Short human label for a timedelta64 (e.g. '3h', '30m')."""
    s = int(d / np.timedelta64(1, 's'))
    if s % 3600 == 0:
        return f"{s // 3600}h"
    if s % 60 == 0:
        return f"{s // 60}m"
    return f"{s}s"


def report_occupancy(name, storm_times, obs, deltas):
    """Console report: obs per lookback window per driver time.

    TEMPORAL counts only — an upper bound on a sample's token count before
    any spatial filtering. Returns the (S, W) per-window count array.
    """
    edges  = build_lookback_pointers(storm_times, obs, deltas)
    counts = np.diff(edges, axis=1)              # (S, W) obs per window
    W      = counts.shape[1]
    reach  = _dlabel(deltas[0])
    S      = len(storm_times)

    labels = [f"{_dlabel(deltas[w])} → {_dlabel(deltas[w + 1])}"
              for w in range(W - 1)]
    labels.append(f"{_dlabel(deltas[-1])} → T")

    bar = "=" * 66
    print()
    print(bar)
    print(f"  {name}   |   {W} windows   |   {reach} reach   |   {S:,} fixes")
    print("  " + "-" * 62)
    print(f"  {'window':<13}{'mean':>9}{'median':>9}{'p95':>9}"
          f"{'max':>11}{'% empty':>10}")
    print("  " + "-" * 62)
    for w in range(W):
        c = counts[:, w]
        print(f"  {labels[w]:<13}{c.mean():>9,.1f}{int(np.median(c)):>9,}"
              f"{int(np.percentile(c, 95)):>9,}{int(c.max()):>11,}"
              f"{100 * (c == 0).mean():>9.1f}%")
    print("  " + "-" * 62)
    tot = counts.sum(axis=1)
    print(f"  {'per-fix total':<13}{tot.mean():>9,.1f}{int(np.median(tot)):>9,}"
          f"{int(np.percentile(tot, 95)):>9,}{int(tot.max()):>11,}"
          f"{100 * (tot == 0).mean():>9.1f}%")
    print(bar)
    return counts
