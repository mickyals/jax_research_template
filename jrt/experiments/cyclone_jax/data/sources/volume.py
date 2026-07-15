"""
experiments/cyclone_jax/data/sources/volume.py

Generic columnar "volume" store (format volume_v1) — the successor to the
npz-backed layout. A volume is a directory of flat .npy column files plus a
CSR entity spine and a manifest:

    <dir>/<col_name>.npy      one per observation column (mmap-friendly)
    <dir>/_entity_int.npy     per-row entity label (int32)
    <dir>/_entity_ids.npy     sorted unique entity ID strings
    <dir>/_entity_order.npy   CSR row-position array (int64)
    <dir>/_entity_offsets.npy CSR indptr (int64, length K+1)
    <dir>/_cat_order.npy      OPTIONAL category spine (driver volumes)
    <dir>/_cat_offsets.npy    OPTIONAL category spine indptr
    <dir>/manifest.json       format tag, column list, row count
    <dir>/meta.json           {column: {units, description}} sidecar
                              (from data/variables.py — volumes self-describe)

Rows are time-sorted on `report_timestamp`, so time-window queries are
O(log N) binary searches. All large arrays are memory-mapped at load —
only the pages actually touched are read from disk, keeping memory full
of examples rather than the dataset.

The library metaphor: a volume is a book whose pages (rows) are ordered
by time; the shelf (sources/shelf.py) is the cross-volume index system.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VOLUME_FORMAT = 'volume_v1'


# ---------------------------------------------------------------------------
# Entity spine
# ---------------------------------------------------------------------------

def build_entity_spine(ids):
    """CSR-style entity index over time-sorted rows.

    Groups row positions by entity (station / storm) so one entity's whole
    time-ordered history is a single slice — no scanning. The stable argsort
    preserves time order within each entity's block.

    Parameters
    ----------
    ids : array-like of str
        Per-row entity identifier, in row (time) order.

    Returns
    -------
    entity_ids : ndarray, str
        Sorted unique entity ID strings (np.unique order).
    entity_int : ndarray, int32
        Per-row integer label indexing into entity_ids.
    entity_order : ndarray, int64
        Row positions grouped by entity, time order kept within each block.
    entity_offsets : ndarray, int64
        CSR indptr; entity e owns
        entity_order[entity_offsets[e] : entity_offsets[e + 1]].
    """
    entity_ids, entity_int = np.unique(np.asarray(ids), return_inverse=True)
    entity_int = entity_int.astype(np.int32)
    K = len(entity_ids)

    entity_order = np.argsort(entity_int, kind='stable').astype(np.int64)
    entity_offsets = np.concatenate(
        [[0], np.cumsum(np.bincount(entity_int, minlength=K))]).astype(np.int64)
    return entity_ids, entity_int, entity_order, entity_offsets


# ---------------------------------------------------------------------------
# Write / load
# ---------------------------------------------------------------------------

def write_volume(out_dir, obs, entity_int, entity_ids, entity_order,
                 entity_offsets, cat_order=None, cat_offsets=None):
    """Persist a volume: one .npy per column + entity spine (+ category spine).

    Object/string columns are cast to fixed-width unicode so they can be
    memory-mapped. `cat_order`/`cat_offsets` (both or neither) attach an
    optional category spine (used by driver volumes, e.g. cyclone SSHS).

    A meta.json sidecar ({column: {units, description}}, from the
    data/variables.py catalogue) makes the volume self-describing;
    uncatalogued columns get an explicit 'unknown' entry.

    Returns the written directory as a Path.
    """
    if (cat_order is None) != (cat_offsets is None):
        raise ValueError("cat_order and cat_offsets must be given together.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = list(obs.keys())
    for k in cols:
        arr = np.asarray(obs[k])
        if arr.dtype == object:
            arr = arr.astype(np.str_)
        np.save(out / f'{k}.npy', arr)
    np.save(out / '_entity_int.npy',     np.asarray(entity_int).astype(np.int32))
    np.save(out / '_entity_ids.npy',     np.asarray(entity_ids).astype(np.str_))
    np.save(out / '_entity_order.npy',   np.asarray(entity_order).astype(np.int64))
    np.save(out / '_entity_offsets.npy', np.asarray(entity_offsets).astype(np.int64))
    if cat_order is not None:
        np.save(out / '_cat_order.npy',   np.asarray(cat_order).astype(np.int64))
        np.save(out / '_cat_offsets.npy', np.asarray(cat_offsets).astype(np.int64))
    (out / 'manifest.json').write_text(json.dumps(
        {'format': VOLUME_FORMAT, 'columns': cols,
         'n_rows': int(len(np.asarray(entity_int)))}))
    # import here: variables.py is policy, this module is store mechanics
    from experiments.cyclone_jax.data.variables import column_meta
    (out / 'meta.json').write_text(json.dumps(column_meta(cols), indent=2))
    return out


def load_volume(d):
    """Load a volume directory into a dict, memory-mapping the large arrays.

    Returns
    -------
    dict with keys:
        obs             — {column: mmap ndarray}
        entity_int      — per-row entity label (mmap)
        entity_ids      — unique entity IDs (small, in RAM)
        entity_order    — CSR row positions (mmap)
        entity_offsets  — CSR indptr (small, in RAM)
        cat_order / cat_offsets — present only if the volume has a
                          category spine on disk
    """
    d = Path(d)
    man = json.loads((d / 'manifest.json').read_text())
    if man.get('format') != VOLUME_FORMAT:
        raise ValueError(
            f"{d} is not a {VOLUME_FORMAT} volume "
            f"(found format={man.get('format')!r}).")
    obs = {k: np.load(d / f'{k}.npy', mmap_mode='r') for k in man['columns']}

    vol = {
        'obs'            : obs,
        'entity_int'     : np.load(d / '_entity_int.npy', mmap_mode='r'),
        'entity_ids'     : np.load(d / '_entity_ids.npy'),
        'entity_order'   : np.load(d / '_entity_order.npy', mmap_mode='r'),
        'entity_offsets' : np.load(d / '_entity_offsets.npy'),
    }
    if (d / '_cat_order.npy').exists():
        vol['cat_order']   = np.load(d / '_cat_order.npy', mmap_mode='r')
        vol['cat_offsets'] = np.load(d / '_cat_offsets.npy')
    return vol


# ---------------------------------------------------------------------------
# Time spine (single volume)
# ---------------------------------------------------------------------------

def build_volume_time_index(obs):
    """Single-volume time spine: distinct timestamps + CSR time-group offsets.

    Rows at distinct timestamp i are obs[k][time_offsets[i]:time_offsets[i+1]].
    """
    t = np.asarray(obs['report_timestamp'], dtype='datetime64[ns]')
    uniq_t, start = np.unique(t, return_index=True)
    time_offsets = np.append(start, len(t)).astype(np.int64)
    return uniq_t, time_offsets


def time_slice(obs, t_lo, t_hi):
    """Bounding row indices for the closed time range [t_lo, t_hi].

    Binary search on the sorted report_timestamp — O(log N). Returns (lo, hi)
    such that obs[k][lo:hi] spans the requested window (lo == hi when empty).
    """
    t = obs['report_timestamp']
    lo = int(np.searchsorted(t, np.datetime64(t_lo), side='left'))
    hi = int(np.searchsorted(t, np.datetime64(t_hi), side='right'))
    return lo, hi


def rows_at(obs, uniq_t, time_offsets, t):
    """All rows at exactly timestamp t (empty columns dict when t is absent)."""
    i = int(np.searchsorted(uniq_t, np.datetime64(t)))
    if i >= len(uniq_t) or uniq_t[i] != np.datetime64(t):
        return {k: v[:0] for k, v in obs.items()}
    lo, hi = time_offsets[i], time_offsets[i + 1]
    return {k: v[lo:hi] for k, v in obs.items()}


# ---------------------------------------------------------------------------
# Entity access
# ---------------------------------------------------------------------------

def get_entity(vol, who):
    """One entity's full history, time-ordered, via the CSR entity spine.

    `who` is an int entity index or an ID string (binary search into the
    sorted entity_ids).
    """
    eids = vol['entity_ids']
    e = who if isinstance(who, (int, np.integer)) else \
        int(np.searchsorted(eids, who))
    eo, eoff = vol['entity_order'], vol['entity_offsets']
    rows = eo[eoff[e]:eoff[e + 1]]
    return {k: v[rows] for k, v in vol['obs'].items()}
