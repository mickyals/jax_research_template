"""
experiments/cyclone_jax/data/identifiability.py

Memorisation precondition for the full-dataset probe: if two fixes present
IDENTICAL model inputs but different targets, no deterministic model can
get both right — the probe's accuracy ceiling drops below 1 before any
capacity question arises. This module finds those collisions so the
headline "which storms/times fail" can separate CANNOT-memorise (input
collision) from DID-NOT-memorise (capacity/optimisation).

Run it once per scenario before training (notebook):

    from experiments.cyclone_jax.data.identifiability import input_collisions
    report = input_collisions(data.loader, data.splits['train'])
    report['max_accuracy']        # ceiling for the memorisation run
    report['conflicts']           # the ambiguous fix groups, with sids/times

Hashing covers the exact x dict the model sees (all arrays, byte-exact,
after selection/normalisation), so "identical" means identical to the
forward pass.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _input_hash(x: dict, fields=None) -> str:
    """Byte-exact content hash of one sample's x dict (key-ordered).

    ``fields`` restricts the hash to the x fields the MODEL consumes
    (encoding.fields + coords) — a field the model never sees must not
    make two fixes count as distinguishable."""
    h = hashlib.sha1()
    for k in sorted(x):
        if fields is not None and k not in fields:
            continue
        arr = np.ascontiguousarray(np.asarray(x[k]))
        h.update(k.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def input_collisions(loader, indices=None, fields=None) -> dict:
    """Group fixes by identical model input; flag different-target groups.

    Parameters
    ----------
    loader : Loader
        The scenario's loader (norms attached, as training will see it).
    indices : array-like, optional
        Fix indices to scan (a split); default = every fix.
    fields : iterable of str, optional
        x fields entering the hash — pass the model's consumed set
        (encoding.fields + 'lat'/'lon') for an honest ceiling; default =
        every x field (the most optimistic model).

    Returns
    -------
    dict with:
        n_fixes            fixes scanned
        n_unique_inputs    distinct input hashes
        collisions         groups sharing one input (list of dicts:
                           indices, sids, times, targets) — any target mix
        conflicts          the subset with >1 DISTINCT target (the ones
                           that cap memorisation)
        n_unmemorisable    fixes a deterministic model must get wrong
                           (group size - majority target count, summed)
        max_accuracy       (n_fixes - n_unmemorisable) / n_fixes
    """
    idx = (np.arange(len(loader)) if indices is None
           else np.asarray(indices))
    fields = set(fields) if fields is not None else None
    buckets: dict[str, list[int]] = {}
    targets: dict[int, int] = {}
    for i in idx:
        i = int(i)
        s = loader.build(i)
        buckets.setdefault(_input_hash(s['x'], fields), []).append(i)
        targets[i] = int(s['y']['target'])

    sids  = np.asarray(loader.fixes['sid'])
    times = np.asarray(loader.fixes['time'])

    def _group(members):
        return {'indices': list(members),
                'sids':    [str(sids[i]) for i in members],
                'times':   [str(times[i]) for i in members],
                'targets': [targets[i] for i in members]}

    collisions, n_unmemorisable = [], 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        g = _group(members)
        g['conflict'] = len(set(g['targets'])) > 1
        if g['conflict']:
            majority = max(np.bincount(g['targets']))
            n_unmemorisable += len(members) - int(majority)
        collisions.append(g)

    n = int(len(idx))
    return {
        'n_fixes':          n,
        'n_unique_inputs':  len(buckets),
        'collisions':       collisions,
        'conflicts':        [g for g in collisions if g['conflict']],
        'n_unmemorisable':  int(n_unmemorisable),
        'max_accuracy':     (n - n_unmemorisable) / n if n else 1.0,
    }
