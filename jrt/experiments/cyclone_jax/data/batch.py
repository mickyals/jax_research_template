"""
experiments/cyclone_jax/data/batch.py

Batching = stacking, nothing else. The sampler produces fixed-shape
(pad_to, TOKEN_DIM) samples, so collation is a literal np.stack and jit
sees one static shape.
"""

from __future__ import annotations

import numpy as np


def collate(samples):
    """Stack sample dicts -> {'X': {tokens, station_mask}, 'y', 'meta'}."""
    return {
        'X': {
            'tokens':       np.stack([s['tokens'] for s in samples]),
            'station_mask': np.stack([s['station_mask'] for s in samples]),
        },
        'y': np.array([s['label'] for s in samples], np.int32),
        'meta': {
            'sid':        [s['sid'] for s in samples],
            'time':       np.array([s['time'] for s in samples],
                                   dtype='datetime64[ns]'),
            'sshs':       np.array([s['sshs'] for s in samples], np.float32),
            'n_stations': np.array([s['n_stations'] for s in samples],
                                   np.int32),
        },
    }
