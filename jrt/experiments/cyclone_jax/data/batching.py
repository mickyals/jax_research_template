"""
experiments/cyclone_jax/data/batching.py

Collate = pad + stack, and the ONLY translator between the sample schema
and the device batch schema:

    {'x': named ragged fields, 'y': named dict}       (sampler.Loader)
      -> {'X':    {field: (B, pad_to, ...)} + station_mask (B, pad_to),
          'y':    (B,) stacked y['target'],
          'meta': remaining y fields + n_stations}

pad_to is a FIXED config value (InputSpec.pad_to) — never the batch max —
so the jitted train step compiles once. Samples longer than pad_to are
truncated (size pad_to to the library maximum to make that impossible;
meta['n_stations'] == pad_to is the truncation tell). batch['meta'] holds
everything jit must never trace (strings, absolute times); the jrt
Trainer drops it before tracing (_to_jax_batch).

Generic over field names: whatever x carries is padded+stacked by dtype
and trailing shape; whatever y carries besides 'target' becomes meta.
"""

from __future__ import annotations

import numpy as np


def collate(samples, pad_to):
    """Stack ragged {'x', 'y'} samples into one fixed-shape batch dict."""
    B = len(samples)
    lengths = [len(next(iter(s['x'].values()))) for s in samples]

    X = {}
    for f, v0 in samples[0]['x'].items():
        out = np.zeros((B, pad_to) + v0.shape[1:], v0.dtype)
        for b, s in enumerate(samples):
            n = min(lengths[b], pad_to)
            out[b, :n] = s['x'][f][:n]
        X[f] = out

    station_mask = np.zeros((B, pad_to), bool)
    for b, n in enumerate(lengths):
        station_mask[b, :min(n, pad_to)] = True
    X['station_mask'] = station_mask

    meta = {}
    for f in samples[0]['y']:
        if f == 'target':
            continue
        vals = [s['y'][f] for s in samples]
        meta[f] = vals if isinstance(vals[0], str) else np.asarray(vals)
    meta['n_stations'] = np.array([min(n, pad_to) for n in lengths],
                                  np.int32)

    return {
        'X': X,
        'y': np.asarray([s['y']['target'] for s in samples]),
        'meta': meta,
    }
