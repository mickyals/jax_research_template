"""
Shared fake device batch for cyclone_jax model tests — the batching.collate
X schema (named fields + station_mask) with tiny, distinctive shapes.
"""

import numpy as np
import pytest

B, N, C = 2, 5, 3


@pytest.fixture()
def X():
    """Named X dict shaped like a collated batch: B=2, N=5, C=3.

    Each field is a distinct constant so packing-order tests can identify
    columns; the last station slot of sample 0 is padding (mask False).
    """
    mask = np.ones((B, N), bool)
    mask[0, -1] = False
    return {
        'lat':          np.full((B, N), 10.0, np.float32),
        'lon':          np.full((B, N), -60.0, np.float32),
        'level':        np.full((B, N), 2.0, np.float32),
        'time':         np.full((B, N), 3.0, np.float32),
        'id':           np.full((B, N), 4.0, np.float32),
        'obs':          np.full((B, N, C), 5.0, np.float32),
        'missing':      np.ones((B, N, C), bool),
        'station_mask': mask,
    }
