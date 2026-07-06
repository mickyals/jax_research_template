"""
Shared fakes for cyclone_jax train-side tests (log / train / evaluate):
the loader/stream/logger surfaces those modules touch, without the
arcana library. Figure LOOKS are covered in tests/.../visualise — these
exist for wiring tests. test_identifiability keeps its own bespoke
FakeLoader (its collision layout IS that test's design).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

import numpy as np

N_CLS = 6           # matches TargetSpec class_set (3..8) in the tests


class FakeStream:
    """Explicit-epoch batch stream; iter() is forbidden on purpose —
    callbacks must never advance the shared stream's epoch counter."""

    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def epoch(self, e):
        return iter(self._batches)

    def __iter__(self):
        raise AssertionError('callbacks must use explicit .epoch(e)')


class FakeLogger:
    def __init__(self):
        self.metrics, self.figures, self.titles = [], [], []
        self.artifacts = []

    def log_metrics(self, m, step):
        self.metrics.append((dict(m), step))

    def log_figure(self, tag, fig, step):
        self.figures.append((tag, step))
        self.titles.append(fig.axes[0].get_title())
        plt.close(fig)

    def log_artifact(self, name, path, artifact_type='profile'):
        self.artifacts.append((name, str(path), artifact_type))


class FakeLoader:
    """loader surface storm_panel/end_of_run touch: fixes['sid'/'time'],
    build(i) -> collatable sample, inputs.pad_to."""

    def __init__(self, sids=('S0', 'S1', 'S2', 'S3'), times=None):
        if times is None:
            times = [np.datetime64('2024-07-01T00') + np.timedelta64(6 * i, 'h')
                     for i in range(len(sids))]
        self.fixes = {'sid': np.asarray(sids), 'time': np.asarray(times)}
        self.inputs = SimpleNamespace(pad_to=8)

    def build(self, i):
        return {
            'x': {'lat': np.float32([10.0 + i, 11.0, 12.0]),
                  'lon': np.float32([-60.0, -61.0, -62.0]),
                  'id':  np.float32([-1.0, 1.0, 0.0])},
            'y': {'target': np.int32(i % N_CLS),
                  'sid':    str(self.fixes['sid'][i]),
                  'lat':    np.float32(13.0),
                  'lon':    np.float32(-58.0),
                  'time':   self.fixes['time'][i]},
        }
