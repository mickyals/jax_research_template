"""Tests for training/prefetch.py — generic multiprocess batch prefetcher.

Fork-guarded: the worker path is exercised on POSIX (fork). The deploy target is
Linux; on Windows the synchronous path (num_workers=0) is what runs.
"""

import os
import numpy as np
import pytest

from training.prefetch import ProcessPrefetcher

_FORK = hasattr(os, "fork")


def _dummy_worker(worker_id, num_workers):
    """Yield 3 NumPy batches per worker, tagged with the worker id."""
    for i in range(3):
        yield {"wid": worker_id, "i": i, "x": np.full(4, worker_id, dtype=np.int32)}


@pytest.mark.skipif(not _FORK, reason="fork-only multiprocessing test")
class TestProcessPrefetcher:

    def test_yields_all_batches(self):
        out = list(ProcessPrefetcher(_dummy_worker, num_workers=2, prefetch_factor=2))
        assert len(out) == 2 * 3                       # all workers drained
        assert {b["wid"] for b in out} == {0, 1}       # both workers ran

    def test_to_device_applied_in_main(self):
        out = list(ProcessPrefetcher(
            _dummy_worker, num_workers=2,
            to_device=lambda b: {**b, "tagged": True}))
        assert len(out) == 6
        assert all(b.get("tagged") for b in out)

    def test_single_worker(self):
        out = list(ProcessPrefetcher(_dummy_worker, num_workers=1))
        assert len(out) == 3

    def test_invalid_num_workers(self):
        with pytest.raises(ValueError):
            ProcessPrefetcher(_dummy_worker, num_workers=0)

    def test_reusable(self):
        pf = ProcessPrefetcher(_dummy_worker, num_workers=2)
        assert len(list(pf)) == 6
        assert len(list(pf)) == 6                       # second pass: fresh pool
