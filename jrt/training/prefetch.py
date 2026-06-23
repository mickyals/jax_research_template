"""
training/prefetch.py

Generic multiprocess batch prefetcher — parallelises CPU-bound sample assembly
across worker processes while the main process keeps feeding the (jitted) train
step. Dataset-agnostic: it owns the worker pool, the bounded queue, sentinel
bookkeeping, and graceful shutdown; the caller supplies a ``worker_fn`` that
yields one worker's DISJOINT SHARD of batches, and an optional ``to_device`` that
moves a batch onto the accelerator in the MAIN process.

Why processes, not threads: the per-sample work is NumPy-heavy assembly, not
Python compute, but the GIL still serialises threads here — and, more
importantly, workers must NOT touch JAX/CUDA (forking after CUDA init is unsafe).
So workers stay pure-NumPy and produce NumPy batches; device transfer happens
only in the main process via ``to_device``.

Start method: defaults to ``fork`` where available (Linux — workers cheaply
inherit any mmap'd dataset), else ``spawn`` (the worker_fn and its captured state
must then be picklable).
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Iterator, Optional


def _worker_loop(worker_fn, worker_id: int, num_workers: int, queue) -> None:
    """Run one worker's shard, pushing each batch onto the queue.

    A ``None`` sentinel is always pushed last (even on error), so the consumer
    can count completed workers and never hang.
    """
    try:
        for batch in worker_fn(worker_id, num_workers):
            queue.put(batch)
    finally:
        queue.put(None)


class ProcessPrefetcher:
    """Iterable that drains ``num_workers`` processes producing batches.

    Parameters
    ----------
    worker_fn : callable(worker_id, num_workers) -> iterator of batches
        Produces this worker's disjoint shard. Must be importable/picklable for
        the ``spawn`` start method; any callable for ``fork``.
    num_workers : int
        Number of worker processes (>= 1).
    prefetch_factor : int
        Queue capacity is ``num_workers * prefetch_factor`` — bounded, so workers
        block (backpressure) rather than running unboundedly ahead. Default 2.
    to_device : callable(batch) -> batch, optional
        Applied in the MAIN process to each batch (e.g. NumPy -> device arrays).
    start_method : {'fork', 'spawn', 'forkserver'}, optional
        Defaults to 'fork' when available, else 'spawn'.
    """

    def __init__(
        self,
        worker_fn:       Callable[[int, int], Iterator],
        num_workers:     int,
        prefetch_factor: int = 2,
        to_device:       Optional[Callable] = None,
        start_method:    Optional[str] = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self._worker_fn = worker_fn
        self._n         = int(num_workers)
        self._pf        = max(1, int(prefetch_factor))
        self._to_device = to_device
        self._start_method = start_method or (
            "fork" if "fork" in mp.get_all_start_methods() else "spawn")

    def __iter__(self) -> Iterator:
        ctx   = mp.get_context(self._start_method)
        queue = ctx.Queue(maxsize=self._n * self._pf)
        procs = [
            ctx.Process(target=_worker_loop,
                        args=(self._worker_fn, wid, self._n, queue),
                        daemon=True)
            for wid in range(self._n)
        ]
        for p in procs:
            p.start()
        done = 0
        try:
            while done < self._n:
                item = queue.get()
                if item is None:           # a worker finished its shard
                    done += 1
                    continue
                yield self._to_device(item) if self._to_device else item
        finally:
            # Graceful shutdown: kill any stragglers (so a killed/erroring main
            # never leaves orphan workers) and reap them all.
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join()
