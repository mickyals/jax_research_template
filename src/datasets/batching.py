"""
datasets/batching.py

JAX-native batching utilities for training loops.

All functions operate on plain arrays or dicts of arrays — there is no
dependency on any specific dataset class. The trainer passes a data dict
(e.g. dataset._data or a subset returned by to_Xy) and gets back an
iterator of batches for one epoch.

Design notes
------------
- Shuffling is done with jax.random.permutation so it is reproducible
  and can be seeded per-epoch (pass rng=jax.random.fold_in(base_rng, epoch)).
- Arrays are kept as JAX arrays throughout — no host copies.
- drop_last=True (default) keeps all batches the same size, which is
  required for JIT-compiled train steps that assume a fixed batch shape.
- Arrays remain on whatever device they were on — batching is purely
  an indexing operation.

Functions
---------
    shuffle_arrays        shuffle a dict of arrays with one shared permutation
    num_batches           how many full batches fit in n samples
    epoch_iterator        yield batches for one epoch (primary entry point)
    as_batches            split pre-shuffled arrays into a list of batches
"""

from __future__ import annotations

from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def shuffle_arrays(
    arrays: dict[str, jax.Array] | jax.Array,
    rng:    jax.Array,
) -> dict[str, jax.Array] | jax.Array:
    """Shuffle all arrays along axis 0 with the same permutation.

    Applying a single permutation to a dict of arrays ensures that row
    correspondence is preserved (e.g. features and targets stay aligned).

    Parameters
    ----------
    arrays : dict[str, jax.Array] or jax.Array
        A single array or a dict of arrays, all with the same leading
        dimension (number of samples).
    rng : jax.Array
        JAX PRNGKey used to generate the permutation.

    Returns
    -------
    dict[str, jax.Array] or jax.Array
        Shuffled arrays in the same container type as the input.

    Example
    -------
    >>> rng = jax.random.PRNGKey(0)
    >>> data = {'X': jnp.arange(6).reshape(3, 2), 'y': jnp.array([10, 20, 30])}
    >>> shuffled = shuffle_arrays(data, rng)
    >>> shuffled['X'].shape
    (3, 2)
    """
    if isinstance(arrays, dict):
        n     = next(iter(arrays.values())).shape[0]
        perm  = jax.random.permutation(rng, n)
        return {k: v[perm] for k, v in arrays.items()}
    else:
        n    = arrays.shape[0]
        perm = jax.random.permutation(rng, n)
        return arrays[perm]


def num_batches(n: int, batch_size: int, drop_last: bool = True) -> int:
    """Number of batches that fit in n samples.

    Parameters
    ----------
    n : int
        Total number of samples.
    batch_size : int
        Batch size.
    drop_last : bool
        If True (default), the last incomplete batch is dropped and the
        return value is n // batch_size. If False, the partial batch is
        counted, giving ceil(n / batch_size).

    Returns
    -------
    int

    Example
    -------
    >>> num_batches(100, 32, drop_last=True)
    3
    >>> num_batches(100, 32, drop_last=False)
    4
    """
    if drop_last:
        return n // batch_size
    return int(np.ceil(n / batch_size))


def as_batches(
    arrays:     dict[str, jax.Array] | jax.Array,
    batch_size: int,
    drop_last:  bool = True,
) -> list[dict[str, jax.Array] | jax.Array]:
    """Split arrays into a list of fixed-size batches.

    Assumes arrays are already shuffled. Typically called from
    epoch_iterator rather than directly.

    Parameters
    ----------
    arrays : dict[str, jax.Array] or jax.Array
        Already-shuffled arrays with the same leading dimension.
    batch_size : int
        Number of samples per batch.
    drop_last : bool
        If True (default), the last partial batch is discarded so every
        batch has exactly batch_size rows.

    Returns
    -------
    list
        List of batches in the same container type as the input.

    Example
    -------
    >>> data = {'X': jnp.arange(10).reshape(5, 2), 'y': jnp.arange(5)}
    >>> batches = as_batches(data, batch_size=2)
    >>> len(batches)
    2
    """
    if isinstance(arrays, dict):
        n = next(iter(arrays.values())).shape[0]
    else:
        n = arrays.shape[0]

    n_full = num_batches(n, batch_size, drop_last=True)
    batches = []
    for i in range(n_full):
        start = i * batch_size
        end   = start + batch_size
        if isinstance(arrays, dict):
            batches.append({k: v[start:end] for k, v in arrays.items()})
        else:
            batches.append(arrays[start:end])

    if not drop_last and n % batch_size != 0:
        start = n_full * batch_size
        if isinstance(arrays, dict):
            batches.append({k: v[start:] for k, v in arrays.items()})
        else:
            batches.append(arrays[start:])

    return batches


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def epoch_iterator(
    arrays:     dict[str, jax.Array] | jax.Array,
    batch_size: int,
    rng:        jax.Array,
    drop_last:  bool = True,
) -> Iterator[dict[str, jax.Array] | jax.Array]:
    """Shuffle and yield batches for one training epoch.

    This is the function the trainer calls each epoch. Pass a different
    rng each epoch to get a different shuffle order — a clean pattern is:

        base_rng = jax.random.PRNGKey(seed)
        for epoch in range(num_epochs):
            epoch_rng = jax.random.fold_in(base_rng, epoch)
            for batch in epoch_iterator(train_data, batch_size, epoch_rng):
                ...

    Parameters
    ----------
    arrays : dict[str, jax.Array] or jax.Array
        Full dataset arrays. All values must share the same leading
        dimension (number of samples).
    batch_size : int
        Number of samples per batch.
    rng : jax.Array
        JAX PRNGKey for this epoch's shuffle. Use jax.random.fold_in
        with the epoch index for reproducible per-epoch shuffles.
    drop_last : bool
        If True (default), the last incomplete batch is dropped. Set
        False only for evaluation loops where you need every sample.

    Yields
    ------
    dict[str, jax.Array] or jax.Array
        One batch at a time, same container type as the input.

    Example
    -------
    >>> rng  = jax.random.PRNGKey(0)
    >>> data = {'X': jnp.ones((100, 4)), 'y': jnp.ones((100, 1))}
    >>> batches = list(epoch_iterator(data, batch_size=32, rng=rng))
    >>> len(batches)           # 100 // 32 = 3 full batches
    3
    >>> batches[0]['X'].shape
    (32, 4)

    >>> # Per-epoch shuffle pattern
    >>> base_rng = jax.random.PRNGKey(42)
    >>> for epoch in range(3):
    ...     for batch in epoch_iterator(data, 32, jax.random.fold_in(base_rng, epoch)):
    ...         pass  # train step here
    """
    shuffled = shuffle_arrays(arrays, rng)
    yield from as_batches(shuffled, batch_size, drop_last=drop_last)
