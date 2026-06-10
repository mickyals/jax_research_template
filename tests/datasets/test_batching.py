"""
Tests for datasets/batching.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datasets.batching import (
    shuffle_arrays,
    num_batches,
    as_batches,
    epoch_iterator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def dict_data():
    """10 samples, 2 features + 1 target. X[:,0] == y (easy correspondence check)."""
    X = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)
    y = jnp.arange(10, dtype=jnp.float32)
    return {"X": X, "y": y}


@pytest.fixture
def arr_data():
    return jnp.arange(20, dtype=jnp.float32).reshape(10, 2)


# ---------------------------------------------------------------------------
# shuffle_arrays
# ---------------------------------------------------------------------------

class TestShuffleArrays:

    def test_dict_shape_preserved(self, dict_data, rng):
        out = shuffle_arrays(dict_data, rng)
        assert out["X"].shape == (10, 2)
        assert out["y"].shape == (10,)

    def test_bare_array_shape_preserved(self, arr_data, rng):
        out = shuffle_arrays(arr_data, rng)
        assert out.shape == (10, 2)

    def test_row_correspondence_preserved(self, dict_data, rng):
        """X[:,0] should still equal y * 2 after shuffling."""
        out = shuffle_arrays(dict_data, rng)
        assert jnp.allclose(out["X"][:, 0], out["y"] * 2)

    def test_all_rows_present(self, dict_data, rng):
        out = shuffle_arrays(dict_data, rng)
        assert set(out["y"].tolist()) == set(range(10))

    def test_different_rngs_give_different_orders(self, dict_data):
        rng_a = jax.random.PRNGKey(0)
        rng_b = jax.random.PRNGKey(1)
        a = shuffle_arrays(dict_data, rng_a)["y"]
        b = shuffle_arrays(dict_data, rng_b)["y"]
        assert not jnp.allclose(a, b)

    def test_same_rng_gives_same_order(self, dict_data):
        rng = jax.random.PRNGKey(42)
        a = shuffle_arrays(dict_data, rng)["y"]
        b = shuffle_arrays(dict_data, rng)["y"]
        assert jnp.allclose(a, b)


# ---------------------------------------------------------------------------
# num_batches
# ---------------------------------------------------------------------------

class TestNumBatches:

    def test_exact_division_drop_last(self):
        assert num_batches(9, 3, drop_last=True) == 3

    def test_remainder_dropped(self):
        assert num_batches(10, 3, drop_last=True) == 3

    def test_remainder_kept(self):
        assert num_batches(10, 3, drop_last=False) == 4

    def test_exact_division_keep_last(self):
        assert num_batches(9, 3, drop_last=False) == 3

    def test_larger_batch_than_data(self):
        assert num_batches(5, 10, drop_last=True) == 0
        assert num_batches(5, 10, drop_last=False) == 1

    def test_batch_size_one(self):
        assert num_batches(7, 1, drop_last=True) == 7


# ---------------------------------------------------------------------------
# as_batches
# ---------------------------------------------------------------------------

class TestAsBatches:

    def test_count_drop_last(self, dict_data):
        batches = as_batches(dict_data, batch_size=3, drop_last=True)
        assert len(batches) == 3  # 10 // 3

    def test_count_keep_last(self, dict_data):
        batches = as_batches(dict_data, batch_size=3, drop_last=False)
        assert len(batches) == 4  # ceil(10 / 3)

    def test_full_batch_shape(self, dict_data):
        batches = as_batches(dict_data, batch_size=3)
        assert all(b["X"].shape == (3, 2) for b in batches)
        assert all(b["y"].shape == (3,)   for b in batches)

    def test_partial_batch_shape(self, dict_data):
        batches = as_batches(dict_data, batch_size=3, drop_last=False)
        assert batches[-1]["X"].shape == (1, 2)  # 10 % 3 = 1

    def test_no_data_lost_keep_last(self, dict_data):
        batches = as_batches(dict_data, batch_size=3, drop_last=False)
        total = sum(b["X"].shape[0] for b in batches)
        assert total == 10

    def test_data_lost_drop_last(self, dict_data):
        batches = as_batches(dict_data, batch_size=3, drop_last=True)
        total = sum(b["X"].shape[0] for b in batches)
        assert total == 9  # last row dropped

    def test_bare_array(self, arr_data):
        batches = as_batches(arr_data, batch_size=4, drop_last=True)
        assert len(batches) == 2
        assert all(b.shape == (4, 2) for b in batches)

    def test_empty_when_batch_larger_than_data(self, dict_data):
        batches = as_batches(dict_data, batch_size=20, drop_last=True)
        assert len(batches) == 0

    def test_single_batch_when_batch_equals_data(self, dict_data):
        batches = as_batches(dict_data, batch_size=10, drop_last=True)
        assert len(batches) == 1
        assert batches[0]["X"].shape == (10, 2)


# ---------------------------------------------------------------------------
# epoch_iterator
# ---------------------------------------------------------------------------

class TestEpochIterator:

    def test_yields_correct_count(self, dict_data, rng):
        batches = list(epoch_iterator(dict_data, batch_size=3, rng=rng))
        assert len(batches) == 3  # 10 // 3

    def test_batch_shapes(self, dict_data, rng):
        for batch in epoch_iterator(dict_data, batch_size=3, rng=rng):
            assert batch["X"].shape == (3, 2)
            assert batch["y"].shape == (3,)

    def test_row_correspondence_in_batches(self, dict_data, rng):
        """X[:,0] == y * 2 must hold within every batch."""
        for batch in epoch_iterator(dict_data, batch_size=3, rng=rng):
            assert jnp.allclose(batch["X"][:, 0], batch["y"] * 2)

    def test_different_epochs_different_order(self, dict_data):
        base = jax.random.PRNGKey(0)
        rng0 = jax.random.fold_in(base, 0)
        rng1 = jax.random.fold_in(base, 1)
        first_ep0 = list(epoch_iterator(dict_data, 3, rng0))[0]["y"]
        first_ep1 = list(epoch_iterator(dict_data, 3, rng1))[0]["y"]
        assert not jnp.allclose(first_ep0, first_ep1)

    def test_same_rng_reproducible(self, dict_data, rng):
        a = [b["y"] for b in epoch_iterator(dict_data, 3, rng)]
        b = [b["y"] for b in epoch_iterator(dict_data, 3, rng)]
        for ai, bi in zip(a, b):
            assert jnp.allclose(ai, bi)

    def test_bare_array(self, arr_data, rng):
        batches = list(epoch_iterator(arr_data, batch_size=4, rng=rng))
        assert len(batches) == 2
        assert batches[0].shape == (4, 2)

    def test_drop_last_false_for_eval(self, dict_data, rng):
        batches = list(epoch_iterator(dict_data, batch_size=3, rng=rng,
                                      drop_last=False))
        assert len(batches) == 4
        total = sum(b["X"].shape[0] for b in batches)
        assert total == 10

    def test_fold_in_pattern(self, dict_data):
        """Canonical per-epoch shuffle pattern should not error."""
        base_rng = jax.random.PRNGKey(42)
        for epoch in range(5):
            epoch_rng = jax.random.fold_in(base_rng, epoch)
            batches = list(epoch_iterator(dict_data, batch_size=5, rng=epoch_rng))
            assert len(batches) == 2
