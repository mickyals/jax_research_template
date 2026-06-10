import numpy as np
import jax.numpy as jnp
import pytest

from utils.plotting.aggregation import (
    masked_mean_np,
    masked_mean_jax,
    bin_to_grid_np,
    bin_to_grid_jax,
    rolling_mean_np,
    take_slice_np,
)


class TestMaskedMeanNp:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        mask = np.array([True, False, True])
        assert masked_mean_np(x, mask) == pytest.approx(2.0)

    def test_all_masked_out(self):
        x = np.array([1.0, 2.0, 3.0])
        mask = np.array([False, False, False])
        assert masked_mean_np(x, mask) == pytest.approx(0.0)

    def test_all_valid(self):
        x = np.array([1.0, 2.0, 3.0])
        mask = np.ones_like(x, dtype=bool)
        assert masked_mean_np(x, mask) == pytest.approx(2.0)

    def test_axis_reduction(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        result = masked_mean_np(x, mask, axis=1)
        np.testing.assert_allclose(result, [1.0, 3.5])

    def test_axis_reduction_with_empty_row(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[False, False], [True, True]])
        result = masked_mean_np(x, mask, axis=1)
        np.testing.assert_allclose(result, [0.0, 3.5])


class TestMaskedMeanJax:
    def test_basic(self):
        x = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.array([True, False, True])
        assert float(masked_mean_jax(x, mask)) == pytest.approx(2.0)

    def test_all_masked_out(self):
        x = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.array([False, False, False])
        assert float(masked_mean_jax(x, mask)) == pytest.approx(0.0)

    def test_axis_reduction(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = jnp.array([[True, False], [True, True]])
        result = masked_mean_jax(x, mask, axis=1)
        np.testing.assert_allclose(np.asarray(result), [1.0, 3.5], rtol=1e-5)

    def test_agrees_with_numpy(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(5, 7))
        mask = rng.random((5, 7)) > 0.3
        np_result = masked_mean_np(x, mask)
        jax_result = float(masked_mean_jax(jnp.asarray(x), jnp.asarray(mask)))
        assert jax_result == pytest.approx(np_result, rel=1e-5)


class TestBinToGridNp:
    def test_mean_reduction(self):
        x = np.array([0.1, 0.1, 0.9])
        y = np.array([0.1, 0.1, 0.9])
        v = np.array([1.0, 3.0, 5.0])
        grid, extent = bin_to_grid_np(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2))
        assert extent == [0, 1, 0, 1]
        np.testing.assert_allclose(grid[0, 0], 2.0)
        np.testing.assert_allclose(grid[1, 1], 5.0)
        assert np.isnan(grid[0, 1])
        assert np.isnan(grid[1, 0])

    def test_count_reduction(self):
        x = np.array([0.1, 0.1, 0.9])
        y = np.array([0.1, 0.1, 0.9])
        v = np.array([1.0, 3.0, 5.0])
        grid, _ = bin_to_grid_np(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2))
        count_grid, _ = bin_to_grid_np(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2), reduce="count")
        assert count_grid[0, 0] == 2.0
        assert count_grid[1, 1] == 1.0
        assert count_grid[0, 1] == 0.0
        assert count_grid[1, 0] == 0.0
        assert grid.shape == count_grid.shape

    def test_max_reduction(self):
        x = np.array([0.1, 0.1])
        y = np.array([0.1, 0.1])
        v = np.array([1.0, 3.0])
        grid, _ = bin_to_grid_np(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2), reduce="max")
        assert grid[0, 0] == 3.0

    def test_out_of_range_points_dropped(self):
        x = np.array([0.1, 5.0])
        y = np.array([0.1, 5.0])
        v = np.array([1.0, 100.0])
        grid, _ = bin_to_grid_np(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2), reduce="count")
        assert grid.sum() == 1.0

    def test_invalid_reduce_raises(self):
        x = np.array([0.1])
        y = np.array([0.1])
        v = np.array([1.0])
        with pytest.raises(ValueError):
            bin_to_grid_np(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2), reduce="median")


class TestBinToGridJax:
    def test_mean_reduction(self):
        x = jnp.array([0.1, 0.1, 0.9])
        y = jnp.array([0.1, 0.1, 0.9])
        v = jnp.array([1.0, 3.0, 5.0])
        grid, extent = bin_to_grid_jax(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2))
        grid = np.asarray(grid)
        np.testing.assert_allclose(grid[0, 0], 2.0)
        np.testing.assert_allclose(grid[1, 1], 5.0)
        assert np.isnan(grid[0, 1])
        assert np.isnan(grid[1, 0])

    def test_count_reduction(self):
        x = jnp.array([0.1, 0.1, 0.9])
        y = jnp.array([0.1, 0.1, 0.9])
        v = jnp.array([1.0, 3.0, 5.0])
        grid, _ = bin_to_grid_jax(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2), reduce="count")
        grid = np.asarray(grid)
        assert grid[0, 0] == 2.0
        assert grid[1, 1] == 1.0
        assert grid[0, 1] == 0.0
        assert grid[1, 0] == 0.0

    def test_max_reduction(self):
        x = jnp.array([0.1, 0.1])
        y = jnp.array([0.1, 0.1])
        v = jnp.array([1.0, 3.0])
        grid, _ = bin_to_grid_jax(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2), reduce="max")
        assert float(grid[0, 0]) == pytest.approx(3.0)

    def test_out_of_range_points_clipped_to_edge(self):
        # Unlike bin_to_grid_np, out-of-range points are clipped into the
        # nearest edge bin rather than dropped (static-shape requirement).
        x = jnp.array([0.1, 5.0])
        y = jnp.array([0.1, 5.0])
        v = jnp.array([1.0, 100.0])
        grid, _ = bin_to_grid_jax(x, y, v, extent=[0, 1, 0, 1], shape=(2, 2), reduce="count")
        assert float(jnp.sum(grid)) == 2.0

    def test_agrees_with_numpy_in_range(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 50)
        y = rng.uniform(0, 1, 50)
        v = rng.normal(size=50)
        grid_np, _ = bin_to_grid_np(x, y, v, extent=[0, 1, 0, 1], shape=(4, 4))
        grid_jax, _ = bin_to_grid_jax(jnp.asarray(x), jnp.asarray(y), jnp.asarray(v), extent=[0, 1, 0, 1], shape=(4, 4))
        grid_jax = np.asarray(grid_jax)
        np.testing.assert_allclose(np.nan_to_num(grid_np), np.nan_to_num(grid_jax), rtol=1e-4, atol=1e-5)


class TestRollingMeanNp:
    def test_constant_array(self):
        values = np.full(10, 3.0)
        result = rolling_mean_np(values, window=3)
        np.testing.assert_allclose(result, np.full(10, 3.0))

    def test_output_aligned_to_input(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_mean_np(values, window=3)
        assert result.shape == values.shape

    def test_known_values(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_mean_np(values, window=3)
        np.testing.assert_allclose(result, [1.5, 2.0, 3.0, 4.0, 4.5])

    def test_window_one_is_identity(self):
        values = np.array([1.0, 2.0, 3.0])
        result = rolling_mean_np(values, window=1)
        np.testing.assert_allclose(result, values)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            rolling_mean_np(np.array([1.0, 2.0]), window=0)


class TestTakeSliceNp:
    def test_basic_slice(self):
        volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        slc = take_slice_np(volume, axis=2, index=1)
        np.testing.assert_array_equal(slc, volume[:, :, 1])
        assert slc.shape == (2, 3)

    def test_negative_index(self):
        volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        slc = take_slice_np(volume, axis=2, index=-1)
        np.testing.assert_array_equal(slc, volume[:, :, -1])

    def test_invalid_axis_raises(self):
        volume = np.zeros((2, 3, 4))
        with pytest.raises(ValueError):
            take_slice_np(volume, axis=3, index=0)

    def test_index_out_of_bounds_raises(self):
        volume = np.zeros((2, 3, 4))
        with pytest.raises(IndexError):
            take_slice_np(volume, axis=2, index=10)
