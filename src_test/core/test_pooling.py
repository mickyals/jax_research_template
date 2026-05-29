# tests/core/test_pooling.py
import pytest
import jax
import jax.numpy as jnp

from core.pooling import (
    get_pooling,
    list_pooling,
    register_pooling,
    MeanPooling,
    MaxPooling,
    MinPooling,
    SumPooling,
    StdPooling,
    MeanMaxPooling,
    SpatialMaxPool,
    SpatialAvgPool,
    GlobalAvgPool,
    GlobalMaxPool,
)

KEY   = jax.random.PRNGKey(0)
BATCH = 8
N_OBS = 16
FEATS = 32
H, W  = 8, 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def x_set():
    """(batch, n_obs, features) -- set aggregation input."""
    return jax.random.normal(KEY, (BATCH, N_OBS, FEATS))


@pytest.fixture
def x_spatial():
    """(batch, height, width, channels) -- conv-style channels-last input."""
    return jax.random.normal(KEY, (BATCH, H, W, FEATS))


@pytest.fixture
def x_seq():
    """(batch, seq, features) -- 1D sequence input."""
    return jax.random.normal(KEY, (BATCH, N_OBS, FEATS))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_pooling_returns_all(self):
        pools = list_pooling()
        expected = {
            "MEAN", "MAX", "MIN", "SUM", "STD", "MEAN_MAX",
            "SPATIAL_MAX", "SPATIAL_AVG",
            "GLOBAL_AVG", "GLOBAL_MAX",
        }
        assert expected == set(pools.keys())

    def test_get_pooling_unknown_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            get_pooling("NONEXISTENT")

    def test_get_pooling_unknown_kwargs_warns(self):
        with pytest.warns(UserWarning, match="unknown kwargs"):
            get_pooling("MEAN", bogus=99)

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            @register_pooling("MEAN", description="duplicate")
            class Duplicate:
                def __call__(self, x, axis=1):
                    return x

    def test_get_pooling_valid_kwargs_applied(self):
        pool = get_pooling("SPATIAL_MAX", kernel_size=(3, 3), strides=(1, 1))
        assert pool.kernel_size == (3, 3)
        assert pool.strides == (1, 1)


# ---------------------------------------------------------------------------
# Axis-based reductions -- set aggregation
# ---------------------------------------------------------------------------

class TestMeanPooling:
    def test_set_aggregation(self, x_set):
        pool = get_pooling("MEAN")
        out = pool(x_set, axis=1)
        assert out.shape == (BATCH, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_spatial_reduction(self, x_spatial):
        pool = get_pooling("MEAN")
        out = pool(x_spatial, axis=(1, 2))
        assert out.shape == (BATCH, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_keepdims(self, x_set):
        pool = get_pooling("MEAN", keepdims=True)
        out = pool(x_set, axis=1)
        assert out.shape == (BATCH, 1, FEATS)

    def test_correct_value(self):
        x = jnp.ones((2, 4, 3))
        pool = get_pooling("MEAN")
        out = pool(x, axis=1)
        assert jnp.allclose(out, jnp.ones((2, 3)))


class TestMaxPooling:
    def test_set_aggregation(self, x_set):
        pool = get_pooling("MAX")
        out = pool(x_set, axis=1)
        assert out.shape == (BATCH, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_spatial_reduction(self, x_spatial):
        pool = get_pooling("MAX")
        out = pool(x_spatial, axis=(1, 2))
        assert out.shape == (BATCH, FEATS)

    def test_keepdims(self, x_set):
        pool = get_pooling("MAX", keepdims=True)
        out = pool(x_set, axis=1)
        assert out.shape == (BATCH, 1, FEATS)

    def test_correct_value(self):
        x = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
        pool = get_pooling("MAX")
        out = pool(x, axis=1)
        assert jnp.allclose(out, jnp.array([[3.0, 4.0]]))


class TestMinPooling:
    def test_set_aggregation(self, x_set):
        pool = get_pooling("MIN")
        out = pool(x_set, axis=1)
        assert out.shape == (BATCH, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_correct_value(self):
        x = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
        pool = get_pooling("MIN")
        out = pool(x, axis=1)
        assert jnp.allclose(out, jnp.array([[1.0, 2.0]]))


class TestSumPooling:
    def test_set_aggregation(self, x_set):
        pool = get_pooling("SUM")
        out = pool(x_set, axis=1)
        assert out.shape == (BATCH, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_correct_value(self):
        x = jnp.ones((2, 4, 3))
        pool = get_pooling("SUM")
        out = pool(x, axis=1)
        assert jnp.allclose(out, 4.0 * jnp.ones((2, 3)))


class TestStdPooling:
    def test_set_aggregation(self, x_set):
        pool = get_pooling("STD")
        out = pool(x_set, axis=1)
        assert out.shape == (BATCH, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_constant_input_zero_std(self):
        x = jnp.ones((2, 4, 3))
        pool = get_pooling("STD")
        out = pool(x, axis=1)
        assert jnp.allclose(out, jnp.zeros((2, 3)))


class TestMeanMaxPooling:
    def test_set_aggregation(self, x_set):
        pool = get_pooling("MEAN_MAX")
        out = pool(x_set, axis=1)
        assert out.shape == (BATCH, FEATS * 2)
        assert jnp.all(jnp.isfinite(out))

    def test_output_dim_doubled(self, x_set):
        pool = get_pooling("MEAN_MAX")
        out = pool(x_set, axis=1)
        assert out.shape[-1] == FEATS * 2

    def test_first_half_is_mean(self, x_set):
        pool = get_pooling("MEAN_MAX")
        out = pool(x_set, axis=1)
        mean = jnp.mean(x_set, axis=1)
        assert jnp.allclose(out[:, :FEATS], mean)

    def test_second_half_is_max(self, x_set):
        pool = get_pooling("MEAN_MAX")
        out = pool(x_set, axis=1)
        max_ = jnp.max(x_set, axis=1)
        assert jnp.allclose(out[:, FEATS:], max_)


# ---------------------------------------------------------------------------
# Spatial pooling
# ---------------------------------------------------------------------------

class TestSpatialMaxPool:
    def test_forward_default(self, x_spatial):
        pool = get_pooling("SPATIAL_MAX")
        out = pool(x_spatial)
        assert out.shape == (BATCH, H // 2, W // 2, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_custom_kernel(self, x_spatial):
        pool = get_pooling("SPATIAL_MAX", kernel_size=(3, 3), strides=(1, 1),
                           padding="SAME")
        out = pool(x_spatial)
        assert out.shape == (BATCH, H, W, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_stride_2(self, x_spatial):
        pool = get_pooling("SPATIAL_MAX", kernel_size=(2, 2), strides=(2, 2))
        out = pool(x_spatial)
        assert out.shape == (BATCH, H // 2, W // 2, FEATS)


class TestSpatialAvgPool:
    def test_forward_default(self, x_spatial):
        pool = get_pooling("SPATIAL_AVG")
        out = pool(x_spatial)
        assert out.shape == (BATCH, H // 2, W // 2, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_custom_kernel(self, x_spatial):
        pool = get_pooling("SPATIAL_AVG", kernel_size=(3, 3), strides=(1, 1),
                           padding="SAME")
        out = pool(x_spatial)
        assert out.shape == (BATCH, H, W, FEATS)

    def test_constant_input_preserved(self):
        x = jnp.ones((2, 4, 4, 8))
        pool = get_pooling("SPATIAL_AVG")
        out = pool(x)
        assert jnp.allclose(out, jnp.ones((2, 2, 2, 8)))


class TestGlobalAvgPool:
    def test_spatial_2d(self, x_spatial):
        pool = get_pooling("GLOBAL_AVG")
        out = pool(x_spatial)
        assert out.shape == (BATCH, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_sequence_1d(self, x_seq):
        pool = get_pooling("GLOBAL_AVG", spatial_axes=(1,))
        out = pool(x_seq)
        assert out.shape == (BATCH, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_correct_value(self):
        x = jnp.ones((2, 4, 4, 3))
        pool = get_pooling("GLOBAL_AVG")
        out = pool(x)
        assert jnp.allclose(out, jnp.ones((2, 3)))


class TestGlobalMaxPool:
    def test_spatial_2d(self, x_spatial):
        pool = get_pooling("GLOBAL_MAX")
        out = pool(x_spatial)
        assert out.shape == (BATCH, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_sequence_1d(self, x_seq):
        pool = get_pooling("GLOBAL_MAX", spatial_axes=(1,))
        out = pool(x_seq)
        assert out.shape == (BATCH, FEATS)
        assert jnp.all(jnp.isfinite(out))

    def test_correct_value(self):
        x = jnp.ones((2, 4, 4, 3)) * 5.0
        pool = get_pooling("GLOBAL_MAX")
        out = pool(x)
        assert jnp.allclose(out, 5.0 * jnp.ones((2, 3)))


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------

class TestCrossCutting:
    @pytest.mark.parametrize("name,kwargs", [
        ("MEAN",     {}),
        ("MAX",      {}),
        ("MIN",      {}),
        ("SUM",      {}),
        ("STD",      {}),
        ("MEAN_MAX", {}),
    ])
    def test_axis_reductions_finite(self, name, kwargs, x_set):
        pool = get_pooling(name, **kwargs)
        out = pool(x_set, axis=1)
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("name,kwargs", [
        ("MEAN",     {}),
        ("MAX",      {}),
        ("MIN",      {}),
        ("SUM",      {}),
        ("STD",      {}),
    ])
    def test_keepdims_shape(self, name, kwargs, x_set):
        pool = get_pooling(name, keepdims=True)
        out = pool(x_set, axis=1)
        assert out.shape == (BATCH, 1, FEATS)

    @pytest.mark.parametrize("name,kwargs", [
        ("SPATIAL_MAX", {}),
        ("SPATIAL_AVG", {}),
    ])
    def test_spatial_pools_finite(self, name, kwargs, x_spatial):
        pool = get_pooling(name, **kwargs)
        out = pool(x_spatial)
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("name", ["GLOBAL_AVG", "GLOBAL_MAX"])
    def test_global_pools_finite(self, name, x_spatial):
        pool = get_pooling(name)
        out = pool(x_spatial)
        assert jnp.all(jnp.isfinite(out))
        assert out.shape == (BATCH, FEATS)