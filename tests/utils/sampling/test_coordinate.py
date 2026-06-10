import pytest
import jax
import jax.numpy as jnp
import numpy as np

from utils.sampling.coordinate import (
    sample_regional,
    sample_sphere_uniform_area,
    sample_sphere_uniform_angle,
    sample_volume,
    lhs_sample,
    lhs_sample_regional,
    lhs_sample_volume,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def volume_bounds():
    return jnp.array([[-1., 1.], [-1., 1.], [0., 1.]])


@pytest.fixture
def regional_bounds():
    return (-100., -40.), (0., 30.)


# ---------------------------------------------------------------------------
# sample_regional
# ---------------------------------------------------------------------------

class TestSampleRegional:

    def test_output_shapes(self, key, regional_bounds):
        lon_b, lat_b = regional_bounds
        lons, lats = sample_regional(key, 100, lon_b, lat_b)
        assert lons.shape == (100,)
        assert lats.shape == (100,)

    def test_lons_in_bounds(self, key, regional_bounds):
        lon_b, lat_b = regional_bounds
        lons, _ = sample_regional(key, 500, lon_b, lat_b)
        assert jnp.all(lons >= lon_b[0])
        assert jnp.all(lons <= lon_b[1])

    def test_lats_in_bounds(self, key, regional_bounds):
        lon_b, lat_b = regional_bounds
        _, lats = sample_regional(key, 500, lon_b, lat_b)
        assert jnp.all(lats >= lat_b[0])
        assert jnp.all(lats <= lat_b[1])

    def test_reproducibility(self, regional_bounds):
        lon_b, lat_b = regional_bounds
        key = jax.random.PRNGKey(0)
        lons1, lats1 = sample_regional(key, 50, lon_b, lat_b)
        lons2, lats2 = sample_regional(key, 50, lon_b, lat_b)
        assert jnp.allclose(lons1, lons2)
        assert jnp.allclose(lats1, lats2)

    def test_different_keys_differ(self, regional_bounds):
        lon_b, lat_b = regional_bounds
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        lons1, _ = sample_regional(k1, 50, lon_b, lat_b)
        lons2, _ = sample_regional(k2, 50, lon_b, lat_b)
        assert not jnp.allclose(lons1, lons2)

    def test_lons_lats_independent(self, key, regional_bounds):
        lon_b, lat_b = regional_bounds
        lons, lats = sample_regional(key, 200, lon_b, lat_b)
        assert not jnp.allclose(lons, lats)


# ---------------------------------------------------------------------------
# sample_sphere_uniform_area
# ---------------------------------------------------------------------------

class TestSampleSphereUniformArea:

    def test_output_shapes(self, key):
        lat, lon = sample_sphere_uniform_area(key, 500)
        assert lat.shape == (500,)
        assert lon.shape == (500,)

    def test_lat_range(self, key):
        lat, _ = sample_sphere_uniform_area(key, 1000)
        assert jnp.all(lat >= -jnp.pi / 2)
        assert jnp.all(lat <= jnp.pi / 2)

    def test_lon_range(self, key):
        _, lon = sample_sphere_uniform_area(key, 1000)
        assert jnp.all(lon >= -jnp.pi)
        assert jnp.all(lon <= jnp.pi)

    def test_area_uniform_sin_lat(self):
        # sin(lat) should be approximately uniform => mean near 0
        key = jax.random.PRNGKey(0)
        lat, _ = sample_sphere_uniform_area(key, 50_000)
        assert abs(float(jnp.sin(lat).mean())) < 0.02

    def test_reproducibility(self):
        key = jax.random.PRNGKey(7)
        lat1, lon1 = sample_sphere_uniform_area(key, 100)
        lat2, lon2 = sample_sphere_uniform_area(key, 100)
        assert jnp.allclose(lat1, lat2)
        assert jnp.allclose(lon1, lon2)

    def test_different_keys_differ(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        lat1, _ = sample_sphere_uniform_area(k1, 100)
        lat2, _ = sample_sphere_uniform_area(k2, 100)
        assert not jnp.allclose(lat1, lat2)


# ---------------------------------------------------------------------------
# sample_sphere_uniform_angle
# ---------------------------------------------------------------------------

class TestSampleSphereUniformAngle:

    def test_output_shapes(self, key):
        lat, lon = sample_sphere_uniform_angle(key, 500)
        assert lat.shape == (500,)
        assert lon.shape == (500,)

    def test_lat_range(self, key):
        lat, _ = sample_sphere_uniform_angle(key, 1000)
        assert jnp.all(lat >= -jnp.pi / 2)
        assert jnp.all(lat <= jnp.pi / 2)

    def test_lon_range(self, key):
        _, lon = sample_sphere_uniform_angle(key, 1000)
        assert jnp.all(lon >= -jnp.pi)
        assert jnp.all(lon <= jnp.pi)

    def test_not_area_uniform(self):
        # sin(lat) mean should deviate from 0 -- poles oversampled
        # with enough samples this reliably fails the area-uniform check
        key = jax.random.PRNGKey(0)
        lat, _ = sample_sphere_uniform_angle(key, 50_000)
        # lat is uniform in [-pi/2, pi/2] so sin(lat) mean ~ 0 too,
        # but variance of sin(lat) differs from area-uniform
        # key distinction: lat itself should be uniform
        lat_np = np.array(lat)
        hist, _ = np.histogram(lat_np, bins=10)
        # all bins should be roughly equal (uniform in angle)
        assert hist.std() / hist.mean() < 0.05

    def test_reproducibility(self):
        key = jax.random.PRNGKey(3)
        lat1, _ = sample_sphere_uniform_angle(key, 100)
        lat2, _ = sample_sphere_uniform_angle(key, 100)
        assert jnp.allclose(lat1, lat2)


# ---------------------------------------------------------------------------
# sample_volume
# ---------------------------------------------------------------------------

class TestSampleVolume:

    def test_output_shape(self, key, volume_bounds):
        coords = sample_volume(key, 100, volume_bounds)
        assert coords.shape == (100, 3)

    def test_coords_in_bounds(self, key, volume_bounds):
        coords = sample_volume(key, 500, volume_bounds)
        for i in range(3):
            assert jnp.all(coords[:, i] >= volume_bounds[i, 0])
            assert jnp.all(coords[:, i] <= volume_bounds[i, 1])

    def test_reproducibility(self, volume_bounds):
        key = jax.random.PRNGKey(0)
        c1 = sample_volume(key, 50, volume_bounds)
        c2 = sample_volume(key, 50, volume_bounds)
        assert jnp.allclose(c1, c2)

    def test_different_keys_differ(self, volume_bounds):
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        c1 = sample_volume(k1, 50, volume_bounds)
        c2 = sample_volume(k2, 50, volume_bounds)
        assert not jnp.allclose(c1, c2)

    def test_equal_bounds_returns_constant(self, key):
        bounds = jnp.array([[1., 1.], [-1., 1.], [0., 1.]])
        coords = sample_volume(key, 10, bounds)
        assert jnp.allclose(coords[:, 0], jnp.ones(10))


# ---------------------------------------------------------------------------
# lhs_sample
# ---------------------------------------------------------------------------

class TestLhsSample:

    def test_output_shape(self, key):
        pts = lhs_sample(key, 50, 2)
        assert pts.shape == (50, 2)

    def test_unit_hypercube_no_bounds(self, key):
        pts = lhs_sample(key, 100, 3)
        assert jnp.all(pts >= 0.)
        assert jnp.all(pts <= 1.)

    def test_bounds_applied(self, key):
        bounds = np.array([[-10., 10.], [0., 5.]])
        pts = lhs_sample(key, 50, 2, bounds=bounds)
        assert jnp.all(pts[:, 0] >= -10.)
        assert jnp.all(pts[:, 0] <= 10.)
        assert jnp.all(pts[:, 1] >= 0.)
        assert jnp.all(pts[:, 1] <= 5.)

    def test_reproducibility(self, key):
        pts1 = lhs_sample(key, 10, 2)
        pts2 = lhs_sample(key, 10, 2)
        assert jnp.allclose(pts1, pts2)

    def test_different_keys_differ(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        pts1 = lhs_sample(k1, 20, 2)
        pts2 = lhs_sample(k2, 20, 2)
        assert not jnp.allclose(pts1, pts2)

    def test_lhs_stratification(self, key):
        # each dimension should have exactly one sample per stratum
        n = 20
        pts = lhs_sample(key, n, 2)
        for d in range(2):
            col = np.array(pts[:, d])
            bins = np.floor(col * n).astype(int)
            # every stratum index 0..n-1 should appear exactly once
            assert sorted(bins) == list(range(n))

    def test_scramble_false(self, key):
        pts = lhs_sample(key, 10, 2, scramble=False)
        assert pts.shape == (10, 2)

    def test_optimization_random_cd(self, key):
        pts = lhs_sample(key, 20, 2, optimization="random-cd")
        assert pts.shape == (20, 2)

    def test_optimization_lloyd(self, key):
        pts = lhs_sample(key, 20, 2, optimization="lloyd")
        assert pts.shape == (20, 2)


# ---------------------------------------------------------------------------
# lhs_sample_regional
# ---------------------------------------------------------------------------

class TestLhsSampleRegional:

    def test_output_shapes(self, key):
        lons, lats = lhs_sample_regional(key, 50, (-100., -40.), (0., 30.))
        assert lons.shape == (50,)
        assert lats.shape == (50,)

    def test_lons_in_bounds(self, key):
        lons, _ = lhs_sample_regional(key, 100, (-100., -40.), (0., 30.))
        assert jnp.all(lons >= -100.)
        assert jnp.all(lons <= -40.)

    def test_lats_in_bounds(self, key):
        _, lats = lhs_sample_regional(key, 100, (-100., -40.), (0., 30.))
        assert jnp.all(lats >= 0.)
        assert jnp.all(lats <= 30.)

    def test_reproducibility(self, key):
        lons1, lats1 = lhs_sample_regional(key, 30, (-100., -40.), (0., 30.))
        lons2, lats2 = lhs_sample_regional(key, 30, (-100., -40.), (0., 30.))
        assert jnp.allclose(lons1, lons2)
        assert jnp.allclose(lats1, lats2)


# ---------------------------------------------------------------------------
# lhs_sample_volume
# ---------------------------------------------------------------------------

class TestLhsSampleVolume:

    def test_output_shape(self, key, volume_bounds):
        coords = lhs_sample_volume(key, 50, np.array(volume_bounds))
        assert coords.shape == (50, 3)

    def test_coords_in_bounds(self, key, volume_bounds):
        bounds_np = np.array(volume_bounds)
        coords = lhs_sample_volume(key, 100, bounds_np)
        for i in range(3):
            assert jnp.all(coords[:, i] >= bounds_np[i, 0])
            assert jnp.all(coords[:, i] <= bounds_np[i, 1])

    def test_reproducibility(self, key, volume_bounds):
        bounds_np = np.array(volume_bounds)
        c1 = lhs_sample_volume(key, 30, bounds_np)
        c2 = lhs_sample_volume(key, 30, bounds_np)
        assert jnp.allclose(c1, c2)

    def test_optimization_random_cd(self, key, volume_bounds):
        coords = lhs_sample_volume(
            key, 30, np.array(volume_bounds), optimization="random-cd"
        )
        assert coords.shape == (30, 3)