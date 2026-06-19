"""
Tests for utils/geoscience/coordinates.py — lat/lon encoders (NumPy) and the
JAX angle/spherical helpers (relocated from jax_core.helpers, r16).
"""

import numpy as np
import jax.numpy as jnp

from utils.geoscience.coordinates import (
    lat_lon_to_unit_sphere,
    lat_lon_to_domain_normalised,
    degrees_to_radians,
    radians_to_degrees,
    latlon_deg_to_rad,
    latlon_rad_to_deg,
    spherical_to_cartesian,
    cartesian_to_spherical,
)


class TestUnitSphere:

    def test_shape(self):
        out = lat_lon_to_unit_sphere([0.0, 45.0], [0.0, 90.0])
        assert out.shape == (2, 3)

    def test_points_on_unit_sphere(self):
        out = lat_lon_to_unit_sphere([0, 30, -60, 89], [0, 120, -170, 45])
        assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-5)

    def test_equator_prime_meridian(self):
        assert np.allclose(lat_lon_to_unit_sphere(0.0, 0.0), [1.0, 0.0, 0.0], atol=1e-6)

    def test_north_pole(self):
        assert np.allclose(lat_lon_to_unit_sphere(90.0, 0.0), [0.0, 0.0, 1.0], atol=1e-6)


class TestDomainNormalised:

    _FOV_LAT = (0.0, 30.0)
    _FOV_LON = (-100.0, -45.0)

    def test_shape(self):
        out = lat_lon_to_domain_normalised(
            [0.0, 30.0], [-100.0, -45.0], self._FOV_LAT, self._FOV_LON)
        assert out.shape == (2, 2)

    def test_corners_map_to_pm1(self):
        out = lat_lon_to_domain_normalised(
            [0.0, 30.0], [-100.0, -45.0], self._FOV_LAT, self._FOV_LON)
        assert np.allclose(out[0], [-1.0, -1.0], atol=1e-6)
        assert np.allclose(out[1], [ 1.0,  1.0], atol=1e-6)

    def test_centre_maps_to_zero(self):
        out = lat_lon_to_domain_normalised(
            15.0, -72.5, self._FOV_LAT, self._FOV_LON)
        assert np.allclose(out, [0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# degrees_to_radians / radians_to_degrees (JAX)
# ---------------------------------------------------------------------------

class TestAngleConversion:

    def test_degrees_to_radians_known_values(self):
        x = jnp.array([0., 90., 180., 360.])
        out = degrees_to_radians(x)
        expected = jnp.array([0., jnp.pi / 2, jnp.pi, 2 * jnp.pi])
        assert jnp.allclose(out, expected, atol=1e-6)

    def test_radians_to_degrees_known_values(self):
        x = jnp.array([0., jnp.pi / 2, jnp.pi, 2 * jnp.pi])
        out = radians_to_degrees(x)
        expected = jnp.array([0., 90., 180., 360.])
        assert jnp.allclose(out, expected, atol=1e-4)

    def test_round_trip(self):
        x = jnp.linspace(-360., 360., 50)
        assert jnp.allclose(radians_to_degrees(degrees_to_radians(x)), x, atol=1e-5)

    def test_output_shape_preserved(self):
        x = jnp.ones((3, 4))
        assert degrees_to_radians(x).shape == (3, 4)
        assert radians_to_degrees(x).shape == (3, 4)

    def test_scalar_input(self):
        assert jnp.allclose(degrees_to_radians(jnp.array(180.)), jnp.pi, atol=1e-6)


# ---------------------------------------------------------------------------
# latlon_deg_to_rad / latlon_rad_to_deg (JAX)
# ---------------------------------------------------------------------------

class TestLatlonConversion:

    def test_deg_to_rad_known_values(self):
        lat_r, lon_r = latlon_deg_to_rad(jnp.array([0., 45.]),
                                          jnp.array([90., 180.]))
        assert jnp.allclose(lat_r, jnp.array([0., jnp.pi / 4]), atol=1e-6)
        assert jnp.allclose(lon_r, jnp.array([jnp.pi / 2, jnp.pi]), atol=1e-6)

    def test_rad_to_deg_known_values(self):
        lat_d, lon_d = latlon_rad_to_deg(jnp.array([0., jnp.pi / 4]),
                                          jnp.array([jnp.pi / 2, jnp.pi]))
        assert jnp.allclose(lat_d, jnp.array([0., 45.]), atol=1e-4)
        assert jnp.allclose(lon_d, jnp.array([90., 180.]), atol=1e-4)

    def test_round_trip(self):
        lat = jnp.linspace(-90., 90., 20)
        lon = jnp.linspace(-180., 180., 20)
        lat_r, lon_r = latlon_deg_to_rad(lat, lon)
        lat_d, lon_d = latlon_rad_to_deg(lat_r, lon_r)
        assert jnp.allclose(lat_d, lat, atol=1e-4)
        assert jnp.allclose(lon_d, lon, atol=1e-4)

    def test_returns_tuple_of_two(self):
        result = latlon_deg_to_rad(jnp.array([10.]), jnp.array([20.]))
        assert len(result) == 2

    def test_output_shapes_match_input(self):
        lat = jnp.ones((5,))
        lon = jnp.ones((5,))
        lat_r, lon_r = latlon_deg_to_rad(lat, lon)
        assert lat_r.shape == lat.shape
        assert lon_r.shape == lon.shape


# ---------------------------------------------------------------------------
# spherical_to_cartesian / cartesian_to_spherical (JAX)
# ---------------------------------------------------------------------------

class TestSphericalCartesian:

    def test_equator_prime_meridian(self):
        xyz = spherical_to_cartesian(jnp.array([0.]), jnp.array([0.]))
        assert jnp.allclose(xyz, jnp.array([[1., 0., 0.]]), atol=1e-6)

    def test_north_pole(self):
        xyz = spherical_to_cartesian(
            jnp.array([jnp.pi / 2]), jnp.array([0.])
        )
        assert jnp.allclose(xyz, jnp.array([[0., 0., 1.]]), atol=1e-6)

    def test_south_pole(self):
        xyz = spherical_to_cartesian(
            jnp.array([-jnp.pi / 2]), jnp.array([0.])
        )
        assert jnp.allclose(xyz, jnp.array([[0., 0., -1.]]), atol=1e-6)

    def test_output_is_unit_vector(self):
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 10)
        lon = jnp.linspace(-jnp.pi, jnp.pi, 10)
        xyz = spherical_to_cartesian(lat, lon)
        norms = jnp.linalg.norm(xyz, axis=-1)
        assert jnp.allclose(norms, jnp.ones(10), atol=1e-6)

    def test_output_shape(self):
        lat = jnp.zeros((8,))
        lon = jnp.zeros((8,))
        assert spherical_to_cartesian(lat, lon).shape == (8, 3)

    def test_round_trip(self):
        lat = jnp.array([0.1, 0.5, -0.3, -1.2, 1.0])
        lon = jnp.array([1.2, -0.7, 2.5, 0.0, -1.5])
        xyz = spherical_to_cartesian(lat, lon)
        lat2, lon2 = cartesian_to_spherical(xyz)
        assert jnp.allclose(lat, lat2, atol=1e-6)
        assert jnp.allclose(lon, lon2, atol=1e-6)

    def test_cartesian_to_spherical_ellipsis_indexing(self):
        # shape (2, 3, 3) -- arbitrary leading dims
        xyz = jnp.ones((2, 3, 3))
        xyz = xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
        lat, lon = cartesian_to_spherical(xyz)
        assert lat.shape == (2, 3)
        assert lon.shape == (2, 3)

    def test_cartesian_to_spherical_clips_z(self):
        # z slightly outside [-1, 1] due to floating point
        xyz = jnp.array([[0., 0., 1.0000001]])
        lat, _ = cartesian_to_spherical(xyz)
        assert jnp.isfinite(lat).all()

    def test_lon_range(self):
        lat = jnp.zeros((100,))
        lon = jnp.linspace(-jnp.pi, jnp.pi, 100)
        xyz = spherical_to_cartesian(lat, lon)
        _, lon2 = cartesian_to_spherical(xyz)
        assert jnp.all(lon2 >= -jnp.pi - 1e-5)
        assert jnp.all(lon2 <= jnp.pi + 1e-5)
