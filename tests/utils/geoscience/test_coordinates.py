"""
Tests for utils/geoscience/coordinates.py — pure lat/lon encoders.
"""

import numpy as np

from utils.geoscience.coordinates import (
    lat_lon_to_unit_sphere,
    lat_lon_to_domain_normalised,
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
