"""
Tests for data/sparsity.py — Nyquist-style network sparsity over the FOV.
"""

import pytest

from utils.geoscience.geodesic import spherical_box_area

from experiments.cyclone_jax.data.sparsity import network_sparsity

DOMAIN = {'lat': [0, 30], 'lon': [-100, -30]}   # basin-shaped FOV


class TestNetworkSparsity:

    def test_keys_and_area(self):
        out = network_sparsity(100, DOMAIN)
        assert set(out) == {'n_stations', 'area_km2', 'spacing_km',
                            'resolvable_km'}
        assert out['area_km2'] == pytest.approx(
            spherical_box_area(-100, -30, 0, 30))

    def test_spacing_is_sqrt_area_over_n(self):
        out = network_sparsity(100, DOMAIN)
        assert out['spacing_km'] == pytest.approx(
            (out['area_km2'] / 100) ** 0.5)
        assert out['resolvable_km'] == pytest.approx(2 * out['spacing_km'])

    def test_more_stations_finer_resolution(self):
        sparse = network_sparsity(10, DOMAIN)
        dense  = network_sparsity(1000, DOMAIN)
        assert dense['resolvable_km'] < sparse['resolvable_km']

    def test_zero_stations_resolve_nothing(self):
        out = network_sparsity(0, DOMAIN)
        assert out['spacing_km'] == float('inf')
        assert out['resolvable_km'] == float('inf')
