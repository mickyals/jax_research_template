"""
Tests for experiments/sparse_obs_encoder/data/transforms/encoding.py.

Round-trip exactness for both encode/decode pairs, the north-seam property
that motivated the local x-y encoding (decision 17), convention checks
(north-up = +y, east = +x), and the sample-level COORD_ENCODERS registry.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.sparse_obs_encoder.data.transforms.encoding import (
    encode_unit_circle, decode_unit_circle,
    encode_domain, decode_domain,
    COORD_ENCODERS, COORD_DECODERS, get_coord_encoder, get_coord_decoder,
)

FOV_LAT = (0.0, 30.0)
FOV_LON = (-100.0, -45.0)


# ---------------------------------------------------------------------------
# unit_circle
# ---------------------------------------------------------------------------

class TestUnitCircle:

    @pytest.mark.parametrize("dist,bearing_deg", [
        (0.1, 0.0), (0.5, 45.0), (1.0, 90.0), (0.3, 180.0),
        (0.7, 270.0), (0.9, 359.0), (0.25, 123.4),
    ])
    def test_roundtrip(self, dist, bearing_deg):
        x, y = encode_unit_circle(dist, np.radians(bearing_deg))
        d, b = decode_unit_circle(x, y)
        assert float(d) == pytest.approx(dist, abs=1e-5)
        assert float(np.degrees(b)) == pytest.approx(bearing_deg, abs=1e-3)

    def test_north_seam_continuity(self):
        # Two stations at equal distance, bearings 359° and 1° — physically
        # ~2° apart across north — must encode near each other (the raw
        # bearing channel put them 358° apart).
        x1, y1 = encode_unit_circle(0.5, np.radians(359.0))
        x2, y2 = encode_unit_circle(0.5, np.radians(1.0))
        gap = np.hypot(x1 - x2, y1 - y2)
        # chord length for 2° at radius 0.5 ≈ 0.0175
        assert gap < 0.02

    def test_conventions(self):
        # due north → (0, +d); due east → (+d, 0)
        x, y = encode_unit_circle(0.8, 0.0)
        assert float(x) == pytest.approx(0.0, abs=1e-6)
        assert float(y) == pytest.approx(0.8, abs=1e-6)
        x, y = encode_unit_circle(0.8, np.pi / 2)
        assert float(x) == pytest.approx(0.8, abs=1e-6)
        assert float(y) == pytest.approx(0.0, abs=1e-6)

    def test_origin_is_storm_position(self):
        d, _ = decode_unit_circle(0.0, 0.0)
        assert float(d) == 0.0

    def test_vectorised(self):
        dist    = np.array([0.2, 0.5, 1.0], dtype=np.float32)
        bearing = np.radians([10.0, 200.0, 350.0]).astype(np.float32)
        x, y = encode_unit_circle(dist, bearing)
        assert x.shape == (3,)
        d, b = decode_unit_circle(x, y)
        assert np.allclose(d, dist, atol=1e-5)
        assert np.allclose(b, bearing, atol=1e-4)

    def test_output_in_unit_square(self):
        rng = np.random.default_rng(0)
        x, y = encode_unit_circle(
            rng.uniform(0, 1, 100), rng.uniform(0, 2 * np.pi, 100)
        )
        assert np.all(np.abs(x) <= 1.0) and np.all(np.abs(y) <= 1.0)


# ---------------------------------------------------------------------------
# domain
# ---------------------------------------------------------------------------

class TestDomain:

    @pytest.mark.parametrize("lat,lon", [
        (0.0, -100.0), (30.0, -45.0), (15.0, -75.0), (7.3, -62.1),
    ])
    def test_roundtrip(self, lat, lon):
        nlat, nlon = encode_domain(lat, lon, FOV_LAT, FOV_LON)
        rlat, rlon = decode_domain(nlat, nlon, FOV_LAT, FOV_LON)
        assert float(rlat) == pytest.approx(lat, abs=1e-4)
        assert float(rlon) == pytest.approx(lon, abs=1e-4)

    def test_bounds_map_to_half_pi(self):
        nlat, nlon = encode_domain(30.0, -45.0, FOV_LAT, FOV_LON)
        assert float(nlat) == pytest.approx(np.pi / 2, abs=1e-5)
        assert float(nlon) == pytest.approx(np.pi / 2, abs=1e-5)
        nlat, nlon = encode_domain(0.0, -100.0, FOV_LAT, FOV_LON)
        assert float(nlat) == pytest.approx(-np.pi / 2, abs=1e-5)
        assert float(nlon) == pytest.approx(-np.pi / 2, abs=1e-5)

    def test_vectorised(self):
        rng = np.random.default_rng(1)
        lats = rng.uniform(*FOV_LAT, 50)
        lons = rng.uniform(*FOV_LON, 50)
        nlat, nlon = encode_domain(lats, lons, FOV_LAT, FOV_LON)
        rlat, rlon = decode_domain(nlat, nlon, FOV_LAT, FOV_LON)
        assert np.allclose(rlat, lats, atol=1e-3)
        assert np.allclose(rlon, lons, atol=1e-3)


# ---------------------------------------------------------------------------
# Sample-level coordinate encoders (COORD_ENCODERS registry)
# ---------------------------------------------------------------------------

def _station_frame():
    """A small per-sample station frame as produced by get_obs_near."""
    return pd.DataFrame({
        'latitude':    np.array([15.0, 16.0, 14.0], dtype=np.float32),
        'longitude':   np.array([-75.0, -74.5, -75.5], dtype=np.float32),
        'distance_km': np.array([0.0, 120.0, 90.0], dtype=np.float32),
    })


class TestCoordEncoders:

    def test_registry_membership(self):
        assert set(COORD_ENCODERS.names()) == {'UNIT_CIRCLE', 'DOMAIN'}
        assert set(COORD_DECODERS.names()) == {'UNIT_CIRCLE', 'DOMAIN'}

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_coord_encoder('polar')

    def test_unit_circle_shapes_and_origin(self):
        df  = _station_frame()
        enc = get_coord_encoder('unit_circle')
        station, query = enc(df, 15.0, -75.0,
                             radius_km=300.0, fov_lat=FOV_LAT, fov_lon=FOV_LON)
        assert station.shape == (3, 2)
        assert station.dtype == np.float32
        assert query.shape == (2,)
        # query sits at the storm origin
        assert np.allclose(query, 0.0)
        # hypot recovers the clipped normalised distance
        expected = np.clip(df['distance_km'].to_numpy() / 300.0, 0.0, 1.0)
        assert np.allclose(np.hypot(station[:, 0], station[:, 1]), expected, atol=1e-3)

    def test_domain_matches_encode_domain(self):
        df  = _station_frame()
        enc = get_coord_encoder('domain')
        station, query = enc(df, 15.0, -75.0,
                             radius_km=300.0, fov_lat=FOV_LAT, fov_lon=FOV_LON)
        assert station.shape == (3, 2)
        nlat, nlon = encode_domain(
            df['latitude'].to_numpy(), df['longitude'].to_numpy(), FOV_LAT, FOV_LON,
        )
        assert np.allclose(station[:, 0], nlat, atol=1e-5)
        assert np.allclose(station[:, 1], nlon, atol=1e-5)
        q = encode_domain(15.0, -75.0, FOV_LAT, FOV_LON)
        assert np.allclose(query, q, atol=1e-5)

    def test_decoder_lookup(self):
        assert get_coord_decoder('domain') is decode_domain
        assert get_coord_decoder('unit_circle') is decode_unit_circle
