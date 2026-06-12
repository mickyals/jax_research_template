import math
import numpy as np
import jax.numpy as jnp
import jax
import pytest

from utils.geoscience.met_conversions import (
    # wind
    kt_to_ms, ms_to_kt,
    kmh_to_ms, ms_to_kmh,
    mph_to_ms, ms_to_mph,
    # pressure
    hpa_to_pa, pa_to_hpa,
    inhg_to_pa, pa_to_inhg,
    # distance
    nmile_to_m, m_to_nmile,
    km_to_m, m_to_km,
    ft_to_m, m_to_ft,
    mi_to_m, m_to_mi,
    # angle
    deg_to_rad, rad_to_deg,
    # temperature
    celsius_to_kelvin, kelvin_to_celsius,
    fahrenheit_to_celsius, celsius_to_fahrenheit,
    fahrenheit_to_kelvin, kelvin_to_fahrenheit,
    # bearing
    bearing_to_components, components_to_bearing,
    # wind
    wind_to_components, components_to_wind,
    # thresholds
    R34_MS_THRESHOLD, R50_MS_THRESHOLD, R64_MS_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_roundtrip(fn_fwd, fn_inv, value, tol=1e-6):
    assert abs(float(fn_inv(fn_fwd(value))) - value) < tol


# ---------------------------------------------------------------------------
# Wind speed
# ---------------------------------------------------------------------------

class TestWindSpeed:
    def test_kt_to_ms_known(self):
        assert abs(float(kt_to_ms(1.0)) - 0.514444) < 1e-6

    def test_roundtrip_kt_ms(self):
        _check_roundtrip(kt_to_ms, ms_to_kt, 35.0)

    def test_roundtrip_kmh_ms(self):
        _check_roundtrip(kmh_to_ms, ms_to_kmh, 100.0)

    def test_roundtrip_mph_ms(self):
        _check_roundtrip(mph_to_ms, ms_to_mph, 60.0)

    def test_numpy_array(self):
        v = np.array([10.0, 20.0, 30.0])
        out = kt_to_ms(v)
        assert out.shape == (3,)
        assert np.allclose(out, v * 0.514444)

    def test_jax_array(self):
        v = jnp.array([10.0, 20.0])
        out = kt_to_ms(v)
        assert out.shape == (2,)
        assert jnp.allclose(out, v * 0.514444)

    def test_scalar_float(self):
        assert isinstance(kt_to_ms(10.0), float)

    def test_thresholds(self):
        assert abs(R34_MS_THRESHOLD - 34 * 0.514444) < 1e-6
        assert abs(R50_MS_THRESHOLD - 50 * 0.514444) < 1e-6
        assert abs(R64_MS_THRESHOLD - 64 * 0.514444) < 1e-6


# ---------------------------------------------------------------------------
# Pressure
# ---------------------------------------------------------------------------

class TestPressure:
    def test_hpa_to_pa_known(self):
        assert float(hpa_to_pa(1013.25)) == pytest.approx(101325.0, rel=1e-5)

    def test_roundtrip_hpa_pa(self):
        _check_roundtrip(hpa_to_pa, pa_to_hpa, 1013.25)

    def test_roundtrip_inhg_pa(self):
        _check_roundtrip(inhg_to_pa, pa_to_inhg, 29.92)

    def test_numpy_array(self):
        p = np.array([900.0, 950.0, 1000.0])
        assert hpa_to_pa(p).shape == (3,)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

class TestDistance:
    def test_nmile_to_m_known(self):
        assert float(nmile_to_m(1.0)) == pytest.approx(1852.0)

    def test_roundtrip_nmile_m(self):
        _check_roundtrip(nmile_to_m, m_to_nmile, 100.0)

    def test_roundtrip_km_m(self):
        _check_roundtrip(km_to_m, m_to_km, 500.0)

    def test_roundtrip_ft_m(self):
        _check_roundtrip(ft_to_m, m_to_ft, 1000.0)

    def test_roundtrip_mi_m(self):
        _check_roundtrip(mi_to_m, m_to_mi, 26.2)

    def test_numpy_array(self):
        d = np.array([1.0, 2.0, 3.0])
        assert km_to_m(d).shape == (3,)


# ---------------------------------------------------------------------------
# Angle
# ---------------------------------------------------------------------------

class TestAngle:
    def test_deg_to_rad_known(self):
        assert abs(float(deg_to_rad(180.0)) - math.pi) < 1e-10

    def test_roundtrip_deg_rad(self):
        _check_roundtrip(deg_to_rad, rad_to_deg, 45.0, tol=1e-10)

    def test_numpy_array(self):
        a = np.array([0.0, 90.0, 180.0, 270.0])
        out = deg_to_rad(a)
        assert out.shape == (4,)
        assert np.allclose(out, np.array([0, math.pi/2, math.pi, 3*math.pi/2]))


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

class TestTemperature:
    def test_celsius_to_kelvin_known(self):
        assert float(celsius_to_kelvin(0.0)) == pytest.approx(273.15)
        assert float(celsius_to_kelvin(100.0)) == pytest.approx(373.15)

    def test_roundtrip_celsius_kelvin(self):
        _check_roundtrip(celsius_to_kelvin, kelvin_to_celsius, 25.0)

    def test_fahrenheit_to_celsius_known(self):
        assert float(fahrenheit_to_celsius(32.0)) == pytest.approx(0.0, abs=1e-10)
        assert float(fahrenheit_to_celsius(212.0)) == pytest.approx(100.0, abs=1e-6)

    def test_roundtrip_fahrenheit_celsius(self):
        _check_roundtrip(fahrenheit_to_celsius, celsius_to_fahrenheit, 98.6)

    def test_fahrenheit_to_kelvin_known(self):
        assert float(fahrenheit_to_kelvin(32.0)) == pytest.approx(273.15, abs=1e-6)

    def test_roundtrip_fahrenheit_kelvin(self):
        _check_roundtrip(fahrenheit_to_kelvin, kelvin_to_fahrenheit, 72.0)

    def test_numpy_array(self):
        t = np.array([0.0, 20.0, 100.0])
        out = celsius_to_kelvin(t)
        assert out.shape == (3,)
        assert np.allclose(out, t + 273.15)


# ---------------------------------------------------------------------------
# Bearing
# ---------------------------------------------------------------------------

class TestBearing:
    @pytest.mark.parametrize("bearing", [0.0, 45.0, 90.0, 180.0, 270.0, 359.0])
    def test_roundtrip(self, bearing):
        s, c = bearing_to_components(bearing)
        recovered = components_to_bearing(s, c)
        assert abs(float(recovered) - bearing) < 1e-4

    def test_unit_circle(self):
        for b in [0.0, 30.0, 90.0, 135.0, 270.0]:
            s, c = bearing_to_components(b)
            assert abs(float(s)**2 + float(c)**2 - 1.0) < 1e-6

    def test_north_is_zero(self):
        s, c = bearing_to_components(0.0)
        assert abs(float(s)) < 1e-10
        assert abs(float(c) - 1.0) < 1e-10

    def test_east_is_90(self):
        s, c = bearing_to_components(90.0)
        assert abs(float(s) - 1.0) < 1e-6
        assert abs(float(c)) < 1e-6

    def test_numpy_array(self):
        bearings = np.array([0.0, 90.0, 180.0, 270.0])
        s, c = bearing_to_components(bearings)
        assert s.shape == (4,)
        assert c.shape == (4,)
        recovered = components_to_bearing(s, c)
        assert np.allclose(recovered, bearings, atol=1e-4)

    def test_jax_array(self):
        bearings = jnp.array([0.0, 90.0, 180.0, 270.0])
        s, c = bearing_to_components(bearings)
        assert isinstance(s, jax.Array)
        recovered = components_to_bearing(s, c)
        assert jnp.allclose(recovered, bearings, atol=1e-4)


# ---------------------------------------------------------------------------
# Wind decomposition (meteorological FROM convention)
# ---------------------------------------------------------------------------

class TestWindComponents:
    @pytest.mark.parametrize("speed,direction", [
        (10.0, 0.0), (5.0, 45.0), (20.0, 90.0), (3.3, 180.0),
        (15.0, 270.0), (7.0, 359.0), (1.0, 123.4),
    ])
    def test_roundtrip(self, speed, direction):
        u, v = wind_to_components(speed, direction)
        s, d = components_to_wind(u, v)
        assert float(s) == pytest.approx(speed, abs=1e-6)
        assert float(d) == pytest.approx(direction, abs=1e-4)

    def test_from_convention(self):
        # Wind FROM north blows toward the south: u=0, v=-speed
        u, v = wind_to_components(10.0, 0.0)
        assert float(u) == pytest.approx(0.0, abs=1e-10)
        assert float(v) == pytest.approx(-10.0)
        # Wind FROM east blows toward the west: u=-speed, v=0
        u, v = wind_to_components(10.0, 90.0)
        assert float(u) == pytest.approx(-10.0)
        assert float(v) == pytest.approx(0.0, abs=1e-6)

    def test_calm_with_nan_direction_is_zero_vector(self):
        u, v = wind_to_components(0.0, np.nan)
        assert float(u) == 0.0
        assert float(v) == 0.0

    def test_calm_inverse_direction_is_nan(self):
        s, d = components_to_wind(0.0, 0.0)
        assert float(s) == 0.0
        assert np.isnan(float(d))

    def test_negative_speed_raises(self):
        with pytest.raises(ValueError, match='non-negative'):
            wind_to_components(-5.0, 90.0)
        with pytest.raises(ValueError, match='non-negative'):
            wind_to_components(np.array([3.0, -0.1]), np.array([0.0, 90.0]))

    def test_nan_speed_passes_negative_guard(self):
        # NaN means missing, not invalid — must not trip the guard
        u, v = wind_to_components(np.array([np.nan, 1.0]),
                                  np.array([0.0, 180.0]))
        assert np.isnan(u[0]) and np.isnan(v[0])
        assert v[1] == pytest.approx(1.0)

    def test_nan_propagation(self):
        # Non-calm speed with missing direction → unknown vector
        u, v = wind_to_components(5.0, np.nan)
        assert np.isnan(float(u)) and np.isnan(float(v))
        # Missing speed → unknown vector
        u, v = wind_to_components(np.nan, 90.0)
        assert np.isnan(float(u)) and np.isnan(float(v))

    def test_numpy_array_mixed(self):
        speed     = np.array([0.0, 10.0, 5.0, np.nan])
        direction = np.array([np.nan, 90.0, np.nan, 45.0])
        u, v = wind_to_components(speed, direction)
        assert u.shape == (4,)
        assert u[0] == 0.0 and v[0] == 0.0          # calm
        assert u[1] == pytest.approx(-10.0)         # from east
        assert np.isnan(u[2]) and np.isnan(v[2])    # speed w/o direction
        assert np.isnan(u[3]) and np.isnan(v[3])    # direction w/o speed

    def test_jax_array(self):
        speed     = jnp.array([0.0, 10.0])
        direction = jnp.array([jnp.nan, 180.0])
        u, v = wind_to_components(speed, direction)
        assert isinstance(u, jax.Array)
        assert float(u[0]) == 0.0 and float(v[0]) == 0.0
        s, d = components_to_wind(u, v)
        assert jnp.isnan(d[0])
        assert float(s[1]) == pytest.approx(10.0)
        assert float(d[1]) == pytest.approx(180.0, abs=1e-4)
