import numpy as np
import jax
import jax.numpy as jnp
import pytest

from utils.geoscience.geodesic import (
    haversine_np,
    haversine_jax,
    vincenty_np,
    vincenty_jax,
)

# ---------------------------------------------------------------------------
# Reference pairs (Toronto -> New York, easy to verify externally)
# ---------------------------------------------------------------------------
_LAT1, _LON1 =  43.6532, -79.3832   # Toronto
_LAT2, _LON2 =  40.7128, -74.0060   # New York

# Computed from the functions themselves and cross-checked manually.
# Toronto (43.6532, -79.3832) -> New York (40.7128, -74.0060)
_REF_KM_VINCENTY  = 551.18   # WGS-84 ellipsoid
_REF_KM_HAVERSINE = 550.44   # spherical mean-radius approx
_DIST_TOL = 1.0              # km — coarse; tests correctness, not 7-sig accuracy


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

class TestHaversineNp:
    def test_scalar(self):
        d = haversine_np(_LAT1, _LON1, _LAT2, _LON2)
        assert abs(float(d) - _REF_KM_HAVERSINE) < _DIST_TOL

    def test_vectorized(self):
        lats1 = np.array([_LAT1, 0.0])
        lons1 = np.array([_LON1, 0.0])
        lats2 = np.array([_LAT2, 0.0])
        lons2 = np.array([_LON2, 0.0])
        d = haversine_np(lats1, lons1, lats2, lons2)
        assert d.shape == (2,)
        assert abs(d[0] - _REF_KM_HAVERSINE) < _DIST_TOL
        assert d[1] == pytest.approx(0.0, abs=1e-6)

    def test_coincident_points(self):
        d = haversine_np(_LAT1, _LON1, _LAT1, _LON1)
        assert float(d) == pytest.approx(0.0, abs=1e-6)

    def test_antipodal_clamp(self):
        # arcsin arg should be clipped to [0, 1]; no NaN/inf
        d = haversine_np(89.9, 0.0, -89.9, 180.0)
        assert np.isfinite(d)


class TestHaversineJax:
    def test_scalar(self):
        d = haversine_jax(
            jnp.array(_LAT1), jnp.array(_LON1),
            jnp.array(_LAT2), jnp.array(_LON2),
        )
        assert abs(float(d) - _REF_KM_HAVERSINE) < _DIST_TOL

    def test_agrees_with_numpy(self):
        d_np  = haversine_np(_LAT1, _LON1, _LAT2, _LON2)
        d_jax = haversine_jax(
            jnp.array(_LAT1), jnp.array(_LON1),
            jnp.array(_LAT2), jnp.array(_LON2),
        )
        assert abs(float(d_np) - float(d_jax)) < 0.01  # float32 vs float64

    def test_differentiable(self):
        def f(lat2):
            return haversine_jax(
                jnp.array(_LAT1), jnp.array(_LON1),
                lat2, jnp.array(_LON2),
            )
        grad = jax.grad(f)(jnp.array(_LAT2))
        assert jnp.isfinite(grad)

    def test_batched_array(self):
        lats1 = jnp.array([_LAT1, 0.0, _LAT1])
        lons1 = jnp.array([_LON1, 0.0, _LON1])
        lats2 = jnp.array([_LAT2, 0.0, _LAT1])
        lons2 = jnp.array([_LON2, 0.0, _LON1])
        d = haversine_jax(lats1, lons1, lats2, lons2)
        assert d.shape == (3,)
        assert abs(float(d[0]) - _REF_KM_HAVERSINE) < _DIST_TOL
        assert float(d[1]) == pytest.approx(0.0, abs=1e-3)  # coincident equator
        assert float(d[2]) == pytest.approx(0.0, abs=1e-3)  # coincident toronto

    def test_batched_vmap(self):
        lats2 = jnp.linspace(30.0, 50.0, 8)
        lons2 = jnp.full(8, _LON2)
        d = jax.vmap(lambda la, lo: haversine_jax(
            jnp.array(_LAT1), jnp.array(_LON1), la, lo
        ))(lats2, lons2)
        assert d.shape == (8,)
        assert jnp.all(jnp.isfinite(d))


# ---------------------------------------------------------------------------
# Vincenty
# ---------------------------------------------------------------------------

class TestVincentyNp:
    def test_scalar_distance(self):
        d, _, _, _ = vincenty_np(_LAT1, _LON1, _LAT2, _LON2)
        assert abs(float(d) - _REF_KM_VINCENTY) < _DIST_TOL

    def test_azimuths_in_range(self):
        _, fwd, back, _ = vincenty_np(_LAT1, _LON1, _LAT2, _LON2)
        assert 0.0 <= float(fwd) < 360.0
        assert 0.0 <= float(back) < 360.0

    def test_coincident_points(self):
        d, fwd, back, _ = vincenty_np(_LAT1, _LON1, _LAT1, _LON1)
        assert float(d)   == pytest.approx(0.0, abs=1e-4)
        assert float(fwd) == pytest.approx(0.0, abs=1e-4)
        assert float(back) == pytest.approx(0.0, abs=1e-4)

    def test_embed_bearing_shape(self):
        _, _, _, emb = vincenty_np(_LAT1, _LON1, _LAT2, _LON2, embed_bearing=True)
        assert emb is not None
        bsin, bcos = emb
        assert np.isfinite(bsin) and np.isfinite(bcos)
        assert abs(bsin**2 + bcos**2 - 1.0) < 1e-6

    def test_no_embed_bearing(self):
        _, _, _, emb = vincenty_np(_LAT1, _LON1, _LAT2, _LON2, embed_bearing=False)
        assert emb is None

    def test_vectorized(self):
        lats1 = np.array([_LAT1, 0.0])
        lons1 = np.array([_LON1, 0.0])
        lats2 = np.array([_LAT2, 0.0])
        lons2 = np.array([_LON2, 0.0])
        d, _, _, _ = vincenty_np(lats1, lons1, lats2, lons2)
        assert d.shape == (2,)
        assert abs(d[0] - _REF_KM_VINCENTY) < _DIST_TOL
        assert d[1] == pytest.approx(0.0, abs=1e-4)


class TestVincentyJax:
    def test_scalar_distance(self):
        d, _, _, _ = vincenty_jax(
            jnp.array(_LAT1), jnp.array(_LON1),
            jnp.array(_LAT2), jnp.array(_LON2),
        )
        assert abs(float(d) - _REF_KM_VINCENTY) < _DIST_TOL

    def test_agrees_with_numpy(self):
        d_np, _, _, _ = vincenty_np(_LAT1, _LON1, _LAT2, _LON2)
        d_jax, _, _, _ = vincenty_jax(
            jnp.array(_LAT1), jnp.array(_LON1),
            jnp.array(_LAT2), jnp.array(_LON2),
        )
        assert abs(float(d_np) - float(d_jax)) < 1e-3

    def test_differentiable(self):
        def f(lat2):
            d, _, _, _ = vincenty_jax(
                jnp.array(_LAT1), jnp.array(_LON1),
                lat2, jnp.array(_LON2),
            )
            return d
        grad = jax.grad(f)(jnp.array(_LAT2))
        assert jnp.isfinite(grad)

    def test_embed_bearing(self):
        _, _, _, emb = vincenty_jax(
            jnp.array(_LAT1), jnp.array(_LON1),
            jnp.array(_LAT2), jnp.array(_LON2),
            embed_bearing=True,
        )
        assert emb is not None
        bsin, bcos = emb
        assert jnp.isfinite(bsin) and jnp.isfinite(bcos)
        assert abs(float(bsin**2 + bcos**2) - 1.0) < 1e-5

    def test_no_embed_bearing(self):
        _, _, _, emb = vincenty_jax(
            jnp.array(_LAT1), jnp.array(_LON1),
            jnp.array(_LAT2), jnp.array(_LON2),
            embed_bearing=False,
        )
        assert emb is None

    def test_azimuths_in_range(self):
        _, fwd, back, _ = vincenty_jax(
            jnp.array(_LAT1), jnp.array(_LON1),
            jnp.array(_LAT2), jnp.array(_LON2),
        )
        assert 0.0 <= float(fwd) < 360.0
        assert 0.0 <= float(back) < 360.0

    def test_coincident_points(self):
        d, fwd, back, _ = vincenty_jax(
            jnp.array(_LAT1), jnp.array(_LON1),
            jnp.array(_LAT1), jnp.array(_LON1),
        )
        assert float(d)    == pytest.approx(0.0, abs=1e-4)
        assert float(fwd)  == pytest.approx(0.0, abs=1e-4)
        assert float(back) == pytest.approx(0.0, abs=1e-4)

    def test_batched_array(self):
        lats1 = jnp.array([_LAT1, 0.0, _LAT1])
        lons1 = jnp.array([_LON1, 0.0, _LON1])
        lats2 = jnp.array([_LAT2, 0.0, _LAT1])
        lons2 = jnp.array([_LON2, 0.0, _LON1])
        d, fwd, back, _ = vincenty_jax(lats1, lons1, lats2, lons2)
        assert d.shape == (3,)
        assert fwd.shape == (3,)
        assert back.shape == (3,)
        assert abs(float(d[0]) - _REF_KM_VINCENTY) < _DIST_TOL
        assert float(d[1]) == pytest.approx(0.0, abs=1e-4)
        assert float(d[2]) == pytest.approx(0.0, abs=1e-4)
        assert 0.0 <= float(fwd[0]) < 360.0
        assert jnp.all(jnp.isfinite(d))

    def test_batched_vmap(self):
        lats2 = jnp.linspace(30.0, 50.0, 8)
        lons2 = jnp.full(8, _LON2)
        d, _, _, _ = jax.vmap(lambda la, lo: vincenty_jax(
            jnp.array(_LAT1), jnp.array(_LON1), la, lo
        ))(lats2, lons2)
        assert d.shape == (8,)
        assert jnp.all(jnp.isfinite(d))
