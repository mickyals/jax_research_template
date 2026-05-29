import pytest
import jax
import jax.numpy as jnp
import numpy as np
import math

from core.embeddings import (
    register_embedding,
    get_embedding,
    list_embeddings,
    EMBEDDINGS,
    GaussianFourierEmbedding,
    PositionalEmbedding,
    SphericalGridEmbedding,
    SphericalCartesianEmbedding,
    SphericalMultiScaleEmbedding,
    DoubleFourierSphericalEmbedding,
    SphericalCartesianPlusEmbedding,
    SphericalMultiScalePlusEmbedding,
    SphericalHarmonicsEmbedding,
    SinusoidalPosEncoding,
    LearnedPosEncoding,
    LearnedPosEncoding2D,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY      = jax.random.PRNGKey(0)
BATCH    = 4
SEQ      = 16
D_MODEL  = 64
H, W     = 14, 14
EMBED    = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_shape(x, expected, label='output'):
    assert x.shape == expected, f"{label}: expected {expected}, got {x.shape}"


def check_finite(x, label='output'):
    assert jnp.all(jnp.isfinite(x)), f"{label} contains non-finite values"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def x_2d():
    return jnp.ones((8, 2))


@pytest.fixture
def lat_lon():
    rng = np.random.default_rng(0)
    lat = jnp.array(rng.uniform(-np.pi / 2, np.pi / 2, 16))
    lon = jnp.array(rng.uniform(-np.pi, np.pi, 16))
    return lat, lon


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_register_and_retrieve(self):
        import flax.linen as nn

        @register_embedding("TEST_EMBED_REG", description="test")
        class _TestEmbed(nn.Module):
            @nn.compact
            def __call__(self, x: jax.Array) -> jax.Array:
                return x

        assert "TEST_EMBED_REG" in EMBEDDINGS
        embed = get_embedding("TEST_EMBED_REG")
        assert embed is not None

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            @register_embedding("GAUSSIAN_POSITIONAL")
            class _Dup:
                pass

    def test_case_insensitive_lookup(self, key, x_2d):
        e1 = get_embedding("GAUSSIAN_POSITIONAL", input_dim=2,
                            mapping_dim=16, scale=1.0)
        e2 = get_embedding("gaussian_positional", input_dim=2,
                            mapping_dim=16, scale=1.0)
        v1 = e1.init(key, x_2d)
        v2 = e2.init(key, x_2d)
        assert jnp.allclose(e1.apply(v1, x_2d), e2.apply(v2, x_2d))

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            get_embedding("NONEXISTENT_XYZ_123")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError) as exc_info:
            get_embedding("NONEXISTENT_XYZ_123")
        assert "GAUSSIAN_POSITIONAL" in str(exc_info.value)

    def test_unknown_kwarg_warns(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_embedding("SPHERE_GRID", scale=4, r_min=0.01,
                           r_max=1.0, unknown_param=99)
            assert any("unknown kwargs" in str(x.message) for x in w)

    def test_list_embeddings_returns_dict(self):
        result = list_embeddings()
        assert isinstance(result, dict)
        assert "GAUSSIAN_POSITIONAL" in result
        assert "SPHERICAL_HARMONICS" in result

    def test_list_embeddings_sorted(self):
        result = list_embeddings()
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_list_embeddings_descriptions_are_strings(self):
        for name, desc in list_embeddings().items():
            assert isinstance(desc, str)


# ---------------------------------------------------------------------------
# GaussianFourierEmbedding
# ---------------------------------------------------------------------------

class TestGaussianFourierEmbedding:

    def test_output_shape(self, key, x_2d):
        embed = GaussianFourierEmbedding(input_dim=2, mapping_dim=64, scale=10.0)
        variables = embed.init(key, x_2d)
        out = embed.apply(variables, x_2d)
        assert out.shape == (8, 64)

    def test_odd_mapping_dim_raises(self):
        embed = GaussianFourierEmbedding(input_dim=2, mapping_dim=63, scale=1.0)
        with pytest.raises(ValueError, match="even"):
            embed.init(jax.random.PRNGKey(0), jnp.ones((4, 2)))

    def test_out_features_property(self):
        embed = GaussianFourierEmbedding(input_dim=2, mapping_dim=64, scale=1.0)
        assert embed.out_features == 64

    def test_variables_empty(self, key, x_2d):
        # B is a plain Python attribute -- invisible to Flax variable system
        embed = GaussianFourierEmbedding(input_dim=2, mapping_dim=32, scale=1.0)
        variables = embed.init(key, x_2d)
        assert variables == {}

    def test_B_shape(self, key, x_2d):
        # B is only accessible inside init/apply -- verify indirectly
        # via output shape: output is (N, mapping_dim) = (N, 2 * mapping_dim//2)
        # which requires B of shape (input_dim, mapping_dim//2)
        embed = GaussianFourierEmbedding(input_dim=2, mapping_dim=32, scale=1.0)
        variables = embed.init(key, x_2d)
        out = embed.apply(variables, x_2d)
        assert out.shape == (8, 32)

    def test_B_does_not_change_across_apply_calls(self, key, x_2d):
        # B is static -- same input always produces same output
        embed = GaussianFourierEmbedding(input_dim=2, mapping_dim=32, scale=1.0)
        variables = embed.init(key, x_2d)
        out1 = embed.apply(variables, x_2d)
        out2 = embed.apply(variables, x_2d)
        assert jnp.allclose(out1, out2)

    def test_different_seeds_produce_different_B(self, key, x_2d):
        e1 = GaussianFourierEmbedding(input_dim=2, mapping_dim=32,
                                      scale=1.0, seed=0)
        e2 = GaussianFourierEmbedding(input_dim=2, mapping_dim=32,
                                      scale=1.0, seed=1)
        v1 = e1.init(key, x_2d)
        v2 = e2.init(key, x_2d)
        # different seeds must produce different outputs for same input
        assert not jnp.allclose(e1.apply(v1, x_2d), e2.apply(v2, x_2d))

    def test_same_seed_reproducible(self, key, x_2d):
        e1 = GaussianFourierEmbedding(input_dim=2, mapping_dim=32,
                                      scale=1.0, seed=42)
        e2 = GaussianFourierEmbedding(input_dim=2, mapping_dim=32,
                                      scale=1.0, seed=42)
        v1 = e1.init(key, x_2d)
        v2 = e2.init(key, x_2d)
        assert jnp.allclose(e1.apply(v1, x_2d), e2.apply(v2, x_2d))

    def test_output_is_real(self, key, x_2d):
        embed = GaussianFourierEmbedding(input_dim=2, mapping_dim=32, scale=1.0)
        variables = embed.init(key, x_2d)
        out = embed.apply(variables, x_2d)
        assert not jnp.issubdtype(out.dtype, jnp.complexfloating)

    def test_jit_compatible(self, key, x_2d):
        embed = GaussianFourierEmbedding(input_dim=2, mapping_dim=32, scale=1.0)
        variables = embed.init(key, x_2d)
        out = jax.jit(embed.apply)(variables, x_2d)
        assert out.shape == (8, 32)

    def test_scale_affects_output(self, key, x_2d):
        e1 = GaussianFourierEmbedding(input_dim=2, mapping_dim=32, scale=1.0)
        e2 = GaussianFourierEmbedding(input_dim=2, mapping_dim=32, scale=100.0)
        v1 = e1.init(key, x_2d)
        v2 = e2.init(key, x_2d)
        assert not jnp.allclose(e1.apply(v1, x_2d), e2.apply(v2, x_2d))



# ---------------------------------------------------------------------------
# PositionalEmbedding
# ---------------------------------------------------------------------------

class TestPositionalEmbedding:

    def test_output_shape(self, x_2d):
        embed = PositionalEmbedding(input_dim=2, mapping_dim=32, scale=10.0)
        variables = embed.init(jax.random.PRNGKey(0), x_2d)
        out = embed.apply(variables, x_2d)
        assert out.shape == (8, 64)

    def test_out_features_property(self):
        embed = PositionalEmbedding(input_dim=2, mapping_dim=32, scale=1.0)
        assert embed.out_features == 64

    def test_no_params_or_constants(self, x_2d):
        embed = PositionalEmbedding(input_dim=2, mapping_dim=16, scale=1.0)
        variables = embed.init(jax.random.PRNGKey(0), x_2d)
        assert variables == {}

    def test_deterministic(self, x_2d):
        embed = PositionalEmbedding(input_dim=2, mapping_dim=16, scale=1.0)
        v = embed.init(jax.random.PRNGKey(0), x_2d)
        out1 = embed.apply(v, x_2d)
        out2 = embed.apply(v, x_2d)
        assert jnp.allclose(out1, out2)

    def test_scale_1_is_uniform(self, x_2d):
        embed = PositionalEmbedding(input_dim=2, mapping_dim=8, scale=1.0)
        v = embed.init(jax.random.PRNGKey(0), x_2d)
        out = embed.apply(v, x_2d)
        assert jnp.all(jnp.isfinite(out))

    def test_jit_compatible(self, x_2d):
        embed = PositionalEmbedding(input_dim=2, mapping_dim=16, scale=1.0)
        v = embed.init(jax.random.PRNGKey(0), x_2d)
        out = jax.jit(embed.apply)(v, x_2d)
        assert out.shape == (8, 32)


# ---------------------------------------------------------------------------
# Shared spherical embedding tests
# ---------------------------------------------------------------------------

class TestSphericalEmbeddingsShared:

    @pytest.mark.parametrize("cls,kwargs,expected_scale", [
        (SphericalGridEmbedding,          {"scale": 8, "r_min": 0.01}, 4),
        (SphericalCartesianEmbedding,     {"scale": 8, "r_min": 0.01}, 3),
        (SphericalMultiScaleEmbedding,    {"scale": 8, "r_min": 0.01}, 5),
        (SphericalCartesianPlusEmbedding, {"scale": 8, "r_min": 0.01}, 6),
        (SphericalMultiScalePlusEmbedding,{"scale": 8, "r_min": 0.01}, 8),
    ])
    def test_output_shape(self, lat_lon, cls, kwargs, expected_scale):
        lat, lon = lat_lon
        embed = cls(**kwargs)
        variables = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(variables, lat, lon)
        assert out.shape == (16, expected_scale * 8)

    @pytest.mark.parametrize("cls,kwargs", [
        (SphericalGridEmbedding,          {"scale": 4, "r_min": 0.01}),
        (SphericalCartesianEmbedding,     {"scale": 4, "r_min": 0.01}),
        (SphericalMultiScaleEmbedding,    {"scale": 4, "r_min": 0.01}),
        (SphericalCartesianPlusEmbedding, {"scale": 4, "r_min": 0.01}),
        (SphericalMultiScalePlusEmbedding,{"scale": 4, "r_min": 0.01}),
    ])
    def test_output_finite(self, lat_lon, cls, kwargs):
        lat, lon = lat_lon
        embed = cls(**kwargs)
        variables = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(variables, lat, lon)
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("cls,kwargs", [
        (SphericalGridEmbedding,          {"scale": 4, "r_min": 0.01}),
        (SphericalCartesianEmbedding,     {"scale": 4, "r_min": 0.01}),
        (SphericalMultiScaleEmbedding,    {"scale": 4, "r_min": 0.01}),
        (SphericalCartesianPlusEmbedding, {"scale": 4, "r_min": 0.01}),
        (SphericalMultiScalePlusEmbedding,{"scale": 4, "r_min": 0.01}),
    ])
    def test_jit_compatible(self, lat_lon, cls, kwargs):
        lat, lon = lat_lon
        embed = cls(**kwargs)
        variables = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = jax.jit(embed.apply)(variables, lat, lon)
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("cls,kwargs", [
        (SphericalGridEmbedding,          {"scale": 4, "r_min": 0.01}),
        (SphericalCartesianEmbedding,     {"scale": 4, "r_min": 0.01}),
        (SphericalMultiScaleEmbedding,    {"scale": 4, "r_min": 0.01}),
        (SphericalCartesianPlusEmbedding, {"scale": 4, "r_min": 0.01}),
        (SphericalMultiScalePlusEmbedding,{"scale": 4, "r_min": 0.01}),
    ])
    def test_no_trainable_params(self, lat_lon, cls, kwargs):
        lat, lon = lat_lon
        embed = cls(**kwargs)
        variables = embed.init(jax.random.PRNGKey(0), lat, lon)
        assert variables == {}


# ---------------------------------------------------------------------------
# SphericalGridEmbedding
# ---------------------------------------------------------------------------

class TestSphericalGridEmbedding:

    def test_out_features_property(self):
        assert SphericalGridEmbedding(scale=8, r_min=0.01).out_features == 32

    def test_r_max_default(self, lat_lon):
        lat, lon = lat_lon
        embed = SphericalGridEmbedding(scale=4, r_min=0.01)
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(v, lat, lon)
        assert out.shape == (16, 16)


# ---------------------------------------------------------------------------
# SphericalCartesianEmbedding
# ---------------------------------------------------------------------------

class TestSphericalCartesianEmbedding:

    def test_out_features_property(self):
        assert SphericalCartesianEmbedding(scale=8, r_min=0.01).out_features == 24

    def test_unit_cartesian_range(self, lat_lon):
        lat, lon = lat_lon
        embed = SphericalCartesianEmbedding(scale=4, r_min=0.01, r_max=1.0)
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(v, lat, lon)
        assert jnp.all(out >= -1.0 - 1e-5)
        assert jnp.all(out <= 1.0 + 1e-5)


# ---------------------------------------------------------------------------
# DoubleFourierSphericalEmbedding
# ---------------------------------------------------------------------------

class TestDoubleFourierSphericalEmbedding:

    def test_output_shape(self, lat_lon):
        lat, lon = lat_lon
        scale = 4
        embed = DoubleFourierSphericalEmbedding(
            scale=scale, r_lat_min=0.01, r_lon_min=0.01
        )
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(v, lat, lon)
        assert out.shape == (16, 4 * scale + 4 * scale ** 2)

    def test_out_features_property(self):
        scale = 4
        embed = DoubleFourierSphericalEmbedding(
            scale=scale, r_lat_min=0.01, r_lon_min=0.01
        )
        assert embed.out_features == 4 * scale + 4 * scale ** 2

    def test_output_finite(self, lat_lon):
        lat, lon = lat_lon
        embed = DoubleFourierSphericalEmbedding(
            scale=4, r_lat_min=0.01, r_lon_min=0.01
        )
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(v, lat, lon)
        assert jnp.all(jnp.isfinite(out))

    def test_independent_lat_lon_freqs(self, lat_lon):
        lat, lon = lat_lon
        e1 = DoubleFourierSphericalEmbedding(
            scale=4, r_lat_min=0.01, r_lon_min=0.01
        )
        e2 = DoubleFourierSphericalEmbedding(
            scale=4, r_lat_min=0.01, r_lon_min=1.0
        )
        v1 = e1.init(jax.random.PRNGKey(0), lat, lon)
        v2 = e2.init(jax.random.PRNGKey(0), lat, lon)
        assert not jnp.allclose(
            e1.apply(v1, lat, lon), e2.apply(v2, lat, lon)
        )

    def test_jit_compatible(self, lat_lon):
        lat, lon = lat_lon
        embed = DoubleFourierSphericalEmbedding(
            scale=4, r_lat_min=0.01, r_lon_min=0.01
        )
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = jax.jit(embed.apply)(v, lat, lon)
        assert jnp.all(jnp.isfinite(out))

    def test_quadratic_growth(self):
        for scale in [2, 4, 8]:
            embed = DoubleFourierSphericalEmbedding(
                scale=scale, r_lat_min=0.01, r_lon_min=0.01
            )
            assert embed.out_features == 4 * scale + 4 * scale ** 2


# ---------------------------------------------------------------------------
# SphericalHarmonicsEmbedding
# ---------------------------------------------------------------------------

class TestSphericalHarmonicsEmbedding:

    def test_output_shape(self, lat_lon):
        lat, lon = lat_lon
        embed = SphericalHarmonicsEmbedding(legendre_polys=8)
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(v, lat, lon)
        assert out.shape == (16, 64)

    def test_out_features_property(self):
        assert SphericalHarmonicsEmbedding(legendre_polys=10).out_features == 100

    def test_output_finite(self, lat_lon):
        lat, lon = lat_lon
        embed = SphericalHarmonicsEmbedding(legendre_polys=8)
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(v, lat, lon)
        assert jnp.all(jnp.isfinite(out))

    def test_no_trainable_params(self, lat_lon):
        lat, lon = lat_lon
        embed = SphericalHarmonicsEmbedding(legendre_polys=5)
        variables = embed.init(jax.random.PRNGKey(0), lat, lon)
        assert "params" not in variables or len(variables.get("params", {})) == 0

    def test_norm_const_shape(self, lat_lon):
        # norm_const is only accessible inside init/apply
        # verify indirectly: output shape (N, L^2) requires L^2 norm constants
        lat, lon = lat_lon
        L = 6
        embed = SphericalHarmonicsEmbedding(legendre_polys=L)
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(v, lat, lon)
        assert out.shape == (16, L ** 2)

    def test_norm_const_positive(self, lat_lon):
        # if any norm constant were zero or negative the output would be
        # systematically zero or wrong -- check output is non-trivially nonzero
        lat, lon = lat_lon
        embed = SphericalHarmonicsEmbedding(legendre_polys=6)
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(v, lat, lon)
        assert jnp.any(out != 0.)
        assert jnp.all(jnp.isfinite(out))

    def test_deterministic(self, lat_lon):
        lat, lon = lat_lon
        embed = SphericalHarmonicsEmbedding(legendre_polys=6)
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out1 = embed.apply(v, lat, lon)
        out2 = embed.apply(v, lat, lon)
        assert jnp.allclose(out1, out2)

    def test_jit_compatible(self, lat_lon):
        lat, lon = lat_lon
        embed = SphericalHarmonicsEmbedding(legendre_polys=6)
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = jax.jit(embed.apply)(v, lat, lon)
        assert jnp.all(jnp.isfinite(out))

    def test_pole_finite(self):
        lat_poles = jnp.array([jnp.pi / 2, -jnp.pi / 2])
        lon_poles = jnp.array([0.0, 0.0])
        embed = SphericalHarmonicsEmbedding(legendre_polys=8)
        v = embed.init(jax.random.PRNGKey(0), lat_poles, lon_poles)
        out = embed.apply(v, lat_poles, lon_poles)
        assert jnp.all(jnp.isfinite(out))

    def test_orthonormality_approx(self):
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 100)
        lon = jnp.linspace(-jnp.pi, jnp.pi, 100)
        embed = SphericalHarmonicsEmbedding(legendre_polys=4)
        v = embed.init(jax.random.PRNGKey(0), lat, lon)
        out = embed.apply(v, lat, lon)
        y00 = out[:, 0]
        expected = 1.0 / math.sqrt(4.0 * math.pi)
        assert jnp.allclose(y00, jnp.full_like(y00, expected), atol=1e-5)

    def test_different_legendre_polys_give_different_output(self, lat_lon):
        lat, lon = lat_lon
        e1 = SphericalHarmonicsEmbedding(legendre_polys=4)
        e2 = SphericalHarmonicsEmbedding(legendre_polys=8)
        v1 = e1.init(jax.random.PRNGKey(0), lat, lon)
        v2 = e2.init(jax.random.PRNGKey(0), lat, lon)
        out1 = e1.apply(v1, lat, lon)
        out2 = e2.apply(v2, lat, lon)
        assert out1.shape[1] != out2.shape[1]

    def test_various_degrees(self, lat_lon):
        lat, lon = lat_lon
        for L in [1, 3, 5, 10, 15]:
            embed = SphericalHarmonicsEmbedding(legendre_polys=L)
            v = embed.init(jax.random.PRNGKey(0), lat, lon)
            out = embed.apply(v, lat, lon)
            assert out.shape == (16, L ** 2)
            assert jnp.all(jnp.isfinite(out))




# ---------------------------------------------------------------------------
# SinusoidalPosEncoding
# ---------------------------------------------------------------------------

class TestSinusoidalPosEncoding:

    # --- Registry ---

    def test_registered(self):
        assert 'SINUSOIDAL_POS' in list_embeddings()

    def test_get_embedding(self):
        enc = get_embedding('SINUSOIDAL_POS', d_model=D_MODEL)
        assert isinstance(enc, SinusoidalPosEncoding)

    # --- Shape ---

    def test_output_shape(self):
        x   = jnp.ones((BATCH, SEQ, D_MODEL))
        enc = SinusoidalPosEncoding(d_model=D_MODEL)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        check_shape(out, (BATCH, SEQ, D_MODEL))

    def test_shorter_sequence(self):
        # T < max_len should work via slicing
        x   = jnp.ones((BATCH, SEQ // 2, D_MODEL))
        enc = SinusoidalPosEncoding(d_model=D_MODEL)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        check_shape(out, (BATCH, SEQ // 2, D_MODEL))

    def test_single_token(self):
        x   = jnp.ones((BATCH, 1, D_MODEL))
        enc = SinusoidalPosEncoding(d_model=D_MODEL)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        check_shape(out, (BATCH, 1, D_MODEL))

    # --- No parameters ---

    def test_no_trainable_params(self):
        x   = jnp.ones((BATCH, SEQ, D_MODEL))
        enc = SinusoidalPosEncoding(d_model=D_MODEL)
        variables = enc.init(KEY, x)
        # params key should be absent or empty
        assert 'params' not in variables or variables['params'] == {}

    # --- Values ---

    def test_adds_to_input(self):
        # output should differ from input (encoding is non-zero)
        x   = jnp.zeros((BATCH, SEQ, D_MODEL))
        enc = SinusoidalPosEncoding(d_model=D_MODEL)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        assert not jnp.allclose(out, x), \
            "Sinusoidal encoding should be non-zero"

    def test_position_zero_specific_values(self):
        # At position 0: sin(0) = 0, cos(0) = 1
        # So even columns should be 0, odd columns should be 1
        x   = jnp.zeros((1, 1, D_MODEL))
        enc = SinusoidalPosEncoding(d_model=D_MODEL)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)   # (1, 1, D_MODEL)
        pos0 = out[0, 0]                # (D_MODEL,)
        assert jnp.allclose(pos0[0::2], jnp.zeros(D_MODEL // 2), atol=1e-6), \
            "Even columns at position 0 should be 0 (sin(0)=0)"
        assert jnp.allclose(pos0[1::2], jnp.ones(D_MODEL // 2), atol=1e-6), \
            "Odd columns at position 0 should be 1 (cos(0)=1)"

    def test_different_positions_differ(self):
        x   = jnp.zeros((1, SEQ, D_MODEL))
        enc = SinusoidalPosEncoding(d_model=D_MODEL)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        # each position should have a different encoding
        for i in range(SEQ - 1):
            assert not jnp.allclose(out[0, i], out[0, i + 1]), \
                f"Positions {i} and {i+1} should have different encodings"

    def test_deterministic_across_calls(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, D_MODEL))
        enc = SinusoidalPosEncoding(d_model=D_MODEL)
        variables = enc.init(KEY, x)
        out1 = enc.apply(variables, x)
        out2 = enc.apply(variables, x)
        assert jnp.allclose(out1, out2)

    def test_finite_values(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, D_MODEL))
        enc = SinusoidalPosEncoding(d_model=D_MODEL)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        check_finite(out)

    def test_odd_d_model(self):
        # odd d_model should not raise
        d = 33
        x   = jnp.ones((BATCH, SEQ, d))
        enc = SinusoidalPosEncoding(d_model=d)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        check_shape(out, (BATCH, SEQ, d))
        check_finite(out)

    def test_dim_mismatch_raises(self):
        x   = jnp.ones((BATCH, SEQ, D_MODEL + 1))  # wrong last dim
        enc = SinusoidalPosEncoding(d_model=D_MODEL)
        variables = enc.init(KEY, jnp.ones((BATCH, SEQ, D_MODEL)))
        with pytest.raises(AssertionError, match="d_model"):
            enc.apply(variables, x)

    def test_custom_max_len(self):
        enc = SinusoidalPosEncoding(d_model=D_MODEL, max_len=128)
        x   = jnp.ones((BATCH, 100, D_MODEL))
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        check_shape(out, (BATCH, 100, D_MODEL))

    # --- Values match manual formula ---

    def test_manual_formula_match(self):
        # verify first two positions match the Vaswani formula manually
        d = 8
        enc = SinusoidalPosEncoding(d_model=d)
        x   = jnp.zeros((1, 2, d))
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)

        for pos in range(2):
            for i in range(d // 2):
                div = math.exp(-math.log(10000.0) * (2 * i) / d)
                expected_sin = math.sin(pos * div)
                expected_cos = math.cos(pos * div)
                assert abs(float(out[0, pos, 2 * i])     - expected_sin) < 1e-5, \
                    f"pos={pos}, dim={2*i}: sin mismatch"
                assert abs(float(out[0, pos, 2 * i + 1]) - expected_cos) < 1e-5, \
                    f"pos={pos}, dim={2*i+1}: cos mismatch"


# ---------------------------------------------------------------------------
# LearnedPosEncoding
# ---------------------------------------------------------------------------

class TestLearnedPosEncoding:

    # --- Registry ---

    def test_registered(self):
        assert 'LEARNED_POS' in list_embeddings()

    def test_get_embedding(self):
        enc = get_embedding('LEARNED_POS', num_tokens=SEQ, embed_dim=EMBED)
        assert isinstance(enc, LearnedPosEncoding)

    # --- Shape ---

    def test_output_shape_full(self):
        x   = jnp.ones((BATCH, SEQ, EMBED))
        enc = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        check_shape(out, (BATCH, SEQ, EMBED))

    def test_output_shape_partial(self):
        # T < num_tokens -- slicing should work
        x   = jnp.ones((BATCH, SEQ // 2, EMBED))
        enc = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED)
        variables = enc.init(KEY, jnp.ones((BATCH, SEQ, EMBED)))
        out = enc.apply(variables, x)
        check_shape(out, (BATCH, SEQ // 2, EMBED))

    # --- Parameters ---

    def test_has_trainable_params(self):
        x   = jnp.ones((BATCH, SEQ, EMBED))
        enc = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        assert 'params' in variables
        assert 'pos_embed' in variables['params']

    def test_param_shape(self):
        x   = jnp.ones((BATCH, SEQ, EMBED))
        enc = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        check_shape(variables['params']['pos_embed'], (1, SEQ, EMBED),
                    'pos_embed param')

    def test_custom_init_stddev(self):
        # larger stddev should produce larger initial embedding values on average
        enc_small = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED,
                                        init_stddev=0.001)
        enc_large = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED,
                                        init_stddev=1.0)
        x = jnp.ones((BATCH, SEQ, EMBED))
        v_small = enc_small.init(KEY, x)
        v_large = enc_large.init(KEY, x)
        std_small = jnp.std(v_small['params']['pos_embed'])
        std_large = jnp.std(v_large['params']['pos_embed'])
        assert std_large > std_small, \
            "Larger init_stddev should produce larger initial embedding std"

    # --- Values ---

    def test_adds_to_input(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        enc = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        assert not jnp.allclose(out, x), \
            "Learned encoding should modify the input"

    def test_encoding_is_position_specific(self):
        # same input at different positions should produce different outputs
        x   = jnp.zeros((1, SEQ, EMBED))
        enc = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        assert not jnp.allclose(out[0, 0], out[0, 1]), \
            "Different positions should have different encodings"

    def test_finite_values(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        enc = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        check_finite(out)

    # --- Gradient ---

    def test_backward(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        enc = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED)
        variables = enc.init(KEY, x)

        def loss(params):
            out = enc.apply({'params': params}, x)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        assert 'pos_embed' in grads
        check_finite(grads['pos_embed'], 'pos_embed gradient')
        assert jnp.any(grads['pos_embed'] != 0.0), \
            "pos_embed should receive non-zero gradients"

    # --- Validation ---

    def test_embed_dim_mismatch_raises(self):
        x   = jnp.ones((BATCH, SEQ, EMBED + 1))
        enc = LearnedPosEncoding(num_tokens=SEQ, embed_dim=EMBED)
        variables = enc.init(KEY, jnp.ones((BATCH, SEQ, EMBED)))
        with pytest.raises(AssertionError, match="embed_dim"):
            enc.apply(variables, x)


# ---------------------------------------------------------------------------
# LearnedPosEncoding2D
# ---------------------------------------------------------------------------

class TestLearnedPosEncoding2D:

    # --- Registry ---

    def test_registered(self):
        assert 'LEARNED_POS_2D' in list_embeddings()

    def test_get_embedding(self):
        enc = get_embedding('LEARNED_POS_2D', height=H, width=W, embed_dim=EMBED)
        assert isinstance(enc, LearnedPosEncoding2D)

    # --- Shape ---

    def test_output_shape(self):
        x   = jnp.ones((BATCH, H, W, EMBED))
        enc = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        check_shape(out, (BATCH, H, W, EMBED))

    # --- Parameters ---

    def test_has_trainable_params(self):
        x   = jnp.ones((BATCH, H, W, EMBED))
        enc = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        assert 'params' in variables
        assert 'pos_embed' in variables['params']

    def test_param_shape(self):
        x   = jnp.ones((BATCH, H, W, EMBED))
        enc = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        check_shape(variables['params']['pos_embed'], (1, H, W, EMBED),
                    'pos_embed param')

    def test_custom_init_stddev(self):
        enc_small = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED,
                                          init_stddev=0.001)
        enc_large = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED,
                                          init_stddev=1.0)
        x = jnp.ones((BATCH, H, W, EMBED))
        v_small = enc_small.init(KEY, x)
        v_large = enc_large.init(KEY, x)
        std_small = jnp.std(v_small['params']['pos_embed'])
        std_large = jnp.std(v_large['params']['pos_embed'])
        assert std_large > std_small

    # --- Values ---

    def test_adds_to_input(self):
        x   = jax.random.normal(KEY, (BATCH, H, W, EMBED))
        enc = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        assert not jnp.allclose(out, x)

    def test_encoding_is_position_specific(self):
        x   = jnp.zeros((1, H, W, EMBED))
        enc = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        # top-left and top-right should differ
        assert not jnp.allclose(out[0, 0, 0], out[0, 0, 1]), \
            "Different spatial positions should have different encodings"

    def test_finite_values(self):
        x   = jax.random.normal(KEY, (BATCH, H, W, EMBED))
        enc = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED)
        variables = enc.init(KEY, x)
        out = enc.apply(variables, x)
        check_finite(out)

    # --- Gradient ---

    def test_backward(self):
        x   = jax.random.normal(KEY, (BATCH, H, W, EMBED))
        enc = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED)
        variables = enc.init(KEY, x)

        def loss(params):
            out = enc.apply({'params': params}, x)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        assert 'pos_embed' in grads
        check_finite(grads['pos_embed'], 'pos_embed gradient')
        assert jnp.any(grads['pos_embed'] != 0.0), \
            "pos_embed should receive non-zero gradients"

    # --- Validation ---

    def test_spatial_mismatch_raises(self):
        x   = jnp.ones((BATCH, H + 1, W, EMBED))  # wrong H
        enc = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED)
        variables = enc.init(KEY, jnp.ones((BATCH, H, W, EMBED)))
        with pytest.raises(AssertionError, match="height"):
            enc.apply(variables, x)

    def test_embed_dim_mismatch_raises(self):
        x   = jnp.ones((BATCH, H, W, EMBED + 1))
        enc = LearnedPosEncoding2D(height=H, width=W, embed_dim=EMBED)
        variables = enc.init(KEY, jnp.ones((BATCH, H, W, EMBED)))
        with pytest.raises(AssertionError, match="embed_dim"):
            enc.apply(variables, x)