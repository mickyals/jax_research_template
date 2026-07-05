"""
Tests for models/features.py — packing order, encoding modes, flatten.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.embeddings import LatLonEmbeddingWrapper
from experiments.cyclone_jax.models.features import (
    COORD_FIELDS, TOKEN_FIELDS, FeatureEncoder, build_encoder, flatten,
    pack_coords, pack_tokens,
)

from .conftest import B, C, N

F_TOKENS = 3 + 2 * C          # level/time/id scalars + obs + missing


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

class TestPacking:

    def test_token_shape_and_dtype(self, X):
        t = pack_tokens(X)
        assert t.shape == (B, N, F_TOKENS)
        assert t.dtype == jnp.float32

    def test_token_field_order_is_the_defined_order(self, X):
        """Column blocks follow TOKEN_FIELDS: level, time, id, obs, missing."""
        assert TOKEN_FIELDS == ('level', 'time', 'id', 'obs', 'missing')
        t = np.asarray(pack_tokens(X))
        assert (t[..., 0] == 2.0).all()               # level
        assert (t[..., 1] == 3.0).all()               # time
        assert (t[..., 2] == 4.0).all()               # id
        assert (t[..., 3:3 + C] == 5.0).all()         # obs
        assert (t[..., 3 + C:] == 1.0).all()          # missing (bool -> 1.0)

    def test_coords_shape_and_order(self, X):
        assert COORD_FIELDS == ('lat', 'lon')
        c = np.asarray(pack_coords(X))
        assert c.shape == (B, N, 2)
        assert (c[..., 0] == 10.0).all() and (c[..., 1] == -60.0).all()

    def test_flatten(self, X):
        assert flatten(pack_tokens(X)).shape == (B, N * F_TOKENS)


# ---------------------------------------------------------------------------
# Encoding modes
# ---------------------------------------------------------------------------

class TestFeatureEncoder:

    def test_concat_raw_appends_coords(self, X):
        enc = FeatureEncoder(mode='concat')
        out = enc.apply(enc.init(jax.random.PRNGKey(0), X), X)
        assert out.shape == (B, N, F_TOKENS + 2)
        np.testing.assert_array_equal(np.asarray(out[..., :F_TOKENS]),
                                      np.asarray(pack_tokens(X)))
        np.testing.assert_array_equal(np.asarray(out[..., F_TOKENS:]),
                                      np.asarray(pack_coords(X)))

    def test_concat_raw_has_no_params(self, X):
        variables = FeatureEncoder(mode='concat').init(
            jax.random.PRNGKey(0), X)
        assert not jax.tree_util.tree_leaves(variables.get('params', {}))

    def test_concat_with_embedding_widens_position_block(self, X):
        enc = build_encoder({'mode': 'concat',
                             'embedding': 'GAUSSIAN_POSITIONAL',
                             'embedding_kwargs': {'input_dim': 2,
                                                  'mapping_dim': 8,
                                                  'scale': 1.0}})
        out = enc.apply(enc.init(jax.random.PRNGKey(0), X), X)
        assert out.shape == (B, N, F_TOKENS + 8)

    def test_additive_keeps_token_width(self, X):
        enc = build_encoder({'mode': 'additive',
                             'embedding': 'GAUSSIAN_POSITIONAL',
                             'embedding_kwargs': {'input_dim': 2,
                                                  'mapping_dim': 8,
                                                  'scale': 1.0}})
        variables = enc.init(jax.random.PRNGKey(0), X)
        out = enc.apply(variables, X)
        assert out.shape == (B, N, F_TOKENS)
        # the projection is the mode's parameter — and the output differs
        # from the bare tokens (position actually entered)
        assert jax.tree_util.tree_leaves(variables['params'])
        assert not np.allclose(np.asarray(out), np.asarray(pack_tokens(X)))

    def test_additive_raw_coords_projects_them(self, X):
        enc = build_encoder({'mode': 'additive'})
        out = enc.apply(enc.init(jax.random.PRNGKey(0), X), X)
        assert out.shape == (B, N, F_TOKENS)

    def test_bad_mode_raises(self, X):
        enc = FeatureEncoder(mode='fourier')
        with pytest.raises(ValueError, match='concat'):
            enc.init(jax.random.PRNGKey(0), X)


# ---------------------------------------------------------------------------
# build_encoder
# ---------------------------------------------------------------------------

class TestBuildEncoder:

    def test_none_block_is_raw_concat(self):
        enc = build_encoder(None)
        assert enc.mode == 'concat' and enc.embedding is None

    def test_latlon_family_gets_wrapped(self, X):
        enc = build_encoder({'mode': 'concat', 'embedding': 'SPHERE_GRID',
                             'embedding_kwargs': {'scale': 4,
                                                  'r_min': 0.01}})
        assert isinstance(enc.embedding, LatLonEmbeddingWrapper)
        out = enc.apply(enc.init(jax.random.PRNGKey(0), X), X)
        assert out.shape == (B, N, F_TOKENS + 4 * 4)   # SPHERE_GRID: 4*scale

    def test_xy_embedding_not_wrapped(self):
        enc = build_encoder({'embedding': 'GAUSSIAN_POSITIONAL',
                             'embedding_kwargs': {'input_dim': 2,
                                                  'mapping_dim': 8,
                                                  'scale': 1.0}})
        assert not isinstance(enc.embedding, LatLonEmbeddingWrapper)
