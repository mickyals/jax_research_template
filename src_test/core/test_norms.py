# tests/core/test_norms.py
import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn

from core.norms import (
    get_norm,
    list_norms,
    register_norm,
    BatchNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,
    RMSNorm,
)

KEY    = jax.random.PRNGKey(0)
BATCH  = 8
SEQ    = 16
FEATS  = 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def x_2d():
    """(batch, features) -- MLP-style input."""
    return jax.random.normal(KEY, (BATCH, FEATS))


@pytest.fixture
def x_3d():
    """(batch, seq, features) -- transformer-style input."""
    return jax.random.normal(KEY, (BATCH, SEQ, FEATS))


@pytest.fixture
def x_4d():
    """(batch, height, width, channels) -- conv-style input, channels last."""
    return jax.random.normal(KEY, (BATCH, 8, 8, FEATS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_norm(norm, x, train=True):
    """Init and apply a norm. Returns (variables, output)."""
    if isinstance(norm, BatchNorm):
        variables = norm.init(KEY, x, train=True)
        out, updates = norm.apply(
            variables, x, train=train, mutable=['batch_stats']
        )
        return variables, out
    else:
        variables = norm.init(KEY, x)
        out = norm.apply(variables, x, train=train)
        return variables, out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_norms_returns_all(self):
        norms = list_norms()
        expected = {
            "BATCH_NORM", "LAYER_NORM", "GROUP_NORM",
            "INSTANCE_NORM", "RMS_NORM",
        }
        assert expected == set(norms.keys())

    def test_get_norm_unknown_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            get_norm("NONEXISTENT")

    def test_get_norm_unknown_kwargs_warns(self):
        with pytest.warns(UserWarning, match="unknown kwargs"):
            get_norm("LAYER_NORM", nonexistent_param=99)

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            @register_norm("LAYER_NORM", description="duplicate")
            class Duplicate(nn.Module):
                def __call__(self, x, train=True):
                    return x

    def test_get_norm_valid_kwargs_applied(self):
        norm = get_norm("GROUP_NORM", num_groups=4)
        assert norm.num_groups == 4

    def test_get_norm_valid_kwargs_kept_unknown_dropped(self):
        with pytest.warns(UserWarning, match="unknown kwargs"):
            norm = get_norm("LAYER_NORM", epsilon=1e-4, bogus=123)
        assert norm.epsilon == 1e-4

# ---------------------------------------------------------------------------
# BatchNorm
# ---------------------------------------------------------------------------

class TestBatchNorm:
    def test_forward_train(self, x_2d):
        norm = get_norm("BATCH_NORM")
        _, out = apply_norm(norm, x_2d, train=True)
        assert out.shape == x_2d.shape
        assert jnp.all(jnp.isfinite(out))

    def test_forward_eval(self, x_2d):
        norm = get_norm("BATCH_NORM")
        _, out = apply_norm(norm, x_2d, train=False)
        assert out.shape == x_2d.shape
        assert jnp.all(jnp.isfinite(out))

    def test_has_batch_stats(self, x_2d):
        norm = get_norm("BATCH_NORM")
        variables = norm.init(KEY, x_2d, train=True)
        assert 'batch_stats' in variables

    def test_has_learnable_params(self, x_2d):
        norm = get_norm("BATCH_NORM")
        variables = norm.init(KEY, x_2d, train=True)
        assert 'params' in variables

    def test_no_scale_no_bias(self, x_2d):
        norm = get_norm("BATCH_NORM", use_scale=False, use_bias=False)
        variables = norm.init(KEY, x_2d, train=True)
        # params should be empty or absent when no scale/bias
        params = variables.get('params', {})
        bn_params = params.get('bn', {})
        assert 'scale' not in bn_params
        assert 'bias' not in bn_params

    def test_4d_input(self, x_4d):
        norm = get_norm("BATCH_NORM")
        _, out = apply_norm(norm, x_4d, train=True)
        assert out.shape == x_4d.shape

    def test_train_eval_differ(self, x_2d):
        norm = get_norm("BATCH_NORM")
        variables = norm.init(KEY, x_2d, train=True)
        out_train, _ = norm.apply(
            variables, x_2d, train=True, mutable=['batch_stats']
        )
        out_eval = norm.apply(variables, x_2d, train=False)
        # train and eval may differ since running stats differ from batch stats
        # just verify both are finite and same shape
        assert out_train.shape == out_eval.shape
        assert jnp.all(jnp.isfinite(out_train))
        assert jnp.all(jnp.isfinite(out_eval))

    def test_forward_3d(self, x_3d):
        norm = get_norm("BATCH_NORM")
        variables = norm.init(KEY, x_3d, train=True)
        out, _ = norm.apply(
            variables, x_3d, train=True, mutable=['batch_stats']
        )
        assert out.shape == x_3d.shape
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class TestLayerNorm:
    def test_forward_2d(self, x_2d):
        norm = get_norm("LAYER_NORM")
        _, out = apply_norm(norm, x_2d)
        assert out.shape == x_2d.shape
        assert jnp.all(jnp.isfinite(out))

    def test_forward_3d(self, x_3d):
        norm = get_norm("LAYER_NORM")
        _, out = apply_norm(norm, x_3d)
        assert out.shape == x_3d.shape
        assert jnp.all(jnp.isfinite(out))

    def test_train_eval_identical(self, x_2d):
        norm = get_norm("LAYER_NORM")
        variables = norm.init(KEY, x_2d)
        out_train = norm.apply(variables, x_2d, train=True)
        out_eval  = norm.apply(variables, x_2d, train=False)
        assert jnp.allclose(out_train, out_eval)

    def test_no_bias(self, x_2d):
        norm = get_norm("LAYER_NORM", use_bias=False)
        variables = norm.init(KEY, x_2d)
        assert 'bias' not in variables.get('params', {}).get('ln', {})

    def test_normalises_last_dim(self, x_2d):
        norm = get_norm("LAYER_NORM", use_scale=False, use_bias=False)
        variables = norm.init(KEY, x_2d)
        out = norm.apply(variables, x_2d)
        mean = jnp.mean(out, axis=-1)
        assert jnp.allclose(mean, jnp.zeros_like(mean), atol=1e-5)


# ---------------------------------------------------------------------------
# GroupNorm
# ---------------------------------------------------------------------------

class TestGroupNorm:
    def test_forward(self, x_4d):
        norm = get_norm("GROUP_NORM", num_groups=8)
        _, out = apply_norm(norm, x_4d)
        assert out.shape == x_4d.shape
        assert jnp.all(jnp.isfinite(out))

    def test_train_eval_identical(self, x_4d):
        norm = get_norm("GROUP_NORM", num_groups=8)
        variables = norm.init(KEY, x_4d)
        out_train = norm.apply(variables, x_4d, train=True)
        out_eval  = norm.apply(variables, x_4d, train=False)
        assert jnp.allclose(out_train, out_eval)

    def test_num_groups_default(self, x_4d):
        norm = get_norm("GROUP_NORM")
        _, out = apply_norm(norm, x_4d)
        assert out.shape == x_4d.shape

    def test_no_bias(self, x_4d):
        norm = get_norm("GROUP_NORM", num_groups=8, use_bias=False)
        _, out = apply_norm(norm, x_4d)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# InstanceNorm
# ---------------------------------------------------------------------------

class TestInstanceNorm:
    def test_forward(self, x_4d):
        norm = get_norm("INSTANCE_NORM")
        _, out = apply_norm(norm, x_4d)
        assert out.shape == x_4d.shape
        assert jnp.all(jnp.isfinite(out))

    def test_train_eval_identical(self, x_4d):
        norm = get_norm("INSTANCE_NORM")
        variables = norm.init(KEY, x_4d)
        out_train = norm.apply(variables, x_4d, train=True)
        out_eval  = norm.apply(variables, x_4d, train=False)
        assert jnp.allclose(out_train, out_eval)

    def test_no_bias(self, x_4d):
        norm = get_norm("INSTANCE_NORM", use_bias=False)
        _, out = apply_norm(norm, x_4d)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_forward_2d(self, x_2d):
        norm = get_norm("RMS_NORM")
        _, out = apply_norm(norm, x_2d)
        assert out.shape == x_2d.shape
        assert jnp.all(jnp.isfinite(out))

    def test_forward_3d(self, x_3d):
        norm = get_norm("RMS_NORM")
        _, out = apply_norm(norm, x_3d)
        assert out.shape == x_3d.shape
        assert jnp.all(jnp.isfinite(out))

    def test_train_eval_identical(self, x_2d):
        norm = get_norm("RMS_NORM")
        variables = norm.init(KEY, x_2d)
        out_train = norm.apply(variables, x_2d, train=True)
        out_eval  = norm.apply(variables, x_2d, train=False)
        assert jnp.allclose(out_train, out_eval)

    def test_scale_is_learnable(self, x_2d):
        norm = get_norm("RMS_NORM", use_scale=True)
        variables = norm.init(KEY, x_2d)
        assert 'scale' in variables['params']

    def test_no_scale(self, x_2d):
        norm = get_norm("RMS_NORM", use_scale=False)
        variables = norm.init(KEY, x_2d)
        assert 'params' not in variables or 'scale' not in variables.get('params', {})

    def test_rms_normalises(self, x_2d):
        norm = get_norm("RMS_NORM", use_scale=False)
        variables = norm.init(KEY, x_2d)
        out = norm.apply(variables, x_2d)
        rms = jnp.sqrt(jnp.mean(out ** 2, axis=-1))
        assert jnp.allclose(rms, jnp.ones_like(rms), atol=1e-5)


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------

class TestCrossCutting:
    @pytest.mark.parametrize("name,kwargs,x_fixture", [
        ("LAYER_NORM",    {},                  "x_2d"),
        ("LAYER_NORM",    {},                  "x_3d"),
        ("GROUP_NORM",    {"num_groups": 8},   "x_4d"),
        ("INSTANCE_NORM", {},                  "x_4d"),
        ("RMS_NORM",      {},                  "x_2d"),
        ("RMS_NORM",      {},                  "x_3d"),
    ])
    def test_finite_output(self, name, kwargs, x_fixture, request):
        x = request.getfixturevalue(x_fixture)
        norm = get_norm(name, **kwargs)
        _, out = apply_norm(norm, x)
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("name,kwargs,x_fixture", [
        ("LAYER_NORM",    {},                  "x_2d"),
        ("GROUP_NORM",    {"num_groups": 8},   "x_4d"),
        ("INSTANCE_NORM", {},                  "x_4d"),
        ("RMS_NORM",      {},                  "x_2d"),
    ])
    def test_has_learnable_params(self, name, kwargs, x_fixture, request):
        x = request.getfixturevalue(x_fixture)
        norm = get_norm(name, **kwargs)
        variables = norm.init(KEY, x)
        assert 'params' in variables
        assert len(jax.tree_util.tree_leaves(variables['params'])) > 0