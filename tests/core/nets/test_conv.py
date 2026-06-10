# tests/core/nets/test_conv.py
import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from core.nets.conv import (
    # blocks 2D
    ConvBlock,
    ResidualBlock,
    DownsampleBlock,
    UpsampleBlock,
    NonLocalBlock,
    InceptionBlock,
    DenseLayer,
    DenseBlock,
    TransitionLayer,
    PatchEmbed,
    # blocks 1D
    ConvBlock1d,
    ResidualBlock1d,
    DownsampleBlock1d,
    UpsampleBlock1d,
    # nets
    ConvEncoder,
    ConvDecoder,
    ResNet,
    DenseNet,
    # registry
    get_conv_net,
    list_conv_nets,
    register_conv_net,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY     = jax.random.PRNGKey(0)
BATCH   = 4
H, W    = 32, 32
C_IN    = 8
C_OUT   = 16
FEATS   = 32
SEQ_LEN = 64
CLASSES = 10

DROP_KEY = jax.random.PRNGKey(1)

PATCH   = 4
IMG_H   = 32
IMG_W   = 32
PEMBED  = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def x_2d():
    """(B, H, W, C) channels-last spatial input."""
    return jax.random.normal(KEY, (BATCH, H, W, C_IN))


@pytest.fixture
def x_2d_sq():
    """Square spatial input with FEATS channels."""
    return jax.random.normal(KEY, (BATCH, H, W, FEATS))


@pytest.fixture
def x_1d():
    """(B, L, C) sequence input."""
    return jax.random.normal(KEY, (BATCH, SEQ_LEN, C_IN))


@pytest.fixture
def key():
    return KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_block(module, x, train=True, with_dropout=False, with_batch_stats=False):
    """Init and apply a block. Returns (variables, output)."""
    rngs = {'dropout': DROP_KEY} if with_dropout else {}
    variables = module.init({'params': KEY, **rngs}, x, train=True)

    mutable = ['batch_stats'] if with_batch_stats else []

    if mutable:
        out, updates = module.apply(
            variables, x, train=train,
            rngs={'dropout': DROP_KEY} if with_dropout else {},
            mutable=mutable,
        )
        return variables, out, updates
    else:
        out = module.apply(
            variables, x, train=train,
            rngs={'dropout': DROP_KEY} if with_dropout else {},
        )
        return variables, out


def check_shape_finite(out, expected_shape):
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    assert jnp.all(jnp.isfinite(out)), "Output contains non-finite values"


def check_backward(module, x, with_dropout=False, with_batch_stats=False):
    rngs = {'dropout': DROP_KEY} if with_dropout else {}
    variables = module.init({'params': KEY, **rngs}, x, train=True)

    def loss_fn(params):
        apply_vars = {'params': params}
        if 'batch_stats' in variables:
            apply_vars['batch_stats'] = variables['batch_stats']
        if with_batch_stats:
            out, _ = module.apply(
                apply_vars, x, train=True,
                rngs={'dropout': DROP_KEY} if with_dropout else {},
                mutable=['batch_stats'],
            )
        else:
            out = module.apply(
                apply_vars, x, train=False,
                rngs={'dropout': DROP_KEY} if with_dropout else {},
            )
        return jnp.mean(out ** 2)

    grads = jax.grad(loss_fn)(variables['params'])
    leaves = jax.tree_util.tree_leaves(grads)
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), "Gradient contains non-finite values"


# ---------------------------------------------------------------------------
# ConvBlock
# ---------------------------------------------------------------------------

class TestConvBlock:
    def test_forward_default(self, x_2d):
        block = ConvBlock(features=FEATS)
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_pre_norm(self, x_2d):
        block = ConvBlock(features=FEATS, pre_norm=True)
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_with_pooling(self, x_2d):
        block = ConvBlock(features=FEATS,
                          pooling='SPATIAL_MAX',
                          pooling_kwargs={'kernel_size': (2, 2), 'strides': (2, 2)})
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H // 2, W // 2, FEATS))

    def test_with_dropout_train(self, x_2d):
        block = ConvBlock(features=FEATS, dropout_rate=0.5)
        _, out = apply_block(block, x_2d, with_dropout=True, train=True)
        check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_dropout_eval_deterministic(self, x_2d):
        block = ConvBlock(features=FEATS, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY}, x_2d, train=True)
        out1 = block.apply(variables, x_2d, train=False)
        out2 = block.apply(variables, x_2d, train=False)
        assert jnp.allclose(out1, out2)

    def test_dropout_train_stochastic(self, x_2d):
        block = ConvBlock(features=FEATS, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY}, x_2d, train=True)
        out1 = block.apply(variables, x_2d, train=True,
                           rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = block.apply(variables, x_2d, train=True,
                           rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)

    def test_activation_variants(self, x_2d):
        for act in ["relu", "tanh", "gelu", "silu"]:
            block = ConvBlock(features=FEATS, activation=act)
            _, out = apply_block(block, x_2d)
            check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_norm_variants(self, x_2d):
        for norm in ["GROUP_NORM", "LAYER_NORM", "INSTANCE_NORM"]:
            block = ConvBlock(features=FEATS, norm=norm)
            _, out = apply_block(block, x_2d)
            check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_backward(self, x_2d):
        check_backward(ConvBlock(features=FEATS), x_2d)

    def test_backward_with_dropout(self, x_2d):
        check_backward(ConvBlock(features=FEATS, dropout_rate=0.1), x_2d,
                       with_dropout=True)

    def test_strides_downsample(self, x_2d):
        block = ConvBlock(features=FEATS, strides=(2, 2))
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H // 2, W // 2, FEATS))


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------

class TestResidualBlock:
    def test_forward_same_channels(self, x_2d_sq):
        block = ResidualBlock(features=FEATS)
        _, out = apply_block(block, x_2d_sq)
        check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_forward_channel_mismatch(self, x_2d):
        block = ResidualBlock(features=FEATS)
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_pre_norm(self, x_2d):
        block = ResidualBlock(features=FEATS, pre_norm=True)
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_with_pooling(self, x_2d_sq):
        block = ResidualBlock(features=FEATS,
                              pooling='SPATIAL_AVG',
                              pooling_kwargs={'kernel_size': (2, 2), 'strides': (2, 2)})
        _, out = apply_block(block, x_2d_sq)
        check_shape_finite(out, (BATCH, H // 2, W // 2, FEATS))

    def test_dropout_eval_deterministic(self, x_2d):
        block = ResidualBlock(features=FEATS, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY}, x_2d, train=True)
        out1 = block.apply(variables, x_2d, train=False)
        out2 = block.apply(variables, x_2d, train=False)
        assert jnp.allclose(out1, out2)

    def test_dropout_train_stochastic(self, x_2d):
        block = ResidualBlock(features=FEATS, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY}, x_2d, train=True)
        out1 = block.apply(variables, x_2d, train=True,
                           rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = block.apply(variables, x_2d, train=True,
                           rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)

    def test_norm_variants(self, x_2d):
        for norm in ["GROUP_NORM", "LAYER_NORM", "INSTANCE_NORM"]:
            block = ResidualBlock(features=FEATS, norm=norm)
            _, out = apply_block(block, x_2d)
            check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_backward(self, x_2d):
        check_backward(ResidualBlock(features=FEATS), x_2d)

    def test_backward_with_dropout(self, x_2d):
        check_backward(ResidualBlock(features=FEATS, dropout_rate=0.1), x_2d,
                       with_dropout=True)

    def test_skip_connection_identity(self, x_2d_sq):
        # zero-init the block weights should give output close to input
        # just verify residual path exists by checking output not all zero
        block = ResidualBlock(features=FEATS)
        variables = block.init(KEY, x_2d_sq, train=False)
        out = block.apply(variables, x_2d_sq, train=False)
        assert not jnp.allclose(out, jnp.zeros_like(out))

    def test_pre_norm_with_pooling(self, x_2d):
        block = ResidualBlock(features=FEATS, pre_norm=True,
                              pooling='SPATIAL_AVG',
                              pooling_kwargs={'kernel_size': (2, 2), 'strides': (2, 2)})
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H // 2, W // 2, FEATS))


# ---------------------------------------------------------------------------
# DownsampleBlock
# ---------------------------------------------------------------------------

class TestDownsampleBlock:
    def test_asymmetric_default(self, x_2d):
        block = DownsampleBlock(features=FEATS)
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H // 2, W // 2, FEATS))

    def test_same_padding(self, x_2d):
        block = DownsampleBlock(features=FEATS, padding_mode='same')
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H // 2, W // 2, FEATS))

    def test_pool_type_avg(self, x_2d):
        block = DownsampleBlock(features=FEATS, pool_type='SPATIAL_AVG')
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H // 2, W // 2, FEATS))

    def test_pool_type_max(self, x_2d):
        block = DownsampleBlock(features=FEATS, pool_type='SPATIAL_MAX')
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H // 2, W // 2, FEATS))

    def test_asymmetric_exact_halving(self):
        # asymmetric padding guarantees exact H//2 for even spatial dims
        # ensuring clean encoder/decoder spatial alignment
        x = jax.random.normal(KEY, (BATCH, 32, 32, C_IN))
        block = DownsampleBlock(features=FEATS, padding_mode='asymmetric')
        variables = block.init(KEY, x)
        out = block.apply(variables, x)
        assert out.shape == (BATCH, 16, 16, FEATS)

    def test_same_padding_odd_dims(self):
        # odd dims: SAME padding gives ceil(H/2) = 17, preserving more spatial info
        x = jax.random.normal(KEY, (BATCH, 33, 33, C_IN))
        block = DownsampleBlock(features=FEATS, padding_mode='same')
        variables = block.init(KEY, x)
        out = block.apply(variables, x)
        assert out.shape == (BATCH, 17, 17, FEATS)

    def test_asymmetric_odd_dims(self):
        # odd dims: asymmetric loses a pixel -- floors to 16 not 17
        # use SAME padding if working with odd spatial dimensions
        x = jax.random.normal(KEY, (BATCH, 33, 33, C_IN))
        block = DownsampleBlock(features=FEATS, padding_mode='asymmetric')
        variables = block.init(KEY, x)
        out = block.apply(variables, x)
        assert out.shape == (BATCH, 16, 16, FEATS)

    def test_asymmetric_vs_same_padding_even_dims(self):
        # both modes should give identical H//2 for even spatial dims
        x = jax.random.normal(KEY, (BATCH, 32, 32, C_IN))
        b_asym = DownsampleBlock(features=FEATS, padding_mode='asymmetric')
        b_same = DownsampleBlock(features=FEATS, padding_mode='same')
        v_asym = b_asym.init(KEY, x)
        v_same = b_same.init(KEY, x)
        out_asym = b_asym.apply(v_asym, x)
        out_same = b_same.apply(v_same, x)
        assert out_asym.shape == out_same.shape == (BATCH, 16, 16, FEATS)

    def test_backward(self, x_2d):
        check_backward(DownsampleBlock(features=FEATS), x_2d)


# ---------------------------------------------------------------------------
# UpsampleBlock
# ---------------------------------------------------------------------------

class TestUpsampleBlock:
    def test_forward(self, x_2d):
        block = UpsampleBlock(features=FEATS)
        _, out = apply_block(block, x_2d)
        check_shape_finite(out, (BATCH, H * 2, W * 2, FEATS))

    def test_encoder_decoder_roundtrip_shape(self, x_2d):
        down = DownsampleBlock(features=FEATS)
        up = UpsampleBlock(features=C_IN)
        v_down = down.init(KEY, x_2d)
        x_small = down.apply(v_down, x_2d)
        v_up = up.init(KEY, x_small)
        x_up = up.apply(v_up, x_small)
        assert x_up.shape == (BATCH, H, W, C_IN)

    def test_backward(self, x_2d):
        check_backward(UpsampleBlock(features=FEATS), x_2d)


# ---------------------------------------------------------------------------
# NonLocalBlock
# ---------------------------------------------------------------------------

class TestNonLocalBlock:
    def test_forward(self, x_2d_sq):
        block = NonLocalBlock()
        _, out = apply_block(block, x_2d_sq)
        check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_output_same_shape_as_input(self, x_2d_sq):
        block = NonLocalBlock()
        variables = block.init(KEY, x_2d_sq, train=False)
        out = block.apply(variables, x_2d_sq, train=False)
        assert out.shape == x_2d_sq.shape

    def test_downsample_factor(self, x_2d_sq):
        block = NonLocalBlock(downsample_factor=2)
        _, out = apply_block(block, x_2d_sq)
        check_shape_finite(out, (BATCH, H, W, FEATS))

    def test_dropout_eval_deterministic(self, x_2d_sq):
        block = NonLocalBlock(dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY},
                               x_2d_sq, train=True)
        out1 = block.apply(variables, x_2d_sq, train=False)
        out2 = block.apply(variables, x_2d_sq, train=False)
        assert jnp.allclose(out1, out2)

    def test_backward(self, x_2d_sq):
        check_backward(NonLocalBlock(), x_2d_sq)

    def test_residual_connection(self, x_2d_sq):
        # output should not be all zeros -- residual ensures input passes through
        block = NonLocalBlock()
        variables = block.init(KEY, x_2d_sq, train=False)
        out = block.apply(variables, x_2d_sq, train=False)
        assert not jnp.allclose(out, jnp.zeros_like(out))

    def test_dropout_train_stochastic(self, x_2d_sq):
        block = NonLocalBlock(dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY},
                               x_2d_sq, train=True)
        out1 = block.apply(variables, x_2d_sq, train=True,
                           rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = block.apply(variables, x_2d_sq, train=True,
                           rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)


# ---------------------------------------------------------------------------
# InceptionBlock
# ---------------------------------------------------------------------------

C_RED = {'3x3': 16, '5x5': 8}
C_OUT = {'1x1': 16, '3x3': 32, '5x5': 8, 'max': 8}
INCEPTION_OUT = sum(C_OUT.values())


class TestInceptionBlock:
    def test_forward(self, x_2d_sq):
        block = InceptionBlock(c_red=C_RED, c_out=C_OUT)
        _, out = apply_block(block, x_2d_sq)
        check_shape_finite(out, (BATCH, H, W, INCEPTION_OUT))

    def test_output_channels_sum(self, x_2d_sq):
        block = InceptionBlock(c_red=C_RED, c_out=C_OUT)
        variables = block.init(KEY, x_2d_sq, train=False)
        out = block.apply(variables, x_2d_sq, train=False)
        assert out.shape[-1] == INCEPTION_OUT

    def test_dropout_eval_deterministic(self, x_2d_sq):
        block = InceptionBlock(c_red=C_RED, c_out=C_OUT, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY},
                               x_2d_sq, train=True)
        out1 = block.apply(variables, x_2d_sq, train=False)
        out2 = block.apply(variables, x_2d_sq, train=False)
        assert jnp.allclose(out1, out2)

    def test_dropout_train_stochastic(self, x_2d_sq):
        block = InceptionBlock(c_red=C_RED, c_out=C_OUT, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY},
                               x_2d_sq, train=True)
        out1 = block.apply(variables, x_2d_sq, train=True,
                           rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = block.apply(variables, x_2d_sq, train=True,
                           rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)

    def test_backward(self, x_2d_sq):
        check_backward(InceptionBlock(c_red=C_RED, c_out=C_OUT), x_2d_sq)

    def test_norm_variants(self, x_2d_sq):
        for norm in ["GROUP_NORM", "LAYER_NORM"]:
            block = InceptionBlock(c_red=C_RED, c_out=C_OUT, norm=norm)
            _, out = apply_block(block, x_2d_sq)
            check_shape_finite(out, (BATCH, H, W, INCEPTION_OUT))


# ---------------------------------------------------------------------------
# DenseNet blocks
# ---------------------------------------------------------------------------

class TestDenseLayer:
    def test_forward(self, x_2d_sq):
        layer = DenseLayer(growth_rate=16)
        _, out = apply_block(layer, x_2d_sq)
        check_shape_finite(out, (BATCH, H, W, FEATS + 16))

    def test_channel_growth(self, x_2d_sq):
        layer = DenseLayer(growth_rate=8)
        variables = layer.init(KEY, x_2d_sq, train=False)
        out = layer.apply(variables, x_2d_sq, train=False)
        assert out.shape[-1] == FEATS + 8

    def test_dropout_eval_deterministic(self, x_2d_sq):
        layer = DenseLayer(growth_rate=16, dropout_rate=0.5)
        variables = layer.init({'params': KEY, 'dropout': DROP_KEY},
                               x_2d_sq, train=True)
        out1 = layer.apply(variables, x_2d_sq, train=False)
        out2 = layer.apply(variables, x_2d_sq, train=False)
        assert jnp.allclose(out1, out2)

    def test_backward(self, x_2d_sq):
        check_backward(DenseLayer(growth_rate=16), x_2d_sq)



class TestDenseBlock:
    def test_forward(self, x_2d_sq):
        block = DenseBlock(num_layers=4, growth_rate=8)
        _, out = apply_block(block, x_2d_sq)
        check_shape_finite(out, (BATCH, H, W, FEATS + 4 * 8))

    def test_channel_growth_formula(self, x_2d_sq):
        # uses LAYER_NORM to isolate shape arithmetic from GroupNorm divisibility
        n, g = 6, 4
        block = DenseBlock(num_layers=n, growth_rate=g, norm='LAYER_NORM')
        variables = block.init(KEY, x_2d_sq, train=False)
        out = block.apply(variables, x_2d_sq, train=False)
        assert out.shape[-1] == FEATS + n * g

    def test_groupnorm_growth_rate_indivisible_raises(self):
        # growth_rate=4 is not divisible by default num_groups=8
        with pytest.raises(ValueError, match="growth_rate"):
            DenseBlock(num_layers=6, growth_rate=4, norm='GROUP_NORM')

    def test_groupnorm_input_channels_indivisible_raises(self):
        # valid growth_rate but C_in not divisible by num_groups
        x_bad = jax.random.normal(KEY, (BATCH, H, W, 12))  # 12 % 8 != 0
        block = DenseBlock(num_layers=4, growth_rate=8, norm='GROUP_NORM')
        with pytest.raises(ValueError, match="input channels"):
            block.init(KEY, x_bad, train=False)

    def test_dropout_propagated(self, x_2d_sq):
        block = DenseBlock(num_layers=4, growth_rate=8, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY},
                               x_2d_sq, train=True)
        out1 = block.apply(variables, x_2d_sq, train=True,
                           rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = block.apply(variables, x_2d_sq, train=True,
                           rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)

    def test_backward(self, x_2d_sq):
        check_backward(DenseBlock(num_layers=4, growth_rate=8), x_2d_sq)


class TestTransitionLayer:
    def test_forward_avg(self, x_2d_sq):
        layer = TransitionLayer(features=FEATS // 2)
        _, out = apply_block(layer, x_2d_sq)
        check_shape_finite(out, (BATCH, H // 2, W // 2, FEATS // 2))

    def test_forward_max(self, x_2d_sq):
        layer = TransitionLayer(features=FEATS // 2, pool_type='SPATIAL_MAX')
        _, out = apply_block(layer, x_2d_sq)
        check_shape_finite(out, (BATCH, H // 2, W // 2, FEATS // 2))

    def test_backward(self, x_2d_sq):
        check_backward(TransitionLayer(features=FEATS // 2), x_2d_sq)


# ---------------------------------------------------------------------------
# 1D Blocks
# ---------------------------------------------------------------------------

class TestConvBlock1d:
    def test_forward(self, x_1d):
        block = ConvBlock1d(features=FEATS)
        _, out = apply_block(block, x_1d)
        check_shape_finite(out, (BATCH, SEQ_LEN, FEATS))

    def test_pre_norm(self, x_1d):
        block = ConvBlock1d(features=FEATS, pre_norm=True)
        _, out = apply_block(block, x_1d)
        check_shape_finite(out, (BATCH, SEQ_LEN, FEATS))

    def test_dropout_eval_deterministic(self, x_1d):
        block = ConvBlock1d(features=FEATS, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY}, x_1d, train=True)
        out1 = block.apply(variables, x_1d, train=False)
        out2 = block.apply(variables, x_1d, train=False)
        assert jnp.allclose(out1, out2)

    def test_dropout_train_stochastic(self, x_1d):
        block = ConvBlock1d(features=FEATS, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY}, x_1d, train=True)
        out1 = block.apply(variables, x_1d, train=True,
                           rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = block.apply(variables, x_1d, train=True,
                           rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)

    def test_backward(self, x_1d):
        check_backward(ConvBlock1d(features=FEATS), x_1d)

    def test_norm_variants(self, x_1d):
        for norm in ["LAYER_NORM", "GROUP_NORM"]:
            block = ConvBlock1d(features=FEATS, norm=norm,
                                norm_kwargs={'num_groups': 4} if norm == "GROUP_NORM" else {})
            _, out = apply_block(block, x_1d)
            check_shape_finite(out, (BATCH, SEQ_LEN, FEATS))


class TestResidualBlock1d:
    def test_forward_same_channels(self, x_1d):
        x = jax.random.normal(KEY, (BATCH, SEQ_LEN, FEATS))
        block = ResidualBlock1d(features=FEATS)
        _, out = apply_block(block, x)
        check_shape_finite(out, (BATCH, SEQ_LEN, FEATS))

    def test_forward_channel_mismatch(self, x_1d):
        block = ResidualBlock1d(features=FEATS)
        _, out = apply_block(block, x_1d)
        check_shape_finite(out, (BATCH, SEQ_LEN, FEATS))

    def test_pre_norm(self, x_1d):
        block = ResidualBlock1d(features=FEATS, pre_norm=True)
        _, out = apply_block(block, x_1d)
        check_shape_finite(out, (BATCH, SEQ_LEN, FEATS))

    def test_dropout_eval_deterministic(self, x_1d):
        block = ResidualBlock1d(features=FEATS, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY}, x_1d, train=True)
        out1 = block.apply(variables, x_1d, train=False)
        out2 = block.apply(variables, x_1d, train=False)
        assert jnp.allclose(out1, out2)

    def test_backward(self, x_1d):
        check_backward(ResidualBlock1d(features=FEATS), x_1d)


class TestDownsampleBlock1d:
    def test_forward(self, x_1d):
        block = DownsampleBlock1d(features=FEATS)
        _, out = apply_block(block, x_1d)
        check_shape_finite(out, (BATCH, SEQ_LEN // 2, FEATS))

    def test_backward(self, x_1d):
        check_backward(DownsampleBlock1d(features=FEATS), x_1d)


class TestUpsampleBlock1d:
    def test_forward(self, x_1d):
        block = UpsampleBlock1d(features=FEATS)
        _, out = apply_block(block, x_1d)
        check_shape_finite(out, (BATCH, SEQ_LEN * 2, FEATS))

    def test_roundtrip_shape(self, x_1d):
        down = DownsampleBlock1d(features=FEATS)
        up = UpsampleBlock1d(features=C_IN)
        v_down = down.init(KEY, x_1d)
        x_small = down.apply(v_down, x_1d)
        v_up = up.init(KEY, x_small)
        x_up = up.apply(v_up, x_small)
        assert x_up.shape == (BATCH, SEQ_LEN, C_IN)

    def test_backward(self, x_1d):
        check_backward(UpsampleBlock1d(features=FEATS), x_1d)


# ---------------------------------------------------------------------------
# ConvEncoder
# ---------------------------------------------------------------------------

class TestConvEncoder:
    @pytest.fixture
    def enc(self):
        return ConvEncoder(channels=(16, 32, 64), num_res_blocks=1)

    def test_forward(self, enc, x_2d):
        variables = enc.init(KEY, x_2d, train=True)
        out = enc.apply(variables, x_2d, train=False)
        check_shape_finite(out, (BATCH, H // 4, W // 4, 64))

    def test_forward_single_level(self, x_2d):
        enc = ConvEncoder(channels=(32,), num_res_blocks=1)
        variables = enc.init(KEY, x_2d, train=True)
        out = enc.apply(variables, x_2d, train=False)
        check_shape_finite(out, (BATCH, H, W, 32))

    def test_with_non_local(self, x_2d):
        enc = ConvEncoder(channels=(16, 32), num_res_blocks=1,
                          use_non_local=True)
        variables = enc.init(KEY, x_2d, train=True)
        out = enc.apply(variables, x_2d, train=False)
        check_shape_finite(out, (BATCH, H // 2, W // 2, 32))

    def test_same_padding_mode(self, x_2d):
        enc = ConvEncoder(channels=(16, 32), num_res_blocks=1,
                          downsample_padding='same')
        variables = enc.init(KEY, x_2d, train=True)
        out = enc.apply(variables, x_2d, train=False)
        check_shape_finite(out, (BATCH, H // 2, W // 2, 32))

    def test_pool_downsample(self, x_2d):
        enc = ConvEncoder(channels=(16, 32), num_res_blocks=1,
                          downsample_pool_type='SPATIAL_AVG')
        variables = enc.init(KEY, x_2d, train=True)
        out = enc.apply(variables, x_2d, train=False)
        check_shape_finite(out, (BATCH, H // 2, W // 2, 32))

    def test_dropout_eval_deterministic(self, x_2d):
        enc = ConvEncoder(channels=(16, 32), num_res_blocks=1, dropout_rate=0.5)
        variables = enc.init({'params': KEY, 'dropout': DROP_KEY}, x_2d, train=True)
        out1 = enc.apply(variables, x_2d, train=False)
        out2 = enc.apply(variables, x_2d, train=False)
        assert jnp.allclose(out1, out2)

    def test_dropout_train_stochastic(self, x_2d):
        enc = ConvEncoder(channels=(16, 32), num_res_blocks=1, dropout_rate=0.5)
        variables = enc.init({'params': KEY, 'dropout': DROP_KEY}, x_2d, train=True)
        out1 = enc.apply(variables, x_2d, train=True,
                         rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = enc.apply(variables, x_2d, train=True,
                         rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)

    def test_backward(self, x_2d):
        enc = ConvEncoder(channels=(16, 32), num_res_blocks=1)
        check_backward(enc, x_2d)

    def test_pre_norm(self, x_2d):
        enc = ConvEncoder(channels=(16, 32), num_res_blocks=1, pre_norm=True)
        variables = enc.init(KEY, x_2d, train=True)
        out = enc.apply(variables, x_2d, train=False)
        check_shape_finite(out, (BATCH, H // 2, W // 2, 32))


# ---------------------------------------------------------------------------
# ConvDecoder
# ---------------------------------------------------------------------------

class TestConvDecoder:
    def test_forward(self):
        x = jax.random.normal(KEY, (BATCH, 8, 8, 64))
        dec = ConvDecoder(channels=(64, 32, 16), out_features=3, num_res_blocks=1)
        variables = dec.init(KEY, x, train=True)
        out = dec.apply(variables, x, train=False)
        check_shape_finite(out, (BATCH, 32, 32, 3))

    def test_no_out_features(self):
        x = jax.random.normal(KEY, (BATCH, 8, 8, 64))
        dec = ConvDecoder(channels=(64, 32), num_res_blocks=1)
        variables = dec.init(KEY, x, train=True)
        out = dec.apply(variables, x, train=False)
        check_shape_finite(out, (BATCH, 16, 16, 32))

    def test_with_non_local(self):
        x = jax.random.normal(KEY, (BATCH, 8, 8, 64))
        dec = ConvDecoder(channels=(64, 32), num_res_blocks=1, use_non_local=True)
        variables = dec.init(KEY, x, train=True)
        out = dec.apply(variables, x, train=False)
        check_shape_finite(out, (BATCH, 16, 16, 32))

    def test_encoder_decoder_roundtrip(self):
        x = jax.random.normal(KEY, (BATCH, 32, 32, 3))
        enc = ConvEncoder(channels=(16, 32), num_res_blocks=1)
        dec = ConvDecoder(channels=(32, 16), out_features=3, num_res_blocks=1)
        v_enc = enc.init(KEY, x, train=True)
        z = enc.apply(v_enc, x, train=False)
        v_dec = dec.init(KEY, z, train=True)
        out = dec.apply(v_dec, z, train=False)
        assert out.shape == x.shape
        assert jnp.all(jnp.isfinite(out))

    def test_dropout_eval_deterministic(self):
        x = jax.random.normal(KEY, (BATCH, 8, 8, 64))
        dec = ConvDecoder(channels=(64, 32), num_res_blocks=1, dropout_rate=0.5)
        variables = dec.init({'params': KEY, 'dropout': DROP_KEY}, x, train=True)
        out1 = dec.apply(variables, x, train=False)
        out2 = dec.apply(variables, x, train=False)
        assert jnp.allclose(out1, out2)

    def test_backward(self):
        x = jax.random.normal(KEY, (BATCH, 8, 8, 32))
        check_backward(ConvDecoder(channels=(32, 16), num_res_blocks=1), x)


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------

class TestResNet:
    @pytest.fixture
    def net(self):
        return ResNet(num_classes=CLASSES, c_hidden=(16, 32), num_blocks=(2, 2))

    def test_forward(self, net, x_2d):
        variables = net.init(KEY, x_2d, train=True)
        out = net.apply(variables, x_2d, train=False)
        check_shape_finite(out, (BATCH, CLASSES))

    def test_pre_norm(self, x_2d):
        net = ResNet(num_classes=CLASSES, c_hidden=(16, 32),
                     num_blocks=(2, 2), pre_norm=True)
        variables = net.init(KEY, x_2d, train=True)
        out = net.apply(variables, x_2d, train=False)
        check_shape_finite(out, (BATCH, CLASSES))

    def test_dropout_eval_deterministic(self, x_2d):
        net = ResNet(num_classes=CLASSES, c_hidden=(16, 32),
                     num_blocks=(2, 2), dropout_rate=0.5)
        variables = net.init({'params': KEY, 'dropout': DROP_KEY}, x_2d, train=True)
        out1 = net.apply(variables, x_2d, train=False)
        out2 = net.apply(variables, x_2d, train=False)
        assert jnp.allclose(out1, out2)

    def test_dropout_train_stochastic(self, x_2d):
        net = ResNet(num_classes=CLASSES, c_hidden=(16, 32),
                     num_blocks=(2, 2), dropout_rate=0.5)
        variables = net.init({'params': KEY, 'dropout': DROP_KEY}, x_2d, train=True)
        out1 = net.apply(variables, x_2d, train=True,
                         rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = net.apply(variables, x_2d, train=True,
                         rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)

    def test_backward(self, x_2d):
        net = ResNet(num_classes=CLASSES, c_hidden=(16, 32), num_blocks=(2, 2))
        check_backward(net, x_2d)

    def test_output_changes_after_grad_step(self, x_2d):
        net = ResNet(num_classes=CLASSES, c_hidden=(16, 32), num_blocks=(2, 2))
        variables = net.init(KEY, x_2d, train=True)

        def loss_fn(params):
            return jnp.mean(net.apply({'params': params}, x_2d, train=False) ** 2)

        grads = jax.grad(loss_fn)(variables['params'])
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(variables['params'])
        updates, _ = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(variables['params'], updates)
        out_before = net.apply(variables, x_2d, train=False)
        out_after = net.apply({'params': new_params}, x_2d, train=False)
        assert not jnp.allclose(out_before, out_after)

    def test_norm_variants(self, x_2d):
        for norm in ["GROUP_NORM", "LAYER_NORM", "INSTANCE_NORM"]:
            net = ResNet(num_classes=CLASSES, c_hidden=(16, 32),
                         num_blocks=(2, 2), norm=norm)
            variables = net.init(KEY, x_2d, train=True)
            out = net.apply(variables, x_2d, train=False)
            check_shape_finite(out, (BATCH, CLASSES))

    def test_activation_variants(self, x_2d):
        for act in ["relu", "silu", "gelu", "tanh"]:
            net = ResNet(num_classes=CLASSES, c_hidden=(16, 32),
                         num_blocks=(2, 2), activation=act)
            variables = net.init(KEY, x_2d, train=True)
            out = net.apply(variables, x_2d, train=False)
            check_shape_finite(out, (BATCH, CLASSES))


# ---------------------------------------------------------------------------
# DenseNet
# ---------------------------------------------------------------------------

# GroupNorm-safe params for DenseNet tests:
# c_hidden = 16*4 = 64, after block: 64+3*16=112, transition: 56 ✓
# after block 1: 56+3*16=104 ✓
DN_GROWTH  = 16
DN_BNSIZE  = 4
DN_LAYERS  = (3, 3)


class TestDenseNet:
    @pytest.fixture
    def net(self):
        return DenseNet(num_classes=CLASSES, num_layers=DN_LAYERS,
                        growth_rate=DN_GROWTH, bn_size=DN_BNSIZE)

    def test_forward(self, net, x_2d):
        variables = net.init(KEY, x_2d, train=True)
        out = net.apply(variables, x_2d, train=False)
        check_shape_finite(out, (BATCH, CLASSES))

    def test_dropout_eval_deterministic(self, x_2d):
        net = DenseNet(num_classes=CLASSES, num_layers=DN_LAYERS,
                       growth_rate=DN_GROWTH, bn_size=DN_BNSIZE, dropout_rate=0.5)
        variables = net.init({'params': KEY, 'dropout': DROP_KEY}, x_2d, train=True)
        out1 = net.apply(variables, x_2d, train=False)
        out2 = net.apply(variables, x_2d, train=False)
        assert jnp.allclose(out1, out2)

    def test_dropout_train_stochastic(self, x_2d):
        net = DenseNet(num_classes=CLASSES, num_layers=DN_LAYERS,
                       growth_rate=DN_GROWTH, bn_size=DN_BNSIZE, dropout_rate=0.5)
        variables = net.init({'params': KEY, 'dropout': DROP_KEY}, x_2d, train=True)
        out1 = net.apply(variables, x_2d, train=True,
                         rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = net.apply(variables, x_2d, train=True,
                         rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)

    def test_transition_pool_type(self, x_2d):
        net = DenseNet(num_classes=CLASSES, num_layers=DN_LAYERS,
                       growth_rate=DN_GROWTH, bn_size=DN_BNSIZE,
                       transition_pool_type='SPATIAL_MAX')
        variables = net.init(KEY, x_2d, train=True)
        out = net.apply(variables, x_2d, train=False)
        check_shape_finite(out, (BATCH, CLASSES))

    def test_norm_variants(self, x_2d):
        for norm in ["GROUP_NORM", "LAYER_NORM", "INSTANCE_NORM"]:
            net = DenseNet(num_classes=CLASSES, num_layers=DN_LAYERS,
                           growth_rate=DN_GROWTH, bn_size=DN_BNSIZE, norm=norm)
            variables = net.init(KEY, x_2d, train=True)
            out = net.apply(variables, x_2d, train=False)
            check_shape_finite(out, (BATCH, CLASSES))

    def test_activation_variants(self, x_2d):
        for act in ["relu", "silu", "gelu", "tanh"]:
            net = DenseNet(num_classes=CLASSES, num_layers=DN_LAYERS,
                           growth_rate=DN_GROWTH, bn_size=DN_BNSIZE, activation=act)
            variables = net.init(KEY, x_2d, train=True)
            out = net.apply(variables, x_2d, train=False)
            check_shape_finite(out, (BATCH, CLASSES))

    def test_backward(self, x_2d):
        net = DenseNet(num_classes=CLASSES, num_layers=DN_LAYERS,
                       growth_rate=DN_GROWTH, bn_size=DN_BNSIZE)
        check_backward(net, x_2d)

    def test_output_changes_after_grad_step(self, x_2d):
        net = DenseNet(num_classes=CLASSES, num_layers=DN_LAYERS,
                       growth_rate=DN_GROWTH, bn_size=DN_BNSIZE)
        variables = net.init(KEY, x_2d, train=True)

        def loss_fn(params):
            return jnp.mean(net.apply({'params': params}, x_2d, train=False) ** 2)

        grads = jax.grad(loss_fn)(variables['params'])
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(variables['params'])
        updates, _ = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(variables['params'], updates)
        out_before = net.apply(variables, x_2d, train=False)
        out_after = net.apply({'params': new_params}, x_2d, train=False)
        assert not jnp.allclose(out_before, out_after)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_conv_nets_returns_all(self):
        nets = list_conv_nets()
        expected = {"CONV_ENCODER", "CONV_DECODER", "RESNET", "DENSENET"}
        assert expected == set(nets.keys())

    def test_get_conv_net_instantiates(self, x_2d):
        net = get_conv_net("RESNET", num_classes=CLASSES,
                           c_hidden=(16, 32), num_blocks=(2, 2))
        variables = net.init(KEY, x_2d, train=True)
        out = net.apply(variables, x_2d, train=False)
        assert out.shape == (BATCH, CLASSES)

    def test_get_conv_net_unknown_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            get_conv_net("NONEXISTENT")

    def test_get_conv_net_unknown_kwargs_warns(self):
        with pytest.warns(UserWarning, match="unknown kwargs"):
            get_conv_net("RESNET", num_classes=CLASSES,
                         c_hidden=(16,), num_blocks=(2,),
                         bogus_param=99)

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            @register_conv_net("RESNET", description="duplicate")
            class Duplicate(nn.Module):
                pass

    def test_get_conv_net_valid_kwargs_applied(self):
        net = get_conv_net("RESNET", num_classes=5,
                           c_hidden=(16, 32), num_blocks=(2, 2))
        assert net.num_classes == 5

    def test_all_registered_nets_forward(self, x_2d):
        x_enc = jax.random.normal(KEY, (BATCH, 8, 8, 32))
        for name in list_conv_nets():
            if name == "CONV_ENCODER":
                net = get_conv_net(name, channels=(16, 32), num_res_blocks=1)
                variables = net.init(KEY, x_2d, train=True)
                out = net.apply(variables, x_2d, train=False)
                assert jnp.all(jnp.isfinite(out))
            elif name == "CONV_DECODER":
                net = get_conv_net(name, channels=(32, 16), num_res_blocks=1)
                variables = net.init(KEY, x_enc, train=True)
                out = net.apply(variables, x_enc, train=False)
                assert jnp.all(jnp.isfinite(out))
            elif name == "RESNET":
                net = get_conv_net(name, num_classes=CLASSES,
                                   c_hidden=(16, 32), num_blocks=(2, 2))
                variables = net.init(KEY, x_2d, train=True)
                out = net.apply(variables, x_2d, train=False)
                assert out.shape == (BATCH, CLASSES)
            elif name == "DENSENET":
                net = get_conv_net(name, num_classes=CLASSES,
                                   num_layers=DN_LAYERS, growth_rate=DN_GROWTH, bn_size=DN_BNSIZE)
                variables = net.init(KEY, x_2d, train=True)
                out = net.apply(variables, x_2d, train=False)
                assert out.shape == (BATCH, CLASSES)

# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

class TestPatchEmbed:

    # --- Shape ---

    def test_output_shape(self):
        x     = jnp.ones((BATCH, IMG_H, IMG_W, C_IN))
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED)
        variables = embed.init(KEY, x)
        out = embed.apply(variables, x)
        num_patches = (IMG_H // PATCH) * (IMG_W // PATCH)
        check_shape_finite(out, (BATCH, num_patches, PEMBED))

    def test_num_patches_formula(self):
        for p in [2, 4, 8]:
            x     = jnp.ones((BATCH, 32, 32, C_IN))
            embed = PatchEmbed(patch_size=p, embed_dim=PEMBED)
            variables = embed.init(KEY, x)
            out = embed.apply(variables, x)
            expected_tokens = (32 // p) ** 2
            assert out.shape == (BATCH, expected_tokens, PEMBED), \
                f"patch_size={p}: expected {expected_tokens} tokens"

    def test_flatten_false_shape(self):
        x = jnp.ones((BATCH, IMG_H, IMG_W, C_IN))
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED, flatten=False)
        variables = embed.init(KEY, x)
        out = embed.apply(variables, x)
        check_shape_finite(out, (BATCH, IMG_H // PATCH, IMG_W // PATCH, PEMBED))

    def test_flatten_true_matches_flatten_false_reshape(self):
        # flatten=True output should equal flatten=False reshaped to sequence
        x = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, C_IN))
        e_flat = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED, flatten=True)
        e_grid = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED, flatten=False)
        # share same params -- init with flat version
        variables = e_flat.init(KEY, x)
        out_flat = e_flat.apply(variables, x)
        out_grid = e_grid.apply(variables, x)
        nH = IMG_H // PATCH
        nW = IMG_W // PATCH
        # grid reshaped to sequence should match flat output exactly
        assert jnp.allclose(
            out_flat,
            out_grid.reshape(BATCH, nH * nW, PEMBED),
            atol=1e-6,
        ), "flatten=True and flatten=False should produce identical values"

    def test_non_square_image(self):
        x     = jnp.ones((BATCH, 32, 64, C_IN))
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED)
        variables = embed.init(KEY, x)
        out = embed.apply(variables, x)
        check_shape_finite(out, (BATCH, (32 // PATCH) * (64 // PATCH), PEMBED))

    def test_multichannel_input(self):
        x     = jnp.ones((BATCH, IMG_H, IMG_W, 16))
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED)
        variables = embed.init(KEY, x)
        out = embed.apply(variables, x)
        num_patches = (IMG_H // PATCH) * (IMG_W // PATCH)
        check_shape_finite(out, (BATCH, num_patches, PEMBED))

    # --- Parameters ---

    def test_has_trainable_params(self):
        x     = jnp.ones((BATCH, IMG_H, IMG_W, C_IN))
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED)
        variables = embed.init(KEY, x)
        assert 'params' in variables
        assert 'proj' in variables['params']

    def test_use_bias_false(self):
        x     = jnp.ones((BATCH, IMG_H, IMG_W, C_IN))
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED, use_bias=False)
        variables = embed.init(KEY, x)
        assert 'bias' not in variables['params']['proj']

    # --- Values ---

    def test_finite_output(self):
        x     = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, C_IN))
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED)
        variables = embed.init(KEY, x)
        out = embed.apply(variables, x)
        assert jnp.all(jnp.isfinite(out))

    def test_linearity(self):
        # PatchEmbed with no bias is a linear map: f(2x) == 2*f(x)
        x     = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, C_IN))
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED, use_bias=False)
        variables = embed.init(KEY, x)
        out1 = embed.apply(variables, x)
        out2 = embed.apply(variables, 2.0 * x)
        assert jnp.allclose(2.0 * out1, out2, atol=1e-5), \
            "PatchEmbed with no bias should be a linear map"

    def test_no_norm_or_activation(self):
        # with large input values, output should be large too (not clipped)
        x     = jnp.ones((BATCH, IMG_H, IMG_W, C_IN)) * 100.0
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED)
        variables = embed.init(KEY, x)
        out = embed.apply(variables, x)
        assert jnp.abs(out).max() > 1.0, \
            "PatchEmbed should not clip or normalise -- pure linear projection"

    # --- Validation ---

    def test_invalid_spatial_dims_raises(self):
        x     = jnp.ones((BATCH, 30, 30, C_IN))  # 30 % 4 != 0
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED)
        with pytest.raises(ValueError, match="divisible"):
            embed.init(KEY, x)

    def test_wrong_rank_raises(self):
        x     = jnp.ones((BATCH, IMG_H, IMG_W))  # 3D not 4D
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED)
        with pytest.raises(AssertionError, match="4D"):
            embed.init(KEY, x)

    # --- Backward ---

    def test_backward(self):
        x     = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, C_IN))
        embed = PatchEmbed(patch_size=PATCH, embed_dim=PEMBED)
        variables = embed.init(KEY, x)

        def loss(params):
            out = embed.apply({'params': params}, x)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf)), "Gradient contains non-finite values"

    # --- Registry ---

    def test_not_in_conv_net_registry(self):
        # PatchEmbed is a block, not a registered net
        nets = list_conv_nets()
        assert 'PATCH_EMBED' not in nets