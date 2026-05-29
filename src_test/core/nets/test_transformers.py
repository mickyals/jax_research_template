# src_test/core/nets/test_transformer.py
import pytest
import jax
import jax.numpy as jnp
import optax

from core.nets.transformers import (
    # blocks
    TransformerBlock,
    CrossAttentionBlock,
    SwinBlock,
    SwinBlockPair,
    PatchMerging,
    # registered nets
    TransformerEncoder,
    TransformerDecoder,
    ViT,
    MaskedViT,
    MAEDecoder,
    ConvMAEDecoder,
    SwinEncoder,
    # registry
    get_transformer,
    list_transformers,
    register_transformer,
)
from core.attention import make_causal_mask, make_padding_mask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY      = jax.random.PRNGKey(0)
DROP_KEY = jax.random.PRNGKey(1)
MASK_KEY = jax.random.PRNGKey(2)

BATCH      = 2
SEQ        = 16
EMBED      = 64
NUM_HEADS  = 4
MLP_RATIO  = 2.0   # small for fast tests

# Swin constants -- H, W divisible by patch_size * window_size
SWIN_H     = 28
SWIN_W     = 28
SWIN_C     = 32
SWIN_HEADS = 4
WINDOW     = 7

# ViT / image constants
IMG_H      = 32
IMG_W      = 32
IMG_C      = 3
PATCH      = 4
VIT_EMBED  = 32
VIT_HEADS  = 4
VIT_LAYERS = 2
NUM_PATCHES = (IMG_H // PATCH) * (IMG_W // PATCH)   # 64

CLASSES    = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_shape(x, expected, label='output'):
    assert x.shape == expected, f"{label}: expected {expected}, got {x.shape}"


def check_finite(x, label='output'):
    assert jnp.all(jnp.isfinite(x)), f"{label} contains non-finite values"


def check_backward(module, *args, with_dropout=False, with_mask_rng=False,
                   init_rngs=None, apply_rngs=None, **kwargs):
    """Generic backward pass checker."""
    if init_rngs is None:
        init_rngs = {'params': KEY}
        if with_dropout:
            init_rngs['dropout'] = DROP_KEY
        if with_mask_rng:
            init_rngs['mask'] = MASK_KEY

    variables = module.init(init_rngs, *args, train=True, **kwargs)

    run_rngs = {}
    if with_dropout:
        run_rngs['dropout'] = DROP_KEY
    if with_mask_rng:
        run_rngs['mask'] = MASK_KEY

    def loss_fn(params):
        out = module.apply({'params': params}, *args, train=False,
                           rngs=run_rngs, **kwargs)
        # handle tuple outputs (MaskedViT)
        if isinstance(out, tuple):
            out = out[0]
        return jnp.mean(out ** 2)

    grads = jax.grad(loss_fn)(variables['params'])
    for leaf in jax.tree_util.tree_leaves(grads):
        check_finite(leaf, 'gradient')


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:

    # --- Shape ---

    def test_forward_shape(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, train=False)
        out = block.apply(variables, x, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_output_same_shape_as_input(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, train=False)
        out = block.apply(variables, x, train=False)
        assert out.shape == x.shape

    # --- Causal ---

    def test_causal_flag(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO, causal=True)
        variables = block.init(KEY, x, train=False)
        out = block.apply(variables, x, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_causal_no_future_influence(self):
        x_full = jax.random.normal(KEY, (1, SEQ, EMBED))
        x_one  = x_full[:, :1, :]
        block  = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                  mlp_ratio=MLP_RATIO, causal=True)
        v = block.init(KEY, x_full, train=False)
        out_full = block.apply(v, x_full, train=False)
        out_one  = block.apply(v, x_one,  train=False)
        assert jnp.allclose(out_full[:, 0, :], out_one[:, 0, :], atol=1e-5), \
            "Causal block: position 0 should not be affected by future tokens"

    # --- Mask ---

    def test_with_padding_mask(self):
        x       = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        lengths = jnp.array([SEQ, SEQ // 2])
        mask    = make_padding_mask(lengths, SEQ)[:, None, :]
        block   = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                   mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, train=False)
        out = block.apply(variables, x, mask=mask, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    # --- return_weights ---

    def test_return_weights_shape(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, train=False)
        out, w = block.apply(variables, x, train=False, return_weights=True)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_shape(w, (BATCH, NUM_HEADS, SEQ, SEQ), 'weights')
        check_finite(w, 'weights')

    def test_return_weights_sum_to_one(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, train=False)
        _, w = block.apply(variables, x, train=False, return_weights=True)
        row_sums = w.sum(axis=-1)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5)

    # --- Dropout ---

    def test_dropout_eval_deterministic(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY},
                               x, train=True)
        out1 = block.apply(variables, x, train=False)
        out2 = block.apply(variables, x, train=False)
        assert jnp.allclose(out1, out2)

    def test_dropout_train_stochastic(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY},
                               x, train=True)
        out1 = block.apply(variables, x, train=True,
                           rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = block.apply(variables, x, train=True,
                           rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)

    # --- Permutation equivariance ---

    def test_permutation_equivariant(self):
        x     = jax.random.normal(KEY, (1, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, train=False)
        perm     = jax.random.permutation(KEY, SEQ)
        out      = block.apply(variables, x, train=False)
        out_perm = block.apply(variables, x[:, perm, :], train=False)
        assert jnp.allclose(out[:, perm, :], out_perm, atol=1e-5), \
            "TransformerBlock should be permutation-equivariant"

    # --- Backward ---

    def test_backward(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO)
        check_backward(block, x)

    def test_backward_causal(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO, causal=True)
        check_backward(block, x)

    def test_backward_with_dropout(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO, dropout_rate=0.1)
        check_backward(block, x, with_dropout=True)

    # --- Submodule accessibility ---

    def test_submodules_accessible(self):
        x     = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = TransformerBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                 mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, train=False)
        bound = block.bind(variables)
        assert hasattr(bound, 'norm1')
        assert hasattr(bound, 'norm2')
        assert hasattr(bound, 'attn')
        assert hasattr(bound, 'ffn')


# ---------------------------------------------------------------------------
# CrossAttentionBlock
# ---------------------------------------------------------------------------

class TestCrossAttentionBlock:

    def test_forward_shape(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ * 2,  EMBED))
        block = CrossAttentionBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                    mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, ctx, train=False)
        out = block.apply(variables, x, ctx, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_output_depends_on_context(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctx1 = jax.random.normal(jax.random.PRNGKey(1), (BATCH, SEQ, EMBED))
        ctx2 = jax.random.normal(jax.random.PRNGKey(2), (BATCH, SEQ, EMBED))
        block = CrossAttentionBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                    mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, ctx1, train=False)
        out1 = block.apply(variables, x, ctx1, train=False)
        out2 = block.apply(variables, x, ctx2, train=False)
        assert not jnp.allclose(out1, out2)

    def test_return_weights_shape(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ * 2,  EMBED))
        block = CrossAttentionBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                    mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, ctx, train=False)
        out, w = block.apply(variables, x, ctx, train=False,
                             return_weights=True)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_shape(w, (BATCH, NUM_HEADS, SEQ, SEQ * 2), 'cross weights')

    def test_separate_query_context_norms(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = CrossAttentionBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                    mlp_ratio=MLP_RATIO)
        variables = block.init(KEY, x, ctx, train=False)
        bound = block.bind(variables)
        assert hasattr(bound, 'norm_q')
        assert hasattr(bound, 'norm_kv')
        assert hasattr(bound, 'norm_ff')

    def test_backward(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ * 2,  EMBED))
        block = CrossAttentionBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                    mlp_ratio=MLP_RATIO)
        check_backward(block, x, ctx)

    def test_dropout_eval_deterministic(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        block = CrossAttentionBlock(embed_dim=EMBED, num_heads=NUM_HEADS,
                                    mlp_ratio=MLP_RATIO, dropout_rate=0.5)
        variables = block.init({'params': KEY, 'dropout': DROP_KEY},
                               x, ctx, train=True)
        out1 = block.apply(variables, x, ctx, train=False)
        out2 = block.apply(variables, x, ctx, train=False)
        assert jnp.allclose(out1, out2)


# ---------------------------------------------------------------------------
# SwinBlock
# ---------------------------------------------------------------------------

class TestSwinBlock:

    def test_wmsa_forward_shape(self):
        x     = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        block = SwinBlock(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                          window_size=WINDOW, shift_size=0)
        variables = block.init(KEY, x, train=False)
        out = block.apply(variables, x, train=False)
        check_shape(out, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        check_finite(out)

    def test_swmsa_forward_shape(self):
        x     = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        block = SwinBlock(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                          window_size=WINDOW, shift_size=WINDOW // 2)
        variables = block.init(KEY, x, train=False)
        out = block.apply(variables, x, train=False)
        check_shape(out, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        check_finite(out)

    def test_return_weights_shape(self):
        x     = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        block = SwinBlock(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                          window_size=WINDOW)
        variables = block.init(KEY, x, train=False)
        out, w = block.apply(variables, x, train=False, return_weights=True)
        nW = (SWIN_H // WINDOW) * (SWIN_W // WINDOW)
        check_shape(w, (BATCH * nW, SWIN_HEADS, WINDOW**2, WINDOW**2),
                    'swin weights')

    def test_backward_wmsa(self):
        x     = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        block = SwinBlock(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                          window_size=WINDOW, shift_size=0)
        check_backward(block, x)

    def test_backward_swmsa(self):
        x     = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        block = SwinBlock(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                          window_size=WINDOW, shift_size=WINDOW // 2)
        check_backward(block, x)

    def test_submodules_accessible(self):
        x     = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        block = SwinBlock(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                          window_size=WINDOW)
        variables = block.init(KEY, x, train=False)
        bound = block.bind(variables)
        assert hasattr(bound, 'norm1')
        assert hasattr(bound, 'norm2')
        assert hasattr(bound, 'attn')
        assert hasattr(bound, 'ffn')


# ---------------------------------------------------------------------------
# SwinBlockPair
# ---------------------------------------------------------------------------

class TestSwinBlockPair:

    def test_forward_shape(self):
        x    = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        pair = SwinBlockPair(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                             window_size=WINDOW)
        variables = pair.init(KEY, x, train=False)
        out = pair.apply(variables, x, train=False)
        check_shape(out, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        check_finite(out)

    def test_output_differs_from_input(self):
        x    = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        pair = SwinBlockPair(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                             window_size=WINDOW)
        variables = pair.init(KEY, x, train=False)
        out = pair.apply(variables, x, train=False)
        assert not jnp.allclose(out, x)

    def test_has_both_blocks(self):
        x    = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        pair = SwinBlockPair(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                             window_size=WINDOW)
        variables = pair.init(KEY, x, train=False)
        bound = pair.bind(variables)
        assert hasattr(bound, 'block_w')
        assert hasattr(bound, 'block_sw')
        assert bound.block_w.shift_size  == 0
        assert bound.block_sw.shift_size == WINDOW // 2

    def test_backward(self):
        x    = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        pair = SwinBlockPair(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                             window_size=WINDOW)
        check_backward(pair, x)


# ---------------------------------------------------------------------------
# PatchMerging
# ---------------------------------------------------------------------------

class TestPatchMerging:

    def test_forward_shape(self):
        x     = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        merge = PatchMerging()
        variables = merge.init(KEY, x, train=False)
        out = merge.apply(variables, x, train=False)
        check_shape(out, (BATCH, SWIN_H // 2, SWIN_W // 2, 2 * SWIN_C))
        check_finite(out)

    def test_channel_doubling(self):
        for C in [16, 32, 64]:
            x     = jax.random.normal(KEY, (BATCH, 28, 28, C))
            merge = PatchMerging()
            variables = merge.init(KEY, x, train=False)
            out = merge.apply(variables, x, train=False)
            assert out.shape[-1] == 2 * C, \
                f"Expected {2*C} output channels, got {out.shape[-1]}"

    def test_spatial_halving(self):
        x     = jax.random.normal(KEY, (BATCH, 56, 56, SWIN_C))
        merge = PatchMerging()
        variables = merge.init(KEY, x, train=False)
        out = merge.apply(variables, x, train=False)
        assert out.shape[1:3] == (28, 28)

    def test_non_square_input(self):
        x     = jax.random.normal(KEY, (BATCH, 28, 56, SWIN_C))
        merge = PatchMerging()
        variables = merge.init(KEY, x, train=False)
        out = merge.apply(variables, x, train=False)
        check_shape(out, (BATCH, 14, 28, 2 * SWIN_C))

    def test_odd_dims_raises(self):
        x     = jax.random.normal(KEY, (BATCH, 27, 28, SWIN_C))
        merge = PatchMerging()
        with pytest.raises(ValueError, match="even"):
            merge.init(KEY, x, train=False)

    def test_backward(self):
        x     = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, SWIN_C))
        merge = PatchMerging()
        check_backward(merge, x)


# ---------------------------------------------------------------------------
# TransformerEncoder
# ---------------------------------------------------------------------------

class TestTransformerEncoder:

    def test_forward_shape(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        enc = TransformerEncoder(num_layers=2, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
        variables = enc.init(KEY, x, train=False)
        out = enc.apply(variables, x, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_with_pos_encoding(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        enc = TransformerEncoder(num_layers=2, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO,
                                 add_pos_encoding=True)
        variables = enc.init(KEY, x, train=False)
        out = enc.apply(variables, x, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))

    def test_without_pos_encoding(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        enc = TransformerEncoder(num_layers=2, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO,
                                 add_pos_encoding=False)
        variables = enc.init(KEY, x, train=False)
        out = enc.apply(variables, x, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))

    def test_permutation_equivariant_no_pos_enc(self):
        x   = jax.random.normal(KEY, (1, SEQ, EMBED))
        enc = TransformerEncoder(num_layers=2, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO,
                                 add_pos_encoding=False)
        variables = enc.init(KEY, x, train=False)
        perm     = jax.random.permutation(KEY, SEQ)
        out      = enc.apply(variables, x,           train=False)
        out_perm = enc.apply(variables, x[:, perm, :], train=False)
        assert jnp.allclose(out[:, perm, :], out_perm, atol=1e-5), \
            "Encoder without pos encoding should be permutation-equivariant"

    def test_causal_encoder(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        enc = TransformerEncoder(num_layers=2, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO,
                                 causal=True, add_pos_encoding=False)
        variables = enc.init(KEY, x, train=False)
        out = enc.apply(variables, x, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))

    def test_get_attention_maps_shape(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        enc = TransformerEncoder(num_layers=3, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO,
                                 add_pos_encoding=False)
        variables = enc.init(KEY, x, train=False)
        maps = enc.apply(variables, x, train=False,
                         method=enc.get_attention_maps)
        assert len(maps) == 3
        for w in maps:
            check_shape(w, (BATCH, NUM_HEADS, SEQ, SEQ), 'attention map')

    def test_backward(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        enc = TransformerEncoder(num_layers=2, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
        check_backward(enc, x)

    def test_registry(self):
        assert 'TRANSFORMER_ENCODER' in list_transformers()

    def test_get_transformer(self):
        enc = get_transformer('TRANSFORMER_ENCODER', num_layers=2,
                              embed_dim=EMBED, num_heads=NUM_HEADS)
        assert isinstance(enc, TransformerEncoder)


# ---------------------------------------------------------------------------
# TransformerDecoder
# ---------------------------------------------------------------------------

class TestTransformerDecoder:

    def test_forward_shared_context(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ * 2,  EMBED))
        dec = TransformerDecoder(num_layers=2, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
        variables = dec.init(KEY, x, ctx, train=False)
        out = dec.apply(variables, x, ctx, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_forward_per_layer_context(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctxs = [jax.random.normal(jax.random.PRNGKey(i),
                                  (BATCH, SEQ * 2, EMBED)) for i in range(2)]
        dec = TransformerDecoder(num_layers=2, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
        variables = dec.init(KEY, x, ctxs[0], train=False)
        out = dec.apply(variables, x, ctxs, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_per_layer_context_wrong_length_raises(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctxs = [jax.random.normal(KEY, (BATCH, SEQ, EMBED))]  # only 1, need 2
        dec  = TransformerDecoder(num_layers=2, embed_dim=EMBED,
                                  num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
        variables = dec.init(KEY, x, ctxs[0], train=False)
        with pytest.raises(ValueError, match="per-layer context"):
            dec.apply(variables, x, ctxs, train=False)

    def test_causal_decoder(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        dec = TransformerDecoder(num_layers=2, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO,
                                 causal=True)
        variables = dec.init(KEY, x, ctx, train=False)
        out = dec.apply(variables, x, ctx, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))

    def test_output_depends_on_context(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctx1 = jax.random.normal(jax.random.PRNGKey(1), (BATCH, SEQ, EMBED))
        ctx2 = jax.random.normal(jax.random.PRNGKey(2), (BATCH, SEQ, EMBED))
        dec  = TransformerDecoder(num_layers=2, embed_dim=EMBED,
                                  num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
        variables = dec.init(KEY, x, ctx1, train=False)
        out1 = dec.apply(variables, x, ctx1, train=False)
        out2 = dec.apply(variables, x, ctx2, train=False)
        assert not jnp.allclose(out1, out2)

    def test_backward(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,     EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ * 2, EMBED))
        dec = TransformerDecoder(num_layers=2, embed_dim=EMBED,
                                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
        check_backward(dec, x, ctx)

    def test_registry(self):
        assert 'TRANSFORMER_DECODER' in list_transformers()


# ---------------------------------------------------------------------------
# ViT
# ---------------------------------------------------------------------------

class TestViT:

    def test_forward_no_head(self):
        x   = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        vit = ViT(patch_size=PATCH, embed_dim=VIT_EMBED, num_heads=VIT_HEADS,
                  num_layers=VIT_LAYERS, mlp_ratio=MLP_RATIO)
        variables = vit.init(KEY, x, train=False)
        out = vit.apply(variables, x, train=False)
        check_shape(out, (BATCH, VIT_EMBED))   # CLS token features
        check_finite(out)

    def test_forward_with_head(self):
        x   = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        vit = ViT(patch_size=PATCH, embed_dim=VIT_EMBED, num_heads=VIT_HEADS,
                  num_layers=VIT_LAYERS, mlp_ratio=MLP_RATIO,
                  num_classes=CLASSES)
        variables = vit.init(KEY, x, train=False)
        out = vit.apply(variables, x, train=False)
        check_shape(out, (BATCH, CLASSES))
        check_finite(out)

    def test_cls_token_exists(self):
        x   = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        vit = ViT(patch_size=PATCH, embed_dim=VIT_EMBED, num_heads=VIT_HEADS,
                  num_layers=VIT_LAYERS, mlp_ratio=MLP_RATIO)
        variables = vit.init(KEY, x, train=False)
        assert 'cls_token' in variables['params']

    def test_pos_encoding_exists(self):
        x   = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        vit = ViT(patch_size=PATCH, embed_dim=VIT_EMBED, num_heads=VIT_HEADS,
                  num_layers=VIT_LAYERS, mlp_ratio=MLP_RATIO)
        variables = vit.init(KEY, x, train=False)
        assert 'pos_encoding' in variables['params']

    def test_different_images_different_output(self):
        x1  = jax.random.normal(jax.random.PRNGKey(0),
                                (BATCH, IMG_H, IMG_W, IMG_C))
        x2  = jax.random.normal(jax.random.PRNGKey(1),
                                (BATCH, IMG_H, IMG_W, IMG_C))
        vit = ViT(patch_size=PATCH, embed_dim=VIT_EMBED, num_heads=VIT_HEADS,
                  num_layers=VIT_LAYERS, mlp_ratio=MLP_RATIO)
        variables = vit.init(KEY, x1, train=False)
        out1 = vit.apply(variables, x1, train=False)
        out2 = vit.apply(variables, x2, train=False)
        assert not jnp.allclose(out1, out2)

    def test_get_attention_maps(self):
        x   = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        vit = ViT(patch_size=PATCH, embed_dim=VIT_EMBED, num_heads=VIT_HEADS,
                  num_layers=VIT_LAYERS, mlp_ratio=MLP_RATIO)
        variables = vit.init(KEY, x, train=False)
        maps = vit.apply(variables, x, train=False,
                         method=vit.get_attention_maps)
        assert len(maps) == VIT_LAYERS
        T_plus_1 = NUM_PATCHES + 1
        for w in maps:
            check_shape(w, (BATCH, VIT_HEADS, T_plus_1, T_plus_1),
                        'vit attention map')

    def test_dropout_eval_deterministic(self):
        x   = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        vit = ViT(patch_size=PATCH, embed_dim=VIT_EMBED, num_heads=VIT_HEADS,
                  num_layers=VIT_LAYERS, mlp_ratio=MLP_RATIO,
                  dropout_rate=0.5)
        variables = vit.init({'params': KEY, 'dropout': DROP_KEY},
                             x, train=True)
        out1 = vit.apply(variables, x, train=False)
        out2 = vit.apply(variables, x, train=False)
        assert jnp.allclose(out1, out2)

    def test_backward(self):
        x   = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        vit = ViT(patch_size=PATCH, embed_dim=VIT_EMBED, num_heads=VIT_HEADS,
                  num_layers=VIT_LAYERS, mlp_ratio=MLP_RATIO)
        check_backward(vit, x)

    def test_output_changes_after_grad_step(self):
        x   = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        vit = ViT(patch_size=PATCH, embed_dim=VIT_EMBED, num_heads=VIT_HEADS,
                  num_layers=VIT_LAYERS, mlp_ratio=MLP_RATIO,
                  num_classes=CLASSES)
        variables = vit.init(KEY, x, train=False)

        def loss_fn(params):
            return jnp.mean(
                vit.apply({'params': params}, x, train=False) ** 2
            )

        grads     = jax.grad(loss_fn)(variables['params'])
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(variables['params'])
        updates, _  = optimizer.update(grads, opt_state)
        new_params  = optax.apply_updates(variables['params'], updates)
        out_before  = vit.apply(variables, x, train=False)
        out_after   = vit.apply({'params': new_params}, x, train=False)
        assert not jnp.allclose(out_before, out_after)

    def test_registry(self):
        assert 'VIT' in list_transformers()

    def test_get_transformer(self):
        vit = get_transformer('VIT', patch_size=PATCH, embed_dim=VIT_EMBED,
                              num_heads=VIT_HEADS, num_layers=VIT_LAYERS)
        assert isinstance(vit, ViT)


# ---------------------------------------------------------------------------
# MaskedViT
# ---------------------------------------------------------------------------

class TestMaskedViT:

    def _init_and_apply(self, mvit, x, train=True):
        init_rngs = {'params': KEY, 'mask': MASK_KEY}
        if train:
            init_rngs['dropout'] = DROP_KEY
        variables = mvit.init(init_rngs, x, train=train)
        apply_rngs = {'mask': MASK_KEY}
        return variables, mvit.apply(variables, x, train=train,
                                     rngs=apply_rngs)

    def test_train_output_shapes(self):
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        variables, (visible, mask, ids_restore) = self._init_and_apply(
            mvit, x, train=True
        )
        T_vis = NUM_PATCHES - int(NUM_PATCHES * 0.75)
        check_shape(visible,     (BATCH, T_vis,       VIT_EMBED), 'visible')
        check_shape(mask,        (BATCH, NUM_PATCHES),            'mask')
        check_shape(ids_restore, (BATCH, NUM_PATCHES),            'ids_restore')

    def test_eval_returns_all_tokens(self):
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        variables = mvit.init({'params': KEY, 'mask': MASK_KEY},
                              x, train=False)
        visible, mask, ids_restore = mvit.apply(variables, x, train=False)
        check_shape(visible,     (BATCH, NUM_PATCHES, VIT_EMBED))
        assert not jnp.any(mask), "eval mask should be all-False"

    def test_mask_ratio_respected(self):
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        for ratio in [0.25, 0.5, 0.75]:
            mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                             num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                             mlp_ratio=MLP_RATIO, mask_ratio=ratio)
            variables = mvit.init({'params': KEY, 'mask': MASK_KEY},
                                  x, train=True)
            visible, mask, _ = mvit.apply(
                variables, x, train=True, rngs={'mask': MASK_KEY}
            )
            expected_masked  = int(NUM_PATCHES * ratio)
            expected_visible = NUM_PATCHES - expected_masked
            assert visible.shape[1] == expected_visible, \
                f"ratio={ratio}: expected {expected_visible} visible tokens"
            assert mask.sum(axis=1)[0] == expected_masked, \
                f"ratio={ratio}: expected {expected_masked} masked positions"

    def test_ids_restore_is_valid_permutation(self):
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        variables = mvit.init({'params': KEY, 'mask': MASK_KEY},
                              x, train=True)
        _, _, ids_restore = mvit.apply(
            variables, x, train=True, rngs={'mask': MASK_KEY}
        )
        # ids_restore must be a permutation of [0, T) for each sample
        for b in range(BATCH):
            sorted_ids = jnp.sort(ids_restore[b])
            assert jnp.array_equal(sorted_ids, jnp.arange(NUM_PATCHES)), \
                f"ids_restore[{b}] is not a valid permutation"

    def test_different_mask_keys_give_different_masks(self):
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        variables = mvit.init({'params': KEY, 'mask': MASK_KEY},
                              x, train=True)
        _, mask1, _ = mvit.apply(variables, x, train=True,
                                 rngs={'mask': jax.random.PRNGKey(0)})
        _, mask2, _ = mvit.apply(variables, x, train=True,
                                 rngs={'mask': jax.random.PRNGKey(99)})
        assert not jnp.array_equal(mask1, mask2), \
            "Different mask keys should produce different masks"

    def test_finite_output(self):
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        variables, (visible, mask, ids_restore) = self._init_and_apply(
            mvit, x, train=True
        )
        check_finite(visible,     'visible tokens')
        check_finite(ids_restore.astype(jnp.float32), 'ids_restore')

    def test_backward(self):
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        variables = mvit.init({'params': KEY, 'mask': MASK_KEY},
                              x, train=True)

        def loss_fn(params):
            visible, _, _ = mvit.apply(
                {'params': params}, x, train=False
            )
            return jnp.mean(visible ** 2)

        grads = jax.grad(loss_fn)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            check_finite(leaf, 'gradient')

    def test_registry(self):
        assert 'MASKED_VIT' in list_transformers()


# ---------------------------------------------------------------------------
# MAEDecoder
# ---------------------------------------------------------------------------

class TestMAEDecoder:

    DECODER_EMBED = 32
    DECODER_HEADS = 4
    DECODER_LAYERS = 2
    PATCH_DIM = PATCH ** 2 * IMG_C   # 48

    def _make_mae_inputs(self, mask_ratio=0.75):
        """Run MaskedViT to get realistic inputs for MAEDecoder."""
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=mask_ratio)
        variables = mvit.init({'params': KEY, 'mask': MASK_KEY},
                              x, train=True)
        visible, mask, ids_restore = mvit.apply(
            variables, x, train=True, rngs={'mask': MASK_KEY}
        )
        return visible, mask, ids_restore

    def test_forward_shape(self):
        visible, mask, ids_restore = self._make_mae_inputs()
        dec = MAEDecoder(num_patches=NUM_PATCHES,
                         patch_dim=self.PATCH_DIM,
                         embed_dim=self.DECODER_EMBED,
                         num_heads=self.DECODER_HEADS,
                         num_layers=self.DECODER_LAYERS,
                         mlp_ratio=MLP_RATIO)
        variables = dec.init(KEY, visible, mask, ids_restore, train=False)
        out = dec.apply(variables, visible, mask, ids_restore, train=False)
        check_shape(out, (BATCH, NUM_PATCHES, self.PATCH_DIM))
        check_finite(out)

    def test_output_covers_all_patches(self):
        # output has T positions (all patches, including masked ones)
        visible, mask, ids_restore = self._make_mae_inputs()
        dec = MAEDecoder(num_patches=NUM_PATCHES,
                         patch_dim=self.PATCH_DIM,
                         embed_dim=self.DECODER_EMBED,
                         num_heads=self.DECODER_HEADS,
                         num_layers=self.DECODER_LAYERS,
                         mlp_ratio=MLP_RATIO)
        variables = dec.init(KEY, visible, mask, ids_restore, train=False)
        out = dec.apply(variables, visible, mask, ids_restore, train=False)
        assert out.shape[1] == NUM_PATCHES, \
            "Decoder should reconstruct all T patch positions"

    def test_different_visible_tokens_different_output(self):
        visible1, mask, ids_restore = self._make_mae_inputs()
        visible2 = jax.random.normal(KEY, visible1.shape)
        dec = MAEDecoder(num_patches=NUM_PATCHES,
                         patch_dim=self.PATCH_DIM,
                         embed_dim=self.DECODER_EMBED,
                         num_heads=self.DECODER_HEADS,
                         num_layers=self.DECODER_LAYERS,
                         mlp_ratio=MLP_RATIO)
        variables = dec.init(KEY, visible1, mask, ids_restore, train=False)
        out1 = dec.apply(variables, visible1, mask, ids_restore, train=False)
        out2 = dec.apply(variables, visible2, mask, ids_restore, train=False)
        assert not jnp.allclose(out1, out2)

    def test_backward(self):
        visible, mask, ids_restore = self._make_mae_inputs()
        dec = MAEDecoder(num_patches=NUM_PATCHES,
                         patch_dim=self.PATCH_DIM,
                         embed_dim=self.DECODER_EMBED,
                         num_heads=self.DECODER_HEADS,
                         num_layers=self.DECODER_LAYERS,
                         mlp_ratio=MLP_RATIO)
        variables = dec.init(KEY, visible, mask, ids_restore, train=False)

        def loss_fn(params):
            out = dec.apply({'params': params}, visible, mask, ids_restore,
                            train=False)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss_fn)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            check_finite(leaf, 'gradient')

    def test_registry(self):
        assert 'MAE_DECODER' in list_transformers()


# ---------------------------------------------------------------------------
# ConvMAEDecoder
# ---------------------------------------------------------------------------

class TestConvMAEDecoder:

    nH = IMG_H // PATCH   # 8
    nW = IMG_W // PATCH   # 8

    def _make_mae_inputs(self):
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        variables = mvit.init({'params': KEY, 'mask': MASK_KEY},
                              x, train=True)
        return mvit.apply(variables, x, train=True, rngs={'mask': MASK_KEY})

    def test_forward_shape(self):
        visible, mask, ids_restore = self._make_mae_inputs()
        dec = ConvMAEDecoder(
            num_patches_h=self.nH, num_patches_w=self.nW,
            encoder_embed_dim=VIT_EMBED, decoder_embed_dim=32,
            channels=(32, 16), out_features=IMG_C,
            num_res_blocks=1,
        )
        variables = dec.init(KEY, visible, mask, ids_restore, train=False)
        out = dec.apply(variables, visible, mask, ids_restore, train=False)
        # ConvDecoder with channels=(32, 16) and 1 upsample:
        # (B, nH, nW, 32) -> (B, nH*2, nW*2, IMG_C)
        check_shape(out, (BATCH, self.nH * 2, self.nW * 2, IMG_C))
        check_finite(out)

    def test_backward(self):
        visible, mask, ids_restore = self._make_mae_inputs()
        dec = ConvMAEDecoder(
            num_patches_h=self.nH, num_patches_w=self.nW,
            encoder_embed_dim=VIT_EMBED, decoder_embed_dim=32,
            channels=(32, 16), out_features=IMG_C,
            num_res_blocks=1,
        )
        variables = dec.init(KEY, visible, mask, ids_restore, train=False)

        def loss_fn(params):
            out = dec.apply({'params': params}, visible, mask, ids_restore,
                            train=False)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss_fn)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            check_finite(leaf, 'gradient')

    def test_registry(self):
        assert 'CONV_MAE_DECODER' in list_transformers()


class TestSwinEncoder:

    # patch_size=4 on 28x28 gives 7x7 grid, divisible by window_size=7
    PATCH_SIZE = 4

    def test_depths_heads_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            SwinEncoder(depths=(2, 2), num_heads=(3, 6, 12))

    def test_forward_no_head(self):
        x   = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, IMG_C))
        enc = SwinEncoder(
            patch_size=self.PATCH_SIZE, embed_dim=SWIN_C,
            depths=(1,), num_heads=(SWIN_HEADS,),
            window_size=WINDOW,
        )
        variables = enc.init(KEY, x, train=False)
        out = enc.apply(variables, x, train=False)
        check_shape(out, (BATCH, SWIN_C))
        check_finite(out)

    def test_forward_with_head(self):
        x   = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, IMG_C))
        enc = SwinEncoder(
            patch_size=self.PATCH_SIZE, embed_dim=SWIN_C,
            depths=(1,), num_heads=(SWIN_HEADS,),
            window_size=WINDOW, num_classes=CLASSES,
        )
        variables = enc.init(KEY, x, train=False)
        out = enc.apply(variables, x, train=False)
        check_shape(out, (BATCH, CLASSES))
        check_finite(out)

    def test_two_stage_channel_doubling(self):
        # after PatchMerging the 7x7 grid becomes 3x3 -- not divisible by 7
        # use a larger input: 56x56 -> 14x14 grid -> 7x7 after PatchMerging
        x   = jax.random.normal(KEY, (BATCH, 56, 56, IMG_C))
        enc = SwinEncoder(
            patch_size=self.PATCH_SIZE, embed_dim=SWIN_C,
            depths=(1, 1), num_heads=(SWIN_HEADS, SWIN_HEADS * 2),
            window_size=WINDOW,
        )
        variables = enc.init(KEY, x, train=False)
        out = enc.apply(variables, x, train=False)
        check_shape(out, (BATCH, SWIN_C * 2))

    def test_non_square_input(self):
        # 28x56 -> 7x14 grid, both divisible by 7
        x   = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W * 2, IMG_C))
        enc = SwinEncoder(
            patch_size=self.PATCH_SIZE, embed_dim=SWIN_C,
            depths=(1,), num_heads=(SWIN_HEADS,),
            window_size=WINDOW,
        )
        variables = enc.init(KEY, x, train=False)
        out = enc.apply(variables, x, train=False)
        check_shape(out, (BATCH, SWIN_C))
        check_finite(out)

    def test_invalid_spatial_dims_raises(self):
        # 15x28: 15 % 4 != 0 so PatchEmbed raises first
        x   = jax.random.normal(KEY, (BATCH, 15, 28, IMG_C))
        enc = SwinEncoder(
            patch_size=self.PATCH_SIZE, embed_dim=SWIN_C,
            depths=(1,), num_heads=(SWIN_HEADS,),
            window_size=WINDOW,
        )
        with pytest.raises(ValueError):
            enc.init(KEY, x, train=False)

    def test_dropout_eval_deterministic(self):
        x   = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, IMG_C))
        enc = SwinEncoder(
            patch_size=self.PATCH_SIZE, embed_dim=SWIN_C,
            depths=(1,), num_heads=(SWIN_HEADS,),
            window_size=WINDOW, dropout_rate=0.5,
        )
        variables = enc.init({'params': KEY, 'dropout': DROP_KEY},
                             x, train=True)
        out1 = enc.apply(variables, x, train=False)
        out2 = enc.apply(variables, x, train=False)
        assert jnp.allclose(out1, out2)

    def test_backward(self):
        x   = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, IMG_C))
        enc = SwinEncoder(
            patch_size=self.PATCH_SIZE, embed_dim=SWIN_C,
            depths=(1,), num_heads=(SWIN_HEADS,),
            window_size=WINDOW,
        )
        check_backward(enc, x)

    def test_registry(self):
        assert 'SWIN_ENCODER' in list_transformers()

    def test_get_transformer(self):
        enc = get_transformer('SWIN_ENCODER', patch_size=self.PATCH_SIZE,
                              embed_dim=SWIN_C, depths=(1,),
                              num_heads=(SWIN_HEADS,))
        assert isinstance(enc, SwinEncoder)
# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_list_transformers_returns_all(self):
        nets = list_transformers()
        expected = {
            'TRANSFORMER_ENCODER', 'TRANSFORMER_DECODER',
            'VIT', 'MASKED_VIT', 'MAE_DECODER',
            'CONV_MAE_DECODER', 'SWIN_ENCODER',
        }
        assert expected == set(nets.keys())

    def test_get_transformer_unknown_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            get_transformer("NONEXISTENT")

    def test_get_transformer_unknown_kwargs_warns(self):
        with pytest.warns(UserWarning, match="unknown kwargs"):
            get_transformer('VIT', patch_size=PATCH, embed_dim=VIT_EMBED,
                            num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                            bogus_param=99)

    def test_register_duplicate_raises(self):
        import flax.linen as nn
        with pytest.raises(ValueError, match="already exists"):
            @register_transformer("VIT", description="duplicate")
            class Duplicate(nn.Module):
                pass

    def test_all_registered_nets_forward(self):
        """Smoke test: every registered net does a forward pass."""
        x_seq  = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        x_img  = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        x_swin = jax.random.normal(KEY, (BATCH, SWIN_H, SWIN_W, IMG_C))

        for name in list_transformers():
            if name == 'TRANSFORMER_ENCODER':
                net = get_transformer(name, num_layers=1, embed_dim=EMBED,
                                      num_heads=NUM_HEADS)
                v   = net.init(KEY, x_seq, train=False)
                out = net.apply(v, x_seq, train=False)
                assert jnp.all(jnp.isfinite(out))

            elif name == 'TRANSFORMER_DECODER':
                ctx = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
                net = get_transformer(name, num_layers=1, embed_dim=EMBED,
                                      num_heads=NUM_HEADS)
                v   = net.init(KEY, x_seq, ctx, train=False)
                out = net.apply(v, x_seq, ctx, train=False)
                assert jnp.all(jnp.isfinite(out))

            elif name == 'VIT':
                net = get_transformer(name, patch_size=PATCH,
                                      embed_dim=VIT_EMBED,
                                      num_heads=VIT_HEADS,
                                      num_layers=VIT_LAYERS,
                                      num_classes=CLASSES)
                v   = net.init(KEY, x_img, train=False)
                out = net.apply(v, x_img, train=False)
                assert out.shape == (BATCH, CLASSES)

            elif name == 'MASKED_VIT':
                net = get_transformer(name, patch_size=PATCH,
                                      embed_dim=VIT_EMBED,
                                      num_heads=VIT_HEADS,
                                      num_layers=VIT_LAYERS,
                                      mask_ratio=0.75)
                v = net.init({'params': KEY, 'mask': MASK_KEY},
                             x_img, train=True)
                visible, mask, ids_restore = net.apply(
                    v, x_img, train=True, rngs={'mask': MASK_KEY}
                )
                assert jnp.all(jnp.isfinite(visible))

            elif name == 'MAE_DECODER':
                # build compatible inputs from MaskedViT
                mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                                 num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                                 mlp_ratio=MLP_RATIO, mask_ratio=0.75)
                mv   = mvit.init({'params': KEY, 'mask': MASK_KEY},
                                 x_img, train=True)
                visible, mask, ids_restore = mvit.apply(
                    mv, x_img, train=True, rngs={'mask': MASK_KEY}
                )
                net = get_transformer(name, num_patches=NUM_PATCHES,
                                      patch_dim=PATCH**2 * IMG_C,
                                      embed_dim=VIT_EMBED,
                                      num_heads=VIT_HEADS, num_layers=1)
                v   = net.init(KEY, visible, mask, ids_restore, train=False)
                out = net.apply(v, visible, mask, ids_restore, train=False)
                assert jnp.all(jnp.isfinite(out))

            elif name == 'CONV_MAE_DECODER':
                mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                                 num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                                 mlp_ratio=MLP_RATIO, mask_ratio=0.75)
                mv   = mvit.init({'params': KEY, 'mask': MASK_KEY},
                                 x_img, train=True)
                visible, mask, ids_restore = mvit.apply(
                    mv, x_img, train=True, rngs={'mask': MASK_KEY}
                )
                nH = IMG_H // PATCH
                nW = IMG_W // PATCH
                net = get_transformer(
                    name, num_patches_h=nH, num_patches_w=nW,
                    encoder_embed_dim=VIT_EMBED, decoder_embed_dim=32,
                    channels=(32, 16), out_features=IMG_C, num_res_blocks=1,
                )
                v   = net.init(KEY, visible, mask, ids_restore, train=False)
                out = net.apply(v, visible, mask, ids_restore, train=False)
                assert jnp.all(jnp.isfinite(out))


            elif name == 'SWIN_ENCODER':
                net = get_transformer(name, patch_size=4,
                                      embed_dim=SWIN_C, depths=(1,),
                                      num_heads=(SWIN_HEADS,),
                                      num_classes=CLASSES)
                v = net.init(KEY, x_swin, train=False)
                out = net.apply(v, x_swin, train=False)
                assert out.shape == (BATCH, CLASSES)


# ---------------------------------------------------------------------------
# Integration: MaskedViT -> MAEDecoder roundtrip
# ---------------------------------------------------------------------------

class TestMAERoundtrip:

    def test_encoder_decoder_shapes_compatible(self):
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        enc_vars = mvit.init({'params': KEY, 'mask': MASK_KEY},
                             x, train=True)
        visible, mask, ids_restore = mvit.apply(
            enc_vars, x, train=True, rngs={'mask': MASK_KEY}
        )

        dec = MAEDecoder(num_patches=NUM_PATCHES,
                         patch_dim=PATCH**2 * IMG_C,
                         embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=2,
                         mlp_ratio=MLP_RATIO)
        dec_vars = dec.init(KEY, visible, mask, ids_restore, train=False)
        recon = dec.apply(dec_vars, visible, mask, ids_restore, train=False)

        check_shape(recon, (BATCH, NUM_PATCHES, PATCH**2 * IMG_C))
        check_finite(recon)

    def test_masked_positions_receive_reconstruction(self):
        # masked positions should produce non-trivial output (not all zeros)
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        enc_vars = mvit.init({'params': KEY, 'mask': MASK_KEY},
                             x, train=True)
        visible, mask, ids_restore = mvit.apply(
            enc_vars, x, train=True, rngs={'mask': MASK_KEY}
        )
        dec = MAEDecoder(num_patches=NUM_PATCHES,
                         patch_dim=PATCH**2 * IMG_C,
                         embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=2,
                         mlp_ratio=MLP_RATIO)
        dec_vars = dec.init(KEY, visible, mask, ids_restore, train=False)
        recon = dec.apply(dec_vars, visible, mask, ids_restore, train=False)

        # extract reconstruction at masked positions for first sample
        masked_recon = recon[0][mask[0]]   # (num_masked, patch_dim)
        assert masked_recon.shape[0] == int(NUM_PATCHES * 0.75)
        assert not jnp.allclose(masked_recon, jnp.zeros_like(masked_recon)), \
            "Masked positions should receive non-zero reconstruction"

    def test_conv_mae_encoder_decoder_shapes_compatible(self):
        x    = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        enc_vars = mvit.init({'params': KEY, 'mask': MASK_KEY},
                             x, train=True)
        visible, mask, ids_restore = mvit.apply(
            enc_vars, x, train=True, rngs={'mask': MASK_KEY}
        )
        nH = IMG_H // PATCH
        nW = IMG_W // PATCH
        dec = ConvMAEDecoder(
            num_patches_h=nH, num_patches_w=nW,
            encoder_embed_dim=VIT_EMBED, decoder_embed_dim=32,
            channels=(32, 16), out_features=IMG_C, num_res_blocks=1,
        )
        dec_vars = dec.init(KEY, visible, mask, ids_restore, train=False)
        recon = dec.apply(dec_vars, visible, mask, ids_restore, train=False)
        check_shape(recon, (BATCH, nH * 2, nW * 2, IMG_C))
        check_finite(recon)

    def test_end_to_end_backward(self):
        # full MAE: encoder + decoder, single backward pass with real MAE loss
        x = jax.random.normal(KEY, (BATCH, IMG_H, IMG_W, IMG_C))
        mvit = MaskedViT(patch_size=PATCH, embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                         mlp_ratio=MLP_RATIO, mask_ratio=0.75)
        dec = MAEDecoder(num_patches=NUM_PATCHES,
                         patch_dim=PATCH ** 2 * IMG_C,
                         embed_dim=VIT_EMBED,
                         num_heads=VIT_HEADS, num_layers=2,
                         mlp_ratio=MLP_RATIO)

        # build target patches from the original image
        nH = IMG_H // PATCH
        nW = IMG_W // PATCH
        target = x.reshape(BATCH, nH, PATCH, nW, PATCH, IMG_C)
        target = target.transpose(0, 1, 3, 2, 4, 5)
        target = target.reshape(BATCH, NUM_PATCHES, -1)  # (B, T, P*P*C)

        enc_vars = mvit.init({'params': KEY, 'mask': MASK_KEY},
                             x, train=True)
        visible, mask, ids_restore = mvit.apply(
            enc_vars, x, train=True, rngs={'mask': MASK_KEY}
        )
        dec_vars = dec.init(KEY, visible, mask, ids_restore, train=False)

        def loss_fn(enc_params, dec_params):
            # train=True with fixed mask key -- deterministic masking,
            # non-zero loss on masked patches, gradients flow through encoder
            vis, m, ids = mvit.apply(
                {'params': enc_params}, x, train=True,
                rngs={'mask': MASK_KEY},
            )
            recon = dec.apply(
                {'params': dec_params}, vis, m, ids, train=False
            )
            # MAE loss: MSE on masked patches only
            return jnp.mean(
                jnp.where(m[:, :, None], (recon - target) ** 2, 0.0)
            )

        enc_grads, dec_grads = jax.grad(loss_fn, argnums=(0, 1))(
            enc_vars['params'], dec_vars['params']
        )
        for leaf in jax.tree_util.tree_leaves(enc_grads):
            check_finite(leaf, 'encoder gradient')
        for leaf in jax.tree_util.tree_leaves(dec_grads):
            check_finite(leaf, 'decoder gradient')

        enc_grad_norm = sum(
            jnp.sum(g ** 2)
            for g in jax.tree_util.tree_leaves(enc_grads)
        )
        dec_grad_norm = sum(
            jnp.sum(g ** 2)
            for g in jax.tree_util.tree_leaves(dec_grads)
        )
        assert enc_grad_norm > 0, "Encoder gradients should be non-zero"
        assert dec_grad_norm > 0, "Decoder gradients should be non-zero"