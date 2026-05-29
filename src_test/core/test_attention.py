# src_test/core/nets/test_attention.py
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from core.attention import (
    # mask utilities
    make_causal_mask,
    make_padding_mask,
    make_swin_shift_mask,
    # modules
    MultiHeadAttention,
    CrossAttention,
    SwinWindowAttention,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY      = jax.random.PRNGKey(0)
DROP_KEY = jax.random.PRNGKey(1)

BATCH      = 4
SEQ        = 16
EMBED      = 64
NUM_HEADS  = 4
HEAD_DIM   = EMBED // NUM_HEADS   # 16

# Swin constants -- must satisfy H % window_size == 0
H, W       = 28, 28
WINDOW     = 7
SHIFT      = 3
SWIN_C     = 32
SWIN_HEADS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_and_apply(module, *args, with_dropout=False, train=True, **kwargs):
    """Init and apply a module. Returns (variables, output)."""
    rngs_init  = {'params': KEY, 'dropout': DROP_KEY} if with_dropout else {'params': KEY}
    rngs_apply = {'dropout': DROP_KEY} if with_dropout else {}
    variables  = module.init(rngs_init, *args, train=train, **kwargs)
    out        = module.apply(variables, *args, train=train, rngs=rngs_apply, **kwargs)
    return variables, out


def check_finite(x, label='output'):
    assert jnp.all(jnp.isfinite(x)), f"{label} contains non-finite values"


def check_shape(x, expected, label='output'):
    assert x.shape == expected, f"{label}: expected {expected}, got {x.shape}"


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

class TestMakeCausalMask:
    def test_shape(self):
        mask = make_causal_mask(8)
        check_shape(mask, (8, 8), 'causal mask')

    def test_dtype_bool(self):
        mask = make_causal_mask(4)
        assert mask.dtype == jnp.bool_

    def test_lower_triangle_true(self):
        mask = make_causal_mask(4)
        # diagonal and below: True
        for i in range(4):
            for j in range(4):
                expected = (i >= j)
                assert bool(mask[i, j]) == expected, \
                    f"mask[{i},{j}] should be {expected}"

    def test_no_future_leakage(self):
        mask = make_causal_mask(6)
        # strict upper triangle must all be False
        assert not jnp.any(jnp.triu(mask, k=1))


class TestMakePaddingMask:
    def test_shape(self):
        lengths = jnp.array([3, 5, 2])
        mask = make_padding_mask(lengths, max_len=6)
        check_shape(mask, (3, 6), 'padding mask')

    def test_dtype_bool(self):
        lengths = jnp.array([3, 5])
        mask = make_padding_mask(lengths, max_len=6)
        assert mask.dtype == jnp.bool_

    def test_valid_positions_true(self):
        lengths = jnp.array([3, 5, 2])
        mask = make_padding_mask(lengths, max_len=6)
        expected = jnp.array([
            [True,  True,  True,  False, False, False],
            [True,  True,  True,  True,  True,  False],
            [True,  True,  False, False, False, False],
        ])
        assert jnp.array_equal(mask, expected)

    def test_full_length(self):
        lengths = jnp.array([6, 6])
        mask = make_padding_mask(lengths, max_len=6)
        assert jnp.all(mask)

    def test_zero_length(self):
        lengths = jnp.array([0, 3])
        mask = make_padding_mask(lengths, max_len=4)
        assert not jnp.any(mask[0])
        assert jnp.all(mask[1, :3])


class TestMakeSwinShiftMask:
    def test_shape(self):
        mask = make_swin_shift_mask(window_size=4, shift_size=2, H=16, W=16)
        nW = (16 // 4) * (16 // 4)
        check_shape(mask, (nW, 16, 16), 'swin shift mask')

    def test_zero_shift_returns_zeros(self):
        mask = make_swin_shift_mask(window_size=4, shift_size=0, H=16, W=16)
        assert jnp.all(mask == 0.0)

    def test_values_zero_or_neginf(self):
        mask = make_swin_shift_mask(window_size=4, shift_size=2, H=16, W=16)
        valid = jnp.logical_or(mask == 0.0, mask == -1e9)
        assert jnp.all(valid), "Shift mask should only contain 0.0 or -1e9"

    def test_dtype_float32(self):
        mask = make_swin_shift_mask(window_size=4, shift_size=2, H=16, W=16)
        assert mask.dtype == jnp.float32

    def test_diagonal_always_zero(self):
        # A token always attends to itself -- diagonal must be 0
        mask = make_swin_shift_mask(window_size=4, shift_size=2, H=16, W=16)
        for i in range(mask.shape[0]):
            assert jnp.all(jnp.diag(mask[i]) == 0.0), \
                f"Window {i}: diagonal should be 0.0"

    def test_symmetric(self):
        # Shift mask should be symmetric: if (i,j) is blocked, so is (j,i)
        mask = make_swin_shift_mask(window_size=4, shift_size=2, H=16, W=16)
        assert jnp.allclose(mask, mask.transpose(0, 2, 1))

    def test_window_size_7(self):
        mask = make_swin_shift_mask(window_size=7, shift_size=3, H=28, W=28)
        nW = (28 // 7) ** 2
        check_shape(mask, (nW, 49, 49), 'swin shift mask 7x7')


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:

    # --- Construction ---

    def test_invalid_embed_dim_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(embed_dim=65, num_heads=4)

    # --- Self-attention shapes ---

    def test_self_attention_shape(self):
        x = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        _, out = init_and_apply(attn, x, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_self_attention_backward(self):
        x = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = attn.init(KEY, x, train=False)

        def loss(params):
            out = attn.apply({'params': params}, x, train=False)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        leaves = jax.tree_util.tree_leaves(grads)
        for leaf in leaves:
            check_finite(leaf, 'gradient')

    # --- Cross-attention shapes ---

    def test_cross_attention_shape(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ // 2, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        _, out = init_and_apply(attn, x, train=False, context=ctx)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_cross_attention_different_seq_lens(self):
        x   = jax.random.normal(KEY, (BATCH, 20, EMBED))
        ctx = jax.random.normal(KEY, (BATCH, 7,  EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        _, out = init_and_apply(attn, x, train=False, context=ctx)
        check_shape(out, (BATCH, 20, EMBED))

    # --- Mask shapes ---

    def test_mask_2d(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        mask = jnp.ones((SEQ, SEQ), dtype=jnp.bool_)
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        _, out = init_and_apply(attn, x, train=False, mask=mask)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_mask_3d(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        mask = jnp.ones((BATCH, SEQ, SEQ), dtype=jnp.bool_)
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        _, out = init_and_apply(attn, x, train=False, mask=mask)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_mask_4d(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        mask = jnp.ones((BATCH, NUM_HEADS, SEQ, SEQ), dtype=jnp.bool_)
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        _, out = init_and_apply(attn, x, train=False, mask=mask)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_float_mask(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        mask = jnp.zeros((SEQ, SEQ))   # float, all attend
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        _, out = init_and_apply(attn, x, train=False, mask=mask)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_padding_mask_integration(self):
        x       = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        lengths = jnp.array([SEQ, SEQ // 2, SEQ // 4, SEQ])
        pad     = make_padding_mask(lengths, SEQ)   # (B, T)
        mask    = pad[:, None, :]                   # (B, 1, T) -- broadcast T_q
        attn    = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        _, out  = init_and_apply(attn, x, train=False, mask=mask)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    # --- Causal mask ---

    def test_causal_mask_shape(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, causal=True)
        _, out = init_and_apply(attn, x, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_causal_no_future_influence(self):
        # With causal masking, position 0 output should be identical
        # whether we feed T tokens or 1 token (no future context)
        x_full = jax.random.normal(KEY, (1, SEQ, EMBED))
        x_one  = x_full[:, :1, :]
        attn   = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, causal=True)
        vars_  = attn.init(KEY, x_full, train=False)
        out_full = attn.apply(vars_, x_full, train=False)
        out_one  = attn.apply(vars_, x_one,  train=False)
        # position 0 output should match regardless of future tokens
        assert jnp.allclose(out_full[:, 0, :], out_one[:, 0, :], atol=1e-5), \
            "Causal masking violated: position 0 is affected by future tokens"

    def test_explicit_mask_overrides_causal(self):
        # Passing an explicit all-True mask with causal=True should not crash
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        mask = jnp.ones((SEQ, SEQ), dtype=jnp.bool_)
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, causal=True)
        _, out = init_and_apply(attn, x, train=False, mask=mask)
        check_shape(out, (BATCH, SEQ, EMBED))

    # --- return_weights ---

    def test_return_weights_shape(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = attn.init(KEY, x, train=False)
        out, w = attn.apply(variables, x, train=False, return_weights=True)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_shape(w, (BATCH, NUM_HEADS, SEQ, SEQ), 'weights')
        check_finite(w, 'weights')

    def test_return_weights_sum_to_one(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = attn.init(KEY, x, train=False)
        _, w = attn.apply(variables, x, train=False, return_weights=True)
        row_sums = w.sum(axis=-1)   # (B, num_heads, T_q)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5), \
            "Attention weights must sum to 1 across T_kv"

    def test_return_weights_no_dropout_contamination(self):
        # Returned weights should be identical with and without train
        # when dropout_rate=0 (no recomputation needed)
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, dropout_rate=0.0)
        variables = attn.init(KEY, x, train=False)
        _, w_train = attn.apply(variables, x, train=True,  return_weights=True)
        _, w_eval  = attn.apply(variables, x, train=False, return_weights=True)
        assert jnp.allclose(w_train, w_eval, atol=1e-5), \
            "Weights should be identical at train/eval when dropout_rate=0"

    def test_return_weights_with_dropout_are_clean(self):
        # When dropout_rate > 0 and train=True, returned weights should
        # still sum to 1 (clean, not dropped)
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, dropout_rate=0.5)
        variables = attn.init({'params': KEY, 'dropout': DROP_KEY}, x, train=True)
        _, w = attn.apply(
            variables, x, train=True, return_weights=True,
            rngs={'dropout': DROP_KEY},
        )
        row_sums = w.sum(axis=-1)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5), \
            "Returned weights must sum to 1 (clean, no dropout) even in train mode"

    def test_causal_weights_upper_triangle_zero(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, causal=True)
        variables = attn.init(KEY, x, train=False)
        _, w = attn.apply(variables, x, train=False, return_weights=True)
        # Upper triangle (future positions) must be ~0
        upper = jnp.triu(w, k=1)
        assert jnp.allclose(upper, jnp.zeros_like(upper), atol=1e-6), \
            "Causal attention: future positions should have zero weight"

    # --- Dropout ---

    def test_dropout_eval_deterministic(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, dropout_rate=0.5)
        variables = attn.init({'params': KEY, 'dropout': DROP_KEY}, x, train=True)
        out1 = attn.apply(variables, x, train=False)
        out2 = attn.apply(variables, x, train=False)
        assert jnp.allclose(out1, out2), "Eval should be deterministic"

    def test_dropout_train_stochastic(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, dropout_rate=0.5)
        variables = attn.init({'params': KEY, 'dropout': DROP_KEY}, x, train=True)
        out1 = attn.apply(variables, x, train=True, rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = attn.apply(variables, x, train=True, rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2), "Train outputs should differ across dropout keys"

    # --- Permutation equivariance ---

    def test_permutation_equivariant(self):
        # Shuffle input tokens -> output should be shuffled identically
        x    = jax.random.normal(KEY, (1, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = attn.init(KEY, x, train=False)

        perm    = jax.random.permutation(KEY, SEQ)
        x_perm  = x[:, perm, :]
        out     = attn.apply(variables, x,      train=False)
        out_perm= attn.apply(variables, x_perm, train=False)
        assert jnp.allclose(out[:, perm, :], out_perm, atol=1e-5), \
            "MHA should be permutation-equivariant without positional encoding"

    # --- forward and backward ---

    def test_cross_attention_backward(self):
        x = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ // 2, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = attn.init(KEY, x, train=False, context=ctx)

        def loss(params):
            out = attn.apply({'params': params}, x, train=False, context=ctx)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            check_finite(leaf, 'gradient')

    def test_causal_backward(self):
        x = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, causal=True)
        variables = attn.init(KEY, x, train=False)

        def loss(params):
            out = attn.apply({'params': params}, x, train=False)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            check_finite(leaf, 'gradient')

    def test_backward_with_dropout(self):
        # Gradients must flow correctly through the dropout path during training
        x = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS,
                                  dropout_rate=0.1)
        variables = attn.init({'params': KEY, 'dropout': DROP_KEY}, x, train=True)

        def loss(params):
            out = attn.apply(
                {'params': params}, x, train=True,
                rngs={'dropout': DROP_KEY},
            )
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            check_finite(leaf, 'gradient')


# ---------------------------------------------------------------------------
# CrossAttention
# ---------------------------------------------------------------------------

class TestCrossAttention:

    def test_forward_shape(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ // 2, EMBED))
        cross = CrossAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        _, out = init_and_apply(cross, x, train=False, context=ctx)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_output_depends_on_context(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctx1 = jax.random.normal(jax.random.PRNGKey(1), (BATCH, SEQ, EMBED))
        ctx2 = jax.random.normal(jax.random.PRNGKey(2), (BATCH, SEQ, EMBED))
        cross = CrossAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = cross.init(KEY, x, train=False, context=ctx1)
        out1 = cross.apply(variables, x, train=False, context=ctx1)
        out2 = cross.apply(variables, x, train=False, context=ctx2)
        assert not jnp.allclose(out1, out2), \
            "Output should differ when context differs"

    def test_return_weights_shape(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ // 2, EMBED))
        cross = CrossAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = cross.init(KEY, x, train=False, context=ctx)
        out, w = cross.apply(variables, x, train=False, context=ctx,
                             return_weights=True)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_shape(w, (BATCH, NUM_HEADS, SEQ, SEQ // 2), 'cross weights')

    def test_return_weights_sum_to_one(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ // 2, EMBED))
        cross = CrossAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = cross.init(KEY, x, train=False, context=ctx)
        _, w = cross.apply(variables, x, train=False, context=ctx,
                           return_weights=True)
        row_sums = w.sum(axis=-1)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5)

    def test_backward(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ // 2, EMBED))
        cross = CrossAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = cross.init(KEY, x, train=False, context=ctx)

        def loss(params):
            out = cross.apply({'params': params}, x, train=False, context=ctx)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            check_finite(leaf, 'gradient')

    def test_not_causal(self):
        # CrossAttention should never be causal -- causal is not a parameter
        # and is hardcoded False in setup(). Verify it is not configurable.
        cross = CrossAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        assert not hasattr(cross, 'causal'), \
            "CrossAttention must not expose causal as a configurable parameter"

        # Also verify it can be initialised and applied -- if causal were
        # accidentally True it would still run, but the above check is the
        # structural guarantee
        x = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        variables = cross.init(KEY, x, train=False, context=ctx)
        out = cross.apply(variables, x, train=False, context=ctx)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_backward_with_dropout(self):
        x = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ // 2, EMBED))
        cross = CrossAttention(embed_dim=EMBED, num_heads=NUM_HEADS,
                               dropout_rate=0.1)
        variables = cross.init(
            {'params': KEY, 'dropout': DROP_KEY}, x, train=True, context=ctx
        )

        def loss(params):
            out = cross.apply(
                {'params': params}, x, train=True, context=ctx,
                rngs={'dropout': DROP_KEY},
            )
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            check_finite(leaf, 'gradient')


# ---------------------------------------------------------------------------
# SwinWindowAttention
# ---------------------------------------------------------------------------

class TestSwinWindowAttention:

    # --- Construction ---

    def test_invalid_embed_dim_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            SwinWindowAttention(embed_dim=33, num_heads=4, window_size=7)

    def test_shift_size_gte_window_raises(self):
        with pytest.raises(ValueError, match="shift_size"):
            SwinWindowAttention(embed_dim=32, num_heads=4, window_size=7,
                                shift_size=7)

    def test_invalid_spatial_dims_raises(self):
        x    = jax.random.normal(KEY, (BATCH, 15, 15, SWIN_C))  # 15 % 7 != 0
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=7)
        with pytest.raises(ValueError, match="divisible"):
            attn.init(KEY, x, train=False)

    # --- W-MSA (no shift) ---

    def test_wmsa_forward_shape(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        _, out = init_and_apply(attn, x, train=False)
        check_shape(out, (BATCH, H, W, SWIN_C))
        check_finite(out)

    def test_wmsa_output_same_shape_as_input(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        variables = attn.init(KEY, x, train=False)
        out = attn.apply(variables, x, train=False)
        assert out.shape == x.shape

    def test_wmsa_backward(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        variables = attn.init(KEY, x, train=False)

        def loss(params):
            out = attn.apply({'params': params}, x, train=False)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            check_finite(leaf, 'gradient')

    # --- SW-MSA (with shift) ---

    def test_swmsa_forward_shape(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW, shift_size=SHIFT)
        _, out = init_and_apply(attn, x, train=False)
        check_shape(out, (BATCH, H, W, SWIN_C))
        check_finite(out)

    def test_swmsa_output_same_shape_as_input(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW, shift_size=SHIFT)
        variables = attn.init(KEY, x, train=False)
        out = attn.apply(variables, x, train=False)
        assert out.shape == x.shape

    def test_swmsa_backward(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW, shift_size=SHIFT)
        variables = attn.init(KEY, x, train=False)

        def loss(params):
            out = attn.apply({'params': params}, x, train=False)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            check_finite(leaf, 'gradient')

    def test_shift_and_no_shift_differ(self):
        x      = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        w_msa  = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                     window_size=WINDOW, shift_size=0)
        sw_msa = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                     window_size=WINDOW, shift_size=SHIFT)
        v_w  = w_msa.init(KEY,  x, train=False)
        v_sw = sw_msa.init(KEY, x, train=False)
        out_w  = w_msa.apply(v_w,   x, train=False)
        out_sw = sw_msa.apply(v_sw, x, train=False)
        assert not jnp.allclose(out_w, out_sw), \
            "W-MSA and SW-MSA should produce different outputs"

    # --- return_weights ---

    def test_return_weights_shape_wmsa(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        variables = attn.init(KEY, x, train=False)
        out, w = attn.apply(variables, x, train=False, return_weights=True)
        nW = (H // WINDOW) * (W // WINDOW)
        check_shape(out, (BATCH, H, W, SWIN_C))
        check_shape(w, (BATCH * nW, SWIN_HEADS, WINDOW**2, WINDOW**2), 'swin weights')
        check_finite(w, 'swin weights')

    def test_return_weights_sum_to_one(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        variables = attn.init(KEY, x, train=False)
        _, w = attn.apply(variables, x, train=False, return_weights=True)
        row_sums = w.sum(axis=-1)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5), \
            "Per-window attention weights must sum to 1"

    def test_return_weights_shape_swmsa(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW, shift_size=SHIFT)
        variables = attn.init(KEY, x, train=False)
        out, w = attn.apply(variables, x, train=False, return_weights=True)
        nW = (H // WINDOW) * (W // WINDOW)
        check_shape(w, (BATCH * nW, SWIN_HEADS, WINDOW**2, WINDOW**2), 'sw weights')

    # --- Relative position bias ---

    def test_rel_pos_bias_table_shape(self):
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        x = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        variables = attn.init(KEY, x, train=False)
        table = variables['params']['rel_pos_bias_table']
        expected = (2 * WINDOW - 1, 2 * WINDOW - 1, SWIN_HEADS)
        check_shape(table, expected, 'rel_pos_bias_table')

    def test_rel_pos_bias_affects_output(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        variables = attn.init(KEY, x, train=False)
        out_with_bias = attn.apply(variables, x, train=False)

        import flax
        zeroed = flax.core.copy(
            variables,
            {'params': {**variables['params'],
                        'rel_pos_bias_table': jnp.zeros_like(
                            variables['params']['rel_pos_bias_table'])}}
        )
        out_no_bias = attn.apply(zeroed, x, train=False)
        assert not jnp.allclose(out_with_bias, out_no_bias), \
            "Relative position bias should affect output"

    def test_rel_pos_bias_gradient_flows(self):
        # Gradients must flow through the relative position bias table
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        variables = attn.init(KEY, x, train=False)

        def loss(params):
            out = attn.apply({'params': params}, x, train=False)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss)(variables['params'])
        bias_grad = grads['rel_pos_bias_table']
        check_finite(bias_grad, 'rel_pos_bias_table gradient')
        assert jnp.any(bias_grad != 0.0), \
            "rel_pos_bias_table should receive non-zero gradients"

    # --- Dropout ---

    def test_dropout_eval_deterministic(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW, dropout_rate=0.5)
        variables = attn.init({'params': KEY, 'dropout': DROP_KEY}, x, train=True)
        out1 = attn.apply(variables, x, train=False)
        out2 = attn.apply(variables, x, train=False)
        assert jnp.allclose(out1, out2)

    def test_dropout_train_stochastic(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW, dropout_rate=0.5)
        variables = attn.init({'params': KEY, 'dropout': DROP_KEY}, x, train=True)
        out1 = attn.apply(variables, x, train=True,
                          rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = attn.apply(variables, x, train=True,
                          rngs={'dropout': jax.random.PRNGKey(99)})
        assert not jnp.allclose(out1, out2)

    # --- Window partition roundtrip ---

    def test_partition_merge_roundtrip(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        variables = attn.init(KEY, x, train=False)
        bound    = attn.bind(variables)
        windows  = bound._partition_windows(x)
        restored = bound._merge_windows(windows, BATCH, H, W)
        assert jnp.allclose(x, restored), \
            "partition -> merge should be identity"

    # --- External mask argument ---

    def test_external_mask_wrong_ndim_raises(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        variables = attn.init(KEY, x, train=False)
        bad_mask  = jnp.zeros((WINDOW**2, WINDOW**2))  # 2D instead of 3D
        with pytest.raises(AssertionError, match="num_windows"):
            attn.apply(variables, x, train=False, mask=bad_mask)

    def test_external_mask_valid_shape_accepted(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        variables = attn.init(KEY, x, train=False)
        nW        = (H // WINDOW) * (W // WINDOW)
        M2        = WINDOW ** 2
        zero_mask = jnp.zeros((nW, M2, M2))
        out_no_mask   = attn.apply(variables, x, train=False)
        out_with_mask = attn.apply(variables, x, train=False, mask=zero_mask)
        assert jnp.allclose(out_no_mask, out_with_mask, atol=1e-5), \
            "Zero external mask should not change output"
        check_shape(out_with_mask, (BATCH, H, W, SWIN_C))
        check_finite(out_with_mask)

    def test_external_mask_neginf_blocks_attention(self):
        x    = jax.random.normal(KEY, (BATCH, H, W, SWIN_C))
        attn = SwinWindowAttention(embed_dim=SWIN_C, num_heads=SWIN_HEADS,
                                   window_size=WINDOW)
        variables = attn.init(KEY, x, train=False)
        nW         = (H // WINDOW) * (W // WINDOW)
        M2         = WINDOW ** 2
        block_mask = jnp.full((nW, M2, M2), -1e9).at[
            :, jnp.arange(M2), jnp.arange(M2)
        ].set(0.0)
        out_no_mask    = attn.apply(variables, x, train=False)
        out_block_mask = attn.apply(variables, x, train=False, mask=block_mask)
        assert not jnp.allclose(out_no_mask, out_block_mask), \
            "Blocking external mask should change output"
        check_shape(out_block_mask, (BATCH, H, W, SWIN_C))
        check_finite(out_block_mask)