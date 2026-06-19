# tests/core/test_attention.py
import pytest
import jax
import jax.numpy as jnp

from core.attention import (
    # mask utilities
    make_causal_mask,
    make_padding_mask,
    # modules
    MultiHeadAttention,
    CrossAttention,
    # registry
    get_attention,
    list_attention,
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

    # --- return_weights (PRE-softmax logits) ---

    def test_return_weights_shape(self):
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = attn.init(KEY, x, train=False)
        out, s = attn.apply(variables, x, train=False, return_weights=True)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_shape(s, (BATCH, NUM_HEADS, SEQ, SEQ), 'scores')
        check_finite(s, 'scores')

    def test_return_weights_are_presoftmax(self):
        # Returned scores are PRE-softmax logits: not a probability distribution
        # (can be negative), but softmax over T_kv recovers one that sums to 1.
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = attn.init(KEY, x, train=False)
        _, s = attn.apply(variables, x, train=False, return_weights=True)
        assert bool(jnp.any(s < 0.0)), \
            "Pre-softmax logits should contain negative values"
        probs = jax.nn.softmax(s, axis=-1)
        row_sums = probs.sum(axis=-1)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5), \
            "softmax(scores) must sum to 1 across T_kv"

    def test_return_weights_no_dropout_contamination(self):
        # Pre-softmax scores depend only on Q, K -- identical with/without train
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, dropout_rate=0.0)
        variables = attn.init(KEY, x, train=False)
        _, s_train = attn.apply(variables, x, train=True,  return_weights=True)
        _, s_eval  = attn.apply(variables, x, train=False, return_weights=True)
        assert jnp.allclose(s_train, s_eval, atol=1e-5), \
            "Scores should be identical at train/eval when dropout_rate=0"

    def test_return_weights_dropout_free(self):
        # Even with dropout on, returned scores are the clean pre-softmax logits
        # (dropout acts on the softmaxed weights in the forward path, not here).
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, dropout_rate=0.5)
        variables = attn.init({'params': KEY, 'dropout': DROP_KEY}, x, train=True)
        _, s_train = attn.apply(
            variables, x, train=True, return_weights=True,
            rngs={'dropout': DROP_KEY},
        )
        _, s_eval = attn.apply(variables, x, train=False, return_weights=True)
        assert jnp.allclose(s_train, s_eval, atol=1e-5), \
            "Returned scores must be dropout-free regardless of train flag"

    def test_causal_weights_future_softmax_zero(self):
        # After softmax, future positions (strict upper triangle) get ~0 weight.
        x    = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        attn = MultiHeadAttention(embed_dim=EMBED, num_heads=NUM_HEADS, causal=True)
        variables = attn.init(KEY, x, train=False)
        _, s = attn.apply(variables, x, train=False, return_weights=True)
        probs = jax.nn.softmax(s, axis=-1)
        upper = jnp.triu(probs, k=1)
        assert jnp.allclose(upper, jnp.zeros_like(upper), atol=1e-6), \
            "Causal attention: future positions should have ~0 weight after softmax"

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
        out, s = cross.apply(variables, x, train=False, context=ctx,
                             return_weights=True)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_shape(s, (BATCH, NUM_HEADS, SEQ, SEQ // 2), 'cross scores')

    def test_return_weights_presoftmax_distribution(self):
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ // 2, EMBED))
        cross = CrossAttention(embed_dim=EMBED, num_heads=NUM_HEADS)
        variables = cross.init(KEY, x, train=False, context=ctx)
        _, s = cross.apply(variables, x, train=False, context=ctx,
                           return_weights=True)
        probs = jax.nn.softmax(s, axis=-1)
        row_sums = probs.sum(axis=-1)
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
# Attention registry
# ---------------------------------------------------------------------------

class TestAttentionRegistry:

    def test_registered_names(self):
        names = list_attention()
        assert set(names) == {'SELF_ATTENTION', 'CROSS_ATTENTION'}

    def test_get_self_attention(self):
        attn = get_attention('self_attention', embed_dim=EMBED, num_heads=NUM_HEADS)
        assert isinstance(attn, MultiHeadAttention)
        x = jax.random.normal(KEY, (BATCH, SEQ, EMBED))
        _, out = init_and_apply(attn, x, train=False)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_get_cross_attention(self):
        attn = get_attention('cross_attention', embed_dim=EMBED, num_heads=NUM_HEADS)
        assert isinstance(attn, CrossAttention)
        x   = jax.random.normal(KEY, (BATCH, SEQ,      EMBED))
        ctx = jax.random.normal(KEY, (BATCH, SEQ // 2, EMBED))
        _, out = init_and_apply(attn, x, train=False, context=ctx)
        check_shape(out, (BATCH, SEQ, EMBED))
        check_finite(out)

    def test_get_is_case_insensitive(self):
        attn = get_attention('Self_Attention', embed_dim=EMBED, num_heads=NUM_HEADS)
        assert isinstance(attn, MultiHeadAttention)

    def test_unknown_attention_raises(self):
        with pytest.raises(ValueError):
            get_attention('not_an_attention', embed_dim=EMBED, num_heads=NUM_HEADS)
