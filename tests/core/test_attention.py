# tests/core/test_attention.py
"""Tests for the v3 ``Attention`` primitive — one class, composed directly."""
import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn

from core.attention import Attention

KEY      = jax.random.PRNGKey(0)
DROP_KEY = jax.random.PRNGKey(1)
BATCH    = 4


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


# ===========================================================================
# v3 primitive: Attention (one class, composed directly — no registry)
# ===========================================================================


QC = 32   # query / latent channels
KC = 17   # key-value / data channels (deliberately != QC and not head-divisible)
TQ = 6    # query tokens (latents)
TK = 12   # key-value tokens (stations)
H3 = 4    # heads


def _mk(num_heads=H3, **kw):
    return Attention(num_heads=num_heads, **kw)


def _data(key=KEY):
    kq, kkv = jax.random.split(key)
    x_q  = jax.random.normal(kq,  (BATCH, TQ, QC))
    x_kv = jax.random.normal(kkv, (BATCH, TK, KC))
    return x_q, x_kv


class TestAttentionPrimitive:

    def test_cross_shapes_asymmetric_channels(self):
        x_q, x_kv = _data()
        _, (out, probs) = init_and_apply(_mk(), x_q, x_kv, train=False)
        check_shape(out, (BATCH, TQ, QC))
        check_shape(probs, (BATCH, H3, TQ, TK), 'probs')
        check_finite(out); check_finite(probs)

    def test_self_attention_is_same_call(self):
        x_q, _ = _data()
        _, (out, probs) = init_and_apply(_mk(), x_q, x_q, train=False)
        check_shape(out, (BATCH, TQ, QC))
        check_shape(probs, (BATCH, H3, TQ, TQ), 'probs')

    def test_probs_are_post_softmax(self):
        x_q, x_kv = _data()
        _, (_, probs) = init_and_apply(_mk(), x_q, x_kv, train=False)
        sums = jnp.sum(probs, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_num_attn_channels_decouples_arithmetic_width(self):
        x_q, x_kv = _data()
        v, (out, _) = init_and_apply(
            _mk(num_attn_channels=16), x_q, x_kv, train=False)
        # arithmetic at 16 channels (4 heads x 4), stream stays QC
        check_shape(out, (BATCH, TQ, QC))
        p = v['params']
        assert p['q_proj']['kernel'].shape == (QC, H3, 4)
        assert p['k_proj']['kernel'].shape == (KC, H3, 4)
        assert p['out_proj']['kernel'].shape == (H3, 4, QC)

    def test_num_out_channels_override(self):
        x_q, x_kv = _data()
        _, (out, _) = init_and_apply(
            _mk(num_out_channels=8), x_q, x_kv, train=False)
        check_shape(out, (BATCH, TQ, 8))

    def test_head_divisibility_enforced(self):
        x_q, x_kv = _data()
        with pytest.raises(ValueError, match="divisible"):
            _mk(num_attn_channels=KC).init({'params': KEY}, x_q, x_kv, train=False)

    def test_default_attn_channels_must_divide_too(self):
        # None -> q channels (32); 5 heads does not divide 32
        x_q, x_kv = _data()
        with pytest.raises(ValueError, match="divisible"):
            _mk(num_heads=5).init({'params': KEY}, x_q, x_kv, train=False)

    def test_mask_zeroes_probs_and_blocks_content(self):
        x_q, x_kv = _data()
        kv_valid = jnp.arange(TK)[None, :] < 7          # (1, TK) -> broadcast (B, TK)
        kv_valid = jnp.broadcast_to(kv_valid, (BATCH, TK))
        mask = nn.make_attention_mask(
            jnp.ones((BATCH, TQ)), kv_valid, dtype=bool)
        attn = _mk()
        v = attn.init({'params': KEY}, x_q, x_kv, mask=mask, train=False)
        out1, probs = attn.apply(v, x_q, x_kv, mask=mask, train=False)
        # masked columns get exactly zero attention weight
        assert jnp.allclose(probs[..., 7:], 0.0, atol=1e-7)
        # and their CONTENT is irrelevant: rewrite masked slots, same output
        x_kv2 = x_kv.at[:, 7:, :].set(123.0)
        out2, _ = attn.apply(v, x_q, x_kv2, mask=mask, train=False)
        assert jnp.allclose(out1, out2, atol=1e-6)

    def test_additive_bias_path(self):
        x_q, x_kv = _data()
        bias = jnp.zeros((1, 1, TQ, TK)).at[..., 7:].set(-1e9)
        attn = _mk()
        v = attn.init({'params': KEY}, x_q, x_kv, bias=bias, train=False)
        _, probs = attn.apply(v, x_q, x_kv, bias=bias, train=False)
        assert jnp.allclose(probs[..., 7:], 0.0, atol=1e-7)

    def test_dropout_needs_rng_and_runs(self):
        x_q, x_kv = _data()
        attn = _mk(dropout_rate=0.5)
        _, (out, _) = init_and_apply(
            attn, x_q, x_kv, with_dropout=True, train=True)
        check_finite(out)

    def test_single_head_works(self):
        x_q, x_kv = _data()
        _, (out, probs) = init_and_apply(_mk(num_heads=1), x_q, x_kv, train=False)
        check_shape(out, (BATCH, TQ, QC))
        check_shape(probs, (BATCH, 1, TQ, TK), 'probs')
