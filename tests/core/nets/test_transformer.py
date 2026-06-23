"""Tests for core/nets/transformer.py — composable transformer blocks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.nets.transformer import BlockConfig, PreLNAttentionBlock, LearnedTokens

B, N, M, D, H = 4, 6, 8, 32, 2
_CFG = BlockConfig(embed_dim=D, num_heads=H, mlp_ratio=2.0)


def _init(mod, *args):
    v = mod.init({'params': jax.random.PRNGKey(0)}, *args, train=False)
    return v, mod.apply(v, *args, train=False)


class TestBlockConfig:
    def test_is_hashable_static_field(self):
        assert hash(_CFG) == hash(BlockConfig(embed_dim=D, num_heads=H, mlp_ratio=2.0))

    def test_ffn_builds_mlp(self):
        from core.nets.mlp import MLP
        assert isinstance(_CFG.ffn(), MLP)


class TestPreLNAttentionBlock:

    def test_self_attention_shapes(self):
        x = jnp.zeros((B, N, D))
        _, (out, scores) = _init(PreLNAttentionBlock(cfg=_CFG), x)
        assert out.shape == (B, N, D)
        assert scores.shape == (B, H, N, N)

    def test_cross_attention_shapes(self):
        q = jnp.zeros((B, 1, D)); ctx = jnp.zeros((B, M, D))
        blk = PreLNAttentionBlock(cfg=_CFG, cross_attention=True)
        _, (out, scores) = _init(blk, q, ctx)
        assert out.shape == (B, 1, D)
        assert scores.shape == (B, H, 1, M)

    def test_always_returns_pair(self):
        # No return_weights flag — the block always yields (out, scores).
        x = jnp.zeros((B, N, D))
        _, res = _init(PreLNAttentionBlock(cfg=_CFG), x)
        assert isinstance(res, tuple) and len(res) == 2

    def test_residual_transforms_input(self):
        rng = np.random.default_rng(0)
        x = jnp.array(rng.standard_normal((B, N, D)).astype(np.float32))
        _, (out, _) = _init(PreLNAttentionBlock(cfg=_CFG), x)
        assert not np.allclose(np.asarray(out), np.asarray(x))

    def test_cross_block_has_kv_norm_self_does_not(self):
        q = jnp.zeros((B, 1, D)); ctx = jnp.zeros((B, M, D))
        vc = PreLNAttentionBlock(cfg=_CFG, cross_attention=True).init(
            {'params': jax.random.PRNGKey(0)}, q, ctx, train=False)
        vs = PreLNAttentionBlock(cfg=_CFG).init(
            {'params': jax.random.PRNGKey(0)}, jnp.zeros((B, N, D)), train=False)
        assert 'norm_kv' in vc['params']
        assert 'norm_kv' not in vs['params']

    def test_mask_changes_output(self):
        rng = np.random.default_rng(1)
        q   = jnp.array(rng.standard_normal((B, 1, D)).astype(np.float32))
        ctx = jnp.array(rng.standard_normal((B, M, D)).astype(np.float32))
        blk = PreLNAttentionBlock(cfg=_CFG, cross_attention=True)
        v   = blk.init({'params': jax.random.PRNGKey(0)}, q, ctx, train=False)
        full = blk.apply(v, q, ctx, train=False)[0]
        mask = jnp.array([[True, True, True, True, False, False, False, False]] * B)[:, None, :]
        part = blk.apply(v, q, ctx, mask=mask, train=False)[0]
        assert not np.allclose(np.asarray(full), np.asarray(part))


class TestLearnedTokens:

    def test_param_shape(self):
        v = LearnedTokens(num_tokens=N, dim=D).init(
            {'params': jax.random.PRNGKey(0)}, B)
        assert v['params']['tokens'].shape == (N, D)

    def test_broadcast_over_batch(self):
        mod = LearnedTokens(num_tokens=N, dim=D)
        v   = mod.init({'params': jax.random.PRNGKey(0)}, B)
        out = mod.apply(v, B)
        assert out.shape == (B, N, D)
        # identical across the batch dim (a broadcast, not per-sample params)
        assert np.allclose(np.asarray(out[0]), np.asarray(out[1]))

    def test_custom_param_name(self):
        v = LearnedTokens(num_tokens=1, dim=D, param_name='query').init(
            {'params': jax.random.PRNGKey(0)}, B)
        assert 'query' in v['params'] and v['params']['query'].shape == (1, D)
