"""Tests for core/nets/transformers.py — v3 blocks + Perceiver compositions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.nets.transformers import (
    BlockConfig,
    SelfAttentionBlock,
    CrossAttentionBlock,
    LearnedTokens,
    PerceiverEncoder,
    PerceiverDecoder,
    PerceiverIO,
    Perceiver,
)

KEY = jax.random.PRNGKey(0)

B  = 4     # batch
T  = 12    # data tokens (stations, padded width)
C  = 17    # data channels (asymmetric on purpose)
N  = 6     # latents
D  = 32    # latent channels
H  = 4     # heads

_CFG = BlockConfig(num_channels=D, num_heads=H)


def _init(mod, *args, **kwargs):
    v = mod.init({'params': KEY}, *args, train=False, **kwargs)
    return v, mod.apply(v, *args, train=False, **kwargs)


def _data(key=KEY):
    return jax.random.normal(key, (B, T, C))


def _latstream(key=KEY):
    return jax.random.normal(key, (B, N, D))


def _param_count(tree):
    return sum(x.size for x in jax.tree_util.tree_leaves(tree))


# ---------------------------------------------------------------------------
# BlockConfig
# ---------------------------------------------------------------------------

class TestBlockConfig:
    def test_is_hashable_static_field(self):
        assert hash(_CFG) == hash(BlockConfig(num_channels=D, num_heads=H))

    def test_widening_factor_sets_ffn_hidden(self):
        cfg = BlockConfig(num_channels=D, num_heads=H, widening_factor=4)
        assert cfg.ffn().hidden_features == 4 * D


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class TestSelfAttentionBlock:
    def test_shapes_and_pair(self):
        x = _latstream()
        _, (out, probs) = _init(SelfAttentionBlock(_CFG), x)
        assert out.shape == (B, N, D)
        assert probs.shape == (B, H, N, N)

    def test_residual_transforms_input(self):
        x = _latstream()
        _, (out, _) = _init(SelfAttentionBlock(_CFG), x)
        assert not np.allclose(np.asarray(out), np.asarray(x))


class TestCrossAttentionBlock:
    def test_asymmetric_channels(self):
        # latent stream D=32 attending data C=17 — no width matching needed
        q, kv = _latstream(), _data()
        _, (out, probs) = _init(CrossAttentionBlock(_CFG), q, kv)
        assert out.shape == (B, N, D)
        assert probs.shape == (B, H, N, T)

    def test_num_attn_channels_never_touches_residual_shapes(self):
        # arithmetic at 16 channels; residual stream stays at D
        cfg = BlockConfig(num_channels=D, num_heads=H, num_attn_channels=16)
        q, kv = _latstream(), _data()
        v, (out, _) = _init(CrossAttentionBlock(cfg), q, kv)
        assert out.shape == (B, N, D)
        attn = v['params']['attn']
        assert attn['q_proj']['kernel'].shape == (D, H, 4)
        assert attn['out_proj']['kernel'].shape == (H, 4, D)


# ---------------------------------------------------------------------------
# LearnedTokens
# ---------------------------------------------------------------------------

class TestLearnedTokens:
    def test_broadcast_shape(self):
        v, out = _initless_tokens(LearnedTokens(N, D))
        assert out.shape == (B, N, D)
        assert v['params']['tokens'].shape == (N, D)

    def test_trunc_sigma_bounds_the_init(self):
        # std 1.0, cutoff 2 sigma -> |v| <= 2 guaranteed
        mod = LearnedTokens(100, 100, init_std=1.0, trunc_sigma=2.0)
        v = mod.init({'params': KEY}, B)
        assert jnp.max(jnp.abs(v['params']['tokens'])) <= 2.0 + 1e-6

    def test_trunc_sigma_none_is_plain_normal(self):
        # 10k draws at std 1.0: some exceed 2 sigma with near-certainty
        mod = LearnedTokens(100, 100, init_std=1.0, trunc_sigma=None)
        v = mod.init({'params': KEY}, B)
        assert jnp.max(jnp.abs(v['params']['tokens'])) > 2.0


def _initless_tokens(mod):
    v = mod.init({'params': KEY}, B)
    return v, mod.apply(v, B)


# ---------------------------------------------------------------------------
# PerceiverEncoder — repeats / shared / pad_handling
# ---------------------------------------------------------------------------

class TestPerceiverEncoder:
    def test_single_repeat_shapes_and_weights(self):
        enc = PerceiverEncoder(_CFG, num_latents=N, depth=2)
        v, (lat, w) = _init(enc, _data())
        assert lat.shape == (B, N, D)
        assert set(w) == {'repeat_1'}
        assert set(w['repeat_1']) == {'cross', 'self_0', 'self_1'}
        assert 'unit_1' in v['params'] and 'unit_n' not in v['params']

    def test_unshared_repeats_have_distinct_units(self):
        enc = PerceiverEncoder(_CFG, num_latents=N, depth=1, repeats=3)
        v, (lat, w) = _init(enc, _data())
        units = {k for k in v['params'] if k.startswith('unit')}
        assert units == {'unit_1', 'unit_2', 'unit_3'}
        assert set(w) == {'repeat_1', 'repeat_2', 'repeat_3'}

    def test_shared_repeats_use_one_extra_unit(self):
        enc = PerceiverEncoder(_CFG, num_latents=N, depth=1, repeats=3,
                               shared=True)
        v, _ = _init(enc, _data())
        units = {k for k in v['params'] if k.startswith('unit')}
        assert units == {'unit_1', 'unit_n'}

    def test_shared_has_fewer_params_than_unshared(self):
        shared = PerceiverEncoder(_CFG, num_latents=N, depth=1, repeats=4,
                                  shared=True)
        unshared = PerceiverEncoder(_CFG, num_latents=N, depth=1, repeats=4)
        vs, _ = _init(shared, _data())
        vu, _ = _init(unshared, _data())
        assert _param_count(vs) < _param_count(vu)

    def test_mask_arm_pad_content_is_irrelevant(self):
        data = _data()
        kv_valid = jnp.broadcast_to(jnp.arange(T)[None, :] < 7, (B, T))
        enc = PerceiverEncoder(_CFG, num_latents=N, depth=1)
        v = enc.init({'params': KEY}, data, kv_valid=kv_valid, train=False)
        lat1, _ = enc.apply(v, data, kv_valid=kv_valid, train=False)
        poisoned = data.at[:, 7:, :].set(999.0)
        lat2, _ = enc.apply(v, poisoned, kv_valid=kv_valid, train=False)
        assert jnp.allclose(lat1, lat2, atol=1e-5)

    def test_learned_arm_substitutes_and_uses_no_mask(self):
        data = _data()
        kv_valid = jnp.broadcast_to(jnp.arange(T)[None, :] < 7, (B, T))
        enc = PerceiverEncoder(_CFG, num_latents=N, depth=1,
                               pad_handling='learned')
        v = enc.init({'params': KEY}, data, kv_valid=kv_valid, train=False)
        assert 'pad_token' in v['params']
        # raw content of invalid slots irrelevant (substituted before attn)
        lat1, w1 = enc.apply(v, data, kv_valid=kv_valid, train=False)
        poisoned = data.at[:, 7:, :].set(999.0)
        lat2, _ = enc.apply(v, poisoned, kv_valid=kv_valid, train=False)
        assert jnp.allclose(lat1, lat2, atol=1e-5)
        # NO mask: the pad token receives nonzero attention mass
        probs = w1['repeat_1']['cross']            # (B, H, N, T)
        assert jnp.max(probs[..., 7:]) > 0.0

    def test_mask_arm_zeroes_pad_columns(self):
        data = _data()
        kv_valid = jnp.broadcast_to(jnp.arange(T)[None, :] < 7, (B, T))
        enc = PerceiverEncoder(_CFG, num_latents=N, depth=1)
        v = enc.init({'params': KEY}, data, kv_valid=kv_valid, train=False)
        _, w = enc.apply(v, data, kv_valid=kv_valid, train=False)
        assert jnp.allclose(w['repeat_1']['cross'][..., 7:], 0.0, atol=1e-7)

    def test_invalid_pad_handling_raises(self):
        enc = PerceiverEncoder(_CFG, num_latents=N, depth=1,
                               pad_handling='zeropad')
        with pytest.raises(ValueError, match="pad_handling"):
            enc.init({'params': KEY}, _data(), train=False)


# ---------------------------------------------------------------------------
# Decoder / PerceiverIO / Perceiver
# ---------------------------------------------------------------------------

class TestPerceiverDecoder:
    def test_head_maps_to_out_channels(self):
        queries = jax.random.normal(KEY, (B, 1, D))
        latents = _latstream()
        dec = PerceiverDecoder(_CFG, num_out_channels=9)
        _, (y, probs) = _init(dec, queries, latents)
        assert y.shape == (B, 1, 9)
        assert probs.shape == (B, H, 1, N)


class TestPerceiverIO:
    def test_default_learned_query_classification(self):
        model = PerceiverIO(_CFG, num_latents=N, depth=2,
                            num_out_channels=9)
        v, (y, w) = _init(model, _data())
        assert y.shape == (B, 1, 9)
        assert set(w) == {'encoder', 'decode'}
        assert 'queries' in v['params']

    def test_caller_queries_with_own_channels(self):
        # Senseiver-style: coord features ⊕ learned token, 24 channels
        queries = jax.random.normal(KEY, (B, 5, 24))
        model = PerceiverIO(_CFG, num_latents=N, depth=1)
        v = model.init({'params': KEY}, _data(), queries=queries, train=False)
        y, _ = model.apply(v, _data(), queries=queries, train=False)
        assert y.shape == (B, 5, 24)
        assert 'queries' not in v['params']    # no learned query built

    def test_senseiver_preset_knobs_compose(self):
        # senseiver = PIO at repeats>1, shared=True, widening_factor=1
        cfg = BlockConfig(num_channels=D, num_heads=H, widening_factor=1)
        model = PerceiverIO(cfg, num_latents=N, depth=2, repeats=3,
                            shared=True, num_out_channels=2)
        kv_valid = jnp.broadcast_to(jnp.arange(T)[None, :] < 9, (B, T))
        v = model.init({'params': KEY}, _data(), kv_valid=kv_valid,
                       train=False)
        y, w = model.apply(v, _data(), kv_valid=kv_valid, train=False)
        assert y.shape == (B, 1, 2)
        assert set(w['encoder']) == {'repeat_1', 'repeat_2', 'repeat_3'}

    def test_deterministic_at_eval(self):
        model = PerceiverIO(_CFG, num_latents=N, depth=1, num_out_channels=3)
        v = model.init({'params': KEY}, _data(), train=False)
        y1, _ = model.apply(v, _data(), train=False)
        y2, _ = model.apply(v, _data(), train=False)
        assert jnp.allclose(y1, y2)


class TestPerceiver:
    def test_classic_pooled_logits(self):
        model = Perceiver(_CFG, num_latents=N, depth=2, num_out_channels=9)
        _, (logits, w) = _init(model, _data())
        assert logits.shape == (B, 9)
        assert set(w) == {'repeat_1'}

    def test_mask_flows_through(self):
        data = _data()
        kv_valid = jnp.broadcast_to(jnp.arange(T)[None, :] < 7, (B, T))
        model = Perceiver(_CFG, num_latents=N, depth=1, num_out_channels=9)
        v = model.init({'params': KEY}, data, kv_valid=kv_valid, train=False)
        l1, _ = model.apply(v, data, kv_valid=kv_valid, train=False)
        l2, _ = model.apply(v, data.at[:, 7:, :].set(-5.0),
                            kv_valid=kv_valid, train=False)
        assert jnp.allclose(l1, l2, atol=1e-5)


# ---------------------------------------------------------------------------
# Dropout plumbing
# ---------------------------------------------------------------------------

class TestDropout:
    def test_train_dropout_needs_rng_and_runs(self):
        cfg = BlockConfig(num_channels=D, num_heads=H,
                          dropout=0.3, residual_dropout=0.3)
        model = PerceiverIO(cfg, num_latents=N, depth=1, num_out_channels=3)
        v = model.init({'params': KEY, 'dropout': jax.random.PRNGKey(1)},
                       _data(), train=True)
        y, _ = model.apply(v, _data(), train=True,
                           rngs={'dropout': jax.random.PRNGKey(2)})
        assert jnp.all(jnp.isfinite(y))
