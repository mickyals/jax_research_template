"""
Tests for experiments/tc_perceiver_io/train/model.py (TCPerceiverIO).

Fake-data tests: no disk access, run always.
Real-data tests: require E:/sparse_obs data files, skipped if absent.

The model is a staged Perceiver-IO (Read cross-attn encode → Processor latent
self-attention → Decoder). It is coordinate-agnostic: there is no query/CLS
token (the learned latent array is the encode query), so station_coords are
just (B, M, 2) features and the model never reads query_coords.
"""

from __future__ import annotations

import os

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from experiments.tc_perceiver_io.train.model import TCPerceiverIO
from experiments.tc_perceiver_io.data.sources.ibtracs import N_CLASSES

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

KEY    = jax.random.PRNGKey(0)
B      = 4       # batch size
M      = 8       # max stations (small for speed)
F      = 5       # obs features
EMBED  = 32      # keep small for fast tests
HEADS  = 2
NLAT   = 6       # num latents (N)
LAYERS = 2       # num processor blocks (L)


# ---------------------------------------------------------------------------
# Fake batch factory
# ---------------------------------------------------------------------------

def _fake_X(
    batch_size:  int = B,
    n_stations:  int = M,
    n_features:  int = F,
    all_present: bool = True,
    n_real:      int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Build a synthetic X dict matching the model's input contract."""
    if rng is None:
        rng = np.random.default_rng(42)

    n_real = n_real if n_real is not None else n_stations

    station_obs    = rng.standard_normal((batch_size, n_stations, n_features)).astype(np.float32)
    station_coords = rng.uniform(-1.0, 1.0, (batch_size, n_stations, 2)).astype(np.float32)

    station_mask             = np.zeros((batch_size, n_stations), dtype=bool)
    station_mask[:, :n_real] = True

    if all_present:
        obs_mask = np.ones((batch_size, n_stations, n_features), dtype=bool)
    else:
        obs_mask = rng.random((batch_size, n_stations, n_features)) > 0.3

    return {
        'station_obs':    jnp.array(station_obs),
        'station_coords': jnp.array(station_coords),
        'station_mask':   jnp.array(station_mask),
        'obs_mask':       jnp.array(obs_mask),
    }


def _make_model(**kwargs) -> TCPerceiverIO:
    # n_classes is set by default so the classifier tests get a head; the
    # headless path (n_classes=None → returns pooled z) is exercised in the
    # split/attach roundtrip test.
    defaults = dict(
        embed_dim          = EMBED,
        num_heads          = HEADS,
        num_latents        = NLAT,
        num_process_layers = LAYERS,
        fourier_dim        = 16,
        n_obs_features     = F,
        n_classes          = N_CLASSES,
    )
    defaults.update(kwargs)
    return TCPerceiverIO(**defaults)


# ---------------------------------------------------------------------------
# TCPerceiverIO — forward / backward on fake data
# ---------------------------------------------------------------------------

class TestForward:

    def _init(self, **extra):
        model = _make_model(**extra)
        X     = _fake_X()
        vs    = model.init(KEY, X, train=False)
        return model, vs, X

    def test_output_shape(self):
        model, vs, X = self._init()
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)

    def test_output_finite(self):
        model, vs, X = self._init()
        logits = model.apply(vs, X, train=False)
        assert jnp.all(jnp.isfinite(logits)), "logits contain NaN or inf"

    def test_missing_obs_changes_output(self):
        model = _make_model()
        X_present = _fake_X(all_present=True)
        X_missing = _fake_X(all_present=False)
        vs = model.init(KEY, X_present, train=False)
        out_present = model.apply(vs, X_present, train=False)
        out_missing = model.apply(vs, X_missing, train=False)
        assert not jnp.allclose(out_present, out_missing)

    def test_padding_changes_output(self):
        model = _make_model()
        X_full = _fake_X(n_real=M)
        X_half = _fake_X(n_real=M // 2)
        vs = model.init(KEY, X_full, train=False)
        out_full = model.apply(vs, X_full, train=False)
        out_half = model.apply(vs, X_half, train=False)
        assert not jnp.allclose(out_full, out_half)

    def test_single_head(self):
        model, vs, X = self._init(num_heads=1)
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)
        assert jnp.all(jnp.isfinite(logits))

    def test_multi_process_layers(self):
        model, vs, X = self._init(num_process_layers=4)
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)
        assert jnp.all(jnp.isfinite(logits))

    def test_avgproj_decode_mode(self):
        model, vs, X = self._init(decode_mode='avgproj')
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)
        assert jnp.all(jnp.isfinite(logits))

    def test_headless_returns_pooled_latents(self):
        model = _make_model(n_classes=None)
        X  = _fake_X()
        vs = model.init(KEY, X, train=False)
        z  = model.apply(vs, X, train=False)
        assert z.shape == (B, EMBED)
        assert jnp.all(jnp.isfinite(z))

    def test_gradient_flows(self):
        """Loss gradient w.r.t. all parameters must be non-None and finite."""
        model = _make_model()
        X     = _fake_X()
        vs    = model.init(KEY, X, train=False)
        labels = jnp.zeros(B, dtype=jnp.int32)

        def loss_fn(params):
            logits = model.apply({'params': params}, X, train=False)
            return jnp.mean(
                jax.vmap(lambda l, y: -l[y] + jax.nn.logsumexp(l))(logits, labels)
            )

        grads = jax.grad(loss_fn)(vs['params'])
        leaves = jax.tree_util.tree_leaves(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves), \
            "non-finite gradient detected"
        assert any(jnp.any(g != 0) for g in leaves), \
            "all gradients are zero — no gradient flow"

    def test_train_vs_eval_differ_with_dropout(self):
        """With dropout, train and eval outputs should differ."""
        model = _make_model(dropout_rate=0.5)
        X     = _fake_X()
        vs    = model.init({'params': KEY, 'dropout': KEY}, X, train=True)
        out_eval  = model.apply(vs, X, train=False)
        out_train = model.apply(
            vs, X, train=True,
            rngs={'dropout': jax.random.PRNGKey(99)},
        )
        assert not jnp.allclose(out_eval, out_train)


# ---------------------------------------------------------------------------
# return_weights — per-component pre-softmax attention dict
# ---------------------------------------------------------------------------

class TestReturnWeights:

    def test_dict_shapes(self):
        model = _make_model()
        X  = _fake_X()
        vs = model.init(KEY, X, train=False)
        logits, attn = model.apply(vs, X, train=False, return_weights=True)
        assert logits.shape == (B, N_CLASSES)
        assert set(attn) == {'read', 'processor', 'decoder'}
        assert attn['read'].shape      == (B, HEADS, NLAT, M)
        assert attn['processor'].shape == (LAYERS, B, HEADS, NLAT, NLAT)
        assert attn['decoder'].shape   == (B, HEADS, 1, NLAT)

    def test_scores_softmax_to_distributions(self):
        # Scores are PRE-softmax; softmax over the last axis must give rows that
        # sum to one (valid attention distributions).
        model = _make_model()
        X  = _fake_X()
        vs = model.init(KEY, X, train=False)
        _, attn = model.apply(vs, X, train=False, return_weights=True)
        for key in ('read', 'processor', 'decoder'):
            p = jax.nn.softmax(attn[key], axis=-1)
            assert jnp.all(jnp.isfinite(p))
            assert jnp.allclose(p.sum(axis=-1), 1.0, atol=1e-5)

    def test_avgproj_has_no_decoder_scores(self):
        model = _make_model(decode_mode='avgproj')
        X  = _fake_X()
        vs = model.init(KEY, X, train=False)
        _, attn = model.apply(vs, X, train=False, return_weights=True)
        assert attn['decoder'] is None

    def test_headless_returns_read_and_processor_only(self):
        model = _make_model(n_classes=None)
        X  = _fake_X()
        vs = model.init(KEY, X, train=False)
        rep, attn = model.apply(vs, X, train=False, return_weights=True)
        assert rep.shape == (B, EMBED)
        assert set(attn) == {'read', 'processor'}


# ---------------------------------------------------------------------------
# Missingness indicator
# ---------------------------------------------------------------------------

def test_missingness_indicator_disambiguates_observed_zero():
    """A missing feature (filled with 0) must produce a different token than a
    real observation that equals 0, when missingness_indicator=True — the mask
    channel carries the disambiguation."""
    model = _make_model(missingness_indicator=True)
    X = _fake_X(all_present=True)

    # A real observation that happens to be exactly 0 everywhere.
    X_real_zero = dict(X)
    X_real_zero['station_obs'] = jnp.zeros_like(X['station_obs'])
    X_real_zero['obs_mask']    = jnp.ones_like(X['obs_mask'])

    # A genuinely missing feature — also fills to 0, but obs_mask=False.
    X_missing = dict(X)
    X_missing['station_obs'] = jnp.zeros_like(X['station_obs'])
    X_missing['obs_mask']    = jnp.zeros_like(X['obs_mask'])

    vs = model.init(KEY, X_real_zero, train=False)
    out_real    = model.apply(vs, X_real_zero, train=False)
    out_missing = model.apply(vs, X_missing, train=False)
    assert not jnp.allclose(out_real, out_missing), \
        "missingness_indicator=True should distinguish observed-0 from absent"


def test_missingness_indicator_false_aliases_observed_zero():
    """Without the mask channel, a missing feature (filled 0) IS aliased with a
    real observation equal to 0 — documents the aliasing the indicator fixes."""
    model = _make_model(missingness_indicator=False)
    X = _fake_X(all_present=True)

    X_real_zero = dict(X)
    X_real_zero['station_obs'] = jnp.zeros_like(X['station_obs'])
    X_real_zero['obs_mask']    = jnp.ones_like(X['obs_mask'])

    X_missing = dict(X)
    X_missing['station_obs'] = jnp.zeros_like(X['station_obs'])
    X_missing['obs_mask']    = jnp.zeros_like(X['obs_mask'])

    vs = model.init(KEY, X_real_zero, train=False)
    out_real    = model.apply(vs, X_real_zero, train=False)
    out_missing = model.apply(vs, X_missing, train=False)
    assert jnp.allclose(out_real, out_missing), \
        "missingness_indicator=False should alias observed-0 with absent (legacy behaviour)"


def test_missingness_indicator_false_runs():
    """missingness_indicator=False (no mask channel) produces finite output."""
    model  = _make_model(missingness_indicator=False)
    X      = _fake_X(all_present=False)   # some missing obs
    vs     = model.init(KEY, X, train=False)
    logits = model.apply(vs, X, train=False)
    assert logits.shape == (B, N_CLASSES)
    assert jnp.all(jnp.isfinite(logits))


def test_processor_weight_sharing_reduces_params_keeps_depth():
    """Recurrent weight tying = one Processor block applied num_process_layers
    times: fewer params, single blocks_0 leaf, but still L attention applications."""
    L = 4
    shared = _make_model(num_process_layers=L, processor_weight_sharing=True)
    indep  = _make_model(num_process_layers=L, processor_weight_sharing=False)
    X  = _fake_X()
    ps = shared.init(KEY, X, train=False)['params']
    pi = indep.init(KEY, X, train=False)['params']

    n_shared = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(ps))
    n_indep  = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(pi))
    assert n_shared < n_indep                       # tying drops parameters

    assert set(ps['processor'].keys()) == {'blocks_0'}      # one shared block
    assert len(pi['processor'].keys()) == L                 # L distinct blocks

    # Still runs, and the Processor still applies L times (depth preserved).
    logits, attn = shared.apply({'params': ps}, X, train=False, return_weights=True)
    assert logits.shape == (B, N_CLASSES)
    assert attn['processor'].shape[0] == L
    assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# Encoder / head split — frozen-encoder probing
# ---------------------------------------------------------------------------
# The seam is the ``decoder`` KEY: the encoder asset is every other leaf
# (latents, norm, processor, read, token_proj), identical to a HEADLESS model's
# param tree.

_ENCODER_KEYS = {'latents', 'norm', 'processor', 'read', 'token_proj'}


def test_param_tree_head_is_separable_leaf():
    """Flat param tree — the head is the separable 'decoder' leaf and the
    encoder is every OTHER leaf (no nested 'encoder' subtree)."""
    model = _make_model()
    X  = _fake_X()
    vs = model.init(KEY, X, train=False)
    keys = set(vs['params'].keys())
    assert 'decoder' in keys and 'encoder' not in keys and 'head' not in keys
    # the encoder leaves are exactly the coordinate-agnostic Perceiver body
    assert keys - {'decoder'} == _ENCODER_KEYS


def test_split_and_attach_encoder_roundtrip():
    """split_encoder_head + attach_encoder transplant the encoder into a fresh
    model with a new head — the frozen-encoder transfer operation."""
    from experiments.tc_perceiver_io.train.model import (
        split_encoder_head, attach_encoder,
    )
    model = _make_model()
    X  = _fake_X()
    vs = model.init(KEY, X, train=False)
    enc_params, head_params = split_encoder_head(vs['params'])
    assert set(head_params) == {'decoder'} and 'decoder' not in enc_params
    assert set(enc_params) == _ENCODER_KEYS

    # A HEADLESS encoder (n_classes=None) runs on the split params → (B, D).
    encoder = TCPerceiverIO(embed_dim=EMBED, num_heads=HEADS, num_latents=NLAT,
                            num_process_layers=LAYERS, fourier_dim=16,
                            n_obs_features=F)   # n_classes=None
    z = encoder.apply({'params': enc_params}, X, train=False)
    assert z.shape == (B, EMBED) and jnp.all(jnp.isfinite(z))

    # Attach the trained encoder into a freshly initialised model (new head).
    fresh  = _make_model().init(jax.random.PRNGKey(7), X, train=False)['params']
    merged = attach_encoder(fresh, enc_params)
    # encoder leaves come from enc_params; the decoder keeps its fresh init.
    assert merged['read']      is enc_params['read']
    assert merged['processor'] is enc_params['processor']
    assert merged['decoder']   is fresh['decoder']
    # the fresh head differs from the original model's head
    assert not jnp.allclose(merged['decoder']['head']['kernel'],
                            vs['params']['decoder']['head']['kernel'])
    logits = model.apply({'params': merged}, X, train=False)
    assert logits.shape == (B, N_CLASSES) and jnp.all(jnp.isfinite(logits))


def test_encoder_freeze_labels_partition():
    """encoder_freeze_labels marks every encoder leaf 'frozen' and the decoder
    leaves 'trainable', matching the param structure."""
    from experiments.tc_perceiver_io.train.model import encoder_freeze_labels
    model = _make_model()
    X  = _fake_X()
    vs = model.init(KEY, X, train=False)
    labels = encoder_freeze_labels(vs['params'])
    head_labels = set(jax.tree_util.tree_leaves(labels['decoder']))
    enc_labels  = set(jax.tree_util.tree_leaves(
        {k: v for k, v in labels.items() if k != 'decoder'}))
    assert enc_labels == {'frozen'}
    assert head_labels == {'trainable'}


# ---------------------------------------------------------------------------
# TCPerceiverIO — real data (skipped if files absent)
# ---------------------------------------------------------------------------

_REAL_DATA_PATHS = {
    'ibtracs_path':    'E:/sparse_obs/ibtracs/ibtracs_full.npz',
    'multi_storm_path':'E:/sparse_obs/ibtracs/ibtracs_multi_storm_times.npz',
    'insitu_obs_path': 'E:/sparse_obs/insitu-land/insitu_land_clean.npz',
    'insitu_meta_path':'E:/sparse_obs/insitu-land/insitu_land_station_meta.npz',
}

_real_data_available = all(
    os.path.exists(p) for p in _REAL_DATA_PATHS.values()
)

_skip_real = pytest.mark.skipif(
    not _real_data_available or not os.environ.get('RUN_REAL_DATA_TESTS'),
    reason="Real data tests disabled. Set RUN_REAL_DATA_TESTS=1 to enable.",
)


@_skip_real
class TestRealData:

    @pytest.fixture(scope='class')
    def loader(self):
        from experiments.tc_perceiver_io.data.datamodule import TCDataModule
        config = {
            **_REAL_DATA_PATHS,
            'reliability_levels':   ['always_active', 'mostly_active'],
            'obs_vars': [
                'air_pressure_at_sea_level',
                'air_temperature',
                'dew_point_temperature',
                'wind_east',
                'wind_north',
            ],
            'radius_km':             500.0,
            'time_window_hours':     0.1,
            'max_stations':          64,
            'min_stations':          1,
            'batch_size':            8,
            'tc_fraction':           0.5,
            'fov_lat':               [0.0, 30.0],
            'fov_lon':               [-100.0, -45.0],
            'background_buffer_hours': 6.0,
            'location_encoding':     'unit_circle',
            'obs_normalisation':     'minmax_11',
            'obs_bounds': {
                'air_pressure_at_sea_level': [87000.0, 108400.0],
                'air_temperature':           [193.0, 333.0],
                'dew_point_temperature':     [193.0, 308.0],
                'wind_east':                 [-115.0, 115.0],
                'wind_north':                [-115.0, 115.0],
            },
        }
        dm = TCDataModule.from_config(config)
        return dm.train_loader(batch_size=8, seed=0, shuffle=False)

    def test_forward_pass_on_real_batch(self, loader):
        batch = next(iter(loader))
        X, y  = batch['X'], batch['y']

        model  = _make_model()
        vs     = model.init(KEY, X, train=False)
        logits = model.apply(vs, X, train=False)

        assert logits.shape == (y.shape[0], N_CLASSES)
        assert jnp.all(jnp.isfinite(logits)), "NaN/inf in logits on real batch"

    def test_real_batch_shapes(self, loader):
        batch = next(iter(loader))
        X     = batch['X']
        assert X['station_obs'].shape[-1]    == F
        assert X['station_coords'].shape[-1] == 2
        assert X['obs_mask'].shape           == X['station_obs'].shape
