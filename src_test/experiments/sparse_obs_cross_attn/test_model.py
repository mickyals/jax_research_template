"""
Tests for experiments/sparse_obs_cross_attn/model.py.

Fake-data tests: no disk access, run always.
Real-data tests: require E:/sparse_obs data files, skipped if absent.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from experiments.sparse_obs_cross_attn.model import (
    TCClassifier,
    SeparateKVCrossAttentionBlock,
    N_CLASSES,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

KEY   = jax.random.PRNGKey(0)
B     = 4       # batch size
N     = 8       # max stations (small for speed)
F     = 5       # obs features
EMBED = 32      # keep small for fast tests
HEADS = 2
LAYERS = 1


# ---------------------------------------------------------------------------
# Fake batch factory
# ---------------------------------------------------------------------------

def _fake_batch(
    batch_size:  int  = B,
    n_stations:  int  = N,
    n_features:  int  = F,
    all_present: bool = True,
    n_real:      int  | None = None,
    location_encoding: str = 'unit_circle',
    rng: np.random.Generator | None = None,
) -> dict:
    """Build a synthetic batch dict matching TCDataModule collate output.

    Parameters
    ----------
    all_present : bool
        If True all obs_mask entries are True (no missing values).
        If False, randomly mask ~30% of obs to False.
    n_real : int or None
        Number of real (non-padded) stations per sample. If None, all N are real.
    location_encoding : str
        'unit_circle' — query_coords = [0, 0] sentinel.
        'domain'      — query_coords = random encoded position.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_real = n_real if n_real is not None else n_stations

    station_obs    = rng.standard_normal((batch_size, n_stations, n_features)).astype(np.float32)
    station_coords = rng.uniform(-1.0, 1.0, (batch_size, n_stations, 2)).astype(np.float32)

    station_mask           = np.zeros((batch_size, n_stations), dtype=bool)
    station_mask[:, :n_real] = True

    if all_present:
        obs_mask = np.ones((batch_size, n_stations, n_features), dtype=bool)
    else:
        obs_mask = rng.random((batch_size, n_stations, n_features)) > 0.3

    if location_encoding == 'unit_circle':
        query_coords = np.zeros((batch_size, 2), dtype=np.float32)
    else:
        query_coords = rng.uniform(-1.5, 1.5, (batch_size, 2)).astype(np.float32)

    return {
        'station_obs':    jnp.array(station_obs),
        'station_coords': jnp.array(station_coords),
        'station_mask':   jnp.array(station_mask),
        'obs_mask':       jnp.array(obs_mask),
        'query_coords':   jnp.array(query_coords),
    }


def _make_model(**kwargs) -> TCClassifier:
    defaults = dict(
        embed_dim=EMBED,
        num_heads=HEADS,
        num_layers=LAYERS,
        num_cross_layers=1,
        fourier_dim=16,
        n_obs_features=F,
    )
    defaults.update(kwargs)
    return TCClassifier(**defaults)


# ---------------------------------------------------------------------------
# SeparateKVCrossAttentionBlock
# ---------------------------------------------------------------------------

class TestSeparateKVCrossAttentionBlock:

    def _make_block(self, **kw):
        defaults = dict(embed_dim=EMBED, num_heads=HEADS)
        defaults.update(kw)
        return SeparateKVCrossAttentionBlock(**defaults)

    def test_output_shape(self):
        block = self._make_block()
        x      = jnp.ones((B, 1, EMBED))
        keys   = jnp.ones((B, N, EMBED))
        values = jnp.ones((B, N, EMBED))
        vs     = block.init(KEY, x, keys, values, train=False)
        out    = block.apply(vs, x, keys, values, train=False)
        assert out.shape == (B, 1, EMBED)

    def test_output_finite(self):
        block = self._make_block()
        rng   = np.random.default_rng(0)
        x      = jnp.array(rng.standard_normal((B, 1, EMBED)).astype(np.float32))
        keys   = jnp.array(rng.standard_normal((B, N, EMBED)).astype(np.float32))
        values = jnp.array(rng.standard_normal((B, N, EMBED)).astype(np.float32))
        vs     = block.init(KEY, x, keys, values, train=False)
        out    = block.apply(vs, x, keys, values, train=False)
        assert jnp.all(jnp.isfinite(out))

    def test_padding_mask_changes_output(self):
        block = self._make_block()
        rng   = np.random.default_rng(1)
        x      = jnp.array(rng.standard_normal((B, 1, EMBED)).astype(np.float32))
        keys   = jnp.array(rng.standard_normal((B, N, EMBED)).astype(np.float32))
        values = jnp.array(rng.standard_normal((B, N, EMBED)).astype(np.float32))
        full_mask = jnp.ones((B, N), dtype=bool)
        half_mask = full_mask.at[:, N // 2:].set(False)
        vs   = block.init(KEY, x, keys, values, mask=full_mask, train=False)
        out_full = block.apply(vs, x, keys, values, mask=full_mask, train=False)
        out_half = block.apply(vs, x, keys, values, mask=half_mask, train=False)
        assert not jnp.allclose(out_full, out_half)

    def test_return_weights(self):
        block  = self._make_block()
        x      = jnp.ones((B, 1, EMBED))
        keys   = jnp.ones((B, N, EMBED))
        values = jnp.ones((B, N, EMBED))
        vs     = block.init(KEY, x, keys, values, return_weights=True, train=False)
        out, w = block.apply(vs, x, keys, values, return_weights=True, train=False)
        assert out.shape == (B, 1, EMBED)
        assert w.shape   == (B, HEADS, 1, N)
        assert jnp.allclose(w.sum(axis=-1), jnp.ones((B, HEADS, 1)), atol=1e-5)


# ---------------------------------------------------------------------------
# TCClassifier — fake data
# ---------------------------------------------------------------------------

# All four path × encoding combinations
_CONFIGS = [
    dict(use_self_attention=True,  location_encoding='unit_circle'),
    dict(use_self_attention=True,  location_encoding='domain'),
    dict(use_self_attention=False, location_encoding='unit_circle'),
    dict(use_self_attention=False, location_encoding='domain'),
]

_CONFIG_IDS = [
    'pathA-unitcircle',
    'pathA-domain',
    'pathB-unitcircle',
    'pathB-domain',
]


@pytest.mark.parametrize('cfg', _CONFIGS, ids=_CONFIG_IDS)
class TestTCClassifierFakeData:

    def _init(self, cfg, **extra):
        model = _make_model(**cfg, **extra)
        X     = _fake_batch(location_encoding=cfg['location_encoding'])
        vs    = model.init(KEY, X, train=False)
        return model, vs, X

    def test_output_shape(self, cfg):
        model, vs, X = self._init(cfg)
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)

    def test_output_finite(self, cfg):
        model, vs, X = self._init(cfg)
        logits = model.apply(vs, X, train=False)
        assert jnp.all(jnp.isfinite(logits)), "logits contain NaN or inf"

    def test_missing_obs_changes_output(self, cfg):
        model = _make_model(**cfg)
        X_present = _fake_batch(location_encoding=cfg['location_encoding'], all_present=True)
        X_missing = _fake_batch(location_encoding=cfg['location_encoding'], all_present=False)
        vs = model.init(KEY, X_present, train=False)
        out_present = model.apply(vs, X_present, train=False)
        out_missing = model.apply(vs, X_missing, train=False)
        assert not jnp.allclose(out_present, out_missing)

    def test_padding_changes_output(self, cfg):
        model = _make_model(**cfg)
        X_full = _fake_batch(location_encoding=cfg['location_encoding'], n_real=N)
        X_half = _fake_batch(location_encoding=cfg['location_encoding'], n_real=N // 2)
        vs = model.init(KEY, X_full, train=False)
        out_full = model.apply(vs, X_full, train=False)
        out_half = model.apply(vs, X_half, train=False)
        assert not jnp.allclose(out_full, out_half)

    def test_single_head(self, cfg):
        model = _make_model(**cfg, num_heads=1)
        X     = _fake_batch(location_encoding=cfg['location_encoding'])
        vs    = model.init(KEY, X, train=False)
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)
        assert jnp.all(jnp.isfinite(logits))

    def test_multi_cross_layers(self, cfg):
        model = _make_model(**cfg, num_cross_layers=3)
        X     = _fake_batch(location_encoding=cfg['location_encoding'])
        vs    = model.init(KEY, X, train=False)
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)
        assert jnp.all(jnp.isfinite(logits))

    def test_gradient_flows(self, cfg):
        """Loss gradient w.r.t. all parameters must be non-None and finite."""
        model = _make_model(**cfg)
        X     = _fake_batch(location_encoding=cfg['location_encoding'])
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

    def test_train_vs_eval_differ_with_dropout(self, cfg):
        """With dropout, train and eval outputs should differ."""
        model = _make_model(**cfg, dropout_rate=0.5)
        X     = _fake_batch(location_encoding=cfg['location_encoding'])
        vs    = model.init({'params': KEY, 'dropout': KEY}, X, train=True)
        out_eval  = model.apply(vs, X, train=False)
        out_train = model.apply(
            vs, X, train=True,
            rngs={'dropout': jax.random.PRNGKey(99)},
        )
        assert not jnp.allclose(out_eval, out_train)


# ---------------------------------------------------------------------------
# TCClassifier — real data (skipped if files absent)
# ---------------------------------------------------------------------------

_REAL_DATA_PATHS = {
    'ibtracs_path':    'E:/sparse_obs/ibtracs/ibtracs_full.npz',
    'multi_storm_path':'E:/sparse_obs/ibtracs/ibtracs_multi_storm_times.npz',
    'insitu_obs_path': 'E:/sparse_obs/insitu-land/insitu_land_clean.npz',
    'insitu_meta_path':'E:/sparse_obs/insitu-land/insitu_land_station_meta.npz',
}

import os

_real_data_available = all(
    os.path.exists(p) for p in _REAL_DATA_PATHS.values()
)

_skip_real = pytest.mark.skipif(
    not _real_data_available or not os.environ.get('RUN_REAL_DATA_TESTS'),
    reason="Real data tests disabled. Set RUN_REAL_DATA_TESTS=1 to enable.",
)


@_skip_real
class TestTCClassifierRealData:

    @pytest.fixture(scope='class')
    def loader(self):
        from experiments.sparse_obs_cross_attn.datamodule import TCDataModule
        config = {
            **_REAL_DATA_PATHS,
            'reliability_levels':   ['always_active', 'mostly_active'],
            'obs_vars': [
                'air_pressure_at_sea_level',
                'air_temperature',
                'dew_point_temperature',
                'wind_speed',
                'wind_from_direction',
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
                'wind_speed':                [0.0, 115.0],
                'wind_from_direction':       [0.0, 360.0],
            },
        }
        dm = TCDataModule.from_config(config)
        return dm.train_loader(batch_size=8, seed=0, shuffle=False)

    def test_forward_pass_on_real_batch(self, loader):
        batch = next(iter(loader))
        X, y  = batch['X'], batch['y']

        model  = _make_model(use_self_attention=True, location_encoding='unit_circle')
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
        assert X['query_coords'].shape[-1]   == 2
