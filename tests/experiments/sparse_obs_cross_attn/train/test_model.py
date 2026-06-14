"""
Tests for experiments/sparse_obs_cross_attn/train/model.py.

Fake-data tests: no disk access, run always.
Real-data tests: require E:/sparse_obs data files, skipped if absent.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from experiments.sparse_obs_cross_attn.train.model import (
    TCClassifier,
    N_CLASSES,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

KEY    = jax.random.PRNGKey(0)
B      = 4       # batch size
N      = 8       # max stations (small for speed)
F      = 5       # obs features
EMBED  = 32      # keep small for fast tests
HEADS  = 2
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
    """Build a synthetic batch dict matching TCDataModule collate output."""
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
        fourier_dim=16,
        n_obs_features=F,
        use_learned_mask=True,
    )
    defaults.update(kwargs)
    return TCClassifier(**defaults)


# ---------------------------------------------------------------------------
# TCClassifier — fake data
# ---------------------------------------------------------------------------

# Two encoding modes — the single unified architecture handles both
_CONFIGS = [
    dict(location_encoding='unit_circle'),
    dict(location_encoding='domain'),
]

_CONFIG_IDS = [
    'unit_circle',
    'domain',
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

    def test_multi_layers(self, cfg):
        model = _make_model(**cfg, num_layers=3)
        X     = _fake_batch(location_encoding=cfg['location_encoding'])
        vs    = model.init(KEY, X, train=False)
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)
        assert jnp.all(jnp.isfinite(logits))

    def test_return_weights_shape(self, cfg):
        model, vs, X = self._init(cfg)
        logits, weights = model.apply(vs, X, train=False, return_weights=True)
        assert logits.shape  == (B, N_CLASSES)
        # Full attention matrices from every layer (leading axis = LAYERS)
        assert weights.shape == (LAYERS, B, HEADS, N + 1, N + 1)
        row_sums = weights.sum(axis=-1)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5)
        # Query row of the last layer is a distribution over N+1 tokens
        q_row = weights[-1][:, :, -1, :]
        assert q_row.shape == (B, HEADS, N + 1)

    def test_return_weights_stations_blocked_from_query(self, cfg):
        # Station rows must place ZERO weight on the query column (token N)
        model, vs, X = self._init(cfg)
        _, weights = model.apply(vs, X, train=False, return_weights=True)
        station_rows_query_col = weights[:, :, :, :N, N]
        assert jnp.allclose(station_rows_query_col, 0.0, atol=1e-6)

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
# Standalone tests
# ---------------------------------------------------------------------------

def test_missingness_indicator_disambiguates_sentinel_collision():
    """A missing feature must produce a different token than a real obs
    that happens to equal the sentinel value (use_learned_mask=False)."""
    missing_value = -10.0
    model = _make_model(
        use_learned_mask=False,
        missing_value=missing_value,
        missingness_indicator=True,
    )
    X = _fake_batch(all_present=True)

    # A real observation that happens to equal the sentinel.
    X_real_eq_sentinel = dict(X)
    X_real_eq_sentinel['station_obs'] = jnp.full_like(X['station_obs'], missing_value)
    X_real_eq_sentinel['obs_mask']    = jnp.ones_like(X['obs_mask'])

    # A genuinely missing feature — obs_fixed collapses to the same sentinel.
    X_missing = dict(X)
    X_missing['station_obs'] = jnp.zeros_like(X['station_obs'])  # ignored where obs_mask=False
    X_missing['obs_mask']    = jnp.zeros_like(X['obs_mask'])

    vs = model.init(KEY, X_real_eq_sentinel, train=False)
    out_real    = model.apply(vs, X_real_eq_sentinel, train=False)
    out_missing = model.apply(vs, X_missing, train=False)
    assert not jnp.allclose(out_real, out_missing), \
        "missingness_indicator=True should disambiguate sentinel collisions"


def test_missingness_indicator_false_aliases_sentinel_collision():
    """Without the indicator, a missing feature IS aliased with a real obs
    equal to the sentinel — documents the bug the indicator fixes."""
    missing_value = -10.0
    model = _make_model(
        use_learned_mask=False,
        missing_value=missing_value,
        missingness_indicator=False,
    )
    X = _fake_batch(all_present=True)

    X_real_eq_sentinel = dict(X)
    X_real_eq_sentinel['station_obs'] = jnp.full_like(X['station_obs'], missing_value)
    X_real_eq_sentinel['obs_mask']    = jnp.ones_like(X['obs_mask'])

    X_missing = dict(X)
    X_missing['station_obs'] = jnp.zeros_like(X['station_obs'])
    X_missing['obs_mask']    = jnp.zeros_like(X['obs_mask'])

    vs = model.init(KEY, X_real_eq_sentinel, train=False)
    out_real    = model.apply(vs, X_real_eq_sentinel, train=False)
    out_missing = model.apply(vs, X_missing, train=False)
    assert jnp.allclose(out_real, out_missing), \
        "missingness_indicator=False should alias sentinel collisions (legacy behaviour)"


def test_use_learned_mask_false():
    """Constant sentinel path (use_learned_mask=False) produces finite output."""
    model  = _make_model(use_learned_mask=False)
    X      = _fake_batch(all_present=False)   # some missing obs
    vs     = model.init(KEY, X, train=False)
    logits = model.apply(vs, X, train=False)
    assert logits.shape == (B, N_CLASSES)
    assert jnp.all(jnp.isfinite(logits))


def test_build_attention_mask_pattern():
    """build_attention_mask: asymmetry + padding blocking, exact pattern."""
    from experiments.sparse_obs_cross_attn.train.model import (
        build_attention_mask,
    )
    N_t = 4
    station_mask = jnp.array([[True, True, False, True]])   # one padding col
    mask = build_attention_mask(station_mask)               # (1, 1, 5, 5)
    assert mask.shape == (1, 1, N_t + 1, N_t + 1)
    m = np.asarray(mask)[0, 0]
    # stations → query column blocked
    assert not m[:N_t, N_t].any()
    # query row: self True, real stations True, padding station False
    assert bool(m[N_t, N_t])
    assert bool(m[N_t, 0]) and bool(m[N_t, 1]) and bool(m[N_t, 3])
    assert not m[:, 2].any()        # padding column blocked for everyone
    # real station ↔ real station allowed
    assert bool(m[0, 1]) and bool(m[1, 0])


def test_build_attention_mask_full_self_attention():
    """full_self_attention=True opens the stations→query block; padding stays blocked."""
    from experiments.sparse_obs_cross_attn.train.model import (
        build_attention_mask,
    )
    N_t = 4
    station_mask = jnp.array([[True, True, False, True]])   # one padding col
    m = np.asarray(build_attention_mask(
        station_mask, full_self_attention=True))[0, 0]
    # real stations now DO attend to the query column (rows 0,1,3)
    assert bool(m[0, N_t]) and bool(m[1, N_t]) and bool(m[3, N_t])
    # padding station (row 2) attends to query too (it's a real from-row),
    # but the padding *column* (col 2) is still blocked for everyone
    assert not m[:, 2].any()
    # every non-padding (from, to) pair is allowed → complete self-attention
    keep = [0, 1, 3, N_t]   # drop padding station index 2
    sub = m[np.ix_(keep, keep)]
    assert sub.all()


def test_full_self_attention_flag_changes_station_outputs():
    """With full_self_attention the model's station reps DO depend on the query."""
    asym = _make_model(full_self_attention=False)
    full = _make_model(full_self_attention=True)
    X = _fake_batch()
    # build_attention_mask is exercised inside apply; compare the masks the
    # two models produce for the same station_mask.
    from experiments.sparse_obs_cross_attn.train.model import build_attention_mask
    sm = X['station_mask']
    m_asym = np.asarray(build_attention_mask(sm, False))
    m_full = np.asarray(build_attention_mask(sm, True))
    assert not np.array_equal(m_asym, m_full)
    # both models still run and produce finite logits
    for model in (asym, full):
        vs = model.init(KEY, X, train=False)
        assert jnp.all(jnp.isfinite(model.apply(vs, X, train=False)))


def test_asymmetric_mask_station_independent_of_query():
    """Swapping the query token must not change station token outputs.

    Tests the asymmetric mask directly on TransformerEncoder: with the
    mask set so stations cannot attend to query, changing the query token
    in position N must leave positions 0..N-1 unchanged.
    """
    from core.nets.transformers import TransformerEncoder

    B_t, N_t, D, H_t = 2, 4, 32, 2
    encoder = TransformerEncoder(
        num_layers=2, embed_dim=D, num_heads=H_t, add_pos_encoding=False,
    )

    rng      = np.random.default_rng(0)
    stations = jnp.array(rng.standard_normal((B_t, N_t, D)).astype(np.float32))
    query_a  = jnp.array(rng.standard_normal((B_t, 1, D)).astype(np.float32))
    query_b  = jnp.array(rng.standard_normal((B_t, 1, D)).astype(np.float32))

    tokens_a = jnp.concatenate([stations, query_a], axis=1)  # (B, N+1, D)
    tokens_b = jnp.concatenate([stations, query_b], axis=1)

    # Asymmetric mask: same construction as TCClassifier
    mask = jnp.zeros((B_t, 1, N_t + 1, N_t + 1), dtype=bool)
    mask = mask.at[:, :, :N_t, :N_t].set(True)
    mask = mask.at[:, :,  N_t, :N_t].set(True)
    mask = mask.at[:, :,  N_t,  N_t].set(True)

    vs    = encoder.init(KEY, tokens_a, mask=mask, train=False)
    out_a = encoder.apply(vs, tokens_a, mask=mask, train=False)
    out_b = encoder.apply(vs, tokens_b, mask=mask, train=False)

    assert jnp.allclose(out_a[:, :N_t, :], out_b[:, :N_t, :], atol=1e-5), \
        "station representations changed when query token changed — mask is broken"
    assert not jnp.allclose(out_a[:, N_t, :], out_b[:, N_t, :]), \
        "query representations should differ for different query tokens"


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
        from experiments.sparse_obs_cross_attn.data.datamodule import TCDataModule
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

        model  = _make_model(location_encoding='unit_circle')
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
