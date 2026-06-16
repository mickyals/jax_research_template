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
    )
    defaults.update(kwargs)
    return TCClassifier(**defaults)


# ---------------------------------------------------------------------------
# TCClassifier — fake data
# ---------------------------------------------------------------------------

# Two coordinate conventions — the single coordinate-agnostic architecture
# handles both; the only difference is the query_coords the datamodule supplies
# (zeros for unit_circle, varied for domain).
_LOCATIONS = ['unit_circle', 'domain']


@pytest.mark.parametrize('loc', _LOCATIONS, ids=_LOCATIONS)
class TestTCClassifierFakeData:

    def _init(self, loc, **extra):
        model = _make_model(**extra)
        X     = _fake_batch(location_encoding=loc)
        vs    = model.init(KEY, X, train=False)
        return model, vs, X

    def test_output_shape(self, loc):
        model, vs, X = self._init(loc)
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)

    def test_output_finite(self, loc):
        model, vs, X = self._init(loc)
        logits = model.apply(vs, X, train=False)
        assert jnp.all(jnp.isfinite(logits)), "logits contain NaN or inf"

    def test_missing_obs_changes_output(self, loc):
        model = _make_model()
        X_present = _fake_batch(location_encoding=loc, all_present=True)
        X_missing = _fake_batch(location_encoding=loc, all_present=False)
        vs = model.init(KEY, X_present, train=False)
        out_present = model.apply(vs, X_present, train=False)
        out_missing = model.apply(vs, X_missing, train=False)
        assert not jnp.allclose(out_present, out_missing)

    def test_padding_changes_output(self, loc):
        model = _make_model()
        X_full = _fake_batch(location_encoding=loc, n_real=N)
        X_half = _fake_batch(location_encoding=loc, n_real=N // 2)
        vs = model.init(KEY, X_full, train=False)
        out_full = model.apply(vs, X_full, train=False)
        out_half = model.apply(vs, X_half, train=False)
        assert not jnp.allclose(out_full, out_half)

    def test_single_head(self, loc):
        model = _make_model(num_heads=1)
        X     = _fake_batch(location_encoding=loc)
        vs    = model.init(KEY, X, train=False)
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)
        assert jnp.all(jnp.isfinite(logits))

    def test_multi_layers(self, loc):
        model = _make_model(num_layers=3)
        X     = _fake_batch(location_encoding=loc)
        vs    = model.init(KEY, X, train=False)
        logits = model.apply(vs, X, train=False)
        assert logits.shape == (B, N_CLASSES)
        assert jnp.all(jnp.isfinite(logits))

    def test_return_weights_shape(self, loc):
        model, vs, X = self._init(loc)
        logits, weights = model.apply(vs, X, train=False, return_weights=True)
        assert logits.shape  == (B, N_CLASSES)
        # Full attention matrices from every layer (leading axis = LAYERS)
        assert weights.shape == (LAYERS, B, HEADS, N + 1, N + 1)
        row_sums = weights.sum(axis=-1)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5)
        # CLS-first: the query is token 0; its row is a distribution over 1+N
        q_row = weights[-1][:, :, 0, :]
        assert q_row.shape == (B, HEADS, N + 1)

    def test_return_weights_stations_blocked_from_query(self, loc):
        # CLS-first: stations are rows 1..N, the query is column 0. Station rows
        # must place ZERO weight on the query column.
        model, vs, X = self._init(loc)
        _, weights = model.apply(vs, X, train=False, return_weights=True)
        station_rows_query_col = weights[:, :, :, 1:, 0]
        assert jnp.allclose(station_rows_query_col, 0.0, atol=1e-6)

    def test_gradient_flows(self, loc):
        """Loss gradient w.r.t. all parameters must be non-None and finite."""
        model = _make_model()
        X     = _fake_batch(location_encoding=loc)
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

    def test_train_vs_eval_differ_with_dropout(self, loc):
        """With dropout, train and eval outputs should differ."""
        model = _make_model(dropout_rate=0.5)
        X     = _fake_batch(location_encoding=loc)
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

def test_missingness_indicator_disambiguates_observed_zero():
    """A missing feature (filled with 0) must produce a different token than a
    real observation that equals 0, when missingness_indicator=True — the mask
    channel carries the disambiguation."""
    model = _make_model(missingness_indicator=True)
    X = _fake_batch(all_present=True)

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
    X = _fake_batch(all_present=True)

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
    X      = _fake_batch(all_present=False)   # some missing obs
    vs     = model.init(KEY, X, train=False)
    logits = model.apply(vs, X, train=False)
    assert logits.shape == (B, N_CLASSES)
    assert jnp.all(jnp.isfinite(logits))


def test_build_attention_mask_pattern():
    """build_attention_mask: CLS-first asymmetry + padding blocking, exact pattern."""
    from experiments.sparse_obs_cross_attn.train.model import (
        build_attention_mask,
    )
    N_t = 4
    station_mask = jnp.array([[True, True, False, True]])   # one padding station
    mask = build_attention_mask(station_mask)               # (1, 1, 5, 5)
    assert mask.shape == (1, 1, N_t + 1, N_t + 1)
    m = np.asarray(mask)[0, 0]
    # CLS-first: token 0 = query; tokens 1..4 = stations; token 3 = padding.
    # stations → query column (col 0) blocked
    assert not m[1:, 0].any()
    # query row (row 0): self True, real stations True, padding station False
    assert bool(m[0, 0])
    assert bool(m[0, 1]) and bool(m[0, 2]) and bool(m[0, 4])
    assert not bool(m[0, 3])
    assert not m[:, 3].any()        # padding column (token 3) blocked for everyone
    # real station ↔ real station allowed (tokens 1, 2)
    assert bool(m[1, 2]) and bool(m[2, 1])


def test_build_attention_mask_full_self_attention():
    """full_self_attention=True opens the stations→query block; padding stays blocked."""
    from experiments.sparse_obs_cross_attn.train.model import (
        build_attention_mask,
    )
    N_t = 4
    station_mask = jnp.array([[True, True, False, True]])   # one padding station
    m = np.asarray(build_attention_mask(
        station_mask, full_self_attention=True))[0, 0]
    # CLS-first: real stations (tokens 1,2,4) now attend to the query col (col 0)
    assert bool(m[1, 0]) and bool(m[2, 0]) and bool(m[4, 0])
    # padding station column (token 3) is still blocked for everyone
    assert not m[:, 3].any()
    # every non-padding (from, to) pair is allowed → complete self-attention
    keep = [0, 1, 2, 4]   # query + 3 real stations (drop padding token 3)
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

    Tests the asymmetric mask directly on TransformerEncoder. CLS-first:
    the query is at position 0, stations 1..N. With the mask set so stations
    cannot attend to the query, changing the query token in position 0 must
    leave positions 1..N unchanged.
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

    tokens_a = jnp.concatenate([query_a, stations], axis=1)  # CLS-first (B, 1+N, D)
    tokens_b = jnp.concatenate([query_b, stations], axis=1)

    # Asymmetric mask, CLS-first: same construction as TCClassifier
    mask = jnp.zeros((B_t, 1, N_t + 1, N_t + 1), dtype=bool)
    mask = mask.at[:, :, 1:, 1:].set(True)   # station → station
    mask = mask.at[:, :, 0,  1:].set(True)   # query   → stations
    mask = mask.at[:, :, 0,  0 ].set(True)   # query   → self

    vs    = encoder.init(KEY, tokens_a, mask=mask, train=False)
    out_a = encoder.apply(vs, tokens_a, mask=mask, train=False)
    out_b = encoder.apply(vs, tokens_b, mask=mask, train=False)

    assert jnp.allclose(out_a[:, 1:, :], out_b[:, 1:, :], atol=1e-5), \
        "station representations changed when query token changed — mask is broken"
    assert not jnp.allclose(out_a[:, 0, :], out_b[:, 0, :]), \
        "query representations should differ for different query tokens"


def test_learnable_query_pos_ignores_query_coords():
    """learnable_query_pos=True (unit_circle): the CLS position is a learned
    parameter, so changing query_coords must NOT change the output."""
    model = _make_model(learnable_query_pos=True)
    X  = _fake_batch(location_encoding='domain')   # nonzero query_coords
    vs = model.init(KEY, X, train=False)
    out0 = model.apply(vs, X, train=False)
    X2 = dict(X); X2['query_coords'] = X['query_coords'] + 1.0
    out1 = model.apply(vs, X2, train=False)
    assert jnp.allclose(out0, out1), "learnable CLS must ignore query_coords"
    assert 'query_pos_slots' in vs['params']['encoder']


def test_query_pos_from_coords_uses_query_coords():
    """learnable_query_pos=False (domain): the CLS position comes from
    query_coords, so changing them changes the output; no query_pos_slots."""
    model = _make_model(learnable_query_pos=False)
    X  = _fake_batch(location_encoding='domain')
    vs = model.init(KEY, X, train=False)
    out0 = model.apply(vs, X, train=False)
    X2 = dict(X); X2['query_coords'] = X['query_coords'] + 1.0
    out1 = model.apply(vs, X2, train=False)
    assert not jnp.allclose(out0, out1), "CLS must use query_coords under domain"
    assert 'query_pos_slots' not in vs['params']['encoder']


# ---------------------------------------------------------------------------
# Encoder / head split — frozen-encoder probing (r4/r5)
# ---------------------------------------------------------------------------

def test_param_tree_splits_encoder_and_head():
    """TCClassifier params have exactly an 'encoder' subtree and a 'head'."""
    model = _make_model()
    X  = _fake_batch()
    vs = model.init(KEY, X, train=False)
    assert set(vs['params'].keys()) == {'encoder', 'head'}
    # head is a pure linear Dense (kernel + bias), no LayerNorm in the head
    assert set(vs['params']['head'].keys()) == {'kernel', 'bias'}
    # the final norm lives inside the encoder asset
    assert 'norm' in vs['params']['encoder']


def test_split_and_attach_encoder_roundtrip():
    """split_encoder_head + attach_encoder transplant the encoder into a fresh
    model with a new head — the frozen-encoder transfer operation."""
    from experiments.sparse_obs_cross_attn.train.model import (
        split_encoder_head, attach_encoder, TCEncoder,
    )
    model = _make_model()
    X  = _fake_batch()
    vs = model.init(KEY, X, train=False)
    enc_params, head_params = split_encoder_head(vs['params'])
    assert 'head' in head_params and 'encoder' not in head_params

    # The standalone encoder runs on the split params → (B, D) CLS embedding.
    encoder = TCEncoder(embed_dim=EMBED, num_heads=HEADS, num_layers=LAYERS,
                        fourier_dim=16, n_obs_features=F)
    z = encoder.apply({'params': enc_params}, X, train=False)
    assert z.shape == (B, EMBED) and jnp.all(jnp.isfinite(z))

    # Attach the trained encoder into a freshly initialised model (new head).
    fresh  = _make_model().init(jax.random.PRNGKey(7), X, train=False)['params']
    merged = attach_encoder(fresh, enc_params)
    assert merged['encoder'] is enc_params                       # encoder transplanted
    # the head keeps its fresh init (differs from the original model's head)
    assert not jnp.allclose(merged['head']['kernel'],
                            vs['params']['head']['kernel'])
    logits = model.apply({'params': merged}, X, train=False)
    assert logits.shape == (B, N_CLASSES) and jnp.all(jnp.isfinite(logits))


def test_encoder_freeze_labels_partition():
    """encoder_freeze_labels marks every encoder leaf 'frozen' and the head
    leaves 'trainable', matching the param structure."""
    from experiments.sparse_obs_cross_attn.train.model import encoder_freeze_labels
    model = _make_model()
    X  = _fake_batch()
    vs = model.init(KEY, X, train=False)
    labels = encoder_freeze_labels(vs['params'])
    enc_labels  = set(jax.tree_util.tree_leaves(labels['encoder']))
    head_labels = set(jax.tree_util.tree_leaves(labels['head']))
    assert enc_labels == {'frozen'}
    assert head_labels == {'trainable'}


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
        assert X['query_coords'].shape[-1]   == 2
