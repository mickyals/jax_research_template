"""
Tests for TCPerceiverIO representation probing (the sown ``intermediates``).

Every stage sows its OUTPUT representation so a linear probe can read what
information is present after Read, after each Processor block, after the
trailing LayerNorm (the encoder asset), and after Decode. ``sow`` is a no-op
unless the 'intermediates' collection is mutable, so the normal forward path is
unchanged — these tests assert both the captured shapes and that overhead-free
default.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from experiments.tc_perceiver_io.train.model import (
    TCPerceiverIO,
    extract_representations,
)
from experiments.tc_perceiver_io.data.sources.ibtracs import N_CLASSES

B      = 4
M      = 8
NLAT   = 6
F      = 5
HEADS  = 2
EMBED  = 32
LAYERS = 3


def _fake_X() -> dict:
    rng = np.random.default_rng(0)
    return {
        'station_obs':    jnp.array(rng.standard_normal((B, M, F)).astype(np.float32)),
        'station_coords': jnp.array(rng.uniform(-1, 1, (B, M, 2)).astype(np.float32)),
        'station_mask':   jnp.ones((B, M), dtype=bool),
        'obs_mask':       jnp.ones((B, M, F), dtype=bool),
        'query_coords':   jnp.zeros((B, 2), dtype=np.float32),
    }


def _model(n_classes=N_CLASSES, decode_mode='attention') -> tuple[TCPerceiverIO, dict, dict]:
    model = TCPerceiverIO(
        embed_dim=EMBED, num_heads=HEADS, num_latents=NLAT,
        num_process_layers=LAYERS, fourier_dim=16, n_obs_features=F,
        n_classes=n_classes, decode_mode=decode_mode,
    )
    X = _fake_X()
    variables = model.init({'params': jax.random.PRNGKey(0)}, X, train=False)
    return model, variables, X


class TestExtractRepresentations:

    def test_stage_shapes(self):
        model, variables, X = _model()
        reps = extract_representations(model, variables, X)
        assert reps['read'].shape    == (B, NLAT, EMBED)
        assert reps['encoded'].shape == (B, NLAT, EMBED)
        assert len(reps['process'])  == LAYERS
        for z in reps['process']:
            assert z.shape == (B, NLAT, EMBED)
        assert reps['decode'].shape  == (B, EMBED)

    def test_values_finite(self):
        model, variables, X = _model()
        reps = extract_representations(model, variables, X)
        for z in [reps['read'], reps['encoded'], reps['decode'], *reps['process']]:
            assert np.all(np.isfinite(np.asarray(z)))

    def test_stages_are_distinct(self):
        # Read, each Processor block, and the encoder asset should differ — the
        # network is transforming the latents, not passing them through.
        model, variables, X = _model()
        reps = extract_representations(model, variables, X)
        assert not np.allclose(reps['read'], reps['process'][0])
        assert not np.allclose(reps['process'][-1], reps['encoded'])

    def test_avgproj_decode_is_latent_mean(self):
        # avgproj decoder representation is exactly the mean over latents of the
        # encoder asset.
        model, variables, X = _model(decode_mode='avgproj')
        reps = extract_representations(model, variables, X)
        assert reps['decode'].shape == (B, EMBED)
        np.testing.assert_allclose(
            np.asarray(reps['decode']),
            np.asarray(reps['encoded']).mean(axis=1),
            rtol=1e-5, atol=1e-5,
        )

    def test_headless_has_no_decode(self):
        model, variables, X = _model(n_classes=None)
        reps = extract_representations(model, variables, X)
        assert reps['decode'] is None
        assert reps['encoded'].shape == (B, NLAT, EMBED)


class TestSowIsNoOpOnForward:

    def test_plain_apply_returns_only_output(self):
        # Without mutable=['intermediates'], apply returns just the logits —
        # sow must not leak a mutated-vars tuple or alter the output.
        model, variables, X = _model()
        logits = model.apply(variables, X, train=False)
        assert isinstance(logits, jnp.ndarray)
        assert logits.shape == (B, N_CLASSES)

    def test_return_weights_still_two_tuple(self):
        # The attention-weights path is unchanged by the added sow calls.
        model, variables, X = _model()
        out, attn = model.apply(variables, X, train=False, return_weights=True)
        assert out.shape == (B, N_CLASSES)
        assert set(attn) == {'read', 'processor', 'decoder'}
        assert attn['processor'].shape == (LAYERS, B, HEADS, NLAT, NLAT)
