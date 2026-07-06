"""
experiments/cyclone_jax/models/features.py

The model-side front end: named device batch X -> (B, N, F) tokens.

Packing is the ONLY place named fields become positional. One defined
order — TOKEN_FIELDS blocks first, then the position block — so every
model sees the same layout. Position (lat, lon) enters exactly once,
through the encoding mode (configs/models/*.yaml, encoding:):

    concat    tokens = [packed; pos]        pos = gamma(lat, lon), or the
                                            raw coords when embedding: null
    additive  tokens = packed + Dense(pos)  gamma projected to the packed
                                            width and added

gamma is any registered core embedding (core/embeddings — Fourier,
spherical, ...); embeddings that take (lat, lon) separately are wrapped
automatically. SIREN/FINER use concat with raw coords — sine IS their
encoding, so no gamma.

No pooling, no masking, no set logic here — models own everything after
tokens exist (the ARCHITECTURE ruling: flat concat, not Deep Sets).
"""

from __future__ import annotations

import inspect
from typing import Optional

import jax.numpy as jnp
import flax.linen as nn

from core.embeddings import EMBEDDINGS, LatLonEmbeddingWrapper, get_embedding

# The defined packing order. Scalar fields contribute one column each;
# obs/missing contribute one per channel (inputs.CHANNEL_ORDER layout).
TOKEN_FIELDS = ('level', 'time', 'id', 'obs', 'missing')
COORD_FIELDS = ('lat', 'lon')


def pack_tokens(X) -> jnp.ndarray:
    """Named batch X -> (B, N, F) float32 tokens, TOKEN_FIELDS order."""
    cols = []
    for f in TOKEN_FIELDS:
        v = jnp.asarray(X[f], jnp.float32)
        cols.append(v[..., None] if v.ndim == 2 else v)
    return jnp.concatenate(cols, axis=-1)


def pack_coords(X) -> jnp.ndarray:
    """Named batch X -> (B, N, 2) float32 positions, COORD_FIELDS order."""
    return jnp.stack([jnp.asarray(X[f], jnp.float32)
                      for f in COORD_FIELDS], axis=-1)


def flatten(tokens: jnp.ndarray) -> jnp.ndarray:
    """(B, N, F) tokens -> (B, N*F) flat vector (slot-sensitive: position
    in the vector = station slot; fine for the memorisation gate)."""
    return tokens.reshape(tokens.shape[0], -1)


class FeatureEncoder(nn.Module):
    """X dict -> (B, N, F') tokens with position folded in per `mode`.

    Parameters
    ----------
    mode : str
        'concat' — append the position block to the packed tokens.
        'additive' — Dense-project the position block to the packed width
        and add (the projection is this module's only parameter; concat
        mode is parameter-free).
    embedding : nn.Module, optional
        gamma(lat, lon) from core/embeddings. None = raw coords.
    """
    mode:      str = 'concat'
    embedding: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, X) -> jnp.ndarray:
        if self.mode not in ('concat', 'additive'):
            raise ValueError(f"encoding mode must be 'concat' or 'additive', "
                             f"got {self.mode!r}")
        tokens = pack_tokens(X)
        pos = pack_coords(X)
        if self.embedding is not None:
            B, N, _ = pos.shape
            pos = self.embedding(pos.reshape(B * N, 2)).reshape(B, N, -1)
        if self.mode == 'concat':
            return jnp.concatenate([tokens, pos], axis=-1)
        return tokens + nn.Dense(tokens.shape[-1])(pos)


def _takes_latlon(module) -> bool:
    """True for embeddings with signature (lat, lon) — need the wrapper."""
    params = list(inspect.signature(type(module).__call__).parameters)
    return params[1:3] == ['lat', 'lon']


def build_encoder(encoding: dict | None = None, seed: int = 0) -> FeatureEncoder:
    """encoding config block -> FeatureEncoder.

    Keys: mode ('concat' default), embedding (core registry name or null =
    raw coords), embedding_kwargs. None/empty block = concat + raw coords
    (the SIREN/FINER front end).

    ``seed`` is the RUN seed (trainer.seed — the ONE-seed principle): it
    becomes the embedding's seed when the embedding takes one (e.g. the
    Gaussian Fourier B matrix) and the yaml does not pin
    ``embedding_kwargs.seed`` (an explicit value wins — controlled
    B-matrix ablations across runs).
    """
    encoding = encoding or {}
    emb = None
    name = encoding.get('embedding')
    if name:
        kwargs = dict(encoding.get('embedding_kwargs') or {})
        if (name in EMBEDDINGS and 'seed' not in kwargs
                and 'seed' in inspect.signature(EMBEDDINGS[name]).parameters):
            kwargs['seed'] = seed
        emb = get_embedding(name, **kwargs)
        if _takes_latlon(emb):
            emb = LatLonEmbeddingWrapper(embedding=emb)
    return FeatureEncoder(mode=encoding.get('mode', 'concat'), embedding=emb)
