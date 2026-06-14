"""
experiments/sparse_obs_cross_attn/train/model.py

TCClassifier: sparse station observation encoder for TC detection
and intensity classification.

Single TransformerEncoder over N+1 tokens (N station tokens + 1 query/CLS
token appended at position N). An asymmetric attention mask ensures stations
never attend to the query; the query attends to all stations. The
classification head reads the query token output.

Senseiver-style single projection (2023): a station token is one linear map
over the concatenation [observations; (missingness mask); Fourier(position)] --
observations and position are projected jointly, once, rather than through
separate obs/position layers that are then summed. The query/CLS token passes
through the SAME projection, with a learned stand-in occupying the obs/mask
slots and Fourier(query_coords) supplying the position slots.

The model is agnostic to the coordinate convention: whatever station_coords /
query_coords the datamodule supplies (storm-centred local x-y for unit_circle,
normalised lat/lon for domain) are Fourier-embedded and projected the same way.
For unit_circle the query sits at (0,0), so its position features are constant
and the learned obs-slot stand-in carries the query identity.

Missingness: missing observations are filled with 0 and the boolean mask is
concatenated as its own channel (missingness_indicator). The mask channel --
not the fill value -- disambiguates "observed 0" from "absent", so no learned
or constant sentinel is needed.

Coordinate conventions (set by the datamodule via data.location_encoding; the
model treats both identically -- it just projects whatever coords it receives):
  unit_circle  station_coords = [x, y] storm-centred local map
               (norm_dist*sin(bearing), norm_dist*cos(bearing)); query at (0,0).
  domain       station_coords = [norm_lat, norm_lon] absolute; query at encoded pos.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from core.nets.transformers import TransformerEncoder
from core.embeddings import GaussianFourierEmbedding

N_CLASSES = 11


def build_attention_mask(
    station_mask: jax.Array,
    full_self_attention: bool = False,
) -> jax.Array:
    """Build the N+1-token attention mask from a station mask.

    Single source of truth — used by TCClassifier's forward pass and by
    the mask-visualisation figure (plotting.plot_attention_mask).

    Convention: True = this (from, to) pair is allowed to attend;
    False = blocked (-inf before softmax).

    Default (asymmetric) pattern:
        stations → stations: True   (stations contextualise each other)
        query    → stations: True   (query reads the station network)
        query    → self:     True   (query self-attention)
        stations → query:    False  (stations never peek at the query)
    With ``full_self_attention=True`` the stations → query block also opens,
    making it complete self-attention (every token attends to every other) —
    the standard unrestricted Transformer pattern.
    A padding override always applies on top: no token may attend to a
    padding station column (station_mask False).

    Parameters
    ----------
    station_mask : jax.Array
        (B, N) bool, True = real station.
    full_self_attention : bool
        If True, stations may also attend to the query token (symmetric /
        complete self-attention). Default False = asymmetric.

    Returns
    -------
    jax.Array
        (B, 1, N+1, N+1) bool — head dim is 1 so it broadcasts across
        all attention heads.
    """
    B, N = station_mask.shape
    attn_mask = jnp.zeros((B, 1, N + 1, N + 1), dtype=bool)
    #                           [batch, head, from_token,  to_token ]
    attn_mask = attn_mask.at[:, :,  :N, :N ].set(True)  # station_rows  → station_cols
    attn_mask = attn_mask.at[:, :,   N, :N ].set(True)  # query_row     → station_cols
    attn_mask = attn_mask.at[:, :,   N,  N ].set(True)  # query_row     → query_col (self)
    if full_self_attention:
        attn_mask = attn_mask.at[:, :, :N, N].set(True)  # station_rows → query_col

    # Padding override: block any column j where station_mask[b, j] == False.
    # station_mask reshaped to (B, 1, 1, N) for broadcast over from-tokens.
    pad_col   = station_mask[:, None, None, :]           # (B, 1, 1, N)
    attn_mask = attn_mask.at[:, :, :, :N].set(
        attn_mask[:, :, :, :N] & pad_col
    )
    return attn_mask


# ---------------------------------------------------------------------------
# TCClassifier
# ---------------------------------------------------------------------------

class TCClassifier(nn.Module):
    """Sparse-observation TC classifier.

    Unified Transformer over N+1 tokens: N station tokens then one
    query/CLS token. Asymmetric attention mask separates "contextualise
    the observation network" (station self-attention) from "classify"
    (query reads all stations). Classification head reads the query token.

    Parameters
    ----------
    embed_dim : int
        Token dimensionality.
    num_heads : int
        Attention heads. Must divide embed_dim.
    num_layers : int
        Total encoder layers.
    mlp_ratio : float
        FFN hidden dim = mlp_ratio * embed_dim. Default 4.0.
    mlp_activation : str
    mlp_initializer : str
    dropout_rate : float
    attn_dropout_rate : float
    fourier_dim : int
        Gaussian Fourier embedding output dim. Must be even. Default 64.
    fourier_scale : float
    n_obs_features : int
        F, number of observation variables. Default 5.
    n_classes : int
        Output classes. Default 11.
    full_self_attention : bool
        False (default) = asymmetric mask: stations contextualise each other
        and the query reads them, but stations never attend to the query.
        True = complete self-attention over all N+1 tokens (every token attends
        to every other, modulo padding) — i.e. the standard, unrestricted
        Transformer pattern.
    missingness_indicator : bool
        True (default) = concatenate obs_mask (as 0./1.) as its own channel in
        the token-projection input, so a missing feature (filled with 0) is
        distinguishable from a real observation that equals 0. False drops the
        mask channel, reproducing the aliased behaviour where missing == 0 ==
        observed-zero.

    Notes
    -----
    Batch dict X must contain:
        station_obs    (B, N, F)  normalised obs, missing → 0 from datamodule
        station_coords (B, N, 2)  encoded station positions
        station_mask   (B, N)     bool True=real station, False=padding
        obs_mask       (B, N, F)  bool True=measurement present
        query_coords   (B, 2)     (0,0) for unit_circle; encoded pos for domain

    Output: (B, n_classes) raw logits.

    Example
    -------
    >>> model = TCClassifier(embed_dim=128, num_heads=4, num_layers=2)
    >>> variables = model.init(jax.random.PRNGKey(0), X, train=False)
    >>> logits = model.apply(variables, X, train=False)
    >>> logits.shape
    (B, 11)
    """

    embed_dim:         int
    num_heads:         int
    num_layers:        int
    mlp_ratio:         float = 4.0
    mlp_activation:    str   = 'gelu'
    mlp_initializer:   str   = 'xavier_uniform'
    dropout_rate:      float = 0.0
    attn_dropout_rate: float = 0.0
    fourier_dim:       int   = 64
    fourier_scale:     float = 1.0
    n_obs_features:    int   = 5
    n_classes:         int   = N_CLASSES
    full_self_attention: bool = False
    missingness_indicator: bool = True

    def setup(self):
        self.coord_embedding = GaussianFourierEmbedding(
            input_dim=2,
            mapping_dim=self.fourier_dim,
            scale=self.fourier_scale,
        )

        # Single shared token projection over [obs; (mask); Fourier(position)].
        # Stations and the query token both pass through this one Dense, so
        # observations and position are projected jointly, once -- not via
        # separate obs/position layers that are summed.
        self.token_proj = nn.Dense(self.embed_dim)

        # Learned query content occupying the obs (+mask) slots of the shared
        # projection input. Its width matches a station's non-position channels
        # so it feeds the same token_proj; Fourier(query_coords) supplies the
        # position slots. For unit_circle (query at (0,0)) the position features
        # are constant, so this stand-in carries the query identity.
        self._query_obs_width = (
            2 * self.n_obs_features if self.missingness_indicator
            else self.n_obs_features
        )
        self.query_obs_slots = self.param(
            'query_obs_slots',
            nn.initializers.normal(stddev=0.02),
            (self._query_obs_width,),
        )

        self.encoder = TransformerEncoder(
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
            add_pos_encoding=False,
            mlp_activation=self.mlp_activation,
            mlp_initializer=self.mlp_initializer,
        )

        self.head_norm = nn.LayerNorm()
        self.head      = nn.Dense(self.n_classes)

    def __call__(
        self,
        X:              dict,
        train:          bool = True,
        return_weights: bool = False,
    ):
        """
        Parameters
        ----------
        X : dict
            Batch dict from TCDataModule.
        train : bool
        return_weights : bool
            If True, also return the full attention matrices from EVERY
            encoder layer: shape (num_layers, B, num_heads, N+1, N+1).
            The query row is [..., N, :] (equivalently [..., -1, :]); its
            last element is the query's self-attention weight — a useful
            diagnostic (high = model relies on CLS prior; low = trusts
            stations).

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
            Logits (B, n_classes), and optionally weights
            (num_layers, B, num_heads, N+1, N+1).
        """
        station_obs    = X['station_obs']     # (B, N, F)
        station_coords = X['station_coords']  # (B, N, 2)
        station_mask   = X['station_mask']    # (B, N) bool
        obs_mask       = X['obs_mask']        # (B, N, F) bool
        query_coords   = X['query_coords']    # (B, 2)

        B, N, _ = station_obs.shape

        # 1. Missing obs -> 0. With the mask concatenated as its own channel
        # below, the projection separates "observed 0" from "absent" cleanly
        # (the mask weights carry the disambiguation), so the fill value is
        # irrelevant; 0 is convenient. No sentinel needed.
        obs_zeroed = jnp.where(obs_mask, station_obs, 0.0)              # (B, N, F)

        # 2. Fourier-embed station positions.
        # GaussianFourierEmbedding expects a flat (..., input_dim) input,
        # so merge batch and station dims before the call, then split them back.
        coord_feats = self.coord_embedding(
            station_coords.reshape(B * N, 2)          # (B*N, 2) — flat over batch x station
        ).reshape(B, N, self.fourier_dim)              # (B, N, fourier_dim)

        # 3. Station tokens — single projection over [obs; (mask); position].
        if self.missingness_indicator:
            station_input = jnp.concatenate(
                [obs_zeroed, obs_mask.astype(obs_zeroed.dtype), coord_feats],
                axis=-1,
            )                                                            # (B, N, 2F+K)
        else:
            station_input = jnp.concatenate(
                [obs_zeroed, coord_feats], axis=-1
            )                                                            # (B, N, F+K)
        station_tokens = self.token_proj(station_input)                 # (B, N, D)

        # 4. Query token — SAME projection over [learned obs-slot stand-in;
        # Fourier(query_coords)]. query_obs_slots occupies the obs(+mask) slots;
        # the position slots come from the query's own coords (constant (0,0)
        # for unit_circle, so the query token reduces to a learned vector).
        query_feats = self.coord_embedding(query_coords)                # (B, K)
        xi = jnp.broadcast_to(
            self.query_obs_slots[None, :], (B, self._query_obs_width)
        )                                                                # (B, obs_slots)
        query_input = jnp.concatenate([xi, query_feats], axis=-1)       # (B, obs_slots+K)
        query_token = self.token_proj(query_input)[:, None, :]          # (B, 1, D)

        # 5. Concatenate: station tokens then query token
        tokens = jnp.concatenate([station_tokens, query_token], axis=1) # (B, N+1, D)

        # 6. Attention mask — see build_attention_mask. Default is asymmetric
        # (stations blocked from the query); full_self_attention=True opens
        # that block for complete self-attention. Padding columns always
        # blocked for everyone.
        attn_mask = build_attention_mask(
            station_mask, self.full_self_attention)          # (B, 1, N+1, N+1)

        # 7. Encoder
        encoder_out = self.encoder(
            tokens, mask=attn_mask, train=train, return_weights=return_weights,
        )
        if return_weights:
            encoded, attn_weights = encoder_out  # (B, N+1, D), (L, B, H, N+1, N+1)
        else:
            encoded = encoder_out                # (B, N+1, D)

        # 8. Classification head reads query token at position N
        # encoded[:, N, :] — index axes: [batch, token_position=N (query slot), embed_dim]
        query_out = encoded[:, N, :]                                    # (B, D)
        logits    = self.head(self.head_norm(query_out))                 # (B, n_classes)

        # 9. Return
        if return_weights:
            # Full per-layer matrices — consumers slice what they need:
            # query row of layer l = attn_weights[l, :, :, N, :].
            return logits, attn_weights                # (L, B, H, N+1, N+1)
        return logits
