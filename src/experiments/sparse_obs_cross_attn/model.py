"""
experiments/sparse_obs_cross_attn/model.py

TCClassifier: sparse station observation encoder for TC detection
and intensity classification.

Single TransformerEncoder over N+1 tokens (N station tokens + 1 query/CLS
token appended at position N). An asymmetric attention mask ensures stations
never attend to the query; the query attends to all stations. The
classification head reads the query token output.

Two location encoding modes (config field location_encoding):
  unit_circle  Learned query content only (no position). station_coords = [norm_dist, bearing_rad].
  domain       Learned query content + pos_proj(Fourier(query_coords)). station_coords = [norm_lat, norm_lon].
               pos_proj is shared between station and query positional encoding.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from core.nets.transformers import TransformerEncoder
from core.embeddings import GaussianFourierEmbedding

N_CLASSES = 11


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
    location_encoding : str
        'unit_circle' — learned query content only; station_coords = [norm_dist, bearing_rad].
        'domain'      — learned query content + shared pos_proj(Fourier(query_coords));
                        station_coords = [norm_lat, norm_lon].
    use_learned_mask : bool
        True  = learned mask token, initialised with normal(stddev=0.02). The token
                is a trainable (F,) parameter updated by the optimizer; missing_value
                is ignored.
        False = fixed scalar sentinel: missing obs are replaced with missing_value
                at every forward pass and the value never changes.
    n_obs_features : int
        F, number of observation variables. Default 5.
    n_classes : int
        Output classes. Default 11.
    missing_value : float
        Used only when use_learned_mask=False. Substituted for every missing
        observation. Should be clearly outside the normalised obs range (e.g. -10.0
        for minmax_11 data in [-1, 1]) but not extreme enough to cause gradient
        pathology. Default -10.0.

    Notes
    -----
    Batch dict X must contain:
        station_obs    (B, N, F)  normalised obs, missing → 0 from datamodule
        station_coords (B, N, 2)  encoded station positions
        station_mask   (B, N)     bool True=real station, False=padding
        obs_mask       (B, N, F)  bool True=measurement present
        query_coords   (B, 2)     [0,0] sentinel for unit_circle; encoded pos for domain

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
    location_encoding: str   = 'unit_circle'
    use_learned_mask:  bool  = True
    n_obs_features:    int   = 5
    n_classes:         int   = N_CLASSES
    missing_value:     float = -10.0

    def setup(self):
        self.coord_embedding = GaussianFourierEmbedding(
            input_dim=2,
            mapping_dim=self.fourier_dim,
            scale=self.fourier_scale,
        )

        # obs content projection (F → D) and additive position projection (fourier_dim → D)
        # pos_proj is shared: used for station positions and (domain mode) query position
        self.station_proj = nn.Dense(self.embed_dim)
        self.pos_proj     = nn.Dense(self.embed_dim)

        if self.use_learned_mask:
            self.mask_token = self.param(
                'mask_token',
                nn.initializers.normal(stddev=0.02),
                (self.n_obs_features,),
            )

        # Learned query content — present in both modes; captures "what it means to be the query"
        # domain mode adds pos_proj(Fourier(query_coords)) on top (shared pos_proj, no extra layer)
        self.learned_query = self.param(
            'learned_query',
            nn.initializers.normal(stddev=0.02),
            (self.embed_dim,),
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
            If True, also return attention weights from the last encoder layer,
            query row only: shape (B, num_heads, N+1).
            The last element is the query's self-attention weight — a useful
            diagnostic (high = model relies on CLS prior; low = trusts stations).

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
            Logits (B, n_classes), and optionally weights (B, num_heads, N+1).
        """
        station_obs    = X['station_obs']     # (B, N, F)
        station_coords = X['station_coords']  # (B, N, 2)
        station_mask   = X['station_mask']    # (B, N) bool
        obs_mask       = X['obs_mask']        # (B, N, F) bool
        query_coords   = X['query_coords']    # (B, 2)

        B, N, _ = station_obs.shape

        # 1. Missing obs handling
        if self.use_learned_mask:
            sentinel = jnp.broadcast_to(self.mask_token, station_obs.shape)
        else:
            sentinel = jnp.full_like(station_obs, self.missing_value)
        obs_fixed = jnp.where(obs_mask, station_obs, sentinel)          # (B, N, F)

        # 2. Station tokens — obs content + additive positional encoding
        station_tokens = self.station_proj(obs_fixed)                   # (B, N, D)

        # GaussianFourierEmbedding expects a flat (..., input_dim) input,
        # so merge batch and station dims before the call, then split them back.
        # Before: (B=batch, N=stations,   2=[coord_0, coord_1])
        # After:  (B=batch, N=stations,   fourier_dim=Fourier features)
        coord_feats = self.coord_embedding(
            station_coords.reshape(B * N, 2)          # (B*N, 2) — flat over batch × station
        ).reshape(B, N, self.fourier_dim)              # (B, N, fourier_dim) — restore batch+station
        pos_embed      = self.pos_proj(coord_feats)                     # (B, N, D)
        station_tokens = station_tokens + pos_embed                     # (B, N, D)

        # 3. Query token — always learned content; domain mode adds shared pos_proj on top
        #
        # learned_query shape: (D,) — a single vector with no batch or token dims.
        # [None, None, :] inserts a batch dim and a token-sequence dim so it can be
        # broadcast to (B=batch, 1=one_query_token, D=embed_dim).
        content = jnp.broadcast_to(
            self.learned_query[None, None, :], (B, 1, self.embed_dim)
        )                                                                # (B, 1, D)
        if self.location_encoding == 'unit_circle':
            query_token = content                                        # (B, 1, D)
        else:  # domain: content + position via shared pos_proj
            query_feats = self.coord_embedding(query_coords)            # (B, fourier_dim)
            # [:, None, :] inserts a token-sequence dim so the (B, D) position
            # vector becomes (B=batch, 1=one_query_token, D=embed_dim) for addition.
            query_pos   = self.pos_proj(query_feats)[:, None, :]        # (B, 1, D)
            query_token = content + query_pos                           # (B, 1, D)

        # 4. Concatenate: station tokens then query token
        tokens = jnp.concatenate([station_tokens, query_token], axis=1) # (B, N+1, D)

        # 5. Asymmetric attention mask
        #
        # Shape: (B=batch, 1=head_broadcast, N+1=from_tokens, N+1=to_tokens)
        # Convention: True  = this (from, to) pair is allowed to attend.
        #             False = blocked (treated as -inf before softmax).
        #
        # Desired pattern:
        #   stations → stations: True   (stations contextualise each other)
        #   query    → stations: True   (query reads the station network)
        #   query    → self:     True   (query self-attention)
        #   stations → query:    False  (stations never peek at the query)
        #
        # The head dim is 1 so it broadcasts across all attention heads.
        attn_mask = jnp.zeros((B, 1, N + 1, N + 1), dtype=bool)
        #                           [batch, head, from_token,  to_token ]
        attn_mask = attn_mask.at[:, :,  :N, :N ].set(True)  # station_rows  → station_cols
        attn_mask = attn_mask.at[:, :,   N, :N ].set(True)  # query_row     → station_cols
        attn_mask = attn_mask.at[:, :,   N,  N ].set(True)  # query_row     → query_col (self)

        # Padding override: block any column j where station_mask[b, j] == False.
        # No token — station or query — should attend to a padding station.
        # station_mask: (B, N) bool, True = real station.
        # Reshape to (B=batch, 1=head, 1=from_broadcast, N=station_cols) for masking.
        pad_col   = station_mask[:, None, None, :]           # (B, 1, 1, N)
        attn_mask = attn_mask.at[:, :, :, :N].set(
            attn_mask[:, :, :, :N] & pad_col
        )

        # 6. Encoder
        encoder_out = self.encoder(
            tokens, mask=attn_mask, train=train, return_weights=return_weights,
        )
        if return_weights:
            encoded, attn_weights = encoder_out  # (B, N+1, D), (B, H, N+1, N+1)
        else:
            encoded = encoder_out                # (B, N+1, D)

        # 7. Classification head reads query token at position N
        # encoded[:, N, :] — index axes: [batch, token_position=N (query slot), embed_dim]
        query_out = encoded[:, N, :]                                    # (B, D)
        logits    = self.head(self.head_norm(query_out))                 # (B, n_classes)

        # 8. Return
        if return_weights:
            # attn_weights[:, :, N, :] slices:
            #   [batch, head, from_token=N (query row), to_token=0..N (all tokens)]
            # This is the query's attention distribution over the N stations + itself.
            return logits, attn_weights[:, :, N, :]                     # (B, H, N+1)
        return logits
