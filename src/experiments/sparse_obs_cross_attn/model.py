"""
experiments/sparse_obs_cross_attn/model.py

TCClassifier: sparse station observation encoder for TC detection
and intensity classification.

Two attention paths (config flag use_self_attention):
  Path A  TransformerEncoder over station tokens then CrossAttentionBlock.
          Stations contextualise each other before the query reads them.
  Path B  Direct cross-attention with separate K (coords) and V (obs).
          Attention weights driven by geometry; aggregated content is obs.

Two location encoding modes (config flag location_encoding):
  unit_circle  Learned query token. station_coords = [norm_dist, bearing_rad].
  domain       Fourier-encoded query_coords. station_coords = [norm_lat, norm_lon].
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from core.nets.transformers import CrossAttentionBlock, TransformerEncoder
from core.nets.mlp import MLP
from core.embeddings import GaussianFourierEmbedding

N_CLASSES = 11


# ---------------------------------------------------------------------------
# SeparateKVCrossAttentionBlock
# ---------------------------------------------------------------------------

class SeparateKVCrossAttentionBlock(nn.Module):
    """Cross-attention block where keys and values come from separate sources.

    Queries attend to keys for routing and aggregate from values.
    Useful when the feature that determines where to attend (keys) and
    the content to aggregate (values) come from different projections.

    Pre-LN ordering:
        x = x + Dropout(Attn(LN_q(x), LN_k(keys), LN_v(values)))
        x = x + Dropout(FFN(LN(x)))

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    mlp_ratio : float
    dropout_rate : float
    attn_dropout_rate : float
    mlp_activation : str
    mlp_initializer : str

    Notes
    -----
    x:      (B, T_q, embed_dim)
    keys:   (B, N, key_dim)   — any input dimensionality; w_k projects to embed_dim
    values: (B, N, val_dim)   — any input dimensionality; w_v projects to embed_dim
    Output: (B, T_q, embed_dim)

    Callers do not need to pre-project keys/values to embed_dim; w_k and w_v
    handle the projection per layer, so raw features (e.g. Fourier-encoded
    coordinates, raw obs) can be passed directly.

    mask: (B, N) bool True=attend. Applied as an additive bias to attention
    logits before softmax.
    """

    embed_dim:         int
    num_heads:         int
    mlp_ratio:         float = 4.0
    dropout_rate:      float = 0.0
    attn_dropout_rate: float = 0.0
    mlp_activation:    str   = 'gelu'
    mlp_initializer:   str   = 'xavier_uniform'

    def setup(self):
        self.norm_q  = nn.LayerNorm()
        self.norm_k  = nn.LayerNorm()
        self.norm_v  = nn.LayerNorm()
        self.norm_ff = nn.LayerNorm()
        self.w_q     = nn.Dense(self.embed_dim)
        self.w_k     = nn.Dense(self.embed_dim)
        self.w_v     = nn.Dense(self.embed_dim)
        self.w_o     = nn.Dense(self.embed_dim)
        self.ffn     = MLP(
            out_features=self.embed_dim,
            hidden_features=int(self.embed_dim * self.mlp_ratio),
            n_layers=1,
            activation=self.mlp_activation,
            initializer=self.mlp_initializer,
            dropout_rate=self.dropout_rate,
        )
        self.drop      = nn.Dropout(rate=self.dropout_rate)
        self.attn_drop = nn.Dropout(rate=self.attn_dropout_rate)

    def __call__(
        self,
        x:      jax.Array,
        keys:   jax.Array,
        values: jax.Array,
        mask:   Optional[jax.Array] = None,
        train:  bool = True,
        return_weights: bool = False,
    ):
        """
        Parameters
        ----------
        x : jax.Array
            Shape (B, T_q, embed_dim).
        keys : jax.Array
            Shape (B, N, key_dim). Any feature dimensionality; w_k projects to embed_dim.
        values : jax.Array
            Shape (B, N, val_dim). Any feature dimensionality; w_v projects to embed_dim.
        mask : jax.Array, optional
            Shape (B, N) bool. True = attend, False = block (padding).
        train : bool
        return_weights : bool
            If True returns (output, weights) where weights is
            (B, num_heads, T_q, N).

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
        """
        B, T_q, D = x.shape
        N  = keys.shape[1]
        H  = self.num_heads
        hd = D // H

        q = self.w_q(self.norm_q(x))      # (B, T_q, D)
        k = self.w_k(self.norm_k(keys))   # (B, N, D)
        v = self.w_v(self.norm_v(values)) # (B, N, D)

        # reshape to (B, H, T, hd)
        q = q.reshape(B, T_q, H, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, N,   H, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, N,   H, hd).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)  # (B, H, T_q, N)

        if mask is not None:
            # (B, N) → (B, 1, 1, N) additive bias
            scores = scores + jnp.where(mask[:, None, None, :], 0.0, -1e9)

        weights  = jax.nn.softmax(scores, axis=-1)
        attn_out = self.attn_drop(weights, deterministic=not train) @ v  # (B, H, T_q, hd)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T_q, D)
        attn_out = self.w_o(attn_out)

        x = x + self.drop(attn_out, deterministic=not train)
        x = x + self.drop(
            self.ffn(self.norm_ff(x), train=train),
            deterministic=not train,
        )

        if return_weights:
            return x, weights
        return x


# ---------------------------------------------------------------------------
# TCClassifier
# ---------------------------------------------------------------------------

class TCClassifier(nn.Module):
    """Sparse-observation TC classifier.

    Encodes N padded station observations and a query position, then
    classifies the query into one of n_classes intensity bins.

    Parameters
    ----------
    embed_dim : int
        Token dimensionality.
    num_heads : int
        Attention heads. Must divide embed_dim.
    num_layers : int
        Self-attention layers in Path A. Ignored when use_self_attention=False.
    num_cross_layers : int
        Cross-attention layers. Default 1.
    mlp_ratio : float
        FFN hidden dim = mlp_ratio * embed_dim. Default 4.0.
    mlp_activation : str
        Registered activation name. Default 'gelu'.
    mlp_initializer : str
        Registered initializer name. Default 'xavier_uniform'.
    dropout_rate : float
    attn_dropout_rate : float
    fourier_dim : int
        Gaussian Fourier embedding output dim. Must be even. Default 64.
    fourier_scale : float
        Frequency spread for Gaussian Fourier embedding. Default 1.0.
    use_self_attention : bool
        True  = Path A: TransformerEncoder over station tokens then cross-attn.
        False = Path B: direct cross-attn, separate K (coords) and V (obs).
    location_encoding : str
        'unit_circle' — learned query token; station_coords = [norm_dist, bearing_rad].
        'domain'      — Fourier-encoded query_coords; same encoding for stations.
    n_obs_features : int
        F, number of observation variables. Default 5.
    n_classes : int
        Output classes. Default 11.
        Label semantics: 0 = no storm; 1–10 = Saffir-Simpson intensity bins
        mapped as label k → SSHS category (k − 5), covering tropical
        depressions (−4, −3) through Category 5 (+5). Ensure the loss
        function and evaluation metrics use the same mapping.
    missing_value : float
        Sentinel substituted for missing obs (where obs_mask=False).
        Use a finite large-negative value to distinguish missing from
        genuine zero without causing NaN in backprop. Default -1e9.

    Notes
    -----
    Input dict X must contain:
        station_obs    (B, N, F)  normalised obs, missing → 0 from datamodule
        station_coords (B, N, 2)  encoded station positions
        station_mask   (B, N)     bool True=real station, False=padding
        obs_mask       (B, N, F)  bool True=measurement present
        query_coords   (B, 2)     [0,0] sentinel for unit_circle; encoded pos for domain

    Output: (B, n_classes) raw logits. Apply softmax cross-entropy loss externally.

    Example
    -------
    >>> model = TCClassifier(embed_dim=128, num_heads=4, num_layers=2)
    >>> variables = model.init(jax.random.PRNGKey(0), X, train=False)
    >>> logits = model.apply(variables, X, train=False)
    >>> logits.shape
    (B, 11)
    """

    embed_dim:          int
    num_heads:          int
    num_layers:         int
    num_cross_layers:   int   = 1
    mlp_ratio:          float = 4.0
    mlp_activation:     str   = 'gelu'
    mlp_initializer:    str   = 'xavier_uniform'
    dropout_rate:       float = 0.0
    attn_dropout_rate:  float = 0.0
    fourier_dim:        int   = 64
    fourier_scale:      float = 1.0
    use_self_attention: bool  = True
    location_encoding:  str   = 'unit_circle'
    n_obs_features:     int   = 5
    n_classes:          int   = N_CLASSES
    missing_value:      float = -1e9

    def setup(self):
        # Fourier coordinate embedding — shared for station coords and (domain) query
        self.coord_embedding = GaussianFourierEmbedding(
            input_dim=2,
            mapping_dim=self.fourier_dim,
            scale=self.fourier_scale,
        )

        # Query token — mode-specific
        if self.location_encoding == 'unit_circle':
            self.query_token = self.param(
                'query_token',
                nn.initializers.normal(stddev=0.02),
                (self.embed_dim,),
            )
        else:  # domain
            self.query_proj = nn.Dense(self.embed_dim)

        # Path A: fused station tokens → self-attention → cross-attention
        if self.use_self_attention:
            # projects concat(obs_fixed, coord_feats) → embed_dim
            self.station_proj = nn.Dense(self.embed_dim)
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
            self.cross_attn_blocks = [
                CrossAttentionBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    attn_dropout_rate=self.attn_dropout_rate,
                    mlp_activation=self.mlp_activation,
                    mlp_initializer=self.mlp_initializer,
                )
                for _ in range(self.num_cross_layers)
            ]

        else:
            # Path B: raw coord/obs features fed directly; w_k/w_v inside each
            # block project to embed_dim, avoiding a redundant shared projection.
            self.cross_attn_blocks = [
                SeparateKVCrossAttentionBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    attn_dropout_rate=self.attn_dropout_rate,
                    mlp_activation=self.mlp_activation,
                    mlp_initializer=self.mlp_initializer,
                )
                for _ in range(self.num_cross_layers)
            ]

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
            If True, also return cross-attention weights from the last
            cross-attention block, shape (B, num_heads, N).

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
            Logits (B, n_classes), and optionally weights (B, num_heads, N).
        """
        station_obs    = X['station_obs']     # (B, N, F)
        station_coords = X['station_coords']  # (B, N, 2)
        station_mask   = X['station_mask']    # (B, N) bool
        obs_mask       = X['obs_mask']        # (B, N, F) bool
        query_coords   = X['query_coords']    # (B, 2)

        B, N, _ = station_obs.shape

        # Replace missing obs with sentinel so model distinguishes missing from zero
        obs_fixed = jnp.where(obs_mask, station_obs, self.missing_value)  # (B, N, F)

        # Fourier-encode station coordinates: (B, N, 2) → (B, N, fourier_dim)
        coord_feats = self.coord_embedding(
            station_coords.reshape(B * N, 2)
        ).reshape(B, N, self.fourier_dim)                                  # (B, N, fourier_dim)

        # Build query token: (B, 1, embed_dim)
        if self.location_encoding == 'unit_circle':
            query = jnp.broadcast_to(
                self.query_token[None, None, :], (B, 1, self.embed_dim)
            )
        else:
            query_feats = self.coord_embedding(query_coords)               # (B, fourier_dim)
            query = self.query_proj(query_feats)[:, None, :]               # (B, 1, embed_dim)

        n_blocks = len(self.cross_attn_blocks)
        weights  = None

        if self.use_self_attention:
            # Path A: fuse obs + coords into one token per station, run SA, then cross-attn
            station_input  = jnp.concatenate([obs_fixed, coord_feats], axis=-1)
            station_tokens = self.station_proj(station_input)              # (B, N, embed_dim)

            # (B, 1, N) — required by MultiHeadAttention._build_bias which interprets
            # 3-D masks as (B, T_q, T_kv) and broadcasts across heads.
            attn_mask = station_mask[:, None, :]
            station_tokens = self.encoder(station_tokens, mask=attn_mask, train=train)

            x = query
            for i, block in enumerate(self.cross_attn_blocks):
                if return_weights and i == n_blocks - 1:
                    x, weights = block(x, context=station_tokens,
                                       mask=attn_mask, train=train,
                                       return_weights=True)
                else:
                    x = block(x, context=station_tokens,
                              mask=attn_mask, train=train)

        else:
            # Path B: separate K (geometry) and V (observations), direct cross-attn.
            # Pass raw features; w_k/w_v inside each block project to embed_dim.
            # SeparateKVCrossAttentionBlock expects (B, N) mask and handles broadcasting.
            x = query
            for i, block in enumerate(self.cross_attn_blocks):
                if return_weights and i == n_blocks - 1:
                    x, weights = block(x, keys=coord_feats, values=obs_fixed,
                                       mask=station_mask, train=train,
                                       return_weights=True)
                else:
                    x = block(x, keys=coord_feats, values=obs_fixed,
                              mask=station_mask, train=train)

        out    = x[:, 0, :]                           # (B, embed_dim)
        logits = self.head(self.head_norm(out))        # (B, n_classes)

        if return_weights:
            # Squeeze T_q=1 dim: (B, H, 1, N) → (B, H, N)
            return logits, weights[:, :, 0, :]
        return logits
