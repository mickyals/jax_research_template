"""
experiments/sparse_obs_cross_attn/model.py

TCClassifier: cross-attention encoder for sparse in-situ observations.

Architecture
------------
Three separate projection heads — Q, K, V never share weights:

  Query  (storm location)
      unit-sphere [x, y, z]  →  GaussianFourierEmbedding(input_dim=3)
                             →  Dense(embed_dim)
                             →  query_tok  (B, 1, embed_dim)

  Key    (station locations)
      [bearing_sin, bearing_cos, log_dist_norm]  →  Dense(embed_dim)
                                                 →  key_tok  (B, N, embed_dim)
      These geometric features describe WHERE each station sits relative
      to the storm.  Attention weights are computed from Q·K^T so the
      model learns to attend based on location.

  Value  (station observations)
      concat([physical_obs, obs_validity_mask_float])  →  Dense(embed_dim)
                                                       →  val_tok  (B, N, embed_dim)
      Physical obs: [pressure, temperature, dew point, wind speed]
      (NaN-zeroed; validity mask lets the model distinguish absent from zero).
      The aggregated output tells the model WHAT was measured at attended
      locations.

  n_layers × SparseObsCrossAttentionBlock
      Q = query_tok, K = key_tok, V = val_tok
      mask = station_mask  (True → real station, False → padding)
      Pre-LN, separate LayerNorm for Q, K, V sources.

  OrdinalHead  →  (B, n_classes-1) logits

forward_with_weights
--------------------
Returns (logits, attn_weights) where attn_weights is a list of
(B, num_heads, 1, N) tensors, one per cross-attention layer.
Average over heads (axis=1) for a scalar per-station importance score.
"""

from __future__ import annotations

from typing import Optional, Union

import jax
import jax.numpy as jnp
import flax.linen as nn

from core.embeddings import GaussianFourierEmbedding
from core.nets.heads import OrdinalHead
from core.nets.mlp import MLP
from datasets.joint.dataset import N_CLASSES, N_PHYSICAL_OBS, N_GEO_FEATURES, OBS_DIM


# ---------------------------------------------------------------------------
# SparseObsCrossAttentionBlock
# ---------------------------------------------------------------------------

class SparseObsCrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention block with separate K and V source tensors.

    Standard CrossAttentionBlock projects a single context tensor into both
    K and V.  This block accepts distinct K and V sources so that station
    locations drive attention routing while station observations supply the
    aggregated content — the two concerns never share a weight matrix.

    Pre-LayerNorm ordering:
        x = x + Dropout(Attn(LN_q(x), LN_k(key_src), LN_v(val_src)))
        x = x + Dropout(FFN(LN(x)))

    Residual connections are applied only to x (the query side).
    key_src and val_src are fixed encoder outputs and are not updated.

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    mlp_ratio : float  FFN hidden dim = mlp_ratio × embed_dim.
    dropout_rate : float
    use_bias : bool
    """

    embed_dim:    int
    num_heads:    int
    mlp_ratio:    float = 4.0
    dropout_rate: float = 0.0
    use_bias:     bool  = True

    def __post_init__(self):
        super().__post_init__()
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by "
                f"num_heads={self.num_heads}."
            )

    @property
    def _head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def setup(self) -> None:
        init = nn.initializers.xavier_uniform()

        self.q_proj = nn.DenseGeneral(
            features=(self.num_heads, self._head_dim), axis=-1,
            use_bias=self.use_bias, kernel_init=init,
        )
        self.k_proj = nn.DenseGeneral(
            features=(self.num_heads, self._head_dim), axis=-1,
            use_bias=self.use_bias, kernel_init=init,
        )
        self.v_proj = nn.DenseGeneral(
            features=(self.num_heads, self._head_dim), axis=-1,
            use_bias=self.use_bias, kernel_init=init,
        )
        self.out_proj = nn.DenseGeneral(
            features=self.embed_dim, axis=(-2, -1),
            use_bias=self.use_bias, kernel_init=init,
        )

        self.norm_q   = nn.LayerNorm()
        self.norm_k   = nn.LayerNorm()
        self.norm_v   = nn.LayerNorm()
        self.norm_ffn = nn.LayerNorm()

        self.ffn = MLP(
            out_features=self.embed_dim,
            hidden_features=int(self.embed_dim * self.mlp_ratio),
            n_layers=1,
            activation='gelu',
            initializer='xavier_uniform',
            dropout_rate=self.dropout_rate,
        )
        self.drop = nn.Dropout(rate=self.dropout_rate)

    def _attend(
        self,
        q:     jax.Array,
        k:     jax.Array,
        v:     jax.Array,
        bias:  Optional[jax.Array],
        train: bool,
    ) -> tuple[jax.Array, jax.Array]:
        """Core attention computation. Returns (output, weights)."""
        weights = nn.dot_product_attention_weights(
            query=q,
            key=k,
            bias=bias,
            dropout_rng=(
                self.make_rng('dropout')
                if (train and self.dropout_rate > 0) else None
            ),
            dropout_rate=self.dropout_rate if train else 0.0,
            deterministic=not train,
        )
        # weights: (B, H, T_q, T_kv); v: (B, T_kv, H, head_dim)
        out = jnp.einsum('bnij,bjnd->bind', weights, v)  # (B, T_q, H, head_dim)
        return self.out_proj(out), weights                # (B, T_q, embed_dim)

    def _bias_from_mask(self, mask: Optional[jax.Array]) -> Optional[jax.Array]:
        if mask is None:
            return None
        bias = jnp.where(mask, 0.0, -1e9).astype(jnp.float32)
        if bias.ndim == 3:
            bias = bias[:, None, :, :]   # (B, 1, T_q, T_kv)
        return bias

    def __call__(
        self,
        x:       jax.Array,
        key_src: jax.Array,
        val_src: jax.Array,
        mask:    Optional[jax.Array] = None,
        train:   bool = True,
        return_weights: bool = False,
    ) -> Union[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        Parameters
        ----------
        x       : (B, T_q,  embed_dim)  query source — storm position tokens.
        key_src : (B, T_kv, embed_dim)  key source   — station location tokens.
        val_src : (B, T_kv, embed_dim)  value source — station observation tokens.
        mask    : (B, T_q, T_kv) or (B, 1, T_kv) bool  True = attend.
        train   : bool
        return_weights : bool  If True returns (output, weights).

        Returns
        -------
        jax.Array  (B, T_q, embed_dim)
        or tuple   (output, weights) where weights is (B, num_heads, T_q, T_kv).
        """
        q = self.q_proj(self.norm_q(x))
        k = self.k_proj(self.norm_k(key_src))
        v = self.v_proj(self.norm_v(val_src))

        bias = self._bias_from_mask(mask)
        attn_out, weights = self._attend(q, k, v, bias, train)

        x = x + self.drop(attn_out, deterministic=not train)
        x = x + self.drop(
            self.ffn(self.norm_ffn(x), train=train),
            deterministic=not train,
        )

        if return_weights:
            # Return clean weights (no dropout) for diagnostics
            if train and self.dropout_rate > 0:
                _, clean_weights = self._attend(q, k, v, bias, train=False)
            else:
                clean_weights = weights
            return x, clean_weights

        return x


# ---------------------------------------------------------------------------
# TCClassifier
# ---------------------------------------------------------------------------

class TCClassifier(nn.Module):
    """Cross-attention TC intensity ordinal classifier.

    Q from storm location, K from station locations, V from station
    observations — three separate projection heads, no shared weights.

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    n_layers  : int
    mlp_ratio : float
    dropout_rate : float
    n_classes : int
    pos_map_dim : int   GaussianFourierEmbedding output dim (must be even).
    pos_scale   : float Gaussian frequency matrix std dev.

    Batch input keys (batch['X'])
    ------------------------------
    query_coords : (B, 3)              unit-sphere storm position
    station_obs  : (B, N, OBS_DIM)    [physical(4) | geo(3)], NaN→0
    station_mask : (B, N) bool         True = real station
    obs_mask     : (B, N, OBS_DIM) bool  True = valid measurement
    """

    embed_dim:    int   = 128
    num_heads:    int   = 4
    n_layers:     int   = 2
    mlp_ratio:    float = 4.0
    dropout_rate: float = 0.1
    n_classes:    int   = N_CLASSES
    pos_map_dim:  int   = 64
    pos_scale:    float = 10.0

    def setup(self) -> None:
        # --- Query: storm location ---
        self.pos_encoder = GaussianFourierEmbedding(
            input_dim=3, mapping_dim=self.pos_map_dim, scale=self.pos_scale,
        )
        self.query_proj = nn.Dense(self.embed_dim)

        # --- Key: station locations [bearing_sin, bearing_cos, log_dist_norm] ---
        # Input dim = N_GEO_FEATURES = 3 (always valid, no mask needed)
        self.station_loc_proj = nn.Dense(self.embed_dim)

        # --- Value: physical observations + validity mask ---
        # Input dim = 2 * N_PHYSICAL_OBS = 8  (4 obs values + 4 validity floats)
        self.station_obs_proj = nn.Dense(self.embed_dim)

        # --- Cross-attention stack ---
        self.cross_attn = [
            SparseObsCrossAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.n_layers)
        ]

        self.head = OrdinalHead(n_classes=self.n_classes)

    def _project(self, x: dict) -> tuple:
        """Project all three input streams into embed_dim tokens."""
        # Query: storm position
        query_tok = self.query_proj(
            self.pos_encoder(x['query_coords'])
        )[:, None, :]                                              # (B, 1, E)

        # Key: station geometric features (last N_GEO_FEATURES columns of station_obs)
        station_locs = x['station_obs'][:, :, N_PHYSICAL_OBS:]    # (B, N, 3)
        key_tok = self.station_loc_proj(station_locs)             # (B, N, E)

        # Value: physical obs + validity mask (first N_PHYSICAL_OBS columns)
        phys_obs  = x['station_obs'][:, :, :N_PHYSICAL_OBS]       # (B, N, 4)
        phys_mask = x['obs_mask'][:, :, :N_PHYSICAL_OBS].astype(jnp.float32)
        val_tok   = self.station_obs_proj(
            jnp.concatenate([phys_obs, phys_mask], axis=-1)
        )                                                          # (B, N, E)

        # Mask: True = attend to this station token
        cross_mask = x['station_mask'][:, None, :]                # (B, 1, N)

        return query_tok, key_tok, val_tok, cross_mask

    def __call__(self, x: dict, train: bool = True) -> jax.Array:
        """
        Parameters
        ----------
        x : dict  batch['X'] from JointDataModule.
        train : bool

        Returns
        -------
        jax.Array  (B, n_classes - 1)  raw ordinal logits.
        """
        q, k, v, mask = self._project(x)
        for block in self.cross_attn:
            q = block(q, key_src=k, val_src=v, mask=mask, train=train)
        return self.head(q[:, 0, :])

    def forward_with_weights(
        self,
        x: dict,
    ) -> tuple[jax.Array, list[jax.Array]]:
        """Forward pass returning logits and per-layer attention weights.

        Parameters
        ----------
        x : dict  same format as __call__.

        Returns
        -------
        logits       : (B, n_classes - 1)
        attn_weights : list of (B, num_heads, 1, N), one per layer.
            Average over heads (axis=1) for a scalar per-station score.
            Final layer weights are most interpretable for visualisation.
        """
        q, k, v, mask = self._project(x)
        all_weights = []
        for block in self.cross_attn:
            q, w = block(
                q, key_src=k, val_src=v, mask=mask,
                train=False, return_weights=True,
            )
            all_weights.append(w)
        return self.head(q[:, 0, :]), all_weights
