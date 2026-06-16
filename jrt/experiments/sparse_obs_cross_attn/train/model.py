"""
experiments/sparse_obs_cross_attn/train/model.py

TCClassifier: sparse station observation encoder for TC detection
and intensity classification.

Single TransformerEncoder over 1+N tokens, CLS-first (ViT-style): one query/CLS
token at position 0 then N station tokens. An asymmetric attention mask ensures
stations never attend to the query; the query attends to all stations. The
classification head reads the query/CLS token output (position 0).

Senseiver-style single projection (2023): a station token is one linear map
over the concatenation [observations; (missingness mask); Fourier(position)] --
observations and position are projected jointly, once, rather than through
separate obs/position layers that are then summed. The query/CLS token passes
through the SAME projection, with a learned stand-in occupying the obs/mask
slots and its position slots either learned (learnable_query_pos, unit_circle)
or supplied by Fourier(query_coords) (domain).

Coordinate handling: whatever station_coords the datamodule supplies (storm-
centred local x-y for unit_circle, normalised lat/lon for domain) are Fourier-
embedded and projected the same way. The CLS token differs by encoding: under
unit_circle the query sits at (0,0), so its position slots are a fully learnable
vector (Fourier((0,0)) is a constant a learned vector subsumes); under domain
the CLS uses Fourier(query_coords) so it carries the absolute query location.

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

# Default label-space size. Canonical label names live in
# data/sources/ibtracs.CLASS_NAMES; the model stays decoupled from the data
# source (n_classes is set from config in practice). TargetSpec will make this
# fully config-driven (plan-encoder-probing-rescope r3).
N_CLASSES = 9


def build_attention_mask(
    station_mask: jax.Array,
    full_self_attention: bool = False,
) -> jax.Array:
    """Build the (1+N)-token attention mask from a station mask.

    Single source of truth — used by TCClassifier's forward pass and by
    the mask-visualisation figure (plotting.plot_attention_mask).

    Token layout is **CLS-first** (ViT-style): token 0 = query/CLS,
    tokens 1..N = stations.

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
        (B, 1, 1+N, 1+N) bool — head dim is 1 so it broadcasts across
        all attention heads. Index 0 = query/CLS, 1..N = stations.
    """
    B, N = station_mask.shape
    T = N + 1
    attn_mask = jnp.zeros((B, 1, T, T), dtype=bool)
    #                       [batch, head, from_token, to_token ]
    #   token 0 = query/CLS, tokens 1..N = stations
    attn_mask = attn_mask.at[:, :, 1:, 1:].set(True)  # station_rows → station_cols
    attn_mask = attn_mask.at[:, :, 0,  1:].set(True)  # query_row    → station_cols
    attn_mask = attn_mask.at[:, :, 0,  0 ].set(True)  # query_row    → query_col (self)
    if full_self_attention:
        attn_mask = attn_mask.at[:, :, 1:, 0].set(True)  # station_rows → query_col

    # Padding override: block any station column where station_mask is False.
    # Stations occupy columns 1..N; station_mask broadcast to (B, 1, 1, N).
    pad_col   = station_mask[:, None, None, :]           # (B, 1, 1, N)
    attn_mask = attn_mask.at[:, :, :, 1:].set(
        attn_mask[:, :, :, 1:] & pad_col
    )
    return attn_mask


# ---------------------------------------------------------------------------
# TCEncoder — the sparse-observation encoder (the frozen-transferable asset)
# ---------------------------------------------------------------------------

class TCEncoder(nn.Module):
    """Sparse-observation encoder: batch dict → normalised CLS representation.

    This is the asset the project cares about (plan r4/r5): a target-agnostic
    encoder whose CLS embedding is read by a swappable linear probe head. It
    owns the Senseiver single projection, the CLS-first token construction, the
    Transformer, and a final LayerNorm. ``TCClassifier = TCEncoder + Dense head``,
    so the param tree splits cleanly into ``params['encoder']`` (this module)
    and ``params['head']`` — see split_encoder_head / attach_encoder /
    encoder_freeze_labels for the frozen-encoder transfer protocol.

    Field meanings match TCClassifier (minus n_classes). ``__call__`` returns the
    normalised CLS embedding ``z`` of shape (B, embed_dim); with
    return_weights it also returns the full per-layer attention matrices
    (num_layers, B, num_heads, 1+N, 1+N).
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
    full_self_attention: bool = False
    missingness_indicator: bool = True
    learnable_query_pos: bool = True

    def setup(self):
        self.coord_embedding = GaussianFourierEmbedding(
            input_dim=2,
            mapping_dim=self.fourier_dim,
            scale=self.fourier_scale,
        )

        # Single shared token projection over [obs; (mask); Fourier(position)].
        # Stations and the query/CLS token both pass through this one Dense.
        self.token_proj = nn.Dense(self.embed_dim)

        # Learned query/CLS content for the obs (+mask) slots.
        self._query_obs_width = (
            2 * self.n_obs_features if self.missingness_indicator
            else self.n_obs_features
        )
        self.query_obs_slots = self.param(
            'query_obs_slots',
            nn.initializers.normal(stddev=0.02),
            (self._query_obs_width,),
        )

        # Learnable CLS position slots (learnable_query_pos, unit_circle); under
        # domain the position slots come from Fourier(query_coords) instead and
        # this parameter is not created. See TCClassifier for the full rationale.
        if self.learnable_query_pos:
            self.query_pos_slots = self.param(
                'query_pos_slots',
                nn.initializers.normal(stddev=0.02),
                (self.fourier_dim,),
            )

        self.transformer = TransformerEncoder(
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

        # Final norm — part of the encoder asset, so a probe head is a pure
        # Dense over a normalised representation (clean linear-probe semantics).
        self.norm = nn.LayerNorm()

    def __call__(
        self,
        X:              dict,
        train:          bool = True,
        return_weights: bool = False,
    ):
        station_obs    = X['station_obs']     # (B, N, F)
        station_coords = X['station_coords']  # (B, N, 2)
        station_mask   = X['station_mask']    # (B, N) bool
        obs_mask       = X['obs_mask']        # (B, N, F) bool
        query_coords   = X['query_coords']    # (B, 2)

        B, N, _ = station_obs.shape

        # 1. Missing obs -> 0. The mask channel (below) disambiguates "observed
        # 0" from "absent", so the fill value is irrelevant. No sentinel.
        obs_zeroed = jnp.where(obs_mask, station_obs, 0.0)              # (B, N, F)

        # 2. Fourier-embed station positions (flatten batch×station for the call).
        coord_feats = self.coord_embedding(
            station_coords.reshape(B * N, 2)
        ).reshape(B, N, self.fourier_dim)                              # (B, N, K)

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

        # 4. Query/CLS token — SAME projection. query_obs_slots fills the
        # obs(+mask) slots; the position slots are a learned parameter
        # (learnable_query_pos) or Fourier(query_coords) (domain).
        xi = jnp.broadcast_to(
            self.query_obs_slots[None, :], (B, self._query_obs_width)
        )                                                                # (B, obs_slots)
        if self.learnable_query_pos:
            query_pos = jnp.broadcast_to(
                self.query_pos_slots[None, :], (B, self.fourier_dim)
            )                                                            # (B, K)
        else:
            query_pos = self.coord_embedding(query_coords)              # (B, K)
        query_input = jnp.concatenate([xi, query_pos], axis=-1)         # (B, obs_slots+K)
        query_token = self.token_proj(query_input)[:, None, :]          # (B, 1, D)

        # 5. Concatenate CLS-first (ViT-style): query/CLS token then stations.
        tokens = jnp.concatenate([query_token, station_tokens], axis=1) # (B, 1+N, D)

        # 6. Attention mask — CLS-first (token 0 = query). Default asymmetric;
        # full_self_attention opens the stations→query block.
        attn_mask = build_attention_mask(
            station_mask, self.full_self_attention)          # (B, 1, 1+N, 1+N)

        # 7. Transformer.
        out = self.transformer(
            tokens, mask=attn_mask, train=train, return_weights=return_weights,
        )
        if return_weights:
            encoded, attn_weights = out          # (B, 1+N, D), (L, B, H, 1+N, 1+N)
        else:
            encoded = out                        # (B, 1+N, D)

        # 8. Normalised CLS representation (token 0).
        z = self.norm(encoded[:, 0, :])                                 # (B, D)
        if return_weights:
            return z, attn_weights
        return z


# ---------------------------------------------------------------------------
# TCClassifier
# ---------------------------------------------------------------------------

class TCClassifier(nn.Module):
    """Sparse-observation TC classifier = TCEncoder + linear probe head.

    A thin wrapper: a TCEncoder (the frozen-transferable asset) produces the
    normalised CLS embedding, and a single ``nn.Dense`` head maps it to logits.
    The param tree therefore splits cleanly into ``params['encoder']`` and
    ``params['head']`` — see split_encoder_head / attach_encoder /
    encoder_freeze_labels for the frozen-encoder transfer protocol.

    The encoder is a unified Transformer over 1+N tokens, CLS-first (ViT-style):
    one query/CLS token at position 0 then N station tokens. An asymmetric
    attention mask separates "contextualise the observation network" (station
    self-attention) from "classify" (query reads all stations); the head reads
    the query/CLS token (position 0).

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
        Output classes. Default 9 (ordinal organisation scale; see
        data/sources/ibtracs.CLASS_NAMES).
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
    learnable_query_pos : bool
        How the CLS/query token's positional slots are produced.
        True (default, unit_circle): the position slots are a learned parameter
        (query_pos_slots) — the whole CLS input [obs; mask; pos] is learnable.
        The query sits at the storm origin (0,0), so its Fourier position is a
        constant that a learned vector subsumes (strictly more expressive).
        False (domain): the position slots come from Fourier(query_coords), so
        the CLS carries the absolute query location; only its obs/mask slots are
        learned. The experiment entry points derive this from location_encoding;
        the two settings have different parameter trees (so checkpoints are not
        interchangeable across the flag).

    Notes
    -----
    Batch dict X must contain:
        station_obs    (B, N, F)  normalised obs, missing → 0 from datamodule
        station_coords (B, N, 2)  encoded station positions
        station_mask   (B, N)     bool True=real station, False=padding
        obs_mask       (B, N, F)  bool True=measurement present
        query_coords   (B, 2)     (0,0) for unit_circle; encoded pos for domain.
                                  Ignored when learnable_query_pos=True.

    Output: (B, n_classes) raw logits.

    Example
    -------
    >>> model = TCClassifier(embed_dim=128, num_heads=4, num_layers=2)
    >>> variables = model.init(jax.random.PRNGKey(0), X, train=False)
    >>> logits = model.apply(variables, X, train=False)
    >>> logits.shape
    (B, 9)
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
    learnable_query_pos: bool = True

    def setup(self):
        # The encoder is the frozen-transferable asset (one param subtree);
        # the head is the swappable, target-specific linear probe.
        self.encoder = TCEncoder(
            embed_dim             = self.embed_dim,
            num_heads             = self.num_heads,
            num_layers            = self.num_layers,
            mlp_ratio             = self.mlp_ratio,
            mlp_activation        = self.mlp_activation,
            mlp_initializer       = self.mlp_initializer,
            dropout_rate          = self.dropout_rate,
            attn_dropout_rate     = self.attn_dropout_rate,
            fourier_dim           = self.fourier_dim,
            fourier_scale         = self.fourier_scale,
            n_obs_features        = self.n_obs_features,
            full_self_attention   = self.full_self_attention,
            missingness_indicator = self.missingness_indicator,
            learnable_query_pos   = self.learnable_query_pos,
        )
        # Pure-linear probe head over the encoder's normalised CLS embedding.
        self.head = nn.Dense(self.n_classes)

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
            encoder layer: shape (num_layers, B, num_heads, 1+N, 1+N).
            CLS-first: the query row is [..., 0, :]; its FIRST element
            ([..., 0, 0]) is the query's self-attention weight — a useful
            diagnostic (high = model relies on CLS prior; low = trusts
            stations).

        Returns
        -------
        jax.Array or tuple[jax.Array, jax.Array]
            Logits (B, n_classes), and optionally weights
            (num_layers, B, num_heads, 1+N, 1+N).
        """
        out = self.encoder(X, train=train, return_weights=return_weights)
        if return_weights:
            z, attn_weights = out                # (B, D), (L, B, H, 1+N, 1+N)
        else:
            z = out                              # (B, D)

        logits = self.head(z)                                           # (B, n_classes)

        if return_weights:
            return logits, attn_weights
        return logits


# ---------------------------------------------------------------------------
# Frozen-encoder / linear-probe helpers (plan-encoder-probing-rescope r4/r5)
# ---------------------------------------------------------------------------
# The TCClassifier param tree separates into the encoder asset and the
# target-specific head. These helpers make "train encoder+head on target A →
# freeze the encoder → train a fresh head on target B" a few clean operations.
# They all operate on the param dict (the tree under ``variables['params']``).

ENCODER_KEY = 'encoder'


def split_encoder_head(params: dict) -> tuple[dict, dict]:
    """Split a TCClassifier param tree into (encoder_params, head_params).

    encoder_params is the frozen-transferable asset (the TCEncoder subtree);
    head_params is everything outside it (the linear probe head). Pass the
    tree under ``variables['params']``.
    """
    encoder = params[ENCODER_KEY]
    head    = {k: v for k, v in params.items() if k != ENCODER_KEY}
    return encoder, head


def attach_encoder(fresh_params: dict, encoder_params: dict) -> dict:
    """Return ``fresh_params`` with its encoder subtree replaced by
    ``encoder_params``.

    Loads a trained/frozen encoder (e.g. from a checkpoint of a model trained
    on target A) into a freshly initialised TCClassifier that carries a new,
    target-specific head — one operation for the frozen-encoder transfer
    protocol. The head keeps its fresh initialisation.
    """
    out = dict(fresh_params)
    out[ENCODER_KEY] = encoder_params
    return out


def encoder_freeze_labels(params: dict) -> dict:
    """Label pytree (matching ``params``) marking the encoder subtree 'frozen'
    and everything else 'trainable'.

    Feed to optax to freeze the encoder during linear-probe / transfer
    training::

        labels = encoder_freeze_labels(state.params)
        tx = optax.multi_transform(
            {'trainable': base_tx, 'frozen': optax.set_to_zero()}, labels)
    """
    return {
        k: jax.tree_util.tree_map(
            lambda _: ('frozen' if k == ENCODER_KEY else 'trainable'), v
        )
        for k, v in params.items()
    }
