"""
experiments/tc_perceiver_io/train/model.py

TCPerceiverIO — a Perceiver IO (Jaegle et al. 2021/2022) for sparse station
observations, built as three explicit stages composed by one top module:

    Read       cross-attention encode: a learned latent array (N, D) queries the
               M station tokens (K/V), compressing M observations -> N latents.
    Processor  L self-attention blocks over the N latents (depth decoupled from
               M; no re-read — that is the original Perceiver, pointless at small
               M).
    Decoder    latents -> logits. Two tracks (Perceiver IO Fig. 6):
                 'attention' — a single learned output query cross-attends the
                               latents, then value-proj + MLP (more expressive);
                 'avgproj'   — mean-pool the latents, then one linear.

Each stage wires the GPT-2-style pre-LN residual block (Perceiver IO eqs 4-6:
LayerNorm -> attention -> +residual -> LayerNorm -> MLP -> +residual) around an
attention primitive pulled from core.attention's registry (self/cross). The
attention's output projection (f_O) lives inside that primitive; the residual
always taps the *unnormalized* input.

Encoder asset vs. head. The frozen-transferable encoder is everything that
produces the normalized latent representation — the latent array, tokenization,
Read, Processor, and the trailing LayerNorm. The Decoder is the only swappable
head. The seam lives in the param pytree (the ``decoder`` leaf), so transfer is
a pure tree operation — see split_encoder_head / attach_encoder /
encoder_freeze_labels.

Headless (n_classes=None): no Decoder is built and __call__ returns the
mean-pooled normalized latents ``z`` (B, embed_dim) — the representation a
linear probe / embedding export reads. Its param tree is exactly the encoder
subtree of a headed model.

Tokenization. A station token is one linear projection over
[obs_zeroed; (missingness mask); Fourier(coords)] -> embed_dim. Missing
observations are filled with 0 and the boolean mask is concatenated as its own
channel, so "observed 0" is distinguishable from "absent" (no sentinel needed).
Only Read needs a padding mask (its K/V are the variable-count, padded stations);
the Processor and Decoder operate over the fixed N latents and need none.

Batch dict X must contain:
    station_obs    (B, M, F)  normalized obs, missing -> 0
    station_coords (B, M, 2)  encoded station positions
    station_mask   (B, M)     bool, True = real station, False = padding
    obs_mask       (B, M, F)  bool, True = measurement present
(query_coords is no longer used — the learned latent array replaces any explicit
 query token.)

Observability. return_weights=True additionally returns a dict of PRE-softmax
attention maps, one per component, for val/test diagnostics:
    {'read':      (B, num_heads, N, M),
     'processor': (num_layers, B, num_heads, N, N),
     'decoder':   (B, num_heads, 1, N)}   # absent / None for headless / avgproj

Representation probing. Every stage also sows its OUTPUT representation into the
Flax ``intermediates`` collection (``self.sow``), so a linear probe can read what
information is present after Read, after each Processor block, and after Decode::

    out, state = model.apply({'params': p}, X, train=False,
                             mutable=['intermediates'])
    reps = state['intermediates']
        reps['read']['output']            -> ((B, N, D),)         post-Read latents
        reps['processor']['blocks_<l>']['output'] -> ((B, N, D),) per-block output
        reps['encoded']                   -> ((B, N, D),)         post-trailing-LN asset
        reps['decoder']['output']         -> ((B, D),)            decoder pre-head repr

``sow`` is a no-op unless the ``intermediates`` collection is mutable, so the
normal forward path (and return_weights) is unchanged and carries zero overhead.
``extract_representations`` returns these as a flat, probe-ready dict.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from core.attention import get_attention
from core.nets.mlp import MLP
from core.embeddings import GaussianFourierEmbedding


def _ffn(embed_dim: int, mlp_ratio: float, activation: str,
         initializer: str, dropout_rate: float) -> MLP:
    """The eqs-6 position-wise MLP: Linear(D -> mlp_ratio*D) -> act -> Linear(-> D).

    Reuses core.nets.mlp.MLP (n_layers=1 == one hidden layer + output layer).
    """
    return MLP(
        out_features=embed_dim,
        hidden_features=int(embed_dim * mlp_ratio),
        n_layers=1,
        activation=activation,
        initializer=initializer,
        dropout_rate=dropout_rate,
    )


# ---------------------------------------------------------------------------
# Read — cross-attention encode
# ---------------------------------------------------------------------------

class Read(nn.Module):
    """Perceiver encode block: latents cross-attend the input tokens (eqs 4-6).

    __call__(latents, tokens, mask) -> z, where latents is (B, N, D) and tokens
    is (B, M, D). Q from latents, K/V from tokens; the residual adds the
    unnormalized latents. mask blocks padded token (station) columns.
    """
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    mlp_activation: str = 'gelu'
    mlp_initializer: str = 'xavier_uniform'
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0

    def setup(self):
        self.norm_q   = nn.LayerNorm()
        self.norm_kv  = nn.LayerNorm()
        self.attn     = get_attention(
            'cross_attention', embed_dim=self.embed_dim,
            num_heads=self.num_heads, dropout_rate=self.attn_dropout_rate,
        )
        self.norm_mlp = nn.LayerNorm()
        self.mlp      = _ffn(self.embed_dim, self.mlp_ratio, self.mlp_activation,
                             self.mlp_initializer, self.dropout_rate)
        self.drop     = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, latents, tokens, mask=None, train=True, return_weights=False):
        attn = self.attn(
            self.norm_q(latents), context=self.norm_kv(tokens),
            mask=mask, train=train, return_weights=return_weights,
        )
        if return_weights:
            attn_out, scores = attn
        else:
            attn_out = attn
        z = latents + self.drop(attn_out, deterministic=not train)              # eq 5
        z = z + self.drop(self.mlp(self.norm_mlp(z), train=train),
                          deterministic=not train)                             # eq 6
        # Sow the Read output for representation probing (no-op unless the
        # 'intermediates' collection is mutable).
        self.sow('intermediates', 'output', z)
        return (z, scores) if return_weights else z


# ---------------------------------------------------------------------------
# Processor — latent self-attention
# ---------------------------------------------------------------------------

class ProcessorBlock(nn.Module):
    """One processor layer: latent self-attention (eqs 4-6)."""
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    mlp_activation: str = 'gelu'
    mlp_initializer: str = 'xavier_uniform'
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.attn  = get_attention(
            'self_attention', embed_dim=self.embed_dim,
            num_heads=self.num_heads, dropout_rate=self.attn_dropout_rate,
        )
        self.norm2 = nn.LayerNorm()
        self.mlp   = _ffn(self.embed_dim, self.mlp_ratio, self.mlp_activation,
                          self.mlp_initializer, self.dropout_rate)
        self.drop  = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, train=True, return_weights=False):
        attn = self.attn(self.norm1(x), train=train, return_weights=return_weights)
        if return_weights:
            attn_out, scores = attn
        else:
            attn_out = attn
        x = x + self.drop(attn_out, deterministic=not train)
        x = x + self.drop(self.mlp(self.norm2(x), train=train),
                          deterministic=not train)
        # Sow this block's output for per-layer representation probing (no-op
        # unless the 'intermediates' collection is mutable).
        self.sow('intermediates', 'output', x)
        return (x, scores) if return_weights else x


class Processor(nn.Module):
    """Stack of ``num_layers`` latent self-attention blocks (no re-read)."""
    num_layers: int
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    mlp_activation: str = 'gelu'
    mlp_initializer: str = 'xavier_uniform'
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0

    def setup(self):
        self.blocks = [
            ProcessorBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio, mlp_activation=self.mlp_activation,
                mlp_initializer=self.mlp_initializer,
                dropout_rate=self.dropout_rate,
                attn_dropout_rate=self.attn_dropout_rate,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x, train=True, return_weights=False):
        if return_weights:
            all_scores = []
            for blk in self.blocks:
                x, s = blk(x, train=train, return_weights=True)
                all_scores.append(s)
            return x, jnp.stack(all_scores, axis=0)   # (L, B, num_heads, N, N)
        for blk in self.blocks:
            x = blk(x, train=train)
        return x


# ---------------------------------------------------------------------------
# Decoder — latents -> logits
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Perceiver decoder producing class logits (Perceiver IO Fig. 6).

    mode='attention' — a single learned output query cross-attends the latents
    (eqs 4-6; the residual is kept because the query is learned, not an input
    feature), then a linear head. More expressive; data-dependent pooling.
    mode='avgproj' — mean-pool the latents, then one linear head.
    """
    embed_dim: int
    num_heads: int
    n_classes: int
    mode: str = 'attention'
    mlp_ratio: float = 4.0
    mlp_activation: str = 'gelu'
    mlp_initializer: str = 'xavier_uniform'
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0

    def setup(self):
        if self.mode == 'attention':
            self.query    = self.param(
                'query', nn.initializers.truncated_normal(stddev=0.02),
                (1, self.embed_dim),
            )
            self.norm_q   = nn.LayerNorm()
            self.norm_kv  = nn.LayerNorm()
            self.attn     = get_attention(
                'cross_attention', embed_dim=self.embed_dim,
                num_heads=self.num_heads, dropout_rate=self.attn_dropout_rate,
            )
            self.norm_mlp = nn.LayerNorm()
            self.mlp      = _ffn(self.embed_dim, self.mlp_ratio, self.mlp_activation,
                                 self.mlp_initializer, self.dropout_rate)
            self.drop     = nn.Dropout(rate=self.dropout_rate)
        elif self.mode != 'avgproj':
            raise ValueError(
                f"Decoder mode must be 'attention' or 'avgproj', got {self.mode!r}."
            )
        self.head = nn.Dense(self.n_classes)

    def __call__(self, latents, train=True, return_weights=False):
        if self.mode == 'avgproj':
            repr = jnp.mean(latents, axis=1)                               # (B, D)
            self.sow('intermediates', 'output', repr)
            logits = self.head(repr)                                       # (B, n_classes)
            return (logits, None) if return_weights else logits

        B = latents.shape[0]
        q    = jnp.broadcast_to(self.query[None], (B, 1, self.embed_dim))  # (B, 1, D)
        attn = self.attn(
            self.norm_q(q), context=self.norm_kv(latents),
            train=train, return_weights=return_weights,
        )
        if return_weights:
            attn_out, scores = attn
        else:
            attn_out = attn
        z = q + self.drop(attn_out, deterministic=not train)               # eq 5
        z = z + self.drop(self.mlp(self.norm_mlp(z), train=train),
                          deterministic=not train)                         # eq 6
        repr = z[:, 0, :]                                                  # (B, D)
        # Sow the decoder's pre-head representation for probing (no-op unless
        # the 'intermediates' collection is mutable).
        self.sow('intermediates', 'output', repr)
        logits = self.head(repr)                                          # (B, n_classes)
        return (logits, scores) if return_weights else logits


# ---------------------------------------------------------------------------
# TCPerceiverIO — the full model
# ---------------------------------------------------------------------------

class TCPerceiverIO(nn.Module):
    """Sparse-observation Perceiver IO: batch dict -> logits (or pooled embedding).

    Parameters
    ----------
    embed_dim : int
        Latent (and token) channel dimensionality D.
    num_heads : int
        Attention heads. Must divide embed_dim.
    num_latents : int
        N, the number of learned latent vectors (the index dim the model
        processes — decoupled from the station count M).
    num_process_layers : int
        L, the number of latent self-attention blocks in the Processor.
    mlp_ratio : float
        FFN hidden dim = mlp_ratio * embed_dim (GPT-2-style widening). Default 4.0.
    mlp_activation, mlp_initializer : str
    dropout_rate, attn_dropout_rate : float
        Default 0.0 (Perceiver found dropout hurt).
    fourier_dim : int
        Gaussian Fourier embedding output dim (even). Default 64.
    fourier_scale : float
    n_obs_features : int
        F, number of observation variables. Default 5.
    n_classes : int or None
        Head size. None (default) = HEADLESS: no Decoder; __call__ returns the
        mean-pooled normalized latents ``z`` (B, embed_dim). An int builds a
        Decoder producing logits (B, n_classes).
    decode_mode : str
        Decoder track when n_classes is set: 'attention' (default) or 'avgproj'.
    missingness_indicator : bool
        True (default) = concatenate obs_mask as its own channel so a missing
        feature (filled 0) is distinguishable from an observed 0.

    Output: (B, n_classes) raw logits if n_classes is set, else (B, embed_dim)
    mean-pooled normalized latents.
    """
    embed_dim: int
    num_heads: int
    num_latents: int
    num_process_layers: int
    mlp_ratio: float = 4.0
    mlp_activation: str = 'gelu'
    mlp_initializer: str = 'xavier_uniform'
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0
    fourier_dim: int = 64
    fourier_scale: float = 1.0
    n_obs_features: int = 5
    n_classes: Optional[int] = None
    decode_mode: str = 'attention'
    missingness_indicator: bool = True

    def setup(self):
        self.coord_embedding = GaussianFourierEmbedding(
            input_dim=2, mapping_dim=self.fourier_dim, scale=self.fourier_scale,
        )
        # Single shared token projection over [obs; (mask); Fourier(coords)].
        self.token_proj = nn.Dense(self.embed_dim)

        # Learned latent array (the encode query) — Perceiver init: truncated
        # normal mean 0, std 0.02, truncated at [-2, 2] (in stddev units).
        self.latents = self.param(
            'latents', nn.initializers.truncated_normal(stddev=0.02),
            (self.num_latents, self.embed_dim),
        )

        self.read = Read(
            embed_dim=self.embed_dim, num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio, mlp_activation=self.mlp_activation,
            mlp_initializer=self.mlp_initializer, dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )
        self.processor = Processor(
            num_layers=self.num_process_layers,
            embed_dim=self.embed_dim, num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio, mlp_activation=self.mlp_activation,
            mlp_initializer=self.mlp_initializer, dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )
        # Trailing LayerNorm (GPT-2 "extra final LN") — the normalized latent
        # array is the encoder asset z.
        self.norm = nn.LayerNorm()

        # Optional head (separable 'decoder' leaf). None -> headless.
        if self.n_classes is not None:
            self.decoder = Decoder(
                embed_dim=self.embed_dim, num_heads=self.num_heads,
                n_classes=self.n_classes, mode=self.decode_mode,
                mlp_ratio=self.mlp_ratio, mlp_activation=self.mlp_activation,
                mlp_initializer=self.mlp_initializer,
                dropout_rate=self.dropout_rate,
                attn_dropout_rate=self.attn_dropout_rate,
            )

    def __call__(self, X: dict, train: bool = True, return_weights: bool = False):
        station_obs    = X['station_obs']     # (B, M, F)
        station_coords = X['station_coords']  # (B, M, 2)
        station_mask   = X['station_mask']    # (B, M) bool
        obs_mask       = X['obs_mask']        # (B, M, F) bool

        B, M, _ = station_obs.shape

        # 1. Tokenize the stations. Missing obs -> 0; the mask channel (below)
        #    disambiguates observed-0 from absent.
        obs_zeroed  = jnp.where(obs_mask, station_obs, 0.0)               # (B, M, F)
        coord_feats = self.coord_embedding(
            station_coords.reshape(B * M, 2)
        ).reshape(B, M, self.fourier_dim)                                 # (B, M, K)
        if self.missingness_indicator:
            station_input = jnp.concatenate(
                [obs_zeroed, obs_mask.astype(obs_zeroed.dtype), coord_feats], axis=-1,
            )
        else:
            station_input = jnp.concatenate([obs_zeroed, coord_feats], axis=-1)
        tokens = self.token_proj(station_input)                          # (B, M, D)

        # 2. Latent array, broadcast across the batch. Read mask blocks padded
        #    station columns (broadcasts over latents and heads).
        latents   = jnp.broadcast_to(
            self.latents[None], (B, self.num_latents, self.embed_dim),
        )                                                                # (B, N, D)
        read_mask = station_mask[:, None, :]                             # (B, 1, M)

        # 3. Read -> Process -> trailing norm.
        if return_weights:
            z, read_scores = self.read(
                latents, tokens, mask=read_mask, train=train, return_weights=True)
            z, proc_scores = self.processor(z, train=train, return_weights=True)
        else:
            z = self.read(latents, tokens, mask=read_mask, train=train)
            z = self.processor(z, train=train)

        z = self.norm(z)                                                 # (B, N, D)
        # Sow the normalized encoder asset (the frozen-transferable
        # representation) for probing (no-op unless 'intermediates' is mutable).
        self.sow('intermediates', 'encoded', z)

        # 4. Headless -> pooled representation; headed -> decoder logits.
        if self.n_classes is None:
            rep = jnp.mean(z, axis=1)                                    # (B, D)
            if return_weights:
                return rep, {'read': read_scores, 'processor': proc_scores}
            return rep

        if return_weights:
            logits, dec_scores = self.decoder(z, train=train, return_weights=True)
            return logits, {
                'read': read_scores, 'processor': proc_scores, 'decoder': dec_scores,
            }
        return self.decoder(z, train=train)


# ---------------------------------------------------------------------------
# Representation probing
# ---------------------------------------------------------------------------

def extract_representations(
    model:     TCPerceiverIO,
    variables: dict,
    X:         dict,
) -> dict:
    """Run a forward pass capturing every stage's OUTPUT representation.

    Collects the ``intermediates`` the model sows at each seam and flattens them
    into a probe-ready dict (the sown values are 1-tuples — Flax's default
    append reducer — so this unwraps the ``[0]``):

        {'read':    (B, N, D)        post-Read latents,
         'process': [(B, N, D)] * L  per-Processor-block outputs (depth order),
         'encoded': (B, N, D)        post-trailing-LN encoder asset,
         'decode':  (B, D) | None    decoder pre-head representation
                                     (None for a headless model)}

    Probe a stage by pooling/flattening its representation and fitting a linear
    classifier on the frozen encoder. ``D`` is ``embed_dim``, ``N`` ``num_latents``,
    ``L`` ``num_process_layers``.

    Parameters
    ----------
    model : TCPerceiverIO
    variables : dict
        Flax variables, e.g. ``{'params': state.params}``.
    X : dict
        Batch input dict (station_obs / station_coords / station_mask / obs_mask).
    """
    _, state = model.apply(variables, X, train=False, mutable=['intermediates'])
    inter    = state['intermediates']

    proc    = inter['processor']
    n_blocks = sum(1 for k in proc if k.startswith('blocks_'))
    process  = [proc[f'blocks_{l}']['output'][0] for l in range(n_blocks)]

    return {
        'read':    inter['read']['output'][0],
        'process': process,
        'encoded': inter['encoded'][0],
        'decode':  inter['decoder']['output'][0] if 'decoder' in inter else None,
    }


# ---------------------------------------------------------------------------
# Frozen-encoder / linear-probe helpers
# ---------------------------------------------------------------------------
# The TCPerceiverIO param tree separates into the encoder asset (every leaf
# except the head) and the target-specific head leaf. The seam is the
# ``decoder`` KEY. These helpers operate on the param dict (the tree under
# ``variables['params']``).

HEAD_KEY = 'decoder'


def split_encoder_head(params: dict) -> tuple[dict, dict]:
    """Split a TCPerceiverIO param tree into (encoder_params, head_params).

    encoder_params is every leaf EXCEPT the decoder — the frozen-transferable
    asset, identical to the param tree of a HEADLESS TCPerceiverIO. head_params
    is ``{'decoder': <decoder subtree>}``. Pass the tree under
    ``variables['params']``.
    """
    encoder = {k: v for k, v in params.items() if k != HEAD_KEY}
    head    = {HEAD_KEY: params[HEAD_KEY]}
    return encoder, head


def attach_encoder(fresh_params: dict, encoder_params: dict) -> dict:
    """Return ``fresh_params`` with its encoder leaves replaced by
    ``encoder_params``, keeping the fresh decoder.

    Loads a trained/frozen encoder into a freshly initialised TCPerceiverIO that
    carries a new, target-specific decoder. ``encoder_params`` must not contain
    the decoder key (it is the output of split_encoder_head).
    """
    out = dict(fresh_params)
    for k, v in encoder_params.items():
        out[k] = v
    return out


def encoder_freeze_labels(params: dict) -> dict:
    """Label pytree (matching ``params``) marking every encoder leaf 'frozen'
    and the decoder leaves 'trainable'.

    Feed to optax to freeze the encoder during linear-probe / transfer
    training::

        labels = encoder_freeze_labels(state.params)
        tx = optax.multi_transform(
            {'trainable': base_tx, 'frozen': optax.set_to_zero()}, labels)
    """
    return {
        k: jax.tree_util.tree_map(
            lambda _: ('trainable' if k == HEAD_KEY else 'frozen'), v
        )
        for k, v in params.items()
    }
