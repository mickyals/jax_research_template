"""
core/nets/transformers.py

Pre-LN transformer blocks + Perceiver-family compositions over the one
``core.attention.Attention`` primitive.

Contents
--------
BlockConfig
    Frozen bundle of the shared block hyperparameters (v3 knobs: num_channels,
    num_heads, num_attn_channels, widening_factor, dropout, residual_dropout).
SelfAttentionBlock / CrossAttentionBlock
    The pre-LN residual blocks (Perceiver IO eqs 4-6: LayerNorm -> attention
    -> +residual -> LayerNorm -> MLP -> +residual). Explicit classes, not a
    boolean flag; both ALWAYS return ``(out, probs)`` (post-softmax attention
    weights) so observability costs no branching. Residuals always live at
    ``num_channels`` — the attention out-projection maps ``num_attn_channels``
    back to the stream BEFORE the add, so tuning ``num_attn_channels`` never
    touches residual shapes.
LearnedTokens
    A ``(num_tokens, num_channels)`` learned array broadcast over the batch —
    the Perceiver latent array, a decoder's learned output query. Init
    ``{init_std, trunc_sigma}``: mean-0 truncated normal with the cutoff in
    SIGMA units (paper/DeepMind convention; ``trunc_sigma=None`` = plain
    normal, the torch ports' effective behaviour).
PerceiverEncoder
    ``[cross(latents, data) -> self x depth] x repeats``. ``repeats=1``
    (default) = single-encode Perceiver IO; ``repeats>1`` re-reads the data
    each repeat (Senseiver's iterative refinement). ``shared=True`` ties the
    weights of repeats 2..R (one extra unit instance called R-1 times;
    repeat 1 never shares — it reads raw data into fresh latents).
    ``pad_handling``: 'mask' (zero pads + attention mask from ``kv_valid``)
    or 'learned' (NO mask; one learned pad vector substituted into invalid
    slots — a soft null sink; NB attention mass on it scales with pad count).
PerceiverDecoder
    One cross-attention block (queries read the latents) + optional linear
    head to ``num_out_channels``.
PerceiverIO
    encoder + learned (or caller-provided) queries + decoder.
Perceiver
    Classic form: encoder + mean-pooled latents + linear head to logits.

Senseiver preset
----------------
Senseiver (Santos et al. 2023) is PerceiverIO at: ``repeats > 1``,
``shared=True``, ``widening_factor=1``, decoder queries = coordinate features
concatenated with ONE learned token (build the query experiment-side and pass
it via ``queries=``). It is a configuration, not a class.

These are COMPOSED (imported as classes), not retrieved from a registry.
Reference implementations (torch, author-provided) for diffing:
``experiments/cyclone_jax/reference/`` (untracked-local).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from core.attention import Attention
from core.nets.mlp import MLP

_PAD_HANDLING = ('mask', 'learned')


@dataclass(frozen=True)
class BlockConfig:
    """Shared hyperparameters for a stack of transformer blocks.

    Frozen (hashable) so it can be a static Flax module field — one field to
    forward instead of six.

    Parameters
    ----------
    num_channels : int
        The residual-stream channel count (latent channels for encoder
        blocks, query channels for a decoder block).
    num_heads : int
        Attention head count.
    num_attn_channels : int, optional
        Channels the attention arithmetic runs in (see
        ``core.attention.Attention``). None -> num_channels. Divisible by
        num_heads. Tune floor heuristic: min(data channels, num_channels).
    widening_factor : int
        MLP hidden = widening_factor * num_channels. Senseiver uses 1,
        transformer convention is 4 — explicit, never inherited silently.
    dropout : float
        Dropout on the post-softmax attention probabilities.
    residual_dropout : float
        Dropout on block outputs (attention and MLP) before the residual add.
    """
    num_channels: int
    num_heads: int
    num_attn_channels: Optional[int] = None
    widening_factor: int = 1
    dropout: float = 0.0
    residual_dropout: float = 0.0
    mlp_activation: str = "gelu"
    use_bias: bool = True

    def attention(self) -> Attention:
        """The block's attention primitive (out projection -> num_channels)."""
        return Attention(
            num_heads=self.num_heads,
            num_attn_channels=self.num_attn_channels,
            num_out_channels=self.num_channels,
            dropout_rate=self.dropout,
            use_bias=self.use_bias,
        )

    def ffn(self) -> MLP:
        """Position-wise FFN: Linear(C -> wf*C) -> act -> Linear(-> C)."""
        return MLP(
            out_features=self.num_channels,
            hidden_features=self.widening_factor * self.num_channels,
            n_layers=1,
            activation=self.mlp_activation,
        )


class SelfAttentionBlock(nn.Module):
    """Pre-LN self-attention block: one LN feeds q/k/v; returns (out, probs)."""
    cfg: BlockConfig

    def setup(self):
        self.norm     = nn.LayerNorm()
        self.attn     = self.cfg.attention()
        self.norm_mlp = nn.LayerNorm()
        self.mlp      = self.cfg.ffn()
        self.drop     = nn.Dropout(rate=self.cfg.residual_dropout)

    def __call__(self, x, mask=None, train: bool = False):
        xn = self.norm(x)
        h, probs = self.attn(xn, xn, mask=mask, train=train)
        x = x + self.drop(h, deterministic=not train)
        x = x + self.drop(self.mlp(self.norm_mlp(x), train=train),
                          deterministic=not train)
        return x, probs


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention block: LN_q on the query stream, LN_kv on the
    data; residuals on the query stream. Returns (out, probs)."""
    cfg: BlockConfig

    def setup(self):
        self.norm_q   = nn.LayerNorm()
        self.norm_kv  = nn.LayerNorm()
        self.attn     = self.cfg.attention()
        self.norm_mlp = nn.LayerNorm()
        self.mlp      = self.cfg.ffn()
        self.drop     = nn.Dropout(rate=self.cfg.residual_dropout)

    def __call__(self, x_q, x_kv, mask=None, train: bool = False):
        h, probs = self.attn(self.norm_q(x_q), self.norm_kv(x_kv),
                             mask=mask, train=train)
        x = x_q + self.drop(h, deterministic=not train)
        x = x + self.drop(self.mlp(self.norm_mlp(x), train=train),
                          deterministic=not train)
        return x, probs


class LearnedTokens(nn.Module):
    """A ``(num_tokens, num_channels)`` learned array, broadcast over batch.

    The Perceiver latent array (``num_tokens=N``) or a decoder's learned
    output query (``num_tokens=1``). ``param_name`` keeps a stable parameter
    key (e.g. 'latents' / 'queries').

    Init: mean-0 truncated normal, std ``init_std``, truncated at
    ``±trunc_sigma`` STANDARD DEVIATIONS (the PIO paper's [-2, 2] read in
    sigma units — DeepMind/TF convention). ``trunc_sigma=None`` -> plain
    normal (what the torch ports' absolute clamp effectively does).
    """
    num_tokens: int
    num_channels: int
    init_std: float = 0.02
    trunc_sigma: Optional[float] = 2.0
    param_name: str = "tokens"

    def setup(self):
        if self.trunc_sigma is None:
            init = nn.initializers.normal(stddev=self.init_std)
        else:
            init = nn.initializers.truncated_normal(
                stddev=self.init_std,
                lower=-self.trunc_sigma, upper=self.trunc_sigma,
            )
        self.tokens = self.param(
            self.param_name, init, (self.num_tokens, self.num_channels),
        )

    def __call__(self, batch_size: int):
        return jnp.broadcast_to(
            self.tokens[None],
            (batch_size, self.num_tokens, self.num_channels),
        )


class _EncoderUnit(nn.Module):
    """One ``[cross -> self x depth]`` unit; returns (latents, weights dict)."""
    cfg: BlockConfig
    depth: int

    def setup(self):
        self.cross = CrossAttentionBlock(self.cfg)
        self.selfs = [SelfAttentionBlock(self.cfg) for _ in range(self.depth)]

    def __call__(self, latents, data, mask=None, train: bool = False):
        weights = {}
        x, weights['cross'] = self.cross(latents, data, mask=mask, train=train)
        for i, block in enumerate(self.selfs):
            x, weights[f'self_{i}'] = block(x, train=train)
        return x, weights


class PerceiverEncoder(nn.Module):
    """``[cross(latents, data) -> self x depth] x repeats``.

    Parameters
    ----------
    cfg : BlockConfig
        Block hyperparameters; ``cfg.num_channels`` = the latent channels.
    num_latents : int
        Latent token count.
    depth : int
        Self-attention blocks per repeat (distinct weights within a repeat).
    repeats : int
        Outer loop count; each repeat re-reads the data (default 1 =
        single-encode Perceiver IO).
    shared : bool
        Tie weights of repeats 2..R (repeat 1 never shares). Dormant at
        repeats=1.
    pad_handling : str
        'mask'    -> attention mask built from ``kv_valid`` (pad content
                     mathematically irrelevant).
        'learned' -> NO mask; a learned pad vector is substituted into
                     invalid data slots (soft null sink; attention mass on it
                     scales with pad count — a density signal).
    latent_init_std, latent_trunc_sigma
        ``LearnedTokens`` init for the latent array.

    Call: ``(data, kv_valid=None, train=False) -> (latents, weights)`` where
    ``weights = {'repeat_1': {'cross': ..., 'self_0': ...}, ...}``
    (post-softmax attention probabilities). ``kv_valid`` is the derived
    station-validity row, e.g. ``arange(pad_to) < n_stations`` — bool
    (B, Tk). ``kv_valid=None`` = every slot real (no mask, no substitution).
    """
    cfg: BlockConfig
    num_latents: int
    depth: int
    repeats: int = 1
    shared: bool = False
    pad_handling: str = 'mask'
    latent_init_std: float = 0.02
    latent_trunc_sigma: Optional[float] = 2.0

    @nn.compact
    def __call__(self, data, kv_valid=None, train: bool = False):
        if self.pad_handling not in _PAD_HANDLING:
            raise ValueError(
                f"PerceiverEncoder: pad_handling={self.pad_handling!r} not in "
                f"{_PAD_HANDLING}."
            )
        batch = data.shape[0]

        latents = LearnedTokens(
            self.num_latents, self.cfg.num_channels,
            init_std=self.latent_init_std, trunc_sigma=self.latent_trunc_sigma,
            param_name='latents', name='latents',
        )(batch)

        mask = None
        if kv_valid is not None:
            if self.pad_handling == 'learned':
                pad_token = self.param(
                    'pad_token',
                    nn.initializers.truncated_normal(stddev=0.02),
                    (data.shape[-1],),
                )
                data = jnp.where(kv_valid[..., None], data, pad_token)
            else:
                mask = nn.make_attention_mask(
                    jnp.ones((batch, self.num_latents)), kv_valid, dtype=bool,
                )

        weights = {}
        x, weights['repeat_1'] = _EncoderUnit(
            self.cfg, self.depth, name='unit_1',
        )(latents, data, mask=mask, train=train)

        if self.repeats > 1:
            if self.shared:
                unit_n = _EncoderUnit(self.cfg, self.depth, name='unit_n')
                rest = [unit_n] * (self.repeats - 1)
            else:
                rest = [
                    _EncoderUnit(self.cfg, self.depth, name=f'unit_{r}')
                    for r in range(2, self.repeats + 1)
                ]
            for r, unit in enumerate(rest, start=2):
                x, weights[f'repeat_{r}'] = unit(x, data, mask=mask, train=train)

        return x, weights


class PerceiverDecoder(nn.Module):
    """Queries read the latents: one cross-attention block + optional head.

    ``cfg.num_channels`` = the QUERY channels (the latents may be a different
    channel count — the attention primitive decouples them). When
    ``num_out_channels`` is set, a final Dense maps each query's output to it
    (e.g. class logits).

    Call: ``(queries, latents, train=False) -> (y, probs)``.
    """
    cfg: BlockConfig
    num_out_channels: Optional[int] = None

    @nn.compact
    def __call__(self, queries, latents, train: bool = False):
        y, probs = CrossAttentionBlock(self.cfg, name='cross')(
            queries, latents, train=train)
        if self.num_out_channels is not None:
            y = nn.Dense(self.num_out_channels, name='head')(y)
        return y, probs


class PerceiverIO(nn.Module):
    """Encoder + (learned or caller-provided) queries + decoder.

    Decoder queries default to ``LearnedTokens(num_queries,
    num_query_channels)`` — the plain-classification recipe (one learned
    embedding). Pass ``queries=`` to override with output-specific features
    (position features ⊕ task embedding — the PIO/Senseiver recipe; built
    experiment-side, never containing measured values).

    Call: ``(data, kv_valid=None, queries=None, train=False) -> (y, weights)``
    with ``y`` (B, num_queries, num_out_channels or query channels) and
    ``weights = {'encoder': {...}, 'decode': probs}``.
    """
    cfg: BlockConfig
    num_latents: int
    depth: int
    repeats: int = 1
    shared: bool = False
    pad_handling: str = 'mask'
    num_queries: int = 1
    num_query_channels: Optional[int] = None
    num_out_channels: Optional[int] = None
    latent_init_std: float = 0.02
    latent_trunc_sigma: Optional[float] = 2.0

    @nn.compact
    def __call__(self, data, kv_valid=None, queries=None, train: bool = False):
        latents, enc_weights = PerceiverEncoder(
            self.cfg, self.num_latents, self.depth,
            repeats=self.repeats, shared=self.shared,
            pad_handling=self.pad_handling,
            latent_init_std=self.latent_init_std,
            latent_trunc_sigma=self.latent_trunc_sigma,
            name='encoder',
        )(data, kv_valid=kv_valid, train=train)
        weights = {'encoder': enc_weights}

        if queries is None:
            query_channels = self.num_query_channels or self.cfg.num_channels
            queries = LearnedTokens(
                self.num_queries, query_channels,
                init_std=self.latent_init_std,
                trunc_sigma=self.latent_trunc_sigma,
                param_name='queries', name='queries',
            )(data.shape[0])

        decoder_cfg = replace(self.cfg, num_channels=queries.shape[-1])
        y, weights['decode'] = PerceiverDecoder(
            decoder_cfg, num_out_channels=self.num_out_channels,
            name='decoder',
        )(queries, latents, train=train)

        return y, weights


class Perceiver(nn.Module):
    """Classic Perceiver: encoder + mean-pooled latents + linear head.

    Call: ``(data, kv_valid=None, train=False) -> (logits, weights)`` with
    ``logits`` (B, num_out_channels).
    """
    cfg: BlockConfig
    num_latents: int
    depth: int
    num_out_channels: int
    repeats: int = 1
    shared: bool = False
    pad_handling: str = 'mask'
    latent_init_std: float = 0.02
    latent_trunc_sigma: Optional[float] = 2.0

    @nn.compact
    def __call__(self, data, kv_valid=None, train: bool = False):
        latents, weights = PerceiverEncoder(
            self.cfg, self.num_latents, self.depth,
            repeats=self.repeats, shared=self.shared,
            pad_handling=self.pad_handling,
            latent_init_std=self.latent_init_std,
            latent_trunc_sigma=self.latent_trunc_sigma,
            name='encoder',
        )(data, kv_valid=kv_valid, train=train)

        pooled = jnp.mean(latents, axis=1)                 # (B, C)
        logits = nn.Dense(self.num_out_channels, name='head')(pooled)
        return logits, weights
