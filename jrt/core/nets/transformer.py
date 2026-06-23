"""
core/nets/transformer.py

Composable transformer building blocks — the pieces a Perceiver / ViT / DETR-
style model wires together, factored out of the experiment so a second model
needs ~zero new block code.

Contents
--------
BlockConfig
    Frozen bundle of the shared block hyperparameters (embed_dim, num_heads,
    mlp_*, dropout_*) — passed as ONE field instead of forwarding seven.
PreLNAttentionBlock
    The GPT-2-style pre-LN residual block (Perceiver IO eqs 4-6:
    ``LayerNorm -> attention -> +residual -> LayerNorm -> MLP -> +residual``)
    around the ``core.attention`` MHA primitive. ``cross_attention=False`` is a
    self-attention block; ``True`` cross-attends a ``context`` (its K/V). The
    residual always taps the *unnormalized* input. It ALWAYS returns
    ``(out, scores)`` (pre-softmax attention logits) — callers that don't want
    the scores ignore them, so no ``return_weights`` branching propagates.
LearnedTokens
    A ``(num_tokens, dim)`` learned parameter, broadcast over the batch — the
    Perceiver latent array or a decoder's learned output query.

These are COMPOSED (imported as classes), not retrieved from a registry — like
``core.nets.conv`` blocks. The attention primitive itself stays in
``core.attention``; this module only adds the residual-block scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import flax.linen as nn

from core.attention import get_attention
from core.nets.mlp import MLP


@dataclass(frozen=True)
class BlockConfig:
    """Shared hyperparameters for a stack of transformer blocks.

    Frozen (hashable) so it can be a static Flax module field — one field to
    forward instead of seven. ``mlp_ratio`` sets the FFN hidden width
    (``mlp_ratio * embed_dim``, GPT-2-style widening).
    """
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    mlp_activation: str = "gelu"
    mlp_initializer: str = "xavier_uniform"
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0

    def ffn(self) -> MLP:
        """The eqs-6 position-wise FFN: Linear(D -> ratio*D) -> act -> Linear(-> D)."""
        return MLP(
            out_features=self.embed_dim,
            hidden_features=int(self.embed_dim * self.mlp_ratio),
            n_layers=1,
            activation=self.mlp_activation,
            initializer=self.mlp_initializer,
            dropout_rate=self.dropout_rate,
        )


class PreLNAttentionBlock(nn.Module):
    """Pre-LN residual attention block (eqs 4-6); returns ``(out, scores)``.

    ``cross_attention=False`` (default): self-attention over ``x``.
    ``cross_attention=True``: Q from ``x``, K/V from the (separately normed)
    ``context`` — pass ``context`` (and an optional padding ``mask`` over its
    columns) at call time.

    Always returns ``(out, scores)``; ``scores`` are the pre-softmax attention
    logits (``(B, H, Tq, Tk)``) for observability — ignore them if unused.
    """
    cfg: BlockConfig
    cross_attention: bool = False

    def setup(self):
        c = self.cfg
        self.norm_q   = nn.LayerNorm()
        self.norm_kv  = nn.LayerNorm() if self.cross_attention else None
        self.attn     = get_attention(
            "self_attention", embed_dim=c.embed_dim,
            num_heads=c.num_heads, dropout_rate=c.attn_dropout_rate,
        )
        self.norm_mlp = nn.LayerNorm()
        self.mlp      = c.ffn()
        self.drop     = nn.Dropout(rate=c.dropout_rate)

    def __call__(self, x, context=None, mask=None, train=True):
        ctx = self.norm_kv(context) if self.cross_attention else None
        out, scores = self.attn(
            self.norm_q(x), context=ctx, mask=mask,
            train=train, return_weights=True,
        )
        z = x + self.drop(out, deterministic=not train)                    # eq 5
        z = z + self.drop(self.mlp(self.norm_mlp(z), train=train),
                          deterministic=not train)                         # eq 6
        return z, scores


class LearnedTokens(nn.Module):
    """A ``(num_tokens, dim)`` learned array, broadcast over the batch.

    The Perceiver latent array (``num_tokens=N``) or a decoder's single learned
    output query (``num_tokens=1``). ``param_name`` lets callers keep a stable
    parameter key (e.g. ``'latents'`` / ``'query'``). Truncated-normal init
    (mean 0, std ``init_std``, ±2σ) — the original Perceiver latent init.
    """
    num_tokens: int
    dim: int
    init_std: float = 0.02
    param_name: str = "tokens"

    def setup(self):
        self.tokens = self.param(
            self.param_name,
            nn.initializers.truncated_normal(stddev=self.init_std),
            (self.num_tokens, self.dim),
        )

    def __call__(self, batch_size: int):
        return jnp.broadcast_to(
            self.tokens[None], (batch_size, self.num_tokens, self.dim),
        )
