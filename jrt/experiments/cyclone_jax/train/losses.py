"""
experiments/cyclone_jax/train/losses.py

trainer.loss yaml -> jrt LossStack. Thin glue: the mechanics (term
folding, LOSSES + MODEL_TERMS registries, per-term values, the model-term
contract (params, apply_fn, batch, pred)) live in jrt training.losses —
this module only normalises the two yaml forms:

    loss: cross_entropy                 # bare string + trainer.loss_kwargs
    loss:                               # weighted term list
      - {name: cross_entropy, kwargs: {focal_gamma: 2.0}}
      - {name: l1_params, weight: 1.0e-4}    # model term; weight 0 = monitor

Experiment-specific loss terms register here (none yet). Re-entry point,
per the step-2 rulings: when a term first needs data-derived constants
(normalisation stats / domain bounds for physics-residual scale factors),
build_loss gains a ctx argument fed from build_data — additive change.
"""

from __future__ import annotations

from training.losses import LossStack, build_loss_stack


def build_loss(trainer_cfg: dict) -> LossStack:
    """trainer yaml block -> LossStack (callable, .needs_model, .term_names).

    Bare-string form folds to a one-term stack, so the Trainer sees one
    contract regardless of form. trainer.loss_kwargs is bare-string-only;
    with a term list each term carries its own kwargs.
    """
    loss = trainer_cfg.get('loss') or 'cross_entropy'
    kwargs = trainer_cfg.get('loss_kwargs') or {}
    if isinstance(loss, str):
        return build_loss_stack([{'name': loss, 'kwargs': kwargs}])
    if kwargs:
        raise ValueError(
            "trainer.loss_kwargs only applies to the bare-string form — "
            "with a trainer.loss term list, put kwargs on each term.")
    return build_loss_stack(loss)
