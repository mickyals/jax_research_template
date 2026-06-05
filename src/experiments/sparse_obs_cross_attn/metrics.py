"""
experiments/sparse_obs_cross_attn/metrics.py

Evaluation metrics for the ordinal TC intensity classifier.

All functions accept raw OrdinalHead logits and integer class labels
and return a scalar jax.Array, satisfying the Trainer's metrics_fns
signature: fn(pred, target) -> scalar.

Metrics
-------
ordinal_loss        CORAL binary cross-entropy (training loss)
accuracy            Exact SSHS class match
mae_class           Mean |pred_class - true_class| — primary ordinal metric
within_1_class      Fraction within 1 class step — lenient accuracy
within_2_class      Fraction within 2 class steps

Physical interpretation
-----------------------
For TC intensity with 10 classes (No Storm → Cat 5):
- accuracy alone is misleading because the distribution is highly
  imbalanced (most samples are No Storm, TS, or TD).
- mae_class gives the average number of SSHS categories the model is
  wrong by — the operationally relevant number.
- within_1_class captures whether the model at least gets the
  neighbourhood right (e.g. Cat-1 vs Cat-2 is an acceptable error;
  No-Storm vs Cat-3 is not).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from training.ordinal_loss import ordinal_loss as _ordinal_loss_fn
from training.ordinal_loss import ordinal_predict
from datasets.joint.dataset import N_CLASSES


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def ordinal_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    """CORAL ordinal loss, n_classes captured from dataset constant."""
    return _ordinal_loss_fn(pred, target, N_CLASSES)


def accuracy(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Fraction of exact SSHS class matches."""
    return jnp.mean(
        ordinal_predict(pred) == target.astype(jnp.int32)
    ).astype(jnp.float32)


def mae_class(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean absolute error in class index.

    The primary ordinal accuracy metric.  A prediction of Cat-2 when
    the true class is Cat-4 scores 2; No-Storm for Cat-4 scores 4.
    """
    preds = ordinal_predict(pred).astype(jnp.float32)
    return jnp.mean(jnp.abs(preds - target.astype(jnp.float32)))


def within_k_classes(
    pred:   jax.Array,
    target: jax.Array,
    k:      int = 1,
) -> jax.Array:
    """Fraction of predictions within k class steps of the true label."""
    diff = jnp.abs(
        ordinal_predict(pred).astype(jnp.int32) - target.astype(jnp.int32)
    )
    return jnp.mean(diff <= k).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Ready-made dict for Trainer
# ---------------------------------------------------------------------------

def build_metrics_fns() -> dict:
    """Return all metrics as a dict ready for Trainer(metrics_fns=...).

    Keys become WandB metric names under the 'train/' and 'val/' prefixes.
    loss_key should be set to 'ordinal_loss' in the trainer config.

    Logged metrics
    --------------
        train/ordinal_loss      step-level (every log_every_n_steps)
        val/ordinal_loss        epoch-level
        val/accuracy
        val/mae_class
        val/within_1_class
        val/within_2_class
    """
    return {
        'ordinal_loss':    ordinal_loss,
        'accuracy':        accuracy,
        'mae_class':       mae_class,
        'within_1_class':  partial(within_k_classes, k=1),
        'within_2_class':  partial(within_k_classes, k=2),
    }
