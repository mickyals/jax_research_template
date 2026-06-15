"""
experiments/sparse_obs_cross_attn/train/metrics.py

Experiment-specific metrics glue for the TC classification task. The
generic metric implementations (cross_entropy, accuracy, binary_accuracy,
mae_class, quadratic_weighted_kappa, expected_calibration_error) live in
training/metrics.py — see [[feedback-metrics-home]] / project memory. This
module only wires them into the Trainer's metrics_fns dict.

For this task, binary_accuracy's default threshold=1 already matches the
TC-detection semantics: class 0 (no storm) vs. class > 0 (any storm).
"""

from __future__ import annotations

from typing import Optional

from training.losses import get_loss
from training.metrics import accuracy, binary_accuracy, cross_entropy, mae_class


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_metrics_fns(
    loss:        str            = 'cross_entropy',
    loss_kwargs: Optional[dict] = None,
) -> dict:
    """Return the metrics_fns dict for the Trainer.

    The first key ('loss') is used as the training loss and the default
    patience metric (val/loss) — resolved from the loss registry in
    training/losses.py via trainer.loss + trainer.loss_kwargs. 'cross_entropy'
    is always reported separately so runs with focal/class-weighting applied
    remain comparable on a common (unweighted) scale.

    Parameters
    ----------
    loss : str
        Name of a registered loss (training/losses.py LOSSES registry);
        currently 'cross_entropy' (the default).
    loss_kwargs : dict, optional
        Forwarded to the loss factory, e.g. {'focal_gamma': 2.0,
        'class_weights': [...]} for class-balanced focal cross-entropy.

    Returns
    -------
    dict[str, Callable]
        Keys: 'loss', 'cross_entropy', 'accuracy', 'binary_accuracy', 'mae_class'.
    """
    loss_fn = get_loss(loss, **(loss_kwargs or {}))
    return {
        'loss':            loss_fn,
        'cross_entropy':   cross_entropy,
        'accuracy':        accuracy,
        'binary_accuracy': binary_accuracy,
        'mae_class':       mae_class,
    }
