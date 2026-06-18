"""
experiments/tc_perceiver_io/train/metrics.py

Experiment-specific metrics glue for the TC classification task. The generic
metric implementations and their registry (METRICS) live in training/metrics.py
— see [[feedback-metrics-home]] / project memory. This module only wires them
into the Trainer's metrics_fns dict.

Only the training 'loss' is hardcoded; every reported metric is selected by name
from the METRICS registry via the trainer.metrics config list (r14). When no
list is given the default is binary_accuracy + mae_class — both meaningful under
the ordinal organisation labels (detection rate and ordinal rank error). Plain
top-1 'accuracy' is registered/available but is no longer a default headline
(misleading under heavy class imbalance); list 'cross_entropy' explicitly to
keep the unweighted comparability anchor (e.g. when patience_metric uses it).

For this task, binary_accuracy's default threshold=1 already matches the
TC-detection semantics: class 0 (no storm) vs. class > 0 (any storm).
"""

from __future__ import annotations

from typing import Optional, Sequence

from training.losses import get_loss
from training.metrics import METRICS


# Default reported metrics when trainer.metrics is omitted (r14): keep the two
# ordinal-meaningful metrics; demote plain top-1 accuracy out of the headline.
DEFAULT_METRICS: tuple[str, ...] = ('binary_accuracy', 'mae_class')


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_metrics_fns(
    loss:        str                     = 'cross_entropy',
    loss_kwargs: Optional[dict]          = None,
    metrics:     Optional[Sequence[str]] = None,
) -> dict:
    """Return the metrics_fns dict for the Trainer.

    The first key ('loss') is the training loss and default patience metric
    (val/loss) — resolved from the loss registry in training/losses.py via
    trainer.loss + trainer.loss_kwargs. It is the ONLY hardcoded entry; every
    other reported metric is looked up by name from the METRICS registry.

    Parameters
    ----------
    loss : str
        Name of a registered loss (training/losses.py LOSSES registry);
        currently 'cross_entropy' (the default).
    loss_kwargs : dict, optional
        Forwarded to the loss factory, e.g. {'focal_gamma': 2.0,
        'class_weights': [...]} for class-balanced focal cross-entropy.
    metrics : sequence of str, optional
        Names of per-batch metrics to report, from the METRICS registry
        (training/metrics.py): cross_entropy / accuracy / binary_accuracy /
        mae_class. None → DEFAULT_METRICS (binary_accuracy + mae_class).

    Returns
    -------
    dict[str, Callable]
        {'loss': <loss_fn>, **{name: <metric_fn> for name in metrics}}.
    """
    loss_fn  = get_loss(loss, **(loss_kwargs or {}))
    selected = DEFAULT_METRICS if metrics is None else tuple(metrics)
    fns: dict = {'loss': loss_fn}
    for name in selected:
        fns[name] = METRICS.get(name)
    return fns
