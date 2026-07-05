"""
experiments/cyclone_jax/train/metrics.py

trainer.metrics yaml -> Trainer metrics_fns. Thin glue over the jrt METRICS
scaffold: the loss entry comes from train/losses.py (a LossStack) and is
always FIRST — the Trainer differentiates the first key, so the logged /
patience name is train/loss regardless of the configured objective;
trainer.metrics adds report-only metrics after it.

Experiment-specific metrics REGISTER here into the jrt METRICS registry
(the pattern established by tc_perceiver_io/train/metrics.py). None yet —
accuracy, macro_precision, macro_recall, mae_class, binary_accuracy are
jrt universals. Caveat worth knowing when reading wandb: macro_precision /
macro_recall are per-batch approximations (ratios don't average across
batches); exact split-level values are evaluate.py territory via
accumulated training.metrics.per_class_counts.
"""

from __future__ import annotations

from training.metrics import get_metric

from experiments.cyclone_jax.train.losses import build_loss


def build_metrics_fns(trainer_cfg: dict) -> dict:
    """trainer yaml block -> Trainer metrics_fns dict ('loss' first)."""
    fns = {'loss': build_loss(trainer_cfg)}
    for name in trainer_cfg.get('metrics') or ():
        if name not in fns:
            fns[name] = get_metric(name)
    return fns
