"""
experiments/cyclone_jax/train/metrics.py

trainer.metrics yaml -> Trainer metrics_fns. Thin glue over the jrt METRICS
scaffold: the loss entry comes from train/losses.py (a LossStack) and is
always FIRST — the Trainer differentiates the first key, so the logged /
patience name is train/loss regardless of the configured objective;
trainer.metrics adds report-only metrics after it.

Experiment-specific metrics REGISTER here into the jrt METRICS registry
(the pattern established by tc_perceiver_io/train/metrics.py). None yet —
accuracy, mae_class, binary_accuracy are jrt universals. Only LINEAR
metrics are registered (PR #5 ruling): macro precision/recall are NOT
valid trainer.metrics values — ratios don't average across batches, so
their exact split-level values come from the accumulated confusion matrix
(training.metrics.update_cm + compute_final_metrics), surfaced by the
confusion-matrix callback and evaluate.py.
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
