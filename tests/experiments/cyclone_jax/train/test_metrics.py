"""
Tests for train/metrics.py — build_metrics_fns glue (moved here from
test_train.py with the step-3 split; metric maths tested in
tests/training/test_metrics.py).
"""

import pytest

from training.losses import LossStack

from experiments.cyclone_jax.train.metrics import build_metrics_fns


class TestBuildMetricsFns:

    def test_loss_is_first_metrics_key(self):
        fns = build_metrics_fns({'loss': 'cross_entropy',
                                 'metrics': ['accuracy', 'mae_class']})
        assert list(fns) == ['loss', 'accuracy', 'mae_class']

    def test_metric_named_like_loss_not_duplicated(self):
        fns = build_metrics_fns({'metrics': ['loss', 'accuracy']})
        assert list(fns) == ['loss', 'accuracy']

    def test_loss_entry_is_a_stack(self):
        fns = build_metrics_fns({'loss': 'cross_entropy'})
        assert isinstance(fns['loss'], LossStack)
        assert fns['loss'].needs_model is False

    def test_term_list_loss_supported(self):
        fns = build_metrics_fns({'loss': [
            {'name': 'cross_entropy'},
            {'name': 'l1_params', 'weight': 1.0e-4},
        ]})
        assert fns['loss'].needs_model is True
        assert fns['loss'].term_names == ('cross_entropy', 'l1_params')

    def test_macro_precision_recall_rejected(self):
        # No longer registered (PR #5): ratios don't average across batches;
        # exact values come from update_cm + compute_final_metrics instead.
        with pytest.raises(ValueError):
            build_metrics_fns({'metrics': ['accuracy', 'macro_precision']})
