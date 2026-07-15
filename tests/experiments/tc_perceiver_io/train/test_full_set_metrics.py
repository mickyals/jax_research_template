"""
Tests for train/full_set_metrics.py (moved from tests/training/test_metrics.py
with the jrt-v2 metrics slim-down, 2026-07-05).

TestFullSetMetrics    mAP / pr_auc scalars, registry, absent classes, no positives
TestPRCurves          curve/scalar agreement, bounds, monotonic recall, per-class
"""

import numpy as np
import pytest

from experiments.tc_perceiver_io.train.full_set_metrics import (
    FULL_SET_METRICS,
    average_precision,
    binary_pr_auc,
    binary_pr_curve,
    compute_full_set_metrics,
    list_full_set_metrics,
    per_class_pr_curves,
    precision_recall_curve,
)

N_CLS = 11


# ---------------------------------------------------------------------------
# TestFullSetMetrics — mAP / pr_auc (full-set, NumPy)
# ---------------------------------------------------------------------------

class TestFullSetMetrics:

    def _separable(self):
        # 2-class, perfectly score-separable (class 0 vs class 1).
        logits = np.array([[6., -6.], [-6., 6.], [6., -6.], [-6., 6.]], np.float32)
        labels = np.array([0, 1, 0, 1], np.int32)
        return logits, labels

    def test_perfect_separation_gives_one(self):
        logits, labels = self._separable()
        assert average_precision(logits, labels) == pytest.approx(1.0, abs=1e-6)
        assert binary_pr_auc(logits, labels)     == pytest.approx(1.0, abs=1e-6)

    def test_metrics_in_unit_interval(self):
        rng    = np.random.default_rng(0)
        logits = rng.standard_normal((50, N_CLS)).astype(np.float32)
        labels = rng.integers(0, N_CLS, size=50).astype(np.int32)
        assert 0.0 <= average_precision(logits, labels) <= 1.0
        assert 0.0 <= binary_pr_auc(logits, labels)     <= 1.0

    def test_average_precision_skips_absent_classes(self):
        # Only classes 0 and 1 occur; the metric must still be well-defined.
        logits, labels = self._separable()
        assert 0.0 <= average_precision(logits, labels) <= 1.0

    def test_no_positives_pr_auc_is_zero(self):
        # All background (class 0) → no positives for the thr=1 detection AP.
        logits = np.zeros((5, N_CLS), np.float32)
        labels = np.zeros(5, np.int32)
        assert binary_pr_auc(logits, labels) == 0.0

    def test_compute_full_set_metrics_dict(self):
        logits, labels = self._separable()
        m = compute_full_set_metrics(logits, labels)
        assert set(m) == {'mAP', 'pr_auc'}
        assert all(0.0 <= v <= 1.0 for v in m.values())

    def test_registry_lists_names(self):
        assert set(list_full_set_metrics()) == {'MAP', 'PR_AUC'}

    def test_registry_get_is_case_insensitive(self):
        assert FULL_SET_METRICS.get('mAP') is average_precision


class TestPRCurves:

    def _rand(self, seed=1, n=60):
        rng    = np.random.default_rng(seed)
        logits = rng.standard_normal((n, N_CLS)).astype(np.float32)
        labels = rng.integers(0, N_CLS, n).astype(np.int32)
        return logits, labels

    def test_curve_ap_equals_scalar(self):
        # The figure's AP must match the pr_auc scalar (shared code path).
        logits, labels = self._rand()
        cv = binary_pr_curve(logits, labels)
        assert cv['ap'] == pytest.approx(binary_pr_auc(logits, labels), abs=1e-9)

    def test_curve_shape_and_bounds(self):
        logits, labels = self._rand()
        cv = binary_pr_curve(logits, labels)
        assert len(cv['precision']) == len(cv['recall'])
        assert cv['recall'][0] == 0.0
        assert cv['recall'][-1] == pytest.approx(1.0)
        assert np.all((cv['precision'] >= 0) & (cv['precision'] <= 1.0 + 1e-9))
        assert 0.0 <= cv['base_rate'] <= 1.0

    def test_recall_is_monotonic(self):
        logits, labels = self._rand(seed=2)
        cv = binary_pr_curve(logits, labels)
        assert np.all(np.diff(cv['recall']) >= -1e-9)

    def test_per_class_curves_match_map(self):
        # Mean of present-class APs equals mAP exactly.
        logits, labels = self._rand(seed=3)
        curves = per_class_pr_curves(logits, labels)
        assert set(curves) == set(np.unique(labels).tolist())
        mean_ap = float(np.mean([cv['ap'] for cv in curves.values()]))
        assert mean_ap == pytest.approx(average_precision(logits, labels), abs=1e-9)

    def test_no_positives_curve_is_degenerate(self):
        cv = precision_recall_curve(np.zeros(5), np.zeros(5, dtype=bool))
        assert cv['ap'] == 0.0 and cv['base_rate'] == 0.0
