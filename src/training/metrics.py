"""
training/metrics.py

Generic evaluation metrics for JAX/Flax training.

Experiment-specific metrics live with their experiment, not here.
See e.g. experiments/sparse_obs_cross_attn/metrics.py for ordinal
classification metrics.

This module is reserved for metrics that are truly dataset-agnostic
and reusable across multiple experiments (e.g. R², skill scores).
Add them here only when a second experiment actually needs them.
"""
