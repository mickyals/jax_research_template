"""
experiments/sparse_obs_cross_attn/tune.py

Hyperparameter search for TCClassifier via Optuna.

Usage
-----
    # Fresh search — results lost on exit (quick experiments)
    python -m experiments.sparse_obs_cross_attn.tune \
        src/experiments/sparse_obs_cross_attn/configs/tc_tune.yaml \
        --n_trials 25

    # Persistent search — resume by running the same command again
    python -m experiments.sparse_obs_cross_attn.tune \
        src/experiments/sparse_obs_cross_attn/configs/tc_tune.yaml \
        --n_trials 50 \
        --storage sqlite:///runs/tc_classifier/hp_search/study.db \
        --study_name tc_classifier_v1

After the study completes the best params are printed via tuner.summary()
and written to <trainer.run_dir>/best_params.json for use in tc_classifier.yaml.

Search space
------------
Training : peak_value, weight_decay
Architecture : embed_dim (= num_heads × dim_per_head), num_layers,
               dropout_rate, attn_dropout_rate, fourier_dim, fourier_scale

embed_dim is constructed as num_heads × dim_per_head so it is always
divisible by num_heads without any rejection sampling.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from experiments.sparse_obs_cross_attn.datamodule import TCDataModule
from experiments.sparse_obs_cross_attn.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.model import TCClassifier
from training.tuner import Tuner, apply_search_space


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE: dict = {
    # Training
    "peak_value":        {"type": "float",       "low": 1e-5, "high": 1e-3, "log": True},
    "weight_decay":      {"type": "float",       "low": 1e-6, "high": 1e-2, "log": True},
    # Regularisation
    "dropout_rate":      {"type": "float",       "low": 0.0,  "high": 0.4},
    "attn_dropout_rate": {"type": "float",       "low": 0.0,  "high": 0.3},
    # Token size — embed_dim = num_heads * dim_per_head (always divisible)
    "num_heads":         {"type": "categorical", "choices": [2, 4, 8]},
    "dim_per_head":      {"type": "categorical", "choices": [16, 32, 64]},
    # Depth — total encoder layers (unified self-attention)
    "num_layers":        {"type": "int",         "low": 1,    "high": 6},
    # Coordinate embedding
    "fourier_dim":       {"type": "categorical", "choices": [32, 64, 128]},
    "fourier_scale":     {"type": "float",       "low": 0.1,  "high": 10.0, "log": True},
}


def suggest_fn(trial, base_config: dict) -> dict:
    """Sample HPs for one trial; return the full modified config dict."""
    hp  = apply_search_space(trial, SEARCH_SPACE)
    cfg = copy.deepcopy(base_config)

    cfg["trainer"]["scheduler_kwargs"]["peak_value"] = hp["peak_value"]
    cfg["trainer"]["optimizer_kwargs"]["weight_decay"] = hp["weight_decay"]

    cfg["model"]["embed_dim"]         = hp["num_heads"] * hp["dim_per_head"]
    cfg["model"]["num_heads"]         = hp["num_heads"]
    cfg["model"]["num_layers"]        = hp["num_layers"]
    cfg["model"]["dropout_rate"]      = hp["dropout_rate"]
    cfg["model"]["attn_dropout_rate"] = hp["attn_dropout_rate"]
    cfg["model"]["fourier_dim"]       = hp["fourier_dim"]
    cfg["model"]["fourier_scale"]     = hp["fourier_scale"]

    return cfg


def model_fn(config: dict) -> TCClassifier:
    return TCClassifier(**config["model"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def tune(
    config_path:      str | Path,
    n_trials:         int,
    storage:          str | None,
    study_name:       str,
    direction:        str,
    n_startup_trials: int,
    n_warmup_steps:   int,
) -> None:
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    seed = int(base_config.get("seed", 42))
    base_config["trainer"].setdefault("seed", seed)

    # DataModule is built once and shared across all trials.
    # TCLoader is re-iterable so calling train_loader_fn() per trial is cheap.
    dm = TCDataModule.from_config(base_config["data"])

    def train_loader_fn():
        return dm.train_loader(seed=seed, shuffle=True)

    def val_loader_fn():
        return dm.val_loader()

    metrics_fns = build_metrics_fns()

    tuner = Tuner(
        suggest_fn       = suggest_fn,
        base_config      = base_config,
        model_fn         = model_fn,
        metrics_fns      = metrics_fns,
        train_loader_fn  = train_loader_fn,
        val_loader_fn    = val_loader_fn,
        study_name       = study_name,
        direction        = direction,
        storage          = storage,
        n_startup_trials = n_startup_trials,
        n_warmup_steps   = n_warmup_steps,
    )

    tuner.run(n_trials=n_trials)
    tuner.summary()

    # Write best params next to the study's run_dir for easy reference
    run_dir = base_config["trainer"].get("run_dir")
    if run_dir:
        out_path = Path(run_dir) / "best_params.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {"best_value": tuner.best_value, "best_params": tuner.best_params},
                f, indent=2,
            )
        print(f"\nBest params written to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Hyperparameter search for TCClassifier."
    )
    p.add_argument("config",             type=str,
                   help="Path to tc_tune.yaml")
    p.add_argument("--n_trials",         type=int, default=25,
                   help="Total trials to run (default 25)")
    p.add_argument("--storage",          type=str, default=None,
                   help="Optuna storage URL, e.g. sqlite:///runs/hp_search.db. "
                        "None = in-memory (results lost on exit).")
    p.add_argument("--study_name",       type=str, default="tc_classifier",
                   help="Optuna study name (default 'tc_classifier')")
    p.add_argument("--direction",        type=str, default="minimize",
                   choices=["minimize", "maximize"],
                   help="Optimization direction (default 'minimize')")
    p.add_argument("--n_startup_trials", type=int, default=5,
                   help="Trials before pruning activates (default 5)")
    p.add_argument("--n_warmup_steps",   type=int, default=10,
                   help="Epochs per trial before pruning is checked (default 10)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    tune(
        config_path      = args.config,
        n_trials         = args.n_trials,
        storage          = args.storage,
        study_name       = args.study_name,
        direction        = args.direction,
        n_startup_trials = args.n_startup_trials,
        n_warmup_steps   = args.n_warmup_steps,
    )
