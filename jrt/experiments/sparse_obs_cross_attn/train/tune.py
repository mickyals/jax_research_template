"""
experiments/sparse_obs_cross_attn/train/tune.py

Hyperparameter search for TCClassifier via Optuna.

Usage
-----
    # Fresh search — results lost on exit (quick experiments)
    python -m experiments.sparse_obs_cross_attn.train.tune \
        jrt/experiments/sparse_obs_cross_attn/configs/tc_tune.yaml \
        --n_trials 25

    # Persistent search — resume by running the same command again
    python -m experiments.sparse_obs_cross_attn.train.tune \
        jrt/experiments/sparse_obs_cross_attn/configs/tc_tune.yaml \
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

from experiments.sparse_obs_cross_attn.data.datamodule import TCDataModule
from experiments.sparse_obs_cross_attn.train.metrics import build_metrics_fns
from experiments.sparse_obs_cross_attn.train.model import N_CLASSES
from training.class_weights import class_weights_from_counts
from experiments.sparse_obs_cross_attn.train.model import TCClassifier
from training.tuner import Tuner, apply_search_space


# ---------------------------------------------------------------------------
# Suggest function
# ---------------------------------------------------------------------------
# The search space (ranges and choices) lives in tc_tune.yaml under search_space:.
# This function defines the MAPPING from sampled HP names to config paths —
# structural code that changes when the model architecture changes, not when
# you want to widen or narrow a search range.

def suggest_fn(trial, base_config: dict) -> dict:
    """Sample HPs for one trial; return the full modified config dict.

    Reads search_space from base_config so ranges and choices are controlled
    entirely from tc_tune.yaml without touching this file.
    """
    hp  = apply_search_space(trial, base_config['search_space'])
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
    # Resolve config path so relative paths inside the config are anchored
    # to the config file's own directory (same convention as train.py).
    config_path = Path(config_path).resolve()

    with open(config_path, encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    seed = int(base_config.get("seed", 42))
    base_config["trainer"].setdefault("seed", seed)

    # Single source of truth: propagate top-level shared values down.
    base_config["data"]["batch_size"] = base_config["trainer"]["batch_size"]

    # Model is coordinate-agnostic; location_encoding only configures the
    # datamodule's coordinate convention.
    loc_enc = base_config.get("location_encoding", "unit_circle")
    base_config["data"]["location_encoding"]  = loc_enc

    # Resolve run_dir relative to the experiment root (two levels up from
    # this script, which lives in train/), NOT the config file directory
    # (configs/).
    _experiment_dir = Path(__file__).resolve().parent.parent
    trainer_cfg = base_config["trainer"]
    if "run_dir" in trainer_cfg:
        rd = Path(trainer_cfg["run_dir"])
        if not rd.is_absolute():
            trainer_cfg["run_dir"] = str(_experiment_dir / rd)

    # DataModule is built once and shared across all trials.
    # TCLoader is re-iterable so calling train_loader_fn() per trial is cheap.
    dm = TCDataModule.from_config(base_config["data"])

    _steps_per_epoch = trainer_cfg.get("steps_per_epoch")

    def train_loader_fn():
        return dm.train_loader(seed=seed, shuffle=True,
                               steps_per_epoch=_steps_per_epoch)

    def val_loader_fn():
        return dm.val_loader()

    # Optional class weighting from the train split's realized class counts
    # (computed once; shared across trials). See train.py for the rationale.
    loss_kwargs = dict(trainer_cfg.get("loss_kwargs") or {})
    cw_scheme   = trainer_cfg.get("class_weight_scheme", "none")
    if cw_scheme != "none":
        n_classes = base_config["model"].get("n_classes", N_CLASSES)
        counts = [int(dm.manifest()["train"]["class_counts"].get(str(c), 0))
                  for c in range(n_classes)]
        loss_kwargs["class_weights"] = class_weights_from_counts(
            counts, scheme=cw_scheme,
            beta=trainer_cfg.get("class_weight_beta", 0.999),
        ).tolist()

    metrics_fns = build_metrics_fns(
        loss        = trainer_cfg.get("loss", "cross_entropy"),
        loss_kwargs = loss_kwargs,
    )

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
