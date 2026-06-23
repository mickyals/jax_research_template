"""
experiments/tc_perceiver_io/train/_config.py

Config helpers shared by the train / tune / evaluate entry points, so the YAML
load and the top-level → sub-block propagation live in one place (they were
duplicated, and the loaders had diverged — evaluate.py opened the file without a
UTF-8 encoding, which trips on non-ASCII config comments on Windows).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file (UTF-8).

    UTF-8 is explicit so config comments with non-ASCII characters (°, ×, →, …)
    load on any platform — the default encoding is cp1252 on Windows and would
    raise UnicodeDecodeError.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def propagate_location_encoding(config: dict) -> str:
    """Push the top-level ``location_encoding`` into the ``data`` block; return it.

    The model is coordinate-agnostic (the learned latent array is the encode
    query), so ``location_encoding`` configures the datamodule's coordinate
    convention only — nothing is injected into ``config['model']``. Falls back to
    ``data.location_encoding`` and then ``'unit_circle'``.
    """
    loc_enc = config.get(
        "location_encoding",
        config["data"].get("location_encoding", "unit_circle"),
    )
    config["data"]["location_encoding"] = loc_enc
    return loc_enc
