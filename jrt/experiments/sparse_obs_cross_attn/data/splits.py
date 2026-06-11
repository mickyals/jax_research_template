"""
experiments/sparse_obs_cross_attn/data/splits.py

Resolves the YAML data.split block into per-split filtered datasets and a
JSON-serialisable run manifest.

Only the 'season' strategy is implemented in this phase (decision 3,
plan-data-splits-sampling). The 'season' strategy reproduces the previous
hardcoded IBTRACS_TRAIN/VAL/TEST_SEASONS behaviour exactly — those constants
become the default config values, not policy baked into the dataset classes.

data.split schema (season strategy)
------------------------------------
    split:
      strategy: season
      train:
        seasons: [2005, ..., 2020]
      val:
        seasons: [2021, 2022]
      test:
        seasons: [2023, 2024, 2025]
        hard_test: multi_storm   # optional, orthogonal row filter
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from experiments.sparse_obs_cross_attn.data.sources.ibtracs import (
    IBTrACSDataset, SSHS_TO_CLASS, N_CLASSES,
)
from experiments.sparse_obs_cross_attn.data.sources.insitu_land import InsituLandDataset


SPLIT_NAMES = ('train', 'val', 'test')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve_splits(
    split_config: dict,
    ibtracs_full: IBTrACSDataset,
    insitu_full: Optional[InsituLandDataset] = None,
) -> dict:
    """Resolve data.split into per-split datasets and a run manifest.

    Parameters
    ----------
    split_config : dict
        The data.split: block from the YAML config.
    ibtracs_full : IBTrACSDataset
        Unfiltered IBTrACS dataset (all seasons).
    insitu_full : InsituLandDataset, optional
        Unfiltered InsituLand dataset (already reliability-filtered).
        If omitted, result[name]['insitu'] is None — useful when only the
        manifest (ibtracs-derived) is needed.

    Returns
    -------
    dict
        'train' / 'val' / 'test' : {'ibtracs': IBTrACSDataset, 'insitu': ...}
        'manifest' : JSON-serialisable dict — resolved seasons/SIDs/row
                     counts/per-class counts per split.
        'hard_test' : {'ibtracs': IBTrACSDataset}, only when
                       data.split.test.hard_test == 'multi_storm'.

    Raises
    ------
    NotImplementedError
        If strategy is not 'season'.
    KeyError, ValueError
        On a malformed or non-disjoint split config.
    """
    strategy = split_config.get('strategy', 'season')
    if strategy != 'season':
        raise NotImplementedError(
            f"split strategy '{strategy}' is not implemented. "
            "Only 'season' is supported in this phase."
        )
    return _resolve_season(split_config, ibtracs_full, insitu_full)


# ---------------------------------------------------------------------------
# 'season' strategy
# ---------------------------------------------------------------------------

def _resolve_season(
    split_config: dict,
    ibtracs_full: IBTrACSDataset,
    insitu_full: Optional[InsituLandDataset],
) -> dict:
    seasons: dict[str, list[int]] = {}
    for name in SPLIT_NAMES:
        if name not in split_config or 'seasons' not in split_config[name]:
            raise KeyError(
                f"data.split.{name}.seasons is required for strategy 'season'."
            )
        seasons[name] = [int(s) for s in split_config[name]['seasons']]

    _validate_disjoint_seasons(seasons)

    has_multi = ibtracs_full._multi_times is not None
    if not has_multi:
        warnings.warn(
            "ibtracs.multi_storm_path not provided — multi-storm timesteps "
            "are NOT excluded from train/val/test, and hard_test is unavailable.",
            UserWarning,
            stacklevel=2,
        )

    result: dict = {'manifest': {'strategy': 'season'}}
    for name in SPLIT_NAMES:
        season_list = seasons[name]
        ib = ibtracs_full.filter_seasons(season_list)
        if has_multi:
            ib = ib.filter_single_storm()
        ins = insitu_full.filter_years(season_list) if insitu_full is not None else None

        result[name] = {'ibtracs': ib, 'insitu': ins}
        result['manifest'][name] = _split_manifest_entry(ib, season_list)

    hard_test_cfg = split_config['test'].get('hard_test')
    if hard_test_cfg == 'multi_storm':
        if not has_multi:
            raise ValueError(
                "data.split.test.hard_test == 'multi_storm' requires "
                "ibtracs.multi_storm_path."
            )
        ht = ibtracs_full.filter_seasons(seasons['test']).filter_multi_storm()
        result['hard_test'] = {'ibtracs': ht}
        result['manifest']['hard_test'] = _split_manifest_entry(ht, seasons['test'])
    elif hard_test_cfg is not None:
        raise ValueError(
            f"Unknown data.split.test.hard_test value: {hard_test_cfg!r}. "
            "Only 'multi_storm' is supported."
        )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_disjoint_seasons(seasons: dict[str, list[int]]) -> None:
    """Raise if any season appears in more than one split."""
    seen: dict[int, str] = {}
    for name, season_list in seasons.items():
        for s in season_list:
            if s in seen:
                raise ValueError(
                    f"Season {s} appears in both data.split.{seen[s]} and "
                    f"data.split.{name} — splits must be disjoint."
                )
            seen[s] = name


def _split_manifest_entry(ib: IBTrACSDataset, season_list: list[int]) -> dict:
    return {
        'seasons':      season_list,
        'sids':         sorted(np.unique(ib['SID']).tolist()),
        'n_rows':       len(ib),
        'n_sids':       ib.n_sids,
        'class_counts': _per_class_counts(ib),
    }


def _per_class_counts(ib: IBTrACSDataset) -> dict[str, int]:
    """Per-class row counts via SSHS_TO_CLASS. Class 0 ('no storm') never appears here."""
    sshs    = np.round(ib['USA_SSHS']).astype(int)
    classes = np.array([SSHS_TO_CLASS.get(int(s), -1) for s in sshs])
    return {str(c): int((classes == c).sum()) for c in range(N_CLASSES)}
