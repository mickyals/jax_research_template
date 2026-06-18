"""
experiments/tc_perceiver_io/data/splits.py

Resolves the YAML data.split block into per-split filtered datasets and a
JSON-serialisable run manifest.

Splits are defined by CALENDAR YEAR (ISO_TIME), never the IBTrACS SEASON column
— both the IBTrACS track rows (filter_years) and the insitu/background stream
(filter_years) use the observation year, so the two streams stay aligned even
for cross-New-Year storms (e.g. Zeta: SEASON 2005, track Jan 2006 → 2006 split).

Two strategies (plan-encoder-probing-rescope r13):

'year' — explicit disjoint year lists per split (train / val / test).

'year_random' — pooled train+val years + a disjoint test year range; the pooled
train+val ROWS (timesteps) are then split into train/val by a seeded fraction.
Because the split is at the row level, adjacent points of one storm can fall in
both train and val — a mild, deliberate train↔val leakage (val is only a tuning
signal; the leakage-free claim set is test, which stays a clean held-out year
range). train and val SHARE the train+val-year insitu/background stream; only
test gets a disjoint year stream.

data.split schema ('year' strategy)
------------------------------------
    split:
      strategy: year
      train: {years: [2005, ..., 2020]}
      val:   {years: [2021, 2022]}
      test:  {years: [2023, 2024, 2025], hard_test: multi_storm}  # hard_test optional

data.split schema ('year_random' strategy)
-------------------------------------------
    split:
      strategy: year_random
      train_val: {years: [2005, ..., 2022]}     # pooled; randomly split into train/val
      test:      {years: [2023, 2024, 2025], hard_test: multi_storm}
      val:       {fraction: 0.2, seed: 42}       # row-level val fraction
      # no train block — train is the random remainder of the train_val pool

Multi-storm timesteps (when ibtracs.multi_storm_path is set) are excluded from
train/val/test; test.hard_test == 'multi_storm' exposes them as a separate
held-out evaluation set.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from datasets.splitting import assign_groups_by_fraction
from experiments.tc_perceiver_io.data.sources.ibtracs import (
    IBTrACSDataset, status_sshs_to_class, N_CLASSES,
)
from experiments.tc_perceiver_io.data.sources.insitu_land import InsituLandDataset


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
        Unfiltered IBTrACS dataset (all years).
    insitu_full : InsituLandDataset, optional
        Unfiltered InsituLand dataset (already reliability-filtered).
        If omitted, result[name]['insitu'] is None — useful when only the
        manifest (ibtracs-derived) is needed.

    Returns
    -------
    dict
        'train' / 'val' / 'test' : {'ibtracs': IBTrACSDataset, 'insitu': ...}
        'manifest' : JSON-serialisable dict — resolved years/SIDs/row
                     counts/per-class counts per split.
        'hard_test' : {'ibtracs': IBTrACSDataset}, only when
                       data.split.test.hard_test == 'multi_storm'.

    Raises
    ------
    NotImplementedError
        If strategy is not 'year' or 'year_random'.
    KeyError, ValueError
        On a malformed or non-disjoint split config.
    """
    strategy = split_config.get('strategy', 'year')
    if strategy == 'year':
        return _resolve_year(split_config, ibtracs_full, insitu_full)
    if strategy == 'year_random':
        return _resolve_year_random(split_config, ibtracs_full, insitu_full)
    raise NotImplementedError(
        f"split strategy '{strategy}' is not implemented. "
        "Supported strategies: 'year', 'year_random'."
    )


def _check_multi(ibtracs_full: IBTrACSDataset) -> bool:
    """True if multi-storm exclusion is available; warn (once) if not."""
    has_multi = ibtracs_full._multi_times is not None
    if not has_multi:
        warnings.warn(
            "ibtracs.multi_storm_path not provided — multi-storm timesteps "
            "are NOT excluded from train/val/test, and hard_test is unavailable.",
            UserWarning,
            stacklevel=3,
        )
    return has_multi


# ---------------------------------------------------------------------------
# 'year' strategy — explicit disjoint year lists per split
# ---------------------------------------------------------------------------

def _resolve_year(
    split_config: dict,
    ibtracs_full: IBTrACSDataset,
    insitu_full: Optional[InsituLandDataset],
) -> dict:
    years: dict[str, list[int]] = {}
    for name in SPLIT_NAMES:
        if name not in split_config or 'years' not in split_config[name]:
            raise KeyError(
                f"data.split.{name}.years is required for strategy 'year'."
            )
        years[name] = [int(y) for y in split_config[name]['years']]

    _validate_disjoint_years(years)
    has_multi = _check_multi(ibtracs_full)

    result: dict = {'manifest': {'strategy': 'year'}}
    for name in SPLIT_NAMES:
        year_list = years[name]
        ib = ibtracs_full.filter_years(year_list)
        if has_multi:
            ib = ib.filter_single_storm()
        ins = insitu_full.filter_years(year_list) if insitu_full is not None else None

        result[name] = {'ibtracs': ib, 'insitu': ins}
        result['manifest'][name] = _split_manifest_entry(ib, year_list)

    _resolve_hard_test(split_config, ibtracs_full, has_multi, result, years['test'])
    return result


# ---------------------------------------------------------------------------
# 'year_random' strategy — pooled train+val years, row-level random val split
# ---------------------------------------------------------------------------

def _resolve_year_random(
    split_config: dict,
    ibtracs_full: IBTrACSDataset,
    insitu_full: Optional[InsituLandDataset],
) -> dict:
    if 'train' in split_config:
        raise ValueError(
            "data.split.train must not be specified for strategy 'year_random' — "
            "train is the random remainder of the train_val pool."
        )
    tv_cfg = split_config.get('train_val', {})
    if 'years' not in tv_cfg:
        raise KeyError(
            "data.split.train_val.years is required for strategy 'year_random'."
        )
    test_cfg = split_config.get('test', {})
    if 'years' not in test_cfg:
        raise KeyError(
            "data.split.test.years is required for strategy 'year_random'."
        )
    val_cfg = split_config.get('val', {})
    for key in ('fraction', 'seed'):
        if key not in val_cfg:
            raise KeyError(
                f"data.split.val.{key} is required for strategy 'year_random'."
            )

    train_val_years = [int(y) for y in tv_cfg['years']]
    test_years      = [int(y) for y in test_cfg['years']]
    fraction        = float(val_cfg['fraction'])
    seed            = int(val_cfg['seed'])
    _validate_disjoint_years({'train_val': train_val_years, 'test': test_years})

    has_multi = _check_multi(ibtracs_full)

    # Pool the train+val years, exclude multi-storm, then random ROW split.
    pooled = ibtracs_full.filter_years(train_val_years)
    if has_multi:
        pooled = pooled.filter_single_storm()
    n = len(pooled)
    if n == 0:
        raise ValueError(
            f"data.split.train_val.years {train_val_years} match no IBTrACS rows."
        )
    # Row-level (timestep) assignment — adjacent points of one storm may split
    # across train/val (accepted mild leakage; test is the clean held-out years).
    val_mask = assign_groups_by_fraction(np.arange(n), fraction, seed)
    train_ib = pooled._mask_to_dataset(~val_mask)
    val_ib   = pooled._mask_to_dataset(val_mask)

    test_ib = ibtracs_full.filter_years(test_years)
    if has_multi:
        test_ib = test_ib.filter_single_storm()

    # Insitu: train & val SHARE the train+val-year stream; only test is separate.
    ins_tv   = (insitu_full.filter_years(train_val_years)
                if insitu_full is not None else None)
    ins_test = (insitu_full.filter_years(test_years)
                if insitu_full is not None else None)

    result: dict = {'manifest': {
        'strategy': 'year_random',
        'assignment': {'fraction': fraction, 'seed': seed, 'level': 'row'},
    }}
    split_data = {
        'train': (train_ib, ins_tv,   train_val_years),
        'val':   (val_ib,   ins_tv,   train_val_years),
        'test':  (test_ib,  ins_test, test_years),
    }
    for name, (ib, ins, year_list) in split_data.items():
        result[name] = {'ibtracs': ib, 'insitu': ins}
        result['manifest'][name] = _split_manifest_entry(ib, year_list)

    _resolve_hard_test(split_config, ibtracs_full, has_multi, result, test_years)
    return result


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_hard_test(
    split_config: dict,
    ibtracs_full: IBTrACSDataset,
    has_multi:    bool,
    result:       dict,
    test_years:   list[int],
) -> None:
    """Append the optional multi-storm hard_test split (test years, multi-storm
    rows) to ``result`` in place."""
    hard_test_cfg = split_config['test'].get('hard_test')
    if hard_test_cfg == 'multi_storm':
        if not has_multi:
            raise ValueError(
                "data.split.test.hard_test == 'multi_storm' requires "
                "ibtracs.multi_storm_path."
            )
        ht = ibtracs_full.filter_years(test_years).filter_multi_storm()
        result['hard_test'] = {'ibtracs': ht}
        result['manifest']['hard_test'] = _split_manifest_entry(ht, test_years)
    elif hard_test_cfg is not None:
        raise ValueError(
            f"Unknown data.split.test.hard_test value: {hard_test_cfg!r}. "
            "Only 'multi_storm' is supported."
        )


def _validate_disjoint_years(years: dict[str, list[int]]) -> None:
    """Raise if any year appears in more than one split."""
    seen: dict[int, str] = {}
    for name, year_list in years.items():
        for y in year_list:
            if y in seen:
                raise ValueError(
                    f"Year {y} appears in both data.split.{seen[y]} and "
                    f"data.split.{name} — splits must be disjoint."
                )
            seen[y] = name


def _split_manifest_entry(ib: IBTrACSDataset, year_list: list[int]) -> dict:
    return {
        'years':        year_list,
        'sids':         sorted(np.unique(ib['SID']).tolist()),
        'n_rows':       len(ib),
        'n_sids':       ib.n_sids,
        'class_counts': _per_class_counts(ib),
    }


def _per_class_counts(ib: IBTrACSDataset) -> dict[str, int]:
    """Per-class row counts via status_sshs_to_class.

    Class 0 (Background) never appears for an IBTrACS row. Off-axis rows
    (status_sshs_to_class → None: extratropical/post-tropical/etc.) are excluded
    and contribute to no class, so the counts sum to n_rows minus excluded rows.
    """
    sshs   = np.round(ib['USA_SSHS']).astype(int)
    status = ib['USA_STATUS'].astype(str)
    counts = {str(c): 0 for c in range(N_CLASSES)}
    for st, sh in zip(status, sshs):
        lab = status_sshs_to_class(st, sh)
        if lab is not None:
            counts[str(lab)] += 1
    return counts
