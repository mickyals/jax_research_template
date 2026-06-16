"""
experiments/sparse_obs_cross_attn/data/splits.py

Resolves the YAML data.split block into per-split filtered datasets and a
JSON-serialisable run manifest.

Two strategies (decision 3, plan-data-splits-sampling):

'season' splits by ISO_TIME calendar year — both the IBTrACS track rows
(filter_seasons, now year-based) and the insitu/background stream
(filter_years) use the observation year, so the two streams stay aligned even
for cross-New-Year storms (the IBTRACS_*_SEASONS constants remain the default
config values). NOTE the config key is still named `seasons:` pending the
season→year rename.

'sid' is the hybrid design: test = edge years (explicit config list);
remaining interior storms are assigned train/val by SID at a seeded,
configurable fraction, stratified by a per-storm sid_meta column
(peak_sshs). Requires ibtracs.sid_meta_path.

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

data.split schema (sid strategy)
------------------------------------
    split:
      strategy: sid
      test:
        seasons: [2005, 2025]    # edge years -> test
        hard_test: multi_storm   # optional, as in 'season'
      val:
        fraction: 0.2            # fraction of interior SIDs assigned to val
        seed: 42
        stratify_by: peak_sshs   # sid_meta column
      # train: no block — train is the implicit remainder of interior storms

    Test membership is decided by TRACK calendar years (from sid_meta
    track_start/track_end), not bare SEASON labels: a storm is a test storm
    iff every calendar year its track touches is in test.seasons. (Concrete
    case: Zeta, SEASON 2005, track entirely Jan 2006 — interior.) This keeps
    the test insitu/background year stream disjoint from the interior one.

    Insitu coverage: train and val deliberately SHARE the full interior-year
    observation stream (decision 3 — val is a tuning signal, the leakage-free
    claim set is test); only test gets a disjoint edge-year stream. Year
    lists are derived from track spans and recorded in the manifest as
    insitu_years.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from datasets.splitting import assign_groups_by_fraction, validate_disjoint_groups
from experiments.sparse_obs_cross_attn.data.sources.ibtracs import (
    IBTrACSDataset, status_sshs_to_class, N_CLASSES,
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
        If strategy is not 'season' or 'sid'.
    KeyError, ValueError
        On a malformed or non-disjoint split config.
    """
    strategy = split_config.get('strategy', 'season')
    if strategy == 'season':
        return _resolve_season(split_config, ibtracs_full, insitu_full)
    if strategy == 'sid':
        return _resolve_sid(split_config, ibtracs_full, insitu_full)
    raise NotImplementedError(
        f"split strategy '{strategy}' is not implemented. "
        "Supported strategies: 'season', 'sid'."
    )


# ---------------------------------------------------------------------------
# 'season' strategy
# ---------------------------------------------------------------------------

def _resolve_season(
    split_config: dict,
    ibtracs_full: IBTrACSDataset,
    insitu_full: Optional[InsituLandDataset],
) -> dict:
    # Both IBTrACS (filter_seasons) and insitu (filter_years) split by ISO_TIME
    # calendar year, so the TC and background streams are year-aligned (a
    # cross-New-Year storm lands in the split matching its observation times,
    # not its SEASON label).
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
# 'sid' strategy (hybrid: edge-year test + stratified SID train/val)
# ---------------------------------------------------------------------------

def _resolve_sid(
    split_config: dict,
    ibtracs_full: IBTrACSDataset,
    insitu_full: Optional[InsituLandDataset],
) -> dict:
    if ibtracs_full._sid_meta is None:
        raise ValueError(
            "split strategy 'sid' requires ibtracs.sid_meta_path "
            "(per-storm metadata table) — it was not provided."
        )
    if 'train' in split_config:
        raise ValueError(
            "data.split.train must not be specified for strategy 'sid' — "
            "train is the implicit remainder of interior storms."
        )
    meta = ibtracs_full._sid_meta

    test_cfg = split_config.get('test', {})
    if 'seasons' not in test_cfg:
        raise KeyError("data.split.test.seasons is required for strategy 'sid'.")
    edge_years = {int(s) for s in test_cfg['seasons']}

    val_cfg = split_config.get('val', {})
    for key in ('fraction', 'seed', 'stratify_by'):
        if key not in val_cfg:
            raise KeyError(f"data.split.val.{key} is required for strategy 'sid'.")
    fraction    = float(val_cfg['fraction'])
    seed        = int(val_cfg['seed'])
    stratify_by = str(val_cfg['stratify_by'])
    if stratify_by not in meta:
        raise ValueError(
            f"data.split.val.stratify_by column {stratify_by!r} not found in "
            f"sid_meta (available: {sorted(meta)})."
        )

    # Test membership by TRACK calendar years, not bare SEASON labels
    # (Zeta: SEASON 2005, track entirely Jan 2006 -> interior).
    track_years = _sid_track_years(meta)
    order = np.argsort(meta['SID'])   # deterministic SID ordering
    sids_sorted = meta['SID'][order]
    is_test = np.array(
        [track_years[s] <= edge_years for s in sids_sorted], dtype=bool
    )
    test_sids     = sids_sorted[is_test]
    interior_sids = sids_sorted[~is_test]
    if len(test_sids) == 0:
        raise ValueError(
            f"data.split.test.seasons {sorted(edge_years)} match no storm "
            "track years — empty test split."
        )
    if len(interior_sids) == 0:
        raise ValueError(
            "data.split.test.seasons cover every storm — empty interior pool."
        )

    strata   = meta[stratify_by][order][~is_test]
    val_mask = assign_groups_by_fraction(interior_sids, fraction, seed, strata)
    sids = {
        'train': interior_sids[~val_mask].tolist(),
        'val':   interior_sids[val_mask].tolist(),
        'test':  test_sids.tolist(),
    }
    validate_disjoint_groups(sids)

    # Insitu year coverage from track spans: train and val SHARE the full
    # interior-year stream (decision 3); only test is time-separated.
    interior_years = sorted(set().union(*(track_years[s] for s in interior_sids)))
    test_years     = sorted(set().union(*(track_years[s] for s in test_sids)))
    insitu_years   = {'train': interior_years, 'val': interior_years,
                      'test': test_years}

    has_multi = ibtracs_full._multi_times is not None
    if not has_multi:
        warnings.warn(
            "ibtracs.multi_storm_path not provided — multi-storm timesteps "
            "are NOT excluded from train/val/test, and hard_test is unavailable.",
            UserWarning,
            stacklevel=2,
        )

    result: dict = {'manifest': {
        'strategy': 'sid',
        'assignment': {'fraction': fraction, 'seed': seed,
                       'stratify_by': stratify_by},
    }}
    for name in SPLIT_NAMES:
        ib = ibtracs_full.filter_sids(sids[name])
        if has_multi:
            ib = ib.filter_single_storm()
        ins = (insitu_full.filter_years(insitu_years[name])
               if insitu_full is not None else None)

        result[name] = {'ibtracs': ib, 'insitu': ins}
        entry = _split_manifest_entry(
            ib, sorted({int(s) for s in ibtracs_full.filter_sids(sids[name])['SEASON']})
        )
        entry['insitu_years'] = insitu_years[name]
        result['manifest'][name] = entry

    hard_test_cfg = test_cfg.get('hard_test')
    if hard_test_cfg == 'multi_storm':
        if not has_multi:
            raise ValueError(
                "data.split.test.hard_test == 'multi_storm' requires "
                "ibtracs.multi_storm_path."
            )
        ht = ibtracs_full.filter_sids(sids['test']).filter_multi_storm()
        result['hard_test'] = {'ibtracs': ht}
        ht_entry = _split_manifest_entry(
            ht, sorted({int(s) for s in ht['SEASON']}) if len(ht) else []
        )
        ht_entry['insitu_years'] = insitu_years['test']
        result['manifest']['hard_test'] = ht_entry
    elif hard_test_cfg is not None:
        raise ValueError(
            f"Unknown data.split.test.hard_test value: {hard_test_cfg!r}. "
            "Only 'multi_storm' is supported."
        )

    return result


def _sid_track_years(meta: dict[str, np.ndarray]) -> dict[str, set[int]]:
    """Calendar years spanned by each storm's track (track_start..track_end)."""
    start_years = pd.to_datetime(meta['track_start']).year
    end_years   = pd.to_datetime(meta['track_end']).year
    return {
        sid: set(range(int(y0), int(y1) + 1))
        for sid, y0, y1 in zip(meta['SID'], start_years, end_years)
    }


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
