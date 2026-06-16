"""
experiments/sparse_obs_cross_attn/data/sources/ibtracs.py

IBTrACSDataset: NpzDataset subclass for IBTrACS best-track data.

Column constants and label mappings live here alongside the class so that
imports are self-contained:

    from experiments.sparse_obs_cross_attn.data.sources.ibtracs import (
        IBTrACSDataset, status_sshs_to_class, CLASS_NAMES, N_CLASSES,
    )

Splitting is not a dataset concern: this class exposes filter primitives
(filter_seasons, filter_sids, filter_single_storm, filter_multi_storm) and
the split policy lives in the data.split config block, resolved by
experiments.sparse_obs_cross_attn.data.splits.resolve_splits. The
IBTRACS_*_SEASONS constants remain only as the reference values of the
original split (used by the config defaults and the resolver's referee test).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from datasets.base import NpzDataset


# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------

IBTRACS_META_COLS: list[str] = [
    'SID', 'NAME', 'SEASON', 'BASIN', 'SUBBASIN', 'ISO_TIME',
    'LAT', 'LON', 'TRACK_TYPE', 'IFLAG', 'USA_AGENCY',
    'USA_ATCF_ID', 'USA_RECORD', 'USA_STATUS', 'USA_SSHS',
]

IBTRACS_PRIMARY_TARGET_COLS: list[str] = [
    'USA_WIND', 'USA_PRES', 'USA_POCI', 'USA_RMW', 'STORM_SPEED', 'STORM_DIR',
]

IBTRACS_SECONDARY_TARGET_COLS: list[str] = [
    'USA_R17MS_NE', 'USA_R17MS_SE', 'USA_R17MS_SW', 'USA_R17MS_NW',
    'USA_R26MS_NE', 'USA_R26MS_SE', 'USA_R26MS_SW', 'USA_R26MS_NW',
    'USA_R33MS_NE', 'USA_R33MS_SE', 'USA_R33MS_SW', 'USA_R33MS_NW',
    'USA_ROCI', 'USA_EYE', 'USA_SEAHGT',
    'USA_SEARAD_NE', 'USA_SEARAD_SE', 'USA_SEARAD_SW', 'USA_SEARAD_NW',
]

IBTRACS_ALL_TARGET_COLS: list[str] = (
    IBTRACS_PRIMARY_TARGET_COLS + IBTRACS_SECONDARY_TARGET_COLS
)

# Temporal splits — season-year based, no row-level randomisation
IBTRACS_TRAIN_SEASONS: list[int] = list(range(2005, 2021))   # 2005–2020 inclusive
IBTRACS_VAL_SEASONS:   list[int] = [2021, 2022]
IBTRACS_TEST_SEASONS:  list[int] = list(range(2023, 2026))    # 2023–2025 inclusive

# ---------------------------------------------------------------------------
# Ordinal organisation label scheme (decision r2, plan-encoder-probing-rescope)
# ---------------------------------------------------------------------------
# Classes are ordered by degree of organisation: background → disturbance →
# depression → storm → Saffir-Simpson categories 1–5. The class is STATUS-driven
# — the agency USA_STATUS code sets it; USA_SSHS supplies ONLY the hurricane
# category number (1–5). Systems off the tropical-organisation axis
# (extratropical, post-tropical, dissipating, inland, extrapolated, unknown) are
# excluded: status_sshs_to_class returns None and TCDataset drops the row.
#
# Class 0 (Background) is assigned by TCDataset for no-storm samples and never
# appears for an IBTrACS row. This is the canonical label space for the
# experiment — model/metrics/plotting all read CLASS_NAMES / N_CLASSES from here.

CLASS_BACKGROUND:  int = 0
CLASS_DISTURBANCE: int = 1
CLASS_DEPRESSION:  int = 2
CLASS_STORM:       int = 3
# 4..8 = Saffir-Simpson category 1..5  (CLASS_STORM + sshs)

CLASS_NAMES: list[str] = [
    'Background',   # 0  no coherent system
    'Disturbance',  # 1  DB / LO / WV / MD
    'Depression',   # 2  TD / SD
    'Storm',        # 3  TS / SS
    'Category 1',   # 4  SSHS 1
    'Category 2',   # 5  SSHS 2
    'Category 3',   # 6  SSHS 3
    'Category 4',   # 7  SSHS 4
    'Category 5',   # 8  SSHS 5
]

N_CLASSES: int = len(CLASS_NAMES)   # 9

# USA_STATUS code groups (status drives the class — see status_sshs_to_class).
_DISTURBANCE_STATUS = frozenset({'DB', 'LO', 'WV', 'MD'})
_DEPRESSION_STATUS  = frozenset({'TD', 'SD'})
_STORM_STATUS       = frozenset({'TS', 'SS'})
_HURRICANE_STATUS   = frozenset({'HU', 'HR', 'TY', 'ST', 'TC'})
# Anything else (EX, ET, PT, DS, IN, XX, …) is off-axis → excluded (None).


def status_sshs_to_class(status, sshs) -> Optional[int]:
    """Map an IBTrACS row to its ordinal organisation class (1..8), or None.

    STATUS-driven: the agency USA_STATUS code sets the class. USA_SSHS supplies
    ONLY the hurricane category number (1–5) for hurricane-status rows. Off-axis
    systems (extratropical, post-tropical, dissipating, inland, extrapolated,
    unknown) return None and are dropped by TCDataset — they are not described
    by the tropical-organisation / Saffir-Simpson scale. Class 0 (Background) is
    assigned elsewhere.

    Parameters
    ----------
    status : str
        USA_STATUS code (e.g. 'TD', 'TS', 'SD', 'SS', 'HU', 'EX').
    sshs : int or float
        USA_SSHS value; only its rounded integer matters, and only for
        hurricane-status rows (the category number).

    Returns
    -------
    int or None
        1..8 for an on-axis TC row; None to exclude the row.
    """
    s = str(status).strip().upper()
    if s in _DISTURBANCE_STATUS:
        return CLASS_DISTURBANCE
    if s in _DEPRESSION_STATUS:
        return CLASS_DEPRESSION
    if s in _STORM_STATUS:
        return CLASS_STORM
    if s in _HURRICANE_STATUS:
        cat = int(round(float(sshs)))
        if 1 <= cat <= 5:
            return CLASS_STORM + cat          # 4..8
        # Hurricane status but sub-category-1 wind (rare inconsistency): keep it
        # on the axis as Storm rather than dropping a real cyclone.
        return CLASS_STORM
    return None


# ---------------------------------------------------------------------------
# IBTrACSDataset
# ---------------------------------------------------------------------------

class IBTrACSDataset(NpzDataset):
    """IBTrACS best-track observations backed by a cleaned .npz file.

    Parameters
    ----------
    npz_path : str or Path
        Path to an IBTrACS npz file (ibtracs_full.npz, ibtracs_tc_clean.npz, etc.).
    multi_storm_path : str or Path, optional
        Path to ibtracs_multi_storm_times.npz. Required for split() and
        filter_single_storm() / filter_multi_storm().
    sid_meta_path : str or Path, optional
        Path to ibtracs_sid_meta.npz (per-storm metadata table). When given,
        the file is validated against npz_path on load: the SID set must
        match exactly and n_timesteps must sum to the row count. Mismatches
        raise ValueError immediately (staleness guard).
    """

    def __init__(
        self,
        npz_path: str | Path,
        multi_storm_path: Optional[str | Path] = None,
        sid_meta_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(npz_path)


        self.multi_storm_path: Optional[Path] = (Path(multi_storm_path) if multi_storm_path is not None else None)
        self._multi_times: Optional[np.ndarray] = None

        # for the timesteps where more than one cyclone is present
        if self.multi_storm_path is not None:
            ms = np.load(self.multi_storm_path, allow_pickle=True)
            self._multi_times = ms['ISO_TIME']   # int64 Unix-ns

        self.sid_meta_path: Optional[Path] = (Path(sid_meta_path) if sid_meta_path is not None else None)
        self._sid_meta: Optional[dict[str, np.ndarray]] = None

        if self.sid_meta_path is not None:
            sid_meta_raw = np.load(self.sid_meta_path, allow_pickle=True)
            self._sid_meta = {k: sid_meta_raw[k] for k in sid_meta_raw.files}
            self._validate_sid_meta()

    # ------------------------------------------------------------------
    # _from_data / _mask_to_dataset — carry multi-storm state through filters
    # ------------------------------------------------------------------

    @classmethod
    def _from_data(cls, data: dict[str, np.ndarray], npz_path: Path, **extra_attrs,) -> IBTrACSDataset:

        obj = cls.__new__(cls)
        obj.npz_path         = npz_path
        obj._data            = data
        obj._n               = len(next(iter(data.values()))) if data else 0
        obj.multi_storm_path = extra_attrs.get('multi_storm_path', None)
        obj._multi_times     = extra_attrs.get('_multi_times', None)
        obj.sid_meta_path    = extra_attrs.get('sid_meta_path', None)
        obj._sid_meta        = extra_attrs.get('_sid_meta', None)
        return obj

    def _mask_to_dataset(self, mask: np.ndarray) -> IBTrACSDataset:
        return self._from_data(
            data             = {k: v[mask] for k, v in self._data.items()},
            npz_path         = self.npz_path,
            multi_storm_path = self.multi_storm_path,
            _multi_times     = self._multi_times,
            sid_meta_path    = self.sid_meta_path,
            _sid_meta        = self._sid_meta,
        )

    # ------------------------------------------------------------------
    # SID metadata validation
    # ------------------------------------------------------------------

    def _validate_sid_meta(self) -> None:
        """Fail loudly if ibtracs_sid_meta.npz is stale relative to this file.

        Checks (decision 1, plan-data-splits-sampling):
          - SID set in sid_meta matches the SID set in this dataset exactly
          - n_timesteps sums to this dataset's row count
        """
        meta_sids = set(np.unique(self._sid_meta['SID']).tolist())
        data_sids = set(np.unique(self._data['SID']).tolist())
        if meta_sids != data_sids:
            missing = data_sids - meta_sids
            extra   = meta_sids - data_sids
            raise ValueError(
                f"{self.sid_meta_path.name}: SID set does not match "
                f"{self.npz_path.name}. "
                f"Missing from sid_meta: {len(missing)}. "
                f"Extra in sid_meta: {len(extra)}."
            )

        meta_n_timesteps = int(self._sid_meta['n_timesteps'].sum())
        if meta_n_timesteps != self._n:
            raise ValueError(
                f"{self.sid_meta_path.name}: n_timesteps sums to "
                f"{meta_n_timesteps}, but {self.npz_path.name} has "
                f"{self._n} rows."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def seasons(self) -> np.ndarray:
        """Storm season years as int32 array."""
        return self['SEASON'].astype(np.int32)

    @property
    def iso_time(self) -> pd.DatetimeIndex:
        """ISO_TIME as a pandas DatetimeIndex (UTC, from Unix-ns int64)."""
        return pd.to_datetime(self['ISO_TIME'])

    @property
    def n_sids(self) -> int:
        """Number of unique storm identifiers."""
        return int(np.unique(self['SID']).size)

    @property
    def is_multi_storm(self) -> np.ndarray:
        """Boolean mask (N, ) — True where ISO_TIME is in the multi-storm set."""
        if self._multi_times is None:
            raise ValueError(
                "multi_storm_path was not provided. "
                "Pass it to IBTrACSDataset() to use is_multi_storm."
            )
        multi_set = set(self._multi_times.tolist())
        return np.array([t in multi_set for t in self['ISO_TIME']], dtype=bool)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_seasons(self, seasons: list[int]) -> IBTrACSDataset:
        """Keep rows whose ISO_TIME calendar year is in ``seasons``.

        Filters on the observation timestamp's year (not the SEASON column), so
        the TC split aligns with the insitu/background stream (also split by
        ISO year). This handles cross-New-Year storms correctly — e.g. Zeta
        (SEASON 2005, track Jan 2006) goes to the 2006 split, matching its
        actual observation times.
        """
        years = self.iso_time.year.to_numpy()
        return self._mask_to_dataset(np.isin(years, list(seasons)))

    def filter_sids(self, sids: list[str]) -> IBTrACSDataset:
        """Keep rows whose SID is in sids."""
        return self.filter_column('SID', sids)

    def filter_single_storm(self) -> IBTrACSDataset:
        """Remove rows where more than one storm was active simultaneously."""
        return self._mask_to_dataset(~self.is_multi_storm)

    def filter_multi_storm(self) -> IBTrACSDataset:
        """Keep only rows where more than one storm was active simultaneously."""
        return self._mask_to_dataset(self.is_multi_storm)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        multi_str = (
            str(int(self.is_multi_storm.sum()))
            if self._multi_times is not None else 'n/a'
        )
        sshs = self['USA_SSHS']
        print(f"IBTrACSDataset -- {self.npz_path.name}")
        print(f"  rows        : {self._n}")
        print(f"  SIDs        : {self.n_sids}")
        print(f"  seasons     : {int(self.seasons.min())}–{int(self.seasons.max())}")
        print(f"  multi-storm : {multi_str}")
        print(f"  SSHS range  : {int(sshs.min())}–{int(sshs.max())}")


# ---------------------------------------------------------------------------
# Self-registration with the generic dataset registry
# ---------------------------------------------------------------------------
# Experiment code registers its own factory (the dependency points
# experiment -> jrt, never the reverse). Importing this module — which the
# experiment's data pipeline always does — makes "IBTRACS" available to the
# generic datasets.datamodule.DataModule.from_config() registry.

from datasets.datamodule import register_dataset


@register_dataset("IBTRACS")
def _ibtracs_factory(config: dict) -> IBTrACSDataset:
    return IBTrACSDataset(
        config["npz_path"],
        config.get("multi_storm_path"),
    )
