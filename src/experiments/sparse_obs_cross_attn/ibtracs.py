"""
experiments/sparse_obs_cross_attn/ibtracs.py

IBTrACSDataset: NpzDataset subclass for IBTrACS best-track data.

Column constants and season splits live here alongside the class so that
imports are self-contained:

    from experiments.sparse_obs_cross_attn.ibtracs import (
        IBTrACSDataset, IBTRACS_TRAIN_SEASONS, SSHS_TO_CLASS, N_CLASSES,
    )
"""

from __future__ import annotations

import warnings
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

# Label mapping — class 0 is reserved for "no storm" (assigned by TCDataset)
# USA_SSHS values present in ibtracs_full.npz: -4 through 5 (no -5)
SSHS_TO_CLASS: dict[int, int] = {
    -4: 1, -3: 2, -2: 3, -1: 4, 0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10,
}
N_CLASSES: int = 11   # 0 = no storm, 1–10 = SSHS -4 to +5


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
    """

    def __init__(self, npz_path: str | Path, multi_storm_path: Optional[str | Path] = None,) -> None:
        super().__init__(npz_path)


        self.multi_storm_path: Optional[Path] = (Path(multi_storm_path) if multi_storm_path is not None else None)
        self._multi_times: Optional[np.ndarray] = None

        # for the timesteps where more than one cyclone is present
        if self.multi_storm_path is not None:
            ms = np.load(self.multi_storm_path, allow_pickle=True)
            self._multi_times = ms['ISO_TIME']   # int64 Unix-ns

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
        return obj

    def _mask_to_dataset(self, mask: np.ndarray) -> IBTrACSDataset:
        return self._from_data(
            data             = {k: v[mask] for k, v in self._data.items()},
            npz_path         = self.npz_path,
            multi_storm_path = self.multi_storm_path,
            _multi_times     = self._multi_times,
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
        """Keep rows whose SEASON is in seasons.

        SEASON is stored as float32 — comparison is done as float.
        """
        return self.filter_column('SEASON', [float(s) for s in seasons])

    def filter_single_storm(self) -> IBTrACSDataset:
        """Remove rows where more than one storm was active simultaneously."""
        return self._mask_to_dataset(~self.is_multi_storm)

    def filter_multi_storm(self) -> IBTrACSDataset:
        """Keep only rows where more than one storm was active simultaneously."""
        return self._mask_to_dataset(self.is_multi_storm)

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def split(self, which: str) -> IBTrACSDataset:
        """Return a predefined data split.

        Parameters
        ----------
        which : 'train' | 'val' | 'test' | 'hard_test'
            train / val / test : season-based, single-storm rows only.
            hard_test          : multi-storm rows (requires multi_storm_path).

        Returns
        -------
        IBTrACSDataset
        """
        if which == 'hard_test':
            return self.filter_multi_storm()

        season_map = {
            'train': IBTRACS_TRAIN_SEASONS,
            'val':   IBTRACS_VAL_SEASONS,
            'test':  IBTRACS_TEST_SEASONS,
        }
        if which not in season_map:
            raise ValueError(
                f"Unknown split '{which}'. "
                "Choose from: 'train', 'val', 'test', 'hard_test'."
            )
        if self._multi_times is None:
            warnings.warn(
                f"{self.__class__.__name__}: multi_storm_path not provided. "
                "Multi-storm timesteps are NOT excluded from this split. "
                "Pass multi_storm_path for clean train/val/test splits.",
                UserWarning,
                stacklevel=2,
            )
            return self.filter_seasons(season_map[which])
        return self.filter_seasons(season_map[which]).filter_single_storm()

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
