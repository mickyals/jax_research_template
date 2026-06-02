"""
ibtracs/dataset.py

IBTrACS best tracks dataset for the North Atlantic / Caribbean-Gulf domain.

Inherits loading, filtering, and export from NpzDataset. Adds:
- Multi-storm secondary npz loading
- IBTrACS-specific season and SID filters
- Predefined train/val/test/hard_test splits
- Domain summary with per-column missingness reporting
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from datasets.base import NpzDataset
from datasets.schema import (
    IBTrACSSchema,
    TRAIN_SEASONS,
    VAL_SEASONS,
    TEST_SEASONS,
)


class IBTrACSDataset(NpzDataset):
    """
    Dataclass interface for the cleaned IBTrACS best tracks npz files.

    Provides named array access, filtering by split/season/SID/time,
    and conversion to pandas for EDA. Framework-agnostic — no JAX or
    PyTorch dependency. Intended to be wrapped by a JAX dataloader or
    torch Dataset at the joint sample stage.

    Parameters
    ----------
    npz_path : str or Path
        Path to a cleaned IBTrACS npz file.
    multi_storm_path : str or Path, optional
        Path to ibtracs_multi_storm_times.npz. Required for split()
        and multi-storm filtering.

    Examples
    --------
    >>> ds    = IBTrACSDataset('data/ibtracs_tc_clean.npz',
    ...                        'data/ibtracs_multi_storm_times.npz')
    >>> train = ds.split('train')
    >>> X, y  = train.to_Xy(
    ...     target_cols=IBTrACSSchema.PRIMARY_TARGETS,
    ...     feature_cols=['LAT', 'LON'],
    ... )
    """

    def __init__(
        self,
        npz_path:         str | Path,
        multi_storm_path: str | Path | None = None,
    ) -> None:
        super().__init__(npz_path)
        self.multi_storm_path: Path | None = (
            Path(multi_storm_path) if multi_storm_path is not None else None
        )
        self._multi_times: np.ndarray | None = None

        if self.multi_storm_path is not None:
            ms = np.load(self.multi_storm_path, allow_pickle=True)
            self._multi_times = ms["ISO_TIME"]

    # ------------------------------------------------------------------
    # Override _from_data to carry multi_storm state through masks
    # ------------------------------------------------------------------

    @classmethod
    def _from_data(
        cls,
        data: dict[str, np.ndarray],
        npz_path: Path,
        **extra_attrs,
    ) -> "IBTrACSDataset":
        obj                  = cls.__new__(cls)
        obj.npz_path         = npz_path
        obj._data            = data
        obj._n               = len(next(iter(data.values()))) if data else 0
        obj.multi_storm_path = extra_attrs.get("multi_storm_path", None)
        obj._multi_times     = extra_attrs.get("_multi_times", None)
        return obj

    def _mask_to_dataset(self, mask: np.ndarray) -> "IBTrACSDataset":
        return self._from_data(
            data             = {k: v[mask] for k, v in self._data.items()},
            npz_path         = self.npz_path,
            multi_storm_path = self.multi_storm_path,
            _multi_times     = self._multi_times,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"IBTrACSDataset("
            f"n={self._n}, "
            f"SIDs={self.n_sids}, "
            f"path='{self.npz_path.name}')"
        )

    @property
    def n_sids(self) -> int:
        """Number of unique storm IDs in this subset."""
        return int(np.unique(self._data["SID"]).shape[0])

    @property
    def iso_time(self) -> pd.DatetimeIndex:
        """ISO_TIME column as a pandas DatetimeIndex."""
        return pd.to_datetime(self._data["ISO_TIME"])

    @property
    def seasons(self) -> np.ndarray:
        """SEASON column as integer array."""
        return self._data["SEASON"].astype(int)

    @property
    def is_multi_storm(self) -> np.ndarray:
        """
        Boolean mask — True where this timestep has 2+ simultaneous active
        storms in the domain.

        Raises
        ------
        ValueError
            If multi_storm_path was not provided at init.
        """
        if self._multi_times is None:
            raise ValueError(
                "multi_storm_path was not provided at init. "
                "Pass it to use multi-storm filtering or split()."
            )
        return np.isin(self._data["ISO_TIME"], self._multi_times)

    # ------------------------------------------------------------------
    # IBTrACS-specific filters
    # ------------------------------------------------------------------

    def filter_seasons(self, seasons: list[int]) -> "IBTrACSDataset":
        """Keep only rows whose SEASON is in seasons."""
        return self._mask_to_dataset(np.isin(self.seasons, seasons))

    def filter_sids(self, sids: list[str]) -> "IBTrACSDataset":
        """Keep only rows whose SID is in sids."""
        return self._mask_to_dataset(np.isin(self._data["SID"], sids))

    def filter_single_storm(self) -> "IBTrACSDataset":
        """Remove timesteps where 2+ storms are simultaneously active."""
        return self._mask_to_dataset(~self.is_multi_storm)

    def filter_multi_storm(self) -> "IBTrACSDataset":
        """Keep only timesteps where 2+ storms are simultaneously active."""
        return self._mask_to_dataset(self.is_multi_storm)

    # ------------------------------------------------------------------
    # Predefined splits
    # ------------------------------------------------------------------

    def split(self, which: str) -> "IBTrACSDataset":
        """
        Return a predefined split.

        Parameters
        ----------
        which : str
            One of:
            ``'train'``     -- single-storm timesteps, seasons 2005-2020
            ``'val'``       -- single-storm timesteps, seasons 2021-2022
            ``'test'``      -- single-storm timesteps, seasons 2023-2025
            ``'hard_test'`` -- all multi-storm timesteps (OOD test set)

        Returns
        -------
        IBTrACSDataset

        Raises
        ------
        ValueError
            If which is not one of the four accepted values, or if
            multi_storm_path was not provided (required for all splits).
        """
        SPLITS = {
            "train":     (True,  TRAIN_SEASONS),
            "val":       (True,  VAL_SEASONS),
            "test":      (True,  TEST_SEASONS),
            "hard_test": (False, None),
        }
        if which not in SPLITS:
            raise ValueError(
                f"Unknown split '{which}'. "
                f"Choose from {list(SPLITS.keys())}."
            )
        single_only, seasons = SPLITS[which]
        if single_only:
            # compute one combined mask rather than chaining two passes
            season_mask = np.isin(self.seasons, seasons)
            single_mask = ~self.is_multi_storm
            return self._mask_to_dataset(season_mask & single_mask)
        else:
            return self.filter_multi_storm()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a concise dataset summary with per-column missingness."""
        print(f"IBTrACSDataset -- {self.npz_path.name}")
        print(f"  rows    : {self._n}")
        print(f"  SIDs    : {self.n_sids}")
        print(f"  seasons : {sorted(np.unique(self.seasons).tolist())}")

        if self._multi_times is not None:
            n_multi = int(self.is_multi_storm.sum())
            print(
                f"  multi-storm : {n_multi} timesteps "
                f"({n_multi / self._n * 100:.1f}%)"
            )

        for label, cols in [
            ("primary targets",   IBTrACSSchema.PRIMARY_TARGETS),
            ("secondary targets", IBTrACSSchema.SECONDARY_TARGETS),
        ]:
            print(f"\n  {label}:")
            for c in cols:
                if c not in self._data:
                    continue
                arr     = self._data[c].astype(np.float32)
                n_valid = int(np.isfinite(arr).sum())
                print(
                    f"    {c:<22} {n_valid}/{self._n} valid "
                    f"({n_valid / self._n * 100:.1f}%)"
                )
