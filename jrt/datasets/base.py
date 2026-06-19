"""
base.py

Base class for npz-backed research datasets.

Handles the loading pattern, row-mask filtering, and common access
protocols shared by every data source (IBTrACS, ICOADS, IGRA, synoptic).
Each source subclass adds its own domain columns, split logic, and
source-specific filters on top.

Will add for zarr, nc files and others as they are needed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class NpzDataset:
    """
    Base class for datasets backed by a cleaned .npz file.

    Each row in the npz corresponds to one observation at one timestep.
    All arrays in the file must share the same leading dimension (n rows).

    Subclasses implement:
        split(which)      -- predefined train/val/test partitions
        summary()         -- domain-specific console summary

    Parameters
    ----------
    npz_path : str or Path
        Path to a cleaned .npz file. Loaded with allow_pickle=True to
        support string arrays (SID, ISO_TIME, etc.).

    Examples
    --------
    >>> ds = NpzDataset('data/my_source.npz')
    >>> ds['LAT'].shape
    (8399,)
    >>> subset = ds._mask_to_dataset(ds['SEASON'].astype(int) >= 2021)
    >>> len(subset)
    412
    """

    def __init__(self, npz_path: str | Path) -> None:
        self.npz_path = Path(npz_path)
        raw           = np.load(self.npz_path, allow_pickle=True)
        self._data: dict[str, np.ndarray] = {k: raw[k] for k in raw.files}
        self._n: int = len(next(iter(self._data.values())))

    # ------------------------------------------------------------------
    # Class-method factory used by _mask_to_dataset
    # ------------------------------------------------------------------

    @classmethod
    def _from_data(
        cls,
        data: dict[str, np.ndarray],
        npz_path: Path,
        **extra_attrs,
    ) -> "NpzDataset":
        """
        Construct an instance from an already-loaded data dict.

        Used internally by _mask_to_dataset to produce filtered subsets
        without re-reading the file. Subclasses that add extra __init__
        arguments should override this method and forward those arguments
        via extra_attrs.

        Parameters
        ----------
        data : dict
            Column-keyed arrays, all sharing the same leading dimension.
        npz_path : Path
            Stored on the instance for provenance; the file is not re-read.
        **extra_attrs
            Additional instance attributes set directly (e.g. _multi_times).
        """
        obj          = cls.__new__(cls)
        obj.npz_path = npz_path
        obj._data    = data
        obj._n       = len(next(iter(data.values()))) if data else 0
        for attr, val in extra_attrs.items():
            setattr(obj, attr, val)
        return obj

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._data:
            raise KeyError(
                f"'{key}' not in dataset. Available columns: {self.columns}"
            )
        return self._data[key]

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n={self._n}, "
            f"path='{self.npz_path.name}')"
        )

    @property
    def columns(self) -> list[str]:
        """List of all column names present in the loaded data."""
        return list(self._data.keys())

    # ------------------------------------------------------------------
    # Row-mask filtering
    # ------------------------------------------------------------------

    def _mask_to_dataset(self, mask: np.ndarray) -> "NpzDataset":
        """
        Apply a boolean row mask and return a new instance of the same type.

        The file is not re-read. Subclasses that carry extra state beyond
        _data (e.g. a secondary npz) should override _from_data to forward
        that state.

        Parameters
        ----------
        mask : np.ndarray of bool
            Shape (n,). True rows are kept.

        Returns
        -------
        NpzDataset
            A new instance of the calling subclass containing only the
            masked rows.
        """
        return self._from_data(
            data     = {k: v[mask] for k, v in self._data.items()},
            npz_path = self.npz_path,
        )

    def filter_column(self, col: str, values) -> "NpzDataset":
        """
        Keep rows where col is in values.

        Parameters
        ----------
        col : str
            Column name to filter on.
        values : array-like
            Accepted values. Uses np.isin so any array-like works.

        Returns
        -------
        NpzDataset
        """
        return self._mask_to_dataset(np.isin(self._data[col], values))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self, cols: Optional[list[str]] = None,
                     time_col: Optional[str] = None) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.

        Datetime handling. If ``time_col`` is given, that one column is cast
        to datetime64. Otherwise columns whose *name* looks date-like
        (matches ``year|month|date|day|time``) AND hold object/string data
        are cast, leaving non-date object columns (e.g. SID, NAME) untouched.
        Columns that fail to parse are left as-is rather than raising.

        Parameters
        ----------
        cols : list[str], optional
            Subset of columns to include. Defaults to all columns.
        time_col : str, optional
            Explicit datetime column to cast. Overrides name detection;
            use when the time column does not match the name pattern.

        Returns
        -------
        pd.DataFrame
        """
        cols = cols or self.columns
        df   = pd.DataFrame({c: self._data[c] for c in cols})

        if time_col is not None:
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
            return df

        # Name-based datetime detection -- cast only date-like *string*
        # columns so string ids (SID, NAME) and numeric fields (a YEAR int
        # column) are never coerced/mangled.
        # Adapted from https://stackoverflow.com/a/79102960
        #   posted by Jayanth MKV, retrieved 2026-06-16, CC BY-SA 4.0.
        # Two project adaptations: 'time' is added to the pattern so
        # ISO_TIME-style columns are caught, and the original
        # ``dtype == 'object'`` guard is broadened to "non-numeric,
        # non-datetime" so it also matches pandas StringDtype columns.
        pattern = "year|month|date|day|time"
        for col in df.columns:
            if (re.search(pattern, col.lower())
                    and not pd.api.types.is_numeric_dtype(df[col])
                    and not pd.api.types.is_datetime64_any_dtype(df[col])):
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    pass
        return df

    def to_Xy(
        self,
        target_cols:  list[str],
        feature_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (X, y) float32 arrays.

        NaN values are preserved — the caller handles masking or imputation.

        Parameters
        ----------
        target_cols : list[str]
            Columns to stack as y.
        feature_cols : list[str]
            Columns to stack as X.

        Returns
        -------
        X : np.ndarray, shape (n, n_features)
        y : np.ndarray, shape (n, n_targets)
        """
        X = np.stack(
            [self._data[c].astype(np.float32) for c in feature_cols], axis=1
        )
        y = np.stack(
            [self._data[c].astype(np.float32) for c in target_cols], axis=1
        )
        return X, y

    # ------------------------------------------------------------------
    # Overrideable hooks
    # ------------------------------------------------------------------

    def split(self, which: str) -> "NpzDataset":
        """Return a predefined split. Must be implemented by subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement split()."
        )

    def summary(self) -> None:
        """Print a concise dataset summary. May be overridden by subclasses."""
        print(f"{self.__class__.__name__} -- {self.npz_path.name}")
        print(f"  rows    : {self._n}")
        print(f"  columns : {len(self.columns)}")
