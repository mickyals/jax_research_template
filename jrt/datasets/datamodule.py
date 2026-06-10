"""
datasets/datamodule.py

DataModule: the interface between raw datasets and the training loop.

Responsibilities
----------------
- Load one or more datasets from a config dict (from the YAML  data:  block)
- Split each into train / val / test using the dataset's own split() method
- Concatenate multiple datasets along the sample axis
- Normalise features and targets using training-set statistics only
  (no leakage from val / test into the normalisation fit)
- Expose train / val / test as ArrayLoader objects that the Trainer iterates
- Store inverse-transform statistics for denormalising predictions back
  to physical units

Dataset vs data loading
-----------------------
DataModule is responsible for WHAT data (splits, normalisation, schema).
ArrayLoader is responsible for HOW it is served (batching, shuffling).

The Trainer calls fit(dm.train_loader(...), dm.val_loader(...)) and iterates
the loaders.  Any object that yields {'X': array, 'y': array} dicts works —
so a PyTorch DataLoader with a thin JAX-conversion wrapper also plugs in.

Dataset registry
----------------
register_dataset  /  list_datasets

Normalisation
-------------
Uses utils.jax_core.helpers.standardise and helpers.minmax_norm.

Config schema  (YAML  data:  block)
------------------------------------
Single dataset
    data:
      dataset: ibtracs
      npz_path: data/ibtracs_tc_clean.npz
      multi_storm_path: data/ibtracs_multi_storm_times.npz  # optional
      target_cols:   [USA_WIND, USA_PRES]
      feature_cols:  [LAT, LON, STORM_SPEED, STORM_DIR]
      feature_norm:  standard   # standard | minmax | none
      target_norm:   standard
      train_shuffle: true       # optional; overrides train_loader default
      val_shuffle:   false      # optional; overrides val_loader default

Multiple datasets
    data:
      datasets:
        - {dataset: ibtracs, npz_path: ..., multi_storm_path: ...}
        - {dataset: ibtracs, npz_path: ...}
      target_cols:  [USA_WIND, USA_PRES]
      feature_cols: [LAT, LON, STORM_SPEED, STORM_DIR]
      feature_norm: standard
      target_norm:  standard

Positional encoding  (optional; applied after to_Xy, before normalisation)
    data:
      ...
      position_encoding_mode: unit_sphere   # storm_relative_polar | unit_sphere | domain_normalised
      coord_cols: [LAT, LON]                # the two columns to encode (must be in feature_cols)

      # storm_relative_polar only:
      storm_coord_cols: [STORM_LAT, STORM_LON]  # storm centre columns; defaults to coord_cols
      radius_km: 500.0

      # domain_normalised only:
      field_of_view:
        lat_min:  0.0
        lat_max: 30.0
        lon_min: -100.0
        lon_max:  -45.0

    coord_cols are removed from X and replaced by the encoded output (3 columns
    for storm_relative_polar / unit_sphere, 2 for domain_normalised).
    storm_coord_cols (if different from coord_cols) are also removed from X.
    All remaining feature_cols are preserved and concatenated after the encoding.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from datasets.batching import as_batches, epoch_iterator, num_batches
from utils.jax_core.helpers import create_rng, minmax_norm, standardise


# ---------------------------------------------------------------------------
# ArrayLoader
# ---------------------------------------------------------------------------

class ArrayLoader:
    """Re-iterable loader that yields batches from an in-memory array dict.

    Each call to ``__iter__`` (i.e. each ``for batch in loader:`` loop)
    produces a fresh pass over the data.  When ``shuffle=True``, the
    ordering varies each pass via ``jax.random.fold_in(seed, call_count)``
    so epochs are shuffled differently without external seed management.

    Parameters
    ----------
    arrays : dict[str, jax.Array]
        {'X': ..., 'y': ...} arrays.  Must share the same leading dimension.
    batch_size : int
    shuffle : bool
        True  → shuffle at the start of each pass (default for train loaders).
        False → fixed order (default for val / test loaders).
    seed : int
        Base RNG seed.  Combined with a per-pass counter so successive
        passes get different shuffles deterministically.
    drop_last : bool or None
        True  → discard the last incomplete batch (keeps all batches the
                 same size — required for JIT-compiled train steps).
        False → yield the last incomplete batch (use for val / test so
                 every sample contributes to the metric).
        None  → mirror the shuffle flag (True when shuffle=True).

    Examples
    --------
    >>> loader = ArrayLoader(arrays, batch_size=32, shuffle=True, seed=0)
    >>> for batch in loader:          # epoch 1 — one shuffle
    ...     train_step(batch)
    >>> for batch in loader:          # epoch 2 — different shuffle
    ...     train_step(batch)
    """

    def __init__(
        self,
        arrays:     dict,
        batch_size: int,
        shuffle:    bool = True,
        seed:       int  = 0,
        drop_last:  bool | None = None,
    ) -> None:
        self._arrays     = arrays
        self._batch_size = batch_size
        self._shuffle    = shuffle
        self._seed       = seed
        self._drop_last  = shuffle if drop_last is None else drop_last
        self._call_count = 0

    def __iter__(self):
        if self._shuffle:
            rng = jax.random.fold_in(create_rng(self._seed), self._call_count)
            gen = epoch_iterator(
                self._arrays, self._batch_size, rng,
                drop_last=self._drop_last,
            )
        else:
            gen = as_batches(
                self._arrays, self._batch_size,
                drop_last=self._drop_last,
            )
        self._call_count += 1
        yield from gen

    def __len__(self) -> int:
        n = next(iter(self._arrays.values())).shape[0]
        return num_batches(n, self._batch_size, drop_last=self._drop_last)


# ---------------------------------------------------------------------------
# Dataset factory registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, callable] = {}


def register_dataset(name: str):
    """Register a dataset factory function by name.

    The factory takes a single config dict and returns a dataset instance
    (any subclass of NpzDataset).  Adding a new data source is just:

        @register_dataset("mysource")
        def _mysource(config):
            from datasets.mysource.dataset import MyDataset
            return MyDataset(config["npz_path"])

    Parameters
    ----------
    name : str
        Registry key (case-insensitive).

    Returns
    -------
    callable
        Function decorator.

    Raises
    ------
    ValueError
        If the name is already registered.

    Example
    -------
    >>> @register_dataset("mock")
    ... def _mock(config):
    ...     from datasets.base import NpzDataset
    ...     return NpzDataset(config["npz_path"])
    """
    name = name.upper()

    def decorator(fn):
        if name in DATASETS:
            raise ValueError(f"Dataset '{name}' is already registered.")
        DATASETS[name] = fn
        return fn

    return decorator


def list_datasets() -> list[str]:
    """Return all registered dataset names.

    Returns
    -------
    list[str]

    Example
    -------
    >>> list_datasets()
    ['IBTRACS']
    """
    return sorted(DATASETS.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_dataset(ds_config: dict):
    """Instantiate a dataset from its config sub-dict."""
    name = ds_config.get("dataset", "").upper()
    if name not in DATASETS:
        available = ", ".join(list_datasets()) or "none"
        raise ValueError(
            f"Dataset '{name}' is not registered. Available: {available}"
        )
    return DATASETS[name](ds_config)


def _split_dataset(dataset, which: str):
    """Call dataset.split(which).

    Parameters
    ----------
    dataset : NpzDataset subclass
        A dataset instance that implements split().
    which : str
        'train', 'val', or 'test'.

    Returns
    -------
    NpzDataset
        The requested split.
    """
    return dataset.split(which)


def _apply_norm(
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit normalisation on train, apply to all three splits.

    Uses utils.jax_core.helpers for the transform. Training statistics
    are computed with np.nanmean / np.nanstd to tolerate NaN values in
    sparse target columns.

    Parameters
    ----------
    tr, va, te : np.ndarray  shape (n_samples, n_cols)
    method : str  'standard' | 'minmax' | 'none'

    Returns
    -------
    (tr_norm, va_norm, te_norm, stats_dict)
    """
    if method == "standard":
        mean = np.nanmean(tr, axis=0, keepdims=True)
        std  = np.nanstd(tr,  axis=0, keepdims=True)
        return (
            np.array(standardise(tr, mean=mean, std=std)),
            np.array(standardise(va, mean=mean, std=std)),
            np.array(standardise(te, mean=mean, std=std)),
            {
                "method": "standard",
                "mean": mean.squeeze(0),
                "std":  std.squeeze(0),
            },
        )
    if method == "minmax":
        lo = np.nanmin(tr, axis=0)
        hi = np.nanmax(tr, axis=0)
        return (
            np.array(minmax_norm(tr, lo, hi, mode="01")),
            np.array(minmax_norm(va, lo, hi, mode="01")),
            np.array(minmax_norm(te, lo, hi, mode="01")),
            {
                "method": "minmax",
                "min": lo,
                "max": hi,
            },
        )
    # 'none'
    return tr, va, te, {"method": "none"}


def _invert_norm(y: np.ndarray, stats: dict) -> np.ndarray:
    """Invert the normalisation applied by _apply_norm."""
    if stats["method"] == "standard":
        return y * stats["std"] + stats["mean"]
    if stats["method"] == "minmax":
        return y * (stats["max"] - stats["min"]) + stats["min"]
    return y


def _apply_position_encoding(
    X:            np.ndarray,
    feature_cols: list[str],
    config:       dict,
) -> np.ndarray:
    """Replace coordinate columns in X with their positional encoding.

    Supported modes: 'unit_sphere', 'domain_normalised'.
    Coordinate columns are extracted by index in feature_cols, encoded,
    and concatenated with the remaining columns.

    Parameters
    ----------
    X : np.ndarray  shape (n, len(feature_cols))
    feature_cols : list[str]
    config : dict
        Must contain 'position_encoding_mode' and 'coord_cols'.
        domain_normalised also requires 'field_of_view' with lat/lon bounds.

    Returns
    -------
    np.ndarray  shape (n, len(feature_cols) - n_dropped + enc_dim)
    """
    import numpy as _np

    mode       = config["position_encoding_mode"]
    coord_cols = config["coord_cols"]

    try:
        coord_idx = [feature_cols.index(c) for c in coord_cols]
    except ValueError as exc:
        raise KeyError(f"coord_cols column not found in feature_cols: {exc}") from exc

    lat      = X[:, coord_idx[0]].astype(_np.float32)
    lon      = X[:, coord_idx[1]].astype(_np.float32)
    drop_idx = set(coord_idx)

    if mode == "unit_sphere":
        lat_r   = _np.radians(lat)
        lon_r   = _np.radians(lon)
        encoded = _np.stack([
            _np.cos(lat_r) * _np.cos(lon_r),
            _np.cos(lat_r) * _np.sin(lon_r),
            _np.sin(lat_r),
        ], axis=1)

    elif mode == "domain_normalised":
        fov      = config["field_of_view"]
        lat_norm = 2.0 * (lat - fov["lat_min"]) / (fov["lat_max"] - fov["lat_min"]) - 1.0
        lon_norm = 2.0 * (lon - fov["lon_min"]) / (fov["lon_max"] - fov["lon_min"]) - 1.0
        encoded  = _np.stack([lat_norm, lon_norm], axis=1)

    else:
        raise ValueError(
            f"Unknown position_encoding_mode '{mode}'. "
            "Choose from: 'unit_sphere', 'domain_normalised'."
        )

    keep_idx = [i for i in range(X.shape[1]) if i not in drop_idx]
    X_keep   = X[:, keep_idx] if keep_idx else _np.empty((X.shape[0], 0), dtype=X.dtype)
    return _np.concatenate([X_keep, encoded], axis=1)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseDataModule(ABC):
    """Protocol that every DataModule must satisfy.

    Primary interface for the Trainer:
        setup(config)
        train_loader(batch_size, seed, shuffle) → ArrayLoader
        val_loader(batch_size, shuffle)         → ArrayLoader
        test_loader(batch_size, shuffle)        → ArrayLoader

    The array accessors (train_arrays / val_arrays / test_arrays) are kept
    for inspection and testing but are not passed directly to the Trainer.
    """

    @abstractmethod
    def setup(self, config: dict) -> None:
        """Load, split, and normalise. Called once before Trainer.fit()."""

    @abstractmethod
    def train_arrays(self) -> dict[str, jnp.ndarray]:
        """{'X': features, 'y': targets} for the training split (all rows)."""

    @abstractmethod
    def val_arrays(self) -> dict[str, jnp.ndarray]:
        """{'X': features, 'y': targets} for the validation split (all rows)."""

    @abstractmethod
    def test_arrays(self) -> dict[str, jnp.ndarray]:
        """{'X': features, 'y': targets} for the test split (all rows)."""

    # ------------------------------------------------------------------
    # Loader interface — default implementations wrap the array accessors.
    # Subclasses may override to customise drop_last, seed, or to serve
    # data from disk rather than from in-memory arrays.
    # ------------------------------------------------------------------

    def train_loader(
        self,
        batch_size: int,
        seed:       int  = 0,
        shuffle:    bool = True,
    ) -> ArrayLoader:
        """Return an ArrayLoader over the training split.

        Parameters
        ----------
        batch_size : int
        seed : int
            Base RNG seed.  Each epoch gets a different shuffle via
            fold_in so successive passes vary deterministically.
        shuffle : bool
            True (default) — reshuffle every epoch.
            False          — fixed order (e.g. for curriculum learning or
                             time-series data where order matters).
        """
        return ArrayLoader(
            self.train_arrays(), batch_size,
            shuffle=shuffle, seed=seed, drop_last=True,
        )

    def val_loader(
        self,
        batch_size: int,
        shuffle:    bool = False,
    ) -> ArrayLoader:
        """Return an ArrayLoader over the validation split.

        drop_last=False so every sample contributes to validation metrics.
        """
        return ArrayLoader(
            self.val_arrays(), batch_size,
            shuffle=shuffle, drop_last=False,
        )

    def test_loader(
        self,
        batch_size: int,
        shuffle:    bool = False,
    ) -> ArrayLoader:
        """Return an ArrayLoader over the test split.

        drop_last=False so every sample contributes to test metrics.
        """
        return ArrayLoader(
            self.test_arrays(), batch_size,
            shuffle=shuffle, drop_last=False,
        )

    @property
    def norm_stats(self) -> dict:
        return {}

    def denormalise_targets(self, y: np.ndarray) -> np.ndarray:
        """Invert target normalisation to recover physical units."""
        return _invert_norm(np.asarray(y), self.norm_stats.get("target", {"method": "none"}))

    def summary(self) -> None:
        """Print a human-readable overview of the prepared splits."""
        print(f"{self.__class__.__name__}: no summary implemented.")


# ---------------------------------------------------------------------------
# Concrete DataModule
# ---------------------------------------------------------------------------

class DataModule(BaseDataModule):
    """Coordinates one or more datasets into train / val / test splits.

    Handles loading, concatenation, and normalisation.  Dataset-specific
    split logic lives in each dataset class — the DataModule just calls
    dataset.split().

    Create via the classmethod:
        dm = DataModule.from_config(config)

    Or manually:
        dm = DataModule()
        dm.setup(config)
    """

    @classmethod
    def from_config(cls, config: dict) -> "DataModule":
        """Convenience constructor: instantiate and set up in one call.

        Parameters
        ----------
        config : dict
            The YAML  data:  block loaded into a dict.

        Returns
        -------
        DataModule  ready for use with Trainer.fit()
        """
        dm = cls()
        dm.setup(config)
        return dm

    def setup(self, config: dict) -> None:
        """Load datasets, split, concatenate, normalise.

        Accepts either a single dataset (config has a 'dataset' key) or
        multiple datasets (config has a 'datasets' list).  All sources must
        share the same feature_cols and target_cols.
        """
        target_cols  = config["target_cols"]
        feature_cols = config["feature_cols"]
        feat_norm    = config.get("feature_norm", "standard")
        tgt_norm     = config.get("target_norm",  "standard")
        enc_mode     = config.get("position_encoding_mode")

        # Normalise to a list of sub-configs
        if "datasets" in config:
            ds_configs = [
                {**config, **sub}
                for sub in config["datasets"]
            ]
        else:
            ds_configs = [config]

        tr_X_parts, tr_y_parts = [], []
        va_X_parts, va_y_parts = [], []
        te_X_parts, te_y_parts = [], []

        for ds_cfg in ds_configs:
            ds       = _build_dataset(ds_cfg)
            train_ds = _split_dataset(ds, "train")
            val_ds   = _split_dataset(ds, "val")
            test_ds  = _split_dataset(ds, "test")

            X_tr, y_tr = train_ds.to_Xy(target_cols, feature_cols)
            X_va, y_va = val_ds.to_Xy(target_cols, feature_cols)
            X_te, y_te = test_ds.to_Xy(target_cols, feature_cols)

            if enc_mode is not None:
                X_tr = _apply_position_encoding(X_tr, feature_cols, config)
                X_va = _apply_position_encoding(X_va, feature_cols, config)
                X_te = _apply_position_encoding(X_te, feature_cols, config)

            tr_X_parts.append(X_tr);  tr_y_parts.append(y_tr)
            va_X_parts.append(X_va);  va_y_parts.append(y_va)
            te_X_parts.append(X_te);  te_y_parts.append(y_te)

        X_tr = np.concatenate(tr_X_parts, axis=0)
        y_tr = np.concatenate(tr_y_parts, axis=0)
        X_va = np.concatenate(va_X_parts, axis=0)
        y_va = np.concatenate(va_y_parts, axis=0)
        X_te = np.concatenate(te_X_parts, axis=0)
        y_te = np.concatenate(te_y_parts, axis=0)

        X_tr, X_va, X_te, feat_stats = _apply_norm(X_tr, X_va, X_te, feat_norm)
        y_tr, y_va, y_te, tgt_stats  = _apply_norm(y_tr, y_va, y_te, tgt_norm)

        self._train      = {"X": jnp.array(X_tr), "y": jnp.array(y_tr)}
        self._val        = {"X": jnp.array(X_va), "y": jnp.array(y_va)}
        self._test       = {"X": jnp.array(X_te), "y": jnp.array(y_te)}
        self._norm_stats = {"feature": feat_stats, "target": tgt_stats}
        self._config     = config

    def train_arrays(self) -> dict[str, jnp.ndarray]:
        return self._train

    def val_arrays(self) -> dict[str, jnp.ndarray]:
        return self._val

    def test_arrays(self) -> dict[str, jnp.ndarray]:
        return self._test

    # ------------------------------------------------------------------
    # Loaders — read shuffle defaults from config, allow caller override
    # ------------------------------------------------------------------

    def train_loader(
        self,
        batch_size: int,
        seed:       int  = 0,
        shuffle:    bool | None = None,
    ) -> ArrayLoader:
        """Train loader; shuffle defaults to config['train_shuffle'] (True)."""
        if shuffle is None:
            shuffle = self._config.get("train_shuffle", True)
        return ArrayLoader(
            self._train, batch_size,
            shuffle=shuffle, seed=seed, drop_last=True,
        )

    def val_loader(
        self,
        batch_size: int,
        shuffle:    bool | None = None,
    ) -> ArrayLoader:
        """Val loader; shuffle defaults to config['val_shuffle'] (False)."""
        if shuffle is None:
            shuffle = self._config.get("val_shuffle", False)
        return ArrayLoader(
            self._val, batch_size,
            shuffle=shuffle, drop_last=False,
        )

    def test_loader(
        self,
        batch_size: int,
        shuffle:    bool | None = None,
    ) -> ArrayLoader:
        """Test loader; shuffle defaults to config['test_shuffle'] (False)."""
        if shuffle is None:
            shuffle = self._config.get("test_shuffle", False)
        return ArrayLoader(
            self._test, batch_size,
            shuffle=shuffle, drop_last=False,
        )

    @property
    def norm_stats(self) -> dict:
        return self._norm_stats

    def summary(self) -> None:
        n_sources = len(self._config.get("datasets", [self._config]))
        print(f"DataModule  ({n_sources} source{'s' if n_sources > 1 else ''})")
        for name, key in [("train", "_train"), ("val", "_val"), ("test", "_test")]:
            arrs  = getattr(self, key)
            n     = arrs["X"].shape[0]
            y     = np.array(arrs["y"])
            pct   = 100.0 * np.isfinite(y).sum() / y.size
            print(f"  {name:<6}: {n:>7} samples  targets {pct:.1f}% finite")
        f_m = self._norm_stats["feature"]["method"]
        t_m = self._norm_stats["target"]["method"]
        print(f"  feature_norm={f_m}   target_norm={t_m}")


# ---------------------------------------------------------------------------
# Built-in dataset factories
# Additional factories can be registered with @register_dataset("NAME")
# in experiment-specific modules and imported before DataModule.from_config().
# ---------------------------------------------------------------------------

@register_dataset("IBTRACS")
def _ibtracs_factory(config: dict):
    from experiments.sparse_obs_cross_attn.data.sources.ibtracs import IBTrACSDataset
    return IBTrACSDataset(
        config["npz_path"],
        config.get("multi_storm_path"),
    )
