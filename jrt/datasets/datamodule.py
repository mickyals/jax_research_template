"""
datasets/datamodule.py

DataModule: the interface between raw datasets and the training loop.

Responsibilities
----------------
- Load one or more datasets from a config dict (from the YAML  data:  block)
- Split each into train / val / test from the required data.split config
  block (disjoint per-split column values, validated up front)
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

This is a generic, domain-agnostic array DataModule (the template default for
tabular / array experiments). On-the-fly sample-assembly experiments subclass
BaseDataModule directly with their own setup/loaders (e.g.
experiments.sparse_obs_encoder.data.datamodule.TCDataModule).

Config schema  (YAML  data:  block)
------------------------------------
Single dataset
    data:
      dataset: mysource              # a name registered via @register_dataset
      npz_path: data/mysource.npz    # forwarded to the dataset factory
      target_cols:   [y0, y1]
      feature_cols:  [x0, x1, x2]
      feature_norm:  standard        # standard | minmax | minmax_01 | minmax_11 | none
      target_norm:   standard
      feature_norm_stats: {...}      # optional precomputed stats (skip fitting; see _apply_norm)
      target_norm_stats:  {...}      # optional
      train_shuffle: true            # optional; overrides train_loader default
      val_shuffle:   false           # optional; overrides val_loader default
      split:
        column: group                # any column present in the dataset
        train: {values: [...]}
        val:   {values: [...]}
        test:  {values: [...]}

Multiple datasets
    data:
      datasets:
        - {dataset: mysource, npz_path: ...}
        - {dataset: mysource, npz_path: ...}
      target_cols:  [y0, y1]
      feature_cols: [x0, x1, x2]
      feature_norm: standard
      target_norm:  standard
      split:
        column: group
        train: {values: [...]}
        val:   {values: [...]}
        test:  {values: [...]}

split is required and is shared across all sources — every source must
have the named column. Splitting is a row filter (dataset.filter_column),
so 'train'/'val'/'test' values must be disjoint (validated up front).

Feature-engineering (e.g. encoding lat/lon columns) is the experiment's job, not
this generic module's — see utils.geoscience.coordinates for reusable lat/lon
encoders an experiment can apply to its feature matrix before normalisation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from datasets.batching import as_batches, epoch_iterator, num_batches
from datasets.splitting import validate_disjoint_groups
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
        Names registered so far. Empty until a dataset module that calls
        ``@register_dataset(...)`` has been imported.

    Example
    -------
    >>> from datasets.base import NpzDataset
    >>> @register_dataset("mock")
    ... def _mock(config):
    ...     return NpzDataset(config["npz_path"])
    >>> list_datasets()
    ['mock']
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


def _split_dataset(dataset, split_config: dict, which: str):
    """Filter dataset to the rows belonging to split `which`.

    Parameters
    ----------
    dataset : NpzDataset subclass
        A dataset instance that implements filter_column().
    split_config : dict
        The data.split block — see module docstring for schema.
    which : str
        'train', 'val', or 'test'.

    Returns
    -------
    NpzDataset
        The requested split.
    """
    column = split_config["column"]
    values = split_config[which]["values"]
    return dataset.filter_column(column, values)


def _apply_norm(
    tr:     np.ndarray,
    va:     np.ndarray,
    te:     np.ndarray,
    method: str,
    stats:  dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit normalization on train (or use precomputed stats), apply to all splits.

    Uses utils.jax_core.helpers for the transform. When ``stats`` is None the
    statistics are fit on ``tr`` with np.nanmean / np.nanstd / np.nanmin /
    np.nanmax (tolerating NaN in sparse columns); when ``stats`` is given they
    are used verbatim (no fitting), so a config can supply known physical
    bounds and val/test never leak into the fit.

    Parameters
    ----------
    tr, va, te : np.ndarray  shape (n_samples, n_cols)
    method : str
        'standard' | 'minmax' (== 'minmax_01') | 'minmax_01' | 'minmax_11' | 'none'.
    stats : dict, optional
        Precomputed statistics. For 'standard': {'mean', 'std'}. For any
        minmax variant: {'min', 'max'}. Each a length-n_cols array-like.

    Returns
    -------
    (tr_norm, va_norm, te_norm, stats_dict)
        stats_dict records 'method' plus the fitted/used statistics, and is
        what _invert_norm consumes.
    """
    if method == "standard":
        if stats is not None:
            mean = np.asarray(stats["mean"], dtype=float)[None, :]
            std  = np.asarray(stats["std"],  dtype=float)[None, :]
        else:
            mean = np.nanmean(tr, axis=0, keepdims=True)
            std  = np.nanstd(tr,  axis=0, keepdims=True)
        return (
            np.array(standardise(tr, mean=mean, std=std)),
            np.array(standardise(va, mean=mean, std=std)),
            np.array(standardise(te, mean=mean, std=std)),
            {"method": "standard", "mean": mean.squeeze(0), "std": std.squeeze(0)},
        )
    if method in ("minmax", "minmax_01", "minmax_11"):
        mode = "-11" if method == "minmax_11" else "01"
        if stats is not None:
            lo = np.asarray(stats["min"], dtype=float)
            hi = np.asarray(stats["max"], dtype=float)
        else:
            lo = np.nanmin(tr, axis=0)
            hi = np.nanmax(tr, axis=0)
        return (
            np.array(minmax_norm(tr, lo, hi, mode=mode)),
            np.array(minmax_norm(va, lo, hi, mode=mode)),
            np.array(minmax_norm(te, lo, hi, mode=mode)),
            {"method": method, "min": lo, "max": hi},
        )
    # 'none'
    return tr, va, te, {"method": "none"}


def _invert_norm(y: np.ndarray, stats: dict) -> np.ndarray:
    """Invert the normalisation applied by _apply_norm."""
    method = stats.get("method", "none")
    if method == "standard":
        return y * stats["std"] + stats["mean"]
    if method in ("minmax", "minmax_01"):
        return y * (stats["max"] - stats["min"]) + stats["min"]
    if method == "minmax_11":
        return (y + 1.0) / 2.0 * (stats["max"] - stats["min"]) + stats["min"]
    return y


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseDataModule(ABC):
    """The contract every DataModule satisfies — exactly what the Trainer needs.

    The Trainer iterates LOADERS; it never touches in-memory arrays. So the
    abstract surface is just:
        setup(config)
        train_loader(...) / val_loader(...) / test_loader(...)
            → re-iterable objects yielding batch dicts
    plus the optional conveniences ``manifest()`` and ``denormalise_targets()``
    the Trainer calls for the run manifest and physical-unit recovery.

    Array-backed datamodules (the generic tabular ``DataModule`` below) ALSO
    expose ``train_arrays`` / ``val_arrays`` / ``test_arrays`` for inspection and
    tests, but those belong to the array path — they are NOT part of this
    contract. On-the-fly samplers (e.g.
    experiments.sparse_obs_encoder.data.datamodule.TCDataModule) implement only
    the loaders, with no arrays to stub out.
    """

    @abstractmethod
    def setup(self, config: dict) -> None:
        """Load / prepare the splits. Called once before Trainer.fit()."""

    @abstractmethod
    def train_loader(self, batch_size: int, **kwargs):
        """Re-iterable of training batches — the object Trainer.fit iterates."""

    @abstractmethod
    def val_loader(self, batch_size: int, **kwargs):
        """Re-iterable of validation batches."""

    @abstractmethod
    def test_loader(self, batch_size: int, **kwargs):
        """Re-iterable of test batches — the object Trainer.test iterates."""

    @property
    def norm_stats(self) -> dict:
        return {}

    def denormalise_targets(self, y: np.ndarray) -> np.ndarray:
        """Invert target normalisation to recover physical units."""
        return _invert_norm(np.asarray(y), self.norm_stats.get("target", {"method": "none"}))

    def manifest(self) -> dict:
        """JSON-serialisable summary of what this run trained on.

        Default is empty. Subclasses that resolve data splits (e.g. via a
        splits resolver) should override this to return the resolved
        seasons/SIDs/row counts per split — see Trainer.write_manifest.
        """
        return {}

    def summary(self) -> None:
        """Print a human-readable overview of the prepared splits."""
        print(f"{self.__class__.__name__}: no summary implemented.")


# ---------------------------------------------------------------------------
# Concrete DataModule
# ---------------------------------------------------------------------------

class DataModule(BaseDataModule):
    """Coordinates one or more datasets into train / val / test splits.

    Handles loading, concatenation, and normalisation.  Splitting is a
    generic column-value filter (see data.split in the module docstring),
    applied via dataset.filter_column().

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
        target_cols   = config["target_cols"]
        feature_cols  = config["feature_cols"]
        feat_norm     = config.get("feature_norm", "standard")
        tgt_norm      = config.get("target_norm",  "standard")
        feat_stats_pre = config.get("feature_norm_stats")   # optional precomputed
        tgt_stats_pre  = config.get("target_norm_stats")    # optional precomputed

        split_config = config["split"]
        validate_disjoint_groups({
            name: split_config[name]["values"] for name in ("train", "val", "test")
        })

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
            train_ds = _split_dataset(ds, split_config, "train")
            val_ds   = _split_dataset(ds, split_config, "val")
            test_ds  = _split_dataset(ds, split_config, "test")

            X_tr, y_tr = train_ds.to_Xy(target_cols, feature_cols)
            X_va, y_va = val_ds.to_Xy(target_cols, feature_cols)
            X_te, y_te = test_ds.to_Xy(target_cols, feature_cols)

            tr_X_parts.append(X_tr);  tr_y_parts.append(y_tr)
            va_X_parts.append(X_va);  va_y_parts.append(y_va)
            te_X_parts.append(X_te);  te_y_parts.append(y_te)

        X_tr = np.concatenate(tr_X_parts, axis=0)
        y_tr = np.concatenate(tr_y_parts, axis=0)
        X_va = np.concatenate(va_X_parts, axis=0)
        y_va = np.concatenate(va_y_parts, axis=0)
        X_te = np.concatenate(te_X_parts, axis=0)
        y_te = np.concatenate(te_y_parts, axis=0)

        X_tr, X_va, X_te, feat_stats = _apply_norm(
            X_tr, X_va, X_te, feat_norm, stats=feat_stats_pre)
        y_tr, y_va, y_te, tgt_stats  = _apply_norm(
            y_tr, y_va, y_te, tgt_norm, stats=tgt_stats_pre)

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
        split_config = self._config["split"]
        print(f"  split column={split_config['column']!r}")
        for name in ("train", "val", "test"):
            print(f"    {name:<6}: values={split_config[name]['values']}")

    def manifest(self) -> dict:
        """Resolved split values and row counts per split.

        See BaseDataModule.manifest for how this is consumed by Trainer.
        """
        split_config = self._config["split"]
        manifest: dict = {"split": {"column": split_config["column"]}}
        for name, key in [("train", "_train"), ("val", "_val"), ("test", "_test")]:
            manifest[name] = {
                "values": split_config[name]["values"],
                "n_rows": int(getattr(self, key)["X"].shape[0]),
            }
        return manifest


# ---------------------------------------------------------------------------
# Dataset factories
# This module ships no built-in dataset factories — keeping it free of any
# dependency on specific experiments. A dataset source registers itself with
# @register_dataset("NAME") in its own module (e.g.
# experiments/sparse_obs_encoder/data/sources/ibtracs.py registers
# "IBTRACS"); importing that module before DataModule.from_config() makes the
# name available.
# ---------------------------------------------------------------------------
