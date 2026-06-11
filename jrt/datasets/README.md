# jrt/datasets

Generic data loading and batching infrastructure. This layer handles how data is read, split, and served to the Trainer — it does not contain experiment-specific logic.

Experiment-specific datasets (custom loaders, domain filtering, coordinate encoding) live inside the experiment directory, typically as a subclass of `NpzDataset` or `BaseDataModule`.

---

## Files

### `base.py` — `NpzDataset`

Loads a `.npz` archive and exposes it as a structured dataset.

```python
from datasets.base import NpzDataset

ds = NpzDataset("data/ibtracs_full.npz")

# Inspect
print(ds.columns)            # list of array keys
print(ds.n_samples)          # number of rows

# Filter to a subset of rows
storm_ds = ds.filter_column("SEASON", lambda v: v >= 2005)

# Export
df = ds.to_dataframe()       # pandas DataFrame
X, y = ds.to_Xy(feature_cols=["LAT", "LON"], target_cols=["USA_WIND"])
```

`NpzDataset` is designed to be subclassed. Experiments override `__init__` to add domain-specific filtering and column derivation. Train/val/test splitting is **not** baked into dataset classes — datasets expose filter primitives (`filter_column` and experiment-specific filters), and split *policy* comes from config (see `datamodule.py` and `splitting.py`).

---

### `batching.py` — iterators and loaders

Low-level utilities for iterating over NumPy arrays in batches.

| Function | Description |
|----------|-------------|
| `as_batches(X, y, batch_size)` | Yield `{"X": ..., "y": ...}` dicts of size `batch_size` |
| `shuffle_arrays(*arrays, rng)` | Shuffle multiple arrays in sync |
| `epoch_iterator(X, y, batch_size, seed, shuffle)` | One full pass with optional shuffle |
| `num_batches(n_samples, batch_size)` | Number of complete + partial batches |

These are the lowest-level building blocks. `ArrayLoader` (in `datamodule.py`) wraps them into a re-iterable object.

---

### `splitting.py` — group-split helpers

Generic mechanism for group-based splits (split by storm, by season, by station — any per-row group id). Policy (which groups go where) belongs to callers: the `data.split` config block or an experiment-side resolver. Pure numpy, no pandas.

| Function | Description |
|----------|-------------|
| `validate_disjoint_groups(groups)` | Raise `ValueError` if any value is assigned to two splits |
| `group_mask(row_groups, groups)` | Boolean row mask: rows whose group id is in `groups` |
| `assign_groups_by_fraction(groups, fraction, seed, stratify_by=None)` | Seeded fraction-based selection of unique groups; with `stratify_by`, the fraction holds per stratum and every non-empty stratum contributes at least one group (floor rule — a 4-group stratum at fraction 0.2 would otherwise round to zero) |

---

### `datamodule.py` — `DataModule` and `ArrayLoader`

**`ArrayLoader`** — a re-iterable loader around in-memory NumPy arrays.

```python
from datasets.datamodule import ArrayLoader

loader = ArrayLoader(X, y, batch_size=64, shuffle=True, seed=42)
for batch in loader:          # batch = {"X": ..., "y": ...}
    ...
# Iterating again starts from scratch (re-iterable, not one-shot)
```

**`BaseDataModule`** — abstract base class for all experiment DataModules.

Subclasses must implement:
- `train_loader(**kwargs) -> iterable`
- `val_loader(**kwargs) -> iterable`
- `test_loader(**kwargs) -> iterable`

Optionally override `manifest() -> dict` (default `{}`): a JSON-serialisable summary of what the run trains on (resolved split membership, row counts, ...). The Trainer's `write_manifest()` persists it next to the checkpoints and pushes it to the logger.

**`DataModule`** — generic concrete implementation for array-based datasets.

```python
from datasets.datamodule import DataModule

dm = DataModule.from_config(config["data"])
trainer.fit(dm.train_loader(batch_size=64), dm.val_loader(batch_size=64))
```

Handles feature/target normalisation (standardise or minmax) using training-set statistics only — no leakage into val/test.

The `data:` config block **requires a `split:` section** — splitting is config policy applied via `dataset.filter_column`, not a method on the dataset:

```yaml
data:
  dataset: ibtracs
  npz_path: ...
  split:
    column: SEASON            # any column present in the dataset
    train: {values: [2005, 2006, ..., 2020]}
    val:   {values: [2021, 2022]}
    test:  {values: [2023, 2024, 2025]}
```

Per-split values must be disjoint (checked with `splitting.validate_disjoint_groups` before any filtering). `DataModule.manifest()` reports the resolved split (column, per-split values, row counts).

---

## Adding a dataset for a new experiment

Subclass `NpzDataset` in your experiment directory. Add domain filtering and derived columns — but no split logic; splits come from config:

```python
# jrt/experiments/my_experiment/dataset.py
from datasets.base import NpzDataset

class MyDataset(NpzDataset):
    def __init__(self, path: str):
        super().__init__(path)
        # domain filtering, column derivation — no split logic here
        self._data = {k: v for k, v in self._data.items()}  # e.g. drop bad rows
```

Then subclass `BaseDataModule` to wrap it, resolving the split from the config (`filter_column` for simple value splits; `datasets.splitting` helpers for group/fraction splits):

```python
# jrt/experiments/my_experiment/datamodule.py
from datasets.datamodule import BaseDataModule, ArrayLoader

class MyDataModule(BaseDataModule):
    def __init__(self, config: dict):
        full     = MyDataset(config["path"])
        split    = config["split"]                       # {column, train/val/test values}
        train_ds = full.filter_column(split["column"], split["train"]["values"])
        val_ds   = full.filter_column(split["column"], split["val"]["values"])
        self._train, self._val = train_ds.to_Xy(...), val_ds.to_Xy(...)

    def train_loader(self, batch_size=64, seed=42, shuffle=True):
        X, y = self._train
        return ArrayLoader(X, y, batch_size=batch_size, shuffle=shuffle, seed=seed)

    def val_loader(self, batch_size=64, **_):
        X, y = self._val
        return ArrayLoader(X, y, batch_size=batch_size, shuffle=False)

    def test_loader(self, **kw):
        ...
```

The Trainer is indifferent to the loader type as long as it yields `{"X": array, "y": array}` dicts and is re-iterable.
