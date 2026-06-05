# src/datasets

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

`NpzDataset` is designed to be subclassed. Experiments override `__init__` to add domain-specific filtering, column derivation, and train/val/test splitting.

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

**`DataModule`** — generic concrete implementation for array-based datasets.

```python
from datasets.datamodule import DataModule

dm = DataModule.from_config(config["data"])
trainer.fit(dm.train_loader(batch_size=64), dm.val_loader(batch_size=64))
```

Handles feature/target normalisation (standardise or minmax) using training-set statistics only — no leakage into val/test.

---

## Adding a dataset for a new experiment

Subclass `NpzDataset` in your experiment directory:

```python
# src/experiments/my_experiment/dataset.py
from datasets.base import NpzDataset

class MyDataset(NpzDataset):
    def __init__(self, path: str, split: str = "train"):
        super().__init__(path)
        # domain filtering, column derivation, split logic
        years = self._data["YEAR"]
        if split == "train":
            mask = years < 2021
        elif split == "val":
            mask = (years >= 2021) & (years < 2023)
        else:
            mask = years >= 2023
        self._data = {k: v[mask] for k, v in self._data.items()}
```

Then subclass `BaseDataModule` to wrap it:

```python
# src/experiments/my_experiment/datamodule.py
from datasets.datamodule import BaseDataModule, ArrayLoader

class MyDataModule(BaseDataModule):
    def __init__(self, config: dict):
        train_ds = MyDataset(config["path"], split="train")
        val_ds   = MyDataset(config["path"], split="val")
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
