# configs/ — usage

Composable yaml, three concerns. A **train config is the entry point**;
notebooks/CLI only ever load one file:

```python
from experiments.cyclone_jax.config import load_config, CONFIG_DIR
cfg = load_config(CONFIG_DIR / 'train' / 'train.yaml')
# -> {'data': <scenario dict>, 'model': <dict|None>, 'trainer': <dict>}
```

```
configs/
├── data/     # run SCENARIOS — self-contained (no inheritance; diffable):
│   #  overfit.yaml     stratified n_per_class subset, train==val (memorisation gate)
│   #  train.yaml       year split, multi-driver fixes excluded  !! edit year lists
│   #  test.yaml        same years, evaluation passes            !! keep in sync
│   #  multistorm.yaml  the excluded multi-driver fixes as OOD test
├── models/   # one yaml per model (empty until the models phase)
└── train/    # train.yaml: data: <scenario>, model: <name|null>, trainer: {...}
```

Rules enforced by `config.py`:

- Pointers resolve to `configs/data/<name>.yaml` / `configs/models/<name>.yaml`;
  a missing file fails with the resolved path.
- Every block is key-set validated — an unknown key (typo) is an ERROR.
  Extend `DATA_KEYS` / `SPLIT_KEYS` / `TRAINER_KEYS` when adding config surface.
- Value-level validation lives with the consumers
  (`resolve_input` / `resolve_target` / `build_data` / Trainer).
- `trainer.seed` is THE seed: jax model init + numpy data order.
- Lookback schedules are NOT config — the bookshelf's baked deltas are the
  source of truth (`library.load_library` verifies against `LOOKBACK_DELTAS`).

Tests round-trip every shipped scenario through the specs
(`tests/experiments/cyclone_jax/test_config.py`) — config drift fails CI.
