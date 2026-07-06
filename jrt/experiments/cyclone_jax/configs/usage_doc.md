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
│   #  train_land.yaml / train_marine.yaml   single-source variants (same
│   #              splits; pad_to stays 1536 so architectures stay identical)
│   #  memorise.yaml    FULL-dataset memorisation probe (train==val==all fixes)
│   #  memorise_land.yaml / memorise_marine.yaml   its source variants
│   #  test.yaml        same years, evaluation passes            !! keep in sync
│   #  multistorm.yaml  the excluded multi-driver fixes as OOD test
│   #  site overrides: shell CYCLONE_JAX_ROOT / CYCLONE_JAX_NUM_WORKERS
│   #  beat the yaml root / num_workers (machine properties)
│   #  channels: filters the sources' union (per-source availability
│   #  table + source-comparable recipe: data/usage_doc.md)
├── models/   # one yaml per model, self-contained + wandb tags:
│   #  mlp.yaml    StationMLP baseline (activation relu|gelu|silu|leaky_relu,
│   #              encoding: concat|additive, embedding null = raw coords)
│   #  siren.yaml  StationSIREN — raw coords, omegas
│   #  finer.yaml  StationFINER — siren keys + bias_k
│   #  n_classes stays null — injected from TargetSpec by models.build_model
└── train/    # train.yaml: data: <scenario>, model: <name|null>, trainer: {...}
          #  top-level gpu: pins CUDA_VISIBLE_DEVICES (--gpu CLI overrides;
          #  a shell setting always wins)
          #  tune_*.yaml: HP search over a base train yaml (train/tune.py)
          #  — dotted data./model./trainer. search paths, direction derives
          #  from the base's patience_direction, record = trials.csv +
          #  per-trial wandb runs (group = study) + merged best.yaml
```

Rules enforced by `config.py`:

- Pointers resolve to `configs/data/<name>.yaml` / `configs/models/<name>.yaml`;
  a missing file fails with the resolved path.
- Every block is key-set validated — an unknown key (typo) is an ERROR.
  Extend `DATA_KEYS` / `SPLIT_KEYS` / `TRAINER_KEYS` / `MODEL_KEYS` when
  adding config surface (model blocks are keyed per model name).
- Value-level validation lives with the consumers
  (`resolve_input` / `resolve_target` / `build_data` / Trainer).
- `trainer.seed` is THE seed: jax model init + numpy data order.
- Lookback schedules are NOT config — the bookshelf's baked deltas are the
  source of truth (`library.load_library` verifies against `LOOKBACK_DELTAS`).

Tests round-trip every shipped scenario through the specs
(`tests/experiments/cyclone_jax/test_config.py`) — config drift fails CI.
