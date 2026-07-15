# models/ — usage

The model side of the experiment. One entry point:

```python
from experiments.cyclone_jax.models import build_model
module, tags = build_model(cfg['model'], data.targets)
# n_classes comes from the TargetSpec — keep `n_classes: null` in yaml
# (a hand-set value raises). tags -> the wandb run.
```

```
models/
├── __init__.py   # MODELS registry + build_model(cfg, targets) -> (module, tags)
├── features.py   # named X -> (B, N, F) tokens; positional-encoding modes
├── mlp.py        # StationMLP: shared perceptron -> flat concat -> core MLP
└── siren.py      # StationSIREN / StationFINER on core SIRENet/FINERNet
```

## Architecture (locked 2026-07-04)

NO Deep Sets / pooling / transformer logic — this is the CLEAN experiment.
Every model is flat-input: each station token goes through one SHARED
per-station perceptron, padding slots are zeroed via `station_mask`, the
result is concatenated FLAT `(B, pad_to * station_features)` and fed to a
core net body. The flat vector is slot-sensitive to order/padding — fine
for the memorisation gate; remember it when reading generalisation runs.

## Positional encoding is model-side

`features.py` owns it, keyed per model yaml (`encoding:` block):

- `mode: concat` — `[tokens; gamma(lat, lon)]`, or the raw coords when
  `embedding: null`.
- `mode: additive` — gamma Dense-projected to the token width and added.

`embedding` is any `core/embeddings` registry name; `(lat, lon)`-signature
embeddings are wrapped automatically. SIREN/FINER take raw coords and no
gamma — sine is their encoding.

## Rules

- Every model = one `configs/models/<name>.yaml` (self-contained, with
  `tags:`) + one registered factory here. Extend `config.MODEL_KEYS` when
  adding config surface (config loading stays jax-free).
- Nets/activations/inits come from `jrt/core` — never re-implement; the
  activation ladder for the MLP is `relu | gelu | silu | leaky_relu`.
