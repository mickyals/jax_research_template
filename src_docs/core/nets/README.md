# `core/nets/`

Complete network implementations built from `core` primitives (activations, embeddings, initializations, norms, pooling, attention).

| Module | Contents |
|--------|----------|
| [`mlp.md`](mlp.md) | MLP registry — `MLP`, `SIREN`, `FINER`, `GAUSSIAN`, `GAUSSIAN_FINER`, `WIRE`, `WIRE_FINER`, `WIRE_COMPLEX`, `HOSC`, `HOSC_FINER`, `SINC` + `_BaseMLP` + embedding wrappers |
| [`conv.md`](conv.md) | Conv net registry — building blocks (`ConvBlock`, `ResidualBlock`, `DownsampleBlock`, `UpsampleBlock`, `NonLocalBlock`, `InceptionBlock`, `DenseLayer/Block/TransitionLayer`, `PatchEmbed`) + 1D variants + registered nets (`CONV_ENCODER`, `CONV_DECODER`, `RESNET`, `DENSENET`) |
| [`transformers.md`](transformers.md) | Transformer registry — blocks (`TransformerBlock`, `CrossAttentionBlock`, `SwinBlock/Pair`, `PatchMerging`) + registered nets (`TRANSFORMER_ENCODER`, `TRANSFORMER_DECODER`, `VIT`, `MASKED_VIT`, `MAE_DECODER`, `CONV_MAE_DECODER`, `SWIN_ENCODER`) |

---

## Common patterns

### Registry access

Each module exposes its own `get_*` / `list_*` / `register_*` trio:

```python
from core.nets.mlp import get_mlp, list_mlps
from core.nets.conv import get_conv_net, list_conv_nets
from core.nets.transformers import get_transformer, list_transformers
```

### Flax Linen usage

All registered nets are `flax.linen.Module` subclasses. Instantiating does not run computation:

```python
net = get_mlp("SIREN", out_features=1, hidden_features=256, n_layers=5)
variables = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3)))
out = net.apply(variables, jnp.ones((8, 3)), train=False)
```

### `train` flag

All nets accept `train: bool = True`. At eval time pass `train=False` — dropout becomes a no-op and `BatchNorm` switches to running statistics. No PRNG keys are needed at eval time.

### Dropout

When `dropout_rate > 0`, pass `rngs={'dropout': key}` at call time:

```python
out = net.apply(variables, x, train=True, rngs={'dropout': jax.random.PRNGKey(1)})
```

### BatchNorm

When using `norm='BATCH_NORM'` in conv nets, pass `mutable=['batch_stats']` and carry the updated stats:

```python
out, updates = net.apply(
    {'params': params, 'batch_stats': batch_stats},
    x, train=True,
    mutable=['batch_stats'],
)
batch_stats = updates['batch_stats']
```
