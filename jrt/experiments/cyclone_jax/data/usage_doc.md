# data/ — usage

Everything between raw sources and device batches. One entry point:

```python
from experiments.cyclone_jax.data.interface import build_data
data = build_data(cfg['data'], seed=0)     # cfg from config.load_config
```

`build_data` returns a `DataBundle`: `lib, inputs, targets, loader,
splits, streams` — all by name, notebook-friendly.

## Module map (dependency order)

| module | role |
|---|---|
| `sources/volume.py` | generic columnar store (volume_v1): one mmap .npy per column, entity spine, time-sorted rows |
| `sources/shelf.py` | `_BOOKSHELF` cross-volume indices: time spines, driver manifest (storm/multi times), causal lookback edges, freshness fingerprints |
| `sources/build.py` | BUILD-TIME raw -> volume converters (LISO/MISO/CUON/IBTrACS; xarray/pandas; never imported at train time) |
| `sources/library.py` | TRAIN-TIME library access: `load_library` (+staleness guards), `build_bookshelf`, `get_fixes`, cyclone vocabulary (SSHS constants, `CYC_TARGETS` allowlist, lookback schedules) |
| `inputs.py` | `InputSpec` / `resolve_input`: per-source channel schemas (`SOURCE_SCHEMAS`), derived wind, canonical union (`CHANNEL_ORDER`), selection policy, `pad_to` |
| `targets.py` | `TargetSpec` / `resolve_target`: label space (class_set; label = position in set), `build_y`, class names |
| `transformations.py` | numpy mechanics (missingness). Wind lives in `utils/geoscience/met_conversions`, normalisers in `utils/jax_core/helpers` — not duplicated here |
| `sampler.py` | `Loader` (deterministic fix -> ragged named sample) + `Sampler` (seeded index streams) + `split_by_year` / `stratified_fixes` |
| `batching.py` | `collate(samples, pad_to)`: the ONLY sample -> device-batch translator |
| `interface.py` | `build_data`, `BatchStream`, split-strategy resolution |

## Sample schema

```python
s = data.loader.build(i)
s['x']   # {lat (n,), lon (n,), level (n,), time (n,),
         #  obs (n, C), missing (n, C) bool, id (n,)}
s['y']   # {'target', 'sid', 'lat', 'lon', 'time'}
```

- `x['time']` = seconds relative to the fix, `<= 0` (causality from the
  bookshelf lookback edges). Absolute fix time = `y['time']`.
- `x['level']` = each volume's vertical coordinate (land: station
  pressure, marine: SLP, upper: z); finite by the build-time gate.
- Channels a source lacks (e.g. sst on land) are `0` with
  `missing=False` — every variable owns one fixed union position;
  `data.inputs.channel_index` maps name -> column.
- `y['target']` is what the loss consumes; the identity fields are
  eval/plot metadata, never model input.

## Batches

```python
batch = next(iter(data.streams['train']))
batch['X']       # x fields padded to (B, pad_to, ...) + station_mask
batch['y']       # (B,) stacked y['target']
batch['meta']    # sid list, lat/lon/time arrays, n_stations
```

- `pad_to` is FIXED from config (never batch max) -> jit compiles once.
- `meta['n_stations'] == pad_to` means truncation (size pad_to to the
  library max so it never happens).
- The jrt Trainer drops `meta` before tracing — strings never reach jit.
- `for batch in stream` reshuffles per pass (epoch auto-increments);
  `stream.epoch(e)` replays epoch e exactly. Train streams shuffle +
  drop_last; val/test are sequential and keep the partial batch.

## Splits (`cfg['split']`)

| strategy | keys | result |
|---|---|---|
| `year` | `years: {train, val, test}`, `exclude_multistorm` (default true) | disjoint year splits; multi-driver fixes excluded |
| `stratified` | `n_per_class` | balanced overfit subset; train == val (memorisation gate) |
| `multistorm` | — | `{'test': ...}` = all fixes at multi-driver timestamps (OOD) |
| *(absent)* | — | `{'all': ...}` for exploration |

## Normalisation (`cfg['normalise']` + `cfg['domain']`)

`normalise: {method: standardise|minmax_01|minmax_11, stats: auto|inline}`;
no block or `method: none` = raw values. Mechanics live in
`utils/normalise` (shared registry); policy in `normalise.py` (NormSpec —
third sibling of Input/TargetSpec).

- **Stats are properties of a training distribution**: `stats: auto`
  computes them over the TRAIN split (or `all`) inside `build_data`, and
  train.py saves them to `run_dir/norm_stats.json` (+ wandb config +
  manifest). Evaluation must REUSE a training run's saved stats.
- **Scenarios with no train split** (multistorm) cannot self-compute —
  `build_data` raises with instructions. Provide inline stats in the yaml
  or point evaluation at the training run (`--stats <run_dir>`), i.e. say
  WHICH training distribution the stress test is relative to.
- Coverage: obs per-channel by `method` **before** the NaN->0 fill (so
  zero-fill == mean-fill); level by `method`; time / lookback -> [-1, 0];
  lat/lon -> [-1, 1] over `domain:` bounds (fallback: train min/max,
  logged). `y` and `meta` stay RAW — they are eval metadata.
- `domain: {lat: [lo, hi], lon: [lo, hi]}` pins the FOV for coord scaling;
  station selection (haversine) always runs on real degrees regardless.

## Rebuilding the library

```python
from experiments.cyclone_jax.data.sources.library import build_bookshelf
build_bookshelf('E:/Caribbean-Obs')   # after any volume rebuild
```

`load_library` raises on stale shelves (fingerprint mismatch) and on
lookback-delta drift from `library.LOOKBACK_DELTAS`.
