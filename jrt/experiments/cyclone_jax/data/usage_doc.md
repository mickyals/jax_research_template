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
| `variables.py` | variable catalogue: canonical units + description per stored column (`VARIABLES`), meta.json sidecar content (`column_meta`), display-unit helper (`to_display`) |
| `inputs.py` | `InputSpec` / `resolve_input`: per-source channel schemas (`SOURCE_SCHEMAS`), derived wind, canonical union (`CHANNEL_ORDER`), selection policy, `pad_to` |
| `targets.py` | `TargetSpec` / `resolve_target`: label space (class_set; label = position in set), `build_y`, class names |
| `transformations.py` | numpy mechanics (missingness). Wind lives in `utils/geoscience/met_conversions`, normalisers in `utils/jax_core/helpers` — not duplicated here |
| `sampler.py` | `Loader` (deterministic fix -> ragged named sample) + `Sampler` (seeded index streams) + `split_by_year` / `stratified_fixes` |
| `identifiability.py` | `input_collisions`: identical-input/different-target groups + memorisation accuracy ceiling (run before the memorise scenario) |
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

## Observation channels (`cfg['channels']`)

What each source can contribute to `x['obs']` (`SOURCE_SCHEMAS`; canonical
column order = `CHANNEL_ORDER`):

| channel | land | marine | upper (deferred) |
|---|---|---|---|
| `station_pressure` | ✓ | — | — |
| `slp` | ✓ | ✓ | — |
| `air_temp` | ✓ | ✓ | ✓ |
| `dewpoint` | ✓ | ✓ | ✓ |
| `sst` | — | ✓ | — |
| `u_wind` / `v_wind` | derived | derived | stored |

(`u/v` are derived at assembly from `wind_speed` + `wind_dir`; upper joins
once its pressure-coordinate encoding is designed.)

- The model input is the **union** of the chosen sources' channels, in
  canonical order; a channel one source lacks rides along as 0 + flag
  (see Sample schema above).
- `channels: [name, ...]` in the data yaml filters that union GLOBALLY
  (one list for all sources — per-source absence is the flag's job).
  Validated: subset of the sources' union, non-empty; canonical order is
  kept regardless of listing order.
- **Source-comparable recipe**: the land∩marine intersection
  `[slp, air_temp, dewpoint, u_wind, v_wind]` gives the identical
  5-channel vector across land-only / marine-only / land+marine
  scenarios (same token width everywhere).
- The startup banner prints `[data] channels (n): ...` and
  `data_manifest.json` records the resolved list — every run states what
  it saw. A channel subset is a NEW scenario: expect the identifiability
  ceiling (`input_collisions`) to move.

## Batches

```python
batch = next(iter(data.streams['train']))
batch['X']       # x fields padded to (B, pad_to, ...) + station_mask
batch['y']       # (B,) stacked y['target']
batch['meta']    # sid list, lat/lon/time arrays, n_stations
```

- `pad_to` is FIXED from config (never batch max) -> jit compiles once.
- `num_workers > 0` (data yaml) = multiprocess assembly for TRAIN streams
  via jrt `training/prefetch` (val/test always sync). Batch contents match
  the sync path exactly; epoch batch ORDER becomes queue-arrival order.
  Linux/fork boxes only — keep 0 on Windows (spawn would pickle mmaps).
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
| `memorise` | `exclude_multistorm` (default true) | train == val == ALL fixes (full-dataset identifiability probe) |
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

## Units

Storage is **canonical SI everywhere** (ruling 2026-07-05). The CDM
sources (LISO/MISO/CUON) deliver SI; IBTrACS is converted inside
`build_storm_volume` (`convert_storm_units`, strictly AFTER `remap_sshs`
— the remap's Saffir-Simpson thresholds are kt on raw `usa_wind`). Every
volume carries a `meta.json` sidecar (`{column: {units, description}}`)
written by `write_volume` from the catalogue. Figures/tables convert at
the display boundary only: `to_display(value, units)` applies the one
project-wide mapping (m → km, Pa → hPa, everything else as stored).
Standardisation absorbs affine unit changes, so model results are
unchanged by the conversion — this is metadata/eval/figure correctness.

| variable | units | notes |
|---|---|---|
| `report_timestamp` (`launch_timestamp` upper) | datetime64[ns] | volume sort key (launch = ascent second, provenance) |
| `lat` / `lon` | degrees | −90..90 / −180..180 symmetric |
| `level` | Pa | vertical coord everywhere: land station p, marine SLP, upper z, storm `usa_pres` (NaN allowed on driver) |
| `elevation`, `geopot` | m | land station height; upper geopotential height |
| `station_pressure`, `slp`, `usa_pres`, `usa_poci` | Pa | storm pair converted from mb |
| `air_temp`, `dewpoint`, `sst`, `dpd` | K | |
| `wind_speed`, `u_wind`, `v_wind`, `usa_wind`, `storm_speed` | m/s | `usa_wind` is a 1-MINUTE average (US agencies); storm pair converted from kt |
| `wind_dir`, `storm_dir` | degrees | 0..360 bearings: wind FROM, storm heading TOWARD |
| `usa_rmw`, `usa_roci`, `usa_r{34,50,64}_{NE,SE,SW,NW}` | m | converted from nmile |
| `rh` / `q` | % / kg kg⁻¹ | upper only |
| `usa_sshs` (`usa_sshs_raw`, `is_subtropical`) | category | project 0..8 scheme (raw −5..5 kept alongside) |
| `sid`, `name`, `basin`, `subbasin`, `iflag`, `usa_status`, `season`, `platform_type` | str / year / code | identity metadata |

## Rebuilding the library

```python
from experiments.cyclone_jax.data.sources.library import build_bookshelf
build_bookshelf('E:/Caribbean-Obs')   # after any volume rebuild
```

`load_library` raises on stale shelves (fingerprint mismatch) and on
lookback-delta drift from `library.LOOKBACK_DELTAS`.
