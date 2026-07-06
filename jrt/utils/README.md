# jrt/utils

General-purpose utilities. None of these are tied to a specific experiment — they are stable helpers that any experiment can import.

---

## `geoscience/`

### `geodesic.py`

Distance and bearing calculations on the Earth's surface.

| Function | Description |
|----------|-------------|
| `haversine_np(lat1, lon1, lat2, lon2)` | Haversine distance in km (NumPy, float64) |
| `haversine_jax(lat1, lon1, lat2, lon2)` | Haversine distance in km (JAX, JIT-compatible) |
| `vincenty_np(lat1, lon1, lat2, lon2)` | Vincenty inverse distance in km (NumPy, float64, higher accuracy) |
| `vincenty_jax(lat1, lon1, lat2, lon2)` | Vincenty distance (JAX, float32) |
| `latlon_box_area(lat_bounds, lon_bounds)` | Exact area (km²) of a lat/lon box on the sphere (network-sparsity / density calculations) |

All functions accept scalar or broadcast-compatible array inputs and expect coordinates in degrees. The `_np` variants run in float64 (NumPy default); the `_jax` variants run in float32 (JAX default). For geophysical distances the float32 precision (±~10 cm at short range) is negligible.

Vincenty is more accurate than Haversine near the poles and for antipodal points; Haversine is simpler and sufficient for regional domains.

### `met_conversions.py`

Meteorological unit and variable conversions (temperature scales, pressure conversions, dewpoint/RH relationships, wind components, etc.).

### `coordinates.py`

Pure lat/lon coordinate encoders (NumPy, vectorised) for feature matrices — kept here so the data-loading framework stays domain-agnostic.

| Function | Description |
|----------|-------------|
| `lat_lon_to_unit_sphere(lat, lon)` | (lat, lon)° → 3-D unit-sphere Cartesian `(x, y, z)`; no pole/meridian discontinuities |
| `lat_lon_to_domain_normalised(lat, lon, fov_lat, fov_lon)` | (lat, lon)° → `[-1, 1]²` over a field-of-view box `(min, max)` |

---

## `jax_core/`

### `helpers.py`

JAX utility functions for model initialisation and array manipulation.

| Function | Description |
|----------|-------------|
| `create_rng(seed)` | `jax.random.PRNGKey` from an integer seed |
| `create_rng_dict(seed, keys)` | Dict of PRNGKeys split from one seed — for `model.init(rngs, ...)` |
| `eval_forward(apply_fn, params, X, batch_stats=None)` | Jitted eval-mode forward (`apply_fn` static; one trace per batch shape) for host-side consumers — figure callbacks, evaluate passes. The Trainer keeps its own fused eval_step by design |
| `standardise(x, axis)` | Zero-mean unit-variance normalisation (device-side; numpy twin in `utils/normalise.py`) |
| `minmax_norm(x, lo, hi)` | Min-max normalisation to `[0, 1]` or `[-1, 1]` |

### `diagnostics.py`

JAX runtime diagnostics: device listing, memory checks, and debugging helpers. Includes `trace_profile(trace_dir, enabled=True)` — a context manager around `jax.profiler.start_trace`/`stop_trace` for ad-hoc profiling; view traces with TensorBoard's Profile plugin (WandB cannot render XLA traces). The Trainer exposes the same capability declaratively via the `profile`/`profile_steps` config keys (see `jrt/training/README.md`).

---

## `normalise.py`

Numpy-only normaliser registry + streaming statistics — jax-free on
purpose (multiprocess data workers must never import jax). The
device-side twins (`standardise`/`minmax_norm`) live in
`jax_core/helpers.py`; cross-referenced both ways.

| Name / class | Description |
|--------------|-------------|
| `NORMALISERS` registry: `minmax_01`, `minmax_11`, `standardise` | `(vals, lo, hi) -> scaled` — the (lo, hi) pair is (min, max) or (mean, std) per method |
| `get_normaliser(name)` | Registry lookup |
| `StatsAccumulator` | NaN-aware streaming per-column stats (mean/std/min/max/count) over ragged sample passes — feeds train-split normalisation stats |

---

## `sampling/`

### `coordinate.py`

Functions for generating spatial coordinate samples within a domain — used for constructing background (non-TC) query positions in the TC experiment, and more generally for OSSE (observing system simulation) setups.

---

## `plotting/`

All plotting functions return a `matplotlib.Figure` (or a tuple `(fig, ...)` for comparison plots) and never call `plt.show()`. This makes them safe to use in headless training loops — pass the returned figure directly to `logger.log_figure(key, fig, step)`.

### `aggregation.py`

Array preparation for plotting — masked reductions, point binning, smoothing, and volume slicing. No matplotlib imports; outputs are plain arrays (or array + extent) ready for the renderers below. A function lives here only if it encodes a decision (mask semantics, empty-bin convention, window centering) — trivial reductions stay inline at call sites.

| Function | Description |
|----------|-------------|
| `masked_mean_np(x, mask, axis=None)` / `masked_mean_jax(...)` | Mean over `mask`-True entries; 0.0 (not NaN) when nothing is valid |
| `bin_to_grid_np(x, y, values, extent, shape, reduce="mean")` / `bin_to_grid_jax(...)` | Scattered `(x, y, value)` points → 2D grid; `reduce` ∈ {"mean", "count", "max"}; empty cells are NaN (mean/max) or 0 (count); returns `(grid, extent)`. `_np` drops out-of-range points, `_jax` clips them to the nearest edge bin (static shape) |
| `rolling_mean_np(values, window)` | Centered moving average, output aligned to input indices (window shrinks near edges) |
| `take_slice_np(volume, axis, index)` | Bounds-checked 2D slice from an N-D volume |

### `_style.py`

Private rendering helpers shared by `curves.py`, `fields.py`, and `volumes.py` — not part of the public API. `DEFAULT_CMAP`, `_symmetric_clim`/`_resolve_clim` (colormap limit resolution), `_imshow_with_colorbar` (image+colorbar core), `_comparison_stats` (residual + shared clims + MSE for target/prediction/residual panels), `_contrast_color` (white/black text colour for annotation contrast against a heatmap cell), `_value_scatter` (scatter points optionally coloured/sized by a value array).

### `geo.py`

Public cartopy canvas helpers behind the `geo=` argument, also importable directly by experiment figure modules (promoted from the private `_geo.py`, PR #5 DRY ruling). `cartopy_available()` is the optional-basemap switch (True/False instead of raising) for figures with a plain-axes fallback; `import_cartopy()` is the lazy import with the clear install error. `make_geoaxes(figsize, extent, scale, color, lw, gridlines, projection, center, fill)` returns `(fig, GeoAxes, transform)` with coastline/border/state linework (`add_map_features`; `fill=True` adds muted land/ocean face colours underneath, basemap style) and labeled dashed gridlines; geo-capable renderers thread the returned `transform` (always the lon/lat PlateCarree CRS) through artist calls whose data is in degrees, so cartopy never leaks to call sites. Two projections: `'platecarree'` (default; extent in degrees) and `'azimuthal'` (`AzimuthalEquidistant` centred on `center=(lat, lon)`; extent in metres from the centre). The azimuthal projection's native axes coordinates are metres along (east, north) from the centre — i.e. (distance·sin(bearing), distance·cos(bearing)) — so storm-centred local x-y data scaled to metres plots with the *default* transform and coastlines land correctly by construction.

cartopy is an **optional dependency**, imported lazily inside `geo.py` only — the template installs and runs without it; calling a geo path without cartopy raises an ImportError with the install command (`pip install cartopy`). Natural Earth shapefiles download on first *render* at the chosen `scale` ('50m' default; pass `geo={"scale": "10m"}` for publication-quality linework).

### `curves.py`

Charts with a value axis — loss curves and grouped bar charts.

| Function | Description |
|----------|-------------|
| `plot_losses(losses, title, window, xlabel, ylabel)` | Smoothed loss curve; smoothing via `aggregation.rolling_mean_np` |
| `plot_grouped_bars(values, categories, ylabel, title, ylim, colors, ...)` | Grouped bar chart — one group per category, one bar per series (e.g. precision/recall/F1 per class) |

### `fields.py`

2D images in index/abstract coordinates. Mollweide stays here (generic matplotlib projection, any sphere). Earth maps are not a separate module: `plot_scatter_overlay` takes `geo=True` (or an options dict, e.g. `geo={"scale": "10m"}`) to draw on a PlateCarree map with coastlines/borders/states via `geo.py` — coordinates must then be lon/lat degrees, and `grid`/`xlabel`/`ylabel` are replaced by labeled map gridlines. Requires cartopy (optional dependency).

| Function | Description |
|----------|-------------|
| `plot_field_2d(field, ...)` | 2D scalar field (imshow) |
| `plot_field_comparison_2d(pred, target, ...)` | Side-by-side prediction vs target + residual; returns `(fig, resid, mse)` |
| `plot_scatter_overlay(field, scatter_x, scatter_y, scatter_values, ...)` | Scattered points, optionally over a 2D field background. `field=None` gives a standalone scatter plot with its own colorbar (driven by `scatter_values`). Optional `marker_x`/`marker_y`/`marker_label` draws a single highlighted reference point (e.g. a query location); `scatter_size_range` scales marker size by value; `grid` adds gridlines when there's no field; `geo` draws on a PlateCarree map (see `geo.py`) |
| `plot_heatmap(matrix, row_labels, col_labels, ..., annotate=True)` | Annotated heatmap (cosine similarity, correlation matrices); annotation text colour auto-contrasts against each cell via `_contrast_color` |
| `confusion_matrix_figure(cm, class_names, title)` | Classification-standard `plot_heatmap` specialisation: integer count annotations, 'Blues', predicted/true axis labels — pass any accumulated confusion matrix (e.g. from `training.metrics.update_cm`) |
| `plot_mollweide(field, lats, lons, ...)` | Global Mollweide projection |
| `plot_mollweide_comparison(pred, target, ...)` | Side-by-side Mollweide; returns `(fig, resid, mse)` |

### `volumes.py`

| Function | Description |
|----------|-------------|
| `plot_volume_comparison(pred, target, slice_index, axis, ...)` | Slices both volumes via `aggregation.take_slice_np`, then renders target/prediction/residual; returns `(fig, resid, mse)` |
| `plot_surface_3d(Z, X, Y, ...)` | 3D surface plot |

For a single slice without a comparison, use `aggregation.take_slice_np` + `fields.plot_field_2d`.
