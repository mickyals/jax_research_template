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

All functions accept scalar or broadcast-compatible array inputs and expect coordinates in degrees. The `_np` variants run in float64 (NumPy default); the `_jax` variants run in float32 (JAX default). For geophysical distances the float32 precision (±~10 cm at short range) is negligible.

Vincenty is more accurate than Haversine near the poles and for antipodal points; Haversine is simpler and sufficient for regional domains.

### `met_conversions.py`

Meteorological unit and variable conversions (temperature scales, pressure conversions, dewpoint/RH relationships, wind components, etc.).

---

## `jax_core/`

### `helpers.py`

JAX utility functions for model initialisation and array manipulation.

| Function | Description |
|----------|-------------|
| `create_rng(seed)` | `jax.random.PRNGKey` from an integer seed |
| `create_rng_dict(seed, keys)` | Dict of PRNGKeys split from one seed — for `model.init(rngs, ...)` |
| `standardise(x, axis)` | Zero-mean unit-variance normalisation |
| `minmax_norm(x, lo, hi)` | Min-max normalisation to `[0, 1]` |

### `diagnostics.py`

JAX runtime diagnostics: device listing, memory checks, and debugging helpers.

---

## `sampling/`

### `coordinate.py`

Functions for generating spatial coordinate samples within a domain — used for constructing background (non-TC) query positions in the TC experiment, and more generally for OSSE (observing system simulation) setups.

---

## `plotting/`

All plotting functions return a `matplotlib.Figure` (or a tuple `(fig, ...)` for comparison plots) and never call `plt.show()`. This makes them safe to use in headless training loops — pass the returned figure directly to `logger.log_figure(key, fig, step)`.

### `plot1d.py`

| Function | Description |
|----------|-------------|
| `plot_losses(losses, title, window, xlabel, ylabel)` | Smoothed loss curve with optional rolling window |

### `plot2d.py`

| Function | Description |
|----------|-------------|
| `plot_field_2d(field, ...)` | 2D scalar field (imshow) |
| `plot_field_comparison_2d(pred, target, ...)` | Side-by-side prediction vs target + residual; returns `(fig, axes)` |
| `plot_scatter_overlay(field, coords, values, ...)` | Field background with scattered observation overlay |
| `plot_heatmap(matrix, xlabels, ylabels, ...)` | Annotated heatmap (confusion matrices, correlation matrices) |
| `plot_mollweide(field, lats, lons, ...)` | Global Mollweide projection |
| `plot_mollweide_comparison(pred, target, ...)` | Side-by-side Mollweide; returns `(fig, axes)` |

### `plot3d.py`

| Function | Description |
|----------|-------------|
| `plot_volume_slice(volume, ...)` | 2D slice through a 3D volume; returns `(fig, slice_ax)` |
| `plot_volume_comparison(pred, target, ...)` | Volume slice comparison + residual; returns `(fig, resid, mse)` |
| `plot_surface_3d(Z, X, Y, ...)` | 3D surface plot |
