# Plotting Utilities

This package provides visualization utilities for scientific plotting, organized into three modules:

- **`plot1d.py`**: 1D plotting functions for time series and loss curves
- **`plot2d.py`**: 2D plotting functions for fields, heatmaps, and global projections
- **`plot3d.py`**: 3D plotting functions for volumes and surfaces

---

## Module: `plot1d.py`

---

### `plot_losses`

```
plot_losses(
    losses: dict[str, list[float]],
    title: str = "Training loss",
    window: int = 20,
) -> None
```

Plot train and optional test loss curves with a smoothed overlay. Produces two side-by-side panels -- left shows raw log-scale curves, right shows moving-average smoothed log-scale curves. Steps before `window` are omitted from the smoothed curve.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `losses` | `dict[str, list[float]]` | Dictionary with keys `"train"` and optionally `"test"`, each a list of scalar loss values one per step | required |
| `title` | `str` | Base title used for both panel headings | `"Training loss"` |
| `window` | `int` | Moving average window size for the smoothed panel | `20` |

---

## Module: `plot2d.py`

---

### Private Helpers

---

#### `_symmetric_clim`

```
_symmetric_clim(data: np.ndarray) -> tuple[float, float]
```

Return `(-vmax, vmax)` where `vmax = max(|data|)`. Used internally for symmetric colormap scaling.

---

#### `_resolve_clim`

```
_resolve_clim(
    data: np.ndarray,
    symmetric: bool,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]
```

Resolve colormap limits from data, symmetric flag, and explicit overrides. Explicit `vmin`/`vmax` always take precedence. Otherwise uses symmetric or data-range scaling. Used internally by all plotting functions that accept a `symmetric_cmap` parameter.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `data` | `np.ndarray` | Input data array | required |
| `symmetric` | `bool` | If True, scale symmetrically around zero | required |
| `vmin` | `float or None` | Explicit minimum override | required |
| `vmax` | `float or None` | Explicit maximum override | required |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[float, float]` | `(vmin, vmax)` resolved colormap limits |

---

### `plot_field_2d`

```
plot_field_2d(
    field: np.ndarray,
    extent: list[float] | None = None,
    cmap: str = "RdBu_r",
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    colorbar_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric_cmap: bool = True,
    figsize: tuple[int, int] = (10, 5),
) -> None
```

Plot a 2D scalar field as an image. By default scales the colormap symmetrically around zero using `max(|field|)`. Set `symmetric_cmap=False` for all-positive or all-negative fields where symmetric scaling would waste half the colorbar range.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `field` | `np.ndarray` | 2D array of shape `(rows, cols)` | required |
| `extent` | `list[float] or None` | `[xmin, xmax, ymin, ymax]` for axis labels. If None, axes show pixel indices | `None` |
| `cmap` | `str` | Matplotlib colormap | `"RdBu_r"` |
| `title` | `str` | Plot title | `""` |
| `xlabel` | `str` | X-axis label | `"x"` |
| `ylabel` | `str` | Y-axis label | `"y"` |
| `colorbar_label` | `str` | Label for the colorbar | `""` |
| `vmin` | `float or None` | Colormap minimum override | `None` |
| `vmax` | `float or None` | Colormap maximum override | `None` |
| `symmetric_cmap` | `bool` | Scale colormap symmetrically around zero | `True` |
| `figsize` | `tuple[int, int]` | Figure size in inches | `(10, 5)` |

---

### `plot_field_comparison_2d`

```
plot_field_comparison_2d(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    extent: list[float] | None = None,
    cmap: str = "RdBu_r",
    title_prefix: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: tuple[int, int] = (16, 4),
    verbose: bool = True,
) -> tuple[np.ndarray, float]
```

Plot target, prediction, and residual side by side as three panels. Target and prediction panels share a symmetric colormap scaled to the maximum absolute value across both. The residual panel uses its own symmetric scale.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `true_field` | `np.ndarray` | Ground truth 2D array of shape `(rows, cols)` | required |
| `pred_field` | `np.ndarray` | Model prediction 2D array of shape `(rows, cols)` | required |
| `extent` | `list[float] or None` | `[xmin, xmax, ymin, ymax]` for axis labels | `None` |
| `cmap` | `str` | Matplotlib colormap | `"RdBu_r"` |
| `title_prefix` | `str` | String prepended to each panel title | `""` |
| `xlabel` | `str` | X-axis label | `"x"` |
| `ylabel` | `str` | Y-axis label | `"y"` |
| `figsize` | `tuple[int, int]` | Figure size in inches | `(16, 4)` |
| `verbose` | `bool` | If True, print the grid MSE after plotting | `True` |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[np.ndarray, float]` | `(residual array of shape (rows, cols), grid MSE scalar)` |

---

### `plot_scatter_overlay`

```
plot_scatter_overlay(
    field: np.ndarray,
    scatter_x: np.ndarray,
    scatter_y: np.ndarray,
    scatter_values: np.ndarray | None = None,
    extent: list[float] | None = None,
    cmap: str = "RdBu_r",
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    scatter_size: int = 30,
    scatter_vmin: float | None = None,
    scatter_vmax: float | None = None,
    symmetric_cmap: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> None
```

Plot a 2D field with a scatter overlay. Scatter points can use independent colour scaling from the field via `scatter_vmin` / `scatter_vmax`. If those are not provided and `scatter_values` is given, the scatter inherits the field's colormap limits. If `scatter_values` is None, points are plotted in black.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `field` | `np.ndarray` | 2D array of shape `(rows, cols)` | required |
| `scatter_x` | `np.ndarray` | X coordinates of scatter points, shape `(n,)` | required |
| `scatter_y` | `np.ndarray` | Y coordinates of scatter points, shape `(n,)` | required |
| `scatter_values` | `np.ndarray or None` | Values used to colour scatter points. If None, points are black | `None` |
| `extent` | `list[float] or None` | `[xmin, xmax, ymin, ymax]` for axis labels | `None` |
| `cmap` | `str` | Matplotlib colormap applied to both field and scatter | `"RdBu_r"` |
| `title` | `str` | Plot title | `""` |
| `xlabel` | `str` | X-axis label | `"x"` |
| `ylabel` | `str` | Y-axis label | `"y"` |
| `scatter_size` | `int` | Scatter marker size | `30` |
| `scatter_vmin` | `float or None` | Colormap minimum for scatter. Defaults to field vmin | `None` |
| `scatter_vmax` | `float or None` | Colormap maximum for scatter. Defaults to field vmax | `None` |
| `symmetric_cmap` | `bool` | Scale field colormap symmetrically around zero | `True` |
| `vmin` | `float or None` | Field colormap minimum override | `None` |
| `vmax` | `float or None` | Field colormap maximum override | `None` |
| `figsize` | `tuple[int, int]` | Figure size in inches | `(10, 5)` |

---

### `plot_heatmap`

```
plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "",
    colorbar_label: str = "",
    annotate: bool = False,
    fmt: str = ".2f",
    annotate_fontsize: int = 8,
    figsize: tuple[int, int] | None = None,
    max_figsize: tuple[int, int] = (14, 14),
) -> None
```

Plot a 2D matrix as a heatmap with optional cell annotations. General purpose -- suitable for cosine similarity matrices, confusion matrices, correlation matrices, or any 2D array where explicit tick labels add meaning. Figure size is auto-computed from matrix shape and capped by `max_figsize`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `matrix` | `np.ndarray` | 2D array of shape `(rows, cols)` | required |
| `row_labels` | `list[str] or None` | Tick labels for the y-axis | `None` |
| `col_labels` | `list[str] or None` | Tick labels for the x-axis | `None` |
| `cmap` | `str` | Matplotlib colormap | `"RdBu_r"` |
| `vmin` | `float or None` | Colormap minimum | `None` |
| `vmax` | `float or None` | Colormap maximum | `None` |
| `title` | `str` | Plot title | `""` |
| `colorbar_label` | `str` | Label for the colorbar | `""` |
| `annotate` | `bool` | If True, write the numeric value of each cell | `False` |
| `fmt` | `str` | Format string for cell annotations | `".2f"` |
| `annotate_fontsize` | `int` | Font size for cell annotations. Reduce for large matrices | `8` |
| `figsize` | `tuple[int, int] or None` | Figure size. Auto-computed from matrix shape if not provided | `None` |
| `max_figsize` | `tuple[int, int]` | Upper bound on auto-computed figure size | `(14, 14)` |

---

### `plot_mollweide`

```
plot_mollweide(
    field: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    cmap: str = "RdBu_r",
    title: str = "",
    colorbar_label: str = "",
    symmetric_cmap: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    scatter_lon: np.ndarray | None = None,
    scatter_lat: np.ndarray | None = None,
    scatter_values: np.ndarray | None = None,
    scatter_cmap: str | None = None,
    scatter_vmin: float | None = None,
    scatter_vmax: float | None = None,
    scatter_size: int = 8,
    figsize: tuple[int, int] = (12, 5),
) -> None
```

Plot a scalar field on a Mollweide projection. Scatter points can be coloured independently from the field via `scatter_vmin` / `scatter_vmax` and `scatter_cmap`. If those are not provided, scatter inherits the field's colormap and colour limits. If `scatter_values` is None, scatter points are plotted in black.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `field` | `np.ndarray` | 2D array of shape `(rows, cols)` matching `lon_grid` / `lat_grid` | required |
| `lon_grid` | `np.ndarray` | 2D array of longitudes in radians, shape `(rows, cols)` | required |
| `lat_grid` | `np.ndarray` | 2D array of latitudes in radians, shape `(rows, cols)` | required |
| `cmap` | `str` | Matplotlib colormap for the field | `"RdBu_r"` |
| `title` | `str` | Plot title | `""` |
| `colorbar_label` | `str` | Colorbar label | `""` |
| `symmetric_cmap` | `bool` | Scale field colormap symmetrically around zero | `True` |
| `vmin` | `float or None` | Field colormap minimum override | `None` |
| `vmax` | `float or None` | Field colormap maximum override | `None` |
| `scatter_lon` | `np.ndarray or None` | Longitudes of scatter points in radians | `None` |
| `scatter_lat` | `np.ndarray or None` | Latitudes of scatter points in radians | `None` |
| `scatter_values` | `np.ndarray or None` | Values used to colour scatter points. If None, points are black | `None` |
| `scatter_cmap` | `str or None` | Colormap for scatter points. Defaults to field cmap | `None` |
| `scatter_vmin` | `float or None` | Colormap minimum for scatter. Defaults to field vmin | `None` |
| `scatter_vmax` | `float or None` | Colormap maximum for scatter. Defaults to field vmax | `None` |
| `scatter_size` | `int` | Scatter marker size | `8` |
| `figsize` | `tuple[int, int]` | Figure size in inches | `(12, 5)` |

---

### `plot_mollweide_comparison`

```
plot_mollweide_comparison(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    cmap: str = "RdBu_r",
    title_prefix: str = "",
    scatter_lon: np.ndarray | None = None,
    scatter_lat: np.ndarray | None = None,
    scatter_values: np.ndarray | None = None,
    scatter_cmap: str | None = None,
    scatter_vmin: float | None = None,
    scatter_vmax: float | None = None,
    figsize: tuple[int, int] = (18, 4),
    verbose: bool = True,
) -> tuple[np.ndarray, float]
```

Plot target, prediction, and residual on three Mollweide panels. Scatter points default to the field's symmetric colour limits -- pass `scatter_vmin` / `scatter_vmax` to decouple when scatter values have a different range to the field (e.g. probabilities in `[0, 1]`).

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `true_field` | `np.ndarray` | Ground truth 2D array of shape `(rows, cols)` | required |
| `pred_field` | `np.ndarray` | Model prediction 2D array of shape `(rows, cols)` | required |
| `lon_grid` | `np.ndarray` | 2D array of longitudes in radians, shape `(rows, cols)` | required |
| `lat_grid` | `np.ndarray` | 2D array of latitudes in radians, shape `(rows, cols)` | required |
| `cmap` | `str` | Matplotlib colormap | `"RdBu_r"` |
| `title_prefix` | `str` | String prepended to each panel title | `""` |
| `scatter_lon` | `np.ndarray or None` | Longitudes of scatter points in radians | `None` |
| `scatter_lat` | `np.ndarray or None` | Latitudes of scatter points in radians | `None` |
| `scatter_values` | `np.ndarray or None` | Values used to colour scatter points. Defaults to field colour limits | `None` |
| `scatter_cmap` | `str or None` | Colormap for scatter points. Defaults to field cmap | `None` |
| `scatter_vmin` | `float or None` | Colormap minimum for scatter. Defaults to field vmin | `None` |
| `scatter_vmax` | `float or None` | Colormap maximum for scatter. Defaults to field vmax | `None` |
| `figsize` | `tuple[int, int]` | Figure size in inches | `(18, 4)` |
| `verbose` | `bool` | If True, print the grid MSE after plotting | `True` |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[np.ndarray, float]` | `(residual array of shape (rows, cols), grid MSE scalar)` |

---

## Module: `plot3d.py`

---

### `plot_volume_slice`

```
plot_volume_slice(
    volume: np.ndarray,
    slice_index: int,
    axis: int = 2,
    extent: list[float] | None = None,
    cmap: str = "RdBu_r",
    title: str = "",
    colorbar_label: str = "",
    symmetric_cmap: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[int, int] = (8, 5),
) -> np.ndarray
```

Plot a single 2D slice of a 3D volume at a specified index along the given axis. Input should be a NumPy array or convertible via `np.asarray()` -- JAX arrays are accepted and converted automatically. Call in a loop to visualise multiple levels. If `title` is empty, defaults to `"z = {slice_index}"` or equivalent for the chosen axis.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `volume` | `np.ndarray` | 3D array of shape `(nx, ny, nz)` or any array-like | required |
| `slice_index` | `int` | Index along `axis` to extract | required |
| `axis` | `int` | Axis to slice along. `0=x`, `1=y`, `2=z` | `2` |
| `extent` | `list[float] or None` | `[xmin, xmax, ymin, ymax]` for the two axes not sliced | `None` |
| `cmap` | `str` | Matplotlib colormap | `"RdBu_r"` |
| `title` | `str` | Plot title. Auto-generated if empty | `""` |
| `colorbar_label` | `str` | Colorbar label | `""` |
| `symmetric_cmap` | `bool` | Scale colormap symmetrically around zero | `True` |
| `vmin` | `float or None` | Colormap minimum override | `None` |
| `vmax` | `float or None` | Colormap maximum override | `None` |
| `figsize` | `tuple[int, int]` | Figure size in inches | `(8, 5)` |

**Returns:**

| Type | Description |
|------|-------------|
| `np.ndarray` | Extracted 2D slice. Shape: `axis=0` → `(ny, nz)`, `axis=1` → `(nx, nz)`, `axis=2` → `(nx, ny)` |

---

### `plot_volume_comparison`

```
plot_volume_comparison(
    true_volume: np.ndarray,
    pred_volume: np.ndarray,
    slice_index: int,
    axis: int = 2,
    extent: list[float] | None = None,
    cmap: str = "RdBu_r",
    title_prefix: str = "",
    figsize: tuple[int, int] = (16, 4),
    verbose: bool = True,
) -> tuple[np.ndarray, float]
```

Plot target, prediction, and residual slices side by side as three panels. Extracts the same slice from both volumes and computes the residual. Target and prediction panels share a symmetric colormap. The residual panel uses its own symmetric scale. Residuals are always plotted symmetrically since they are naturally centred around zero. Input arrays should be NumPy arrays or convertible via `np.asarray()`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `true_volume` | `np.ndarray` | Ground truth 3D array of shape `(nx, ny, nz)` | required |
| `pred_volume` | `np.ndarray` | Model prediction 3D array, same shape as `true_volume` | required |
| `slice_index` | `int` | Index along `axis` to extract | required |
| `axis` | `int` | Axis to slice along. `0=x`, `1=y`, `2=z` | `2` |
| `extent` | `list[float] or None` | `[xmin, xmax, ymin, ymax]` for the two axes not sliced | `None` |
| `cmap` | `str` | Matplotlib colormap | `"RdBu_r"` |
| `title_prefix` | `str` | String prepended to each panel title | `""` |
| `figsize` | `tuple[int, int]` | Figure size in inches | `(16, 4)` |
| `verbose` | `bool` | If True, print the slice MSE after plotting | `True` |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[np.ndarray, float]` | `(residual 2D slice, MSE scalar for this slice)` |

---

### `plot_surface_3d`

```
plot_surface_3d(
    z: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    cmap: str = "viridis",
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    alpha: float = 1.0,
    stride: int = 1,
    symmetric_cmap: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[int, int] = (10, 7),
) -> None
```

Plot a 2D array as a 3D surface. Colour limits are computed from the full array before striding so extreme values at positions skipped by `stride` are not lost from the colour scale. Input should be a NumPy array or convertible via `np.asarray()`.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `z` | `np.ndarray` | 2D array of shape `(rows, cols)` representing surface heights | required |
| `x` | `np.ndarray or None` | 1D array of x coordinates, shape `(cols,)`. If None, uses column indices | `None` |
| `y` | `np.ndarray or None` | 1D array of y coordinates, shape `(rows,)`. If None, uses row indices | `None` |
| `cmap` | `str` | Matplotlib colormap | `"viridis"` |
| `title` | `str` | Plot title | `""` |
| `xlabel` | `str` | X-axis label | `"x"` |
| `ylabel` | `str` | Y-axis label | `"y"` |
| `zlabel` | `str` | Z-axis label | `"z"` |
| `alpha` | `float` | Surface transparency. `1.0` is fully opaque | `1.0` |
| `stride` | `int` | Subsampling stride for rendering. `stride=2` reduces vertex count by 4x | `1` |
| `symmetric_cmap` | `bool` | Scale colormap symmetrically around zero. Default False since surfaces like loss landscapes are typically all-positive | `False` |
| `vmin` | `float or None` | Colormap minimum override | `None` |
| `vmax` | `float or None` | Colormap maximum override | `None` |
| `figsize` | `tuple[int, int]` | Figure size in inches | `(10, 7)` |