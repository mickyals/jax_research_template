# `pooling.py`

Registry-based pooling operations for JAX/Flax models. Covers global reductions (axis-parameterised), 2D spatial windowed pooling, and global spatial pooling.

Global reductions take an `axis` argument at call time, making them usable for both spatial pooling over `(H, W)` and set aggregation over a sequence dimension:

```python
pool = get_pooling("MEAN")
pool(x, axis=1)        # set aggregation: (B, N, D) -> (B, D)
pool(x, axis=(1, 2))   # spatial:         (B, H, W, C) -> (B, C)
```

Spatial pooling modules (`SPATIAL_MAX`, `SPATIAL_AVG`) are `nn.Module` subclasses with a fixed kernel and stride, used for downsampling between conv blocks.

---

## Registry Functions

---

### `register_pooling`

```
register_pooling(name: str, description: str = "") -> callable
```

Class decorator. Names are stored uppercase and must be unique.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name used for lookup. Stored uppercase | required |
| `description` | `str` | Short description shown in `list_pooling()` | `""` |

**Raises:** `ValueError` if a pooling with the same name is already registered.

---

### `get_pooling`

```
get_pooling(name: str, **kwargs) -> callable
```

Retrieve and instantiate a registered pooling operation by name. Inspects constructor signatures and warns about / drops unknown kwargs.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `name` | `str` | Name of the registered pooling (case-insensitive) | required |
| `**kwargs` | any | Forwarded to constructor | |

**Returns:** Instantiated pooling callable.

**Raises:** `ValueError` if no pooling with the given name exists.

---

### `list_pooling`

```
list_pooling() -> dict[str, str]
```

Return a sorted dictionary of all registered pooling names and their descriptions.

---

## Global Reductions

All reductions accept `axis: int | Sequence[int] = 1` at call time.

---

### `MEAN`

```python
MeanPooling(keepdims: bool = False)
```

Computes the mean over the specified axis.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `keepdims` | `bool` | Keep the reduced dimensions | `False` |

---

### `MAX`

```python
MaxPooling(keepdims: bool = False)
```

Computes the max over the specified axis.

---

### `MIN`

```python
MinPooling(keepdims: bool = False)
```

Computes the min over the specified axis.

---

### `SUM`

```python
SumPooling(keepdims: bool = False)
```

Computes the sum over the specified axis.

---

### `STD`

```python
StdPooling(keepdims: bool = False)
```

Computes the standard deviation over the specified axis. Useful as a second-order statistic alongside mean pooling for richer set representations.

---

### `MEAN_MAX`

```python
MeanMaxPooling(keepdims: bool = False)
```

Concatenates mean and max pooling along the feature dimension. Produces a richer representation than either alone. Output feature dimension is **2×** the input feature dimension.

```python
pool = get_pooling("MEAN_MAX")
out = pool(x, axis=1)   # (B, N, D) -> (B, 2D)
```

---

## Spatial Pooling (2D conv nets)

Fixed 2D window operations over H and W. These are `nn.Module` subclasses since they carry kernel/stride state. Input is channels-last: `(B, H, W, C)`.

---

### `SPATIAL_MAX`

```python
SpatialMaxPool(
    kernel_size: tuple = (2, 2),
    strides: tuple = (2, 2),
    padding: str = "VALID",
)
```

2D max pooling over a spatial window. Used in conv nets between blocks for downsampling.

```python
pool = get_pooling("SPATIAL_MAX", kernel_size=(2, 2), strides=(2, 2))
out = pool(x)   # (B, H, W, C) -> (B, H//2, W//2, C)
```

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `kernel_size` | `tuple[int, int]` | Size of the pooling window | `(2, 2)` |
| `strides` | `tuple[int, int]` | Stride of the pooling window | `(2, 2)` |
| `padding` | `str` | `'VALID'` or `'SAME'` | `'VALID'` |

---

### `SPATIAL_AVG`

```python
SpatialAvgPool(
    kernel_size: tuple = (2, 2),
    strides: tuple = (2, 2),
    padding: str = "VALID",
)
```

2D average pooling over a spatial window. Same parameters as `SPATIAL_MAX`.

---

## Global Spatial Pooling

---

### `GLOBAL_AVG`

```python
GlobalAvgPool(spatial_axes: tuple = (1, 2))
```

Global average pooling over specified spatial dimensions. Reduces the full spatial extent to a single vector per sample.

```python
pool = get_pooling("GLOBAL_AVG")
out = pool(x)   # (B, H, W, C) -> (B, C)

pool = get_pooling("GLOBAL_AVG", spatial_axes=(1,))
out = pool(x)   # (B, T, C) -> (B, C)
```

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `spatial_axes` | `tuple[int, ...]` | Axes to reduce over | `(1, 2)` for channels-last 2D |

---

### `GLOBAL_MAX`

```python
GlobalMaxPool(spatial_axes: tuple = (1, 2))
```

Global max pooling over specified spatial dimensions. Same parameters as `GLOBAL_AVG`.
