# Coordinate Sampling Utilities

This package provides coordinate sampling functions for spatial and volumetric domains, including uniform random sampling and Latin Hypercube Sampling (LHS) via `scipy.stats.qmc` for better space-filling properties.

- **`coordinate.py`**: Uniform random and Latin Hypercube samplers for regional lon/lat boxes, spherical surfaces, and 3D volumes.

---

## Module: `coordinate.py`

---

### Uniform Random Samplers

---

#### `sample_regional`

```
sample_regional(
    key: jax.Array,
    n: int,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
) -> tuple[jax.Array, jax.Array]
```

Uniform random sampling in a lon/lat box.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `key` | `jax.Array` | JAX PRNGKey | required |
| `n` | `int` | Number of samples | required |
| `lon_bounds` | `tuple[float, float]` | `(lon_min, lon_max)` in degrees | required |
| `lat_bounds` | `tuple[float, float]` | `(lat_min, lat_max)` in degrees | required |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[jax.Array, jax.Array]` | `(lons, lats)` each shape `(n,)` |

---

#### `sample_sphere_uniform_area`

```
sample_sphere_uniform_area(
    key: jax.Array,
    n: int,
) -> tuple[jax.Array, jax.Array]
```

Uniform sampling on the sphere with respect to surface area. Uses the inverse CDF method `lat = arcsin(2u - 1)`, `lon = 2*pi*v - pi` to avoid pole-clustering that occurs with uniform-in-angle sampling. Use this for unbiased global coverage.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `key` | `jax.Array` | JAX PRNGKey | required |
| `n` | `int` | Number of samples | required |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[jax.Array, jax.Array]` | `(lat, lon)` in radians, each shape `(n,)`. lat in `[-pi/2, pi/2]`, lon in `[-pi, pi]` |

---

#### `sample_sphere_uniform_angle`

```
sample_sphere_uniform_angle(
    key: jax.Array,
    n: int,
) -> tuple[jax.Array, jax.Array]
```

Uniform-in-angle sampling on the sphere. Samples lat uniformly in `[-pi/2, pi/2]` and lon uniformly in `[-pi, pi]`. This is NOT area-uniform -- it oversamples the poles relative to surface area. Use `sample_sphere_uniform_area` for unbiased global coverage.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `key` | `jax.Array` | JAX PRNGKey | required |
| `n` | `int` | Number of samples | required |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[jax.Array, jax.Array]` | `(lat, lon)` in radians, each shape `(n,)` |

---

#### `sample_volume`

```
sample_volume(
    key: jax.Array,
    n: int,
    bounds: jax.Array,
) -> jax.Array
```

Uniform random sampling in a 3D box. All three dimensions are sampled in a single vectorised call.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `key` | `jax.Array` | JAX PRNGKey | required |
| `n` | `int` | Number of samples | required |
| `bounds` | `jax.Array` | Shape `(3, 2)`. Each row is `[min, max]` for one dimension, ordered as `[x, y, z]` or `[lon, lat, alt]` | required |

**Returns:**

| Type | Description |
|------|-------------|
| `jax.Array` | Sampled coordinates of shape `(n, 3)` |

---

### Latin Hypercube Samplers

LHS guarantees one sample per stratum in each dimension, giving substantially better space-filling coverage than uniform random at the same `n`. Particularly relevant for sparse observation experiments where uniform random can leave large empty regions.

All LHS functions bridge the JAX PRNGKey to `scipy.stats.qmc` via `_key_to_seed` for reproducibility. These functions are not purely functional -- they call scipy's stateful RNG internally -- but sequential calls with different keys produce independent, reproducible results.

Requires `scipy >= 1.8`.

---

#### `lhs_sample`

```
lhs_sample(
    key: jax.Array,
    n: int,
    d: int,
    bounds: np.ndarray | None = None,
    scramble: bool = True,
    optimization: Literal["random-cd", "lloyd"] | None = None,
) -> jax.Array
```

Latin Hypercube Sampling in `d` dimensions via `scipy.stats.qmc.LatinHypercube`. If `bounds` is None, points are returned in the unit hypercube `[0, 1]^d` with no scaling applied.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `key` | `jax.Array` | JAX PRNGKey used to seed scipy's RNG | required |
| `n` | `int` | Number of samples | required |
| `d` | `int` | Number of dimensions | required |
| `bounds` | `np.ndarray or None` | Shape `(d, 2)`. Each row is `[min, max]` for one dimension. If None, returns points in `[0, 1]^d` | `None` |
| `scramble` | `bool` | If True, randomly place samples within strata. If False, center samples within strata | `True` |
| `optimization` | `"random-cd"`, `"lloyd"`, or `None` | Post-processing optimisation to improve space-filling quality | `None` |

**Returns:**

| Type | Description |
|------|-------------|
| `jax.Array` | Sampled points of shape `(n, d)` |

**Notes:**

| optimization | behaviour |
|-------------|-----------|
| `None` | No optimisation -- fastest |
| `"random-cd"` | Minimise centered discrepancy via random coordinate permutations |
| `"lloyd"` | Move points toward a more uniform distribution via Lloyd's algorithm |

---

#### `lhs_sample_regional`

```
lhs_sample_regional(
    key: jax.Array,
    n: int,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    scramble: bool = True,
    optimization: Literal["random-cd", "lloyd"] | None = None,
) -> tuple[jax.Array, jax.Array]
```

Latin Hypercube Sampling in a lon/lat box.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `key` | `jax.Array` | JAX PRNGKey | required |
| `n` | `int` | Number of samples | required |
| `lon_bounds` | `tuple[float, float]` | `(lon_min, lon_max)` in degrees | required |
| `lat_bounds` | `tuple[float, float]` | `(lat_min, lat_max)` in degrees | required |
| `scramble` | `bool` | Randomly place samples within strata | `True` |
| `optimization` | `"random-cd"`, `"lloyd"`, or `None` | Post-processing optimisation. See `lhs_sample` | `None` |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[jax.Array, jax.Array]` | `(lons, lats)` each shape `(n,)` |

---

#### `lhs_sample_volume`

```
lhs_sample_volume(
    key: jax.Array,
    n: int,
    bounds: np.ndarray,
    scramble: bool = True,
    optimization: Literal["random-cd", "lloyd"] | None = None,
) -> jax.Array
```

Latin Hypercube Sampling in a 3D volume.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `key` | `jax.Array` | JAX PRNGKey | required |
| `n` | `int` | Number of samples | required |
| `bounds` | `np.ndarray` | Shape `(3, 2)`. Each row is `[min, max]` for one dimension | required |
| `scramble` | `bool` | Randomly place samples within strata | `True` |
| `optimization` | `"random-cd"`, `"lloyd"`, or `None` | Post-processing optimisation. See `lhs_sample` | `None` |

**Returns:**

| Type | Description |
|------|-------------|
| `jax.Array` | Sampled coordinates of shape `(n, 3)` |

---

### Private Helpers

#### `_key_to_seed`

```
_key_to_seed(key: jax.Array) -> int
```

Derive a reproducible integer seed in `[0, 2^31 - 1]` from a JAX PRNGKey. Used internally to bridge JAX's functional PRNG to scipy's stateful RNG.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `key` | `jax.Array` | JAX PRNGKey | required |

**Returns:**

| Type | Description |
|------|-------------|
| `int` | Integer seed in `[0, 2^31 - 1]` |