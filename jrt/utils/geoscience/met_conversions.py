"""
Meteorological unit conversion utilities.

All functions accept scalars, NumPy arrays, or JAX arrays and return the
same type. Linear conversions are built from a shared factory — each is an
explicit module-level name so static analysers and IDEs can resolve them.

Sources:
    NOAA/NWS unit definitions
    IBTrACS v04r01 column documentation
"""

import math

# ---------------------------------------------------------------------------
# Wind radii kt thresholds in m/s — used for imputation / masking decisions.
# ---------------------------------------------------------------------------

_KT_TO_MS = 0.514444

R34_MS_THRESHOLD = 34 * _KT_TO_MS   # 17.49 m/s
R50_MS_THRESHOLD = 50 * _KT_TO_MS   # 25.72 m/s
R64_MS_THRESHOLD = 64 * _KT_TO_MS   # 32.92 m/s

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _linear(factor: float, doc: str):
    def _fn(x):
        return x * factor
    _fn.__doc__ = doc
    return _fn


def _affine(scale: float, offset: float, doc: str):
    def _fn(x):
        return x * scale + offset
    _fn.__doc__ = doc
    return _fn


# ---------------------------------------------------------------------------
# Wind speed
# ---------------------------------------------------------------------------

kt_to_ms  = _linear(0.514444,       "Knots to metres per second.")
ms_to_kt  = _linear(1 / 0.514444,   "Metres per second to knots.")
kmh_to_ms = _linear(1 / 3.6,        "Kilometres per hour to metres per second.")
ms_to_kmh = _linear(3.6,            "Metres per second to kilometres per hour.")
mph_to_ms = _linear(0.44704,        "Miles per hour to metres per second.")
ms_to_mph = _linear(1 / 0.44704,    "Metres per second to miles per hour.")

# ---------------------------------------------------------------------------
# Pressure
# ---------------------------------------------------------------------------

hpa_to_pa  = _linear(100.0,          "Hectopascals (millibars) to Pascals.")
pa_to_hpa  = _linear(0.01,           "Pascals to hectopascals (millibars).")
inhg_to_pa = _linear(3386.389,       "Inches of mercury to Pascals.")
pa_to_inhg = _linear(1 / 3386.389,   "Pascals to inches of mercury.")

# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

nmile_to_m = _linear(1852.0,         "Nautical miles to metres.")
m_to_nmile = _linear(1 / 1852.0,     "Metres to nautical miles.")
km_to_m    = _linear(1000.0,         "Kilometres to metres.")
m_to_km    = _linear(0.001,          "Metres to kilometres.")
ft_to_m    = _linear(0.3048,         "Feet to metres.")
m_to_ft    = _linear(1 / 0.3048,     "Metres to feet.")
mi_to_m    = _linear(1609.344,       "Statute miles to metres.")
m_to_mi    = _linear(1 / 1609.344,   "Metres to statute miles.")

# ---------------------------------------------------------------------------
# Angle
# ---------------------------------------------------------------------------

deg_to_rad = _linear(math.pi / 180.0,  "Degrees to radians.")
rad_to_deg = _linear(180.0 / math.pi,  "Radians to degrees.")

# ---------------------------------------------------------------------------
# Temperature  (affine: output = input * scale + offset)
# ---------------------------------------------------------------------------

celsius_to_kelvin     = _affine(1.0,     273.15,               "Celsius to Kelvin.")
kelvin_to_celsius     = _affine(1.0,    -273.15,               "Kelvin to Celsius.")
fahrenheit_to_celsius = _affine(5/9,    -32 * 5/9,             "Fahrenheit to Celsius.")
celsius_to_fahrenheit = _affine(9/5,     32.0,                 "Celsius to Fahrenheit.")
fahrenheit_to_kelvin  = _affine(5/9,    -32 * 5/9 + 273.15,   "Fahrenheit to Kelvin.")
kelvin_to_fahrenheit  = _affine(9/5,    -273.15 * 9/5 + 32,   "Kelvin to Fahrenheit.")

# ---------------------------------------------------------------------------
# Bearing helpers (non-trivial; kept as explicit functions)
# ---------------------------------------------------------------------------

def bearing_to_components(bearing_deg):
    """
    Convert bearing in degrees to (sin, cos) circular embedding.

    Bearing is clockwise from north, in [0, 360). The returned components
    are continuous features suitable for use in ML models — avoids the
    0/360 discontinuity that arises from using bearing directly.

    Parameters
    ----------
    bearing_deg : array-like
        Bearing(s) in degrees.

    Returns
    -------
    tuple
        (sin_component, cos_component), both in [-1, 1].
    """
    rad = bearing_deg * (math.pi / 180.0)
    import numpy as _np
    try:
        import jax
        import jax.numpy as _jnp
        if isinstance(bearing_deg, jax.Array):
            return _jnp.sin(rad), _jnp.cos(rad)
    except ImportError:
        pass
    rad = _np.asarray(rad)
    return _np.sin(rad), _np.cos(rad)


def components_to_bearing(sin_component, cos_component):
    """
    Recover bearing in degrees from (sin, cos) circular components.

    Parameters
    ----------
    sin_component, cos_component : array-like
        Circular embedding components as returned by ``bearing_to_components``.

    Returns
    -------
    array-like
        Bearing(s) in [0, 360).
    """
    import numpy as _np
    try:
        import jax
        import jax.numpy as _jnp
        if isinstance(sin_component, jax.Array):
            return _jnp.degrees(_jnp.arctan2(sin_component, cos_component)) % 360
    except ImportError:
        pass
    return _np.degrees(_np.arctan2(
        _np.asarray(sin_component), _np.asarray(cos_component)
    )) % 360


# ---------------------------------------------------------------------------
# Wind vector decomposition (meteorological FROM convention)
# ---------------------------------------------------------------------------

def wind_to_components(speed, direction_deg):
    """
    Decompose wind speed + direction into (east, north) velocity components.

    Uses the meteorological FROM convention: direction is the bearing the
    wind blows FROM, clockwise from north in [0, 360). A wind from the
    north (0°) therefore blows toward the south: u = 0, v = -speed.

        u_east  = -speed * sin(direction)
        v_north = -speed * cos(direction)

    Calm rule: wherever speed == 0 the components are (0, 0) even if
    direction is NaN — calm wind has no direction, but its velocity vector
    is exactly zero. Non-zero speed with NaN direction propagates NaN
    (genuinely unknown vector), as does NaN speed.

    Parameters
    ----------
    speed : array-like
        Wind speed(s), must be non-negative (any consistent unit,
        typically m/s). A negative speed would silently flip the
        direction by 180°, so the NumPy path rejects it. NaN = missing
        and passes through.
    direction_deg : array-like
        Direction(s) the wind blows from, degrees clockwise from north.

    Returns
    -------
    tuple
        (u_east, v_north) in the same unit as speed.

    Raises
    ------
    ValueError
        If any element of speed is negative (NumPy path only — the JAX
        path skips validation because value checks cannot run inside a
        jit trace).
    """

    import numpy as _np

    rad = direction_deg * (_np.pi / 180.0)
    try:
        import jax
        import jax.numpy as _jnp
        if isinstance(speed, jax.Array) or isinstance(direction_deg, jax.Array):
            calm = speed == 0
            u = _jnp.where(calm, 0.0, -speed * _jnp.sin(rad))
            v = _jnp.where(calm, 0.0, -speed * _jnp.cos(rad))
            return u, v
    except ImportError:
        pass
    speed = _np.asarray(speed)
    if _np.any(speed < 0):
        raise ValueError(
            "Wind speed must be non-negative — a negative speed flips the "
            "FROM direction by 180°. Negative values are not physically "
            "meaningful."
        )
    rad   = _np.asarray(rad)
    calm  = speed == 0
    u = _np.where(calm, 0.0, -speed * _np.sin(rad))
    v = _np.where(calm, 0.0, -speed * _np.cos(rad))
    return u, v


def components_to_wind(u_east, v_north):
    """
    Recover wind speed + FROM direction from (east, north) components.

        speed         = hypot(u_east, v_north)
        direction_deg = degrees(arctan2(-u_east, -v_north)) % 360

    Calm rule (inverse of ``wind_to_components``): wherever speed == 0 the
    direction is NaN — a calm wind has no direction, and 0 would falsely
    read as "from north".

    Parameters
    ----------
    u_east, v_north : array-like
        Wind velocity components as returned by ``wind_to_components``.

    Returns
    -------
    tuple
        (speed, direction_deg) with direction in [0, 360) or NaN at calm.
    """
    import numpy as _np
    try:
        import jax
        import jax.numpy as _jnp
        if isinstance(u_east, jax.Array) or isinstance(v_north, jax.Array):
            speed = _jnp.hypot(u_east, v_north)
            direction = _jnp.degrees(_jnp.arctan2(-u_east, -v_north)) % 360
            direction = _jnp.where(speed == 0, _jnp.nan, direction)
            return speed, direction
    except ImportError:
        pass
    u_east  = _np.asarray(u_east)
    v_north = _np.asarray(v_north)
    speed = _np.hypot(u_east, v_north)
    direction = _np.degrees(_np.arctan2(-u_east, -v_north)) % 360
    direction = _np.where(speed == 0, _np.nan, direction)
    return speed, direction
