"""
experiments/cyclone_jax/data/sparsity.py

Network sparsity over the field of view: how well can N surface sensors
(marine + land) physically resolve a storm over the FOV area? Nyquist-style
quantities only (step-3c ruling; clustering/gap statistics are deliberate
future additions):

    spacing_km    = sqrt(area / n)   mean inter-station spacing if the
                                     sensors were evenly spread
    resolvable_km = 2 * spacing_km   smallest resolvable feature — sampling
                                     needs two samples per wavelength, so
                                     anything smaller is invisible to the
                                     raw network (the "why a learned prior"
                                     number for a data-limited basin)

The FOV is the declared normalise.surface_coordinate bounds, read off the
NormSpec as norms.domain ({lat: [lo, hi], lon: [lo, hi]});
its area is exact on the sphere (utils.geoscience.latlon_box_area).
n_stations means DISTINCT sensor locations in the sample/window, not
observation rows. Consumers: EDA notebooks now; per-sample prediction-table
columns when train/evaluate.py lands.
"""

from __future__ import annotations

from utils.geoscience.geodesic import latlon_box_area


def network_sparsity(n_stations: int, domain: dict) -> dict:
    """Nyquist-style sparsity of ``n_stations`` distinct sensors over the
    domain FOV.

    Parameters
    ----------
    n_stations : int
        Distinct sensor locations present (e.g. station_mask.sum() for a
        sample, or unique active stations in a time window).
    domain : dict
        The FOV (NormSpec.domain): {'lat': [lo, hi], 'lon': [lo, hi]}.

    Returns
    -------
    dict  {n_stations, area_km2, spacing_km, resolvable_km}
          (spacing/resolvable are inf when n_stations == 0).
    """
    lat, lon = domain['lat'], domain['lon']
    area = latlon_box_area(lon[0], lon[1], lat[0], lat[1])
    n = int(n_stations)
    spacing = (area / n) ** 0.5 if n > 0 else float('inf')
    return {
        'n_stations':    n,
        'area_km2':      float(area),
        'spacing_km':    float(spacing),
        'resolvable_km': float(2.0 * spacing),
    }
