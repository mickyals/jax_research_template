"""
experiments/sparse_obs_cross_attn/plotting/plotting.py

Plotting functions for the sparse_obs_cross_attn experiment: confusion
matrix / per-class metric charts, and geographic attention visualizations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import jax
import matplotlib.pyplot as plt

from experiments.sparse_obs_cross_attn.train.model import TCClassifier
from utils.plotting._style import _value_scatter
from utils.plotting.curves import plot_grouped_bars
from utils.plotting.fields import plot_heatmap, plot_scatter_overlay


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: list[str],
    normalize:   bool = True,
    title:       str  = 'Confusion Matrix',
) -> plt.Figure:
    """Heatmap of confusion matrix.

    Thin wrapper over ``utils.plotting.fields.plot_heatmap``: normalisation
    is the only confusion-matrix-specific decision, made here.

    Parameters
    ----------
    normalize : bool
        True  → row-normalised (each row sums to 1; value = recall per class).
        False → raw counts.

    Returns
    -------
    plt.Figure
    """
    if normalize:
        display = np.where(cm.sum(axis=1, keepdims=True) > 0,
                            cm.astype(float) / cm.sum(axis=1, keepdims=True), 0.0)
        vmax    = 1.0
        clabel  = 'Recall (fraction of true class)'
        fmt     = '.2f'
    else:
        display = cm.astype(float)
        vmax    = float(cm.max()) if cm.max() > 0 else 1.0
        clabel  = 'Count'
        fmt     = '.0f'

    return plot_heatmap(
        display, row_labels=class_names, col_labels=class_names,
        xlabel='Predicted class', ylabel='True class',
        cmap='Blues', vmin=0.0, vmax=vmax, title=title,
        colorbar_label=clabel, annotate=True, fmt=fmt,
        annotate_fontsize=6, figsize=(11, 9),
    )


def plot_class_metrics(
    metrics:     dict[int, dict[str, float]],
    class_names: list[str],
) -> plt.Figure:
    """Grouped bar chart: per-class precision, recall, F1.

    Thin wrapper over ``utils.plotting.curves.plot_grouped_bars``.

    Parameters
    ----------
    metrics : dict returned by per_class_metrics()
    """
    n    = len(class_names)
    prec = [metrics[k]['precision'] for k in range(n)]
    rec  = [metrics[k]['recall']    for k in range(n)]
    f1   = [metrics[k]['f1']        for k in range(n)]

    return plot_grouped_bars(
        {'Precision': prec, 'Recall': rec, 'F1': f1}, class_names,
        ylabel='Score', title='Per-class Precision / Recall / F1',
        ylim=(0, 1.05), colors=['steelblue', 'darkorange', 'seagreen'],
    )


# ---------------------------------------------------------------------------
# Attention visualization
# ---------------------------------------------------------------------------

def extract_attention_weights(
    model:     TCClassifier,
    variables: dict,
    batch:     dict,
) -> np.ndarray:
    """Run batch through model with return_weights=True.

    Returns
    -------
    np.ndarray float32 (B, num_heads, N+1)
        Self-attention weights (query row) from the last encoder layer.
        N = max_stations; the N+1-th element is the query's self-attention
        weight; padding positions have weight ≈ 0.
    """
    apply_fn = jax.jit(
        lambda X: model.apply(variables, X, train=False, return_weights=True)
    )
    _, weights = apply_fn(batch['X'])
    return np.asarray(weights, dtype=np.float32)


def _decode_domain_coords(
    coords:       np.ndarray,
    query_coords: np.ndarray,
    fov_lat:      tuple[float, float],
    fov_lon:      tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Decode normalised domain-encoded coordinates to lon/lat degrees.

    Parameters
    ----------
    coords : np.ndarray (n, 2)
        Normalised station coordinates, columns (lat, lon) each in
        [-pi/2, pi/2].
    query_coords : np.ndarray (2,)
        Normalised query coordinate, (lat, lon).
    fov_lat, fov_lon : tuple[float, float]
        (min, max) field-of-view bounds in degrees.

    Returns
    -------
    lons, lats : np.ndarray (n,)
        Decoded station longitudes/latitudes in degrees.
    q_lon, q_lat : float
        Decoded query longitude/latitude in degrees.
    """
    half_pi = float(np.pi / 2)
    lat_min, lat_max = fov_lat
    lon_min, lon_max = fov_lon
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min

    lats = (coords[:, 0] / half_pi + 1) / 2 * lat_span + lat_min
    lons = (coords[:, 1] / half_pi + 1) / 2 * lon_span + lon_min
    q_lat = (query_coords[0] / half_pi + 1) / 2 * lat_span + lat_min
    q_lon = (query_coords[1] / half_pi + 1) / 2 * lon_span + lon_min
    return lons, lats, q_lon, q_lat


def plot_attention_geographic(
    weights:           np.ndarray,
    batch:             dict,
    location_encoding: str,
    fov_lat:           Optional[tuple[float, float]] = None,
    fov_lon:           Optional[tuple[float, float]] = None,
    radius_km:         float = 500.0,
    sample_idx:        int   = 0,
    head_agg:          str   = 'mean',
    geo:               bool | dict = False,
) -> plt.Figure:
    """Plot per-station attention weight for one sample.

    For ``unit_circle`` encoding: polar axes — radius = normalised distance
    from storm centre, angle = bearing.  The storm is at the origin.
    Compass conventions and km-scaled radial ticks are bespoke to this
    plot, so it composes ``_value_scatter`` directly.

    For ``domain`` encoding: Cartesian axes — decoded lat/lon from the
    normalised coord representation.  Query position marked with a star.
    Renders via ``utils.plotting.fields.plot_scatter_overlay``.

    Parameters
    ----------
    weights : np.ndarray (B, H, N)
        From extract_attention_weights().
    batch : dict
        Raw batch dict (contains 'X' with station_coords, station_mask,
        query_coords).
    location_encoding : {'unit_circle', 'domain'}
    fov_lat, fov_lon : required for domain mode.
    radius_km : float
        Search radius used when building samples (unit_circle mode label).
    sample_idx : int
        Which sample in the batch to visualise.
    head_agg : {'mean', 'max'}
        How to collapse the head dimension before plotting.
    geo : bool or dict
        Domain mode only (ignored for unit_circle): forwarded to
        ``plot_scatter_overlay`` to draw on a PlateCarree map with
        coastlines/borders. Requires cartopy (optional dependency);
        default False keeps the cartopy-free plain-axes plot.
    """
    X            = batch['X']
    coords       = np.asarray(X['station_coords'][sample_idx])   # (N, 2)
    mask         = np.asarray(X['station_mask'][sample_idx])     # (N,) bool
    query_coords = np.asarray(X['query_coords'][sample_idx])     # (2,)

    # Aggregate attention over heads: (H, N+1) → (N+1,) then drop query self-weight
    w = weights[sample_idx]                                       # (H, N+1)
    w_station = w.mean(axis=0) if head_agg == 'mean' else w.max(axis=0)
    N = mask.shape[0]
    w_station = w_station[:N]                                     # (N,) drop query self-attn
    w_real = w_station[mask]                                      # (n_real,)

    if location_encoding == 'unit_circle':
        norm_dist   = coords[mask, 0]          # [0, 1]
        bearing_rad = coords[mask, 1]          # [0, 2π)

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
        sc = _value_scatter(
            ax, bearing_rad, norm_dist, values=w_real,
            cmap='YlOrRd', size_range=(30, 280),
            alpha=0.85, edgecolors='k', linewidths=0.4, zorder=3,
        )
        # Storm centre
        ax.scatter([0], [0], marker='*', s=200, color='royalblue',
                   zorder=5, label='Storm centre')

        ax.set_theta_zero_location('N')   # North at top
        ax.set_theta_direction(-1)        # Clockwise (compass convention)
        ax.set_rlim(0, 1)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_rlabel_position(45)
        ax.yaxis.set_tick_params(labelsize=7)
        tick_labels = [f'{r * radius_km:.0f} km' for r in [0.25, 0.5, 0.75, 1.0]]
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_title('Self-attention weights (query row)\n(polar: distance × bearing from storm)',
                     pad=15, fontsize=10)
        fig.colorbar(sc, ax=ax, label='Attention weight', shrink=0.7, pad=0.1)
        fig.tight_layout()
        return fig

    # domain
    if fov_lat is None or fov_lon is None:
        raise ValueError("fov_lat and fov_lon required for domain encoding.")

    lons, lats, q_lon, q_lat = _decode_domain_coords(
        coords[mask], query_coords, fov_lat, fov_lon,
    )
    lat_min, lat_max = fov_lat
    lon_min, lon_max = fov_lon

    return plot_scatter_overlay(
        None, lons, lats, scatter_values=w_real,
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap='YlOrRd',
        title='Self-attention weights (query row) (domain encoding)',
        xlabel='Longitude', ylabel='Latitude',
        colorbar_label='Attention weight',
        scatter_size_range=(30, 280), grid=True, geo=geo,
        marker_x=q_lon, marker_y=q_lat, marker_label='Query (storm centre)',
        marker_kwargs={'s': 250}, figsize=(9, 7),
    )
