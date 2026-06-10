"""
experiments/sparse_obs_cross_attn/plotting/plotting.py

Plotting functions for the sparse_obs_cross_attn experiment: confusion
matrix / per-class metric charts, and geographic attention visualizations.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import jax
import matplotlib.pyplot as plt

from experiments.sparse_obs_cross_attn.train.model import TCClassifier


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: list[str],
    normalize:   bool = True,
    title:       str  = 'Confusion Matrix',
    ax:          Optional[Any] = None,
) -> plt.Figure:
    """Heatmap of confusion matrix.

    Parameters
    ----------
    normalize : bool
        True  → row-normalised (each row sums to 1; value = recall per class).
        False → raw counts.
    ax : matplotlib Axes, optional
        If None a new figure is created.

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 9))
    else:
        fig = ax.figure

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display  = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
        vmax     = 1.0
        clabel   = 'Recall (fraction of true class)'
        cell_fmt = '{:.2f}'
    else:
        display  = cm.astype(float)
        vmax     = float(cm.max()) if cm.max() > 0 else 1.0
        clabel   = 'Count'
        cell_fmt = '{:d}'

    im   = ax.imshow(display, interpolation='nearest', cmap='Blues',
                     vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(clabel, fontsize=9)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted class', fontsize=9)
    ax.set_ylabel('True class', fontsize=9)
    ax.set_title(title, fontsize=11)

    thresh = display.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = display[i, j]
            txt = cell_fmt.format(val if normalize else int(cm[i, j]))
            ax.text(j, i, txt, ha='center', va='center', fontsize=6,
                    color='white' if val > thresh else 'black')

    fig.tight_layout()
    return fig


def plot_class_metrics(
    metrics:     dict[int, dict[str, float]],
    class_names: list[str],
    ax:          Optional[Any] = None,
) -> plt.Figure:
    """Grouped bar chart: per-class precision, recall, F1.

    Parameters
    ----------
    metrics : dict returned by per_class_metrics()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    else:
        fig = ax.figure

    n     = len(class_names)
    x     = np.arange(n)
    w     = 0.25

    prec = [metrics[k]['precision'] for k in range(n)]
    rec  = [metrics[k]['recall']    for k in range(n)]
    f1   = [metrics[k]['f1']        for k in range(n)]

    ax.bar(x - w, prec, w, label='Precision', color='steelblue',  alpha=0.85)
    ax.bar(x,     rec,  w, label='Recall',    color='darkorange', alpha=0.85)
    ax.bar(x + w, f1,   w, label='F1',        color='seagreen',   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-class Precision / Recall / F1')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    return fig


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


def plot_attention_geographic(
    weights:           np.ndarray,
    batch:             dict,
    location_encoding: str,
    fov_lat:           Optional[tuple[float, float]] = None,
    fov_lon:           Optional[tuple[float, float]] = None,
    radius_km:         float = 500.0,
    sample_idx:        int   = 0,
    head_agg:          str   = 'mean',
    ax:                Optional[Any] = None,
) -> plt.Figure:
    """Plot per-station attention weight for one sample.

    For ``unit_circle`` encoding: polar axes — radius = normalised distance
    from storm centre, angle = bearing.  The storm is at the origin.

    For ``domain`` encoding: Cartesian axes — decoded lat/lon from the
    normalised coord representation.  Query position marked with a star.

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

    # Normalise to [0, 1] for sizing/colouring
    w_norm = (w_real - w_real.min()) / (w_real.max() - w_real.min() + 1e-12)

    if location_encoding == 'unit_circle':
        norm_dist   = coords[mask, 0]          # [0, 1]
        bearing_rad = coords[mask, 1]          # [0, 2π)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7),
                                   subplot_kw={'projection': 'polar'})
        else:
            fig = ax.figure

        sc = ax.scatter(
            bearing_rad, norm_dist,
            s=30 + w_norm * 250,
            c=w_real,
            cmap='YlOrRd',
            alpha=0.85,
            edgecolors='k',
            linewidths=0.4,
            zorder=3,
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

    else:  # domain
        if fov_lat is None or fov_lon is None:
            raise ValueError("fov_lat and fov_lon required for domain encoding.")

        half_pi  = float(np.pi / 2)
        lat_min, lat_max = fov_lat
        lon_min, lon_max = fov_lon
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min

        # Decode station positions
        lats = (coords[mask, 0] / half_pi + 1) / 2 * lat_span + lat_min
        lons = (coords[mask, 1] / half_pi + 1) / 2 * lon_span + lon_min

        # Decode query position
        q_lat = (query_coords[0] / half_pi + 1) / 2 * lat_span + lat_min
        q_lon = (query_coords[1] / half_pi + 1) / 2 * lon_span + lon_min

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 7))
        else:
            fig = ax.figure

        sc = ax.scatter(
            lons, lats,
            s=30 + w_norm * 250,
            c=w_real,
            cmap='YlOrRd',
            alpha=0.85,
            edgecolors='k',
            linewidths=0.4,
            zorder=3,
        )
        ax.scatter([q_lon], [q_lat], marker='*', s=250, color='royalblue',
                   zorder=5, label='Query (storm centre)')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Self-attention weights (query row) (domain encoding)', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.colorbar(sc, ax=ax, label='Attention weight')

    fig.tight_layout()
    return fig
