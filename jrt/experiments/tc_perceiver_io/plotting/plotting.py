"""
experiments/tc_perceiver_io/plotting/plotting.py

Plotting functions for the tc_perceiver_io experiment: confusion
matrix / per-class metric charts, and per-component Perceiver-IO attention
visualizations.

The model's ``model.apply(v, X, train=False, return_weights=True)`` returns
``(output, attn_dict)`` where attn_dict holds PRE-softmax scores, one per
component (see model.TCPerceiverIO):

    read       (B, H, N, M)        N latents cross-attend M station tokens
    processor  (L, B, H, N, N)     L blocks of latent self-attention
    decoder    (B, H, 1, N)        single output query cross-attends N latents
                                   (None for decode_mode='avgproj' / headless)

Apply ``softmax`` over the LAST axis to turn any of these into attention
distributions. The three plotters below each take the softmaxed weights:

    plot_attention_geographic   Read map  — which stations the latents attend to
    plot_attention_matrix_grid  Processor — layers × heads grid of N×N matrices
    plot_decoder_query          Decoder   — output query's weight over N latents
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

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
# Attention visualization — one plotter per Perceiver-IO component
# ---------------------------------------------------------------------------

def plot_attention_matrix_grid(
    weights:    np.ndarray,
    sample_idx: int = 0,
    cmap:       str = 'viridis',
    title:      Optional[str] = None,
) -> plt.Figure:
    """Layers × heads grid of the Processor's N×N latent self-attention matrices.

    One panel per (layer, head) for a single sample, plain ``imshow`` with NO
    per-latent tick labels (unreadable at N = 16+) and a shared colour scale.
    Rows are query latents, columns the latents they attend to; each row of a
    softmaxed matrix sums to 1.

    Parameters
    ----------
    weights : np.ndarray (num_layers, B, num_heads, N, N)
        Softmaxed Processor scores — ``softmax(attn['processor'], axis=-1)``.
    sample_idx : int
        Which sample in the batch to visualise.
    cmap : str
    title : str, optional
        Figure suptitle. Default names the sample.

    Returns
    -------
    plt.Figure
    """
    w = np.asarray(weights)[:, sample_idx]          # (L, H, N, N)
    L, H, _, _ = w.shape
    vmax = float(w.max()) if w.max() > 0 else 1.0

    fig, axes = plt.subplots(
        L, H, figsize=(2.2 * H + 1.2, 2.2 * L),
        squeeze=False, sharex=True, sharey=True,
    )
    for l in range(L):
        for h in range(H):
            ax = axes[l, h]
            im = ax.imshow(w[l, h], cmap=cmap, vmin=0.0, vmax=vmax,
                           origin='upper', aspect='equal',
                           interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if l == 0:
                ax.set_title(f'head {h}', fontsize=8)
            if h == 0:
                ax.set_ylabel(f'layer {l}', fontsize=8)

    fig.colorbar(im, ax=axes, label='Attention weight',
                 shrink=0.85, pad=0.02)
    fig.suptitle(
        title if title is not None
        else f'Processor self-attention — sample {sample_idx} '
             f'(latent rows attend to latent columns)',
        fontsize=10,
    )
    return fig


def plot_decoder_query(
    weights:    np.ndarray,
    sample_idx: int = 0,
    title:      Optional[str] = None,
) -> plt.Figure:
    """Heads × latents heatmap of the Decoder output query's attention.

    The Decoder's single learned output query cross-attends the N latents
    before classifying (decode_mode='attention'). This shows, per head, how
    that query weights each latent — each row (head) is a distribution over the
    N latents and sums to 1.

    Parameters
    ----------
    weights : np.ndarray (B, num_heads, 1, N)
        Softmaxed Decoder scores — ``softmax(attn['decoder'], axis=-1)``.
        Pass only when decode_mode='attention' (it is None for 'avgproj').
    sample_idx : int
        Which sample in the batch to visualise.
    title : str, optional

    Returns
    -------
    plt.Figure
    """
    w = np.asarray(weights)[sample_idx, :, 0, :]    # (H, N)
    H, N = w.shape
    vmax = float(w.max()) if w.max() > 0 else 1.0

    return plot_heatmap(
        w,
        row_labels=[f'head {h}' for h in range(H)],
        col_labels=[str(i) for i in range(N)],
        xlabel='Latent index', ylabel='Head',
        cmap='magma', vmin=0.0, vmax=vmax,
        title=title if title is not None
        else 'Decoder output-query attention over latents',
        colorbar_label='Attention weight',
        annotate=(N <= 24), fmt='.2f', annotate_fontsize=6,
        figsize=(min(0.7 * N + 2.5, 14), 0.6 * H + 2.0),
    )


def plot_attention_geographic(
    read_weights:      np.ndarray,
    batch:             dict,
    location_encoding: str,
    fov_lat:           Optional[tuple[float, float]] = None,
    fov_lon:           Optional[tuple[float, float]] = None,
    radius_km:         float = 500.0,
    sample_idx:        int   = 0,
    head_agg:          str   = 'mean',
    geo:               bool | dict = False,
    storm_latlon:      Optional[tuple[float, float]] = None,
    station_latlon:    Optional[tuple[np.ndarray, np.ndarray]] = None,
    query_latlon:      Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Geographic Read map — which stations the model attends to.

    Aggregates the Read cross-attention over the N latents (and heads) to a
    single per-station weight, then plots it on the station geometry. Each
    latent's softmaxed row sums to 1 over the M stations, so the latent-mean is
    a per-station share of attention mass (summing to ~1 across real stations).

    For ``unit_circle`` encoding: the station coords ARE a storm-centred local
    map (x = east, y = north, storm at the origin) — scatter them directly on
    equal-aspect Cartesian axes, with distance rings at 0.25/0.5/0.75/1.0 of
    radius_km. Bespoke to this plot, so it composes ``_value_scatter`` directly.

    For ``domain`` encoding: lat/lon axes. The decoded positions are supplied by
    the caller (``station_latlon`` / ``query_latlon``) rather than decoded here
    — this module is a viz layer and does not depend on the experiment's
    coordinate encoding; the caller decodes via
    ``evaluate.domain_latlon_for_sample`` (which wraps
    ``data.transforms.encoding.decode_domain``) and passes the result in. The
    storm centre is marked with a star. Renders via
    ``utils.plotting.fields.plot_scatter_overlay``.

    Parameters
    ----------
    read_weights : np.ndarray (B, H, N, M)
        Softmaxed Read scores — ``softmax(attn['read'], axis=-1)`` — where N is
        the number of latents and M the (padded) station count.
    batch : dict
        Raw batch dict (contains 'X' with station_coords, station_mask).
    location_encoding : {'unit_circle', 'domain'}
    fov_lat, fov_lon : required for domain mode.
    radius_km : float
        Search radius used when building samples (unit_circle mode label).
    sample_idx : int
        Which sample in the batch to visualise.
    head_agg : {'mean', 'max'}
        How to collapse the head dimension before the latent-mean.
    geo : bool or dict
        Draw on a cartopy map with coastlines/borders (requires cartopy,
        optional dependency; default False keeps the cartopy-free plain-axes
        plot). Domain mode: forwarded to ``plot_scatter_overlay`` (PlateCarree).
        unit_circle mode: an AzimuthalEquidistant map centred on the storm — the
        local x-y encoding times radius_km IS that projection's native metre
        grid, so stations, km rings, and coastlines align exactly; requires
        ``storm_latlon``. Dict keys (scale/color/lw/gridlines) are forwarded to
        the canvas factory.
    storm_latlon : (lat, lon), optional
        Absolute storm position in degrees — the projection centre for the
        unit_circle geo map (available as
        batch['meta']['query_lat']/['query_lon']). Ignored unless unit_circle
        mode with geo enabled.
    station_latlon : (lats, lons), optional
        DOMAIN mode only: decoded latitudes/longitudes of the REAL (masked)
        stations, aligned with the masked attention weights — decoded by the
        caller. Required for domain encoding.
    query_latlon : (lat, lon), optional
        DOMAIN mode only: decoded query/storm position in degrees. Required for
        domain encoding.
    """
    X      = batch['X']
    coords = np.asarray(X['station_coords'][sample_idx])   # (M, 2)
    mask   = np.asarray(X['station_mask'][sample_idx])     # (M,) bool

    # Aggregate the Read attention to a single per-station weight: collapse the
    # head axis (mean/max), then mean over the N latents -> (M,).
    w = np.asarray(read_weights)[sample_idx]               # (H, N, M)
    w = w.mean(axis=0) if head_agg == 'mean' else w.max(axis=0)   # (N, M)
    w_station = w.mean(axis=0)                              # (M,) latent-mean
    w_real = w_station[mask]                                # (n_real,)

    if location_encoding == 'unit_circle':
        x = coords[mask, 0]                    # east offset,  [-1, 1]
        y = coords[mask, 1]                    # north offset, [-1, 1]

        geo_opts = {} if geo is True else dict(geo) if isinstance(geo, dict) else None
        if geo_opts is not None:
            # Azimuthal-equidistant map centred on the storm: native axes
            # coordinates are metres east/north of the centre, which is
            # exactly the local x-y encoding scaled by the radius — plot
            # in native metres with the default transform.
            if storm_latlon is None:
                raise ValueError(
                    "plot_attention_geographic: geo=True with unit_circle "
                    "encoding requires storm_latlon=(lat, lon) — available "
                    "as batch['meta']['query_lat']/['query_lon']."
                )
            from utils.plotting._geo import _make_geoaxes
            r_m = radius_km * 1000.0
            fig, ax, _ = _make_geoaxes(
                figsize=(7, 7),
                extent=[-1.08 * r_m, 1.08 * r_m, -1.08 * r_m, 1.08 * r_m],
                projection='azimuthal',
                center=(float(storm_latlon[0]), float(storm_latlon[1])),
                **geo_opts,
            )
            unit = r_m   # data coords below are in metres
        else:
            fig, ax = plt.subplots(figsize=(7, 7))
            unit = 1.0   # data coords below are fractions of the radius

        sc = _value_scatter(
            ax, x * unit, y * unit, values=w_real,
            cmap='YlOrRd', size_range=(30, 280),
            alpha=0.85, edgecolors='k', linewidths=0.4, zorder=3,
        )
        # Storm centre — the query/origin of the local map
        ax.scatter([0], [0], marker='*', s=200, color='royalblue',
                   zorder=5, label='Storm centre')

        # Distance rings with km labels (north-up: +y = north)
        for r in (0.25, 0.5, 0.75, 1.0):
            ax.add_patch(plt.Circle(
                (0, 0), r * unit, fill=False, color='grey',
                linewidth=0.6, linestyle='--', zorder=2,
            ))
            ax.annotate(
                f'{r * radius_km:.0f} km',
                xy=(r * unit / np.sqrt(2), r * unit / np.sqrt(2)),
                fontsize=7, color='grey', ha='left', va='bottom',
            )

        if geo_opts is None:
            ax.set_xlim(-1.08, 1.08)
            ax.set_ylim(-1.08, 1.08)
            ax.set_aspect('equal')
            ax.set_xlabel('East offset (× radius)')
            ax.set_ylabel('North offset (× radius)')
        else:
            # Re-assert the radius-box extent AFTER plotting: scatter/rings
            # autoscale the GeoAxes to the (clustered) data, which can shrink
            # the view and clip the coastlines. Pinning it back guarantees the
            # cfeatures within the radius box render. ax.projection is the
            # azimuthal CRS whose native units are metres from the centre.
            ax.set_extent(
                [-1.08 * r_m, 1.08 * r_m, -1.08 * r_m, 1.08 * r_m],
                crs=ax.projection,
            )
        ax.set_title('Read attention — stations the latents attend to\n'
                     '(storm-centred local map, north up)',
                     pad=15, fontsize=10)
        fig.colorbar(sc, ax=ax, label='Attention weight', shrink=0.7, pad=0.1)
        fig.tight_layout()
        return fig

    # domain — positions are decoded by the caller (this viz layer does not
    # import the experiment's coordinate encoding).
    if fov_lat is None or fov_lon is None:
        raise ValueError("fov_lat and fov_lon required for domain encoding.")
    if station_latlon is None or query_latlon is None:
        raise ValueError(
            "plot_attention_geographic: domain encoding requires station_latlon "
            "and query_latlon, decoded by the caller "
            "(evaluate.domain_latlon_for_sample) — plotting does not decode coords."
        )

    lats, lons   = station_latlon          # decoded REAL (masked) stations
    q_lat, q_lon = query_latlon
    lat_min, lat_max = fov_lat
    lon_min, lon_max = fov_lon

    return plot_scatter_overlay(
        None, lons, lats, scatter_values=w_real,
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap='YlOrRd',
        title='Read attention — stations the latents attend to (domain encoding)',
        xlabel='Longitude', ylabel='Latitude',
        colorbar_label='Attention weight',
        scatter_size_range=(30, 280), grid=True, geo=geo,
        marker_x=q_lon, marker_y=q_lat, marker_label='Storm centre',
        marker_kwargs={'s': 250}, figsize=(9, 7),
    )
