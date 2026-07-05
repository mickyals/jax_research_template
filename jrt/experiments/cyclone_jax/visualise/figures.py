"""
experiments/cyclone_jax/visualise/figures.py

HOW figures look; train/log.py decides when/what (layering ruling). The
three v1 figures ruled 2026-07-05 — confusion matrix, storm panel, storm
sequence + gif — new figures land only when a need requires them.

Storm panel encoding: station dots + a STAR coloured by the TRUE class
sitting inside a RING coloured by the PREDICTED class — agreement reads as
ring-matches-star, a miss pops visually (one figure, not paired plots).

cartopy is OPTIONAL: panels draw a coastline basemap when importable and
``basemap=True``, else plain lon/lat axes. Tests never render cartopy
features (Natural Earth shapefiles download at draw time).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _CARTOPY = True
except ImportError:                                   # pragma: no cover
    _CARTOPY = False


def class_colour(k: int, n_classes: int):
    """Stable colour for class k: viridis sampled over the class range."""
    return plt.get_cmap('viridis')(k / max(n_classes - 1, 1))


def confusion_matrix_figure(cm, class_names=None, title='confusion matrix'):
    """(C, C) counts (rows = true, cols = predicted) -> annotated figure."""
    cm = np.asarray(cm)
    n = cm.shape[0]
    names = list(class_names) if class_names else [str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{int(cm[i, j])}', ha='center', va='center',
                    fontsize=8,
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def storm_panel_figure(station_lon, station_lat, storm_lon, storm_lat,
                       true_class, pred_class, n_classes, title='',
                       domain=None, basemap=True):
    """One storm fix: station dots, truth-star inside a predicted-class ring.

    Parameters
    ----------
    station_lon, station_lat : arrays of VALID stations (mask pre-applied)
    storm_lon, storm_lat : the fix position
    true_class, pred_class : int class indices (colour the star / the ring)
    n_classes : int   colour-scale range
    title : str       caller composes it (name/sid, time, true vs pred,
                      sensor count, resolvable_km — log.py territory)
    domain : dict, optional   data yaml domain block -> fixed extent
    basemap : bool    coastline basemap when cartopy is available
    """
    use_map = basemap and _CARTOPY
    if use_map:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(6.4, 4.8),
                               subplot_kw={'projection': proj})
        if domain:
            ax.set_extent([domain['lon'][0], domain['lon'][1],
                           domain['lat'][0], domain['lat'][1]], crs=proj)
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f2efe9')
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#dceaf3')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='grey',
                          alpha=0.4)
        gl.top_labels = gl.right_labels = False
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        if domain:
            ax.set_xlim(domain['lon'])
            ax.set_ylim(domain['lat'])
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')

    lon, lat = np.asarray(station_lon), np.asarray(station_lat)
    if lon.size:
        ax.scatter(lon, lat, s=12, c='#3b6ea5', edgecolors='none', zorder=4)
    ax.scatter([storm_lon], [storm_lat], s=900, marker='o',
               facecolors='none',
               edgecolors=[class_colour(int(pred_class), n_classes)],
               linewidths=2.5, zorder=5)
    ax.scatter([storm_lon], [storm_lat], s=300, marker='*',
               c=[class_colour(int(true_class), n_classes)],
               edgecolors='black', linewidths=0.6, zorder=6)
    ax.set_title(title, fontsize=8)
    return fig


def storm_sequence_figures(samples, n_classes, domain=None, basemap=True):
    """Time-ordered sample dicts -> list of storm-panel figures.

    Each sample: {station_lon, station_lat, storm_lon, storm_lat,
    true_class, pred_class, title}. The gif is these frames via save_gif;
    the caller saves a FEW of them as stills (not every frame).
    """
    return [storm_panel_figure(s['station_lon'], s['station_lat'],
                               s['storm_lon'], s['storm_lat'],
                               s['true_class'], s['pred_class'], n_classes,
                               title=s.get('title', ''), domain=domain,
                               basemap=basemap)
            for s in samples]


def save_gif(figures, path, duration_ms=600):
    """Rasterise figures (Agg draw) into one looping gif via Pillow.

    Figures are not closed here — callers own their lifecycle (and may
    still save some frames as svg/png stills).
    """
    from PIL import Image
    frames = []
    for fig in figures:
        fig.canvas.draw()
        frames.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())).convert('RGB'))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    return path
