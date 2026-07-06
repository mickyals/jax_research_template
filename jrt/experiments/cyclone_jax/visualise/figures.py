"""
experiments/cyclone_jax/visualise/figures.py

HOW figures look; train/log.py decides when/what (layering ruling). The
storm panel + storm sequence/gif figures ruled 2026-07-05 — new figures
land only when a need requires them. The confusion-matrix figure is
model-agnostic and lives in jrt (utils.plotting.fields
.confusion_matrix_figure) — import it from there.

Storm panel encoding: station dots + the storm's track so far (grey
trail) + a STAR coloured by the TRUE class sitting inside a RING coloured
by the PREDICTED class — agreement reads as ring-matches-star, a miss
pops visually (one figure, not paired plots). Class colours follow the
standard Saffir–Simpson track-map palette (SSHS_COLORS, keyed by the
TargetSpec class NAME); classes without a known name fall back to
viridis. Sources and classes get SEPARATE legends (sources on the axes,
classes below the figure) — never one shared box.

cartopy is OPTIONAL: panels draw a coastline basemap via
utils.plotting.geo when importable and ``basemap=True``, else plain
lon/lat axes. Tests never render cartopy features (Natural Earth
shapefiles download at draw time).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.plotting.geo import (add_map_features, cartopy_available,
                                make_geoaxes)

# Station-dot styling per source-ID code (phase-0 scalar lock: land -1,
# upper 0, marine +1 — the x['id'] field). Marine keeps the historic
# single-colour dot blue; unknown codes fall back to it, labelled 'id <k>'.
SOURCE_STYLE = {
    -1: ('land',   '#8c6d31'),
    0:  ('upper',  '#7b52ab'),
    1:  ('marine', '#3b6ea5'),
}
_DOT_FALLBACK = '#3b6ea5'

# Standard NHC/track-map Saffir–Simpson palette, keyed by targets.SSHS_NAMES
# entries (the TargetSpec class-name strings). Below-TS categories use
# greys/blues consistent with common track maps.
SSHS_COLORS = {
    'Post-Tropical':  '#c3c3c3',
    'Disturbance':    '#a5a5a5',
    'Depression':     '#5ebaff',
    'Tropical Storm': '#00faf4',
    'Cat 1':          '#ffffcc',
    'Cat 2':          '#ffe775',
    'Cat 3':          '#ffc140',
    'Cat 4':          '#ff8f20',
    'Cat 5':          '#ff6060',
}


def class_colour(k: int, n_classes: int, class_names=None):
    """Colour for class k: the SSHS palette when its name is known,
    else viridis sampled over the class range (stable fallback)."""
    if class_names is not None:
        name = str(class_names[k])
        if name in SSHS_COLORS:
            return SSHS_COLORS[name]
    return plt.get_cmap('viridis')(k / max(n_classes - 1, 1))


def storm_panel_figure(station_lon, station_lat, storm_lon, storm_lat,
                       true_class, pred_class, n_classes, title='',
                       domain=None, basemap=True, station_id=None,
                       class_names=None, track_lon=None, track_lat=None):
    """One storm fix: station dots, truth-star inside a predicted-class ring.

    Parameters
    ----------
    station_lon, station_lat : arrays of VALID stations (mask pre-applied)
    storm_lon, storm_lat : the fix position
    true_class, pred_class : int class indices (colour the star / the ring)
    n_classes : int   colour-scale range
    title : str       caller composes it (name/sid, time, true vs pred,
                      per-source counts, resolvable_km — log.py territory)
    domain : dict, optional   data yaml domain block -> fixed extent
    basemap : bool    coastline basemap when cartopy is available
    station_id : array, optional   per-station source codes (x['id'],
                      same mask as the coords) -> dots coloured per
                      SOURCE_STYLE + a small legend; None = single colour
    class_names : sequence of str, optional   TargetSpec.class_names —
                      switches star/ring to the SSHS palette and adds a
                      class legend below the axes (separate from the
                      source legend)
    track_lon, track_lat : arrays, optional   the storm's fixes up to and
                      including this one, time-ordered -> grey trail
    """
    use_map = basemap and cartopy_available()
    if use_map:
        extent = ([domain['lon'][0], domain['lon'][1],
                   domain['lat'][0], domain['lat'][1]] if domain else None)
        fig, ax, transform = make_geoaxes(figsize=(6.4, 4.8), extent=extent,
                                          fill=True)
        tkw = {'transform': transform}
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        if domain:
            ax.set_xlim(domain['lon'])
            ax.set_ylim(domain['lat'])
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        tkw = {}

    lon, lat = np.asarray(station_lon), np.asarray(station_lat)
    if lon.size:
        if station_id is None:
            ax.scatter(lon, lat, s=12, c=_DOT_FALLBACK, edgecolors='none',
                       zorder=4, **tkw)
        else:
            codes = np.rint(np.asarray(station_id)).astype(int)
            for code, (label, colour) in SOURCE_STYLE.items():
                sel = codes == code
                if sel.any():
                    ax.scatter(lon[sel], lat[sel], s=12, c=colour,
                               edgecolors='none', zorder=4, label=label,
                               **tkw)
            for code in np.unique(codes[~np.isin(codes,
                                                 list(SOURCE_STYLE))]):
                sel = codes == code
                ax.scatter(lon[sel], lat[sel], s=12, c=_DOT_FALLBACK,
                           edgecolors='none', zorder=4, label=f'id {code}',
                           **tkw)
            ax.legend(loc='upper right', fontsize=6, markerscale=1.5,
                      framealpha=0.8)
    if track_lon is not None and track_lat is not None and len(track_lon):
        ax.plot(np.asarray(track_lon), np.asarray(track_lat),
                color='grey', lw=0.8, alpha=0.5, zorder=3, **tkw)
    ax.scatter([storm_lon], [storm_lat], s=380, marker='o',
               facecolors='none',
               edgecolors=[class_colour(int(pred_class), n_classes,
                                        class_names)],
               linewidths=1.8, zorder=5, **tkw)
    ax.scatter([storm_lon], [storm_lat], s=140, marker='*',
               c=[class_colour(int(true_class), n_classes, class_names)],
               edgecolors='black', linewidths=0.5, zorder=6, **tkw)
    if class_names is not None:
        handles = [Line2D([], [], marker='*', linestyle='',
                          color=class_colour(k, n_classes, class_names),
                          markeredgecolor='black', markersize=9,
                          label=str(class_names[k]))
                   for k in range(n_classes)]
        fig.legend(handles=handles, loc='lower center',
                   ncol=min(n_classes, 6), fontsize=7, framealpha=0.9,
                   title='SSHS class (star = true, ring = predicted)',
                   title_fontsize=7)
        fig.subplots_adjust(bottom=0.2)
    ax.set_title(title, fontsize=8)
    return fig


def accuracy_hexbin_figure(lon, lat, correct, domain=None, basemap=True,
                           gridsize=70, title=''):
    """Spatial correctness: hexbin over the FOV, each bin the mean of the
    per-fix correct flags falling in it (1 = always right there, 0 =
    always wrong; RdYlGn). Empty bins stay blank (mincnt=1). On the
    cartopy path the bins sit on the geoaxes UNDER re-drawn coastline
    linework, so the geography stays readable through the field.

    Parameters
    ----------
    lon, lat : arrays   fix positions (raw/display coordinates)
    correct : bool/0-1 array   per-fix prediction correctness
    domain : dict, optional    data yaml domain block -> fixed bin extent,
                               so figures tile identically across runs
    basemap : bool             coastline basemap when cartopy is available
    gridsize : int             hexbin resolution across the lon span
    title : str                caller composes it (split, step, accuracy)
    """
    use_map = basemap and cartopy_available()
    if use_map:
        extent = ([domain['lon'][0], domain['lon'][1],
                   domain['lat'][0], domain['lat'][1]] if domain else None)
        fig, ax, transform = make_geoaxes(figsize=(6.4, 4.8), extent=extent,
                                          fill=True)
        tkw = {'transform': transform}
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        if domain:
            ax.set_xlim(domain['lon'])
            ax.set_ylim(domain['lat'])
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        tkw = {}
    hex_extent = ((domain['lon'][0], domain['lon'][1],
                   domain['lat'][0], domain['lat'][1]) if domain else None)
    hb = ax.hexbin(np.asarray(lon, float), np.asarray(lat, float),
                   C=np.asarray(correct, float),
                   reduce_C_function=np.mean, gridsize=gridsize,
                   extent=hex_extent, cmap='RdYlGn', vmin=0.0, vmax=1.0,
                   mincnt=1, linewidths=0.2, zorder=2, **tkw)
    if use_map:
        # coastline/border linework ABOVE the bins (make_geoaxes drew the
        # basemap below them)
        add_map_features(ax, zorder=3)
    fig.colorbar(hb, ax=ax, label='fraction correct', shrink=0.85)
    ax.set_title(title, fontsize=8)
    return fig


def storm_sequence_figures(samples, n_classes, domain=None, basemap=True,
                           class_names=None):
    """Time-ordered sample dicts -> list of storm-panel figures.

    Each sample: {station_lon, station_lat, storm_lon, storm_lat,
    true_class, pred_class, title, station_id?, track_lon?, track_lat?}.
    The gif is these frames via save_gif; the caller saves a FEW of them
    as stills (not every frame).
    """
    return [storm_panel_figure(s['station_lon'], s['station_lat'],
                               s['storm_lon'], s['storm_lat'],
                               s['true_class'], s['pred_class'], n_classes,
                               title=s.get('title', ''), domain=domain,
                               basemap=basemap,
                               station_id=s.get('station_id'),
                               class_names=class_names,
                               track_lon=s.get('track_lon'),
                               track_lat=s.get('track_lat'))
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
