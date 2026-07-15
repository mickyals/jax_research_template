"""
Tests for visualise/figures.py. Headless (Agg) and basemap=False throughout
so cartopy features are never rendered (Natural Earth downloads at draw).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.cyclone_jax.visualise.figures import (
    SOURCE_STYLE, SSHS_COLORS, accuracy_hexbin_figure,
    accuracy_vs_resolution_figure, class_colour, save_gif,
    storm_panel_figure, storm_sequence_figures,
    storm_track_correctness_figure,
)

N_CLS = 6
CLASS_NAMES = ('Tropical Storm', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5')
DOMAIN = {'lat': [0, 30], 'lon': [-100, -30]}


def _panel(true_class=2, pred_class=4):
    return storm_panel_figure(
        station_lon=np.array([-60.0, -55.0]),
        station_lat=np.array([12.0, 14.0]),
        storm_lon=-58.0, storm_lat=13.0, true_class=true_class,
        pred_class=pred_class, n_classes=N_CLS,
        title='TEST 2024  true 2 vs pred 4', domain=DOMAIN, basemap=False)


def _panel_ids(station_id):
    return storm_panel_figure(
        station_lon=np.array([-60.0, -55.0]),
        station_lat=np.array([12.0, 14.0]),
        storm_lon=-58.0, storm_lat=13.0, true_class=0, pred_class=0,
        n_classes=N_CLS, basemap=False, station_id=station_id)


# confusion_matrix_figure moved to utils.plotting.fields (PR #5 DRY ruling);
# its tests live in tests/utils/plotting/test_fields.py.


class TestStormPanel:

    def test_star_true_ring_pred_colours(self):
        fig = _panel(true_class=2, pred_class=4)
        ax = fig.axes[0]
        stations, ring, star = ax.collections[:3]
        assert np.allclose(star.get_facecolor()[0], class_colour(2, N_CLS))
        assert np.allclose(ring.get_edgecolor()[0], class_colour(4, N_CLS))
        assert ax.get_title() == 'TEST 2024  true 2 vs pred 4'
        assert tuple(ax.get_xlim()) == (-100, -30)      # domain extent
        assert ax.get_legend() is None                  # no ids -> no legend
        assert fig.legends == []              # no class_names -> no legend
        plt.close('all')

    def test_class_names_switch_to_sshs_palette(self):
        fig = storm_panel_figure(
            np.array([-60.0]), np.array([12.0]), -58.0, 13.0,
            true_class=0, pred_class=5, n_classes=N_CLS,
            class_names=CLASS_NAMES, basemap=False)
        ax = fig.axes[0]
        _, ring, star = ax.collections[:3]
        assert np.allclose(star.get_facecolor()[0], matplotlib.colors
                           .to_rgba(SSHS_COLORS['Tropical Storm']))
        assert np.allclose(ring.get_edgecolor()[0], matplotlib.colors
                           .to_rgba(SSHS_COLORS['Cat 5']))
        plt.close('all')

    def test_unknown_class_name_falls_back_to_viridis(self):
        assert (class_colour(1, N_CLS, ('weird', 'names') + CLASS_NAMES[2:])
                == plt.get_cmap('viridis')(1 / (N_CLS - 1)))

    def test_class_legend_separate_from_source_legend(self):
        fig = storm_panel_figure(
            np.array([-60.0]), np.array([12.0]), -58.0, 13.0,
            true_class=0, pred_class=1, n_classes=N_CLS,
            class_names=CLASS_NAMES, station_id=np.float32([-1]),
            basemap=False)
        ax = fig.axes[0]
        # sources on the axes, classes on the FIGURE — never one shared box
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ['land']
        (cls_leg,) = fig.legends
        assert ([t.get_text() for t in cls_leg.get_texts()]
                == list(CLASS_NAMES))
        plt.close('all')

    def test_track_draws_grey_trail(self):
        fig = storm_panel_figure(
            np.array([-60.0]), np.array([12.0]), -58.0, 13.0,
            true_class=0, pred_class=0, n_classes=N_CLS, basemap=False,
            track_lon=np.array([-61.0, -59.5, -58.0]),
            track_lat=np.array([11.0, 12.0, 13.0]))
        (trail,) = fig.axes[0].lines
        assert trail.get_xdata().shape == (3,)
        assert trail.get_color() == 'grey'
        plt.close('all')

    def test_no_track_no_lines(self):
        fig = _panel()
        assert len(fig.axes[0].lines) == 0
        plt.close('all')

    def test_no_stations_still_draws(self):
        fig = storm_panel_figure(np.array([]), np.array([]), -58.0, 13.0,
                                 0, 0, N_CLS, basemap=False)
        assert len(fig.axes[0].collections) == 2        # ring + star only
        plt.close('all')

    def test_station_ids_group_dots_with_legend(self):
        fig = storm_panel_figure(
            station_lon=np.array([-60.0, -55.0, -50.0, -45.0]),
            station_lat=np.array([12.0, 14.0, 16.0, 18.0]),
            storm_lon=-58.0, storm_lat=13.0, true_class=1, pred_class=1,
            n_classes=N_CLS, basemap=False,
            station_id=np.float32([-1, -1, 1, 0]))
        ax = fig.axes[0]
        # one dot group per source present (SOURCE_STYLE order) + ring + star
        assert len(ax.collections) == 5
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['land', 'upper', 'marine']
        land, upper, marine = ax.collections[:3]
        assert land.get_offsets().shape[0] == 2         # two land stations
        assert np.allclose(land.get_facecolor()[0],
                           matplotlib.colors.to_rgba(SOURCE_STYLE[-1][1]))
        assert np.allclose(marine.get_facecolor()[0],
                           matplotlib.colors.to_rgba(SOURCE_STYLE[1][1]))
        plt.close('all')

    def test_only_present_sources_in_legend(self):
        fig = _panel_ids(np.float32([1, 1]))
        labels = [t.get_text() for t in
                  fig.axes[0].get_legend().get_texts()]
        assert labels == ['marine']
        plt.close('all')

    def test_unknown_id_code_falls_back(self):
        fig = _panel_ids(np.float32([7, 7]))
        labels = [t.get_text() for t in
                  fig.axes[0].get_legend().get_texts()]
        assert labels == ['id 7']
        plt.close('all')


class TestAccuracyHexbin:

    def test_bins_hold_mean_correctness(self):
        # two far-apart clusters: one all-correct, one half-correct
        lon = np.array([-80.0, -80.0, -80.0, -40.0, -40.0])
        lat = np.array([10.0, 10.0, 10.0, 25.0, 25.0])
        correct = np.array([1, 1, 1, 1, 0], float)
        fig = accuracy_hexbin_figure(lon, lat, correct, domain=DOMAIN,
                                     basemap=False, gridsize=8,
                                     title='val correctness')
        ax = fig.axes[0]
        vals = np.asarray(ax.collections[0].get_array())
        assert sorted(vals.tolist()) == [0.5, 1.0]      # mean per bin
        assert ax.get_title() == 'val correctness'
        assert tuple(ax.get_xlim()) == (-100, -30)      # domain extent
        plt.close('all')

    def test_bool_correct_and_no_domain(self):
        fig = accuracy_hexbin_figure(
            np.array([-60.0, -61.0]), np.array([12.0, 13.0]),
            np.array([True, False]), basemap=False)
        assert len(fig.axes[0].collections) == 1
        plt.close('all')


class TestAccuracyVsResolution:

    def test_bins_hold_mean_accuracy(self):
        # two resolution clusters: fine-network fixes all correct,
        # coarse-network fixes 1/4 correct
        r = np.array([100.0] * 4 + [1000.0] * 4)
        correct = np.array([1, 1, 1, 1, 0, 0, 0, 1], float)
        fig = accuracy_vs_resolution_figure(r, correct, n_bins=2,
                                            title='val vs resolution')
        ax = fig.axes[0]
        assert sorted(ax.lines[0].get_ydata().tolist()) == [0.25, 1.0]
        assert ax.get_title() == 'val vs resolution'
        ax2 = fig.axes[1]                        # count bars on the twin
        assert sorted(p.get_height() for p in ax2.patches) == [4, 4]
        plt.close('all')

    def test_nonfinite_dropped_and_empty_safe(self):
        fig = accuracy_vs_resolution_figure(
            np.array([np.inf, np.nan]), np.array([1, 0]))
        assert len(fig.axes[0].lines) == 0       # nothing left to plot
        plt.close('all')


class TestStormTrackCorrectness:

    def test_correct_and_wrong_split_into_collections(self):
        lon = np.array([-80.0, -75.0, -70.0, -65.0])
        lat = np.array([10.0, 12.0, 14.0, 16.0])
        correct = np.array([True, True, False, True])
        fig = storm_track_correctness_figure(
            lon, lat, correct, domain=DOMAIN, basemap=False,
            title='train B track')
        ax = fig.axes[0]
        assert len(ax.lines) == 1                       # the grey trail
        ok, bad = ax.collections[:2]
        assert ok.get_offsets().shape[0] == 3           # green dots
        assert bad.get_offsets().shape[0] == 1          # red X
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['correct', 'wrong']
        assert ax.get_title() == 'train B track'
        assert tuple(ax.get_xlim()) == (-100, -30)      # domain extent
        plt.close('all')

    def test_all_correct_has_no_wrong_collection(self):
        fig = storm_track_correctness_figure(
            np.array([-60.0, -59.0]), np.array([12.0, 13.0]),
            np.array([1, 1]), basemap=False)
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['correct']
        plt.close('all')


class TestSequenceAndGif:

    def test_sequence_and_gif(self, tmp_path):
        samples = [dict(station_lon=np.array([-60.0]),
                        station_lat=np.array([12.0]),
                        storm_lon=-58.0 + i, storm_lat=13.0,
                        true_class=i % N_CLS, pred_class=(i + 1) % N_CLS,
                        title=f'frame {i}')
                   for i in range(3)]
        figs = storm_sequence_figures(samples, N_CLS, domain=DOMAIN,
                                      basemap=False)
        assert len(figs) == 3
        out = save_gif(figs, tmp_path / 's.gif', duration_ms=100)
        assert out.exists() and out.stat().st_size > 0
        plt.close('all')
