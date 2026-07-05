"""
Tests for visualise/figures.py. Headless (Agg) and basemap=False throughout
so cartopy features are never rendered (Natural Earth downloads at draw).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.cyclone_jax.visualise.figures import (
    class_colour, save_gif,
    storm_panel_figure, storm_sequence_figures,
)

N_CLS = 6
DOMAIN = {'lat': [0, 30], 'lon': [-100, -30]}


def _panel(true_class=2, pred_class=4):
    return storm_panel_figure(
        station_lon=np.array([-60.0, -55.0]),
        station_lat=np.array([12.0, 14.0]),
        storm_lon=-58.0, storm_lat=13.0, true_class=true_class,
        pred_class=pred_class, n_classes=N_CLS,
        title='TEST 2024  true 2 vs pred 4', domain=DOMAIN, basemap=False)


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
        plt.close('all')

    def test_no_stations_still_draws(self):
        fig = storm_panel_figure(np.array([]), np.array([]), -58.0, 13.0,
                                 0, 0, N_CLS, basemap=False)
        assert len(fig.axes[0].collections) == 2        # ring + star only
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
