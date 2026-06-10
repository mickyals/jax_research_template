import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from unittest.mock import patch

from utils.plotting.curves import plot_losses, plot_grouped_bars


class TestPlotLosses:

    @pytest.fixture
    def short_losses(self):
        return {
            "train": [1.0, 0.8, 0.6, 0.4, 0.2],
            "test":  [1.1, 0.9, 0.7, 0.5, 0.3],
        }

    @pytest.fixture
    def long_losses(self):
        rng = np.random.default_rng(0)
        n = 200
        return {
            "train": list(np.exp(-np.linspace(0, 3, n)) + rng.normal(0, 0.01, n)),
            "test":  list(np.exp(-np.linspace(0, 3, n)) + rng.normal(0, 0.02, n)),
        }

    @patch("matplotlib.pyplot.show")
    def test_runs_with_train_and_test(self, mock_show, long_losses):
        plot_losses(long_losses)

    @patch("matplotlib.pyplot.show")
    def test_runs_train_only(self, mock_show, long_losses):
        plot_losses({"train": long_losses["train"]})

    @patch("matplotlib.pyplot.show")
    def test_runs_short_sequence(self, mock_show, short_losses):
        # sequence shorter than default window -- smoothed panel skipped
        plot_losses(short_losses)

    @patch("matplotlib.pyplot.show")
    def test_custom_title(self, mock_show, long_losses):
        plot_losses(long_losses, title="Sphere INR")

    @patch("matplotlib.pyplot.show")
    def test_custom_window(self, mock_show, long_losses):
        plot_losses(long_losses, window=50)

    @patch("matplotlib.pyplot.show")
    def test_window_larger_than_sequence(self, mock_show, short_losses):
        # window > len(losses) -- smoothed curve simply not drawn, no error
        plot_losses(short_losses, window=100)

    @patch("matplotlib.pyplot.show")
    def test_empty_test_list(self, mock_show, long_losses):
        # test key present but empty -- treated as train-only
        plot_losses({"train": long_losses["train"], "test": []})

    @patch("matplotlib.pyplot.show")
    def test_returns_figure(self, mock_show, long_losses):
        result = plot_losses(long_losses)
        assert isinstance(result, Figure)
        plt.close(result)

    @patch("matplotlib.pyplot.show")
    def test_produces_two_axes(self, mock_show, long_losses):
        plot_losses(long_losses)
        fig = plt.gcf()
        assert len(fig.axes) == 2
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_single_step(self, mock_show):
        # degenerate case -- one step only
        plot_losses({"train": [1.0]})


class TestPlotGroupedBars:

    @pytest.fixture
    def metrics(self):
        return {
            "Precision": [0.8, 0.6, 0.9],
            "Recall":    [0.7, 0.5, 0.95],
            "F1":        [0.75, 0.55, 0.92],
        }

    @pytest.fixture
    def categories(self):
        return ["A", "B", "C"]

    @patch("matplotlib.pyplot.show")
    def test_runs(self, mock_show, metrics, categories):
        plot_grouped_bars(metrics, categories)

    @patch("matplotlib.pyplot.show")
    def test_returns_figure(self, mock_show, metrics, categories):
        result = plot_grouped_bars(metrics, categories)
        assert isinstance(result, Figure)
        plt.close(result)

    @patch("matplotlib.pyplot.show")
    def test_one_bar_per_series_per_category(self, mock_show, metrics, categories):
        fig = plot_grouped_bars(metrics, categories)
        ax = fig.axes[0]
        n_bars = sum(1 for patch_ in ax.patches)
        assert n_bars == len(metrics) * len(categories)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_xtick_labels_match_categories(self, mock_show, metrics, categories):
        fig = plot_grouped_bars(metrics, categories)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == categories
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_ylim_applied(self, mock_show, metrics, categories):
        fig = plot_grouped_bars(metrics, categories, ylim=(0, 1.05))
        ax = fig.axes[0]
        assert ax.get_ylim() == (0, 1.05)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_custom_colors(self, mock_show, metrics, categories):
        import matplotlib.colors as mcolors
        colors = ["steelblue", "darkorange", "seagreen"]
        fig = plot_grouped_bars(metrics, categories, colors=colors)
        ax = fig.axes[0]
        n_cat = len(categories)
        for i, color in enumerate(colors):
            patch_ = ax.patches[i * n_cat]
            expected = mcolors.to_rgba(color, alpha=0.85)
            assert patch_.get_facecolor() == pytest.approx(expected)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_legend_labels_match_series(self, mock_show, metrics, categories):
        fig = plot_grouped_bars(metrics, categories)
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend_labels == list(metrics.keys())
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_single_series(self, mock_show, categories):
        fig = plot_grouped_bars({"Accuracy": [0.9, 0.8, 0.7]}, categories)
        ax = fig.axes[0]
        assert len(ax.patches) == len(categories)
        plt.close(fig)