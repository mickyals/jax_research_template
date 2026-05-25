import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch

from utils.plotting.plot1d import plot_losses


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
    def test_returns_none(self, mock_show, long_losses):
        result = plot_losses(long_losses)
        assert result is None

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