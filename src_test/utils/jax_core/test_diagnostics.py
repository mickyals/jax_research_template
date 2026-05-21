import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
from unittest.mock import patch

from utils.jax_core.diagnostics import (
    get_grads,
    get_layer_gradients,
    get_layer_activations,
    count_inactive_neurons,
    vis_act_fn,
    visualize_weight_distribution,
    visualize_gradients,
    visualize_activations,
    plot_loss_landscape,
    _plot_dists,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Two-layer dense network for testing."""
    hidden: int = 8
    out: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out)(x)
        return x


@pytest.fixture
def model_and_params():
    model = TinyModel()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
    return model, params


@pytest.fixture
def dummy_batch():
    return jax.random.normal(jax.random.PRNGKey(1), (16, 5))


@pytest.fixture
def make_loss_fn(model_and_params, dummy_batch):
    model, _ = model_and_params
    batch = dummy_batch

    def loss_fn(p):
        logits = model.apply(p, batch)
        return jnp.mean(logits ** 2)

    return loss_fn


# ---------------------------------------------------------------------------
# get_grads
# ---------------------------------------------------------------------------

class TestGetGrads:

    def test_relu_gradient(self):
        x = jnp.array([-2.0, -1.0, 1.0, 2.0])
        grads = get_grads(jax.nn.relu, x)
        expected = jnp.array([0.0, 0.0, 1.0, 1.0])
        assert jnp.allclose(grads, expected)

    def test_identity_gradient(self):
        x = jnp.linspace(-3, 3, 7)
        grads = get_grads(lambda x: x, x)
        assert jnp.allclose(grads, jnp.ones_like(x))

    def test_quadratic_gradient(self):
        x = jnp.array([3.0])
        grads = get_grads(lambda x: x ** 2, x)
        assert jnp.allclose(grads, jnp.array([6.0]))

    def test_output_shape_matches_input(self):
        x = jnp.linspace(-5, 5, 50)
        grads = get_grads(jax.nn.sigmoid, x)
        assert grads.shape == x.shape


# ---------------------------------------------------------------------------
# get_layer_gradients
# ---------------------------------------------------------------------------

class TestGetLayerGradients:

    def test_returns_list(self, model_and_params, make_loss_fn):
        _, params = model_and_params
        result = get_layer_gradients(params, make_loss_fn)
        assert isinstance(result, list)

    def test_excludes_bias_by_default(self, model_and_params, make_loss_fn):
        _, params = model_and_params
        without_bias = get_layer_gradients(params, make_loss_fn, include_bias=False)
        with_bias = get_layer_gradients(params, make_loss_fn, include_bias=True)
        assert len(with_bias) > len(without_bias)

    def test_weight_count_matches_layers(self, model_and_params, make_loss_fn):
        _, params = model_and_params
        # TinyModel has 2 Dense layers, so 2 weight matrices
        result = get_layer_gradients(params, make_loss_fn, include_bias=False)
        assert len(result) == 2

    def test_all_arrays_are_1d(self, model_and_params, make_loss_fn):
        _, params = model_and_params
        result = get_layer_gradients(params, make_loss_fn)
        for g in result:
            assert g.ndim == 1


# ---------------------------------------------------------------------------
# get_layer_activations
# ---------------------------------------------------------------------------

class TestGetLayerActivations:

    def test_returns_dict(self, model_and_params, dummy_batch):
        model, params = model_and_params
        result = get_layer_activations(model, params, dummy_batch)
        assert isinstance(result, dict)

    def test_non_empty(self, model_and_params, dummy_batch):
        model, params = model_and_params
        result = get_layer_activations(model, params, dummy_batch)
        leaves = jax.tree_util.tree_leaves(result)
        assert len(leaves) > 0

    def test_batch_dimension_preserved(self, model_and_params, dummy_batch):
        model, params = model_and_params
        result = get_layer_activations(model, params, dummy_batch)
        leaves = jax.tree_util.tree_leaves(result)
        for leaf in leaves:
            if leaf.ndim >= 2:
                assert leaf.shape[0] == dummy_batch.shape[0]


# ---------------------------------------------------------------------------
# count_inactive_neurons
# ---------------------------------------------------------------------------

class TestCountInactiveNeurons:

    def test_all_zero_magnitude(self):
        activations = {"layer0": (jnp.zeros((16, 10)),)}
        result = count_inactive_neurons(activations, threshold=1e-6, mode="magnitude")
        assert result["layer0"]["inactive"] == 10
        assert result["layer0"]["percent"] == 100.0

    def test_all_active_magnitude(self):
        activations = {"layer0": (jnp.ones((16, 10)),)}
        result = count_inactive_neurons(activations, threshold=1e-6, mode="magnitude")
        assert result["layer0"]["inactive"] == 0
        assert result["layer0"]["percent"] == 0.0

    def test_constant_nonzero_variance_mode(self):
        # All neurons output 0.5 for every sample: zero variance but nonzero magnitude
        activations = {"layer0": (0.5 * jnp.ones((16, 10)),)}
        result_mag = count_inactive_neurons(activations, threshold=1e-6, mode="magnitude")
        result_var = count_inactive_neurons(activations, threshold=1e-6, mode="variance")
        assert result_mag["layer0"]["inactive"] == 0
        assert result_var["layer0"]["inactive"] == 10

    def test_mixed_neurons(self):
        # 3 dead neurons (zero), 7 active neurons
        data = jnp.concatenate([
            jnp.zeros((16, 3)),
            jnp.ones((16, 7)),
        ], axis=1)
        activations = {"layer0": (data,)}
        result = count_inactive_neurons(activations, threshold=1e-6, mode="magnitude")
        assert result["layer0"]["inactive"] == 3
        assert result["layer0"]["total"] == 10

    def test_both_mode_stricter(self):
        # Constant nonzero: magnitude says active, variance says inactive
        activations = {"layer0": (0.5 * jnp.ones((16, 10)),)}
        result_both = count_inactive_neurons(activations, threshold=1e-6, mode="both")
        result_mag = count_inactive_neurons(activations, threshold=1e-6, mode="magnitude")
        result_var = count_inactive_neurons(activations, threshold=1e-6, mode="variance")
        # "both" requires BOTH conditions, so fewer or equal inactive
        assert result_both["layer0"]["inactive"] <= result_var["layer0"]["inactive"]
        assert result_both["layer0"]["inactive"] <= result_mag["layer0"]["inactive"]

    def test_aggregate_spatial_conv(self):
        # (batch=8, h=4, w=4, channels=3), all zeros
        activations = {"conv": (jnp.zeros((8, 4, 4, 3)),)}
        result = count_inactive_neurons(
            activations, threshold=1e-6, mode="magnitude",
            aggregate_spatial=True, channel_axis=-1,
        )
        assert result["conv"]["total"] == 3
        assert result["conv"]["inactive"] == 3

    def test_aggregate_spatial_preserves_active(self):
        # (batch=8, h=4, w=4, channels=3), channel 0 is active
        data = jnp.zeros((8, 4, 4, 3))
        data = data.at[:, :, :, 0].set(1.0)
        activations = {"conv": (data,)}
        result = count_inactive_neurons(
            activations, threshold=1e-6, mode="magnitude",
            aggregate_spatial=True, channel_axis=-1,
        )
        assert result["conv"]["inactive"] == 2

    def test_invalid_mode_raises(self):
        activations = {"layer0": (jnp.ones((4, 4)),)}
        with pytest.raises(ValueError):
            count_inactive_neurons(activations, mode="invalid")

    def test_invalid_spatial_reduction_raises(self):
        activations = {"layer0": (jnp.ones((4, 4, 4, 2)),)}
        with pytest.raises(ValueError):
            count_inactive_neurons(
                activations, aggregate_spatial=True, spatial_reduction="invalid"
            )

    def test_skips_1d_arrays(self):
        activations = {"bias": (jnp.zeros((10,)),)}
        result = count_inactive_neurons(activations)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Plotting functions (smoke tests)
# ---------------------------------------------------------------------------

class TestPlotSmoke:

    @patch("matplotlib.pyplot.show")
    def test_vis_act_fn_runs(self, mock_show):
        fig, ax = plt.subplots()
        x = jnp.linspace(-5, 5, 50)
        vis_act_fn(jax.nn.relu, ax, x)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_visualize_weight_distribution_runs(self, mock_show, model_and_params):
        _, params = model_and_params
        visualize_weight_distribution(params)

    @patch("matplotlib.pyplot.show")
    def test_visualize_gradients_runs(self, mock_show, model_and_params, make_loss_fn):
        _, params = model_and_params
        visualize_gradients(params, make_loss_fn)

    @patch("matplotlib.pyplot.show")
    def test_visualize_gradients_print_variance(self, mock_show, model_and_params,
                                                 make_loss_fn, capsys):
        _, params = model_and_params
        visualize_gradients(params, make_loss_fn, print_variance=True)
        captured = capsys.readouterr()
        assert "Variance" in captured.out

    @patch("matplotlib.pyplot.show")
    def test_visualize_activations_runs(self, mock_show, model_and_params, dummy_batch):
        model, params = model_and_params
        visualize_activations(model, params, dummy_batch)

    @patch("matplotlib.pyplot.show")
    def test_visualize_activations_print_variance(self, mock_show, model_and_params,
                                                   dummy_batch, capsys):
        model, params = model_and_params
        visualize_activations(model, params, dummy_batch, print_variance=True)
        captured = capsys.readouterr()
        assert "Variance" in captured.out

    @patch("matplotlib.pyplot.show")
    def test_plot_dists_runs(self, mock_show):
        val_dict = {"Layer 0": jnp.ones(100), "Layer 1": jnp.zeros(100)}
        fig = _plot_dists(val_dict)
        assert fig is not None
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_plot_dists_single_layer(self, mock_show):
        val_dict = {"Layer 0": jnp.linspace(-1, 1, 100)}
        fig = _plot_dists(val_dict)
        assert fig is not None
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_plot_loss_landscape_runs(self, mock_show, model_and_params, make_loss_fn):
        _, params = model_and_params
        plot_loss_landscape(params, make_loss_fn, grid_size=5)

    @patch("matplotlib.pyplot.show")
    def test_plot_loss_landscape_3d(self, mock_show, model_and_params, make_loss_fn):
        _, params = model_and_params
        plot_loss_landscape(params, make_loss_fn, grid_size=5, plot_3d=True)