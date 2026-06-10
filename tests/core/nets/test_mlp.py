# tests/test_mlp.py
import pytest
import jax
import jax.numpy as jnp
import optax

from core.nets.mlp import (
    MLP,
    SIREN,
    FINERNet,
    GaussianNet,
    GaussianFINERNet,
    WireNet,
    WireFINERNet,
    WireComplexNet,
    HOSCNet,
    HOSCFINERNet,
    SincNet,
    LatLonEmbeddingWrapper,
    CombinedEmbedding,
    get_mlp,
    list_mlps,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH    = 16
IN_DIM   = 4
OUT_DIM  = 2
HIDDEN   = 32
N_LAYERS = 3
KEY      = jax.random.PRNGKey(0)




@pytest.fixture
def x():
    return jax.random.normal(KEY, (BATCH, IN_DIM))


@pytest.fixture
def key():
    return KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ALL_NETS = [
    (MLP,              {}),
    (SIREN,            {}),
    (FINERNet,         {}),
    (GaussianNet,      {}),
    (GaussianFINERNet, {}),
    (WireNet,          {}),
    (WireFINERNet,     {}),
    (WireComplexNet,   {}),
    (HOSCNet,          {}),
    (HOSCFINERNet,     {}),
    (SincNet,          {}),
]


def init_and_forward(model, x, key):
    variables = model.init(key, x)
    out = model.apply(variables, x, train=False)
    return variables, out


def check_forward(model, x, key):
    variables, out = init_and_forward(model, x, key)
    assert out.shape == (BATCH, OUT_DIM), (
        f"Expected ({BATCH}, {OUT_DIM}), got {out.shape}"
    )
    assert jnp.all(jnp.isfinite(out)), "Output contains non-finite values"
    assert out.dtype == jnp.float32, f"Expected float32 output, got {out.dtype}"
    return variables, out


def check_backward(model, x, key):
    variables, _ = init_and_forward(model, x, key)

    def loss_fn(params):
        out = model.apply({'params': params}, x, train=False)
        return jnp.mean(out ** 2)

    grads = jax.grad(loss_fn)(variables['params'])
    leaves = jax.tree_util.tree_leaves(grads)
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), "Gradient contains non-finite values"
        assert jnp.any(leaf != 0.0), "Gradient is all zeros"
    return grads


def check_output_shape_varies_with_input(model, key):
    for batch in [1, 8, 64]:
        x = jax.random.normal(key, (batch, IN_DIM))
        variables = model.init(key, x)
        out = model.apply(variables, x)
        assert out.shape == (batch, OUT_DIM)


def check_n_layers_one(model_cls, key, x, **kwargs):
    model = model_cls(out_features=OUT_DIM, hidden_features=HIDDEN,
                      n_layers=1, **kwargs)
    variables, out = init_and_forward(model, x, key)
    assert out.shape == (BATCH, OUT_DIM)
    assert jnp.all(jnp.isfinite(out))


def check_invalid_n_layers(model_cls, **kwargs):
    model = model_cls(out_features=OUT_DIM, hidden_features=HIDDEN,
                      n_layers=0, **kwargs)
    x = jax.random.normal(KEY, (BATCH, IN_DIM))
    with pytest.raises(ValueError, match="n_layers must be >= 1"):
        model.init(KEY, x)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    @pytest.fixture
    def model(self):
        return MLP(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_batch_size_invariance(self, key):
        model = MLP(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)
        check_output_shape_varies_with_input(model, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(MLP, key, x)

    def test_invalid_n_layers(self):
        check_invalid_n_layers(MLP)

    def test_activation_variants(self, x, key):
        for act in ["relu", "tanh", "gelu", "silu", "sigmoid", "softplus"]:
            model = MLP(out_features=OUT_DIM, hidden_features=HIDDEN,
                        n_layers=N_LAYERS, activation=act)
            check_forward(model, x, key)

    def test_output_bias_configurable(self, x, key):
        model = MLP(out_features=OUT_DIM, hidden_features=HIDDEN,
                    n_layers=N_LAYERS, output_bias_initializer="normal",
                    output_bias_initializer_kwargs={"std": 0.1})
        variables, _ = init_and_forward(model, x, key)
        output_bias = variables['params']['output_layer']['bias']
        assert jnp.any(output_bias != 0.0)

    def test_hidden_bias_configurable(self, x, key):
        model = MLP(out_features=OUT_DIM, hidden_features=HIDDEN,
                    n_layers=N_LAYERS, bias_initializer="normal",
                    bias_initializer_kwargs={"std": 0.1})
        variables, _ = init_and_forward(model, x, key)
        first_bias = variables['params']['first_layer']['bias']
        assert jnp.any(first_bias != 0.0)

    def test_no_bias(self, x, key):
        model = MLP(out_features=OUT_DIM, hidden_features=HIDDEN,
                    n_layers=N_LAYERS, use_bias=False)
        check_forward(model, x, key)


# ---------------------------------------------------------------------------
# SIREN
# ---------------------------------------------------------------------------

class TestSIREN:
    @pytest.fixture
    def model(self):
        return SIREN(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(SIREN, key, x)

    def test_invalid_n_layers(self):
        check_invalid_n_layers(SIREN)

    def test_first_layer_init_bounds(self, x, key):
        model = SIREN(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)
        variables, _ = init_and_forward(model, x, key)
        first_kernel = variables['params']['first_layer']['kernel']
        bound = 1.0 / IN_DIM
        assert jnp.all(jnp.abs(first_kernel) <= bound + 1e-6), (
            "SIREN first layer kernel out of U(-1/fan_in, 1/fan_in) bounds"
        )

    def test_different_omegas(self, x, key):
        model = SIREN(out_features=OUT_DIM, hidden_features=HIDDEN,
                      n_layers=N_LAYERS, first_omega=30.0, hidden_omega=30.0)
        check_forward(model, x, key)

    def test_bias_is_zero(self, x, key):
        model = SIREN(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)
        variables, _ = init_and_forward(model, x, key)
        first_bias = variables['params']['first_layer']['bias']
        assert jnp.all(first_bias == 0.0), "SIREN first layer bias should be zero"

    def test_bias_override(self, x, key):
        model = SIREN(out_features=OUT_DIM, hidden_features=HIDDEN,
                      n_layers=N_LAYERS, bias_initializer="normal",
                      bias_initializer_kwargs={"std": 0.1})
        variables, out = init_and_forward(model, x, key)
        assert jnp.all(jnp.isfinite(out))
        first_bias = variables['params']['first_layer']['bias']
        assert jnp.any(first_bias != 0.0)


# ---------------------------------------------------------------------------
# FINERNet
# ---------------------------------------------------------------------------

class TestFINERNet:
    @pytest.fixture
    def model(self):
        return FINERNet(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(FINERNet, key, x)

    def test_invalid_n_layers(self):
        check_invalid_n_layers(FINERNet)

    def test_first_layer_init_bounds(self, x, key):
        model = FINERNet(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)
        variables, _ = init_and_forward(model, x, key)
        first_kernel = variables['params']['first_layer']['kernel']
        bound = 1.0 / IN_DIM
        assert jnp.all(jnp.abs(first_kernel) <= bound + 1e-6), (
            "FINER first layer kernel out of U(-1/fan_in, 1/fan_in) bounds"
        )

    def test_bias_k_range(self, x, key):
        k = 0.5
        model = FINERNet(out_features=OUT_DIM, hidden_features=HIDDEN,
                         n_layers=N_LAYERS, bias_k=k)
        variables, _ = init_and_forward(model, x, key)
        first_bias = variables['params']['first_layer']['bias']
        assert jnp.all(jnp.abs(first_bias) <= k + 1e-6), (
            f"FINER bias out of U(-{k}, {k}) bounds"
        )

    def test_bias_override(self, x, key):
        model = FINERNet(out_features=OUT_DIM, hidden_features=HIDDEN,
                         n_layers=N_LAYERS, bias_initializer="normal",
                         bias_initializer_kwargs={"std": 0.01})
        variables, out = init_and_forward(model, x, key)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Gaussian variants
# ---------------------------------------------------------------------------

class TestGaussianNet:
    @pytest.fixture
    def model(self):
        return GaussianNet(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(GaussianNet, key, x)


class TestGaussianFINERNet:
    @pytest.fixture
    def model(self):
        return GaussianFINERNet(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(GaussianFINERNet, key, x)


# ---------------------------------------------------------------------------
# WIRE real variants
# ---------------------------------------------------------------------------

class TestWireNet:
    @pytest.fixture
    def model(self):
        return WireNet(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(WireNet, key, x)


class TestWireFINERNet:
    @pytest.fixture
    def model(self):
        return WireFINERNet(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(WireFINERNet, key, x)


# ---------------------------------------------------------------------------
# WIRE complex
# ---------------------------------------------------------------------------

class TestWireComplexNet:
    @pytest.fixture
    def model(self):
        return WireComplexNet(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_output_is_real(self, model, x, key):
        variables, out = init_and_forward(model, x, key)
        assert out.dtype == jnp.float32, "Complex WIRE output should be real float32"
        assert not jnp.iscomplexobj(out), "Output should not be complex"

    def test_params_are_complex(self, model, x, key):
        variables, _ = init_and_forward(model, x, key)
        leaves = jax.tree_util.tree_leaves(variables['params'])
        assert any(jnp.issubdtype(leaf.dtype, jnp.complexfloating)
                   for leaf in leaves), "Expected at least one complex parameter"

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(WireComplexNet, key, x)


# ---------------------------------------------------------------------------
# HOSC variants
# ---------------------------------------------------------------------------

class TestHOSCNet:
    @pytest.fixture
    def model(self):
        return HOSCNet(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(HOSCNet, key, x)


class TestHOSCFINERNet:
    @pytest.fixture
    def model(self):
        return HOSCFINERNet(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(HOSCFINERNet, key, x)


# ---------------------------------------------------------------------------
# SincNet
# ---------------------------------------------------------------------------

class TestSincNet:
    @pytest.fixture
    def model(self):
        return SincNet(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)

    def test_forward(self, model, x, key):
        check_forward(model, x, key)

    def test_backward(self, model, x, key):
        check_backward(model, x, key)

    def test_n_layers_one(self, x, key):
        check_n_layers_one(SincNet, key, x)

# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class TestDropout:
    def test_zero_dropout_matches_no_dropout(self, x, key):
        m1 = MLP(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS)
        m2 = MLP(out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS,
                 dropout_rate=0.0)
        v1 = m1.init(key, x)
        # use same params for both
        out1 = m1.apply(v1, x, train=False)
        out2 = m2.apply(v1, x, train=False)
        assert jnp.allclose(out1, out2)

    def test_dropout_train_stochastic(self, x, key):
        model = MLP(out_features=OUT_DIM, hidden_features=HIDDEN,
                    n_layers=N_LAYERS, dropout_rate=0.5)
        variables = model.init(key, x)
        out1 = model.apply(variables, x, train=True,
                           rngs={'dropout': jax.random.PRNGKey(0)})
        out2 = model.apply(variables, x, train=True,
                           rngs={'dropout': jax.random.PRNGKey(1)})
        assert not jnp.allclose(out1, out2), (
            "Dropout with different keys should produce different outputs at train time"
        )

    def test_dropout_eval_deterministic(self, x, key):
        model = MLP(out_features=OUT_DIM, hidden_features=HIDDEN,
                    n_layers=N_LAYERS, dropout_rate=0.5)
        variables = model.init(key, x)
        out1 = model.apply(variables, x, train=False)
        out2 = model.apply(variables, x, train=False)
        assert jnp.allclose(out1, out2), (
            "Dropout should be deterministic at eval time"
        )

    def test_dropout_rates_per_layer(self, x, key):
        rates = [0.0, 0.1, 0.5]
        model = MLP(out_features=OUT_DIM, hidden_features=HIDDEN,
                    n_layers=N_LAYERS, dropout_rates=rates)
        variables = model.init(key, x)
        out = model.apply(variables, x, train=False)
        assert out.shape == (BATCH, OUT_DIM)
        assert jnp.all(jnp.isfinite(out))

    def test_dropout_rates_wrong_length_raises(self, x, key):
        model = MLP(out_features=OUT_DIM, hidden_features=HIDDEN,
                    n_layers=N_LAYERS, dropout_rates=[0.1, 0.2])
        with pytest.raises(ValueError, match="dropout_rates length"):
            model.init(key, x)

    def test_dropout_backward(self, x, key):
        model = MLP(out_features=OUT_DIM, hidden_features=HIDDEN,
                    n_layers=N_LAYERS, dropout_rate=0.1)
        variables = model.init(key, x)

        def loss_fn(params):
            return jnp.mean(
                model.apply({'params': params}, x, train=True,
                            rngs={'dropout': jax.random.PRNGKey(0)}) ** 2
            )

        grads = jax.grad(loss_fn)(variables['params'])
        leaves = jax.tree_util.tree_leaves(grads)
        for leaf in leaves:
            assert jnp.all(jnp.isfinite(leaf))

    def test_all_nets_accept_train_flag(self, x, key):
        for cls, _ in ALL_NETS:
            model = cls(out_features=OUT_DIM, hidden_features=HIDDEN,
                        n_layers=N_LAYERS)
            variables = model.init(key, x)
            out = model.apply(variables, x, train=False)
            assert out.shape == (BATCH, OUT_DIM)
            assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Embedding wrappers
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_latlon_wrapper_forward(self, key):
        from core.embeddings import get_embedding
        embed = LatLonEmbeddingWrapper(
            embedding=get_embedding("SPHERE_GRID", scale=8, r_min=0.01)
        )
        x = jax.random.normal(key, (BATCH, 2))
        variables = embed.init(key, x)
        out = embed.apply(variables, x)
        assert out.shape == (BATCH, 4 * 8)
        assert jnp.all(jnp.isfinite(out))

    def test_combined_spatial_and_time(self, key):
        from core.embeddings import get_embedding
        embed = CombinedEmbedding(
            spatial_dim=2,
            spatial_embedding=LatLonEmbeddingWrapper(
                embedding=get_embedding("SPHERE_GRID", scale=8, r_min=0.01)
            ),
            time_embedding=get_embedding(
                "GENERAL_POSITIONAL", input_dim=1, mapping_dim=16, scale=2.0
            ),
        )
        x = jax.random.normal(key, (BATCH, 3))  # lat, lon, time
        variables = embed.init(key, x)
        out = embed.apply(variables, x)
        # spatial: 4*8=32, time: 2*16=32
        assert out.shape == (BATCH, 64)
        assert jnp.all(jnp.isfinite(out))

    def test_combined_spatial_only_raw_time(self, key):
        from core.embeddings import get_embedding
        embed = CombinedEmbedding(
            spatial_dim=2,
            spatial_embedding=LatLonEmbeddingWrapper(
                embedding=get_embedding("SPHERE_GRID", scale=8, r_min=0.01)
            ),
        )
        x = jax.random.normal(key, (BATCH, 3))  # lat, lon, pressure raw
        variables = embed.init(key, x)
        out = embed.apply(variables, x)
        # spatial: 4*8=32, pressure: 1 raw
        assert out.shape == (BATCH, 33)
        assert jnp.all(jnp.isfinite(out))

    def test_combined_both_raw(self, key):
        embed = CombinedEmbedding(spatial_dim=2)
        x = jax.random.normal(key, (BATCH, 4))
        variables = embed.init(key, x)
        out = embed.apply(variables, x)
        assert out.shape == (BATCH, 4)
        assert jnp.all(jnp.isfinite(out))

    def test_combined_invalid_spatial_dim(self, key):
        embed = CombinedEmbedding(spatial_dim=5)
        x = jax.random.normal(key, (BATCH, 3))
        with pytest.raises(ValueError, match="spatial_dim"):
            embed.init(key, x)

    def test_net_with_gaussian_embedding_forward(self, key):
        from core.embeddings import get_embedding
        x = jax.random.normal(key, (BATCH, IN_DIM))
        net = SIREN(
            out_features=OUT_DIM,
            hidden_features=HIDDEN,
            n_layers=N_LAYERS,
            embedding=get_embedding(
                "GAUSSIAN_POSITIONAL", input_dim=IN_DIM, mapping_dim=32, scale=10.0
            ),
        )
        variables = net.init(key, x)
        out = net.apply(variables, x)
        assert out.shape == (BATCH, OUT_DIM)
        assert jnp.all(jnp.isfinite(out))

    def test_net_with_combined_embedding_forward(self, key):
        from core.embeddings import get_embedding
        x = jax.random.normal(key, (BATCH, 3))  # lat, lon, time
        net = SIREN(
            out_features=OUT_DIM,
            hidden_features=HIDDEN,
            n_layers=N_LAYERS,
            embedding=CombinedEmbedding(
                spatial_dim=2,
                spatial_embedding=LatLonEmbeddingWrapper(
                    embedding=get_embedding("SPHERE_GRID", scale=4, r_min=0.01)
                ),
                time_embedding=get_embedding(
                    "GENERAL_POSITIONAL", input_dim=1, mapping_dim=16, scale=2.0
                ),
            ),
        )
        variables = net.init(key, x)
        out = net.apply(variables, x)
        assert out.shape == (BATCH, OUT_DIM)
        assert jnp.all(jnp.isfinite(out))

    def test_net_with_combined_embedding_backward(self, key):
        from core.embeddings import get_embedding
        x = jax.random.normal(key, (BATCH, 3))
        net = SIREN(
            out_features=OUT_DIM,
            hidden_features=HIDDEN,
            n_layers=N_LAYERS,
            embedding=CombinedEmbedding(
                spatial_dim=2,
                spatial_embedding=LatLonEmbeddingWrapper(
                    embedding=get_embedding("SPHERE_GRID", scale=4, r_min=0.01)
                ),
            ),
        )
        variables = net.init(key, x)

        def loss_fn(params):
            return jnp.mean(net.apply({'params': params}, x) ** 2)

        grads = jax.grad(loss_fn)(variables['params'])
        leaves = jax.tree_util.tree_leaves(grads)
        for leaf in leaves:
            assert jnp.all(jnp.isfinite(leaf))

    def test_complex_net_with_embedding(self, key):
        from core.embeddings import get_embedding
        x = jax.random.normal(key, (BATCH, IN_DIM))
        net = WireComplexNet(
            out_features=OUT_DIM,
            hidden_features=HIDDEN,
            n_layers=N_LAYERS,
            embedding=get_embedding(
                "GAUSSIAN_POSITIONAL", input_dim=IN_DIM, mapping_dim=32, scale=10.0
            ),
        )
        variables = net.init(key, x)
        out = net.apply(variables, x)
        assert out.shape == (BATCH, OUT_DIM)
        assert out.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------

class TestCrossCutting:
    @pytest.mark.parametrize("model_cls,kwargs", ALL_NETS)
    def test_deterministic_given_same_key(self, model_cls, kwargs, x, key):
        model = model_cls(out_features=OUT_DIM, hidden_features=HIDDEN,
                          n_layers=N_LAYERS, **kwargs)
        variables = model.init(key, x)
        out1 = model.apply(variables, x)
        out2 = model.apply(variables, x)
        assert jnp.allclose(out1, out2), "Forward pass is not deterministic"

    @pytest.mark.parametrize("model_cls,kwargs", ALL_NETS)
    def test_different_keys_different_params(self, model_cls, kwargs, x):
        model = model_cls(out_features=OUT_DIM, hidden_features=HIDDEN,
                          n_layers=N_LAYERS, **kwargs)
        v1 = model.init(jax.random.PRNGKey(0), x)
        v2 = model.init(jax.random.PRNGKey(1), x)
        leaves1 = jax.tree_util.tree_leaves(v1['params'])
        leaves2 = jax.tree_util.tree_leaves(v2['params'])
        assert any(not jnp.allclose(l1, l2) for l1, l2 in zip(leaves1, leaves2)), (
            "Different keys produced identical parameters"
        )

    @pytest.mark.parametrize("model_cls,kwargs", ALL_NETS)
    def test_output_changes_after_grad_step(self, model_cls, kwargs, x, key):
        model = model_cls(out_features=OUT_DIM, hidden_features=HIDDEN,
                          n_layers=N_LAYERS, **kwargs)
        variables = model.init(key, x)

        def loss_fn(params):
            return jnp.mean(model.apply({'params': params}, x) ** 2)

        grads = jax.grad(loss_fn)(variables['params'])
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(variables['params'])
        updates, _ = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(variables['params'], updates)

        out_before = model.apply(variables, x)
        out_after  = model.apply({'params': new_params}, x)
        assert not jnp.allclose(out_before, out_after), (
            "Output unchanged after gradient step"
        )

    @pytest.mark.parametrize("model_cls", [cls for cls, _ in ALL_NETS])
    def test_invalid_n_layers_all(self, model_cls, x):
        check_invalid_n_layers(model_cls)

    @pytest.mark.parametrize("model_cls,kwargs", ALL_NETS)
    def test_bias_initializer_inherited(self, model_cls, kwargs, x, key):
        model = model_cls(
            out_features=OUT_DIM, hidden_features=HIDDEN, n_layers=N_LAYERS,
            output_bias_initializer="normal",
            output_bias_initializer_kwargs={"std": 0.1},
            **kwargs,
        )
        variables = model.init(key, x)
        out = model.apply(variables, x)
        assert jnp.all(jnp.isfinite(out))
        output_bias = variables['params']['output_layer']['bias']
        assert jnp.any(output_bias != 0.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_mlps_returns_all(self):
        nets = list_mlps()
        expected = {
            "MLP", "SIREN", "FINER", "GAUSSIAN", "GAUSSIAN_FINER",
            "WIRE", "WIRE_FINER", "WIRE_COMPLEX",
            "HOSC", "HOSC_FINER", "SINC",
        }
        assert expected == set(nets.keys())

    def test_get_mlp_instantiates(self, x, key):
        net = get_mlp("SIREN", out_features=OUT_DIM,
                      hidden_features=HIDDEN, n_layers=N_LAYERS)
        variables = net.init(key, x)
        out = net.apply(variables, x)
        assert out.shape == (BATCH, OUT_DIM)

    def test_get_mlp_unknown_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            get_mlp("NONEXISTENT", out_features=2, hidden_features=32, n_layers=2)

    def test_get_mlp_unknown_kwargs_warns(self):
        with pytest.warns(UserWarning, match="unknown kwargs"):
            get_mlp("SIREN", out_features=OUT_DIM, hidden_features=HIDDEN,
                    n_layers=N_LAYERS, nonexistent_param=99)

    def test_get_mlp_all_registered(self, x, key):
        for name in list_mlps():
            net = get_mlp(name, out_features=OUT_DIM,
                          hidden_features=HIDDEN, n_layers=N_LAYERS)
            variables = net.init(key, x)
            out = net.apply(variables, x)
            assert out.shape == (BATCH, OUT_DIM), f"{name} produced wrong output shape"
            assert jnp.all(jnp.isfinite(out)), f"{name} produced non-finite output"