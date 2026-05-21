import jax
import jax.numpy as jnp
from typing import Callable


def check_environment() -> None:
    """Check JAX environment and print details for GPUs, backend, JAX version, and available memory.

    Example
    -------
    >>> check_environment()
    JAX version: 0.4.35
    Backend: gpu
    GPUs: 1 available
      [0] NVIDIA A100-SXM4-40GB - 40.0 GB
    """
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    try:
        gpus = jax.devices("gpu")
        print(f"GPUs: {len(gpus)} available")
        for i, gpu in enumerate(gpus):
            try:
                mem_stats = gpu.memory_stats()
                mem_gb = mem_stats.get("bytes_limit", 0) / 1e9
            except (AttributeError, KeyError):
                mem_gb = 0.0
            print(f"  [{i}] {gpu.device_kind} - {mem_gb:.1f} GB")
    except RuntimeError:
        print("GPUs: none, using CPU")


def create_rng(seed: int = 42) -> jax.Array:
    """Create a JAX PRNGKey from a seed for reproducible randomness.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    jax.Array
        A JAX PRNGKey array.

    Example
    -------
    >>> create_rng(0)
    Array([0, 0], dtype=uint32)
    """
    return jax.random.PRNGKey(seed)


def create_rng_dict(seed: int = 42, keys: list[str] | None = None) -> dict[str, jax.Array]:
    """Create a dictionary of PRNGKeys from a single seed.

    Splits a root key into named subkeys for use with Flax model
    init and apply calls.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    keys : list[str], optional
        Key names for the dictionary. Defaults to ["params", "dropout"].

    Returns
    -------
    dict[str, jax.Array]
        Dictionary mapping key names to their respective PRNGKeys.

    Example
    -------
    >>> rngs = create_rng_dict(0, keys=["params", "dropout"])
    >>> list(rngs.keys())
    ['params', 'dropout']
    >>> rngs["params"].shape
    (2,)
    """
    if keys is None:
        keys = ["params", "dropout"]
    root = jax.random.PRNGKey(seed)
    splits = jax.random.split(root, len(keys))
    return {k: v for k, v in zip(keys, splits)}


def split_rng(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split a PRNGKey into a new root key and a subkey.

    This is the standard JAX pattern to avoid accidental key reuse.

    Parameters
    ----------
    rng : jax.Array
        Current PRNGKey to split.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        A pair of (new_rng, subkey).

    Example
    -------
    >>> rng = create_rng(0)
    >>> rng, key = split_rng(rng)
    >>> jax.random.normal(key)
    Array(-1.2515389, dtype=float32)
    """
    return jax.random.split(rng)


def show_jaxpr(fn: Callable, *sample_inputs, static_argnums: int | tuple[int, ...] | None = None) -> None:
    """Print the jaxpr (JAX expression) representation of a function.

    Parameters
    ----------
    fn : callable
        Function to trace.
    *sample_inputs
        Example inputs that define the shapes and dtypes for tracing.
    static_argnums : int or tuple[int, ...], optional
        Positional arguments to treat as static (not traced).

    Example
    -------
    >>> show_jaxpr(lambda x: x ** 2 + 1, jnp.array(3.0))
    { lambda ; a:f32[]. let
        b:f32[] = integer_pow[y=2] a
        c:f32[] = add b 1.0
      in (c,) }
    """
    kwargs = {}
    if static_argnums is not None:
        kwargs["static_argnums"] = static_argnums
    print(jax.make_jaxpr(fn, **kwargs)(*sample_inputs))


def grad_fn(fn: Callable, argnums: int | tuple[int, ...] = 0, has_aux: bool = False) -> Callable:
    """Return a function that computes both value and gradients.

    Parameters
    ----------
    fn : callable
        Function to differentiate.
    argnums : int or tuple[int]
        Which positional argument(s) to differentiate with respect to.
    has_aux : bool
        If True, ``fn`` returns a pair ``(value, aux)`` and the gradient
        is computed with respect to ``value`` only.

    Returns
    -------
    callable
        A function that returns ``(value, grads)`` or ``((value, aux), grads)``.

    Example
    -------
    >>> f = lambda x: x ** 3
    >>> val_and_grad = grad_fn(f)
    >>> val_and_grad(2.0)
    (Array(8., dtype=float32), Array(12., dtype=float32))
    """
    return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)