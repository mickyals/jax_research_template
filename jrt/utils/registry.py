"""
utils/registry.py

A small string-addressable registry, shared by the factory-style registries
across jrt (losses, optimizers, schedulers, ...). Each registered entry is a
**factory**: a function taking config kwargs and returning the actual object
(a loss callable, an optax transform, a schedule, ...).

Centralises the boilerplate that was previously copy-pasted per registry:
case-insensitive names, duplicate-registration guard, kwarg filtering with a
warning for unknown keys, and listing.

Usage
-----
    from utils.registry import Registry

    OPTIMIZERS = Registry("Optimizer")

    @OPTIMIZERS.register("adam", description="Adam (Kingma & Ba 2015)")
    def _adam(learning_rate=1e-3, **kw):
        return optax.adam(learning_rate, **kw)

    opt = OPTIMIZERS.get("adam", learning_rate=3e-4)   # case-insensitive
    OPTIMIZERS.describe()                               # {name: description}
    "ADAM" in OPTIMIZERS                                # membership

A registry that needs a different contract (e.g. the datasets registry, whose
factories take a single positional config dict) can stay bespoke — this class
targets the common ``factory(**kwargs)`` shape.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Callable


class Registry:
    """Name -> factory registry with kwarg filtering.

    Parameters
    ----------
    kind : str
        Human-readable label used in error/warning messages (e.g. "Loss").
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, dict] = {}

    # -- registration -------------------------------------------------------

    def register(self, name: str, description: str = "") -> Callable:
        """Decorator registering a factory under ``name`` (case-insensitive).

        Raises
        ------
        ValueError
            If ``name`` is already registered.
        """
        key = name.upper()

        def decorator(fn: Callable) -> Callable:
            if key in self._entries:
                raise ValueError(f"{self.kind} '{name}' is already registered.")
            self._entries[key] = {"fn": fn, "description": description}
            return fn

        return decorator

    # -- lookup -------------------------------------------------------------

    def get(self, name: str, **kwargs):
        """Instantiate a registered entry by name, forwarding kwargs.

        Unknown kwargs (not in the factory signature) trigger a UserWarning
        and are dropped rather than raising a TypeError.

        Raises
        ------
        ValueError
            If ``name`` is not registered.
        """
        key = name.upper()
        if key not in self._entries:
            available = ", ".join(sorted(self._entries)) or "none"
            raise ValueError(
                f"{self.kind} '{name}' is not registered. Available: {available}"
            )

        fn = self._entries[key]["fn"]
        if kwargs:
            params = inspect.signature(fn).parameters.values()
            # A factory with **kwargs accepts anything — no filtering.
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                valid = {
                    p.name for p in params
                    if p.kind != inspect.Parameter.VAR_POSITIONAL
                }
                unknown = set(kwargs) - valid
                if unknown:
                    warnings.warn(
                        f"{self.kind} '{name}': unknown kwargs {unknown} will be "
                        f"ignored. Valid kwargs: {valid or 'none'}.",
                        UserWarning,
                        stacklevel=2,
                    )
                kwargs = {k: v for k, v in kwargs.items() if k in valid}
        return fn(**kwargs)

    # -- introspection ------------------------------------------------------

    def names(self) -> list[str]:
        """Sorted registered names."""
        return sorted(self._entries)

    def describe(self) -> dict[str, str]:
        """Map of registered name -> description."""
        return {name: e["description"] for name, e in self._entries.items()}

    def __contains__(self, name: str) -> bool:
        return name.upper() in self._entries

    def __getitem__(self, name: str) -> Callable:
        """The raw factory registered under ``name``."""
        return self._entries[name.upper()]["fn"]

    def __len__(self) -> int:
        return len(self._entries)
