"""
datasets/splitting.py

Generic group-split helpers — turn a per-row group id array plus a
group-level split assignment into per-row boolean masks, and produce
seeded fraction-based group assignments stratified by an optional
per-group label.

Mechanism only: which groups go where (policy) is decided by callers
(e.g. an experiment's data/splits.py resolver). Pure numpy, no pandas.
"""

from __future__ import annotations

import numpy as np


def validate_disjoint_groups(groups: dict[str, list]) -> None:
    """Raise if any value appears in more than one named group.

    Parameters
    ----------
    groups : dict[str, list]
        Maps a split name (e.g. 'train', 'val', 'test') to the list of
        column values assigned to that split.

    Raises
    ------
    ValueError
        If a value appears in two different splits.
    """
    seen: dict = {}
    for name, values in groups.items():
        for v in values:
            if v in seen:
                raise ValueError(
                    f"Value {v!r} appears in both '{seen[v]}' and '{name}' "
                    "— splits must be disjoint."
                )
            seen[v] = name


def group_mask(row_groups: np.ndarray, groups) -> np.ndarray:
    """Boolean row mask, True where row_groups[i] is in groups.

    Parameters
    ----------
    row_groups : np.ndarray, shape (n_rows,)
        Group identifier (e.g. SID) for each row.
    groups : array-like
        Group identifiers selected for this split.

    Returns
    -------
    np.ndarray of bool, shape (n_rows,)
    """
    return np.isin(row_groups, groups)


def assign_groups_by_fraction(
    groups: np.ndarray,
    fraction: float,
    seed: int,
    stratify_by: np.ndarray | None = None,
) -> np.ndarray:
    """Seeded fraction-based binary assignment of unique groups.

    Without `stratify_by`, shuffles `groups` with a seeded RNG and selects
    `int(len(groups) * fraction)` of them.

    With `stratify_by`, assignment is performed independently within each
    stratum so the selected fraction holds per-stratum.  Floor rule: every
    non-empty stratum contributes at least one group to the selected set,
    even when `int(stratum_size * fraction)` would otherwise be 0 (e.g. a
    4-group stratum at fraction=0.2 would round down to 0 without this
    rule). A stratum of size 1 therefore goes entirely to the selected set
    when fraction > 0.

    Parameters
    ----------
    groups : np.ndarray, shape (n_groups,)
        Unique group identifiers (e.g. SIDs). Must not contain duplicates.
    fraction : float
        Target fraction of groups to select, in (0, 1).
    seed : int
        RNG seed — identical inputs always produce the identical assignment.
    stratify_by : np.ndarray, shape (n_groups,), optional
        Per-group stratum label, aligned with `groups`.

    Returns
    -------
    np.ndarray of bool, shape (n_groups,)
        True = selected (e.g. "val"), False = remainder (e.g. "train").
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    groups = np.asarray(groups)
    if len(np.unique(groups)) != len(groups):
        raise ValueError("groups must not contain duplicates")

    rng = np.random.default_rng(seed)
    selected = np.zeros(len(groups), dtype=bool)

    if stratify_by is None:
        n_select = int(len(groups) * fraction)
        idx = rng.permutation(len(groups))[:n_select]
        selected[idx] = True
        return selected

    stratify_by = np.asarray(stratify_by)
    for stratum in np.unique(stratify_by):
        idx = np.flatnonzero(stratify_by == stratum)
        n_select = int(len(idx) * fraction)
        if n_select == 0:
            n_select = 1   # floor rule: every non-empty stratum contributes
        chosen = rng.permutation(idx)[:n_select]
        selected[chosen] = True

    return selected
