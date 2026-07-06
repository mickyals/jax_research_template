"""
experiments/cyclone_jax/data/interface.py

THE experiment's data entry point — notebooks and train call this, nothing
deeper. One call turns a data config into ready-to-iterate data:

    data = build_data(cfg['data'], seed=cfg['trainer']['seed'])
    data.loader.build(i)                # one named sample (inspection)
    for batch in data.streams['train']: ...   # device-ready batches

Composition (each piece independently tested):
    library.load_library -> resolve_input/resolve_target -> Loader
    -> split policy (cfg['split']) -> Sampler per split
    -> BatchStream = Sampler x Loader x batching.collate

Split strategies (cfg['split']['strategy']):
    year        disjoint year lists per split:
                  split: {strategy: year, years: {train: [...], val: [...],
                                                  test: [...]}}
                Fixes at MULTI-driver timestamps (>1 qualifying storm =
                ambiguous supervision) are excluded by default; set
                exclude_multistorm: false to keep them.
    stratified  balanced overfit subset (memorisation gate) — train and
                val are the SAME indices (watch train loss; val = sanity):
                  split: {strategy: stratified, n_per_class: 8}
    memorise    FULL-dataset memorisation (identifiability probe): train
                and val are ALL fixes (same indices; multistorm fixes
                excluded by default, as for 'year'):
                  split: {strategy: memorise}
    multistorm  the multi-driver fixes as an OOD test set:
                  split: {strategy: multistorm}   -> {'test': idx}

Seeding: pass ONE seed (trainer.seed) — it orders every split's epoch
streams here (numpy) and the caller feeds the same seed to jax model init
(utils.jax_core.helpers.create_rng). This module is jax-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.cyclone_jax.data.sources.library import load_library
from experiments.cyclone_jax.data.inputs import InputSpec, resolve_input
from experiments.cyclone_jax.data.targets import TargetSpec, resolve_target
from experiments.cyclone_jax.data.normalise import NormSpec, resolve_normalise
from experiments.cyclone_jax.data.sampler import (
    Loader, Sampler, split_by_year, stratified_fixes,
)
from experiments.cyclone_jax.data.batching import collate


class BatchStream:
    """Re-iterable batch stream over one split.

    `for batch in stream` reshuffles each pass (epoch auto-increments —
    PyTorch-DataLoader semantics, deterministic overall given the seed);
    stream.epoch(e) gives explicit epoch control (notebooks, resume).
    """

    def __init__(self, loader, indices, batch_size, seed=0, shuffle=True,
                 drop_last=True):
        self._loader  = loader
        self._sampler = Sampler(indices, batch_size, seed=seed,
                                shuffle=shuffle, drop_last=drop_last)
        self._epoch   = 0

    def __len__(self):
        """Batches per epoch."""
        return len(self._sampler)

    @property
    def indices(self):
        return self._sampler.indices

    def epoch(self, epoch):
        """Yield collated batches for one explicit epoch."""
        pad_to = self._loader.inputs.pad_to
        for idx in self._sampler.epoch(epoch):
            yield collate([self._loader.build(int(i)) for i in idx], pad_to)

    def __iter__(self):
        it = self.epoch(self._epoch)
        self._epoch += 1
        return it


@dataclass
class DataBundle:
    """Everything build_data resolved, by name (notebook-friendly)."""
    lib:     dict
    inputs:  InputSpec
    targets: TargetSpec
    loader:  Loader
    splits:  dict[str, np.ndarray]        # split name -> fix indices
    streams: dict[str, BatchStream] = field(default_factory=dict)
    norms:   NormSpec | None = None       # ALSO attached as loader.norms


def _multistorm_mask(loader):
    """Bool mask over the fix table: True at multi-driver timestamps
    (shelf multi_times — which storm do these obs belong to?)."""
    multi = loader.lib['shelves']['cyclone'].get('multi_times')
    if multi is None or not len(multi):
        return np.zeros(len(loader), bool)
    return np.isin(np.asarray(loader.fixes['time']), np.asarray(multi))


def _resolve_splits(cfg, loader, seed):
    split = cfg.get('split')
    if not split:
        return {'all': np.arange(len(loader))}
    strategy = split.get('strategy')
    if strategy == 'year':
        years = split.get('years') or {}
        missing = {'train', 'val', 'test'} - set(years)
        if missing:
            raise ValueError(f"split.years missing {sorted(missing)}.")
        out = split_by_year(loader.fixes['time'], years['train'],
                            years['val'], years['test'])
        if split.get('exclude_multistorm', True):
            keep = ~_multistorm_mask(loader)
            out = {k: v[keep[v]] for k, v in out.items()}
        return out
    if strategy == 'stratified':
        if 'n_per_class' not in split:
            raise ValueError("split.strategy 'stratified' requires "
                             "n_per_class.")
        idx = stratified_fixes(loader, int(split['n_per_class']), seed=seed)
        return {'train': idx, 'val': idx.copy()}
    if strategy == 'memorise':
        idx = np.arange(len(loader))
        if split.get('exclude_multistorm', True):
            idx = idx[~_multistorm_mask(loader)]
        return {'train': idx, 'val': idx.copy()}
    if strategy == 'multistorm':
        return {'test': np.nonzero(_multistorm_mask(loader))[0]}
    raise ValueError(f"unknown split.strategy {strategy!r} — "
                     f"'year', 'stratified', 'memorise' or 'multistorm'.")


def build_data(cfg, seed=0, check_fresh=True):
    """Data config block -> DataBundle (see module docstring).

    Keys read beyond the spec keys (inputs/targets resolve their own):
    root, sshs_min, drop_subtropical, split, batch_size. Streams are built
    only when batch_size is set (pure-inspection use skips it); empty
    splits get no stream. Train streams shuffle and drop the last partial
    batch; val/test streams are sequential and keep it (full coverage).
    """
    inputs  = resolve_input(cfg)
    targets = resolve_target(cfg)
    lib = load_library(cfg['root'], names=tuple(inputs.sources) + ('cyclone',),
                       check_fresh=check_fresh)
    loader = Loader(lib, inputs, targets,
                    sshs_min=int(cfg.get('sshs_min', 3)),
                    drop_subtropical=bool(cfg.get('drop_subtropical', False)))

    splits = _resolve_splits(cfg, loader, seed)
    norms = _resolve_norms(cfg, loader, splits)

    streams = {}
    batch_size = cfg.get('batch_size')
    if batch_size:
        for name, idx in splits.items():
            if not len(idx):
                continue
            train = name in ('train', 'all')
            streams[name] = BatchStream(loader, idx, int(batch_size),
                                        seed=seed, shuffle=train,
                                        drop_last=train)

    return DataBundle(lib=lib, inputs=inputs, targets=targets,
                      loader=loader, splits=splits, streams=streams,
                      norms=norms)


def _resolve_norms(cfg, loader, splits):
    """normalise block -> NormSpec attached to the loader (or None).

    'stats: auto' computes over the TRAIN split (the 'all' split when no
    split block exists) — stats are properties of a training distribution
    and get saved with the run (train.py -> run_dir/norm_stats.json).
    """
    policy = resolve_normalise(cfg)
    if policy is None:
        return None
    if policy.auto:
        idx = splits.get('train', splits.get('all'))
        if idx is None or not len(idx):
            raise ValueError(
                "normalise: stats: auto, but this scenario has no train/all "
                "split to compute statistics from. A stress-test scenario "
                "must name WHICH training distribution it is relative to: "
                "either paste inline stats into its normalise.stats block, "
                "or evaluate it with the training run's saved stats "
                "(run_dir/norm_stats.json — evaluate's --stats pointer).")
        norms = policy.materialise(loader, idx)
    else:
        norms = policy.materialise(loader)
    loader.norms = norms
    return norms
