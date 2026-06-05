"""
experiments/sparse_obs_cross_attn/datamodule.py

JointDataModule: wraps JointTCDataset into Trainer-compatible loaders.

Batch format produced by loaders
---------------------------------
Each batch is a dict {'X': model_input_dict, 'y': label_array} as
expected by the generic Trainer:

    batch['X'] = {
        'query_coords': jnp.array  (B, 3)              unit-sphere position
        'station_obs':  jnp.array  (B, max_stations, 7) NaN→0
        'station_mask': jnp.array  (B, max_stations)   bool  real vs padding
        'obs_mask':     jnp.array  (B, max_stations, 7) bool  valid vs missing
    }
    batch['y'] = jnp.array  (B,)  int32 ordinal class label

Metadata (sid, iso_time, n_stations) are NOT in the training batch.
Use get_eval_samples() to retrieve annotated samples for evaluation.

Class balance
-------------
Each batch is batch_size // 2 TC samples + batch_size // 2 background
samples, interleaved so every batch is balanced regardless of ordering.

Background timestamp pool
-------------------------
Built at setup time: all unique InsituLand timestamps NOT within
background_buffer_hours of any IBTrACS TC observation and NOT in the
multi-storm blackout set.
"""

from __future__ import annotations

import warnings
from typing import Iterator, Optional

import numpy as np
import jax.numpy as jnp

from datasets.ibtracs.dataset import IBTrACSDataset
from datasets.insitu_land.dataset import InsituLandDataset
from datasets.joint.dataset import JointTCDataset


# ---------------------------------------------------------------------------
# Background pool construction
# ---------------------------------------------------------------------------

def _build_background_pool(
    insitu:              InsituLandDataset,
    ibtracs:             IBTrACSDataset,
    exclude_multi_times: Optional[np.ndarray] = None,
    buffer_hours:        float = 6.0,
) -> np.ndarray:
    """Return InsituLand timestamps clear of TC observations.

    Uses binary search on sorted IBTrACS timestamps: O(N log M).
    """
    all_ts = np.unique(insitu['report_timestamp'])
    tc_ts  = np.sort(ibtracs['ISO_TIME'])
    buf_ns = int(buffer_hours * 3600 * 1e9)

    lo = np.searchsorted(tc_ts, all_ts - buf_ns, side='left')
    hi = np.searchsorted(tc_ts, all_ts + buf_ns, side='right')
    near_tc = (hi - lo) > 0

    if exclude_multi_times is not None and len(exclude_multi_times) > 0:
        multi_set = set(int(t) for t in exclude_multi_times)
        in_multi  = np.array([int(t) in multi_set for t in all_ts])
    else:
        in_multi = np.zeros(len(all_ts), dtype=bool)

    pool = all_ts[~near_tc & ~in_multi]
    if len(pool) == 0:
        warnings.warn(
            "Background timestamp pool is empty after filtering. "
            "Check background_buffer_hours and InsituLand split coverage.",
            UserWarning, stacklevel=2,
        )
    return pool


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def _collate(samples: list[dict]) -> dict:
    """Stack list of single-sample dicts into {'X': dict, 'y': array}."""
    X = {
        k: np.stack([s[k] for s in samples], axis=0)
        for k in ('query_coords', 'station_obs', 'station_mask', 'obs_mask')
    }
    y = np.stack([s['label'] for s in samples], axis=0)
    return {
        'X': {k: jnp.array(v) for k, v in X.items()},
        'y': jnp.array(y, dtype=jnp.int32),
    }


# ---------------------------------------------------------------------------
# JointLoader
# ---------------------------------------------------------------------------

class JointLoader:
    """Re-iterable loader producing balanced {'X': dict, 'y': label} batches.

    Parameters
    ----------
    dataset : JointTCDataset
    batch_size : int
        Total batch size.
    tc_fraction : float
        Fraction of each batch that is TC samples.  Default 0.5 (equal
        halves).  Set below 0.5 (e.g. 0.4) to oversample background and
        reinforce the no-storm base rate.
    shuffle : bool
        Reshuffle TC sample order each epoch.
    seed : int
        Combined with epoch count for deterministic per-epoch shuffles.
    fov_lat, fov_lon : tuple[float, float]
        Field of view for random background query positions.
    """

    def __init__(
        self,
        dataset:     JointTCDataset,
        batch_size:  int,
        tc_fraction: float = 0.5,
        shuffle:     bool  = True,
        seed:        int   = 0,
        fov_lat:     tuple[float, float] = (0.0, 30.0),
        fov_lon:     tuple[float, float] = (-100.0, -45.0),
    ) -> None:
        if not (0.0 < tc_fraction < 1.0):
            raise ValueError(f"tc_fraction must be in (0, 1), got {tc_fraction}")
        self._dataset     = dataset
        self._batch_size  = batch_size
        self._tc_half     = max(1, round(batch_size * tc_fraction))
        self._bg_half     = batch_size - self._tc_half
        self._shuffle     = shuffle
        self._base_seed   = seed
        self._fov_lat     = fov_lat
        self._fov_lon     = fov_lon
        self._epoch_count = 0

    def __iter__(self) -> Iterator[dict]:
        tc_half  = self._tc_half
        bg_half  = self._bg_half

        rng = np.random.default_rng(self._base_seed + self._epoch_count)
        self._epoch_count += 1

        indices = np.arange(len(self._dataset))
        if self._shuffle:
            rng.shuffle(indices)

        tc_buf:  list[dict] = []
        bg_buf:  list[dict] = []
        bg_tries = 0

        for idx in indices:
            sample = self._dataset.get_tc_sample(int(idx), rng=rng)
            if sample is None:
                continue
            tc_buf.append(sample)

            if len(tc_buf) == tc_half:
                while len(bg_buf) < bg_half:
                    bg_tries += 1
                    if bg_tries > bg_half * 50:
                        warnings.warn(
                            f"Could not draw {bg_half} background samples "
                            f"after {bg_tries} attempts.",
                            UserWarning, stacklevel=2,
                        )
                        break
                    bg = self._dataset.get_background_sample(
                        rng, self._fov_lat, self._fov_lon
                    )
                    if bg is not None:
                        bg_buf.append(bg)

                if len(bg_buf) == bg_half:
                    yield _collate(tc_buf + bg_buf)

                tc_buf = []
                bg_buf = []
                bg_tries = 0

    def __len__(self) -> int:
        return max(1, len(self._dataset) // (self._batch_size // 2))


# ---------------------------------------------------------------------------
# JointDataModule
# ---------------------------------------------------------------------------

class JointDataModule:
    """DataModule for the sparse_obs_cross_attn experiment.

    Loads IBTrACS and InsituLand, splits by season, builds per-split
    background timestamp pools, and exposes Trainer-compatible loaders.

    Config keys (YAML  data:  block)
    ---------------------------------
        ibtracs_path             str
        multi_storm_path         str   optional
        insitu_obs_path          str
        insitu_meta_path         str
        reliability_levels       list  default [always_active, mostly_active]
        radius_km                float default 500.0
        time_window_hours        float default 3.0
        max_stations             int   default 64
        min_stations             int   default 1
        batch_size               int   default 64
        tc_fraction              float default 0.5  (TC share per batch; < 0.5 oversamples background)
        fov_lat                  list  default [0.0, 30.0]
        fov_lon                  list  default [-100.0, -45.0]
        background_buffer_hours  float default 3.0
    """

    @classmethod
    def from_config(cls, config: dict) -> 'JointDataModule':
        dm = cls()
        dm.setup(config)
        return dm

    def setup(self, config: dict) -> None:
        ibtracs_path  = config['ibtracs_path']
        multi_path    = config.get('multi_storm_path')
        obs_path      = config['insitu_obs_path']
        meta_path     = config['insitu_meta_path']
        reliability   = config.get('reliability_levels', ['always_active', 'mostly_active'])
        radius_km     = float(config.get('radius_km', 500.0))
        time_window_h = float(config.get('time_window_hours', 3.0))
        max_stations  = int(config.get('max_stations', 64))
        min_stations  = int(config.get('min_stations', 1))
        buf_hours     = float(config.get('background_buffer_hours', 3.0))

        self._batch_size  = int(config.get('batch_size', 64))
        self._tc_fraction = float(config.get('tc_fraction', 0.5))
        self._fov_lat     = tuple(config.get('fov_lat', [0.0, 30.0]))
        self._fov_lon     = tuple(config.get('fov_lon', [-100.0, -45.0]))
        self._config     = config

        ibtracs_full = IBTrACSDataset(ibtracs_path, multi_path)
        insitu_full  = InsituLandDataset(obs_path, meta_path)
        if reliability:
            insitu_full = insitu_full.filter_reliability(reliability)

        multi_times: Optional[np.ndarray] = None
        if multi_path is not None:
            ms = np.load(multi_path, allow_pickle=True)
            multi_times = ms['ISO_TIME']

        for split in ('train', 'val', 'test'):
            ib = ibtracs_full.split(split)
            ins = insitu_full.split(split)
            bg_pool = _build_background_pool(
                ins, ib,
                exclude_multi_times=multi_times,
                buffer_hours=buf_hours,
            )
            ds = JointTCDataset(
                ibtracs=ib, insitu=ins,
                radius_km=radius_km, time_window_hours=time_window_h,
                max_stations=max_stations, min_stations=min_stations,
                background_timestamps=bg_pool,
            )
            setattr(self, f'_{split}_ds', ds)

    # ------------------------------------------------------------------
    # Loaders (Trainer-compatible)
    # ------------------------------------------------------------------

    def train_loader(self, seed: int = 0) -> JointLoader:
        return JointLoader(
            self._train_ds, self._batch_size,
            tc_fraction=self._tc_fraction,
            shuffle=True, seed=seed,
            fov_lat=self._fov_lat, fov_lon=self._fov_lon,
        )

    def val_loader(self) -> JointLoader:
        return JointLoader(
            self._val_ds, self._batch_size,
            tc_fraction=self._tc_fraction,
            shuffle=False, seed=0,
            fov_lat=self._fov_lat, fov_lon=self._fov_lon,
        )

    def test_loader(self) -> JointLoader:
        return JointLoader(
            self._test_ds, self._batch_size,
            tc_fraction=self._tc_fraction,
            shuffle=False, seed=0,
            fov_lat=self._fov_lat, fov_lon=self._fov_lon,
        )

    # ------------------------------------------------------------------
    # Evaluation helper — includes metadata for plotting
    # ------------------------------------------------------------------

    def get_eval_samples(
        self,
        split: str = 'val',
        n: int = 16,
        seed: int = 0,
    ) -> list[dict]:
        """Return n assembled samples with full metadata for evaluation.

        Unlike the training loader, these samples include 'sid', 'iso_time',
        'n_stations', and the raw station lat/lon from InsituLand so that
        evaluate.py can build attention maps on a geographic plot.

        Parameters
        ----------
        split : 'train' | 'val' | 'test'
        n : int
        seed : int

        Returns
        -------
        list[dict]  each dict has all JointTCDataset keys plus 'station_latlons'
        """
        ds  = getattr(self, f'_{split}_ds')
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), min(n, len(ds)), replace=False)

        samples = []
        for i in idx:
            s = ds.get_tc_sample(int(i), rng=rng)
            if s is None:
                continue
            # Attach storm lat/lon for geographic plotting
            s['storm_lat'] = float(ds._lat[int(i)])
            s['storm_lon'] = float(ds._lon[int(i)])
            samples.append(s)
            if len(samples) >= n:
                break
        return samples

    def summary(self) -> None:
        for name in ('train', 'val', 'test'):
            ds = getattr(self, f'_{name}_ds')
            bg = len(ds.background_timestamps) if ds.background_timestamps is not None else 0
            print(f"  {name:<6}: {len(ds):>6} TC samples  {bg:>8} background timestamps")
