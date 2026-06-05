"""
experiments/sparse_obs_cross_attn/datamodule.py

TCDataModule: coordinates IBTrACSDataset and InsituLandDataset into
Trainer-compatible loaders for the sparse_obs_cross_attn experiment.

As additional data sources are added (reanalysis, radar, satellite) they
appear alongside ibtracs and insitu_land in setup(), each contributing
observations to the same sample structure via TCDataset.

Batch format
------------
    {'X': dict_of_jnp_arrays, 'y': jnp.int32_array}

    X keys: query_lat, query_lon, station_obs, station_coords,
            station_mask, obs_mask
    y     : (B,) int32 ordinal class labels

Class balance
-------------
tc_fraction controls the TC share per batch (default 0.5). Set it below
0.5 to oversample background and counteract the natural bias toward storm
samples in the IBTrACS data (~10 K TC rows vs millions of background
timestamps).
"""

from __future__ import annotations

import warnings
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np

from datasets.datamodule import BaseDataModule
from utils.jax_core.helpers import create_rng

from experiments.sparse_obs_cross_attn.ibtracs import IBTrACSDataset
from experiments.sparse_obs_cross_attn.insitu_land import InsituLandDataset
from experiments.sparse_obs_cross_attn.dataset import TCDataset


# ---------------------------------------------------------------------------
# Background pool construction
# ---------------------------------------------------------------------------

def _build_background_pool(
    insitu:              InsituLandDataset,
    ibtracs:             IBTrACSDataset,
    exclude_multi_times: Optional[np.ndarray] = None,
    buffer_hours:        float = 6.0,
) -> np.ndarray:
    """Return InsituLand timestamps that are clear of any active TC observation.

    Parameters
    ----------
    insitu : InsituLandDataset (already split to a single season range)
    ibtracs : IBTrACSDataset (already split to the same season range)
    exclude_multi_times : int64 array, optional
        Additional timestamps to exclude (multi-storm blackout set).
    buffer_hours : float
        Timestamps within this many hours of any IBTrACS ISO_TIME are excluded.

    Returns
    -------
    np.ndarray  int64 Unix-ns timestamps
    """
    all_ts = np.unique(insitu.timestamps)
    tc_ts  = np.sort(ibtracs['ISO_TIME'])
    buf_ns = int(buffer_hours * 3600 * 1e9)

    lo = np.searchsorted(tc_ts, all_ts - buf_ns, side='left')
    hi = np.searchsorted(tc_ts, all_ts + buf_ns, side='right')
    near_tc = (hi - lo) > 0

    if exclude_multi_times is not None and len(exclude_multi_times) > 0:
        multi_set = set(int(t) for t in exclude_multi_times)
        in_multi  = np.array([int(t) in multi_set for t in all_ts], dtype=bool)
    else:
        in_multi = np.zeros(len(all_ts), dtype=bool)

    pool = all_ts[~near_tc & ~in_multi]
    if len(pool) == 0:
        warnings.warn(
            "Background timestamp pool is empty after filtering. "
            "Check background_buffer_hours and InsituLand split coverage.",
            UserWarning,
            stacklevel=2,
        )
    return pool


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------

def _collate(samples: list[dict]) -> dict:
    """Stack a list of sample dicts into {'X': dict_of_jnp_arrays, 'y': jnp.array}."""
    X = {
        k: jnp.array(np.stack([s[k] for s in samples], axis=0))
        for k in (
            'query_coords',
            'station_obs', 'station_coords',
            'station_mask', 'obs_mask',
        )
    }
    y = jnp.array(
        np.stack([s['label'] for s in samples], axis=0),
        dtype=jnp.int32,
    )
    return {'X': X, 'y': y}


# ---------------------------------------------------------------------------
# TCLoader
# ---------------------------------------------------------------------------

class TCLoader:
    """Re-iterable loader producing balanced {'X': dict, 'y': labels} batches.

    Each batch contains tc_fraction * batch_size TC samples interleaved
    with background samples drawn on-the-fly from the background pool.

    Parameters
    ----------
    dataset : TCDataset
    batch_size : int
    tc_fraction : float
        Fraction of each batch that is TC samples (default 0.5).
    shuffle : bool
        Reshuffle TC sample order each epoch.
    seed : int
    fov_lat, fov_lon : tuple[float, float]
        Field-of-view bounds for random background query positions.
    """

    def __init__(
        self,
        dataset:     TCDataset,
        batch_size:  int,
        tc_fraction: float = 0.5,
        shuffle:     bool  = True,
        seed:        int   = 0,
        fov_lat:     tuple[float, float] = (0.0, 30.0),
        fov_lon:     tuple[float, float] = (-100.0, -45.0),
    ) -> None:

        
        if not (0.0 < tc_fraction < 1.0):
            raise ValueError(f"tc_fraction must be in (0, 1), got {tc_fraction}")
        self._dataset    = dataset
        self._batch_size = batch_size
        self._tc_half    = max(1, round(batch_size * tc_fraction))
        self._bg_half    = batch_size - self._tc_half
        self._shuffle    = shuffle
        self._base_seed  = seed
        self._fov_lat    = fov_lat
        self._fov_lon    = fov_lon
        self._epoch      = 0

    def __iter__(self) -> Iterator[dict]:
        # Derive a per-epoch numpy seed from a JAX key — same pattern as ArrayLoader.
        jax_key  = jax.random.fold_in(create_rng(self._base_seed), self._epoch)
        np_seed  = int(jax_key[0])
        rng      = np.random.default_rng(np_seed)
        self._epoch += 1

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

            if len(tc_buf) == self._tc_half:
                while len(bg_buf) < self._bg_half:
                    bg_tries += 1
                    if bg_tries > self._bg_half * 50:
                        warnings.warn(
                            f"Could not draw {self._bg_half} background samples "
                            f"after {bg_tries} attempts.",
                            UserWarning,
                            stacklevel=2,
                        )
                        break
                    bg = self._dataset.get_background_sample(
                        rng, self._fov_lat, self._fov_lon,
                    )
                    if bg is not None:
                        bg_buf.append(bg)

                if len(bg_buf) == self._bg_half:
                    yield _collate(tc_buf + bg_buf)

                tc_buf   = []
                bg_buf   = []
                bg_tries = 0

    def __len__(self) -> int:
        return max(1, len(self._dataset) // max(1, self._tc_half))


# ---------------------------------------------------------------------------
# TCDataModule
# ---------------------------------------------------------------------------

class TCDataModule(BaseDataModule):
    """DataModule for the sparse_obs_cross_attn experiment.

    Subclasses BaseDataModule. Because samples are assembled on-the-fly from
    two large datasets (74 M obs rows), this module overrides train_loader /
    val_loader / test_loader to return TCLoader instances rather than
    ArrayLoaders. The in-memory array accessors (train_arrays etc.) are not
    applicable and raise NotImplementedError.

    Config keys (data: block in YAML)
    ----------------------------------
        ibtracs_path             str
        multi_storm_path         str   optional
        insitu_obs_path          str
        insitu_meta_path         str
        reliability_levels       list  default [always_active, mostly_active]
        obs_vars                 list  default DEFAULT_OBS_VARS (from insitu_land)
        radius_km                float default 500.0
        time_window_hours        float default 3.0
        max_stations             int   default 64
        min_stations             int   default 1
        batch_size               int   default 64
        tc_fraction              float default 0.5
        fov_lat                  list  default [0.0, 30.0]
        fov_lon                  list  default [-100.0, -45.0]
        background_buffer_hours  float default 6.0
        location_encoding        str   default 'unit_circle' ('unit_circle'|'domain')
        obs_normalisation        str   default 'minmax_01'
                                        ('minmax_01'|'minmax_11'|'standardise')
        obs_bounds               dict  optional — {var: [min, max]} for minmax_*,
                                        {var: [mean, std]} for standardise
    """

    @classmethod
    def from_config(cls, config: dict) -> TCDataModule:
        dm = cls()
        dm.setup(config)
        return dm

    # ------------------------------------------------------------------
    # BaseDataModule abstract methods — not applicable for on-the-fly sampling
    # ------------------------------------------------------------------

    def train_arrays(self) -> dict:
        raise NotImplementedError(
            "TCDataModule assembles samples on-the-fly. Use train_loader() instead."
        )

    def val_arrays(self) -> dict:
        raise NotImplementedError(
            "TCDataModule assembles samples on-the-fly. Use val_loader() instead."
        )

    def test_arrays(self) -> dict:
        raise NotImplementedError(
            "TCDataModule assembles samples on-the-fly. Use test_loader() instead."
        )

    def setup(self, config: dict) -> None:
        ibtracs_path      = config['ibtracs_path']
        multi_path        = config.get('multi_storm_path')
        obs_path          = config['insitu_obs_path']
        meta_path         = config['insitu_meta_path']
        reliability       = config.get('reliability_levels', ['always_active', 'mostly_active'])
        obs_vars          = config.get('obs_vars', None)
        radius_km         = float(config.get('radius_km', 500.0))
        time_window_h     = float(config.get('time_window_hours', 3.0))
        max_stations      = int(config.get('max_stations', 64))
        min_stations      = int(config.get('min_stations', 1))
        buf_hours         = float(config.get('background_buffer_hours', 6.0))
        location_encoding = config.get('location_encoding', 'unit_circle')
        obs_normalisation = config.get('obs_normalisation', 'minmax_01')

        # obs_bounds: dict[var, [min, max]] or dict[var, [mean, std]] for standardise
        obs_bounds_raw = config.get('obs_bounds', None)
        obs_bounds: Optional[dict[str, tuple[float, float]]] = None
        if obs_bounds_raw is not None:
            obs_bounds = {k: tuple(v) for k, v in obs_bounds_raw.items()}

        self._batch_size  = int(config.get('batch_size', 64))
        self._tc_fraction = float(config.get('tc_fraction', 0.5))
        self._fov_lat     = tuple(config.get('fov_lat', [0.0, 30.0]))
        self._fov_lon     = tuple(config.get('fov_lon', [-100.0, -45.0]))

        ibtracs_full = IBTrACSDataset(ibtracs_path, multi_path)
        insitu_full  = InsituLandDataset(obs_path, meta_path)
        if reliability:
            insitu_full = insitu_full.filter_reliability(reliability)

        multi_times: Optional[np.ndarray] = None
        if multi_path is not None:
            ms = np.load(multi_path, allow_pickle=True)
            multi_times = ms['ISO_TIME']

        for split_name in ('train', 'val', 'test'):
            ib  = ibtracs_full.split(split_name)
            ins = insitu_full.split(split_name)
            bg_pool = _build_background_pool(
                insitu=ins,
                ibtracs=ib,
                exclude_multi_times=multi_times,
                buffer_hours=buf_hours,
            )
            ds = TCDataset(
                ibtracs=ib,
                insitu=ins,
                radius_km=radius_km,
                time_window_hours=time_window_h,
                max_stations=max_stations,
                min_stations=min_stations,
                obs_vars=obs_vars,
                background_timestamps=bg_pool,
                location_encoding=location_encoding,
                fov_lat=self._fov_lat,
                fov_lon=self._fov_lon,
                obs_bounds=obs_bounds,
                obs_normalisation=obs_normalisation,
            )
            setattr(self, f'_{split_name}_ds', ds)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def train_loader(self, batch_size: Optional[int] = None, seed: int = 0, shuffle: bool = True) -> TCLoader:
        return TCLoader(
            self._train_ds,
            batch_size or self._batch_size,
            tc_fraction=self._tc_fraction,
            shuffle=shuffle,
            seed=seed,
            fov_lat=self._fov_lat,
            fov_lon=self._fov_lon,
        )

    def val_loader(self, batch_size: Optional[int] = None, shuffle: bool = False) -> TCLoader:
        return TCLoader(
            self._val_ds,
            batch_size or self._batch_size,
            tc_fraction=self._tc_fraction,
            shuffle=shuffle,
            seed=0,
            fov_lat=self._fov_lat,
            fov_lon=self._fov_lon,
        )

    def test_loader(self, batch_size: Optional[int] = None, shuffle: bool = False) -> TCLoader:
        return TCLoader(
            self._test_ds,
            batch_size or self._batch_size,
            tc_fraction=self._tc_fraction,
            shuffle=shuffle,
            seed=0,
            fov_lat=self._fov_lat,
            fov_lon=self._fov_lon,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        for name in ('train', 'val', 'test'):
            ds = getattr(self, f'_{name}_ds')
            bg = len(ds.background_timestamps) if ds.background_timestamps is not None else 0
            print(
                f"  {name:<6}: {len(ds):>6} TC samples  "
                f"{bg:>8} background timestamps"
            )
