"""
Tests for cyclone_jax sampler / batching — the assembly contract: ragged
named samples, causality (relative time <= 0), the union-channel mask
invariant, y via TargetSpec, deterministic seeded index streams, and the
sample -> device-batch translation (fixed pad_to, meta routing).
"""

import numpy as np
import pytest

from experiments.cyclone_jax.data.inputs import CHANNEL_ORDER, resolve_input
from experiments.cyclone_jax.data.targets import resolve_target
from experiments.cyclone_jax.data.sampler import (
    X_FIELDS, Loader, Sampler, split_by_year, stratified_fixes,
)
from experiments.cyclone_jax.data.batching import collate

# library fixture: conftest.py (shared with test_interface)


@pytest.fixture(scope='module')
def loader(library):
    return Loader(library, resolve_input({}), resolve_target({}))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestLoader:

    def test_ragged_named_schema(self, loader):
        s = loader.build(5)
        assert set(s) == {'x', 'y'}
        assert set(s['x']) == set(X_FIELDS)
        n = len(s['x']['lat'])
        assert n > 0
        for f in ('lon', 'level', 'time', 'id'):
            assert s['x'][f].shape == (n,)
        C = len(CHANNEL_ORDER)
        assert s['x']['obs'].shape == (n, C)
        assert s['x']['missing'].shape == (n, C)
        assert s['x']['missing'].dtype == bool

    def test_level_is_finite(self, loader):
        x = loader.build(5)['x']
        assert np.isfinite(x['level']).all()        # vertical gate upstream

    def test_causality_relative_time_nonpositive(self, loader):
        x = loader.build(10)['x']
        assert np.all(x['time'] <= 0.0)

    def test_deterministic(self, loader):
        a, b = loader.build(7), loader.build(7)
        for f in X_FIELDS:
            np.testing.assert_array_equal(a['x'][f], b['x'][f])

    def test_shuffle_samples_permutes_rows_deterministically(self, library,
                                                             loader):
        """shuffle_samples: station rows are co-permuted (a permutation of
        the plain build, rows intact), the permutation differs per fix,
        and the SAME fix rebuilds the SAME bytes (input_collisions and the
        prediction records stay valid with the knob on)."""
        sh = Loader(library, resolve_input({}), resolve_target({}),
                    shuffle_samples=True, seed=3)
        a, b = sh.build(7)['x'], sh.build(7)['x']
        for f in X_FIELDS:
            np.testing.assert_array_equal(a[f], b[f])   # per-fix frozen
        plain = loader.build(7)['x']
        order = np.argsort(a['lat'], kind='stable')
        base = np.argsort(plain['lat'], kind='stable')
        np.testing.assert_array_equal(a['lat'][order], plain['lat'][base])
        np.testing.assert_array_equal(a['obs'][order], plain['obs'][base])
        assert not np.array_equal(a['lat'], plain['lat'])   # order changed
        sh2 = Loader(library, resolve_input({}), resolve_target({}),
                     shuffle_samples=True, seed=4)
        assert not np.array_equal(sh2.build(7)['x']['lat'], a['lat'])

    def test_union_channels_masked_per_source(self, loader):
        """Every token carries the FULL channel union; channels a source
        lacks are zeroed with missing False (land: sst; marine:
        station_pressure)."""
        x = loader.build(8)['x']
        ch = loader.inputs.channel_index
        land, marine = x['id'] == -1.0, x['id'] == 1.0
        assert land.any() and marine.any()
        assert not x['missing'][land, ch['sst']].any()
        assert np.all(x['obs'][land, ch['sst']] == 0.0)
        assert not x['missing'][marine, ch['station_pressure']].any()
        assert x['missing'][land, ch['station_pressure']].all()

    def test_y_contract(self, loader):
        s = loader.build(0)
        y = s['y']
        assert set(y) == {'target', 'sid', 'lat', 'lon', 'time'}
        sshs = int(loader.fixes['usa_sshs'][0])
        assert y['target'] == loader.targets.label(sshs)
        assert y['sid'] == 'AL012020'
        assert y['time'] == loader.fixes['time'][0]

    def test_class_set_filters_fix_table(self, library):
        """Fixes outside the label space are not samples — a class_set
        narrower than sshs_min must never reach build_y."""
        full = Loader(library, resolve_input({}), resolve_target({}))
        cat45 = Loader(library, resolve_input({}),
                       resolve_target({'class_set': [4, 5]}))
        sshs = np.asarray(cat45.fixes['usa_sshs']).astype(int)
        assert len(cat45) < len(full) and len(cat45) > 0
        assert set(sshs) <= {4, 5}
        assert cat45.build(0)['y']['target'] in (0, 1)

    def test_max_stations_keeps_nearest(self, library):
        spec_all = resolve_input({})
        target = resolve_target({})
        n_all = len(Loader(library, spec_all, target).build(8)['x']['lat'])
        k = max(1, n_all // 2)
        spec_k = resolve_input({'selection': 'max_stations',
                                'max_stations': k})
        ld = Loader(library, spec_k, target)
        x = ld.build(8)['x']
        assert len(x['lat']) == k
        # kept set = the k smallest haversine distances to the fix
        from utils.geoscience.geodesic import haversine_np
        x_all = Loader(library, spec_all, target).build(8)['x']
        d_all = haversine_np(np.float32(ld.fixes['lat'][8]),
                             np.float32(ld.fixes['lon'][8]),
                             x_all['lat'], x_all['lon'])
        d_kept = haversine_np(np.float32(ld.fixes['lat'][8]),
                              np.float32(ld.fixes['lon'][8]),
                              x['lat'], x['lon'])
        assert d_kept.max() <= np.sort(d_all)[k - 1] + 1e-6

    def test_leakage_x_fields_fixed(self, loader):
        """x carries exactly the allowlisted fields — nothing from the
        cyclone volume's target columns can ride along unnoticed."""
        assert set(loader.build(3)['x']) == set(X_FIELDS)


# ---------------------------------------------------------------------------
# Splits + overfit sets
# ---------------------------------------------------------------------------

class TestSelections:

    def test_split_by_year_disjoint(self, loader):
        t = np.asarray(loader.fixes['time'])
        out = split_by_year(t, [2019], [2021], [2020])
        assert len(out['test']) == len(t)               # all fixes 2020
        assert len(out['train']) == len(out['val']) == 0

    def test_split_overlap_raises(self, loader):
        t = np.asarray(loader.fixes['time'])
        with pytest.raises(ValueError, match='disjoint'):
            split_by_year(t, [2020], [2020], [2021])

    def test_stratified_counts_and_seed(self, loader):
        idx = stratified_fixes(loader, n_per_class=2, seed=0)
        sshs = np.asarray(loader.fixes['usa_sshs']).astype(int)[idx]
        for c in loader.targets.class_set:
            assert np.sum(sshs == c) <= 2
        np.testing.assert_array_equal(
            idx, stratified_fixes(loader, n_per_class=2, seed=0))


# ---------------------------------------------------------------------------
# Sampler — seeded index streams
# ---------------------------------------------------------------------------

class TestSampler:

    def test_len_and_full_batches(self):
        s = Sampler(np.arange(10), batch_size=3, seed=0)
        assert len(s) == 3                              # drop_last
        assert all(len(b) == 3 for b in s.epoch(0))

    def test_epoch_deterministic_in_seed_and_epoch(self):
        a = list(Sampler(np.arange(20), 5, seed=1).epoch(2))
        b = list(Sampler(np.arange(20), 5, seed=1).epoch(2))
        c = list(Sampler(np.arange(20), 5, seed=1).epoch(3))
        np.testing.assert_array_equal(np.concatenate(a), np.concatenate(b))
        assert not np.array_equal(np.concatenate(a), np.concatenate(c))

    def test_seed_changes_order(self):
        a = np.concatenate(list(Sampler(np.arange(20), 5, seed=1).epoch(0)))
        b = np.concatenate(list(Sampler(np.arange(20), 5, seed=2).epoch(0)))
        assert not np.array_equal(a, b)

    def test_no_shuffle_keeps_order_and_partial_batch(self):
        s = Sampler(np.arange(10), 4, shuffle=False, drop_last=False)
        batches = list(s.epoch(0))
        assert len(s) == 3 and len(batches[-1]) == 2
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(10))

    def test_shuffle_is_permutation(self):
        s = Sampler(np.arange(17), 17, seed=5)
        np.testing.assert_array_equal(np.sort(next(iter(s))), np.arange(17))

    def test_guards(self):
        with pytest.raises(ValueError, match='indices'):
            Sampler(np.array([]), 4)
        with pytest.raises(ValueError, match='batch_size'):
            Sampler(np.arange(4), 0)


# ---------------------------------------------------------------------------
# Collate — sample -> device batch
# ---------------------------------------------------------------------------

class TestCollate:

    def test_shapes_and_routing(self, loader):
        pad_to = 128
        batch = collate([loader.build(i) for i in range(4)], pad_to)
        C = loader.inputs.n_channels
        assert batch['X']['obs'].shape == (4, pad_to, C)
        assert batch['X']['missing'].shape == (4, pad_to, C)
        for f in ('lat', 'lon', 'level', 'time', 'id'):
            assert batch['X'][f].shape == (4, pad_to)
        assert batch['X']['station_mask'].shape == (4, pad_to)
        assert batch['y'].shape == (4,) and batch['y'].dtype == np.int32
        assert set(batch['meta']) == {'sid', 'lat', 'lon', 'time',
                                      'n_stations'}
        assert isinstance(batch['meta']['sid'], list)

    def test_mask_matches_padding(self, loader):
        pad_to = 512
        batch = collate([loader.build(i) for i in range(3)], pad_to)
        for b in range(3):
            n = batch['meta']['n_stations'][b]
            assert batch['X']['station_mask'][b].sum() == n
            assert np.all(batch['X']['obs'][b, n:] == 0.0)
            assert not batch['X']['missing'][b, n:].any()

    def test_truncation_capped_at_pad_to(self, loader):
        pad_to = 4
        batch = collate([loader.build(8)], pad_to)
        assert batch['meta']['n_stations'][0] == pad_to
        assert batch['X']['station_mask'][0].all()

    def test_pipeline_sampler_to_batch(self, loader):
        """Indices -> Loader -> collate, the P5 interface composition."""
        s = Sampler(np.arange(len(loader)), 4, seed=0)
        idx = next(iter(s))
        batch = collate([loader.build(int(i)) for i in idx], 64)
        assert batch['y'].shape == (4,)
        np.testing.assert_array_equal(
            batch['meta']['time'],
            np.asarray(loader.fixes['time'])[idx])
