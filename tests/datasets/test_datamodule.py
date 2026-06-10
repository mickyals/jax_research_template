"""
Tests for datasets/datamodule.py.

Coverage
--------
TestArrayLoader         shuffle / no-shuffle; drop_last; len; per-epoch seed
                        variation; shuffle=False flag; correct batch shapes
TestDatasetRegistry     register / list / duplicate / unknown
TestApplyNorm           standard, minmax, none; no-leakage; NaN tolerance
TestInvertNorm          round-trip for each method
TestDataModule          single dataset: shapes, normalisation, NaN targets,
                        denormalise, no-leakage, all three norm methods,
                        missing multi_storm warning;
                        train/val/test loaders yield batches of correct size;
                        shuffle flag honoured; config shuffle override
TestDataModuleMulti     multiple datasets: combined size, normalised on union,
                        consistent dims, denormalise round-trip
TestDataModuleInterface BaseDataModule is abstract; DataModule is concrete;
                        from_config convenience constructor; summary output
"""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datasets.datamodule import (
    DATASETS,
    ArrayLoader,
    BaseDataModule,
    DataModule,
    _apply_norm,
    _invert_norm,
    list_datasets,
    register_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ibtracs_npz(path: Path, n: int = 300, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    seasons = np.array([2010] * 200 + [2021] * 50 + [2023] * 50, dtype=np.float32)
    data = {
        "SID":         np.array([f"2010001N{i:05d}" for i in range(n)]),
        "ISO_TIME":    np.array([f"2010-0{(i % 9) + 1}-01 00:00:00" for i in range(n)]),
        "SEASON":      seasons,
        "LAT":         rng.uniform(-10, 30, n).astype(np.float32),
        "LON":         rng.uniform(-100, 0, n).astype(np.float32),
        "STORM_SPEED": rng.uniform(0, 15, n).astype(np.float32),
        "STORM_DIR":   rng.uniform(0, 360, n).astype(np.float32),
        "USA_WIND":    rng.uniform(15, 80, n).astype(np.float32),
        "USA_PRES":    rng.uniform(920, 1010, n).astype(np.float32),
    }
    # ~10 % of USA_PRES missing (sparse secondary target behaviour)
    data["USA_PRES"][rng.choice(n, n // 10, replace=False)] = np.nan
    np.savez(path, **data)
    return path


def _make_multi_storm_npz(path: Path) -> Path:
    """Empty multi-storm file — no timestep is multi-storm."""
    np.savez(path, ISO_TIME=np.array([], dtype=object))
    return path


@pytest.fixture
def npz(tmp_path):
    return _make_ibtracs_npz(tmp_path / "ibtracs.npz")


@pytest.fixture
def ms_npz(tmp_path):
    return _make_multi_storm_npz(tmp_path / "ms.npz")


@pytest.fixture
def single_cfg(npz, ms_npz):
    return {
        "dataset":          "ibtracs",
        "npz_path":         str(npz),
        "multi_storm_path": str(ms_npz),
        "target_cols":      ["USA_WIND", "USA_PRES"],
        "feature_cols":     ["LAT", "LON", "STORM_SPEED", "STORM_DIR"],
        "feature_norm":     "standard",
        "target_norm":      "standard",
    }


@pytest.fixture
def dm(single_cfg):
    return DataModule.from_config(single_cfg)


# ---------------------------------------------------------------------------
# TestArrayLoader
# ---------------------------------------------------------------------------

class TestArrayLoader:

    @pytest.fixture
    def small_arrays(self):
        rng = np.random.default_rng(0)
        return {
            "X": jnp.array(rng.normal(0, 1, (50, 4)).astype(np.float32)),
            "y": jnp.array(rng.normal(0, 1, (50, 2)).astype(np.float32)),
        }

    def test_yields_dicts(self, small_arrays):
        loader = ArrayLoader(small_arrays, batch_size=16, shuffle=False)
        batch  = next(iter(loader))
        assert isinstance(batch, dict)
        assert "X" in batch and "y" in batch

    def test_correct_batch_shape(self, small_arrays):
        loader = ArrayLoader(small_arrays, batch_size=16, shuffle=False)
        for batch in loader:
            assert batch["X"].shape[1] == 4
            assert batch["y"].shape[1] == 2

    def test_drop_last_true_all_full_batches(self, small_arrays):
        # 50 samples / batch_size 16 = 3 full batches (drop 2 remainder)
        loader  = ArrayLoader(small_arrays, batch_size=16, shuffle=False, drop_last=True)
        batches = list(loader)
        assert len(batches) == 3
        assert all(b["X"].shape[0] == 16 for b in batches)

    def test_drop_last_false_includes_remainder(self, small_arrays):
        # 50 / 16 = 3 full + 1 partial (2 rows)
        loader  = ArrayLoader(small_arrays, batch_size=16, shuffle=False, drop_last=False)
        batches = list(loader)
        assert len(batches) == 4
        assert batches[-1]["X"].shape[0] == 2

    def test_len_drop_last_true(self, small_arrays):
        loader = ArrayLoader(small_arrays, batch_size=16, shuffle=False, drop_last=True)
        assert len(loader) == 3

    def test_len_drop_last_false(self, small_arrays):
        loader = ArrayLoader(small_arrays, batch_size=16, shuffle=False, drop_last=False)
        assert len(loader) == 4

    def test_shuffle_false_fixed_order(self, small_arrays):
        loader  = ArrayLoader(small_arrays, batch_size=16, shuffle=False, drop_last=True)
        first_pass  = [b["X"][0, 0].item() for b in loader]
        second_pass = [b["X"][0, 0].item() for b in loader]
        assert first_pass == second_pass

    def test_shuffle_true_varies_between_passes(self, small_arrays):
        loader = ArrayLoader(small_arrays, batch_size=16, shuffle=True, seed=0, drop_last=True)
        pass1  = [b["X"][0, 0].item() for b in loader]
        pass2  = [b["X"][0, 0].item() for b in loader]
        assert pass1 != pass2  # different epoch → different shuffle

    def test_shuffle_true_covers_all_samples(self, small_arrays):
        # All 48 samples (3 × 16) from drop_last=True must be unique
        loader = ArrayLoader(small_arrays, batch_size=16, shuffle=True, seed=0, drop_last=True)
        seen   = []
        for batch in loader:
            seen.extend(batch["X"][:, 0].tolist())
        assert len(set(seen)) == 48

    def test_seed_reproducibility(self, small_arrays):
        l1 = ArrayLoader(small_arrays, batch_size=16, shuffle=True, seed=42)
        l2 = ArrayLoader(small_arrays, batch_size=16, shuffle=True, seed=42)
        for b1, b2 in zip(l1, l2):
            assert jnp.array_equal(b1["X"], b2["X"])

    def test_different_seeds_different_order(self, small_arrays):
        l1 = ArrayLoader(small_arrays, batch_size=16, shuffle=True, seed=0)
        l2 = ArrayLoader(small_arrays, batch_size=16, shuffle=True, seed=99)
        first_b1 = next(iter(l1))["X"][0, 0].item()
        first_b2 = next(iter(l2))["X"][0, 0].item()
        assert first_b1 != first_b2

    def test_is_reiterable(self, small_arrays):
        loader = ArrayLoader(small_arrays, batch_size=16, shuffle=False, drop_last=True)
        cnt1   = sum(1 for _ in loader)
        cnt2   = sum(1 for _ in loader)
        assert cnt1 == cnt2 == 3


# ---------------------------------------------------------------------------
# TestDatasetRegistry
# ---------------------------------------------------------------------------

class TestDatasetRegistry:

    def test_list_datasets_returns_list(self):
        assert isinstance(list_datasets(), list)

    def test_ibtracs_registered(self):
        assert "IBTRACS" in list_datasets()

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_dataset("IBTRACS")
            def _dup(config):
                pass

    def test_unknown_dataset_raises_on_setup(self, single_cfg):
        bad = {**single_cfg, "dataset": "NOT_A_THING"}
        with pytest.raises(ValueError, match="not registered"):
            DataModule.from_config(bad)

    def test_case_insensitive(self, single_cfg):
        upper = DataModule.from_config({**single_cfg, "dataset": "IBTRACS"})
        lower = DataModule.from_config({**single_cfg, "dataset": "ibtracs"})
        assert upper.train_arrays()["X"].shape == lower.train_arrays()["X"].shape


# ---------------------------------------------------------------------------
# TestApplyNorm
# ---------------------------------------------------------------------------

class TestApplyNorm:

    def _make(self, seed=0):
        rng = np.random.default_rng(seed)
        tr = rng.normal(5.0, 2.0, (100, 3)).astype(np.float32)
        va = rng.normal(5.0, 2.0, (20, 3)).astype(np.float32)
        te = rng.normal(5.0, 2.0, (20, 3)).astype(np.float32)
        return tr, va, te

    def test_standard_train_mean_zero(self):
        tr, va, te = self._make()
        tr_n, _, _, stats = _apply_norm(tr, va, te, "standard")
        assert np.allclose(tr_n.mean(axis=0), 0.0, atol=1e-5)

    def test_standard_train_std_one(self):
        tr, va, te = self._make()
        tr_n, _, _, _ = _apply_norm(tr, va, te, "standard")
        assert np.allclose(tr_n.std(axis=0), 1.0, atol=1e-4)

    def test_standard_val_uses_train_stats_not_own(self):
        rng = np.random.default_rng(7)
        tr = rng.normal(0.0, 1.0, (200, 2)).astype(np.float32)
        va = rng.normal(5.0, 1.0, (50, 2)).astype(np.float32)
        te = rng.normal(5.0, 1.0, (50, 2)).astype(np.float32)
        _, va_n, _, _ = _apply_norm(tr, va, te, "standard")
        # val was drawn from mean=5, train from mean=0 → val_n mean ≈ 5
        assert abs(float(va_n[:, 0].mean())) > 1.0

    def test_minmax_train_range_01(self):
        tr, va, te = self._make()
        tr_n, _, _, stats = _apply_norm(tr, va, te, "minmax")
        assert float(tr_n.min()) >= -1e-6
        assert float(tr_n.max()) <= 1.0 + 1e-6
        assert stats["method"] == "minmax"

    def test_none_passthrough(self):
        tr, va, te = self._make()
        tr_n, va_n, te_n, stats = _apply_norm(tr, va, te, "none")
        assert stats["method"] == "none"
        assert np.array_equal(tr_n, tr)

    def test_nan_in_targets_tolerated(self):
        tr, va, te = self._make()
        tr[::5, 0] = np.nan
        tr_n, _, _, stats = _apply_norm(tr, va, te, "standard")
        assert np.isfinite(stats["mean"][0])
        assert np.isfinite(stats["std"][0])

    def test_all_nan_column_propagates_nan(self):
        # If an entire training column is NaN, np.nanmean returns NaN
        # (with a RuntimeWarning). The normalised column is also all-NaN.
        # Downstream masked losses handle this correctly; this test documents
        # the behaviour so it cannot regress silently.
        tr, va, te = self._make()
        tr[:, 1] = np.nan   # column 1 entirely NaN in training data
        with pytest.warns(RuntimeWarning):
            tr_n, va_n, _, stats = _apply_norm(tr, va, te, "standard")
        assert np.isnan(stats["mean"][1])
        assert np.isnan(stats["std"][1])
        assert np.all(np.isnan(tr_n[:, 1]))
        assert np.all(np.isnan(va_n[:, 1]))


# ---------------------------------------------------------------------------
# TestInvertNorm
# ---------------------------------------------------------------------------

class TestInvertNorm:

    def _round_trip(self, method):
        rng = np.random.default_rng(0)
        tr = rng.uniform(10, 50, (100, 3)).astype(np.float32)
        tr_n, _, _, stats = _apply_norm(tr, tr, tr, method)
        return tr, tr_n, stats

    def test_standard_round_trip(self):
        orig, norm, stats = self._round_trip("standard")
        assert np.allclose(_invert_norm(norm, stats), orig, atol=1e-5)

    def test_minmax_round_trip(self):
        orig, norm, stats = self._round_trip("minmax")
        assert np.allclose(_invert_norm(norm, stats), orig, atol=1e-5)

    def test_none_passthrough(self):
        orig, norm, stats = self._round_trip("none")
        assert np.array_equal(_invert_norm(norm, stats), orig)


# ---------------------------------------------------------------------------
# TestDataModule  (single source)
# ---------------------------------------------------------------------------

class TestDataModule:

    def test_three_splits_exist(self, dm):
        assert dm.train_arrays() is not None
        assert dm.val_arrays()   is not None
        assert dm.test_arrays()  is not None

    def test_splits_have_x_and_y_keys(self, dm):
        for split in [dm.train_arrays(), dm.val_arrays(), dm.test_arrays()]:
            assert set(split.keys()) == {"X", "y"}

    def test_feature_dim(self, dm, single_cfg):
        n_feat = len(single_cfg["feature_cols"])
        for split in [dm.train_arrays(), dm.val_arrays(), dm.test_arrays()]:
            assert split["X"].shape[1] == n_feat

    def test_target_dim(self, dm, single_cfg):
        n_tgt = len(single_cfg["target_cols"])
        for split in [dm.train_arrays(), dm.val_arrays(), dm.test_arrays()]:
            assert split["y"].shape[1] == n_tgt

    def test_all_splits_non_empty(self, dm):
        for split in [dm.train_arrays(), dm.val_arrays(), dm.test_arrays()]:
            assert split["X"].shape[0] > 0

    def test_train_largest(self, dm):
        assert (
            dm.train_arrays()["X"].shape[0]
            > dm.val_arrays()["X"].shape[0]
        )

    def test_returns_jax_arrays(self, dm):
        assert isinstance(dm.train_arrays()["X"], jnp.ndarray)

    def test_standard_norm_train_mean_zero(self, dm):
        X = np.array(dm.train_arrays()["X"])
        assert np.allclose(X.mean(axis=0), 0.0, atol=0.1)

    def test_standard_norm_train_std_one(self, dm):
        X = np.array(dm.train_arrays()["X"])
        assert np.allclose(X.std(axis=0), 1.0, atol=0.1)

    def test_nan_preserved_in_sparse_target(self, dm):
        y = np.array(dm.train_arrays()["y"])
        # USA_PRES (col 1) has NaN in the fixture
        assert not np.all(np.isfinite(y[:, 1]))

    def test_norm_stats_has_feature_and_target(self, dm):
        assert "feature" in dm.norm_stats
        assert "target"  in dm.norm_stats

    def test_denormalise_round_trip(self, dm):
        y_norm = np.array(dm.train_arrays()["y"])
        y_phys = dm.denormalise_targets(y_norm)
        stats  = dm.norm_stats["target"]
        if stats["method"] == "standard":
            y_back = (y_phys - stats["mean"]) / stats["std"]
            assert np.allclose(
                np.nan_to_num(y_back), np.nan_to_num(y_norm), atol=1e-5
            )

    def test_minmax_feature_norm(self, npz, ms_npz):
        cfg = {
            "dataset": "ibtracs", "npz_path": str(npz),
            "multi_storm_path": str(ms_npz),
            "target_cols": ["USA_WIND"], "feature_cols": ["LAT", "LON"],
            "feature_norm": "minmax", "target_norm": "none",
        }
        dm = DataModule.from_config(cfg)
        X  = np.array(dm.train_arrays()["X"])
        assert float(X.min()) >= -1e-6
        assert float(X.max()) <= 1.0 + 1e-6

    def test_none_norm_keeps_physical_scale(self, npz, ms_npz):
        cfg = {
            "dataset": "ibtracs", "npz_path": str(npz),
            "multi_storm_path": str(ms_npz),
            "target_cols": ["USA_WIND"], "feature_cols": ["LAT"],
            "feature_norm": "none", "target_norm": "none",
        }
        dm = DataModule.from_config(cfg)
        X  = np.array(dm.train_arrays()["X"])
        # LAT should be in [-10, 30] as generated by the fixture
        assert float(X.min()) > -15
        assert float(X.max()) <  35

    def test_missing_multi_storm_warns_and_works(self, npz):
        cfg = {
            "dataset": "ibtracs", "npz_path": str(npz),
            "target_cols": ["USA_WIND"], "feature_cols": ["LAT", "LON"],
            "feature_norm": "standard", "target_norm": "standard",
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dm = DataModule.from_config(cfg)
        assert any("multi_storm_path" in str(warning.message) for warning in w)
        assert dm.train_arrays()["X"].shape[0] > 0

    def test_summary_output(self, dm, capsys):
        dm.summary()
        out = capsys.readouterr().out
        assert "DataModule" in out
        assert "train" in out

    # --- loader interface ---

    def test_train_loader_yields_correct_feature_dim(self, dm, single_cfg):
        n_feat = len(single_cfg["feature_cols"])
        for batch in dm.train_loader(batch_size=32):
            assert batch["X"].shape[1] == n_feat
            break

    def test_val_loader_yields_correct_feature_dim(self, dm, single_cfg):
        n_feat = len(single_cfg["feature_cols"])
        for batch in dm.val_loader(batch_size=32):
            assert batch["X"].shape[1] == n_feat
            break

    def test_train_loader_shuffles_by_default(self, dm):
        assert dm.train_loader(batch_size=32)._shuffle is True

    def test_val_loader_no_shuffle_by_default(self, dm):
        assert dm.val_loader(batch_size=32)._shuffle is False

    def test_train_loader_shuffle_false_override(self, dm):
        assert dm.train_loader(batch_size=32, shuffle=False)._shuffle is False

    def test_config_train_shuffle_false(self, npz, ms_npz):
        cfg = {
            "dataset": "ibtracs", "npz_path": str(npz),
            "multi_storm_path": str(ms_npz),
            "target_cols": ["USA_WIND"], "feature_cols": ["LAT", "LON"],
            "feature_norm": "standard", "target_norm": "standard",
            "train_shuffle": False,
        }
        assert DataModule.from_config(cfg).train_loader(batch_size=32)._shuffle is False

    def test_val_loader_drop_last_false_covers_all(self, dm):
        n_val  = dm.val_arrays()["X"].shape[0]
        total  = sum(b["X"].shape[0] for b in dm.val_loader(batch_size=16))
        assert total == n_val

    def test_train_loader_drop_last_true(self, dm):
        loader = dm.train_loader(batch_size=16)
        assert loader._drop_last is True


# ---------------------------------------------------------------------------
# TestDataModuleMulti  (multiple datasets)
# ---------------------------------------------------------------------------

class TestDataModuleMulti:

    @pytest.fixture
    def multi_cfg(self, tmp_path):
        p1 = _make_ibtracs_npz(tmp_path / "a.npz", seed=0)
        p2 = _make_ibtracs_npz(tmp_path / "b.npz", seed=42)
        ms = _make_multi_storm_npz(tmp_path / "ms.npz")
        return {
            "datasets": [
                {"dataset": "ibtracs", "npz_path": str(p1),
                 "multi_storm_path": str(ms)},
                {"dataset": "ibtracs", "npz_path": str(p2),
                 "multi_storm_path": str(ms)},
            ],
            "target_cols":  ["USA_WIND", "USA_PRES"],
            "feature_cols": ["LAT", "LON", "STORM_SPEED", "STORM_DIR"],
            "feature_norm": "standard",
            "target_norm":  "standard",
        }

    def test_combined_train_roughly_double_single(self, multi_cfg, single_cfg):
        dm_multi  = DataModule.from_config(multi_cfg)
        dm_single = DataModule.from_config(single_cfg)
        n_multi  = dm_multi.train_arrays()["X"].shape[0]
        n_single = dm_single.train_arrays()["X"].shape[0]
        assert abs(n_multi - 2 * n_single) < 5

    def test_feature_dim_unchanged(self, multi_cfg):
        dm = DataModule.from_config(multi_cfg)
        assert dm.train_arrays()["X"].shape[1] == len(multi_cfg["feature_cols"])

    def test_target_dim_unchanged(self, multi_cfg):
        dm = DataModule.from_config(multi_cfg)
        assert dm.train_arrays()["y"].shape[1] == len(multi_cfg["target_cols"])

    def test_combined_train_normalised(self, multi_cfg):
        dm = DataModule.from_config(multi_cfg)
        X  = np.array(dm.train_arrays()["X"])
        assert np.allclose(X.mean(axis=0), 0.0, atol=0.1)
        assert np.allclose(X.std(axis=0),  1.0, atol=0.1)

    def test_denormalise_round_trip(self, multi_cfg):
        dm     = DataModule.from_config(multi_cfg)
        y_norm = np.array(dm.train_arrays()["y"])
        y_phys = dm.denormalise_targets(y_norm)
        stats  = dm.norm_stats["target"]
        if stats["method"] == "standard":
            y_back = (y_phys - stats["mean"]) / stats["std"]
            assert np.allclose(
                np.nan_to_num(y_back), np.nan_to_num(y_norm), atol=1e-5
            )

    def test_summary_shows_two_sources(self, multi_cfg, capsys):
        dm = DataModule.from_config(multi_cfg)
        dm.summary()
        out = capsys.readouterr().out
        assert "2 source" in out


# ---------------------------------------------------------------------------
# TestDataModuleInterface
# ---------------------------------------------------------------------------

class TestDataModuleInterface:

    def test_base_datamodule_is_abstract(self):
        import inspect
        assert inspect.isabstract(BaseDataModule)

    def test_datamodule_is_concrete(self, dm):
        assert isinstance(dm, DataModule)
        assert isinstance(dm, BaseDataModule)

    def test_from_config_is_equivalent_to_manual_setup(self, single_cfg):
        dm1 = DataModule.from_config(single_cfg)
        dm2 = DataModule()
        dm2.setup(single_cfg)
        assert dm1.train_arrays()["X"].shape == dm2.train_arrays()["X"].shape

    def test_norm_stats_keys(self, dm):
        stats = dm.norm_stats
        assert "feature" in stats and "target" in stats
        assert "method" in stats["feature"]
        assert "method" in stats["target"]
