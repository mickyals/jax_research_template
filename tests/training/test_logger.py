"""
Tests for training/logger.py.

Only NullLogger is tested end-to-end (no external deps required).
WandbLogger and TensorBoardLogger are tested for interface completeness
only — their backends are soft dependencies that may not be installed.
"""

import json
import inspect
from pathlib import Path

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt

from training.logger import (
    BaseLogger,
    NullLogger,
    WandbLogger,
    TensorBoardLogger,
    create_logger,
    _to_hwc_uint8,
    _save_hparams_json,
)

_REQUIRED_METHODS = [
    "log_metrics",
    "log_hyperparams",
    "log_figure",
    "log_image",
    "log_histogram",
    "log_artifact",
    "finalize",
    "log_dir",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def null_logger(tmp_path):
    return NullLogger(log_dir=tmp_path / "null_run", verbose=False)


@pytest.fixture
def null_logger_verbose(tmp_path):
    return NullLogger(log_dir=tmp_path / "null_verbose", verbose=True)


def _make_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


# ---------------------------------------------------------------------------
# Interface completeness
# ---------------------------------------------------------------------------

class TestInterface:

    def test_base_logger_is_abstract(self):
        assert inspect.isabstract(BaseLogger)

    @pytest.mark.parametrize("cls", [NullLogger, WandbLogger, TensorBoardLogger])
    def test_all_methods_present(self, cls, tmp_path):
        for method in _REQUIRED_METHODS:
            assert hasattr(cls, method), f"{cls.__name__} missing '{method}'"

    def test_null_logger_is_base_logger(self, null_logger):
        assert isinstance(null_logger, BaseLogger)


# ---------------------------------------------------------------------------
# _to_hwc_uint8
# ---------------------------------------------------------------------------

class TestToHwcUint8:

    def test_float32_01_converted(self):
        img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        out = _to_hwc_uint8(img)
        assert out.dtype == np.uint8
        assert out.min() == 127 or out.min() == 128  # 0.5 * 255

    def test_uint8_passthrough(self):
        img = np.ones((4, 4, 3), dtype=np.uint8) * 200
        out = _to_hwc_uint8(img)
        assert out.dtype == np.uint8
        assert (out == 200).all()

    def test_grayscale_hw_gets_channel_dim(self):
        img = np.ones((8, 8), dtype=np.uint8)
        out = _to_hwc_uint8(img)
        assert out.shape == (8, 8, 1)

    def test_float_out_of_range_clipped(self):
        img = np.array([[[500.0, -10.0, 128.0]]], dtype=np.float32)
        out = _to_hwc_uint8(img)
        assert int(out[0, 0, 0]) == 255
        assert int(out[0, 0, 1]) == 0
        assert int(out[0, 0, 2]) == 128


# ---------------------------------------------------------------------------
# _save_hparams_json
# ---------------------------------------------------------------------------

class TestSaveHparamsJson:

    def test_writes_file(self, tmp_path):
        _save_hparams_json(tmp_path, {"lr": 1e-3, "n_layers": 4})
        assert (tmp_path / "hparams.json").exists()

    def test_content_correct(self, tmp_path):
        hparams = {"lr": 1e-3, "model": "SIREN"}
        _save_hparams_json(tmp_path, hparams)
        with open(tmp_path / "hparams.json") as f:
            loaded = json.load(f)
        assert loaded["lr"] == pytest.approx(1e-3)
        assert loaded["model"] == "SIREN"

    def test_does_not_overwrite(self, tmp_path):
        _save_hparams_json(tmp_path, {"lr": 1e-3})
        _save_hparams_json(tmp_path, {"lr": 99.0})  # second call should be a no-op
        with open(tmp_path / "hparams.json") as f:
            loaded = json.load(f)
        assert loaded["lr"] == pytest.approx(1e-3)

    def test_creates_directory(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        _save_hparams_json(deep, {"x": 1})
        assert (deep / "hparams.json").exists()


# ---------------------------------------------------------------------------
# NullLogger — log_dir and init
# ---------------------------------------------------------------------------

class TestNullLoggerInit:

    def test_log_dir_created(self, tmp_path):
        log_dir = tmp_path / "run"
        logger = NullLogger(log_dir=log_dir)
        assert log_dir.exists()

    def test_log_dir_property(self, null_logger, tmp_path):
        assert null_logger.log_dir == tmp_path / "null_run"

    def test_config_written_at_init(self, tmp_path):
        logger = NullLogger(log_dir=tmp_path / "r", config={"lr": 0.001})
        assert (tmp_path / "r" / "hparams.json").exists()


# ---------------------------------------------------------------------------
# NullLogger — log_metrics
# ---------------------------------------------------------------------------

class TestNullLoggerMetrics:

    def test_no_error(self, null_logger):
        null_logger.log_metrics({"train/loss": 0.5, "val/loss": 0.6}, step=1)

    def test_verbose_prints(self, null_logger_verbose, capsys):
        null_logger_verbose.log_metrics({"loss": 0.123}, step=5)
        out = capsys.readouterr().out
        assert "loss" in out
        assert "0.1230" in out

    def test_accepts_numpy_scalars(self, null_logger):
        null_logger.log_metrics({"loss": np.float32(0.5)}, step=1)

    def test_accepts_zero_step(self, null_logger):
        null_logger.log_metrics({"loss": 1.0}, step=0)


# ---------------------------------------------------------------------------
# NullLogger — log_hyperparams
# ---------------------------------------------------------------------------

class TestNullLoggerHparams:

    def test_writes_json(self, null_logger):
        null_logger.log_hyperparams({"lr": 1e-3, "n_layers": 5})
        assert (null_logger.log_dir / "hparams.json").exists()

    def test_content_preserved(self, null_logger):
        null_logger.log_hyperparams({"omega": 30, "model": "SIREN"})
        with open(null_logger.log_dir / "hparams.json") as f:
            d = json.load(f)
        assert d["omega"] == 30
        assert d["model"] == "SIREN"


# ---------------------------------------------------------------------------
# NullLogger — log_figure
# ---------------------------------------------------------------------------

class TestNullLoggerFigure:

    def test_saves_png(self, null_logger):
        null_logger.log_figure("val/field", _make_figure(), step=1)
        pngs = list((null_logger.log_dir / "figures").glob("*.png"))
        assert len(pngs) == 1

    def test_filename_contains_tag_and_step(self, null_logger):
        null_logger.log_figure("val/wind_field", _make_figure(), step=42)
        pngs = list((null_logger.log_dir / "figures").glob("*.png"))
        name = pngs[0].name
        assert "val_wind_field" in name
        assert "000042" in name

    def test_multiple_figures_saved(self, null_logger):
        for i in range(3):
            null_logger.log_figure("train/fig", _make_figure(), step=i)
        pngs = list((null_logger.log_dir / "figures").glob("*.png"))
        assert len(pngs) == 3

    def test_verbose_prints(self, null_logger_verbose, capsys):
        null_logger_verbose.log_figure("test/fig", _make_figure(), step=1)
        out = capsys.readouterr().out
        assert "figure" in out.lower()


# ---------------------------------------------------------------------------
# NullLogger — log_image
# ---------------------------------------------------------------------------

class TestNullLoggerImage:

    def test_float32_image(self, null_logger):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        null_logger.log_image("val/img", img, step=1)
        # no error = pass; file saving requires Pillow (optional)

    def test_uint8_image(self, null_logger):
        img = np.ones((16, 16, 3), dtype=np.uint8) * 128
        null_logger.log_image("val/img", img, step=1)

    def test_grayscale_image(self, null_logger):
        img = np.ones((16, 16), dtype=np.uint8)
        null_logger.log_image("val/grey", img, step=1)

    def test_verbose_prints(self, null_logger_verbose, capsys):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        null_logger_verbose.log_image("val/img", img, step=5)
        out = capsys.readouterr().out
        assert "image" in out.lower()


# ---------------------------------------------------------------------------
# NullLogger — log_histogram
# ---------------------------------------------------------------------------

class TestNullLoggerHistogram:

    def test_no_error(self, null_logger):
        null_logger.log_histogram("params/w0", np.random.randn(256), step=1)

    def test_verbose_prints_stats(self, null_logger_verbose, capsys):
        null_logger_verbose.log_histogram("w", np.ones(10) * 2.0, step=1)
        out = capsys.readouterr().out
        assert "mean" in out
        assert "std" in out

    def test_2d_array_flattened(self, null_logger):
        null_logger.log_histogram("w", np.random.randn(16, 16), step=1)


# ---------------------------------------------------------------------------
# NullLogger — finalize
# ---------------------------------------------------------------------------

class TestNullLoggerFinalize:

    def test_success(self, null_logger):
        null_logger.finalize("success")

    def test_failed(self, null_logger):
        null_logger.finalize("failed")

    def test_verbose_prints(self, null_logger_verbose, capsys):
        null_logger_verbose.finalize("success")
        out = capsys.readouterr().out
        assert "success" in out


# ---------------------------------------------------------------------------
# create_logger factory
# ---------------------------------------------------------------------------

class TestCreateLogger:

    def test_null_backend(self, tmp_path):
        logger = create_logger("null", log_dir=tmp_path / "r")
        assert isinstance(logger, NullLogger)

    def test_none_backend(self, tmp_path):
        logger = create_logger("none", log_dir=tmp_path / "r")
        assert isinstance(logger, NullLogger)

    def test_case_insensitive(self, tmp_path):
        logger = create_logger("NULL", log_dir=tmp_path / "r")
        assert isinstance(logger, NullLogger)

    def test_config_forwarded(self, tmp_path):
        logger = create_logger("null", log_dir=tmp_path / "r",
                               config={"lr": 1e-3})
        assert (tmp_path / "r" / "hparams.json").exists()

    def test_unknown_backend_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_logger("fakebackend", log_dir=tmp_path)

    def test_tensorboard_without_log_dir_raises(self):
        with pytest.raises(ValueError, match="log_dir is required"):
            create_logger("tensorboard")

    def test_verbose_kwarg_forwarded(self, tmp_path, capsys):
        logger = create_logger("null", log_dir=tmp_path / "r", verbose=True)
        logger.log_metrics({"loss": 0.5}, step=1)
        out = capsys.readouterr().out
        assert "loss" in out


# ---------------------------------------------------------------------------
# log_artifact
# ---------------------------------------------------------------------------

class TestLogArtifact:

    def test_null_silent_by_default(self, null_logger, tmp_path, capsys):
        null_logger.log_artifact("trace", tmp_path)
        assert capsys.readouterr().out == ""

    def test_null_verbose_prints_path(self, null_logger_verbose, tmp_path, capsys):
        null_logger_verbose.log_artifact("trace", tmp_path / "profile")
        out = capsys.readouterr().out
        assert "trace" in out and "profile" in out

    def test_wandb_uploads_dir_as_artifact(self, tmp_path):
        from unittest.mock import MagicMock
        # Build a WandbLogger without running __init__ (no wandb account
        # in CI) and inject mocks for the wandb module + run.
        logger = object.__new__(WandbLogger)
        logger._wandb = MagicMock()
        logger._run   = MagicMock()
        artifact_dir  = tmp_path / "profile"
        artifact_dir.mkdir()
        (artifact_dir / "trace.pb").write_bytes(b"x")

        logger.log_artifact("profile-trace", artifact_dir,
                            artifact_type="profile")

        logger._wandb.Artifact.assert_called_once_with(
            "profile-trace", type="profile")
        artifact = logger._wandb.Artifact.return_value
        artifact.add_dir.assert_called_once_with(str(artifact_dir))
        artifact.add_file.assert_not_called()
        logger._run.log_artifact.assert_called_once_with(artifact)

    def test_wandb_uploads_single_file(self, tmp_path):
        from unittest.mock import MagicMock
        logger = object.__new__(WandbLogger)
        logger._wandb = MagicMock()
        logger._run   = MagicMock()
        f = tmp_path / "manifest.json"
        f.write_text("{}")

        logger.log_artifact("manifest", f, artifact_type="manifest")

        artifact = logger._wandb.Artifact.return_value
        artifact.add_file.assert_called_once_with(str(f))
        artifact.add_dir.assert_not_called()
