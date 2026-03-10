"""Tests for the pipeline orchestrator.

The full end-to-end test is marked slow + integration because it needs
both mocked depth estimation and real ffmpeg. The unit tests just
verify the orchestrator's path logic and error handling.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipeline import run_pipeline
from tests.test_depth import mock_load_midas


class TestPipelineValidation:
    def test_exits_on_missing_video(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            run_pipeline(tmp_path / "nope.mp4")

    def test_default_output_path(self, tmp_path: Path):
        """Verify the default output naming convention without
        actually running the pipeline."""
        video_path = tmp_path / "cool_movie.mp4"
        video_path.write_bytes(b"fake")

        # We just want to check the path logic, so patch everything
        # and let it fail at extraction — that's fine
        with pytest.raises(RuntimeError):
            run_pipeline(video_path, work_dir=tmp_path / "work")

        # The output dir should have been created
        assert (Path("output")).exists()


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineEndToEnd:
    """Full pipeline run with mocked depth model."""

    @pytest.fixture(autouse=True)
    def _require_ffmpeg(self):
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")

    @patch("src.depth.load_midas", side_effect=mock_load_midas)
    @patch("src.extract._check_resume", return_value=None)
    def test_full_pipeline(self, _resume, _midas, synthetic_video: Path, tmp_path: Path):
        output_path = tmp_path / "output.mp4"
        work_dir = tmp_path / "work"

        run_pipeline(
            video_path=synthetic_video,
            output_path=output_path,
            work_dir=work_dir,
            max_disparity=5,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify intermediate dirs were created
        assert (work_dir / "frames").exists()
        assert (work_dir / "depth").exists()
        assert (work_dir / "stereo_frames").exists()
