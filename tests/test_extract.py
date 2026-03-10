"""Tests for frame extraction — the boring-but-essential first stage."""

from pathlib import Path
from unittest.mock import patch

import cv2
import pytest

from src.extract import (
    _find_last_frame,
    _load_metadata,
    _save_metadata,
    _video_metadata,
    extract_frames,
)
from tests.conftest import FRAME_H, FRAME_W, NUM_FRAMES


class TestVideoMetadata:
    def test_returns_correct_fields(self, synthetic_video: Path):
        cap = cv2.VideoCapture(str(synthetic_video))
        meta = _video_metadata(cap)
        cap.release()

        assert meta["width"] == FRAME_W
        assert meta["height"] == FRAME_H
        assert meta["total_frames"] == NUM_FRAMES
        assert meta["fps"] > 0


class TestMetadataRoundTrip:
    def test_save_and_load(self, tmp_path: Path):
        meta = {"width": 100, "height": 50, "fps": 24.0, "total_frames": 42}
        _save_metadata(tmp_path, Path("/fake/video.mp4"), meta)

        loaded = _load_metadata(tmp_path)
        assert loaded is not None
        assert loaded["width"] == 100
        assert loaded["total_frames"] == 42
        assert "source_video" in loaded

    def test_load_missing_returns_none(self, tmp_path: Path):
        assert _load_metadata(tmp_path) is None

    def test_load_corrupt_json_returns_none(self, tmp_path: Path):
        (tmp_path / ".extract_meta.json").write_text("not valid json {{{")
        assert _load_metadata(tmp_path) is None


class TestFindLastFrame:
    def test_empty_dir_returns_zero(self, tmp_path: Path):
        assert _find_last_frame(tmp_path) == 0

    def test_finds_highest_numbered_frame(self, tmp_path: Path):
        for i in (1, 2, 5, 10):
            (tmp_path / f"frame_{i:06d}.png").touch()

        assert _find_last_frame(tmp_path) == 10

    def test_ignores_non_frame_files(self, tmp_path: Path):
        (tmp_path / "frame_000003.png").touch()
        (tmp_path / "something_else.png").touch()
        (tmp_path / ".extract_meta.json").touch()

        assert _find_last_frame(tmp_path) == 3


class TestExtractFrames:
    def test_extracts_all_frames(self, synthetic_video: Path, tmp_path: Path):
        output_dir = tmp_path / "frames"

        with patch("src.extract._check_resume", return_value=None):
            count = extract_frames(synthetic_video, output_dir)

        assert count == NUM_FRAMES
        assert len(list(output_dir.glob("frame_*.png"))) == NUM_FRAMES

    def test_frames_are_numbered_sequentially(self, synthetic_video: Path, tmp_path: Path):
        output_dir = tmp_path / "frames"

        with patch("src.extract._check_resume", return_value=None):
            extract_frames(synthetic_video, output_dir)

        for i in range(1, NUM_FRAMES + 1):
            assert (output_dir / f"frame_{i:06d}.png").exists()

    def test_saves_metadata_file(self, synthetic_video: Path, tmp_path: Path):
        output_dir = tmp_path / "frames"

        with patch("src.extract._check_resume", return_value=None):
            extract_frames(synthetic_video, output_dir)

        meta = _load_metadata(output_dir)
        assert meta is not None
        assert meta["width"] == FRAME_W

    def test_skips_when_fully_extracted(self, synthetic_video: Path, tmp_path: Path):
        """If _check_resume says all frames exist, don't re-extract."""
        output_dir = tmp_path / "frames"
        output_dir.mkdir()

        with patch("src.extract._check_resume", return_value=NUM_FRAMES):
            count = extract_frames(synthetic_video, output_dir)

        assert count == NUM_FRAMES
        # No actual frames written — we just returned early
        assert len(list(output_dir.glob("frame_*.png"))) == 0

    def test_raises_on_missing_video(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="Cannot open video"), \
             patch("src.extract._check_resume", return_value=None):
            extract_frames(tmp_path / "nonexistent.mp4", tmp_path / "frames")
