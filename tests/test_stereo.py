"""Tests for stereo synthesis — the fun part where flat images learn depth.

synthesize_stereo_pair is a pure function (image + depth → SBS image),
which makes it the most naturally testable module in the pipeline.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.stereo import _process_single_frame, synthesize_all, synthesize_stereo_pair
from tests.conftest import FRAME_H, FRAME_W, NUM_FRAMES


class TestSynthesizeSterePair:
    """Core stereo synthesis function — pure numpy/cv2, no side effects."""

    def _make_frame(self, color=(128, 128, 128)):
        return np.full((FRAME_H, FRAME_W, 3), color, dtype=np.uint8)

    def _make_depth(self, value=128):
        """Uniform depth map — every pixel at the same distance."""
        return np.full((FRAME_H, FRAME_W), value, dtype=np.uint8)

    def _make_gradient_depth(self):
        """Left-to-right gradient: black (far) → white (near)."""
        return np.tile(
            np.linspace(0, 255, FRAME_W, dtype=np.uint8),
            (FRAME_H, 1),
        )

    def test_output_is_double_width(self):
        frame = self._make_frame()
        depth = self._make_depth()

        sbs = synthesize_stereo_pair(frame, depth, max_disparity=10)

        assert sbs.shape == (FRAME_H, FRAME_W * 2, 3)

    def test_left_eye_is_original_frame(self):
        """Left half of SBS output should be the unmodified input."""
        frame = self._make_frame((42, 99, 200))
        depth = self._make_depth()

        sbs = synthesize_stereo_pair(frame, depth, max_disparity=10)
        left_eye = sbs[:, :FRAME_W, :]

        np.testing.assert_array_equal(left_eye, frame)

    def test_zero_disparity_means_identical_eyes(self):
        """With no stereo shift, both eyes should see the same thing."""
        frame = self._make_frame()
        depth = self._make_depth()

        sbs = synthesize_stereo_pair(frame, depth, max_disparity=0)
        left = sbs[:, :FRAME_W, :]
        right = sbs[:, FRAME_W:, :]

        np.testing.assert_array_equal(left, right)

    def test_uniform_depth_shifts_uniformly(self):
        """All-white depth (everything near) should shift every pixel
        by the same amount. The right eye view should look like the
        original shifted right with border replication on the left edge."""
        frame = self._make_frame()
        depth = self._make_depth(255)  # all near

        sbs = synthesize_stereo_pair(frame, depth, max_disparity=5)
        right = sbs[:, FRAME_W:, :]

        # With a solid-color frame, shifting doesn't change anything
        # visually — but the remap should still produce valid output
        assert right.shape == frame.shape
        assert right.dtype == np.uint8

    def test_output_dtype_and_range(self):
        frame = self._make_frame()
        depth = self._make_gradient_depth()

        sbs = synthesize_stereo_pair(frame, depth, max_disparity=15)

        assert sbs.dtype == np.uint8
        assert sbs.min() >= 0
        assert sbs.max() <= 255


class TestProcessSingleFrame:
    """The multiprocessing worker function."""

    def test_writes_output_file(self, synthetic_frames: Path, synthetic_depth_maps: Path, tmp_path: Path):
        output_dir = tmp_path / "stereo"
        output_dir.mkdir()

        frame_path = str(synthetic_frames / "frame_000001.png")
        depth_path = str(synthetic_depth_maps / "frame_000001.png")
        out_path = str(output_dir / "frame_000001.png")

        result = _process_single_frame((frame_path, depth_path, out_path, 15.0))

        assert result is True
        assert Path(out_path).exists()

        # Verify it's actually a double-width SBS image
        img = cv2.imread(out_path)
        assert img.shape[1] == FRAME_W * 2

    def test_returns_false_when_depth_missing(self, synthetic_frames: Path, tmp_path: Path):
        result = _process_single_frame((
            str(synthetic_frames / "frame_000001.png"),
            str(tmp_path / "nonexistent_depth.png"),
            str(tmp_path / "output.png"),
            15.0,
        ))

        assert result is False


class TestSynthesizeAll:
    """End-to-end stereo synthesis across multiple frames."""

    def test_processes_all_frames(self, synthetic_frames: Path, synthetic_depth_maps: Path, tmp_path: Path):
        output_dir = tmp_path / "stereo"

        count = synthesize_all(synthetic_frames, synthetic_depth_maps, output_dir, max_disparity=10, workers=2)

        assert count == NUM_FRAMES
        assert len(list(output_dir.glob("frame_*.png"))) == NUM_FRAMES

    def test_raises_on_empty_frames_dir(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            synthesize_all(empty_dir, tmp_path / "depth", tmp_path / "out")

    def test_skips_frames_without_depth(self, synthetic_frames: Path, tmp_path: Path):
        """If depth maps are missing for some frames, skip them gracefully."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        # Only create depth for frame 1 of 3
        gradient = np.tile(np.linspace(0, 255, FRAME_W, dtype=np.uint8), (FRAME_H, 1))
        cv2.imwrite(str(depth_dir / "frame_000001.png"), gradient)

        output_dir = tmp_path / "stereo"
        count = synthesize_all(synthetic_frames, depth_dir, output_dir, workers=1)

        assert count == 1
