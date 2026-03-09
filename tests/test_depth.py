"""Tests for depth estimation — the GPU-heavy stage, tested without a GPU.

The real MiDaS model is ~1GB and requires downloading weights, so we
mock it with a tiny fake that returns random depth predictions. The
actual neural network accuracy isn't our problem to test — Intel already
did that. We're testing our pipeline code: batching, normalization,
temporal smoothing, and file I/O.
"""

import cv2
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.depth import estimate_depth
from tests.conftest import FRAME_W, FRAME_H, NUM_FRAMES


class FakeMiDaS(torch.nn.Module):
    """Pretend to be MiDaS — return random depth for any input.

    The real model outputs shape [batch, 384, 384]. We match that so
    the interpolation resize in estimate_depth actually exercises real code.
    """
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.rand(batch_size, 384, 384)


def fake_transform(img):
    """Mimic the MiDaS transform: image → [1, 3, 384, 384] tensor."""
    return torch.rand(1, 3, 384, 384)


def mock_load_midas(model_type="DPT_Large"):
    """Stand-in for load_midas that skips the ~1GB model download."""
    return FakeMiDaS(), fake_transform, torch.device("cpu")


class TestEstimateDepth:
    @patch("src.depth.load_midas", side_effect=mock_load_midas)
    def test_produces_depth_maps(self, _mock, synthetic_frames: Path, tmp_path: Path):
        depth_dir = tmp_path / "depth"

        count = estimate_depth(synthetic_frames, depth_dir, batch_size=2)

        assert count == NUM_FRAMES
        assert len(list(depth_dir.glob("frame_*.png"))) == NUM_FRAMES

    @patch("src.depth.load_midas", side_effect=mock_load_midas)
    def test_depth_maps_are_grayscale(self, _mock, synthetic_frames: Path, tmp_path: Path):
        depth_dir = tmp_path / "depth"
        estimate_depth(synthetic_frames, depth_dir, batch_size=2)

        for depth_file in depth_dir.glob("frame_*.png"):
            img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            # Grayscale PNGs are 2D (h, w), not 3D (h, w, channels)
            assert img.ndim == 2, f"{depth_file.name} is not grayscale"
            assert img.dtype == np.uint8

    @patch("src.depth.load_midas", side_effect=mock_load_midas)
    def test_depth_values_span_full_range(self, _mock, synthetic_frames: Path, tmp_path: Path):
        """After normalization, depth maps should use most of [0, 255].

        We can't guarantee exact 0-255 because temporal smoothing
        blends frames, but they shouldn't all be the same flat value.
        """
        depth_dir = tmp_path / "depth"
        estimate_depth(synthetic_frames, depth_dir, batch_size=4)

        for depth_file in depth_dir.glob("frame_*.png"):
            img = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
            # Should have some variation, not a flat gray slab
            assert img.std() > 1, f"{depth_file.name} has suspiciously low variance"

    @patch("src.depth.load_midas", side_effect=mock_load_midas)
    def test_batch_size_one(self, _mock, synthetic_frames: Path, tmp_path: Path):
        """batch_size=1 is the degenerate case — should still work."""
        depth_dir = tmp_path / "depth"
        count = estimate_depth(synthetic_frames, depth_dir, batch_size=1)
        assert count == NUM_FRAMES

    @patch("src.depth.load_midas", side_effect=mock_load_midas)
    def test_batch_size_larger_than_frames(self, _mock, synthetic_frames: Path, tmp_path: Path):
        """batch_size > total frames shouldn't crash — just one batch."""
        depth_dir = tmp_path / "depth"
        count = estimate_depth(synthetic_frames, depth_dir, batch_size=100)
        assert count == NUM_FRAMES

    def test_raises_on_empty_dir(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            estimate_depth(empty_dir, tmp_path / "depth")


class TestTemporalSmoothing:
    """Verify the exponential moving average behaves correctly."""

    @patch("src.depth.load_midas", side_effect=mock_load_midas)
    def test_alpha_one_means_no_smoothing(self, _mock, synthetic_frames: Path, tmp_path: Path):
        """With alpha=1.0, each frame's depth should be independent
        (no blending with previous frames)."""
        depth_dir = tmp_path / "depth"
        estimate_depth(synthetic_frames, depth_dir, temporal_alpha=1.0, batch_size=2)

        # All frames should exist and be valid
        assert len(list(depth_dir.glob("frame_*.png"))) == NUM_FRAMES

    @patch("src.depth.load_midas", side_effect=mock_load_midas)
    def test_alpha_zero_freezes_first_frame(self, _mock, synthetic_frames: Path, tmp_path: Path):
        """With alpha=0.0, the smoothed depth never updates past the
        first frame — all outputs should be identical."""
        depth_dir = tmp_path / "depth"
        estimate_depth(synthetic_frames, depth_dir, temporal_alpha=0.0, batch_size=2)

        images = []
        for f in sorted(depth_dir.glob("frame_*.png")):
            images.append(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))

        # All depth maps should be identical (frozen at frame 1's depth)
        for img in images[1:]:
            np.testing.assert_array_equal(images[0], img)
