"""Shared test fixtures for the stereo3d pipeline.

Generates tiny synthetic test data so we never need real video files
or model weights in the test suite. Everything is 16x16 pixels and
3 frames — just enough to exercise the code paths without waiting
around for actual depth estimation.
"""

import cv2
import numpy as np
import pytest
from pathlib import Path


# -- Dimensions for synthetic test data --
# Small enough to be instant, large enough that OpenCV doesn't choke
FRAME_W = 16
FRAME_H = 16
NUM_FRAMES = 3
TEST_FPS = 1.0


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a tiny AVI video with 3 distinct solid-color frames.

    Red, green, blue — so tests can verify frames are distinct after
    extraction. Uses MJPG codec because it works everywhere, even
    headless Linux CI with no GPU.
    """
    video_path = tmp_path / "test_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, TEST_FPS, (FRAME_W, FRAME_H))

    colors = [
        (0, 0, 255),    # red (BGR)
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
    ]
    for color in colors:
        frame = np.full((FRAME_H, FRAME_W, 3), color, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def synthetic_frames(tmp_path: Path) -> Path:
    """Write 3 numbered PNG frames with known pixel values.

    Each frame is a different solid color, matching what synthetic_video
    would produce after extraction.
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    colors = [
        (0, 0, 255),    # red
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
    ]
    for i, color in enumerate(colors, 1):
        frame = np.full((FRAME_H, FRAME_W, 3), color, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), frame)

    return frames_dir


@pytest.fixture
def synthetic_depth_maps(tmp_path: Path) -> Path:
    """Write 3 grayscale depth maps with a left-to-right gradient.

    Left edge = black (far), right edge = white (near). This produces
    predictable stereo shift behavior: right side shifts more than left.
    """
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()

    gradient = np.tile(
        np.linspace(0, 255, FRAME_W, dtype=np.uint8),
        (FRAME_H, 1),
    )

    for i in range(1, NUM_FRAMES + 1):
        cv2.imwrite(str(depth_dir / f"frame_{i:06d}.png"), gradient)

    return depth_dir


@pytest.fixture
def synthetic_stereo_frames(tmp_path: Path) -> Path:
    """Write 3 side-by-side stereo frames for encode tests.

    Double width (left eye | right eye), each half a solid color.
    """
    stereo_dir = tmp_path / "stereo_frames"
    stereo_dir.mkdir()

    for i in range(1, NUM_FRAMES + 1):
        left = np.full((FRAME_H, FRAME_W, 3), (100, 100, 100), dtype=np.uint8)
        right = np.full((FRAME_H, FRAME_W, 3), (200, 200, 200), dtype=np.uint8)
        sbs = np.hstack([left, right])
        cv2.imwrite(str(stereo_dir / f"frame_{i:06d}.png"), sbs)

    return stereo_dir
