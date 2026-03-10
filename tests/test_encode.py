"""Tests for video encoding — the FFmpeg wrapper stage."""

import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.encode import encode_video
from tests.conftest import NUM_FRAMES


def _mock_popen(returncode=0, stderr_lines=None):
    """Create a mock Popen that simulates FFmpeg's stderr output.

    FFmpeg writes progress lines like 'frame=  10 fps=30.0 ...' to stderr.
    We feed these to the progress bar parser so it doesn't hang.
    """
    if stderr_lines is None:
        stderr_lines = [f"frame=  {NUM_FRAMES} fps=30.0\n"]

    mock_proc = MagicMock()
    mock_proc.stderr = iter(stderr_lines)
    mock_proc.stdout = iter([])
    mock_proc.wait.return_value = returncode
    mock_proc.returncode = returncode
    return mock_proc


class TestEncodeValidation:
    """Input validation — these don't need ffmpeg installed."""

    def test_raises_without_ffmpeg(self, tmp_path: Path):
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                encode_video(tmp_path, tmp_path / "out.mp4")

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_raises_on_empty_frames_dir(self, _mock, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No frames found"):
            encode_video(empty_dir, tmp_path / "out.mp4")


class TestEncodeCommand:
    """Verify the ffmpeg command is built correctly without actually running it."""

    @patch("src.encode.subprocess.Popen")
    @patch("src.encode.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_basic_command_structure(self, _which, mock_popen, synthetic_stereo_frames: Path, tmp_path: Path):
        mock_popen.return_value = _mock_popen()
        output_path = tmp_path / "out.mp4"
        output_path.write_bytes(b"fake video data")

        encode_video(synthetic_stereo_frames, output_path, fps=30.0, crf=20)

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-framerate" in cmd
        assert "30.0" in cmd[cmd.index("-framerate") + 1]
        assert "-crf" in cmd
        assert "20" in cmd[cmd.index("-crf") + 1]

    @patch("src.encode.subprocess.Popen")
    @patch("src.encode.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_audio_source_included(self, _which, mock_popen, synthetic_stereo_frames: Path, tmp_path: Path):
        mock_popen.return_value = _mock_popen()
        output_path = tmp_path / "out.mp4"
        output_path.write_bytes(b"fake")

        audio = tmp_path / "source.mp4"
        audio.write_bytes(b"fake audio")

        encode_video(synthetic_stereo_frames, output_path, audio_source=audio)

        cmd = mock_popen.call_args[0][0]
        assert str(audio) in cmd
        assert "-c:a" in cmd
        assert "copy" in cmd

    @patch("src.encode.subprocess.Popen")
    @patch("src.encode.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_raises_on_ffmpeg_failure(self, _which, mock_popen, synthetic_stereo_frames: Path, tmp_path: Path):
        mock_popen.return_value = _mock_popen(returncode=1, stderr_lines=["Error: something went wrong\n"])

        with pytest.raises(RuntimeError, match="FFmpeg failed"):
            encode_video(synthetic_stereo_frames, tmp_path / "out.mp4")


@pytest.mark.integration
class TestEncodeIntegration:
    """Actually encode a video — requires ffmpeg installed."""

    @pytest.fixture(autouse=True)
    def _require_ffmpeg(self):
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")

    def test_produces_video_file(self, synthetic_stereo_frames: Path, tmp_path: Path):
        output_path = tmp_path / "test_output.mp4"

        result = encode_video(synthetic_stereo_frames, output_path, fps=1.0, crf=28)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_with_audio_source(self, synthetic_stereo_frames: Path, synthetic_video: Path, tmp_path: Path):
        """Use the synthetic video as an audio source (it has no audio,
        but ffmpeg should handle that gracefully with the '?' stream selector)."""
        output_path = tmp_path / "test_with_audio.mp4"

        result = encode_video(
            synthetic_stereo_frames, output_path,
            fps=1.0, crf=28, audio_source=synthetic_video,
        )

        assert output_path.exists()
