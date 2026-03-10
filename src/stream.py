"""Streaming pipeline — all stages in one pass, no intermediate files.

Instead of writing 100+ GB of PNGs to disk, this reads frames from the
video, runs depth estimation, synthesizes stereo, and pipes the result
straight to FFmpeg. Memory usage is roughly 3 frames worth of data at
any given time (current frame, depth map, SBS output).

Tradeoff vs the disk-based pipeline: not restartable. If it crashes at
frame 15,000 you start over. Use the regular pipeline if you need that
safety net, use this one when disk space is tight.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.depth import load_midas
from src.stereo import synthesize_stereo_pair


def run_streaming(
    video_path: str | Path,
    output_path: str | Path | None = None,
    model_type: str = "DPT_Large",
    max_disparity: float = 15.0,
    temporal_alpha: float = 0.3,
    crf: int = 18,
):
    """Convert 2D video to stereoscopic 3D without intermediate files.

    Reads one frame at a time from the input video, runs it through
    depth estimation and stereo synthesis in memory, and pipes the
    SBS result directly to FFmpeg's stdin as raw pixels.

    Args:
        video_path: Input video file
        output_path: Where to write the final SBS video
        model_type: MiDaS model variant
        max_disparity: Maximum pixel shift for stereo effect
        temporal_alpha: Depth temporal smoothing (0=frozen, 1=no smoothing)
        crf: H.264 quality (lower = better)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg not found. Install it:\n"
            "  Mac:   brew install ffmpeg\n"
            "  Linux: sudo apt install ffmpeg"
        )

    if output_path is None:
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_sbs.mp4"
    else:
        output_path = Path(output_path)

    # Open the video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sbs_width = width * 2

    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    print("Streaming mode — no intermediate files written to disk")

    # Load MiDaS
    model, transform, device = load_midas(model_type)

    # Start FFmpeg reading raw BGR frames from stdin.
    # We tell it the pixel format, dimensions, and framerate upfront
    # so it knows how to interpret the raw byte stream.
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        # Input: raw video from pipe
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{sbs_width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]

    # Audio from source video
    if video_path.exists():
        ffmpeg_cmd += ["-i", str(video_path)]

    ffmpeg_cmd += [
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    ]

    if video_path.exists():
        ffmpeg_cmd += ["-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0?", "-shortest"]

    ffmpeg_cmd.append(str(output_path))

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    start = time.time()
    smooth_depth = None
    count = 0

    try:
        with torch.no_grad():
            for _ in tqdm(range(total_frames), desc="Streaming"):
                ret, frame = cap.read()
                if not ret:
                    break

                # -- Depth estimation (single frame) --
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_batch = transform(img_rgb).to(device)
                prediction = model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth = prediction.cpu().numpy().astype(np.float64)

                # Normalize to [0, 1]
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max - depth_min > 0:
                    depth = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth = np.zeros_like(depth)

                # Temporal smoothing
                if smooth_depth is None:
                    smooth_depth = depth
                else:
                    smooth_depth = temporal_alpha * depth + (1.0 - temporal_alpha) * smooth_depth

                depth_u8 = (smooth_depth * 255).astype(np.uint8)

                # -- Stereo synthesis --
                sbs = synthesize_stereo_pair(frame, depth_u8, max_disparity)

                # -- Pipe to FFmpeg --
                # sbs is a contiguous BGR numpy array, tobytes() gives raw pixels
                ffmpeg_proc.stdin.write(sbs.tobytes())
                count += 1

    except BrokenPipeError:
        # FFmpeg exited early — grab its stderr for the error message
        pass
    finally:
        cap.release()
        if ffmpeg_proc.stdin:
            ffmpeg_proc.stdin.close()

    # communicate() drains stdout/stderr fully, preventing the deadlock
    # where FFmpeg blocks trying to write to a full stderr pipe while
    # we block waiting for it to exit. Classic subprocess footgun.
    _, stderr_bytes = ffmpeg_proc.communicate()
    if ffmpeg_proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{stderr_bytes.decode(errors='replace')}")

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nDone! {count} frames in {minutes}m {seconds:.1f}s")
    print(f"Output: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Streaming 2D-to-3D conversion — no intermediate files.",
    )
    parser.add_argument("video_path", help="Input video file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output SBS video path (default: output/<name>_sbs.mp4)",
    )
    parser.add_argument("--model", default="DPT_Large", help="MiDaS model variant")
    parser.add_argument("--max-disparity", type=float, default=15.0, help="Max pixel shift (default: 15.0)")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality (default: 18)")

    args = parser.parse_args()

    run_streaming(
        video_path=args.video_path,
        output_path=args.output,
        model_type=args.model,
        max_disparity=args.max_disparity,
        crf=args.crf,
    )
