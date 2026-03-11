"""Full 2D→3D pipeline orchestrator.

Runs all stages in sequence: extract → depth → stereo → encode.
Each stage reads from and writes to disk, so you can re-run individual
stages without starting over if something goes sideways.
"""

import shutil
import sys
import time
from pathlib import Path

from src.depth import estimate_depth
from src.encode import encode_video
from src.extract import extract_frames
from src.stereo import synthesize_all


def run_pipeline(
    video_path: str | Path,
    output_path: str | Path | None = None,
    work_dir: str | Path = "work",
    model_type: str = "DPT_Large",
    max_disparity: float = 15.0,
    fps: float | None = None,
    crf: int = 18,
    quest: bool = False,
):
    """Convert a 2D video to stereoscopic 3D side-by-side format.

    Args:
        video_path: Input video file
        output_path: Where to write the final SBS video
        work_dir: Intermediate files directory
        model_type: MiDaS model variant (DPT_Large, DPT_Hybrid, MiDaS_small)
        max_disparity: Maximum pixel shift for stereo effect (12-18 recommended)
        fps: Output framerate (None = match input video)
        crf: H.264 quality (lower = better, 18 = visually lossless)
        quest: Optimize encoding for Meta Quest 2/3 headsets
    """
    video_path = Path(video_path)
    work_dir = Path(work_dir)

    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    # Default output: output/<original_name>.mp4
    if output_path is None:
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_sbs.mp4"
    else:
        output_path = Path(output_path)

    frames_dir = work_dir / "frames"
    depth_dir = work_dir / "depth"
    stereo_dir = work_dir / "stereo_frames"

    start = time.time()

    # Detect input FPS if not specified
    if fps is None:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        print(f"Detected input FPS: {fps:.1f}")

    # Stage 1: Extract frames (handles its own resume logic)
    print("\n=== Stage 1/4: Frame Extraction ===")
    extract_frames(video_path, frames_dir)

    # Wipe downstream stages — depth and stereo depend on extracted frames,
    # so stale outputs from a different run would corrupt the final video
    for d in (depth_dir, stereo_dir):
        if d.exists():
            shutil.rmtree(d)

    # Stage 2: Depth estimation
    print("\n=== Stage 2/4: Depth Estimation ===")
    estimate_depth(frames_dir, depth_dir, model_type)

    # Stage 3: Stereo synthesis
    print("\n=== Stage 3/4: Stereo Synthesis ===")
    synthesize_all(frames_dir, depth_dir, stereo_dir, max_disparity)

    # Stage 4: Video encoding
    print("\n=== Stage 4/4: Video Encoding ===")
    encode_video(stereo_dir, output_path, fps, crf, audio_source=video_path, quest=quest)

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"\nDone! Total time: {minutes}m {seconds:.1f}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert 2D video to stereoscopic 3D side-by-side format.",
    )
    parser.add_argument("video_path", help="Input video file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output SBS video path (default: output/<name>_sbs.mp4)",
    )
    parser.add_argument(
        "-w", "--work-dir",
        default="work",
        help="Working directory for intermediate files (default: work)",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Streaming mode — no intermediate files, much less disk space",
    )
    parser.add_argument(
        "--quest", action="store_true",
        help="Optimize encoding for Meta Quest 2/3 headsets",
    )

    args = parser.parse_args()

    if args.stream:
        from src.stream import run_streaming
        run_streaming(
            video_path=args.video_path,
            output_path=args.output,
        )
    else:
        run_pipeline(
            video_path=args.video_path,
            output_path=args.output,
            work_dir=args.work_dir,
            quest=args.quest,
        )
