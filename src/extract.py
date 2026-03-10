"""Frame extraction from video using OpenCV."""

import json
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def _video_metadata(cap: cv2.VideoCapture) -> dict:
    """Grab the stats we need to identify a video across runs."""
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def _save_metadata(output_dir: Path, video_path: Path, meta: dict) -> None:
    """Stash video metadata so future runs can check if frames match."""
    info = {"source_video": str(video_path.resolve()), **meta}
    (output_dir / ".extract_meta.json").write_text(json.dumps(info, indent=2))


def _load_metadata(output_dir: Path) -> dict | None:
    """Load previous extraction metadata, or None if it doesn't exist."""
    meta_path = output_dir / ".extract_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _find_last_frame(output_dir: Path) -> int:
    """Find the highest frame number already extracted.

    Returns 0 if no frames exist. Checks actual file existence rather
    than trusting the count — handles partial writes from crashes.
    """
    existing = sorted(output_dir.glob("frame_*.png"))
    if not existing:
        return 0
    # frame_000042.png → 42
    return int(existing[-1].stem.split("_")[1])


def _check_resume(output_dir: Path, video_path: Path, meta: dict) -> int | None:
    """Check if we can resume a previous extraction.

    Returns the frame number to resume from, or None to start fresh.
    Asks the user interactively if resumable frames are found.
    """
    prev_meta = _load_metadata(output_dir)
    if prev_meta is None:
        return None

    # Different video or different properties → can't resume
    if (
        prev_meta.get("width") != meta["width"]
        or prev_meta.get("height") != meta["height"]
        or prev_meta.get("total_frames") != meta["total_frames"]
    ):
        return None

    last_frame = _find_last_frame(output_dir)
    if last_frame == 0:
        return None

    if last_frame >= meta["total_frames"]:
        # Already fully extracted — nothing to do
        print(f"All {last_frame} frames already extracted in {output_dir}")
        return last_frame

    pct = (last_frame / meta["total_frames"]) * 100
    print(f"\nFound {last_frame}/{meta['total_frames']} frames ({pct:.0f}%) from a previous run.")
    print(f"  [r] Resume from frame {last_frame + 1}")
    print("  [s] Start over (delete existing frames)")

    while True:
        choice = input("Choice [r/s]: ").strip().lower()
        if choice == "r":
            return last_frame
        elif choice == "s":
            return None
        print("  Please enter 'r' to resume or 's' to start over.")


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path = "work/frames",
) -> int:
    """Pull every frame out of a video file and save as numbered PNGs.

    If frames from the same video already exist in output_dir, offers
    to resume from where extraction left off — handy when a long
    extraction gets interrupted halfway through.

    Returns the total number of frames extracted (including any
    previously existing frames when resuming).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    meta = _video_metadata(cap)
    total = meta["total_frames"]
    print(f"Video: {meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps, {total} frames")

    # Check for resumable state before we blow anything away
    resume_from = _check_resume(output_dir, video_path, meta)

    if resume_from is not None and resume_from >= total:
        # Fully extracted already — skip the whole thing
        cap.release()
        return resume_from

    if resume_from is not None and resume_from > 0:
        # Seek past already-extracted frames
        print(f"Resuming from frame {resume_from + 1}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume_from)
        start_count = resume_from
    else:
        # Fresh start — wipe any stale frames
        if any(output_dir.glob("frame_*.png")):
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        start_count = 0

    # Write metadata so future runs can detect this video
    _save_metadata(output_dir, video_path, meta)

    count = start_count
    remaining = total - start_count
    with tqdm(total=remaining, desc="Extracting frames", initial=0) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            out_path = output_dir / f"frame_{count:06d}.png"
            cv2.imwrite(str(out_path), frame)
            pbar.update(1)

    cap.release()
    print(f"Extracted {count} frames to {output_dir}")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("video_path", help="Input video file")
    parser.add_argument(
        "-w", "--work-dir", default="work", help="Working directory (frames saved to <work-dir>/frames/)",
    )

    args = parser.parse_args()

    extract_frames(args.video_path, Path(args.work_dir) / "frames")
