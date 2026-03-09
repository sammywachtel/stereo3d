"""Stereo view synthesis — the heart of the 2D-to-3D conversion.

Takes an original frame + its depth map, shifts pixels horizontally based
on depth to create a synthetic second eye view. Uses reverse (backward)
warping via cv2.remap() so every output pixel gets a value — no holes.
"""

import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm


def synthesize_stereo_pair(
    frame: np.ndarray,
    depth: np.ndarray,
    max_disparity: float = 15.0,
) -> np.ndarray:
    """Create a side-by-side stereo image from a frame and its depth map.

    Uses reverse warping: for each pixel in the right-eye output, we look
    *back* into the left-eye (source) image to find its color. This means
    every output pixel gets a value — no holes, no inpainting needed.

    cv2.remap handles sub-pixel interpolation, so we get smooth results
    even at fractional disparity values. The only artifacts show up at the
    left edge where the remap samples beyond the frame boundary.
    """
    h, w = frame.shape[:2]

    # Normalize depth to [0, 1] float
    depth_norm = depth.astype(np.float32) / 255.0

    # Guided filter: use the original frame as the guide to snap depth
    # edges onto real image edges. Kills the wobbly depth noise at
    # foreground/background boundaries that MiDaS leaves behind.
    # radius=16 catches the fuzzy transition zone, eps=1e-4 keeps edges sharp.
    guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    depth_norm = cv2.ximgproc.guidedFilter(
        guide=guide, src=depth_norm, radius=16, eps=1e-4,
    )

    # Disparity map: near (white/1.0) = large shift, far (black/0.0) = small shift
    disparity = depth_norm * max_disparity

    # Build reverse warp maps for cv2.remap
    # For right eye: sample from (x + disparity, y) in the source.
    # Moving the virtual camera right means objects shift left in the
    # right-eye view, so we look rightward in the source to find them.
    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(x_coords, y_coords)

    map_x = map_x + disparity

    # Remap with bilinear interpolation — border pixels get replicated
    # so edges fade gracefully instead of going black
    right_eye = cv2.remap(
        frame, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Side-by-side: left eye | right eye
    sbs = np.hstack([frame, right_eye])
    return sbs


def _process_single_frame(args: tuple) -> bool:
    """Worker function for parallel stereo synthesis.

    Takes a tuple because Pool.map only passes one arg. Returns True on
    success so we can count completions without shared state.
    """
    frame_path_str, depth_path_str, out_path_str, max_disparity = args

    depth_path = Path(depth_path_str)
    if not depth_path.exists():
        return False

    frame = cv2.imread(frame_path_str)
    depth = cv2.imread(depth_path_str, cv2.IMREAD_GRAYSCALE)

    sbs = synthesize_stereo_pair(frame, depth, max_disparity)
    cv2.imwrite(out_path_str, sbs)
    return True


def synthesize_all(
    frames_dir: str | Path = "work/frames",
    depth_dir: str | Path = "work/depth",
    output_dir: str | Path = "work/stereo_frames",
    max_disparity: float = 15.0,
    workers: int | None = None,
) -> int:
    """Process all frame/depth pairs into SBS stereo images.

    Uses multiprocessing to saturate all CPU cores — each frame is
    independent so this parallelizes perfectly.

    Args:
        workers: Number of parallel workers. None = os.cpu_count().

    Returns the number of stereo frames generated.
    """
    frames_dir = Path(frames_dir)
    depth_dir = Path(depth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    if workers is None:
        workers = os.cpu_count() or 4

    # Build work items as plain strings — Path objects add pickling overhead
    work_items = [
        (
            str(frame_path),
            str(depth_dir / frame_path.name),
            str(output_dir / frame_path.name),
            max_disparity,
        )
        for frame_path in frame_files
    ]

    count = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # chunksize > 1 reduces IPC overhead when there are thousands of frames
        chunksize = max(1, len(work_items) // (workers * 4))
        results = pool.map(_process_single_frame, work_items, chunksize=chunksize)
        for success in tqdm(results, total=len(work_items), desc=f"Synthesizing stereo ({workers} workers)"):
            if success:
                count += 1

    print(f"Generated {count} stereo frames in {output_dir}")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthesize stereo SBS frames from frame/depth pairs.")
    parser.add_argument("-w", "--work-dir", default="work", help="Working directory (reads frames/ and depth/, writes stereo_frames/)")
    parser.add_argument("--max-disparity", type=float, default=15.0, help="Max pixel shift for stereo effect (default: 15.0)")

    args = parser.parse_args()

    synthesize_all(
        frames_dir=Path(args.work_dir) / "frames",
        depth_dir=Path(args.work_dir) / "depth",
        output_dir=Path(args.work_dir) / "stereo_frames",
        max_disparity=args.max_disparity,
    )
