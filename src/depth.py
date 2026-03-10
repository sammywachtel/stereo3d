"""Depth estimation using MiDaS (via torch.hub)."""

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def load_midas(model_type: str = "DPT_Large") -> tuple:
    """Load MiDaS model and transforms. Downloads weights on first run.

    DPT_Large is the beefiest option — slower but noticeably better depth
    maps than DPT_Hybrid. Worth the wait for offline processing.
    """
    # M4 Mac → MPS backend, NVIDIA → CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Loading MiDaS ({model_type}) on {device}...")

    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to(device)
    model.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if model_type in ("DPT_Large", "DPT_Hybrid") else transforms.small_transform

    return model, transform, device


def estimate_depth(
    frames_dir: str | Path = "work/frames",
    output_dir: str | Path = "work/depth",
    model_type: str = "DPT_Large",
    temporal_alpha: float = 0.3,
    batch_size: int = 4,
) -> int:
    """Run MiDaS on every frame, save temporally-smoothed depth maps as PNGs.

    White = close, black = far. The depth values are relative (not metric),
    but consistent enough across frames for stereo synthesis.

    temporal_alpha controls how much the current frame's raw depth contributes
    vs the running average. Lower = smoother/more stable, higher = more
    responsive to actual depth changes. 0.3 is a good starting point —
    enough to prevent frame-to-frame jitter while still tracking real motion.

    batch_size controls how many frames are fed to the GPU at once. Higher
    values use more VRAM but reduce CPU↔GPU round-trip overhead. 4 is safe
    for most GPUs; bump to 8-16 if you have VRAM to spare.

    Returns the number of depth maps generated.
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    model, transform, device = load_midas(model_type)

    # Running average for temporal smoothing. None until first frame.
    smooth_depth = None

    count = 0
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(frame_files), batch_size), desc=f"Estimating depth (batch={batch_size})"):
            batch_paths = frame_files[batch_start : batch_start + batch_size]

            # Read and transform all frames in this batch
            imgs = []
            input_tensors = []
            for frame_path in batch_paths:
                img = cv2.imread(str(frame_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
                input_tensors.append(transform(img_rgb))

            # Stack into a single batch tensor and run inference once
            input_batch = torch.cat(input_tensors, dim=0).to(device)
            predictions = model(input_batch)

            # Process each prediction — temporal smoothing is sequential
            # so we handle one at a time after the GPU does its thing
            for i, frame_path in enumerate(batch_paths):
                prediction = torch.nn.functional.interpolate(
                    predictions[i].unsqueeze(0).unsqueeze(0),
                    size=imgs[i].shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth = prediction.cpu().numpy().astype(np.float64)

                # Normalize to [0, 1] — do this BEFORE smoothing so the
                # running average doesn't drift as MiDaS rescales per-frame
                depth_min = depth.min()
                depth_max = depth.max()
                if depth_max - depth_min > 0:
                    depth = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth = np.zeros_like(depth)

                # Exponential moving average: new = α * current + (1-α) * previous
                # First frame seeds the accumulator, no blending needed.
                if smooth_depth is None:
                    smooth_depth = depth
                else:
                    smooth_depth = temporal_alpha * depth + (1.0 - temporal_alpha) * smooth_depth

                depth_u8 = (smooth_depth * 255).astype(np.uint8)

                out_path = output_dir / frame_path.name
                cv2.imwrite(str(out_path), depth_u8)
                count += 1

    print(f"Generated {count} depth maps in {output_dir}")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MiDaS depth estimation on extracted frames.")
    parser.add_argument("-w", "--work-dir", default="work", help="Working directory (reads frames/, writes depth/)")
    parser.add_argument("--model", default="DPT_Large", help="MiDaS model (DPT_Large, DPT_Hybrid, MiDaS_small)")
    parser.add_argument("--batch-size", type=int, default=4, help="Frames per GPU batch (default: 4)")

    args = parser.parse_args()

    estimate_depth(
        frames_dir=Path(args.work_dir) / "frames",
        output_dir=Path(args.work_dir) / "depth",
        model_type=args.model,
        batch_size=args.batch_size,
    )
