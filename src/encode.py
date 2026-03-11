"""Video encoding — assembles stereo frames into a playable SBS video via FFmpeg."""

import re
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm


def encode_video(
    frames_dir: str | Path = "work/stereo_frames",
    output_path: str | Path = "output_sbs.mp4",
    fps: float = 24.0,
    crf: int = 18,
    audio_source: str | Path | None = None,
    quest: bool = False,
) -> Path:
    """Encode a directory of numbered PNGs into an H.264 MP4.

    CRF 18 is visually lossless for most content. Go lower (e.g. 15) if
    you're a pixel-peeper, higher (e.g. 23) if you need smaller files.

    If audio_source is provided, the audio track is copied straight from
    the original video — no re-encoding, no quality loss.

    If quest=True, encodes with settings optimized for Meta Quest 2/3
    playback: High profile for hardware decode, high bitrate to
    discourage DLNA servers from transcoding, and SBS 3D metadata so
    VR players auto-detect the stereo format.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg not found. Install it:\n"
            "  Mac:   brew install ffmpeg\n"
            "  Linux: sudo apt install ffmpeg"
        )

    frames_dir = Path(frames_dir)
    output_path = Path(output_path)

    # Verify we have frames to encode
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    if quest:
        print(f"Encoding {len(frame_files)} frames at {fps} fps (Quest mode, CRF {crf})...")
    else:
        print(f"Encoding {len(frame_files)} frames at {fps} fps (CRF {crf})...")

    cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
    ]

    # Pull audio from the original video if provided
    if audio_source is not None:
        audio_source = Path(audio_source)
        if audio_source.exists():
            cmd += ["-i", str(audio_source)]

    cmd += [
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        # Ensure dimensions are divisible by 2 (x264 requirement)
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    ]

    if quest:
        # Quest 2/3 hardware-decodes High profile Level 5.1 natively.
        # maxrate/bufsize cap the bitrate high enough for quality but
        # within what the XR2 chip can handle. This also tells DLNA
        # servers like UMS "this stream is already optimized, don't
        # transcode it into mush."
        cmd += [
            "-profile:v", "high",
            "-level", "5.1",
            "-maxrate", "60M",
            "-bufsize", "120M",
        ]

    if audio_source is not None and Path(audio_source).exists():
        # Copy audio as-is — no re-encode, no sync drift
        cmd += ["-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0?"]
        # shortest: stop when the shorter stream ends (frames or audio)
        cmd += ["-shortest"]

    if quest:
        # Tag the file as side-by-side 3D so VR players (Skybox, Oculus
        # Gallery, etc.) auto-detect the format instead of making the
        # user hunt through menus every time.
        cmd += ["-metadata:s:v", "stereo_mode=left_right"]

    cmd.append(str(output_path))

    # Stream stderr so we can parse FFmpeg's frame counter in real time.
    # FFmpeg spits out lines like "frame=  123 fps= 45.2 ..." to stderr.
    total_frames = len(frame_files)
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    frame_re = re.compile(r"frame=\s*(\d+)")
    stderr_lines = []

    with tqdm(total=total_frames, desc="Encoding video", unit="frame") as pbar:
        for line in process.stderr:
            stderr_lines.append(line)
            match = frame_re.search(line)
            if match:
                current = int(match.group(1))
                pbar.update(current - pbar.n)

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{''.join(stderr_lines)}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Encoded to {output_path} ({size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Encode stereo frames into an SBS video via FFmpeg.",
    )
    parser.add_argument("audio_source", nargs="?", default=None, help="Original video to copy audio from")
    parser.add_argument("-o", "--output", default="output_sbs.mp4", help="Output video path (default: output_sbs.mp4)")
    parser.add_argument(
        "-w", "--work-dir", default="work", help="Working directory containing stereo_frames/",
    )
    parser.add_argument("--fps", type=float, default=24.0, help="Output framerate (default: 24.0)")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality, lower=better (default: 18)")
    parser.add_argument(
        "--quest", action="store_true",
        help="Optimize for Meta Quest 2/3: High profile, high bitrate, SBS metadata",
    )

    args = parser.parse_args()

    encode_video(
        frames_dir=Path(args.work_dir) / "stereo_frames",
        output_path=args.output,
        fps=args.fps,
        crf=args.crf,
        audio_source=args.audio_source,
        quest=args.quest,
    )
