
# 3D Video Generator

A pipeline for converting **2D video into stereoscopic 3D video** using AI depth estimation and pixel warping.

The system extracts frames from a video, predicts a **depth map** for each frame using the MiDaS model, synthesizes a second camera perspective, and encodes the result into a **side-by-side (SBS) stereoscopic movie**.

Playable in VR headsets (Quest, Vision Pro), 3D TVs, and SBS-compatible media players (Skybox, DeoVR, VLC).

---

## Quick Start

```bash
# Install dependencies
uv add torch torchvision opencv-python opencv-contrib-python numpy pillow tqdm

# Run the full pipeline
uv run python -m src.pipeline input/movie.mp4
```

Output lands in `output/movie_sbs.mp4`. That's it.

---

## Full Pipeline Usage

```bash
uv run python -m src.pipeline <video> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `output/<name>_sbs.mp4` | Output video path |
| `-w, --work-dir` | `work` | Working directory for intermediate files |

### Examples

```bash
# Defaults — output goes to output/movie_sbs.mp4, work files in work/
uv run python -m src.pipeline input/movie.mp4

# Custom output file
uv run python -m src.pipeline input/movie.mp4 -o ~/Desktop/my_3d_movie.mp4

# Custom work directory (useful for external drives or multiple jobs)
uv run python -m src.pipeline input/movie.mp4 -w /Volumes/external/work

# Both
uv run python -m src.pipeline input/movie.mp4 -o /tmp/result.mp4 -w /tmp/work
```

---

## Running Individual Stages

Each stage reads from and writes to disk, so you can re-run any stage independently. All stages use `-w` to set the work directory (default: `work`).

### Stage 1: Extract Frames

```bash
uv run python -m src.extract <video> [-w work-dir]
```

Pulls every frame from the video as numbered PNGs into `<work-dir>/frames/`.

If frames from the same video already exist, you'll be asked whether to **resume** from where you left off or start over — handy when a long extraction gets interrupted.

### Stage 2: Depth Estimation

```bash
uv run python -m src.depth [-w work-dir] [--model MODEL] [--batch-size N]
```

Runs MiDaS on every frame, saves temporally-smoothed depth maps to `<work-dir>/depth/`.

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `DPT_Large` | MiDaS variant (`DPT_Large`, `DPT_Hybrid`, `MiDaS_small`) |
| `--batch-size` | `4` | Frames per GPU batch — higher uses more VRAM but runs faster |

### Stage 3: Stereo Synthesis

```bash
uv run python -m src.stereo [-w work-dir] [--max-disparity N]
```

Creates side-by-side stereo frames in `<work-dir>/stereo_frames/`. Runs across all CPU cores in parallel.

| Option | Default | Description |
|--------|---------|-------------|
| `--max-disparity` | `15.0` | Max pixel shift for 3D effect (12–18 recommended) |

### Stage 4: Encode Video

```bash
uv run python -m src.encode [audio-source] -o output.mp4 [-w work-dir] [--fps N] [--crf N]
```

Assembles stereo frames from `<work-dir>/stereo_frames/` into a final H.264 MP4. Pass the original video as the first argument to copy its audio track.

| Option | Default | Description |
|--------|---------|-------------|
| `audio-source` | *(none)* | Original video to copy audio from |
| `-o, --output` | `output_sbs.mp4` | Output video path |
| `--fps` | `24.0` | Output framerate |
| `--crf` | `18` | H.264 quality — lower is better, 18 is visually lossless |

### Example: Re-run encode after tweaking settings

```bash
uv run python -m src.encode input/movie.mp4 \
  -o /Volumes/external/movie_sbs.mp4 \
  -w /Volumes/external/work \
  --crf 15
```

---

## How It Works

```text
Video → Frame Extraction → Depth Estimation (MiDaS) → Stereo Synthesis → Video Encoding
                                                              ↓
                                                    Left Eye | Right Eye
```

| Stage | What happens |
|-------|-------------|
| **Extract** | Decodes video into individual PNG frames |
| **Depth** | MiDaS neural network estimates per-pixel depth (white=near, black=far) |
| **Stereo** | Shifts pixels horizontally based on depth to create a second eye view |
| **Encode** | FFmpeg assembles stereo frames into a playable H.264 MP4 |

---

## Directory Layout

```text
stereo3d/
├── input/              ← source videos
├── output/             ← final SBS videos
├── work/               ← intermediate files (configurable with -w)
│   ├── frames/         ← extracted PNG frames
│   ├── depth/          ← depth maps
│   └── stereo_frames/  ← side-by-side stereo frames
├── src/
│   ├── pipeline.py     ← full pipeline orchestrator
│   ├── extract.py      ← stage 1: frame extraction
│   ├── depth.py        ← stage 2: MiDaS depth estimation
│   ├── stereo.py       ← stage 3: stereo synthesis
│   └── encode.py       ← stage 4: FFmpeg encoding
└── blender/
    └── stereo_composite.py  ← optional Blender compositing
```

---

## Required Software

- **Python 3.14+**
- **uv** — `curl -Ls https://astral.sh/uv/install.sh | sh`
- **FFmpeg** — `brew install ffmpeg` (Mac) or `sudo apt install ffmpeg` (Linux)
- **Blender** *(optional)* — [blender.org/download](https://www.blender.org/download/)

### Hardware Notes

Target machine is Mac M4 (Apple Silicon). PyTorch uses the **MPS** backend for GPU acceleration — not CUDA. Docker containers can't access Metal, so run depth estimation on the host for best performance.

---

## Optional Blender Integration

Blender can be used for stereo convergence adjustments, depth-based fog, compositing, and titles.

```bash
blender -b -P blender/stereo_composite.py
```

---

## Recommended Test Settings

| Setting | Value |
|---------|-------|
| Resolution | 1080p |
| FPS | 24 |
| Clip Length | 10–20 seconds |
| MiDaS Model | `DPT_Hybrid` (fast) or `DPT_Large` (better quality) |
| Max Disparity | 12–18 px |

---

## License

MIT
