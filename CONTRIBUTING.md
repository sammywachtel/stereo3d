# Contributing

Thanks for wanting to help turn flat videos into 3D. Here's how to get going.

## Setup

```bash
# Clone and install everything (including dev tools)
git clone https://github.com/sammywachtel/stereo3d.git
cd stereo3d
uv sync --group dev

# Install pre-commit hooks — these run ruff on every commit
uv run pre-commit install
```

You'll also need **FFmpeg** for the encode stage:
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

## Running Tests

```bash
# All fast tests (skips model downloads)
uv run pytest

# Include integration tests (requires ffmpeg)
uv run pytest -m "not slow"

# Everything, including slow tests that download MiDaS weights
uv run pytest -m ""

# Single module
uv run pytest tests/test_stereo.py -v
```

### Test markers

| Marker | Meaning |
|--------|---------|
| `slow` | Downloads model weights or runs full pipeline |
| `integration` | Requires ffmpeg installed |

## Code Style

Ruff handles formatting and linting. The pre-commit hooks run it automatically, but you can also run it manually:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Config lives in `pyproject.toml` under `[tool.ruff]`.

### Comment style

Write comments like a veteran programmer sharing hard-won wisdom — honest about hacks, pragmatic about trade-offs. Skip the corporate boilerplate.

```python
# Good
# Guided filter snaps depth edges to image edges. Without this,
# MiDaS leaves wobbly boundaries that cause stereo shimmer.

# Bad
# This function applies a guided filter to the depth map
```

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Write tests for new functionality
3. Make sure `uv run pytest` passes
4. Pre-commit hooks will run ruff on commit — fix any issues they flag
5. Open a PR with a clear description of what and why

CI runs ruff and the test suite automatically on every PR.

## Architecture

The pipeline has four independent stages, each reading from and writing to disk:

```
extract → depth → stereo → encode
```

This design means you can hack on one stage without touching the others. See the README for details on each stage's inputs and outputs.
