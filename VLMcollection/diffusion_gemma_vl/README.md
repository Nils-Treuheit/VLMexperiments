# DiffusionGemma VLM

Pluggable vision encoder + DiffusionGemma text-only diffusion model pipeline.

## Architecture

```
Image -> Vision Encoder (YOLO | SigLIP2 | MoonViT | DINOv3) -> structured text -> llama-diffusion-cli -> answer
```

Since DiffusionGemma GGUF (`llama-diffusion-cli`) is text-only, a vision encoder bridges the gap
by converting visual content into structured text that the diffusion model can reason about.

## Requirements

- Python 3.10+
- Vision encoder: `ultralytics` (YOLO) or `transformers` (SigLIP2 / MoonViT / DINOv3)
- `llama-diffusion-cli` built from `ggml-org/llama.cpp` PR #24423
- DiffusionGemma 26B Q8_0 GGUF (~26 GiB)

## Setup

```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install ultralytics
```

Build `llama-diffusion-cli` from https://github.com/ggml-org/llama.cpp/pull/24423.

Set `GGUF_PATH` env var to override the default GGUF path, or place the model at:
`/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/diffusiongemma_local/diffusiongemma-26B-A4B-it-Q8_0.gguf`

## Usage

```bash
# Descriptive caption (YOLO default)
python run.py --image path/to/img.jpg --task caption

# Choose vision encoder backend
python run.py --image path/to/img.jpg --task caption --encoder yolo
python run.py --image path/to/img.jpg --task caption --encoder siglip2
python run.py --image path/to/img.jpg --task caption --encoder moonvit

# Combine YOLO task types for richer scene description
python run.py --image path/to/img.jpg --task caption --encoder yolo --yolo-tasks aabb
python run.py --image path/to/img.jpg --task caption --encoder yolo --yolo-tasks aabb,pose
python run.py --image path/to/img.jpg --task caption --encoder yolo --yolo-tasks aabb,pose,obb

# Persistent mode (keep llama-diffusion-cli alive across prompts, avoid ~60s reload)
python run.py --image path/to/img.jpg --task caption --encoder yolo --persist --n-predict 512

# Visual QA
python run.py --image path/to/img.jpg --task vqa --prompt "Is the person sitting?"

# YOLO-only modes (fast, no LLM)
python run.py --image path/to/img.jpg --task detect
python run.py --image path/to/img.jpg --task pose
python run.py --image path/to/img.jpg --task obb

# Pure text chat (no image)
python run.py --task chat --prompt "What is 2+2?"
```

## Task descriptions

| Mode | Encoder | Description |
|------|---------|-------------|
| `caption` | yolo / siglip2 / moonvit | Descriptive image caption -> llama-diffusion-cli |
| `vqa` | yolo / siglip2 / moonvit | Visual question answering |
| `chat` | none | Pure text-only chat (no image) |
| `detect` | yolo | YOLO object detection only (no LLM) |
| `pose` | yolo | YOLO pose estimation only (no LLM) |
| `obb` | yolo | YOLO oriented bounding box only (no LLM) |
| `detect+pose` / `detect+obb` / `all` | yolo | Combined YOLO tasks via `--yolo-tasks` |

## Persistent Engine (`--persist`)

Keeps `llama-diffusion-cli --conversation` alive in a PTY subprocess, eliminating the
~60s GGUF reload between images. Uses `/clear` to reset conversation history per prompt.

**Status**: Single-image works. Multi-image in progress (second generation may return empty).

```bash
# First call loads model (~60s + generation time)
python run.py --image img1.jpg --task caption --persist

# Subsequent calls reuse the loaded model (no reload)
# (in the same process; subprocess-based benchmarks start fresh each time)
```

## Benchmark

```bash
python ../Benchmark/scripts/benchmark_all.py --tasks captioning vqa --max-images 5
```

## Key Files

- `run.py` - main pipeline entry point
- `llama.cpp/build/bin/llama-diffusion-cli` - built CLI binary (CPU-only)

## Limitations

- CPU-only inference ~1-5 tok/s (26B Q8_0)
- Vision quality limited by encoder accuracy
- Persistent engine multi-image stability in progress
- CUDA acceleration requires system CUDA toolkit install
