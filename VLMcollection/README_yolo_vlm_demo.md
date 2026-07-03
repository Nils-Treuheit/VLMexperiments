# YOLO → Diffusion Gemma Demo

End-to-end visual understanding pipeline: runs **YOLO detection**, **YOLO pose estimation**, and **YOLO oriented bounding boxes** (OBB) simultaneously on an image, then feeds the combined structured observations to a **Diffusion LLM** for semantic interpretation.

## Pipeline

```
Image
  │
  ├─► YOLO Detection  ──► objects + bboxes
  ├─► YOLO Pose       ──► human keypoints/skeletons
  └─► YOLO OBB        ──► oriented bboxes
        │
        ▼
  Structured text prompt
        │
        ▼
  Diffusion Gemma LLM (llama-diffusion-cli)
        │
        ▼
  Semantic context / Human intent
```

## Setup

### YOLO models
The demo uses pre-downloaded YOLO weights from `yolo11-26/models/`:
- `yolo11m.pt` — detection (COCO 80 classes)
- `yolo11n-pose.pt` — pose estimation (17 keypoints)
- `yolo11n-obb.pt` — oriented bboxes (DOTA 15 classes)

### Diffusion Gemma
Requires the GGUF model at `~/.cache/huggingface/hub/diffusiongemma_local/diffusiongemma-26B-A4B-it-Q8_0.gguf`

## Usage

```bash
# Basic
python yolo_vlm_demo.py /path/to/image.jpg

# With annotated output
python yolo_vlm_demo.py /path/to/image.jpg -o output.jpg

# Adjust generation
python yolo_vlm_demo.py /path/to/image.jpg --diffusion-steps 32 --temperature 0.3 --max-tokens 256
```

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `image` | (required) | Path to input image |
| `--output` / `-o` | None | Save annotated visualization |
| `--conf` | 0.25 | YOLO confidence threshold |
| `--max-tokens` | 256 | Max tokens for LLM |
| `--diffusion-steps` | 64 | Diffusion denoising steps (lower = faster) |
| `--temperature` | 0.3 | LLM sampling temperature |
| `--json` | false | Output structured JSON (YOLO results only) |

## Speed

| Component | Time |
|-----------|------|
| YOLO (3 models, RTX 5090) | ~11s |
| Diffusion Gemma (CPU, 26B Q8) | ~75s for 24 steps, ~175s for 48 steps |

The Diffusion Gemma model runs on CPU because llama.cpp was compiled without CUDA. Recompile with `-DGGML_CUDA=ON` for GPU acceleration.
