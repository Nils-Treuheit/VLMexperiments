# Qwen3-VL-8B-Thinking

Local inference setup for [unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit) — a 4-bit quantized vision-language model from the Qwen3-VL family.

## Model

- **Architecture:** Qwen3-VL-8B (Dense, ~9B params)
- **Quantization:** 4-bit BnB (NF4) via bitsandbytes
- **Type:** Thinking (reasoning-enhanced)
- **Context:** 256K tokens native, expandable to 1M
- **Capabilities:** Image description, visual reasoning, OCR (32 languages), object detection (2D grounding with bounding boxes), 3D grounding, visual agent, coding from images/video

## Requirements

- Python 3.11
- NVIDIA GPU with ~6GB+ VRAM (tested on RTX 5090)
- CUDA 12.x

## Setup

```bash
cd /mnt/HDD1/Project_Code/qwen3-vl_thinking

# Create virtual environment with uv
uv venv --python 3.11

# Install dependencies
uv pip install -r requirements.txt
```

Set the HuggingFace cache (model is ~7.9GB):

```bash
export HF_HOME=/mnt/HDD1/tmp/hf_home
```

## One-shot inference

```bash
HF_HOME=/mnt/HDD1/tmp/hf_home .venv/bin/python test_inference_unsloth.py
```

The model loads from disk (~1s), runs, and exits. Weights are re-loaded every run.

---

## Persistent inference (recommended for live feeds)

The model stays loaded in memory — no reload between calls.

### Option A: Import as a module (zero overhead)

```python
from qwen_detector import QwenVLDetector

detector = QwenVLDetector()          # loads once
result = detector.detect("img.jpg")  # reuses model
desc   = detector.describe("img.jpg")
```

`detect()` prompts for bounding boxes in JSON format `[{"bbox_2d": [x1,y1,x2,y2], "label": "..."}]` with coordinates normalized to 0–1000.

Example watching a folder for new images:

```bash
.venv/bin/python live_feed.py /path/to/image/folder
```

### Option B: FastAPI server (for parallel/multi-process)

```bash
.venv/bin/python server.py --port 8000
```

```bash
# Detect
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/photo.jpg", "prompt": "Detect all cars"}'

# Health check
curl http://localhost:8000/health
```

Supports `image_url`, `image_path`, and `image_base64`.

### Option C: Named pipe (for piping from another process)

```bash
mkfifo /tmp/qwen_pipe
.venv/bin/python live_feed.py pipe /tmp/qwen_pipe &
echo "/path/to/image.jpg" > /tmp/qwen_pipe
```

## Object detection

Qwen3-VL has native 2D grounding. Prompt it with:

```
Detect all objects. Return JSON with bbox_2d (0-1000 range) and label.
```

Output example: `[{"bbox_2d": [218, 103, 343, 205], "label": "yellow taxi"}]`

Convert to pixel coordinates: `x / 1000 * image_width`

## Generation parameters (Thinking model)

| Parameter         | Recommended |
|-------------------|-------------|
| Temperature       | 1.0         |
| Top-P             | 0.95        |
| Top-K             | 20          |
| Presence penalty  | 0.0         |
| Max tokens        | 40960       |

## Files

| File | Purpose |
|------|---------|
| `test_inference_unsloth.py` | One-shot image description |
| `qwen_detector.py` | Reusable `QwenVLDetector` class |
| `server.py` | FastAPI server (keeps model alive) |
| `live_feed.py` | Folder watcher / named pipe consumer |
| `requirements.txt` | Python dependencies |

## Tested & Working (2026-07-04)

Verified on RTX 5090 (32 GB VRAM):
- `QwenVLDetector().describe(img.jpg)` — chain-of-thought description with visual reasoning
- `QwenVLDetector().detect(img.jpg)` — object detection with JSON `bbox_2d` output
- Benchmark integration: `benchmark_caption.py --model qwen3_thinking --max-images 2` — 29.4s/image avg
- Benchmark integration: `benchmark_vqa.py --model qwen3_thinking --max-questions 8` — works
- Benchmark integration: `benchmark_od.py --model qwen3_thinking --max-images 100` — works
- VLMshowcase: `vlm-demo intent img.jpg --all` — human intent/action/emotion analysis

Notes:
- 4-bit BnB quantized model (~6 GB VRAM), loads in ~45s
- Longer inference time due to chain-of-thought reasoning tokens
- `python` in `.venv/bin/` correctly resolves to Python 3.13
