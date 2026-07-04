# Qwen3-VL-8B-Instruct

Standalone inference setup for **Qwen3-VL-8B-Instruct** — a general-purpose vision-language model that can output bounding boxes via `<box>` tags.

This is **not** integrated with NVIDIA's LocateAnything-3B architecture (Parallel Box Decoding / PBD). It runs Qwen3-VL-8B as a native VL model that happens to understand `<box>` syntax.

## Setup

```bash
source .venv/bin/activate
```

Dependencies: `torch`, `transformers>=5.12`, `pillow`, `numpy`, `bitsandbytes>=0.49`.

## Usage

```bash
# Describe an image
python infer_qwen3.py ~/photo.jpg

# Ask a specific question
python infer_qwen3.py ~/photo.jpg "What color is the car?"

# Object detection (model outputs <box> tags)
python infer_qwen3.py ~/photo.jpg "Find all people" --json

# Disable thinking mode for faster responses
python infer_qwen3.py ~/photo.jpg "Describe this" --no-thinking
```

## Model directories

| Path | Content | Size |
|------|---------|------|
| `model_vl/` | **Qwen3-VL-8B-Instruct** (active) | ~16 GB, 4 shards |
| `model/` | Qwen3.6-35B-A3B (inactive, too large for 32GB GPU) | ~67 GB, 26 shards |

The `model/` directory (original Qwen3.6-35B-A3B) is kept for reference but requires >48GB VRAM. Run `infer_qwen3.py` — it auto-detects `model_vl/` first.

## Benchmark results (100 COCO val2017 images, RTX 5090)

| Metric | LA-3B | Qwen3-VL-8B-Instruct |
|--------|-------|----------------------|
| FPS | 4.70 | 0.80 |
| mAP@50:95 | 0.170 | 0.078 |
| mAP@50 | 0.208 | 0.130 |
| Detected | 504/655 | 329/655 |

Qwen3-VL-8B is ~63% of LA's mAP@50. A general 8.8B VL model can't match a specialized 3B detection model in speed or accuracy, but it's the only Qwen3-VL variant that fits 32GB VRAM.

## Hardware

RTX 5090 (32 GB VRAM). At BF16, the model uses ~16 GB leaving room for KV cache and activations.

## License

See `model_vl/README.md` (Qwen3-VL-8B-Instruct) and `model/README.md` (Qwen3.6-35B-A3B).

## Tested & Working (2026-07-04)

Verified on RTX 5090 (32 GB VRAM):
- `python infer_qwen3.py img.jpg` — image description
- `python infer_qwen3.py img.jpg "What color is the car?" --no-thinking` — VQA without thinking tokens
- `python infer_qwen3.py img.jpg "Find all people" --json` — detection with `<box>` output
- Benchmark integration: `benchmark_caption.py --model qwen3_native --max-images 2` — 5.4s/image avg
- Benchmark integration: `benchmark_vqa.py --model qwen3_native --max-questions 8` — works
- Benchmark integration: `benchmark_od.py --model qwen3_native --max-images 100` — mAP@50:95 0.078
- Benchmark integration: `benchmark_grounding.py --model qwen3_native --max-images 100` — works
- VLMshowcase: `vlm-demo scene img.jpg` — scene analysis with YOLO

Notes:
- Model is ~16 GB at BF16, loads in ~2 min on first run
- Python 3.10 venv (venv was created with Python 3.10, `python` symlink points to system python3)
- Set `HF_HOME` to the shared HuggingFace cache before running
