# DINOtool

Unified ViT feature extraction and zero-shot description using the [`dinotool`](https://pypi.org/project/dinotool/) pip package. Supports DINOv2, DINOv3, SigLIP, OpenCLIP, RADIO, and TIPSv2 models.

## Quick Start

```bash
cd VLMcollection/DINOtool
uv venv .venv --python 3.13
uv pip install dinotool sentence-transformers

# Run description
python run.py --image path/to/img.jpg --task describe

# Run with a specific model
python run.py --image path/to/img.jpg --task describe --model siglip2

# Encode embeddings
python run.py --image path/to/img.jpg --task encode

# Extract PCA features
python run.py --image path/to/img.jpg --task features
```

## Supported Models

| Shortcut | Base Model | Source |
|---|---|---|
| `vit-s`, `dinov2-s` | DINOv2 ViT-S/14 | Meta |
| `vit-b`, `dinov2-b` | DINOv2 ViT-B/14 | Meta |
| `vit-l`, `dinov2-l` | DINOv2 ViT-L/14 | Meta |
| `vit-g`, `dinov2-g` | DINOv2 ViT-g/14 | Meta |
| `dinov3-s` | DINOv3 ViT-S/16 | Meta |
| `dinov3-b` | DINOv3 ViT-B/16 | Meta |
| `dinov3-l` | DINOv3 ViT-L/16 | Meta |
| `dinov3-hplus` | DINOv3 ViT-H+/16 | Meta |
| `siglip1` | SigLIP ViT-B/16 | Google |
| `siglip2` | SigLIP2 ViT-B/16 | Google |
| `siglip2-so400m-384` | SigLIP2 ViT-SO/14 (384) | Google |
| `siglip2-so400m-512` | SigLIP2 ViT-SO/14 (512) | Google |
| `siglip2-b16-256` | SigLIP2 ViT-B/16 (256) | Google |
| `siglip2-b16-512` | SigLIP2 ViT-B/16 (512) | Google |
| `clip` | OpenCLIP ViT-B/16 | LAION |
| `radio-b` | RADIO ViT-B/16 | Google |
| `radio-l` | RADIO ViT-L/16 | Google |
| `radio-h` | RADIO ViT-H/14 | Google |
| `radio-g` | RADIO ViT-g/14 | Google |
| `tipsv2-b` | TIPSv2 ViT-B/16 | — |
| `tipsv2-l` | TIPSv2 ViT-L/16 | — |
| `tipsv2-so400m` | TIPSv2 ViT-SO/14 | — |
| `tipsv2-g` | TIPSv2 ViT-g/14 | — |

## Tasks

### `describe`
Zero-shot object/scene/attribute classification via DINO/transformers features + sentence-transformers text embeddings.

### `encode`
Extract CLS token embedding as base64-encoded numpy array.

### `features`  
Extract local patch features with PCA visualization data.

## Programmatic API

```python
from dinotool_wrapper import DINoToolWorker

worker = DINoToolWorker(model_name="dinov2_vits14_reg")
text, predictions = worker.describe("image.jpg", top_k=5)
embedding = worker.encode("image.jpg")
features = worker.extract_features("image.jpg")
```

## Benchmark Integration

```bash
cd ../../VLMbenchmark/scripts
python benchmark_caption.py dinotool --max-images 50
python benchmark_all.py --caption
```

## Notes

- Models are cached at `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface` (HF) or `~/.cache/torch/hub` (torch hub).
- First run downloads any missing model weights automatically.
- xFormers is disabled for Blackwell GPU compatibility (falls back to PyTorch SDPA).
- For best zero-shot accuracy, use `siglip2`, `dinov3-b`, or `dinov2-g` models.
