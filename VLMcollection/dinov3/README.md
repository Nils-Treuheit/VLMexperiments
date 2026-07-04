# DINOv3 Vision Encoder

Zero-shot scene understanding using Meta's [DINOv3](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009) self-supervised vision foundation model.

## Modes

| Flag | Description |
|---|---|
| `--task encode` | Extract DINOv3 patch features + pooled embedding (JSON, numpy base64) |
| `--task describe` | Zero-shot classification → structured text description |
| `--task encode+describe` | Both |

## Usage

```bash
# Describe an image (zero-shot classification + structured text)
python run.py --image /path/to/image.jpg --task describe

# Extract vision embeddings
python run.py --image /path/to/image.jpg --task encode

# Use a larger model variant
python run.py --image /path/to/image.jpg --task describe --model facebook/dinov3-vitb16-pretrain-lvd1689m

# Custom device and top-k
python run.py --image /path/to/image.jpg --task describe --device cuda --top-k 12
```

## Models

| Model | Params | Hidden Size |
|---|---|---|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | ~22M | 384 |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | ~86M | 768 |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | ~307M | 1024 |
| `facebook/dinov3-vithplus16-pretrain-lvd1689m` | ~2B | 1536 |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m` | ~7B | 4096 |

Note: These are **gated models** — you must accept the license on Hugging Face and provide a valid `HF_TOKEN`.

## Task: describe

`--task describe` runs zero-shot classification using:
- **Vision encoder**: DINOv3 (any variant) — extracts image embeddings
- **Text encoder**: SigLIP2 SO-400M (1152-dim) — encodes candidate labels
- Labels: COCO objects (80) + scene types (20) + image attributes (15)
- Outputs structured text like: `"Objects detected: person (92.3%), chair (87.1%), ... Scene: indoor scene. Attributes: daytime, bright."`

## Tested & Working (2026-07-03)

Verified on RTX 5090:
- `--task describe` — zero-shot classification (uses `sentence-transformers/all-MiniLM-L6-v2` text encoder, 384-dim matching DINOv3 vits16/vitb16)
- `--task encode` — extract patch features + pooled embedding
- Both `encode+describe` also works

Notes:
- Default DINOv3 model: `facebook/dinov3-vits16-pretrain-lvd1689m` (384-dim)
- Default text model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, matches)
- For larger DINOv3 variants, use a matching text model via `--text-model`
- Gated models: requires `HF_TOKEN`

Programmatic usage:
```python
import subprocess, json, sys
result = subprocess.run(
    [sys.executable, "run.py", "--image", img_path, "--task", "describe"],
    capture_output=True, text=True, timeout=60,
)
data = json.loads(result.stdout)
scene_text = data["description_text"]
```

## Integration

This project is designed to be called as a subprocess from `diffusion_gemma_vl/run.py`:

```python
import subprocess, json, sys
result = subprocess.run(
    [sys.executable, "dinov3/run.py", "--image", img_path, "--task", "describe"],
    capture_output=True, text=True, timeout=60,
)
data = json.loads(result.stdout)
scene_text = data["description_text"]
```
