# SigLIP2 Vision Encoder

Google's multilingual vision-language encoder for standalone feature extraction
and zero-shot scene description. Two variants: FixRes (fixed resolution, backwards
compatible with SigLIP) and NaFlex (dynamic resolution, preserves aspect ratio).

## Setup

```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
```

## Usage

### Describe an image (zero-shot classification)

```bash
python run.py --image path/to/img.jpg --task describe
```

Output includes top-k COCO object detections, scene type, and attributes
with SigLIP2 zero-shot probabilities.

### Encode image to embedding vector

```bash
python run.py --image path/to/img.jpg --task encode
```

Outputs a base64-encoded numpy array of the pooled vision embedding.

### Both at once

```bash
python run.py --image path/to/img.jpg --task encode+describe
```

### Choose a different model size

```bash
# Larger model (384px, better accuracy)
python run.py --image path/to/img.jpg --task describe \
  --model google/siglip2-large-patch16-384

# NaFlex (dynamic resolution)
python run.py --image path/to/img.jpg --task describe \
  --model google/siglip2-base-patch16-naflex

# Shape-optimized
python run.py --image path/to/img.jpg --task describe \
  --model google/siglip2-so400m-patch14-384
```

## Feed into DiffusionGemma

```bash
python ../diffusion_gemma_vl/run.py \
  --image path/to/img.jpg \
  --encoder siglip2 \
  --task caption
```

This calls `siglip2/run.py --task describe` internally and feeds the
structured text description to `llama-diffusion-cli`.

## Task descriptions

| Mode | Description |
|------|-------------|
| `describe` | Zero-shot classification -> structured text |
| `encode` | Extract vision embedding as base64 numpy |
| `encode+describe` | Both embedding and description |

## Models

| HF ID | Size | Params | Patch |
|-------|------|--------|-------|
| `google/siglip2-base-patch16-224` | Base | 0.4B | 16x16 |
| `google/siglip2-large-patch16-256` | Large | 0.9B | 16x16 |
| `google/siglip2-large-patch16-384` | Large | 0.9B | 16x16 |
| `google/siglip2-so400m-patch14-384` | Shape-opt | 1.0B | 14x14 |
| `google/siglip2-base-patch16-naflex` | NaFlex | 0.4B | dynamic |
| `google/siglip2-large-patch16-naflex` | NaFlex | 0.9B | dynamic |

All models are Apache 2.0 licensed.
