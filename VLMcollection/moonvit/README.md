# MoonViT Vision Encoder

Native-resolution vision encoder from Moonshot AI (Kimi-VL team).
Initialized from SigLIP-SO-400M and continually pre-trained to handle
dynamic image resolutions while preserving aspect ratio.

## Setup

```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
```

The venv inherits the system-site-packages which includes
`transformers` with SigLIP2 support.

## Usage

### Describe an image

```bash
python run.py --image path/to/img.jpg --task describe
```

This loads MoonViT for vision + SigLIP2 text encoder for zero-shot
classification. Outputs top-k COCO objects, scene type, and attributes.

### Encode image to features

```bash
python run.py --image path/to/img.jpg --task encode
```

Outputs:
- `patch_features`: list of per-patch feature tensors (base64 numpy)
- `embedding`: global pooled embedding (base64 numpy)

MoonViT's output shape is `[num_patches, num_heads, dim]`, which varies
per image since it supports native resolution.

### Both

```bash
python run.py --image path/to/img.jpg --task encode+describe
```

## Feed into DiffusionGemma

```bash
python ../diffusion_gemma_vl/run.py \
  --image path/to/img.jpg \
  --encoder moonvit \
  --task caption
```

This calls `moonvit/run.py --task describe` internally and feeds the
structured text description to `llama-diffusion-cli`.

## Task descriptions

| Mode | Description |
|------|-------------|
| `describe` | Zero-shot classification -> structured text with objects, scene, attributes |
| `encode` | Extract MoonViT patch features + global embedding as base64 numpy |
| `encode+describe` | Both embedding and description |

## Model details

| Property | Value |
|----------|-------|
| HF ID | `moonshotai/MoonViT-SO-400M` |
| Params | 0.4B |
| Base | SigLIP-SO-400M (shape-optimized) |
| Resolution | Native/dynamic (preserves aspect ratio) |
| Output feature dim | 1152 per head |
| Output heads | 4 |
| License | MIT |

## Notes

- MoonViT uses `trust_remote_code=True` (custom model architecture)
- The `describe` mode also loads a SigLIP2 text encoder for zero-shot
  classification (MoonViT is vision-only)
- Dynamic resolution means inference time varies with image size
