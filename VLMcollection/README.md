# VLMcollection — Vision-Language Models

Ready-to-run implementations of 20+ vision-language models for server inference (RTX 5090, 32 GB VRAM) and edge deployment.

## Models

| Model | Params | Folder | Type | Attention | Capabilities |
|-------|--------|--------|------|-----------|-------------|
| **YOLO11/26** (Ultralytics) | 2.7M–68M | `yolo11-26/` | CNN detector | N/A (CNN) | Detection, pose, OBB, tracking |
| **Florence-2** (Microsoft) | 0.23B–0.77B | `florence-2/` | Task-prompt VLM | `sdpa` | Captioning, OD, grounding, OCR, segmentation |
| **PaliGemma 2** (Google) | 3B | `paligemma/` | Encoder-decoder VLM | `sdpa` | Caption, detection, VQA, segmentation |
| **LocateAnything-3B** (NVIDIA) | 3B | `locate_anything/` | Grounding VLM | `sdpa` (PT) / TensorRT | Visual grounding, detection, OCR, pointing |
| **Phi-3.5 Vision** (Microsoft) | 4.2B | `phi-vision/` | Decoder VLM | `eager` (only option) | Caption, VQA, document/chart QA |
| **Cosmos Reason1** (NVIDIA) | 7B | `cosmos-nemotron/` | Reasoning VLM | `sdpa` | Physical AI reasoning, VQA |
| **LLaVA-v1.6-Mistral-7B** (Microsoft) | 7B | `Llava/` | LLaVA VLM | `sdpa` | Caption, VQA, grounding, reasoning |
| **LLaVA-OneVision-Qwen2-7B** (Microsoft) | 7B | `Llava/` | LLaVA VLM | `sdpa` | Caption, VQA, reasoning |
| **LLaVA-NeXT-Video-7B** (Microsoft) | 7B | `Llava/` | LLaVA VLM | `sdpa` | Caption, VQA, video understanding |
| **Phi-3-Mini-4B (LLaVA)** (Microsoft) | 4B | `Llava/` | LLaVA VLM | `sdpa` | Caption, VQA |
| **Qwen3-VL-8B Instruct** (Alibaba) | 8.8B | `qwen3-vl_instruct/` | General VLM | `sdpa` | Description, detection, VQA, OCR |
| **Qwen3-VL-8B Thinking** (Unsloth 4-bit) | ~9B | `qwen3-vl_thinking/` | Reasoning VLM | Unsloth (internal) | CoT reasoning, intent, detection |
| **Llama 3.2 Vision** (Meta, 4-bit) | 11B | `llama-vision/` | Multimodal LM | `sdpa` | Description, reasoning, VQA |
| **LLaVA-NeXT-Video-34B** (Microsoft) | 34B | `Llava/` | LLaVA VLM | `sdpa` (4-bit quant) | Caption, VQA, video understanding |
| **Phi-4 Multimodal** (Microsoft) | ~14B | `phi-4_multimodal/` | Multimodal VLM | `flash_attention_2` | Image + audio + text, video, webcam |
| **DiffusionGemma-26B** (Google) | 26B | `diffusion_gemma_vl/` | Diffusion + YOLO | `sdpa` (subprocess) | Caption, VQA (text-only diffusion) |
| **DINOtool** (Meta etc.) | 21M–1.1B | `DINOtool/` | Vision encoder (30+ archs) | N/A (own impl.) | Zero-shot classification, description |
| **DINOv3** (Meta) | 37M–304M | `dinov3/` | Vision encoder | `sdpa` | Zero-shot structured description |
| **SigLIP2** (Google) | 0.4B | `siglip2/` | Vision encoder | `sdpa` | Zero-shot structured description |
| **MoonViT** (Moonshot AI) | 0.4B | `moonvit/` | Vision encoder | `sdpa` | Zero-shot structured description |

### DINOtool Variants

DINOtool wraps 30+ vision backbone variants in a unified interface. Key families:

| Family | Variants | Params |
|--------|----------|--------|
| **DINOv2** (Meta) | `vit-s/vit-b/vit-l/vit-g` | 21M–1.1B |
| **DINOv3** (Meta) | `dinov3-s/dinov3-b/dinov3-l/dinov3-hplus/dinov3-7b` | 37M–7B |
| **SigLIP / SigLIP2** (Google) | `siglip1/siglip2/siglip2-so400m-384/512` | 86M–400M |
| **RADIO** (NVlabs) | `radio-b/radio-l/radio-h/radio-g` | 90M–1.1B |
| **TIPSv2** (DeepMind) | `tipsv2-b/tipsv2-l/tipsv2-so400m/tipsv2-g` | 90M–1.1B |

Select variant with `--model <shortcut>` (e.g. `--model vit-g`).

## Quick Start

Each model has its own venv and entry script:

```bash
cd <model-folder>
source .venv/bin/activate
python run.py  # or inference.py / predict.py for task-specific scripts
```

See each model's README for detailed usage instructions.

## Tested Platform

- **GPU:** RTX 5090 (32 GB VRAM)
- **CUDA:** 13.2
- **Python:** 3.10.12 / 3.11 / 3.13
- **Package manager:** uv 0.11.17
- **Model cache:** `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/`

## Model Cache

All HuggingFace models are cached at:
```
/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/
```

Set `HF_HOME` to this path before running any model:
```bash
export HF_HOME=/mnt/HDD1/unsloth_and_hugging_face_models/huggingface
```

## Attention Implementation Notes

The `attention_implementation` choice significantly impacts inference speed on Blackwell GPUs (RTX 5090, CC 12.0):

- **`sdpa`** (Scaled Dot-Product Attention) — best overall balance. Chosen for most models.
- **`flash_attention_2`** — theoretically faster, but Blackwell kernel compilation is slow (244s first load for Qwen3) and often no faster than sdpa in practice.
- **`eager`** — fallback when FA2/sdpa unsupported (Phi-3.5-Vision).
- Model-specific recommendations determined via comprehensive 50-image benchmark across all implementations.

See `VLMbenchmark/attention_implementation_benchmark.md` for full benchmark results.

## Quick Reference

| Task | Best Model | Reason |
|------|-----------|--------|
| Object detection | YOLO26n | Fastest, most accurate for 80-class COCO |
| Visual grounding | LocateAnything-3B | Specialized for free-form text queries |
| Image captioning | Florence-2 / PaliGemma2 | Fast (<0.25s), good quality |
| Physical reasoning | Cosmos-Reason1-7B | Trained for physics/robotics |
| Chain-of-thought | Qwen3-Thinking | Built-in reasoning tokens |
| Document/chart QA | Phi-3.5 Vision | 128K context, strong at documents |
| Multimodal (image+audio) | Phi-4 Multimodal | Native audio understanding |
| Zero-shot classification | DINOtool (any variant) | 30+ backbones, label-agnostic |
| Zero-shot description | SigLIP2 / MoonViT / DINOv3 | Vision encoders, no LLM needed |
