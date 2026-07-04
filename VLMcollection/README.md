# VLMcollection — Vision-Language Models

Ready-to-run implementations of 14 vision-language models for edge deployment (NVIDIA Jetson, Orin NX/AGX, etc.) and server inference.

## Models

| Model | Params | Folder | Type | Capabilities |
|-------|--------|--------|------|-------------|
| **YOLO11/26** (Ultralytics) | 2.7M–68M | `yolo11-26/` | CNN detector | Detection, pose, OBB, tracking |
| **Florence-2** (Microsoft) | 0.23B–0.77B | `florence-2/` | Task-prompt VLM | Captioning, OD, grounding, OCR |
| **PaliGemma 2** (Google) | 3B | `paligemma/` | Encoder-decoder VLM | Caption, detection, VQA, segmentation |
| **LocateAnything-3B** (NVIDIA) | 3B | `locate_anything/` | Grounding VLM | Visual grounding, detection |
| **Phi-3.5 Vision** (Microsoft) | 4.2B | `phi-vision/` | Decoder VLM | Caption, VQA, document/chart QA |
| **Cosmos Reason1** (NVIDIA) | 7B | `cosmos-nemotron/` | Reasoning VLM | Physical AI reasoning, VQA |
| **Qwen3-VL-8B Instruct** (Alibaba) | 8.8B | `qwen3-vl_instruct/` | General VLM | Description, detection, VQA, OCR |
| **Qwen3-VL-8B Thinking** (Unsloth 4-bit) | ~9B | `qwen3-vl_thinking/` | Reasoning VLM | CoT reasoning, intent, detection |
| **Llama 3.2 Vision** (Meta, 4-bit) | 11B | `llama-vision/` | Multimodal LM | Description, reasoning, VQA |
| **Phi-4 Multimodal** (Microsoft) | ~14B | `phi-4_multimodal/` | Multimodal VLM | Image + audio + text, video, webcam |
| **DiffusionGemma-26B** (Google) | 26B | `diffusion_gemma_vl/` | Diffusion + YOLO | Caption, VQA (text-only diffusion) |
| **DINOv3** (Meta) | 22M–7B | `dinov3/` | Vision encoder | Zero-shot structured description |
| **SigLIP2** (Google) | 0.4B | `siglip2/` | Vision encoder | Zero-shot structured description |
| **MoonViT** (Moonshot AI) | 0.4B | `moonvit/` | Vision encoder | Zero-shot structured description |

## Quick Start

Each model has its own venv and entry script:

```bash
cd <model-folder>
source .venv/bin/activate
python run.py  # or inference.py / predict.py for task-specific scripts
```

See each model's README for detailed usage instructions.

## Tested Platform

- **GPU:** RTX 5090 (32 GB VRAM, ~16 GB free)
- **CUDA:** 13.2
- **Python:** 3.10.12 / 3.13
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

## Quick Reference

| Task | Best Model | Reason |
|------|-----------|--------|
| Object detection | YOLO26n | Fastest, most accurate for 80-class COCO |
| Visual grounding | LocateAnything-3B | Specialized for free-form text queries |
| Image captioning | Florence-2 | Fast (<1s), good quality |
| Physical reasoning | Cosmos-Reason1-7B | Trained for physics/robotics |
| Chain-of-thought | Qwen3-Thinking | Built-in reasoning tokens |
| Document/chart QA | Phi-3.5 Vision | 128K context, strong at documents |
| Multimodal (image+audio) | Phi-4 Multimodal | Native audio understanding |
| Zero-shot description | SigLIP2 / MoonViT / DINOv3 | Vision encoders, no LLM needed |
