# VLM Experiments

Vision-Language Model experimentation, benchmarking, and demo suite. Designed for RTX 5090 (Blackwell CC 12.0, 32 GB VRAM).

## Project Structure

```
VLMexperiments/
├── VLMcollection/       # 20+ pre-installed VLM models, each in own venv
│   ├── yolo11-26/       # YOLO11/26 weights in models/ (20 weight files)
│   ├── florence-2/
│   ├── paligemma/
│   ├── locate_anything/
│   ├── phi-vision/
│   ├── cosmos-nemotron/
│   ├── qwen3-vl_instruct/
│   ├── qwen3-vl_thinking/  # Unsloth 4-bit
│   ├── llama-vision/
│   ├── phi-4_multimodal/
│   ├── diffusion_gemma_vl/
│   ├── Llava/             # 5 LLaVA variants
│   ├── DINOtool/          # 30+ vision backbone variants
│   ├── dinov3/
│   ├── siglip2/
│   ├── moonvit/
│   └── README.md          # Per-model details + attention implementations
│
├── VLMbenchmark/         # Benchmark suite: 19 tasks across all models
│   ├── scripts/          # benchmark_*.py drivers + common.py (model loaders)
│   ├── benchmark_descriptions/  # Task definitions (*.md)
│   ├── results/          # Per-model JSON results
│   ├── samples/          # Pre-generated fixed sample lists
│   ├── charts/           # Report visualizations
│   ├── generate_report.py
│   ├── attention_implementation_benchmark.md  # FA2/sdpa/eager choices
│   └── README.md
│
├── VLMshowcase/          # CLI demo tool (vlm-demo command)
│   ├── vlm_showcase/     # Python package with per-task modules
│   ├── ros2_ws/          # ROS2 live webcam node
│   ├── scripts/          # Shell helpers
│   └── README.md         # Full usage guide
│
├── VLMforRobotPlan/      # Robot planning with VLMs
└── unsloth_compiled_cache/  # Unsloth compilation artifacts
```

## Quick Start

```bash
# --- Demos ---
cd VLMshowcase
source .venv/bin/activate
vlm-demo list                          # Available models
vlm-demo run /path/to/images/ --model yolo26n --batch
vlm-demo compare image.jpg             # Side-by-side all models
# See VLMshowcase/README.md for full usage

# --- Benchmarks ---
cd VLMbenchmark
python3 scripts/benchmark_all.py --max-images 50   # Full suite (hours)
python3 scripts/benchmark_all.py --tasks captioning vqa --max-images 50
python3 generate_report.py                          # Produce Benchmark_Results.md
# See VLMbenchmark/README.md for per-task commands

# --- Standalone model inference ---
cd VLMcollection/<model>
source .venv/bin/activate
python run.py --image /path/to/img.jpg --task caption
```

## Models

| Category | Models | Count | Attention |
|----------|--------|-------|-----------|
| CNN detectors | YOLO11 (n/s/m/l/x), YOLO26 (n/s/m/l/x) | 10 | N/A (CNN) |
| Task-prompt VLMs | Florence-2-large, PaliGemma2-3B | 2 | `sdpa` |
| Grounding VLMs | LocateAnything-3B (PyTorch + TRT) | 2 | `sdpa` |
| General VLMs | Qwen3-VL-8B-Instruct, Qwen3-Thinking | 2 | `sdpa` / Unsloth |
| Reasoning VLMs | Cosmos-Reason1-7B, Llama-3.2-11B-Vision | 2 | `sdpa` |
| Document VLMs | Phi-3.5-Vision-4.2B | 1 | `eager` |
| Multimodal VLMs | Phi-4-Multimodal | 1 | `flash_attention_2` |
| LLaVA family | Mistral-7B, OneVision-7B, NeXT-Video-7B/34B, Phi-3-4B | 5 | `sdpa` |
| Diffusion VLMs | DiffusionGemma-26B (+ YOLO/SigLIP2/MoonViT encoders) | 6 | `sdpa` |
| Vision encoders | SigLIP2, MoonViT, DINOv3, DINOtool (30+ backbones) | 4+ | `sdpa` |

See `VLMcollection/README.md` for full details and `VLMbenchmark/attention_implementation_benchmark.md` for selection rationale.

## Attention Implementation

Each model uses its fastest tested `attn_implementation`:

| Model | Chosen | Rationale |
|-------|--------|-----------|
| Florence-2, PaliGemma2, Cosmos-Reason1, Qwen3-Instruct | `sdpa` | Fastest FPS on Blackwell |
| Llama-3.2-Vision | `sdpa` | Only robust option (FA2 broken) |
| Phi-3.5-Vision | `eager` | Only supported option |
| Phi-4-Multimodal | `flash_attention_2` | Native FA2 support |
| LLaVA, vision encoders | `sdpa` | Default in their run.py |

## Environment

- **GPU:** RTX 5090 (32 GB VRAM)
- **CUDA:** 13.2 at `/usr/local/cuda-13.2`
- **Build arch:** `TORCH_CUDA_ARCH_LIST="12.0"`
- **HF cache:** `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface`
- **Unsloth cache:** `/mnt/HDD1/unsloth_and_hugging_face_models/unsloth_compiled_cache`
- **Python:** Per-model venvs (3.10–3.13)
