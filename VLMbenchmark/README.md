# VLM Benchmark

Multi-task benchmark suite for vision-language models, vision encoders, and object detectors.

Tests 20+ benchmarks across object detection, captioning, VQA, classification, pose estimation, segmentation, embedding quality, and more on RTX 5090.

## Quick Start

```bash
# Run all tasks with 50 images per model
python3 scripts/benchmark_all.py --max-images 50

# Run specific tasks
python3 scripts/benchmark_all.py --tasks zeroshot_od zeroshot_cls --max-images 100

# Generate report + charts after benchmarks complete
python3 generate_report.py
```

Each model is automatically dispatched to its own virtual environment under `../VLMcollection/`.

## Benchmarks

| # | Benchmark | Script | Dataset | Description |
|---|-----------|--------|---------|-------------|
| 1 | Object Detection | `benchmark_od.py` | COCO val2017 | [Instance Localization](benchmark_descriptions/InstanceLocalization.md) |
| 2 | Zero-Shot OD | `benchmark_zeroshot_od.py` | COCO val2017 | [Zero-Shot Object Detection](benchmark_descriptions/0ShotObjDet.md) |
| 3 | Pose Estimation | `benchmark_pose.py` | COCO Keypoints | 2D keypoint detection (YOLO only) |
| 4 | Oriented BBox | `benchmark_obb.py` | DOTA-v1.0 | Rotated bounding box detection |
| 5 | Phrase Grounding | `benchmark_grounding.py` | COCO val2017 | [Visual Grounding](benchmark_descriptions/VisualGrounding.md) |
| 6 | Image Captioning | `benchmark_caption.py` | COCO Captions | [Image Captioning](benchmark_descriptions/ImageCaptioning.md) |
| 7 | Visual QA | `benchmark_vqa.py` | COCO val2017 | [Visual QA](benchmark_descriptions/VQA.md) |
| 8 | Zero-Shot Classification | `benchmark_zeroshot_cls.py` | Tiny ImageNet | [Zero-Shot Image Classification](benchmark_descriptions/0ShotImgClassify.md) |
| 9 | Segmentation | `benchmark_segmentation.py` | COCO val2017 | [Image Segmentation](benchmark_descriptions/ImageSegmentation.md) |
| 10 | Scene Analysis | `benchmark_scene.py` | COCO val2017 | [Semantic Scene Analysis](benchmark_descriptions/SemanticSceneAnalysis.md) |
| 11 | Multi-Object Tracking | `benchmark_tracking.py` | MOT17 | ByteTrack multi-object tracking |
| 12 | 6D Pose | `benchmark_6dpose.py` | Linemod | 6D object pose estimation |
| 13 | OCR / Text Detection | `benchmark_ocr.py` | Synthetic COCO | Text detection on synthetic overlays |
| 14 | Pointing / 2D KP | `benchmark_pointing.py` | COCO Keypoints | [Instance Localization](benchmark_descriptions/InstanceLocalization.md) |
| 15 | Object Counting | `benchmark_counting.py` | COCO val2017 | [Object Counting](benchmark_descriptions/ObjectCounting.md) |
| 16 | Visual Reasoning | `benchmark_visual_reasoning.py` | COCO val2017 | [Visual Reasoning](benchmark_descriptions/VisualReasoning.md) |
| 17 | Document VQA | `benchmark_docvqa.py` | COCO val2017 | [Document VQA](benchmark_descriptions/DocumentVQA.md) |
| 18 | Emotion Detection | `benchmark_emotion.py` | COCO val2017 | [Emotion Detection](benchmark_descriptions/EmotionDetection.md) |
| 19 | Human Intention Rec. | `benchmark_hir.py` | COCO val2017 | [HIR](benchmark_descriptions/HIR.md) |
| 20 | Document Understanding | `benchmark_doc_understanding.py` | COCO val2017 | [Document Understanding](benchmark_descriptions/DocumentUnderstanding.md) |
| 21 | Embedding Extraction | `benchmark_embedding.py` | COCO val2017 | Vision encoder embedding extraction |
| 22 | Zero-Shot Detection | `benchmark_zeroshot_detection.py` | COCO val2017 | Open-vocabulary detection (B3/Avg Acc@50) |
| 23 | VEQ (Vision Encoders) | `benchmark_veq.py` | COCO val2017 | [Visual Embedding Quality](benchmark_descriptions/VEQ.md) |
| 24 | VEQ (VLMs) | `benchmark_veq_vlm.py` | COCO val2017 | [Visual Embedding Quality](benchmark_descriptions/VEQ.md) |

## Models per Benchmark

| Model | Det. | ZS-Det | Pose | OBB | Ground | Caption | VQA | ZS-Cls | Seg. | Scene | Track | 6D | OCR | Pt. | Count | Reas. | DocVQA | Emo. | HIR | Doc. | Emb. | ZS-Det(v2) | VEQ | VEQ-V |
|-------|:----:|:------:|:----:|:---:|:------:|:-------:|:---:|:------:|:----:|:-----:|:-----:|:--:|:---:|:---:|:-----:|:-----:|:------:|:----:|:---:|:----:|:----:|:---------:|:---:|:-----:|
| Florence-2 | ✓ | ✓ | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| PaliGemma2 | ✓* | ✓ | — | — | — | ✓ | ✓ | ✓ | — | ✓ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| Qwen3-Instruct | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| Qwen3-Thinking | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| Phi-4-Multimodal | ✓ | ✓ | — | — | — | ✓ | ✓ | ✓ | — | ✓ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| Phi-3.5-Vision | — | ✓ | — | — | — | ✓ | ✓ | ✓ | — | ✓ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| Cosmos-Reason1 | — | ✓ | — | — | — | ✓ | ✓ | ✓ | — | ✓ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| Llama-3.2-11B | — | — | — | — | — | ✓ | ✓ | ✓ | — | ✓ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| LLaVA-v1.6 | — | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| LLaVA-OneVision | — | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| LLaVA-NeXT-Video | — | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| DiffusionGemma | — | — | — | — | — | ✓ | ✓ | — | — | — | — | — | — | — | ✓ | — | — | — | — | — | — | ✓ | — | — |
| YOLO26/11 (Det) | ✓ | — | — | — | — | — | — | — | — | — | ✓ | ✓ | — | — | — | — | — | — | — | — | — | — | — | — |
| YOLO26/11 (Pose) | — | — | ✓ | — | — | — | — | — | — | — | ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| YOLO26/11 (OBB) | — | — | — | ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| YOLO-World | — | ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| YOLOE | — | ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| LocateAnything | ✓ | — | — | — | ✓ | — | — | — | ✓ | — | — | — | ✓ | ✓ | — | — | — | — | — | — | — | — | — | — |
| LocateAnything (TRT) | ✓ | — | — | — | ✓ | — | — | — | ✓ | — | — | — | ✓ | ✓ | — | — | — | — | — | — | — | — | — | — |
| SigLIP2 | — | — | — | — | — | ✓† | — | ✓ | — | — | — | — | — | — | — | — | — | — | — | — | ✓ | — | ✓ | ✓ |
| MoonViT | — | — | — | — | — | ✓† | — | ✓ | — | — | — | — | — | — | — | — | — | — | — | — | ✓ | — | ✓ | ✓ |
| DINOv3 | — | — | — | — | — | ✓† | — | ✓ | — | — | — | — | — | — | — | — | — | — | — | — | ✓ | — | ✓ | ✓ |
| DINOtool | — | — | — | — | — | ✓† | — | ✓ | — | — | — | — | — | — | — | — | — | — | — | — | ✓ | — | ✓ | ✓ |

> `✓*` = PaliGemma OD is prompt-based. `✓†` = Vision encoders output structured text, not freeform captions.

## Commands

### Run all benchmarks

```bash
python3 scripts/benchmark_all.py --max-images 50
```

### Run selected tasks

```bash
python3 scripts/benchmark_all.py --tasks captioning vqa zeroshot_od --max-images 50
```

### Run a single model on a single task

Each model must be run with its own venv Python:

```bash
# Florence-2 captioning
../VLMcollection/florence-2/.venv/bin/python scripts/benchmark_caption.py --model florence2 --max-images 100

# Phi-4 zero-shot classification
../VLMcollection/phi-4_multimodal/.venv/bin/python scripts/benchmark_zeroshot_cls.py --model phi4_multimodal --max-images 50

# YOLO26 object detection
../VLMcollection/yolo11-26/.venv/bin/python scripts/benchmark_od.py --model yolo26 --max-images 100

# Qwen3 zero-shot OD
../VLMcollection/qwen3-vl_instruct/.venv/bin/python scripts/benchmark_zeroshot_od.py --model qwen3_native --max-images 100

# SigLIP2 VEQ
../VLMcollection/siglip2/.venv/bin/python scripts/benchmark_veq.py --model siglip2 --max-images 200
```

### Generate report

```bash
python3 generate_report.py
# Produces: Benchmark_Results.md, charts/*.png
```

### Attention Implementation Benchmark

```bash
python3 attention_impl_results.json  # or run the benchmark directly
```

See [attention_implementation_benchmark.md](attention_implementation_benchmark.md) for flash-attn vs SDPA vs eager comparison results.

## Model Reference

| `--model` key | Model | Venv | VRAM |
|---------------|-------|------|------|
| `florence2` | Florence-2-large-ft | `florence-2/.venv` | ~4 GB |
| `paligemma` | PaliGemma2-3B-mix | `paligemma/.venv` | ~4 GB |
| `phi_vision` | Phi-3.5-Vision-4.2B | `phi-vision/.venv` | ~10 GB |
| `phi4_multimodal` | Phi-4-Multimodal-14B | `phi-4_multimodal/.venv` | ~16 GB |
| `qwen3_native` | Qwen3-VL-8B-Instruct | `qwen3-vl_instruct/.venv` | ~16 GB |
| `qwen3_thinking` | Qwen3-VL-8B-Thinking | `qwen3-vl_thinking/.venv` | ~16 GB |
| `cosmos_nemotron` | Cosmos-Reason1-7B | `cosmos-nemotron/.venv` | ~14 GB |
| `llama_vision` | Llama-3.2-11B-Vision | `llama-vision/.venv` | ~12 GB |
| `locate_anything` | LocateAnything-3B | `locate_anything/.venv` | ~8 GB |
| `locate_anything_trt` | LocateAnything-3B (TRT) | `locate_anything/.venv` | ~6 GB |
| `siglip2` | SigLIP2 | `siglip2/.venv` | ~2 GB |
| `moonvit` | MoonViT | `moonvit/.venv` | ~2 GB |
| `dinov3` | DINOv2/v3 | `dinov3/.venv` | ~1 GB |
| `dinotool` | DINOtool | `DINOtool/.venv` | ~1 GB |
| `yolo26` / `yolo11` | YOLO26/11 (Detect) | `yolo11-26/.venv` | ~1 GB |
| `yolo26_pose` / `yolo11_pose` | YOLO26/11 (Pose) | `yolo11-26/.venv` | ~1 GB |
| `yolo26_obb` / `yolo11_obb` | YOLO26/11 (OBB) | `yolo11-26/.venv` | ~1 GB |
| `yolo_world` | YOLO-World-v2-x | `yolo11-26/.venv` | ~2 GB |
| `yoloe` | YOLOE-26m | `yolo11-26/.venv` | ~1 GB |
| `diffusion_gemma` | DiffusionGemma-26B | `diffusion_gemma_vl/.venv` | ~20 GB |
| `llava_*` | LLaVA models (5) | `Llava/.venv` | ~14-34 GB |

### Aliases

| Shortcut | Full key | Shortcut | Full key |
|----------|----------|----------|----------|
| `f2` / `florence` | `florence2` | `la` | `locate_anything` |
| `pg` / `gemma` | `paligemma` | `qwen3` | `qwen3_native` |
| `phi` / `phi3` | `phi_vision` | `thinking` | `qwen3_thinking` |
| `phi4` / `phi4mm` | `phi4_multimodal` | `cosmos` | `cosmos_nemotron` |
| `llama` | `llama_vision` | `yolo` / `yolo26n` | `yolo26` |
| `s2` / `siglip` | `siglip2` | `yolo11n` | `yolo11` |
| `mv` / `moon` | `moonvit` | `yoloe` | `yoloe` |
| `d3` / `dino` | `dinov3` | `yolo_world` | `yolo_world` |

## Results

Results are stored in `results/` as `{model}_{task}_stats.json`.

```bash
# View all results
ls results/*_stats.json

# Generate report + charts
python3 generate_report.py
```

- **[Full Benchmark Results](Benchmark_Results.md)** — detailed metrics for all tasks
- **[Attention Implementation Benchmark](attention_implementation_benchmark.md)** — flash-attn vs SDPA vs eager comparison

## Datasets

| Dataset | Path | Content |
|---------|------|---------|
| COCO val2017 | `/mnt/HDD1/Project_Data/public_datasets/coco/` | 5000 images, 80 classes |
| COCO Keypoints | `/mnt/HDD1/Project_Data/public_datasets/coco/` | 17 keypoints per image |
| COCO Captions | `/mnt/HDD1/Project_Data/public_datasets/coco/annotations/captions_val2017.json` | 5 captions per image |
| DOTA-v1.0 | `/mnt/HDD1/Project_Data/public_datasets/dotav1/` | Aerial images, 15 classes |
| Tiny ImageNet | `/mnt/HDD1/Project_Data/public_datasets/tiny-imagenet-200/` | 200 classes, 50 val each |
| MOT17 | `/mnt/HDD1/Project_Data/public_datasets/MOT17/` | 7 sequences |
| Linemod | `/mnt/HDD1/Project_Data/public_datasets/linemod/LINEMOD/` | 13 objects, 6D poses |

## Reproducibility

Each benchmark script selects images deterministically:
- **COCO-based tasks**: First `N` images from val2017 annotations (fixed order).
- **VQA**: Shuffled with `random.Random(42)` for a fixed split.
- **Classification**: First `N` Tiny ImageNet validation images.
- **OBB**: First `N` DOTA validation images.

This ensures every model sees the same images for a given task.

## LLaVA Models

LLaVA models run via **subprocess dispatch**: `benchmark_all.py` calls the benchmark script from `Llava/.venv`, which then subprocess-calls `Llava/run.py` for each image. All 5 LLaVA models share one venv. The 34B model uses 4-bit NF4 quantization.

## Project Structure

```
VLMbenchmark/
├── scripts/                     # All benchmark scripts
│   ├── benchmark_all.py         # Main orchestrator
│   ├── benchmark_od.py          # Object detection
│   ├── benchmark_zeroshot_od.py # Zero-shot OD (mAP)
│   ├── benchmark_zeroshot_cls.py # Zero-shot classification
│   ├── benchmark_caption.py     # Image captioning
│   ├── benchmark_vqa.py         # Visual QA
│   ├── benchmark_veq.py         # VEQ (vision encoders)
│   ├── benchmark_veq_vlm.py     # VEQ (VLMs)
│   ├── common.py                # Shared model loaders, metrics, utilities
│   └── ...
├── benchmark_descriptions/      # Detailed benchmark descriptions
├── results/                     # JSON stats per model per task
├── charts/                      # Generated chart PNGs
├── samples/                     # Sample images
├── datasets/                    # Dataset download scripts
├── Benchmark_Results.md         # Full benchmark results
├── attention_implementation_benchmark.md  # Attn impl comparison
├── generate_report.py           # Generates Benchmark_Results.md + charts
└── run_benchmarks.sh            # Shell script to run all benchmarks
```
