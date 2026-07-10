# VLM Benchmark

Multi-task benchmark for vision-language models on COCO, DOTA-v1.0, Tiny ImageNet, and MOT17.

## Quick Start

```bash
# Run all tasks with 50 images per model (takes hours)
python3 scripts/benchmark_all.py --max-images 50

# Run specific tasks
python3 scripts/benchmark_all.py --tasks captioning vqa --max-images 50

# Generate report after benchmark is complete
python3 generate_report.py
```

Each model is automatically dispatched to its own virtual environment under `../VLMcollection/`.

## Skill Matrix

| Model | Det. | Pose | OBB | Ground | Caption | VQA | Class. | Seg. | Scene | Track | 6D Pose | OCR | Point |
|-------|:----:|:----:|:---:|:------:|:-------:|:---:|:------:|:----:|:-----:|:-----:|:-------:|:---:|:-----:|
| LocateAnything-3B | ✓ | — | — | ✓ | — | — | — | ✓ | — | — | — | ✓ | ✓ |
| LocateAnything-3B (TRT) | ✓ | — | — | ✓ | — | — | — | ✓ | — | — | — | ✓ | ✓ |
| Qwen3-VL-8B-Instruct | ✓ | — | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| Qwen3-VL-8B-Thinking | ✓ | — | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| YOLO26n / YOLO11n | ✓ | ✓ | ✓ | — | — | — | — | — | — | ✓ | ✓ | — | — |
| Florence-2-large-ft | ✓ | — | — | ✓ | ✓ | ✓ | — | ✓ | ✓ | — | — | ✓ | — |
| PaliGemma2-3B-mix | ✓* | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| Llama-3.2-11B-Vision | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| Phi-3.5-Vision-4.2B | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| Cosmos-Reason1-7B | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| LLaVA-v1.6-Mistral-7B | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| LLaVA-OneVision-Qwen2-7B | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| LLaVA-NeXT-Video-7B | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| LLaVA-NeXT-Video-34B (4-bit) | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| LLaVA-Phi-3-Mini-4B | — | — | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — |
| DiffusionGemma-26B | — | — | — | — | ✓ | ✓ | — | — | — | — | — | — | — |
| SigLIP2 (ZS) | — | — | — | — | ✓† | — | ✓ | — | — | — | — | — | — |
| MoonViT (ZS) | — | — | — | — | ✓† | — | ✓ | — | — | — | — | — | — |
| DINOv3 (ZS) | — | — | — | — | ✓† | — | ✓ | — | — | — | — | — | — |
| DINOtool (ZS) | — | — | — | — | ✓† | — | ✓ | — | — | — | — | — | — |

> `✓*` = PaliGemma OD is prompt-based (bbox tokens). `✓†` = Vision-only encoders output structured text, not freeform captions.

## Tasks & Commands

| Script | Task | Dataset | `--model` keys |
|--------|------|---------|----------------|
| `benchmark_od.py` | Object Detection | COCO val2017 | `locate_anything`, `locate_anything_trt`, `qwen3_native`, `qwen3_thinking`, `yolo26`, `yolo26s-yolo26x`, `yolo11-yolo11x`, `florence2`, `paligemma` |
| `benchmark_pose.py` | Pose Estimation | COCO Keypoints | `yolo26_pose`, `yolo26s_pose`, `yolo11_pose`, `yolo11s_pose` |
| `benchmark_obb.py` | Oriented BBox | DOTA-v1.0 | `yolo26_obb`, `yolo26s_obb`, `yolo11_obb`, `yolo11s_obb` |
| `benchmark_grounding.py` | Phrase Grounding | COCO val2017 | `locate_anything`, `locate_anything_trt`, `qwen3_native`, `qwen3_thinking`, `florence2` |
| `benchmark_caption.py` | Image Captioning | COCO Captions | `florence2`, `paligemma`, `llama_vision`, `phi_vision`, `cosmos_nemotron`, `qwen3_native`, `qwen3_thinking`, `diffusion_gemma*`, `siglip2`, `moonvit`, `dinov3`, `dinotool`, `llava_v16_mistral`, `llava_onevision`, `llava_next_video_7b`, `llava_next_video_34b`, `phi3_vision` |
| `benchmark_vqa.py` | Visual QA | COCO val2017 | `florence2`, `paligemma`, `llama_vision`, `phi_vision`, `cosmos_nemotron`, `qwen3_native`, `qwen3_thinking`, `diffusion_gemma*`, `llava_v16_mistral`, `llava_onevision`, `llava_next_video_7b`, `llava_next_video_34b`, `phi3_vision` |
| `benchmark_classification.py` | Zero-Shot Class. | Tiny ImageNet | `dinotool`, `dinov3`, `siglip2`, `moonvit` |
| `benchmark_segmentation.py` | Segmentation | COCO val2017 | `florence2`, `locate_anything`, `locate_anything_trt` |
| `benchmark_scene.py` | Scene Analysis | COCO val2017 | `florence2`, `paligemma`, `llama_vision`, `phi_vision`, `cosmos_nemotron`, `qwen3_native`, `qwen3_thinking`, `llava_v16_mistral`, `llava_onevision`, `llava_next_video_7b`, `llava_next_video_34b`, `phi3_vision` |
| `benchmark_tracking.py` | Multi-Object Track | MOT17 | `yolo26`, `yolo26s-yolo26m`, `yolo11`, `yolo11s-yolo11m` |
| `benchmark_6dpose.py` | 6D Pose Detection | Linemod | `yolo26`, `yolo26s-yolo26m`, `yolo11`, `yolo11s-yolo11m` |
| `benchmark_ocr.py` | OCR / Text Det. | Synthetic COCO | `locate_anything`, `locate_anything_trt`, `florence2` |
| `benchmark_pointing.py` | Pointing / 2D KP | COCO val2017 | `locate_anything`, `locate_anything_trt` |

## Model Reference

| `--model` key | Model | Tasks | Venv | Notes |
|---------------|-------|-------|------|-------|
| `locate_anything` | LocateAnything-3B | OD, Ground, Seg, OCR, Point | `locate_anything/.venv` | |
| `locate_anything_trt` | LocateAnything-3B (TRT) | OD, Ground, Seg, OCR, Point | `locate_anything/.venv` | TensorRT engine |
| `qwen3_native` | Qwen3-VL-8B-Instruct | OD, OBB, Ground, Caption, VQA, Scene | `qwen3-vl_instruct/.venv` | |
| `qwen3_thinking` | Qwen3-VL-8B-Thinking | OD, OBB, Ground, Caption, VQA, Scene | `qwen3-vl_thinking/.venv` | |
| `yolo26` / `yolo26s/m/l/x` | YOLO26n/s/m/l/x (Detect) | OD, Track, 6D Pose | `yolo11-26/.venv` | |
| `yolo11` / `yolo11s/m/l/x` | YOLO11n/s/m/l/x (Detect) | OD, Track, 6D Pose | `yolo11-26/.venv` | |
| `yolo26_pose` / `yolo26s_pose` | YOLO26n/s (Pose) | Pose | `yolo11-26/.venv` | |
| `yolo11_pose` / `yolo11s_pose` | YOLO11n/s (Pose) | Pose | `yolo11-26/.venv` | |
| `yolo26_obb` / `yolo26s_obb` | YOLO26n/s (OBB) | OBB | `yolo11-26/.venv` | |
| `yolo11_obb` / `yolo11s_obb` | YOLO11n/s (OBB) | OBB | `yolo11-26/.venv` | |
| `florence2` | Florence-2-large-ft | OD, Ground, Caption, VQA, Seg, Scene, OCR | `florence-2/.venv` | |
| `paligemma` | PaliGemma2-3B-mix | OD*, Caption, VQA, Scene | `paligemma/.venv` | |
| `llama_vision` | Llama-3.2-11B-Vision | Caption, VQA, Scene | `llama-vision/.venv` | |
| `phi_vision` | Phi-3.5-Vision-4.2B | Caption, VQA, Scene | `phi-vision/.venv` | |
| `cosmos_nemotron` | Cosmos-Reason1-7B | Caption, VQA, Scene | `cosmos-nemotron/.venv` | |
| `diffusion_gemma*` | DiffusionGemma-26B | Caption, VQA | `diffusion_gemma_vl/.venv` | Variants: `diffusion_gemma`, `_yolo`, `_yolo_pose`, `_yolo_obb`, `_siglip2`, `_moonvit` |
| `llava_v16_mistral` | LLaVA-v1.6-Mistral-7B | Caption, VQA, Scene | `Llava/.venv` | Subprocess dispatch |
| `llava_onevision` | LLaVA-OneVision-Qwen2-7B | Caption, VQA, Scene | `Llava/.venv` | Subprocess dispatch |
| `llava_next_video_7b` | LLaVA-NeXT-Video-7B | Caption, VQA, Scene | `Llava/.venv` | Subprocess dispatch |
| `llava_next_video_34b` | LLaVA-NeXT-Video-34B-DPO | Caption, VQA, Scene | `Llava/.venv` | 4-bit quantized, subprocess dispatch |
| `phi3_vision` | LLaVA-Phi-3-Mini-4B | Caption, VQA, Scene | `Llava/.venv` | Subprocess dispatch |
| `siglip2` | SigLIP2 (ZS) | Caption, Class. | `siglip2/.venv` | |
| `moonvit` | MoonViT (ZS) | Caption, Class. | `moonvit/.venv` | |
| `dinov3` | DINOv2/v3 (ZS) | Caption, Class. | `dinov3/.venv` | |
| `dinotool` | DINOtool | Caption, Class. | `DINOtool/.venv` | |

### LLaVA Models

LLaVA models run via **subprocess dispatch**: `benchmark_all.py` calls `benchmark_caption.py` (etc.) from the `Llava/.venv`, which then subprocess-calls `Llava/run.py` for each image. This means:

- All 5 LLaVA models share one venv (`Llava/.venv`).
- The 34B model uses 4-bit NF4 quantization (`--quantize` flag).
- The 34B model takes ~20 min to load from HDD and ~80s per image.
- Each image triggers a separate model load (no batching across images yet).

### Aliases

| Shortcut | Full key |
|----------|----------|
| `la` | `locate_anything` |
| `qwen3` | `qwen3_native` |
| `thinking` | `qwen3_thinking` |
| `yolo` / `yolo26n` | `yolo26` |
| `yolo11n` | `yolo11` |
| `yolo_pose` | `yolo26_pose` |
| `yolo_obb` | `yolo26_obb` |
| `f2` / `florence` | `florence2` |
| `pg` / `gemma` | `paligemma` |
| `llama` / `llama3` | `llama_vision` |
| `phi` / `phi3` | `phi_vision` |
| `cosmos` / `nemotron` | `cosmos_nemotron` |
| `dg` / `diffusion_gemma_vl` | `diffusion_gemma` |
| `d3` / `dino` / `dinov3` | `dinov3` |
| `s2` / `siglip` | `siglip2` |
| `mv` / `moon` | `moonvit` |
| `llava` / `llava_mistral` | `llava_v16_mistral` |
| `llava_onevision` | `llava_onevision` |
| `llava_next7b` | `llava_next_video_7b` |
| `llava_next34b` | `llava_next_video_34b` |
| `phi3v` | `phi3_vision` |

## Running a Single Model / Task

Each script must be run with the model's own venv Python. Examples:

```bash
# Captioning with Florence-2
../VLMcollection/florence-2/.venv/bin/python \
    scripts/benchmark_caption.py --model florence2 --max-images 100

# Captioning with a LLaVA model (all share Llava venv)
../VLMcollection/Llava/.venv/bin/python \
    scripts/benchmark_caption.py --model phi3_vision --max-images 50

# VQA with Qwen3
../VLMcollection/qwen3-vl_instruct/.venv/bin/python \
    scripts/benchmark_vqa.py --model qwen3_native --max-questions 100

# Object Detection with YOLO26
../VLMcollection/yolo11-26/.venv/bin/python \
    scripts/benchmark_od.py --model yolo26 --max-images 100
```

## Run All Benchmarks

```bash
# Full run (50 images per model, all tasks)
python3 scripts/benchmark_all.py --max-images 50

# Or use any model venv (they all have the required benchmark dependencies)
../VLMcollection/yolo11-26/.venv/bin/python \
    scripts/benchmark_all.py --max-images 50

# Selected tasks only
python3 scripts/benchmark_all.py --tasks captioning vqa scene_analysis --max-images 50
```

The orchestrator (`benchmark_all.py`) dispatches each model to its own `{collection}/.venv/bin/python`, so each model runs in its correct environment automatically.

## Reproducibility

Each benchmark script selects images deterministically:
- **COCO-based tasks**: First `N` images from COCO val2017 annotations (fixed order).
- **VQA**: Shuffled with `random.Random(42)` for a fixed split.
- **Classification**: First `N` Tiny ImageNet validation images.
- **OBB**: First `N` DOTA validation images.

This ensures every model sees the same images for a given task.

## Results

Stored in `results/` as `{model}_{task}_stats.json`. Run `benchmark_all.py` to produce a combined summary at `results/all_benchmarks_summary.json`.

```bash
# Generate report with charts after benchmark is done
python3 generate_report.py
# Produces: Benchmark_Results.md, charts/*.png
```

## Datasets

| Dataset | Path | Content |
|---------|------|---------|
| COCO val2017 | `/mnt/HDD1/Project_Data/public_datasets/coco/` | 5000 images, 80 classes |
| COCO Keypoints | `/mnt/HDD1/Project_Data/public_datasets/coco/` | 5000 images, 17 keypoints |
| COCO Captions | `/mnt/HDD1/Project_Data/public_datasets/coco/annotations/captions_val2017.json` | 5000 images, 5 captions each |
| DOTA-v1.0 | `/mnt/HDD1/Project_Data/public_datasets/dotav1/` | Aerial images, 15 classes |
| Tiny ImageNet | `/mnt/HDD1/Project_Data/public_datasets/tiny-imagenet-200/` | 200 classes, 50 val each |
| MOT17 | `/mnt/HDD1/Project_Data/public_datasets/MOT17/` | 7 sequences |
| Linemod | `/mnt/HDD1/Project_Data/public_datasets/linemod/LINEMOD/` | 13 objects, 6D poses |

## Metrics

| Task | Metrics |
|------|---------|
| Object Detection | mAP@50:95, mAP@50, FPS, avg inference time |
| Pose Estimation | AP@50:95 (keypoints), AP@50 (keypoints), FPS |
| OBB Detection | mAP@50:95, mAP@50, FPS |
| Phrase Grounding | Acc@50 (IoU≥0.5), FPS |
| Image Captioning | BLEU-4, ROUGE-L, CIDEr, FPS |
| Visual Question Answering | Accuracy (exact match), FPS |
| Classification | Top-1 accuracy, Top-5 accuracy, FPS |
| Segmentation | mIoU, FPS |
| Scene Analysis | Accuracy (exact label match), FPS |
| Multi-Object Tracking | HOTA, MOTA, IDF1, FPS |
| 6D Pose Detection | ADD(-S) accuracy, FPS |
| OCR / Text Detection | F1 (text spotting), FPS |
| Pointing | Accuracy (pixel distance < threshold), FPS |
