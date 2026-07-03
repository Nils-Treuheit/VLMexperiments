# Benchmark

Multi-task benchmark for visual models on COCO and DOTA-v1.0.

## Skill Matrix — Which Model Does What?

| Model | Object Det. | Pose Est. | OBB Det. | Phrase Grounding | Captioning | VQA |
|-------|:-----------:|:---------:|:--------:|:----------------:|:----------:|:---:|
| LocateAnything-3B | ✓ | — | — | ✓ | — | — |
| Qwen3-VL-8B-Instruct | ✓ | — | — | ✓ | ✓ | ✓ |
| Qwen3-VL-8B-Thinking | ✓ | — | — | ✓ | ✓ | ✓ |
| YOLO26n (Detect) | ✓ | — | — | — | — | — |
| YOLO26n (Pose) | — | ✓ | — | — | — | — |
| YOLO26n (OBB) | — | — | ✓ | — | — | — |
| Florence-2-large-ft | ✓ | — | — | ✓ | ✓ | ✓ |
| PaliGemma2-3B-mix | ✓* | — | — | — | ✓ | ✓ |
| Llama-3.2-11B-Vision | — | — | — | — | ✓ | ✓ |
| Phi-3.5-Vision-4.2B | — | — | — | — | ✓ | ✓ |
| Cosmos-Reason1-7B | — | — | — | — | ✓ | ✓ |
| DiffusionGemma-26B (YOLO+DG) | — | — | — | — | ✓ | ✓ |
| SigLIP2 (Zero-shot Description) | — | — | — | — | ✓† | — |
| MoonViT (Zero-shot Description) | — | — | — | — | ✓† | — |
| DINOv3 (Zero-shot Description) | — | — | — | — | ✓† | — |

> `✓*` = PaliGemma OD is prompt-based (tokenizes bbox tokens), results may vary.
> `✓†` = Vision encoders output structured zero-shot text (objects, scene, attributes) rather than freeform captions.

## Tasks

| Script | Task | Dataset | Models |
|--------|------|---------|--------|
| `benchmark_od.py` | Object Detection | COCO val2017 | locate_anything, qwen3_native, qwen3_thinking, yolo26, florence2, paligemma |
| `benchmark_od.py` | Object Detection | DOTA-v1.0 | locate_anything, qwen3_native, qwen3_thinking |
| `benchmark_pose.py` | Pose Estimation | COCO Keypoints | yolo26_pose |
| `benchmark_obb.py` | Oriented BBox | DOTA-v1.0 | yolo26_obb |
| `benchmark_grounding.py` | Phrase Grounding | COCO val2017 | locate_anything, qwen3_native, qwen3_thinking, florence2 |
| `benchmark_caption.py` | Image Captioning | COCO Captions val2017 | florence2, paligemma, llama_vision, phi_vision, cosmos_nemotron, qwen3_native, qwen3_thinking, diffusion_gemma, siglip2, moonvit, dinov3 |
| `benchmark_vqa.py` | Visual Question Answering | COCO val2017 | florence2, paligemma, llama_vision, phi_vision, cosmos_nemotron, qwen3_native, qwen3_thinking, diffusion_gemma |
| `benchmark_all.py` | All supported tasks | — | All models |

## Quick Start

Each model must be run with **its own virtual environment**.

All models are located under `/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/`.

### Object Detection (COCO)
```bash
# LocateAnything-3B
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/locate_anything/.venv/bin/python \
    scripts/benchmark_od.py --model locate_anything --max-images 100

# Florence-2 (native <OD> task, multi-label)
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/florence-2/.venv/bin/python \
    scripts/benchmark_od.py --model florence2 --max-images 100

# YOLO11n
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/yolo11-26/.venv/bin/python \
    scripts/benchmark_od.py --model yolo26 --max-images 100
```

### Phrase Grounding
```bash
# Florence-2 (referring expression segmentation)
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/florence-2/.venv/bin/python \
    scripts/benchmark_grounding.py --model florence2 --max-images 100

# LocateAnything
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/locate_anything/.venv/bin/python \
    scripts/benchmark_grounding.py --model locate_anything --max-images 100
```

### Image Captioning
```bash
# Florence-2
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/florence-2/.venv/bin/python \
    scripts/benchmark_caption.py --model florence2 --max-images 100

# Qwen3-VL-Instruct
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/qwen3-vl_instruct/.venv/bin/python \
    scripts/benchmark_caption.py --model qwen3_native --max-images 100

# Llama-3.2-Vision
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/llama-vision/.venv/bin/python \
    scripts/benchmark_caption.py --model llama_vision --max-images 100

# DiffusionGemma (YOLO feeder + text diffusion)
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/diffusion_gemma_vl/.venv/bin/python \
    scripts/benchmark_caption.py --model diffusion_gemma --max-images 100

# SigLIP2 (zero-shot structured description)
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/siglip2/.venv/bin/python \
    scripts/benchmark_caption.py --model siglip2 --max-images 100

# MoonViT (zero-shot structured description)
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/moonvit/.venv/bin/python \
    scripts/benchmark_caption.py --model moonvit --max-images 100

# DINOv3 (zero-shot structured description, requires HF login + license)
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/dinov3/.venv/bin/python \
    scripts/benchmark_caption.py --model dinov3 --max-images 100
```

### Visual Question Answering
```bash
# PaliGemma2
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/paligemma/.venv/bin/python \
    scripts/benchmark_vqa.py --model paligemma --max-questions 200

# Phi-3.5-Vision
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/phi-vision/.venv/bin/python \
    scripts/benchmark_vqa.py --model phi_vision --max-questions 200

# DiffusionGemma
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/diffusion_gemma_vl/.venv/bin/python \
    scripts/benchmark_vqa.py --model diffusion_gemma --max-questions 200
```

### Run Everything
```bash
# Uses yolo venv (has pycocotools); spawns subprocesses with each model's venv
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/yolo11-26/.venv/bin/python \
    scripts/benchmark_all.py --max-images 25

# Run only specific tasks
/mnt/HDD1/Project_Code/to_be_merged/VLMexperiments/VLMcollection/yolo11-26/.venv/bin/python \
    scripts/benchmark_all.py --tasks captioning vqa --max-images 50
```

## Models

| Key | Model | Tasks | Venv |
|-----|-------|-------|------|
| `locate_anything` | LocateAnything-3B | OD, Grounding | `locate_anything/.venv` |
| `qwen3_native` | Qwen3-VL-8B-Instruct | OD, Grounding, OD(DOTA), Caption, VQA | `qwen3-vl_instruct/.venv` |
| `qwen3_thinking` | Qwen3-VL-8B-Thinking | OD, Grounding, OD(DOTA), Caption, VQA | `qwen3-vl_thinking/.venv` |
| `yolo26` | YOLO26n (Detect) | OD | `yolo11-26/.venv` |
| `yolo26_pose` | YOLO26n (Pose) | Pose | `yolo11-26/.venv` |
| `yolo26_obb` | YOLO26n (OBB) | OBB | `yolo11-26/.venv` |
| `florence2` | Florence-2-large-ft | OD, Grounding, Caption, VQA | `florence-2/.venv` |
| `paligemma` | PaliGemma2-3B-mix | OD*, Caption, VQA | `paligemma/.venv` |
| `llama_vision` | Llama-3.2-11B-Vision | Caption, VQA | `llama-vision/.venv` |
| `phi_vision` | Phi-3.5-Vision-4.2B | Caption, VQA | `phi-vision/.venv` |
| `cosmos_nemotron` | Cosmos-Reason1-7B | Caption, VQA | `cosmos-nemotron/.venv` |
| `diffusion_gemma` | DiffusionGemma-26B (YOLO+DG) | Caption, VQA | `diffusion_gemma_vl/.venv` |
| `dinov3` | DINOv3 (Zero-shot Description) | Caption† | `dinov3/.venv` |
| `siglip2` | SigLIP2 (Zero-shot Description) | Caption† | `siglip2/.venv` |
| `moonvit` | MoonViT (Zero-shot Description) | Caption† | `moonvit/.venv` |

> `✓*` = PaliGemma OD is prompt-based.
> `†` = Vision encoders output structured zero-shot text, not freeform captions.

## Aliases

| Shortcut | Full key |
|----------|----------|
| `la` | `locate_anything` |
| `qwen3` | `qwen3_native` |
| `thinking` | `qwen3_thinking` |
| `yolo` / `yolo26n` | `yolo26` |
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

## Metrics

| Task | Metrics |
|------|---------|
| Object Detection | mAP@50:95, mAP@50, FPS, avg inference time |
| Pose Estimation | AP@50:95 (keypoints), AP@50 (keypoints), FPS |
| OBB Detection | mAP@50:95, mAP@50, FPS |
| Phrase Grounding | Acc@50 (IoU≥0.5), FPS |
| Image Captioning | BLEU-4, ROUGE-L, CIDEr, FPS |
| Visual Question Answering | Accuracy (exact match), FPS |

## Datasets

| Dataset | Path | Content |
|---------|------|---------|
| COCO val2017 | `/mnt/HDD1/Project_Data/public_datasets/coco/` | 5000 images, 80 classes |
| COCO Keypoints | `/mnt/HDD1/Project_Data/public_datasets/coco/` | 5000 images, 17 keypoints |
| COCO Captions | `/mnt/HDD1/Project_Data/public_datasets/coco/annotations/captions_val2017.json` | 5000 images, 5 captions each |
| DOTA-v1.0 | `/mnt/HDD1/Project_Data/public_datasets/dotav1/` | Aerial images, 15 classes |

Download DOTA: `python download_datasets.py --dota 200`

## Results

Stored in `results/` as `{model}_{task}_stats.json`. Run `benchmark_all.py` to produce a combined summary at `results/all_benchmarks_summary.json`.
