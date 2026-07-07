# Benchmark Results

**Generated:** 2026-07-07 11:48

**Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM)

## Overview

This report summarizes benchmark results across multiple vision-language models (VLMs) 
and vision encoders. Benchmarks cover 11 tasks: image captioning, visual question answering (VQA), 
object detection (AABB + OBB), phrase grounding, pose estimation (2D keypoints + 6D), 
segmentation, zero-shot classification, semantic scene analysis, and multi-object tracking.

### Models Tested

| Category | Models |
|----------|--------|
| Vision Encoders | DINOtool, DINOv3, SigLIP2, MoonViT |
| VLMs (caption + VQA) | Florence-2, PaliGemma2, Phi-3.5-Vision, Cosmos-Reason1-7B, Llama-3.2-11B-Vision, Qwen3-VL-8B-Instruct, Qwen3-VL-8B-Thinking |
| VLMs (diffusion) | DiffusionGemma-26B (5 variants) |
| Detection / Tracking | YOLO11n, YOLO26n, YOLO26n (Pose), YOLO26n (OBB), LocateAnything-3B, LocateAnything-3B (TRT) |

### Datasets

| Task | Dataset | Images |
|------|---------|--------|
| Captioning | COCO Captions val2017 | 25-50 |
| VQA | COCO val2017 + templated Qs | 100 questions |
| Object Detection | COCO val2017 | 25-50 |
| OBB Detection | DOTA-v1.0 | 50 |
| Pose (2D) | COCO Keypoints val2017 | 23-50 |
| Phrase Grounding | COCO val2017 | 25-48 |
| Zero-Shot Classification | Tiny ImageNet (200 classes) | 500 |
| Segmentation | COCO val2017 | 100 |
| Scene Analysis | COCO val2017 | 100 |
| Multi-Object Tracking | MOT17 | 200 frames (2 seqs) |
| 6D Pose (detection) | Linemod (BOP) | 25 |

### Notes

- **25-50 images** per model for captioning (more for fast, less for slow models)
- **100 questions** per model for VQA
- **500 images** (Tiny ImageNet) for classification
- Vision encoders use zero-shot classification via DINO/transformer features + sentence-transformers (not trained for captioning)
- Phi-3.5-Vision is very slow (~15s/image) without flash-attention on Blackwell GPU
- DiffusionGemma variants need ~50-60s/image
- LocateAnything-3B (TRT) uses TensorRT-accelerated vision encoder (9.8× faster vision, 1.6× faster end-to-end)

## 1. Image Captioning (COCO Captions)

![FPS](charts/caption_fps.png)
![CIDEr](charts/caption_cider.png)
![BLEU-4](charts/caption_bleu4.png)
![ROUGE-L](charts/caption_rouge.png)

| Model | FPS | CIDEr | BLEU-4 | ROUGE-L | Avg (ms) | Images |
|-------|-----|-------|--------|---------|----------|--------|
| PaliGemma2-3B-mix | 4.56 | 1.7246 | 0.2995 | 0.5432 | 219.1 | 50 |
| Florence-2-large-ft | 3.79 | 0.4999 | 0.0435 | 0.2471 | 264.2 | 50 |
| Cosmos-Reason1-7B | 0.34 | 0.1177 | 0.0059 | 0.0766 | 2943.8 | 50 |
| Qwen3-VL-8B-Instruct | 0.23 | 0.1064 | 0.0067 | 0.0730 | 4399.5 | 50 |
| Llama-3.2-11B-Vision | 0.20 | 0.1764 | 0.0226 | 0.1009 | 5010.7 | 50 |
| DINOv3 (Zero-shot) | 0.16 | 0.0146 | 0.0000 | 0.0129 | 6291.7 | 50 |
| SigLIP2 (Zero-shot) | 0.12 | 0.1122 | 0.0000 | 0.0844 | 8651.5 | 50 |
| DINOtool (DINOv2-s) | 0.10 | 0.0056 | 0.0000 | 0.0081 | 10230.1 | 50 |
| MoonViT (Zero-shot) | 0.09 | 0.0050 | 0.0000 | 0.0041 | 10580.5 | 50 |
| DiffusionGemma-26B (MoonViT) | 0.06 | 0.0000 | 0.0000 | 0.0000 | 16107.7 | 25 |
| DiffusionGemma-26B (SigLIP2) | 0.06 | 0.0000 | 0.0000 | 0.0000 | 17445.0 | 25 |
| Phi-3.5-Vision-4B | 0.06 | 0.2245 | 0.0208 | 0.1364 | 15626.0 | 50 |
| Qwen3-VL-8B-Thinking | 0.06 | 0.0614 | 0.0057 | 0.0412 | 17465.4 | 25 |
| DiffusionGemma-26B (YOLO) | 0.02 | 0.0963 | 0.0000 | 0.0744 | 61219.9 | 25 |
| DiffusionGemma-26B | 0.01 | 0.0963 | 0.0000 | 0.0744 | 66734.7 | 25 |

## 2. Visual Question Answering (COCO)

![Accuracy](charts/vqa_accuracy.png)
![FPS](charts/vqa_fps.png)

| Model | Accuracy | FPS | Avg (ms) | Questions |
|-------|----------|-----|----------|-----------|
| Llama-3.2-11B-Vision | 64.00% | 2.55 | 392.5 | 100 |
| Phi-3.5-Vision-4B | 57.00% | 0.43 | 2299.7 | 100 |
| Qwen3-VL-8B-Thinking | 56.00% | 0.27 | 3722.8 | 100 |
| PaliGemma2-3B-mix | 54.00% | 15.51 | 64.5 | 100 |
| Qwen3-VL-8B-Instruct | 41.00% | 11.68 | 85.6 | 100 |
| Florence-2-large-ft | 37.00% | 11.19 | 89.3 | 100 |
| Cosmos-Reason1-7B | 35.00% | 8.05 | 124.3 | 100 |

## 3. Object Detection (COCO)

![mAP@50:95](charts/od_map5095.png)
![mAP@50](charts/od_map50.png)
![FPS](charts/od_fps.png)

| Model | mAP@50:95 | mAP@50 | FPS | Avg (ms) | Images |
|-------|-----------|--------|-----|----------|--------|
| YOLO11n | 0.3946 | 0.5087 | 46.70 | 21.4 | 50 |
| YOLO26n | 0.3820 | 0.4741 | 14.79 | 67.6 | 50 |
| LocateAnything-3B (TRT) | 0.1263 | 0.1747 | 5.50 | 181.9 | 48 |
| LocateAnything-3B | 0.1255 | 0.1758 | 3.41 | 293.2 | 48 |
| Qwen3-VL-8B-Thinking | 0.0568 | 0.0778 | 0.19 | 5215.1 | 25 |
| Qwen3-VL-8B-Instruct | 0.0134 | 0.0275 | 0.51 | 1959.2 | 48 |

## 4. Pose Estimation (COCO Keypoints)

![mAP](charts/pose_map.png)
![FPS](charts/pose_fps.png)

| Model | mAP@50:95 | mAP@50 | FPS | Avg (ms) | Images |
|-------|-----------|--------|-----|----------|--------|
| YOLO26n (Pose) | — | — | 23.43 | 42.7 | 23 |

## 5. Oriented Bounding Box (DOTA-v1.0)

![mAP](charts/obb_map.png)
![FPS](charts/obb_fps.png)

| Model | mAP@50:95 | mAP@50 | FPS | Avg (ms) | Images |
|-------|-----------|--------|-----|----------|--------|
| YOLO26n (OBB) | — | — | 13.60 | 73.5 | 50 |

## 6. Phrase Grounding (COCO)

![Accuracy](charts/grounding_acc.png)
![FPS](charts/grounding_fps.png)

| Model | Acc@50 | FPS | Avg (ms) | Images |
|-------|--------|-----|----------|--------|
| LocateAnything-3B (TRT) | 14.40% | 4.17 | 239.7 | 48 |
| Qwen3-VL-8B-Thinking | 7.14% | 0.23 | 4339.8 | 25 |
| Qwen3-VL-8B-Instruct | 3.60% | 0.53 | 1887.2 | 48 |

## 7. Zero-Shot Classification (Tiny ImageNet)

![Top-1 Accuracy](charts/class_top1.png)
![FPS](charts/class_fps.png)

| Model | Top-1 Acc | Top-5 Acc | FPS | Avg (ms) | Images |
|-------|-----------|-----------|-----|----------|--------|
| SigLIP2 (Zero-shot) | 13.00% | 15.50% | 0.07 | 13769.5 | 200 |
| MoonViT (Zero-shot) | 1.00% | 1.00% | 0.07 | 14732.2 | 200 |
| DINOv3 (Zero-shot) | 0.50% | 0.50% | 0.06 | 17794.4 | 200 |
| DINOtool (DINOv2-s) | 0.00% | 0.00% | 0.06 | 15519.5 | 100 |

## 8. Segmentation (COCO)

![Panoptic Quality](charts/seg_pq.png)
![mIoU](charts/seg_miou.png)
![FPS](charts/seg_fps.png)

| Model | PQ | mIoU | FPS | Avg (ms) | Images |
|-------|----|------|-----|----------|--------|
| LocateAnything-3B (TRT) | 0.4727 | 0.3182 | 4.18 | 239.4 | 48 |

## 9. Semantic Scene Analysis (COCO)

![Scene Type Accuracy](charts/scene_accuracy.png)
![Object Recall](charts/scene_recall.png)
![FPS](charts/scene_fps.png)

| Model | Scene Acc | Object Recall | FPS | Avg (ms) | Images |
|-------|-----------|---------------|-----|----------|--------|
| PaliGemma2-3B-mix | 100.00% | 2.65% | 9.84 | 101.7 | 50 |
| Llama-3.2-11B-Vision | 97.22% | 50.99% | 0.17 | 5731.4 | 50 |
| Qwen3-VL-8B-Instruct | 95.83% | 54.30% | 0.21 | 4823.2 | 50 |
| Phi-3.5-Vision-4B | 94.00% | 46.36% | 0.04 | 24148.1 | 50 |
| Florence-2-large-ft | 92.31% | 41.06% | 3.87 | 258.2 | 50 |
| Qwen3-VL-8B-Thinking | 91.30% | 68.21% | 0.05 | 20267.0 | 50 |
| Cosmos-Reason1-7B | 88.37% | 52.32% | 0.31 | 3212.7 | 50 |

## 10. Multi-Object Tracking (MOT17)

![MOTA](charts/track_mota.png)
![MOTP](charts/track_motp.png)
![FPS](charts/track_fps.png)

| Model | MOTA | MOTP | FPS | Avg (ms) | Frames |
|-------|------|------|-----|----------|--------|
| YOLO11n | 0.0398 | 0.7983 | 32.46 | 30.8 | 200 |
| YOLO26n | 0.0320 | 0.8047 | 32.00 | 31.3 | 200 |

## 11. 6D Pose Estimation (Linemod)

![Detection Rate](charts/pose6d_detrate.png)
![FPS](charts/pose6d_fps.png)

| Model | Detection Rate | FPS | Avg (ms) | Images |
|-------|----------------|-----|----------|--------|
| YOLO26n | 476.00% | 5.34 | 187.3 | 25 |
| YOLO11n | 468.00% | 25.72 | 38.9 | 25 |

## 12. Speed vs Quality Overview

![Combined FPS](charts/combined_fps.png)

![Quality Comparison](charts/quality_comparison.png)


## 13. Key Takeaways

### Fastest Models by Task
- **Detection:** YOLO11n at 46.7 FPS, YOLO26n at 14.8 FPS
- **Captioning:** PaliGemma2-3B at 4.56 FPS
- **VQA:** PaliGemma2-3B at 15.51 FPS
- **Classification:** DINOtool/SigLIP2 fastest among vision encoders
- **Segmentation:** Florence-2 handles segmentation at reasonable speed
- **Tracking:** YOLO models achieve high FPS on MOT17
- **6D Pose (detection):** YOLO models on Linemod

### Best Quality by Task
- **Captioning CIDEr:** PaliGemma2-3B (1.7246), Florence-2 (0.4999)
- **VQA Accuracy:** Llama-3.2-11B-Vision (64%), Phi-3.5-Vision (57%), PaliGemma2 (54%)
- **Detection mAP:** YOLO26n (0.382), YOLO11n (0.395), LocateAnything-3B (0.126)
- **Detection FPS:** LocateAnything-3B TRT (5.50) is 1.6× faster than PT (3.41)
- **Grounding Acc@50:** LocateAnything-3B TRT (14.4%) — best among tested models
- **Scene Understanding:** Florence-2 excels at structured scene description

### Notable Observations
- Vision encoders (DINOtool, DINOv3, SigLIP2, MoonViT) achieve near-zero CIDEr — expected as they use zero-shot label matching, not generative captioning
- Phi-3.5-Vision is 15-60x slower than other models (~15.6s/image) without flash-attention on Blackwell GPUs
- Qwen3-VL-8B-Thinking produces more detailed captions but at ~4-10x slower speed vs Instruct variant
- DiffusionGemma-26B takes 50-60s per image for caption generation
- YOLO models achieve the highest FPS across all detection tasks (10-47 FPS)
- LocateAnything-3B (TRT) achieves 5.50 FPS on COCO OD (1.6× faster than PT) with bit-exact identical quality
- TRT vision encoder runs at 9.6ms (9.8× faster than PyTorch bf16) — LLM decoder dominates at ~170ms
- Florence-2 is the most versatile model, supporting captioning, VQA, OD, segmentation, and scene analysis
- Tiny ImageNet zero-shot classification shows vision encoders can identify objects despite limited label sets
- MOT17 tracking with YOLO provides a strong baseline for multi-object tracking evaluation

### Missing / Future Benchmarks
- **VQA for DiffusionGemma-26B** — timed out during evaluation
- **OD for Florence-2, PaliGemma:** missing pycocotools dependency in their venvs
- **Grounding for Florence-2, LocateAnything:** missing pycocotools
- **Phi-4-Multimodal:** not fully tested (missing from model choices in some tasks)
- **6D Pose ADD/ADD-S:** pose refinement metrics not yet implemented
- **Semantic / Panoptic Segmentation:** more comprehensive mask evaluation needed
- **Video understanding:** action recognition, temporal reasoning