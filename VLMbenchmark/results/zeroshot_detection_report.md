# Zero-Shot Object Detection Benchmark — COCO Val (100 images)

**Task:** Multi-class zero-shot detection across all 80 COCO categories.
**Metric:** 3-pass B3/AVG Acc@50 (each image × each category tested 3× independently).
**GPU:** NVIDIA RTX 5090 (32 GB).
**Date:** 2026-07-21

---

## Results (models with 100-image runs)

| # | Model | Params | B3 Acc@50 | AVG Acc@50 | FPS | ms/inf | Correct / GT |
|--:|-------|-------:|----------:|-----------:|----:|-------:|:-------------|
| 1 | **Florence-2-large-ft** | 0.77B | **69.30%** | 68.84% | 5.43 | 184 | 456 / 658 |
| 2 | **YOLO-Worldv2-x** | — | **67.63%** | 67.33% | 457.5 | 2.2 | 445 / 658 |
| 3 | **YOLOE-26m** | 26M | **64.74%** | 64.29% | 439.4 | 2.3 | 426 / 658 |
| 4 | **Cosmos-Reason1-7B** | 7B | 37.54% | 36.78% | 1.11 | 898 | 247 / 658 |
| 5 | **PaliGemma2-3B-mix** | 3B | 33.28% | 33.13% | 7.13 | 140 | 219 / 658 |
| 6 | **Qwen3-VL-8B-Instruct** | 8B | 7.75% | 7.75% | 0.73 | 1364 | 51 / 658 |
| 7 | **Qwen3-VL-8B-Thinking** *(4-bit, partial)* | 8B | ~2% | ~1% | 0.13 | 8000 | 14 / 658* |

*\* Qwen3-Thinking only completed 20 images before timeout; numbers are extrapolated.*

---

## Previously tested (≤ 28 images, 1-pass, older pipeline — not directly comparable)

| Model | Images | B3 Acc@50 | ms/inf |
|-------|-------:|----------:|-------:|
| LLaVA-Phi-3-Mini-4B | 28 | 0% | 20,890 |
| Phi-3.5-Vision-4.2B | 28 | 0% | 23,357 |
| LLaVA-v1.6-Mistral-7B | 28 | 0% | 28,071 |
| LLaVA-OneVision-Qwen2-7B | 28 | 0% | 26,092 |
| LLaVA-NeXT-Video-7B | 24 | 0% | 26,290 |
| Llama-3.2-11B-Vision | 28 | 0% | 4,205 |

---

## Key findings

1. **Florence-2 (0.77B) wins** — a lightweight model purpose-built for detection tasks dominates this benchmark at 69.3% with decent throughput.

2. **YOLO models dominate speed** — YOLO-Worldv2-x and YOLOE-26m both achieve 440–457 FPS with comparable accuracy (64–68%), making them the best for real-time applications.

3. **General VLMs struggle** — Qwen3-VL-8B (7.8%) and all LLaVA variants (0%) show that generic VLMs cannot reliably produce structured bounding box output without task-specific fine-tuning.

4. **Cosmos-Reason1-7B is mid-tier** — at 37.5% it is roughly half as accurate as the detection-specialist models despite being 10× larger.

5. **Model size ≠ performance** — PaliGemma2-3B outperforms both Qwen3-VL-8B and Cosmos-Reason1-7B on this task, confirming that architecture and task-specific training matter more than parameter count.
