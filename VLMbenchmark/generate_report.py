"""Generate benchmark report with graphs from all results."""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPORT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = REPORT_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (14, 7),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})


def load_stats(glob_pattern):
    results = {}
    for fpath in sorted(RESULTS_DIR.glob(glob_pattern)):
        try:
            with open(fpath) as f:
                data = json.load(f)
            mk = data.get("model_key", fpath.stem.replace("_stats", ""))
            results[mk] = data
        except (json.JSONDecodeError, KeyError):
            pass
    return results


def make_chart(data_dict, metric_key, title, ylabel, filename, higher_better=True,
               fmt="{:.2f}", skip_zero=False, model_filter=None):
    items = []
    for mk, d in data_dict.items():
        if model_filter and mk not in model_filter:
            continue
        if metric_key not in d:
            continue
        val = d[metric_key]
        if val is None:
            continue
        if skip_zero and val == 0:
            continue
        display = d.get("display", mk)
        items.append((display, val))

    if not items:
        print(f"  No data for {title}")
        return

    items.sort(key=lambda x: x[1], reverse=higher_better)

    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    colors = plt.cm.tab20(np.linspace(0, 1, len(items)))

    fig, ax = plt.subplots()
    bars = ax.barh(range(len(items)), vals, color=colors, edgecolor="gray", linewidth=0.5)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        label = fmt.format(v)
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8)

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


print("Loading results...")

caption_data = load_stats("*_caption_stats.json")
vqa_data = load_stats("*_vqa_stats.json")
od_data = load_stats("*_coco_od_stats.json")
pose_data = load_stats("*_pose_stats.json")
obb_data = load_stats("*_obb_stats.json")
grounding_data = load_stats("*_grounding_stats.json")
classification_data = load_stats("*_classification_stats.json")
segmentation_data = load_stats("*_segmentation_stats.json")
scene_data = load_stats("*_scene_stats.json")
tracking_data = load_stats("*_tracking_stats.json")
pose6d_data = load_stats("*_6dpose_stats.json")
ocr_data = load_stats("*_ocr_stats.json")
pointing_data = load_stats("*_pointing_stats.json")
visual_reasoning_data = load_stats("*_visual_reasoning_stats.json")
emotion_data = load_stats("*_emotion_stats.json")
hir_data = load_stats("*_hir_stats.json")
doc_understanding_data = load_stats("*_doc_understanding_stats.json")
embedding_data = load_stats("*_embedding_stats.json")
zeroshot_detection_data = load_stats("*_zeroshot_detection_stats.json")

print(f"  Captioning: {len(caption_data)} models")
print(f"  VQA: {len(vqa_data)} models")
print(f"  OD: {len(od_data)} models")
print(f"  Pose: {len(pose_data)} models")
print(f"  OBB: {len(obb_data)} models")
print(f"  Grounding: {len(grounding_data)} models")
print(f"  Classification: {len(classification_data)} models")
print(f"  Segmentation: {len(segmentation_data)} models")
print(f"  Scene Analysis: {len(scene_data)} models")
print(f"  Tracking: {len(tracking_data)} models")
print(f"  6D Pose: {len(pose6d_data)} models")
print(f"  OCR: {len(ocr_data)} models")
print(f"  Pointing: {len(pointing_data)} models")
print(f"  Visual Reasoning: {len(visual_reasoning_data)} models")
print(f"  Emotion: {len(emotion_data)} models")
print(f"  HIR: {len(hir_data)} models")
print(f"  Doc Understanding: {len(doc_understanding_data)} models")
print(f"  Embedding: {len(embedding_data)} models")
print(f"  Zero-shot Detection: {len(zeroshot_detection_data)} models")

# =============================================
# Override display names for consistency
# =============================================
DISPLAY_NAMES = {
    "cosmos_nemotron": "Cosmos-Reason1-7B",
    "diffusion_gemma": "DiffusionGemma-26B",
    "diffusion_gemma_yolo": "DiffusionGemma-26B (YOLO)",
    "diffusion_gemma_yolo_pose": "DiffusionGemma-26B (YOLO+pose)",
    "diffusion_gemma_yolo_obb": "DiffusionGemma-26B (YOLO+pose+obb)",
    "diffusion_gemma_siglip2": "DiffusionGemma-26B (SigLIP2)",
    "diffusion_gemma_moonvit": "DiffusionGemma-26B (MoonViT)",
    "dinotool": "DINOtool (DINOv2-s)",
    "dinov3": "DINOv3 (Zero-shot)",
    "siglip2": "SigLIP2 (Zero-shot)",
    "moonvit": "MoonViT (Zero-shot)",
    "florence2": "Florence-2-large-ft",
    "paligemma": "PaliGemma2-3B-mix",
    "phi_vision": "Phi-3.5-Vision-4B",
    "llama_vision": "Llama-3.2-11B-Vision",
    "qwen3_native": "Qwen3-VL-8B-Instruct",
    "qwen3_thinking": "Qwen3-VL-8B-Thinking",
    "yolo26": "YOLO26n",
    "yolo26s": "YOLO26s",
    "yolo26m": "YOLO26m",
    "yolo26l": "YOLO26l",
    "yolo26x": "YOLO26x",
    "yolo11": "YOLO11n",
    "yolo11s": "YOLO11s",
    "yolo11m": "YOLO11m",
    "yolo11l": "YOLO11l",
    "yolo11x": "YOLO11x",
    "yolo26_pose": "YOLO26n (Pose)",
    "yolo26s_pose": "YOLO26s (Pose)",
    "yolo11_pose": "YOLO11n (Pose)",
    "yolo11s_pose": "YOLO11s (Pose)",
    "yolo26_obb": "YOLO26n (OBB)",
    "yolo26s_obb": "YOLO26s (OBB)",
    "yolo11_obb": "YOLO11n (OBB)",
    "yolo11s_obb": "YOLO11s (OBB)",
    "locate_anything": "LocateAnything-3B",
    "locate_anything_trt": "LocateAnything-3B (TRT)",
    "locate_anything_ocr": "LocateAnything-3B (OCR)",
    "locate_anything_trt_ocr": "LocateAnything-3B TRT (OCR)",
    "dinotool_dinov2_s": "DINOtool (DINOv2-s)",
    "dinotool_dinov2_b": "DINOtool (DINOv2-b)",
    "dinotool_dinov2_l": "DINOtool (DINOv2-l)",
    "dinotool_dinov2_g": "DINOtool (DINOv2-g)",
    "dinotool_dinov3_s": "DINOtool (DINOv3-s)",
    "dinotool_dinov3_b": "DINOtool (DINOv3-b)",
    "dinotool_dinov3_l": "DINOtool (DINOv3-l)",
    "dinotool_siglip1": "DINOtool (SigLIP-B)",
    "dinotool_siglip2_so400m": "DINOtool (SigLIP2-SO400M-384)",
    "dinotool_radio_b": "DINOtool (RADIO-B)",
    "dinotool_radio_h": "DINOtool (RADIO-H)",
    "dinotool_tipsv2_b": "DINOtool (TIPSv2-B)",
    "dinotool_tipsv2_l": "DINOtool (TIPSv2-L)",
    "dinotool_tipsv2_so400m": "DINOtool (TIPSv2-SO400M)",
    "dinotool_tipsv2_g": "DINOtool (TIPSv2-G)",
    "yolo26_botsort": "YOLO26n + BoTSORT",
    "yolo26_bytetrack": "YOLO26n + ByteTrack",
    "yolo11_botsort": "YOLO11n + BoTSORT",
    "yolo11_bytetrack": "YOLO11n + ByteTrack",
    "phi4_multimodal": "Phi-4-Multimodal",
}

for d in [caption_data, vqa_data, od_data, pose_data, obb_data, grounding_data,
          classification_data, segmentation_data, scene_data, tracking_data, pose6d_data,
          visual_reasoning_data, emotion_data, hir_data, doc_understanding_data,
          embedding_data, zeroshot_detection_data]:
    for mk in list(d.keys()):
        if mk in DISPLAY_NAMES:
            d[mk]["display"] = DISPLAY_NAMES[mk]
        else:
            d[mk]["display"] = d[mk].get("model_key", mk)

print("\nGenerating charts...")

# =============================================
# CAPTIONING: FPS, CIDEr, BLEU-4, ROUGE-L
# =============================================
print("\n--- Captioning ---")

make_chart(caption_data, "fps", "Image Captioning — FPS (higher is better)",
           "FPS", "caption_fps.png")

make_chart(caption_data, "cider", "Image Captioning — CIDEr Score (higher is better)",
           "CIDEr", "caption_cider.png", fmt="{:.4f}")

make_chart(caption_data, "bleu_4", "Image Captioning — BLEU-4 (higher is better)",
           "BLEU-4", "caption_bleu4.png", fmt="{:.4f}")

make_chart(caption_data, "rouge_l", "Image Captioning — ROUGE-L (higher is better)",
           "ROUGE-L", "caption_rouge.png", fmt="{:.4f}")

# =============================================
# VQA: Accuracy, FPS
# =============================================
print("\n--- VQA ---")

make_chart(vqa_data, "accuracy", "Visual Question Answering — Accuracy (higher is better)",
           "Accuracy", "vqa_accuracy.png", fmt="{:.2%}")

make_chart(vqa_data, "fps", "Visual Question Answering — FPS (higher is better)",
           "FPS", "vqa_fps.png")

# =============================================
# DETECTION: mAP@50:95, mAP@50, FPS
# =============================================
print("\n--- Object Detection ---")

make_chart(od_data, "mAP@50:95", "Object Detection — mAP@50:95 (higher is better)",
           "mAP@50:95", "od_map5095.png", fmt="{:.4f}")

make_chart(od_data, "mAP@50", "Object Detection — mAP@50 (higher is better)",
           "mAP@50", "od_map50.png", fmt="{:.4f}")

make_chart(od_data, "fps", "Object Detection — FPS (higher is better)",
           "FPS", "od_fps.png")

# =============================================
# OTHER TASKS
# =============================================
print("\n--- Pose / OBB / Grounding ---")

make_chart(pose_data, "mAP@50:95", "Pose Estimation — mAP@50:95",
           "mAP@50:95", "pose_map.png", fmt="{:.4f}")
make_chart(pose_data, "fps", "Pose Estimation — FPS",
           "FPS", "pose_fps.png")

make_chart(obb_data, "mAP@50:95", "OBB Detection — mAP@50:95",
           "mAP@50:95", "obb_map.png", fmt="{:.4f}")
make_chart(obb_data, "fps", "OBB Detection — FPS",
           "FPS", "obb_fps.png")

make_chart(grounding_data, "acc@50", "Phrase Grounding — Acc@50",
           "Acc@50", "grounding_acc.png", fmt="{:.2%}")
make_chart(grounding_data, "fps", "Phrase Grounding — FPS",
           "FPS", "grounding_fps.png")

# =============================================
# NEW TASK CHARTS
# =============================================
print("\n--- Classification ---")

make_chart(classification_data, "top1_accuracy", "Zero-Shot Classification — Top-1 Accuracy",
           "Top-1 Accuracy", "class_top1.png", fmt="{:.2%}")
make_chart(classification_data, "fps", "Zero-Shot Classification — FPS",
           "FPS", "class_fps.png")

print("\n--- Segmentation ---")
make_chart(segmentation_data, "PQ", "Segmentation — Panoptic Quality",
           "PQ", "seg_pq.png", fmt="{:.4f}")
make_chart(segmentation_data, "mIoU", "Segmentation — mIoU",
           "mIoU", "seg_miou.png", fmt="{:.4f}")
make_chart(segmentation_data, "fps", "Segmentation — FPS",
           "FPS", "seg_fps.png")

print("\n--- Scene Analysis ---")
make_chart(scene_data, "scene_type_accuracy", "Semantic Scene Analysis — Scene Type Accuracy",
           "Accuracy", "scene_accuracy.png", fmt="{:.2%}")
make_chart(scene_data, "object_recall", "Semantic Scene Analysis — Object Recall",
           "Recall", "scene_recall.png", fmt="{:.2%}")
make_chart(scene_data, "fps", "Semantic Scene Analysis — FPS",
           "FPS", "scene_fps.png")

print("\n--- Tracking ---")
make_chart(tracking_data, "MOTA", "Multi-Object Tracking — MOTA",
           "MOTA", "track_mota.png", fmt="{:.4f}")
make_chart(tracking_data, "MOTP", "Multi-Object Tracking — MOTP",
           "MOTP", "track_motp.png", fmt="{:.4f}")
make_chart(tracking_data, "fps", "Multi-Object Tracking — FPS",
           "FPS", "track_fps.png")

print("\n--- 6D Pose ---")
make_chart(pose6d_data, "detection_rate", "6D Pose (Linemod) — Detection Rate",
           "Detection Rate", "pose6d_detrate.png", fmt="{:.2%}")
make_chart(pose6d_data, "fps", "6D Pose (Linemod) — FPS",
           "FPS", "pose6d_fps.png")

print("\n--- OCR ---")
make_chart(ocr_data, "detection_rate", "OCR / Text Detection — Detection Rate",
           "Detection Rate", "ocr_detrate.png", fmt="{:.2%}")
make_chart(ocr_data, "fps", "OCR / Text Detection — FPS",
           "FPS", "ocr_fps.png")

print("\n--- Pointing ---")
make_chart(pointing_data, "acc@0.05", "Pointing / 2D Keypoint — Acc@0.05",
           "Acc@0.05", "pointing_acc05.png", fmt="{:.2%}")
make_chart(pointing_data, "acc@0.10", "Pointing / 2D Keypoint — Acc@0.10",
           "Acc@0.10", "pointing_acc10.png", fmt="{:.2%}")
make_chart(pointing_data, "fps", "Pointing / 2D Keypoint — FPS",
           "FPS", "pointing_fps.png")

# =============================================
# Combined FPS overview across tasks
# =============================================
print("\n--- Embedding ---")
make_chart(embedding_data, "fps", "Embedding Extraction — FPS (higher is better)",
           "FPS", "embedding_fps.png")
make_chart(embedding_data, "embedding_dim", "Embedding Extraction — Output Dimension",
           "Embedding dim", "embedding_dim.png", higher_better=False)

print("\n--- Zero-shot Detection ---")
make_chart(zeroshot_detection_data, "acc@50", "Zero-Shot Detection — Acc@50",
           "Acc@50", "zeroshot_detection_acc.png", fmt="{:.2%}")
make_chart(zeroshot_detection_data, "fps", "Zero-Shot Detection — FPS",
           "FPS", "zeroshot_detection_fps.png")

print("\n--- Combined charts ---")

# Gather FPS for each model across tasks
model_fps = {}
for mk, d in caption_data.items():
    if "fps" in d:
        model_fps.setdefault(mk, {})["captioning"] = d["fps"]
for mk, d in vqa_data.items():
    if "fps" in d:
        model_fps.setdefault(mk, {})["vqa"] = d["fps"]
for mk, d in od_data.items():
    if "fps" in d:
        model_fps.setdefault(mk, {})["detection"] = d["fps"]
for mk, d in classification_data.items():
    if "fps" in d:
        model_fps.setdefault(mk, {})["classification"] = d["fps"]
for mk, d in tracking_data.items():
    if "fps" in d:
        model_fps.setdefault(mk, {})["tracking"] = d["fps"]

if model_fps:
    tasks = ["captioning", "vqa", "detection", "classification", "tracking"]
    models = sorted(model_fps.keys(), key=lambda m: max(
        model_fps[m].get(t, 0) for t in tasks), reverse=True)
    models = models[:15]  # top 15

    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(len(models))
    width = 0.15
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860"]

    for i, task in enumerate(tasks):
        vals = [model_fps.get(m, {}).get(task, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=task.capitalize(),
                      color=colors[i], edgecolor="gray", linewidth=0.5)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in models], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("FPS")
    ax.set_title("Speed Comparison Across Tasks (top 15 models)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "combined_fps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved combined_fps.png")

# Quality summary: caption CIDEr + VQA accuracy
quality_models = set()
for mk in caption_data:
    if caption_data[mk].get("images", 0) >= 20:
        quality_models.add(mk)
for mk in vqa_data:
    if vqa_data[mk].get("questions", 0) >= 50:
        quality_models.add(mk)

models_with_both = [m for m in quality_models if m in caption_data and m in vqa_data]
if models_with_both:
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    names = [DISPLAY_NAMES.get(m, m) for m in models_with_both]
    ciders = [caption_data[m].get("cider", 0) for m in models_with_both]
    accs = [vqa_data[m].get("accuracy", 0) for m in models_with_both]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, ciders, width, label="CIDEr (captioning)",
                    color="#4C72B0", edgecolor="gray")
    bars2 = ax2.bar(x + width / 2, accs, width, label="Accuracy (VQA)",
                    color="#DD8452", edgecolor="gray")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("CIDEr")
    ax2.set_ylabel("Accuracy")
    ax1.set_title("Quality: Captioning CIDEr vs VQA Accuracy")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "quality_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved quality_comparison.png")


# =============================================
# Write Markdown Report
# =============================================
def model_link(mk):
    name = DISPLAY_NAMES.get(mk, mk)
    return name

def fmt_val(d, key, default="—", fmt="{:.2f}"):
    if key in d and d[key] is not None:
        return fmt.format(d[key])
    return default

print("\n--- Writing report ---")

report = []
report.append("# Benchmark Results")
report.append("")
report.append(f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
report.append("")
report.append(f"**Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM)")
report.append("")
report.append("## Overview")
report.append("")
report.append("This report summarizes benchmark results across multiple vision-language models (VLMs) ")
report.append("and vision encoders. Benchmarks cover 15 tasks: image captioning, visual question answering (VQA), ")
report.append("object detection (AABB + OBB), phrase grounding, pose estimation (2D keypoints + 6D), ")
report.append("segmentation, zero-shot classification, semantic scene analysis, multi-object tracking, ")
report.append("OCR / text detection, pointing / 2D keypoint localization, ")
report.append("embedding extraction, and zero-shot detection.")
report.append("")

report.append("### Models Tested")
report.append("")
report.append("| Category | Models |")
report.append("|----------|--------|")
report.append(f"| Vision Encoders | DINOtool, DINOv3, SigLIP2, MoonViT |")
report.append(f"| VLMs (caption + VQA) | Florence-2, PaliGemma2, Phi-3.5-Vision, Cosmos-Reason1-7B, Llama-3.2-11B-Vision, Qwen3-VL-8B-Instruct, Qwen3-VL-8B-Thinking |")
report.append(f"| VLMs (diffusion) | DiffusionGemma-26B (5 variants) |")
report.append(f"| Detection / OBB / Pose | YOLO11n/s/m, YOLO26n/s/m (detect, pose, OBB), LocateAnything-3B, LocateAnything-3B (TRT) |")
report.append(f"| OCR / Pointing | LocateAnything-3B, LocateAnything-3B (TRT) |")
report.append("")

report.append("### Datasets")
report.append("")
report.append("| Task | Dataset | Images |")
report.append("|------|---------|--------|")
report.append("| Captioning | COCO Captions val2017 | 25-50 |")
report.append("| VQA | COCO val2017 + templated Qs | 100 questions |")
report.append("| Object Detection | COCO val2017 | 25-50 |")
report.append("| OBB Detection | DOTA-v1.0 | 50 |")
report.append("| Pose (2D) | COCO Keypoints val2017 | 23-50 |")
report.append("| Phrase Grounding | COCO val2017 | 25-48 |")
report.append("| Zero-Shot Classification | Tiny ImageNet (200 classes) | 500 |")
report.append("| Segmentation | COCO val2017 | 100 |")
report.append("| Scene Analysis | COCO val2017 | 100 |")
report.append("| Multi-Object Tracking | MOT17 | 200 frames (2 seqs) |")
report.append("| 6D Pose (detection) | Linemod (BOP) | 25 |")
report.append("| OCR / Text Detection | Synthetic text on COCO | 25 |")
report.append("| Pointing / 2D Keypoint | COCO Keypoints val2017 | 10-25 |")
report.append("| Embedding Extraction | COCO val2017 | 100 |")
report.append("| Zero-Shot Detection | COCO val2017 | 100 |")
report.append("")

report.append("### Notes")
report.append("")
report.append("- **25-50 images** per model for captioning (more for fast, less for slow models)")
report.append("- **100 questions** per model for VQA")
report.append("- **500 images** (Tiny ImageNet) for classification")
report.append("- Vision encoders use zero-shot classification via DINO/transformer features + sentence-transformers (not trained for captioning)")
report.append("- Phi-3.5-Vision is very slow (~15s/image) without flash-attention on Blackwell GPU")
report.append("- DiffusionGemma variants need ~50-60s/image")
report.append("- LocateAnything-3B (TRT) uses TensorRT-accelerated vision encoder (9.8× faster vision, 1.6× faster end-to-end)")
report.append("- OCR benchmark uses synthetic text overlays on COCO images (5 random words per image)")
report.append("- Pointing benchmark evaluates COCO keypoints (nose, eyes, shoulders, etc.) with normalized distance thresholds")

# =============================================
# CAPTIONING TABLE
# =============================================
report.append("")
report.append("## 1. Image Captioning (COCO Captions)")
report.append("")
report.append("![FPS](charts/caption_fps.png)")
report.append("![CIDEr](charts/caption_cider.png)")
report.append("![BLEU-4](charts/caption_bleu4.png)")
report.append("![ROUGE-L](charts/caption_rouge.png)")
report.append("")

# Table
report.append("| Model | FPS | CIDEr | BLEU-4 | ROUGE-L | Avg (ms) | Images |")
report.append("|-------|-----|-------|--------|---------|----------|--------|")
rows = []
for mk, d in sorted(caption_data.items(), key=lambda x: x[1].get("fps", 0), reverse=True):
    if d.get("images", 0) < 5:
        continue
    rows.append(
        f"| {model_link(mk)} | {fmt_val(d, 'fps')} | {fmt_val(d, 'cider', fmt='{:.4f}')} | "
        f"{fmt_val(d, 'bleu_4', fmt='{:.4f}')} | {fmt_val(d, 'rouge_l', fmt='{:.4f}')} | "
        f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('images', '—')} |"
    )
report.extend(rows)

# =============================================
# VQA TABLE
# =============================================
report.append("")
report.append("## 2. Visual Question Answering (COCO)")
report.append("")
report.append("![Accuracy](charts/vqa_accuracy.png)")
report.append("![FPS](charts/vqa_fps.png)")
report.append("")

report.append("| Model | Accuracy | FPS | Avg (ms) | Questions |")
report.append("|-------|----------|-----|----------|-----------|")
for mk, d in sorted(vqa_data.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
    if d.get("questions", 0) < 20:
        continue
    report.append(
        f"| {model_link(mk)} | {fmt_val(d, 'accuracy', fmt='{:.2%}')} | {fmt_val(d, 'fps')} | "
        f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('questions', '—')} |"
    )

# =============================================
# OBJECT DETECTION TABLE
# =============================================
report.append("")
report.append("## 3. Object Detection (COCO)")
report.append("")
report.append("![mAP@50:95](charts/od_map5095.png)")
report.append("![mAP@50](charts/od_map50.png)")
report.append("![FPS](charts/od_fps.png)")
report.append("")

report.append("| Model | mAP@50:95 | mAP@50 | FPS | Avg (ms) | Images |")
report.append("|-------|-----------|--------|-----|----------|--------|")
for mk, d in sorted(od_data.items(), key=lambda x: x[1].get("mAP@50:95", 0), reverse=True):
    if d.get("images", 0) < 5:
        continue
    report.append(
        f"| {model_link(mk)} | {fmt_val(d, 'mAP@50:95', fmt='{:.4f}')} | {fmt_val(d, 'mAP@50', fmt='{:.4f}')} | "
        f"{fmt_val(d, 'fps')} | {fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('images', '—')} |"
    )

# =============================================
# OTHER TASKS
# =============================================
if pose_data:
    report.append("")
    report.append("## 4. Pose Estimation (COCO Keypoints)")
    report.append("")
    report.append("![mAP](charts/pose_map.png)")
    report.append("![FPS](charts/pose_fps.png)")
    report.append("")
    report.append("| Model | mAP@50:95 | mAP@50 | FPS | Avg (ms) | Images |")
    report.append("|-------|-----------|--------|-----|----------|--------|")
    for mk, d in sorted(pose_data.items(), key=lambda x: x[1].get("AP@50:95_keypoints", 0) or 0, reverse=True):
        if d.get("images", 0) < 5:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'mAP@50:95', fmt='{:.4f}')} | {fmt_val(d, 'mAP@50', fmt='{:.4f}')} | "
            f"{fmt_val(d, 'fps')} | {fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('images', '—')} |"
        )

if obb_data:
    report.append("")
    report.append("## 5. Oriented Bounding Box (DOTA-v1.0)")
    report.append("")
    report.append("![mAP](charts/obb_map.png)")
    report.append("![FPS](charts/obb_fps.png)")
    report.append("")
    report.append("| Model | mAP@50:95 | mAP@50 | FPS | Avg (ms) | Images |")
    report.append("|-------|-----------|--------|-----|----------|--------|")
    for mk, d in sorted(obb_data.items(), key=lambda x: x[1].get("mAP@50:95") or 0, reverse=True):
        if d.get("images", 0) < 5:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'mAP@50:95', fmt='{:.4f}')} | {fmt_val(d, 'mAP@50', fmt='{:.4f}')} | "
            f"{fmt_val(d, 'fps')} | {fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('images', '—')} |"
        )

if grounding_data:
    report.append("")
    report.append("## 6. Phrase Grounding (COCO)")
    report.append("")
    report.append("![Accuracy](charts/grounding_acc.png)")
    report.append("![FPS](charts/grounding_fps.png)")
    report.append("")
    report.append("| Model | Acc@50 | FPS | Avg (ms) | Images |")
    report.append("|-------|--------|-----|----------|--------|")
    for mk, d in sorted(grounding_data.items(), key=lambda x: x[1].get("acc@50", 0), reverse=True):
        if d.get("images", 0) < 5:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'acc@50', fmt='{:.2%}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('images', '—')} |"
        )

# =============================================
# CLASSIFICATION
# =============================================
if classification_data:
    report.append("")
    report.append("## 7. Zero-Shot Classification (Tiny ImageNet)")
    report.append("")
    report.append("![Top-1 Accuracy](charts/class_top1.png)")
    report.append("![FPS](charts/class_fps.png)")
    report.append("")
    report.append("| Model | Top-1 Acc | Top-5 Acc | FPS | Avg (ms) | Images |")
    report.append("|-------|-----------|-----------|-----|----------|--------|")
    for mk, d in sorted(classification_data.items(), key=lambda x: x[1].get("top1_accuracy", 0), reverse=True):
        if d.get("images", 0) < 10:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'top1_accuracy', fmt='{:.2%}')} | "
            f"{fmt_val(d, 'top5_accuracy', fmt='{:.2%}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('images', '—')} |"
        )

# =============================================
# SEGMENTATION
# =============================================
if segmentation_data:
    report.append("")
    report.append("## 8. Segmentation (COCO)")
    report.append("")
    report.append("![Panoptic Quality](charts/seg_pq.png)")
    report.append("![mIoU](charts/seg_miou.png)")
    report.append("![FPS](charts/seg_fps.png)")
    report.append("")
    report.append("| Model | PQ | mIoU | FPS | Avg (ms) | Images |")
    report.append("|-------|----|------|-----|----------|--------|")
    for mk, d in sorted(segmentation_data.items(), key=lambda x: x[1].get("PQ", 0) or 0, reverse=True):
        if d.get("images", 0) < 5:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'PQ', fmt='{:.4f}')} | "
            f"{fmt_val(d, 'mIoU', fmt='{:.4f}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('images', '—')} |"
        )

# =============================================
# SCENE ANALYSIS
# =============================================
if scene_data:
    report.append("")
    report.append("## 9. Semantic Scene Analysis (COCO)")
    report.append("")
    report.append("![Scene Type Accuracy](charts/scene_accuracy.png)")
    report.append("![Object Recall](charts/scene_recall.png)")
    report.append("![FPS](charts/scene_fps.png)")
    report.append("")
    report.append("| Model | Scene Acc | Object Recall | FPS | Avg (ms) | Images |")
    report.append("|-------|-----------|---------------|-----|----------|--------|")
    for mk, d in sorted(scene_data.items(), key=lambda x: x[1].get("scene_type_accuracy", 0), reverse=True):
        if d.get("images", 0) < 5:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'scene_type_accuracy', fmt='{:.2%}')} | "
            f"{fmt_val(d, 'object_recall', fmt='{:.2%}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('images', '—')} |"
        )

# =============================================
# TRACKING
# =============================================
if tracking_data:
    report.append("")
    report.append("## 10. Multi-Object Tracking (MOT17)")
    report.append("")
    report.append("![MOTA](charts/track_mota.png)")
    report.append("![MOTP](charts/track_motp.png)")
    report.append("![FPS](charts/track_fps.png)")
    report.append("")
    report.append("| Model | MOTA | MOTP | FPS | Avg (ms) | Frames |")
    report.append("|-------|------|------|-----|----------|--------|")
    for mk, d in sorted(tracking_data.items(), key=lambda x: x[1].get("MOTA", 0), reverse=True):
        if d.get("frames", 0) < 10:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'MOTA', fmt='{:.4f}')} | "
            f"{fmt_val(d, 'MOTP', fmt='{:.4f}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('frames', '—')} |"
        )

# =============================================
# 6D POSE
# =============================================
if pose6d_data:
    report.append("")
    report.append("## 11. 6D Pose Estimation (Linemod)")
    report.append("")
    report.append("![Detection Rate](charts/pose6d_detrate.png)")
    report.append("![FPS](charts/pose6d_fps.png)")
    report.append("")
    report.append("| Model | Detection Rate | FPS | Avg (ms) | Images |")
    report.append("|-------|----------------|-----|----------|--------|")
    for mk, d in sorted(pose6d_data.items(), key=lambda x: x[1].get("detection_rate", 0), reverse=True):
        if d.get("images", 0) < 5:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'detection_rate', fmt='{:.2%}')} | "
            f"{fmt_val(d, 'fps')} | {fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | "
            f"{d.get('images', '—')} |"
        )

# =============================================
# OCR
# =============================================
if ocr_data:
    report.append("")
    report.append("## 12. OCR / Text Detection (Synthetic COCO)")
    report.append("")
    report.append("![Detection Rate](charts/ocr_detrate.png)")
    report.append("![FPS](charts/ocr_fps.png)")
    report.append("")
    report.append("| Model | Detection Rate | FPS | Avg (ms) | Images |")
    report.append("|-------|----------------|-----|----------|--------|")
    for mk, d in sorted(ocr_data.items(), key=lambda x: x[1].get("detection_rate", 0), reverse=True):
        if d.get("images", 0) < 5:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'detection_rate', fmt='{:.2%}')} | "
            f"{fmt_val(d, 'fps')} | {fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | "
            f"{d.get('images', '—')} |"
        )

# =============================================
# POINTING (2D Keypoint)
# =============================================
if pointing_data:
    report.append("")
    report.append("## 13. Pointing / 2D Keypoint (COCO Keypoints)")
    report.append("")
    report.append("![Acc@0.05](charts/pointing_acc05.png)")
    report.append("![Acc@0.10](charts/pointing_acc10.png)")
    report.append("![FPS](charts/pointing_fps.png)")
    report.append("")
    report.append("| Model | Acc@0.05 | Acc@0.10 | FPS | Avg (ms) | Keypoints |")
    report.append("|-------|----------|----------|-----|----------|-----------|")
    for mk, d in sorted(pointing_data.items(), key=lambda x: x[1].get("acc@0.05", 0), reverse=True):
        if d.get("total_keypoints", 0) < 10:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'acc@0.05', fmt='{:.2%}')} | "
            f"{fmt_val(d, 'acc@0.10', fmt='{:.2%}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('total_keypoints', '—')} |"
        )

# =============================================
# VISUAL REASONING
# =============================================
if visual_reasoning_data:
    report.append("")
    report.append("## 14. Visual Reasoning (COCO)")
    report.append("")
    report.append("| Model | Accuracy | FPS | Avg (ms) | Questions |")
    report.append("|-------|----------|-----|----------|-----------|")
    for mk, d in sorted(visual_reasoning_data.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
        if d.get("questions", 0) < 20:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'accuracy', fmt='{:.2%}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('questions', '—')} |"
        )

# =============================================
# EMOTION
# =============================================
if emotion_data:
    report.append("")
    report.append("## 15. Emotion Detection (COCO)")
    report.append("")
    report.append("| Model | Accuracy | FPS | Avg (ms) | Questions |")
    report.append("|-------|----------|-----|----------|-----------|")
    for mk, d in sorted(emotion_data.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
        if d.get("questions", 0) < 20:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'accuracy', fmt='{:.2%}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('questions', '—')} |"
        )

# =============================================
# HIR (Human Intention Recognition)
# =============================================
if hir_data:
    report.append("")
    report.append("## 16. Human Intention Recognition (COCO)")
    report.append("")
    report.append("| Model | Accuracy | FPS | Avg (ms) | Questions |")
    report.append("|-------|----------|-----|----------|-----------|")
    for mk, d in sorted(hir_data.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
        if d.get("questions", 0) < 20:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'accuracy', fmt='{:.2%}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('questions', '—')} |"
        )

# =============================================
# DOCUMENT UNDERSTANDING
# =============================================
if doc_understanding_data:
    report.append("")
    report.append("## 17. Document Understanding (COCO)")
    report.append("")
    report.append("| Model | Accuracy | FPS | Avg (ms) | Questions |")
    report.append("|-------|----------|-----|----------|-----------|")
    for mk, d in sorted(doc_understanding_data.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
        if d.get("questions", 0) < 20:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'accuracy', fmt='{:.2%}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('questions', '—')} |"
        )

# =============================================
# EMBEDDING
# =============================================
if embedding_data:
    report.append("")
    report.append("## 18. Embedding Extraction (COCO)")
    report.append("")
    report.append("![FPS](charts/embedding_fps.png)")
    report.append("![Dimension](charts/embedding_dim.png)")
    report.append("")
    report.append("| Model | FPS | Embedding Dim | Avg (ms) | Images |")
    report.append("|-------|-----|---------------|----------|--------|")
    for mk, d in sorted(embedding_data.items(), key=lambda x: x[1].get("fps", 0), reverse=True):
        if d.get("images", 0) < 5:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'fps')} | {d.get('embedding_dim', '—')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('images', '—')} |"
        )

# =============================================
# ZERO-SHOT DETECTION
# =============================================
if zeroshot_detection_data:
    report.append("")
    report.append("## 19. Zero-Shot Detection (COCO)")
    report.append("")
    report.append("![Acc@50](charts/zeroshot_detection_acc.png)")
    report.append("![FPS](charts/zeroshot_detection_fps.png)")
    report.append("")
    report.append("| Model | Acc@50 | FPS | Avg (ms) | Images |")
    report.append("|-------|--------|-----|----------|--------|")
    for mk, d in sorted(zeroshot_detection_data.items(), key=lambda x: x[1].get("acc@50", 0), reverse=True):
        if d.get("images", 0) < 5:
            continue
        report.append(
            f"| {model_link(mk)} | {fmt_val(d, 'acc@50', fmt='{:.2%}')} | {fmt_val(d, 'fps')} | "
            f"{fmt_val(d, 'avg_inference_ms', fmt='{:.1f}')} | {d.get('images', '—')} |"
        )

# =============================================
# Speed vs Quality
# =============================================
report.append("")
report.append("## 20. Speed vs Quality Overview")
report.append("")
report.append("![Combined FPS](charts/combined_fps.png)")
report.append("")
report.append("![Quality Comparison](charts/quality_comparison.png)")
report.append("")

# =============================================
# Key Takeaways
# =============================================
report.append("")
report.append("## 21. Key Takeaways")
report.append("")
report.append("### Fastest Models by Task")
report.append("- **Detection:** YOLO11n at 46.7 FPS, YOLO26n at 14.8 FPS (medium: 35 FPS)")
report.append("- **Captioning:** PaliGemma2-3B at 4.56 FPS")
report.append("- **VQA:** PaliGemma2-3B at 15.51 FPS")
report.append("- **Classification:** Vision encoders via embedding matching")
report.append("- **Segmentation:** Florence-2 handles segmentation at reasonable speed")
report.append("- **Tracking:** YOLO + ByteTrack achieves ~50 FPS on MOT17")
report.append("- **6D Pose (detection):** YOLO models on Linemod")
report.append("- **OCR (text detection):** LocateAnything-3B TRT at 3.08 FPS, 85.6% detection rate")
report.append("- **Visual Reasoning:** Best models achieve moderate accuracy on template-based COCO questions")
report.append("- **Emotion Detection:** Models show limited accuracy on COCO without dedicated emotion training data")
report.append("- **Human Intention Recognition:** Best models achieve decent accuracy on interaction-focused questions")
report.append("- **Document Understanding:** COCO images contain few documents; models default to 'no'")
report.append("")
report.append("### Best Quality by Task")
report.append("- **Captioning CIDEr:** PaliGemma2-3B (1.7246), Florence-2 (0.4999)")
report.append("- **VQA Accuracy:** Llama-3.2-11B-Vision (64%), Phi-3.5-Vision (57%), PaliGemma2 (54%)")
report.append("- **Detection mAP:** YOLO26m (0.514), YOLO11m (0.497), YOLO26s (0.458)")
report.append("- **Detection FPS:** LocateAnything-3B TRT (5.50) is 1.6× faster than PT (3.41)")
report.append("- **Grounding Acc@50:** LocateAnything-3B TRT (14.4%) — best among tested models")
report.append("- **OCR:** LocateAnything-3B TRT 85.6% detection — better than PT 75.2%")
report.append("- **Scene Understanding:** Florence-2 excels at structured scene description")
report.append("")
report.append("### Notable Observations")
report.append("- YOLO26m achieves highest detection mAP@50:95 (0.514) at 35 FPS — best accuracy-speed trade-off")
report.append("- ByteTrack is ~1.5-2× faster than BoTSORT with nearly identical MOTA accuracy")
report.append("- LocateAnything-3B OCR: TRT achieves 85.6% detection rate vs 75.2% PT (~10% improvement)")
report.append("- LocateAnything-3B Pointing: ~20-28% accuracy at 0.05-0.10 normalized distance thresholds")
report.append("- Vision encoders (DINOtool, DINOv3, SigLIP2, MoonViT) achieve near-zero CIDEr — expected as they use zero-shot label matching, not generative captioning")
report.append("- Phi-3.5-Vision is 15-60x slower than other models (~15.6s/image) without flash-attention on Blackwell GPUs")
report.append("- Qwen3-VL-8B-Thinking produces more detailed captions but at ~4-10x slower speed vs Instruct variant")
report.append("- DiffusionGemma-26B takes 50-60s per image for caption generation")
report.append("- YOLO models achieve the highest FPS across all detection tasks (10-50 FPS)")
report.append("- LocateAnything-3B (TRT) achieves 5.50 FPS on COCO OD (1.6× faster than PT) with bit-exact identical quality")
report.append("- TRT vision encoder runs at 9.6ms (9.8× faster than PyTorch bf16) — LLM decoder dominates at ~170ms")
report.append("- Florence-2 is the most versatile model, supporting captioning, VQA, OD, segmentation, and scene analysis")
report.append("")
report.append("### Missing / Future Benchmarks")
report.append("- **Phi-4-Multimodal:** not fully tested (missing from model choices in some tasks)")
report.append("- **6D Pose ADD/ADD-S:** pose refinement metrics not yet implemented")
report.append("- **Semantic / Panoptic Segmentation:** more comprehensive mask evaluation needed")
report.append("- **Video understanding:** action recognition, temporal reasoning")
report.append("- **Visual Reasoning / Emotion / HIR / Doc Understanding:** template-based evaluation on COCO (no dedicated labels)")

report_str = "\n".join(report)
report_path = REPORT_DIR / "Benchmark_Results.md"
with open(report_path, "w") as f:
    f.write(report_str)

print(f"\nReport saved to: {report_path}")
print("Done!")
