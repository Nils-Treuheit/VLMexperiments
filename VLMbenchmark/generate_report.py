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

print(f"  Captioning: {len(caption_data)} models")
print(f"  VQA: {len(vqa_data)} models")
print(f"  OD: {len(od_data)} models")
print(f"  Pose: {len(pose_data)} models")
print(f"  OBB: {len(obb_data)} models")
print(f"  Grounding: {len(grounding_data)} models")

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
    "yolo11": "YOLO11n",
    "yolo26_pose": "YOLO26n (Pose)",
    "yolo26_obb": "YOLO26n (OBB)",
    "locate_anything": "LocateAnything-3B",
    "phi4_multimodal": "Phi-4-Multimodal",
}

for d in [caption_data, vqa_data, od_data, pose_data, obb_data, grounding_data]:
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
# Combined FPS overview across tasks
# =============================================
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

if model_fps:
    tasks = ["captioning", "vqa", "detection"]
    models = sorted(model_fps.keys(), key=lambda m: max(
        model_fps[m].get(t, 0) for t in tasks), reverse=True)
    models = models[:15]  # top 15

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(models))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, task in enumerate(tasks):
        vals = [model_fps.get(m, {}).get(task, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=task.capitalize(),
                      color=colors[i], edgecolor="gray", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in models], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("FPS")
    ax.set_title("Speed Comparison Across Tasks (top 15 models)")
    ax.legend()
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
report.append("and vision encoders. Benchmarks cover image captioning, visual question answering (VQA), ")
report.append("object detection, phrase grounding, pose estimation, and oriented bounding box (OBB) detection.")
report.append("")

report.append("### Models Tested")
report.append("")
report.append("| Category | Models |")
report.append("|----------|--------|")
report.append(f"| Vision Encoders | DINOtool, DINOv3, SigLIP2, MoonViT |")
report.append(f"| VLMs (caption + VQA) | Florence-2, PaliGemma2, Phi-3.5-Vision, Cosmos-Reason1-7B, Llama-3.2-11B-Vision, Qwen3-VL-8B-Instruct, Qwen3-VL-8B-Thinking |")
report.append(f"| VLMs (diffusion) | DiffusionGemma-26B (5 variants) |")
report.append(f"| Detection | YOLO11n, YOLO26n, YOLO26n (Pose), YOLO26n (OBB), LocateAnything-3B |")
report.append("")

report.append("### Notes")
report.append("")
report.append("- **50 images** per model for captioning (25 for slow models: qwen3_thinking, phi4_multimodal, diffusion_gemma variants)")
report.append("- **100 questions** per model for VQA")
report.append("- **50 images** per model for OD, pose, OBB, grounding")
report.append("- Vision encoders use zero-shot classification via DINO/transformer features + sentence-transformers (not trained for captioning)")
report.append("- Phi-3.5-Vision is very slow (~15s/image) without flash-attention on Blackwell GPU")
report.append("- DiffusionGemma variants need ~50-60s/image")

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
    for mk, d in sorted(pose_data.items(), key=lambda x: x[1].get("mAP@50:95", 0), reverse=True):
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
    for mk, d in sorted(obb_data.items(), key=lambda x: x[1].get("mAP@50:95", 0), reverse=True):
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
# Speed vs Quality
# =============================================
report.append("")
report.append("## 7. Speed vs Quality Overview")
report.append("")
report.append("![Combined FPS](charts/combined_fps.png)")
report.append("")
report.append("![Quality Comparison](charts/quality_comparison.png)")
report.append("")

# =============================================
# Key Takeaways
# =============================================
report.append("")
report.append("## 8. Key Takeaways")
report.append("")
report.append("### Fastest Models")
report.append("- **Detection:** YOLO26n dominates at ~13 FPS for detection, pose, and OBB")
report.append("- **Captioning:** PaliGemma2-3B is fastest at 4.56 FPS with highest CIDEr (1.72)")
report.append("- **VQA:** Llama-3.2-11B-Vision achieves highest accuracy (64%) at 2.55 FPS")
report.append("")
report.append("### Best Quality")
report.append("- **Captioning CIDEr:** PaliGemma2-3B (1.7246), Florence-2 (0.4999)")
report.append("- **VQA Accuracy:** Llama-3.2-11B-Vision (64%), Phi-3.5-Vision (57%), PaliGemma2 (54%)")
report.append("- **Detection mAP:** YOLO26n (0.480), LocateAnything-3B (0.126)")
report.append("")
report.append("### Notable Observations")
report.append("- Vision encoders (DINOtool, DINOv3, SigLIP2, MoonViT) achieve near-zero CIDEr — expected as they use zero-shot label matching, not generative captioning")
report.append("- Phi-3.5-Vision is 15-60x slower than other models (~15.6s/image) without flash-attention on Blackwell GPUs")
report.append("- Qwen3-VL-8B-Thinking produces more detailed captions but at ~4-10x slower speed vs Instruct variant")
report.append("- DiffusionGemma-26B takes 50-60s per image for caption generation")
report.append("- YOLO models achieve the highest FPS across all detection tasks (10-14 FPS)")
report.append("")
report.append("### Missing Benchmarks")
report.append("- **VQA for DiffusionGemma-26B** — timed out during evaluation")
report.append("- **OD for Florence-2, PaliGemma:** missing pycocotools dependency in their venvs")
report.append("- **Grounding for Florence-2, LocateAnything:** missing pycocotools")
report.append("- **Phi-4-Multimodal:** not fully tested (missing from model choices in some tasks)")

report_str = "\n".join(report)
report_path = REPORT_DIR / "Benchmark_Results.md"
with open(report_path, "w") as f:
    f.write(report_str)

print(f"\nReport saved to: {report_path}")
print("Done!")
