#!/usr/bin/env python3
"""
Zero-shot classification benchmark using vision encoder models.
Evaluates on Tiny ImageNet (200 classes, 10K validation images).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_DIR, RESULTS_DIR, PROJECT_DIR, save_stats, print_comparison

TINY_IMAGENET_DIR = Path("/mnt/HDD1/Project_Data/public_datasets/tiny-imagenet-200")

CLASSIFICATION_MODELS = {"dinotool", "dinov3", "siglip2", "moonvit"}

MODEL_VENV = {
    "dinotool": PROJECT_DIR / "DINOtool" / ".venv" / "bin" / "python",
    "dinov3": PROJECT_DIR / "dinov3" / ".venv" / "bin" / "python",
    "siglip2": PROJECT_DIR / "siglip2" / ".venv" / "bin" / "python",
    "moonvit": PROJECT_DIR / "moonvit" / ".venv" / "bin" / "python",
}

MODEL_RUN_SCRIPT = {
    "dinotool": PROJECT_DIR / "DINOtool" / "run.py",
    "dinov3": PROJECT_DIR / "dinov3" / "run.py",
    "siglip2": PROJECT_DIR / "siglip2" / "run.py",
    "moonvit": PROJECT_DIR / "moonvit" / "run.py",
}


def load_tiny_imagenet(max_images=None):
    """Load Tiny ImageNet validation images and labels."""
    val_annot = TINY_IMAGENET_DIR / "val" / "val_annotations.txt"
    if not val_annot.exists():
        print(f"Error: Tiny ImageNet not found at {TINY_IMAGENET_DIR}")
        return None, None
    
    # Load class ID -> human name mapping
    class_names = {}
    with open(TINY_IMAGENET_DIR / "words.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                class_names[parts[0]] = parts[1]
    
    # Load validation annotations
    val_dir = TINY_IMAGENET_DIR / "val" / "images"
    images = []
    labels = []
    all_valid = []
    with open(val_annot) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                fname, cls_id = parts[0], parts[1]
                all_valid.append((fname, cls_id))
    
    if max_images:
        all_valid = all_valid[:max_images]
    
    for fname, cls_id in all_valid:
        img_path = val_dir / fname
        if img_path.exists():
            images.append(img_path)
            labels.append({
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, cls_id),
            })
    
    return images, labels


def run_model_inference(model_key, image_path):
    """Run a model on a single image via subprocess, return predictions."""
    venv_py = MODEL_VENV[model_key]
    run_py = MODEL_RUN_SCRIPT[model_key]
    
    try:
        result = subprocess.run(
            [str(venv_py), str(run_py), "--image", str(image_path),
             "--task", "describe", "--top-k", "5"],
            capture_output=True, text=True, timeout=120,
        )
        data = json.loads(result.stdout)
        predictions = data.get("predictions", [])
        return predictions
    except Exception as e:
        print(f"    Error: {e}")
        return []


def evaluate_predictions(predictions, gt_label):
    """Check if any prediction matches the ground truth class."""
    gt_name = gt_label["class_name"].lower().strip()
    gt_id = gt_label["class_id"].lower().strip()
    
    top1_label = ""
    top5_labels = []
    
    for i, pred in enumerate(predictions[:5]):
        pred_label = pred.get("label", "").lower().strip()
        if i == 0:
            top1_label = pred_label
        top5_labels.append(pred_label)
    
    # Check if ground truth name or ID is in predictions
    top1_correct = False
    top5_correct = False
    
    # Try matching by class name substring
    for pred_label in [top1_label]:
        if gt_name and (gt_name in pred_label or pred_label in gt_name):
            top1_correct = True
            break
        # Also check if words from gt_name appear in prediction
        gt_words = set(gt_name.split())
        if gt_words:
            pred_words = set(pred_label.split())
            if gt_words & pred_words:  # any word overlap
                top1_correct = True
                break
    
    # More lenient top-5 matching
    for pred_label in top5_labels:
        if gt_name and (gt_name in pred_label or pred_label in gt_name):
            top5_correct = True
            break
        gt_words = set(gt_name.split())
        if gt_words:
            pred_words = set(pred_label.split())
            if gt_words & pred_words:
                top5_correct = True
                break
    
    return top1_correct, top5_correct


def main():
    parser = argparse.ArgumentParser(description="Zero-shot classification benchmark")
    parser.add_argument("--model", type=str, default="dinotool",
                        choices=sorted(CLASSIFICATION_MODELS),
                        help="Model to benchmark")
    parser.add_argument("--max-images", type=int, default=500,
                        help="Number of images to test (default: 500)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-k predictions to consider (default: 5)")
    args = parser.parse_args()
    
    images, gt_labels = load_tiny_imagenet(args.max_images)
    if images is None:
        sys.exit(1)
    
    print(f"\nClassification Benchmark: {args.model}")
    print(f"  Images: {len(images)}")
    print(f"  Dataset: Tiny ImageNet")
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    total_time = 0.0
    
    for idx, (img_path, gt_label) in enumerate(zip(images, gt_labels)):
        t0 = time.time()
        predictions = run_model_inference(args.model, img_path)
        elapsed = time.time() - t0
        total_time += elapsed
        
        top1, top5 = evaluate_predictions(predictions, gt_label)
        if top1:
            top1_correct += 1
        if top5:
            top5_correct += 1
        total += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(images)}] top1={top1_correct/total:.2%} top5={top5_correct/total:.2%} "
                  f"avg={total_time/total:.1f}s/img")
    
    acc_top1 = top1_correct / total if total > 0 else 0
    acc_top5 = top5_correct / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    fps = total / total_time if total_time > 0 else 0
    
    stats = {
        "model": f"Tiny ImageNet Classification ({args.model})",
        "model_key": args.model,
        "task": "classification",
        "dataset": "tiny-imagenet-200",
        "images": total,
        "top1_accuracy": round(acc_top1, 4),
        "top5_accuracy": round(acc_top5, 4),
        "total_inference_time_s": round(total_time, 2),
        "avg_inference_ms": round(avg_time * 1000, 1),
        "fps": round(fps, 2),
    }
    
    save_stats(stats, f"{args.model}_classification")
    
    print(f"\nClassification Results: {args.model}")
    print(f"  Top-1 Accuracy: {acc_top1:.2%} ({top1_correct}/{total})")
    print(f"  Top-5 Accuracy: {acc_top5:.2%} ({top5_correct}/{total})")
    print(f"  Avg per image: {avg_time*1000:.0f}ms")
    print(f"  FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
