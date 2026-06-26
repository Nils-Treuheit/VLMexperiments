import argparse
import json
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import (
    RESULTS_DIR, DOTA_DIR, DOTA_CATEGORIES, DOTA_CAT_NAME_TO_ID,
    MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY, print_comparison, save_stats,
)

OBB_MODELS = {"yolo26_obb", "yolo_obb"}

DOTA_CLASS_NAMES = [
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field",
    "roundabout", "harbor", "swimming-pool", "helicopter",
]


def ensure_dota_data(max_images=None):
    images_dir = DOTA_DIR / "images"
    labels_dir = DOTA_DIR / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return None

    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        return None
    if max_images:
        image_files = image_files[:max_images]

    gt = []
    for imp in image_files:
        lp = labels_dir / f"{imp.stem}.txt"
        if not lp.exists():
            continue
        try:
            with Image.open(imp) as im:
                w, h = im.size
        except Exception:
            continue
        annotations = []
        with open(lp) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "imagesource", "gsd")):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                try:
                    coords = [float(p) for p in parts[:8]]
                    cname = parts[8]
                    diff = int(parts[9]) if len(parts) > 9 else 0
                except (ValueError, IndexError):
                    continue
                if diff != 0 or cname not in DOTA_CAT_NAME_TO_ID:
                    continue
                annotations.append({
                    "category": cname,
                    "cat_id": DOTA_CAT_NAME_TO_ID[cname],
                    "poly": coords,
                })
        if annotations:
            gt.append({
                "file": imp,
                "width": w,
                "height": h,
                "annotations": annotations,
            })
    return gt, image_files


def benchmark_obb(model_name, max_images=100, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")

    display = MODEL_DISPLAY.get(mn, mn)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"OBB Detection: {display}  |  Dataset: DOTA-v1.0")
        print(f"{'=' * 60}")

    dota_data = ensure_dota_data(max_images=max_images)
    if dota_data is None:
        print(f"Error: DOTA data not found at {DOTA_DIR}")
        print("Run: python download_datasets.py --dota N")
        return None

    gt_list, _ = dota_data

    obj, _ = MODEL_LOADERS[mn]()
    model = obj

    results = []
    times = []
    total_gt = 0
    total_det = 0
    skipped = 0

    pbar = tqdm(
        total=len(gt_list), unit="img", desc=f"{mn:>16}",
        bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    pbar.set_postfix_str("avg=-  fps=-  det=0/0")

    for idx, entry in enumerate(gt_list):
        imp = entry["file"]
        try:
            image = Image.open(imp).convert("RGB")
        except Exception:
            skipped += 1
            pbar.update(1)
            continue

        t0 = time.perf_counter()
        try:
            yolo_out = model(image, verbose=False)[0]
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            if yolo_out.obb is not None and len(yolo_out.obb) > 0:
                obb_data = yolo_out.obb
                xyxyxyxyn = obb_data.xyxyxyxy.cpu().numpy() if hasattr(obb_data, 'xyxyxyxy') else None
                if xyxyxyxyn is None:
                    xyxyxyxyn = obb_data.xyxyxyxyn.cpu().numpy()
                    ow, oh = entry["width"], entry["height"]
                    xyxyxyxyn[:, :, 0] *= ow
                    xyxyxyxyn[:, :, 1] *= oh

                confs = obb_data.conf.cpu().numpy()
                cls = obb_data.cls.cpu().numpy().astype(int)

                for i in range(len(cls)):
                    if confs[i] < 0.25:
                        continue
                    poly = xyxyxyxyn[i].reshape(-1).tolist()
                    poly = [float(v) for v in poly]
                    if len(poly) != 8:
                        continue
                    results.append({
                        "file": imp.name,
                        "poly": poly,
                        "score": float(confs[i]),
                        "class_id": int(cls[i]),
                    })
                    total_det += 1
        except Exception as e:
            tqdm.write(f"  Error on {imp.name}: {e}")
            skipped += 1
            pbar.update(1)
            continue

        total_gt += len(entry["annotations"])
        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    avg_time = sum(times) / len(times)
    fps = len(times) / sum(times)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Speed Results: {display} on DOTA-v1.0")
        print(f"{'=' * 60}")
        print(f"  Images processed:  {len(times)}")
        print(f"  Skipped:           {skipped}")
        print(f"  Total inference:   {sum(times):.2f}s")
        print(f"  Avg per image:     {avg_time * 1000:.1f}ms")
        print(f"  FPS:               {fps:.2f}")
        print(f"  Total GT objects:  {total_gt}")
        print(f"  Total detected:    {total_det}")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    rp = RESULTS_DIR / f"{safe_name}_obb_results.json"
    with open(rp, "w") as f:
        json.dump({"results": results, "total_gt": total_gt}, f, indent=2)
    if verbose:
        print(f"\n  Raw results saved to: {rp}")

    stats = {
        "model": display,
        "model_key": mn,
        "task": "obb_detection",
        "dataset": "dota",
        "images": len(times),
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "total_gt": total_gt,
        "total_detected": total_det,
        "mAP@50:95": None,
        "mAP@50": None,
    }

    save_stats(stats, f"{safe_name}_obb")
    return stats


def main():
    parser = argparse.ArgumentParser(description="OBB Detection Benchmark (DOTA-v1.0)")
    parser.add_argument("--model", default="yolo26_obb",
                        help="Model to benchmark (use yolo_obb or yolo26_obb)")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_obb(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "OBB DETECTION COMPARISON — DOTA-v1.0")


if __name__ == "__main__":
    main()
