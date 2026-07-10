import argparse
import json
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics.utils.metrics import ap_per_class, batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

from common import (
    RESULTS_DIR, DOTA_DIR, DOTA_CATEGORIES, DOTA_CAT_NAME_TO_ID,
    MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY, print_comparison, save_stats,
)

OBB_MODELS = {"yolo26_obb", "yolo_obb"}

DOTA_CLASS_NAMES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
]

# Mapping from DOTA category names (from common.py) to YOLO model class IDs
DOTA_NAME_TO_YOLO_ID = {
    "plane": 0, "ship": 1, "storage-tank": 2, "baseball-diamond": 3,
    "tennis-court": 4, "basketball-court": 5, "ground-track-field": 6,
    "harbor": 7, "bridge": 8, "large-vehicle": 9, "small-vehicle": 10,
    "helicopter": 11, "roundabout": 12, "soccer-ball-field": 13,
    "swimming-pool": 14,
}


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
                    "cat_id": DOTA_NAME_TO_YOLO_ID.get(cname, DOTA_CAT_NAME_TO_ID[cname]),
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

    # Compute OBB mAP
    mAP_50_95 = None
    mAP_50 = None
    if results and gt_list:
        try:
            all_tp = []   # (N, 10) - N predictions across all images
            all_conf = []
            all_pred_cls = []
            iou_thresholds = np.linspace(0.5, 0.95, 10)
            for entry in gt_list:
                imp_name = entry["file"].name
                img_preds = [r for r in results if r["file"] == imp_name]
                if not img_preds:
                    continue
                gt_anns = entry["annotations"]
                if not gt_anns:
                    continue
                img_preds = sorted(img_preds, key=lambda x: x["score"], reverse=True)
                gt_polys = [np.array(a["poly"], dtype=np.float32).reshape(1, -1) for a in gt_anns]
                gt_cls = [a["cat_id"] for a in gt_anns]
                gt_xywhr = np.concatenate([xyxyxyxy2xywhr(gp) for gp in gt_polys], axis=0).astype(np.float32)
                pred_polys = [np.array(p["poly"], dtype=np.float32).reshape(1, -1) for p in img_preds]
                pred_cls = [p["class_id"] for p in img_preds]
                pred_conf = [p["score"] for p in img_preds]

                if not pred_polys:
                    continue
                pred_xywhr = np.concatenate([xyxyxyxy2xywhr(pp) for pp in pred_polys], axis=0)
                ious = batch_probiou(torch.from_numpy(gt_xywhr).float(),
                                     torch.from_numpy(pred_xywhr).float()).cpu().numpy()

                # Per-threshold matching, building (P, T) TP array per image
                img_tp = np.zeros((len(img_preds), len(iou_thresholds)), dtype=bool)
                for ti, thresh in enumerate(iou_thresholds):
                    matched_gt = set()
                    for pi in range(len(img_preds)):
                        best_iou = 0.0
                        best_gi = -1
                        for gi in range(len(gt_anns)):
                            if gi in matched_gt:
                                continue
                            if gt_cls[gi] != pred_cls[pi]:
                                continue
                            if ious[gi, pi] > best_iou:
                                best_iou = ious[gi, pi]
                                best_gi = gi
                        if best_iou >= thresh and best_gi >= 0:
                            matched_gt.add(best_gi)
                            img_tp[pi, ti] = True

                all_tp.append(img_tp)
                all_conf.extend(pred_conf)
                all_pred_cls.extend(pred_cls)

            if all_tp:
                tp_arr = np.concatenate(all_tp, axis=0)
                conf_arr = np.array(all_conf)
                pred_cls_arr = np.array(all_pred_cls)
                _, _, _, _, _, ap, _, _, _, _, _, _ = ap_per_class(
                    tp_arr, conf_arr, pred_cls_arr, pred_cls_arr,
                    names={i: DOTA_CLASS_NAMES[i] for i in range(len(DOTA_CLASS_NAMES))},
                )
                mAP_50_95 = float(ap.mean()) if len(ap) > 0 else 0.0
                mAP_50 = float(np.nanmean(ap[:, 0])) if ap.shape[1] > 0 else 0.0
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  mAP computation error: {e}")

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
        "mAP@50:95": mAP_50_95,
        "mAP@50": mAP_50,
    }

    save_stats(stats, f"{safe_name}_obb")
    return stats


def main():
    parser = argparse.ArgumentParser(description="OBB Detection Benchmark (DOTA-v1.0)")
    parser.add_argument("--model", default="yolo26_obb",
                        help="Model to benchmark (use yolo_obb or yolo26_obb)")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--samples-file", type=str, default=None, help="Path to samples file (unused, for compatibility)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_obb(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "OBB DETECTION COMPARISON — DOTA-v1.0")


if __name__ == "__main__":
    main()
