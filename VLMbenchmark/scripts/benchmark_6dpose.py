import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import (
    RESULTS_DIR, MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY,
    print_comparison, save_stats,
)

LINEMOD_DIR = Path("/mnt/HDD1/Project_Data/public_datasets/linemod")
LM_DIR = LINEMOD_DIR / "lm"
TEST_DIR = LINEMOD_DIR / "test"

LINEMOD_OBJECT_IDS = list(range(1, 16))

LINEMOD_OBJECT_NAMES = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}


def load_test_targets():
    tp = LM_DIR / "test_targets_bop19.json"
    with open(tp) as f:
        targets = json.load(f)
    return targets


def load_camera():
    cp = LM_DIR / "camera.json"
    with open(cp) as f:
        return json.load(f)


def load_scene_gt(scene_id):
    sp = TEST_DIR / f"{scene_id:06d}" / "scene_gt.json"
    with open(sp) as f:
        return json.load(f)


def load_scene_gt_info(scene_id):
    sp = TEST_DIR / f"{scene_id:06d}" / "scene_gt_info.json"
    with open(sp) as f:
        return json.load(f)


def compute_detection_rate(results, targets):
    """Compute detection rate: fraction of target objects that were detected."""
    target_set = set()
    for t in targets:
        target_set.add((t["scene_id"], t["im_id"], t["obj_id"]))
    if not target_set:
        return 0.0, len(target_set), 0
    detected = 0
    for r in results:
        key = (r["scene_id"], r["im_id"], r["obj_id"])
        if key in target_set:
            detected += 1
    return detected / len(target_set), len(target_set), detected


def benchmark(model_name, max_images=None, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")

    is_yolo = mn.startswith("yolo")

    display = MODEL_DISPLAY.get(mn, mn)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"6D Pose (Detection): {display}  |  Dataset: Linemod (BOP)")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    if is_yolo:
        model = obj
    else:
        model = obj

    targets = load_test_targets()
    camera = load_camera()

    if max_images is not None and max_images > 0:
        targets = targets[:max_images]

    scene_cache = {}
    gt_info_cache = {}

    all_results = []
    times = []
    total_gt_objects = 0
    total_detected = 0
    skipped = 0

    pbar = tqdm(
        total=len(targets), unit="img", desc=f"{mn:>16}",
        bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    pbar.set_postfix_str("avg=-  fps=-  det=0/0")

    for target in targets:
        scene_id = target["scene_id"]
        im_id = target["im_id"]
        obj_id = target["obj_id"]

        img_path = TEST_DIR / f"{scene_id:06d}" / "rgb" / f"{im_id:06d}.png"
        if not img_path.exists():
            skipped += 1
            pbar.update(1)
            continue

        if scene_id not in scene_cache:
            scene_cache[scene_id] = load_scene_gt(scene_id)
            gt_info_cache[scene_id] = load_scene_gt_info(scene_id)

        scene_gt = scene_cache[scene_id]
        scene_gt_info = gt_info_cache[scene_id]

        str_im_id = str(im_id)
        gt_anns = scene_gt.get(str_im_id, [])
        gt_info_anns = scene_gt_info.get(str_im_id, [])

        gt_box = None
        for ann, info in zip(gt_anns, gt_info_anns):
            if ann["obj_id"] == obj_id:
                bb = info.get("bbox_visib") or info.get("bbox_obj")
                if bb is not None:
                    gt_box = bb
                break

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            pbar.update(1)
            continue

        ow, oh = image.size

        t0 = time.perf_counter()
        try:
            detections = []
            if is_yolo:
                yolo_out = model(image, verbose=False)[0]
                if yolo_out.boxes is not None:
                    xyxy = yolo_out.boxes.xyxy.cpu().numpy()
                    scores = yolo_out.boxes.conf.cpu().numpy()
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(scores[i]),
                            "obj_id": obj_id,
                        })

            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except Exception as e:
            tqdm.write(f"  Error on scene {scene_id}/img {im_id}: {e}")
            skipped += 1
            pbar.update(1)
            continue

        total_gt_objects += 1
        total_detected += len(detections)

        for det in detections:
            all_results.append({
                "scene_id": scene_id,
                "im_id": im_id,
                "obj_id": det["obj_id"],
                "bbox": det["bbox"],
                "score": det["score"],
                "gt_bbox": gt_box,
            })

        avg_ms = sum(times) / len(times) * 1000
        fps_now = len(times) / sum(times)
        pbar.set_postfix_str(f"avg={avg_ms:.0f}ms  fps={fps_now:.2f}  det={total_detected}/{total_gt_objects}")
        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    avg_time = sum(times) / len(times)
    fps = len(times) / sum(times)
    det_rate, total_gt, total_det = compute_detection_rate(all_results, targets)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Speed Results: {display} on Linemod")
        print(f"{'=' * 60}")
        print(f"  Images processed:  {len(times)}")
        print(f"  Skipped:           {skipped}")
        print(f"  Total inference:   {sum(times):.2f}s")
        print(f"  Avg per image:     {avg_time * 1000:.1f}ms")
        print(f"  FPS:               {fps:.2f}")
        print(f"  Total GT objects:  {total_gt}")
        print(f"  Total detected:    {total_det}")
        print(f"  Detection rate:    {det_rate:.4f}")

    stats = {
        "model": display,
        "model_key": mn,
        "task": "6d_pose",
        "dataset": "linemod",
        "images": len(times),
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "detection_rate": round(det_rate, 4),
        "total_gt_objects": total_gt,
        "total_detected": total_det,
    }

    safe_name = mn.replace("/", "_").replace(" ", "_")
    save_stats(stats, f"{safe_name}_linemod_6dpose")

    if verbose:
        print(f"\n{'─' * 60}")
        print("  6D Pose Metrics Placeholder")
        print(f"{'─' * 60}")
        print("  ADD / ADD-S computation not yet implemented.")
        print("  To compute ADD(-S):")
        print("    1. Load the estimated pose (cam_R_m2c, cam_t_m2c)")
        print("    2. Load the ground truth pose from scene_gt.json")
        print("    3. Load the 3D model points from models/*.ply")
        print("    4. Transform model points using estimated and GT poses")
        print("    5. Compute mean point distance (ADD) or closest point distance (ADD-S)")
        print("    6. Threshold by object diameter from models_info.json")
        print()

    return stats


def main():
    parser = argparse.ArgumentParser(description="6D Pose Estimation Benchmark (Linemod BOP)")
    all_choices = list(MODEL_LOADERS) + list(MODEL_ALIASES)
    parser.add_argument("--model", choices=all_choices, default="yolo26",
                        help="Model to benchmark")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of test images (default: all BOP targets)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "6D POSE ESTIMATION — Linemod")


if __name__ == "__main__":
    main()
