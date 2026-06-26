import argparse
import json
import time

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from common import RESULTS_DIR, COCO_DIR, MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY, print_comparison, save_stats

POSE_MODELS = {"yolo26_pose", "yolo_pose"}


def benchmark(model_name, max_images=100, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")
    if mn not in POSE_MODELS and "pose" not in mn:
        raise ValueError(f"Model {mn!r} does not support pose estimation. Use --model yolo_pose or yolo26_pose")

    display = MODEL_DISPLAY.get(mn, mn)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Pose Estimation: {display}")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    model = obj

    ap = COCO_DIR / "annotations" / "person_keypoints_val2017.json"
    if not ap.exists():
        print(f"Error: keypoints annotations not found at {ap}")
        return None
    coco_gt = COCO(ap)
    img_ids = coco_gt.getImgIds()
    if max_images:
        img_ids = img_ids[:max_images]
    img_dir = COCO_DIR / "val2017"

    results = []
    times = []
    total_gt = 0
    skipped = 0

    pbar = tqdm(
        total=len(img_ids), unit="img", desc=f"{mn:>16}",
        bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    pbar.set_postfix_str("avg=-  fps=-  kpts=0")

    for idx, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            skipped += 1
            pbar.update(1)
            continue

        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        if not anns:
            skipped += 1
            pbar.update(1)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            pbar.update(1)
            continue

        ow, oh = image.size
        t0 = time.perf_counter()
        try:
            yolo_out = model(image, verbose=False)[0]
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            if yolo_out.keypoints is not None and len(yolo_out.keypoints) > 0:
                kpts_xy = yolo_out.keypoints.xy.cpu().numpy()
                kpts_conf = yolo_out.keypoints.conf.cpu().numpy()
                box_scores = yolo_out.boxes.conf.cpu().numpy() if yolo_out.boxes is not None else None

                for i in range(len(kpts_xy)):
                    keypoints = []
                    for j in range(17):
                        x, y = kpts_xy[i, j]
                        c = kpts_conf[i, j]
                        v = 2 if c > 0.5 else 0
                        keypoints.extend([float(x), float(y), v])

                    overall_score = float(box_scores[i]) if box_scores is not None else float(kpts_conf[i].mean())

                    results.append({
                        "image_id": img_id,
                        "category_id": 1,
                        "keypoints": keypoints,
                        "score": overall_score,
                    })
        except Exception as e:
            tqdm.write(f"  Error on {img_info['file_name']}: {e}")
            skipped += 1
            pbar.update(1)
            continue

        total_gt += len(anns)
        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    avg_time = sum(times) / len(times)
    fps = len(times) / sum(times)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Speed Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Images processed:  {len(times)}")
        print(f"  Skipped:           {skipped}")
        print(f"  Total inference:   {sum(times):.2f}s")
        print(f"  Avg per image:     {avg_time * 1000:.1f}ms")
        print(f"  FPS:               {fps:.2f}")
        print(f"  Total GT persons:  {total_gt}")

    AP = None
    AP_50 = None
    count_person_detections = sum(1 for r in results if r.get("category_id") == 1)

    if results and count_person_detections > 0:
        safe_name = mn.replace("/", "_").replace(" ", "_")
        rp = RESULTS_DIR / f"{safe_name}_pose_results.json"
        with open(rp, "w") as f:
            json.dump(results, f)

        try:
            coco_dt = coco_gt.loadRes(results)
            ev = COCOeval(coco_gt, coco_dt, "keypoints")
            ev.params.imgIds = list(set(r["image_id"] for r in results))
            ev.evaluate()
            ev.accumulate()
            if verbose:
                ev.summarize()
            AP = float(ev.stats[0])
            AP_50 = float(ev.stats[1])
            if verbose:
                print(f"\n  AP@50:95 (keypoints) = {AP:.4f}")
                print(f"  AP@50   (keypoints) = {AP_50:.4f}")
        except Exception as e:
            print(f"  Keypoint AP computation skipped: {e}")

    kpt_stats = {
        "model": display,
        "model_key": mn,
        "task": "pose_estimation",
        "images": len(times),
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "total_gt_persons": total_gt,
        "total_detected_persons": count_person_detections,
        "AP@50:95_keypoints": AP,
        "AP@50_keypoints": AP_50,
    }

    save_stats(kpt_stats, f"{safe_name}_pose")
    return kpt_stats


def main():
    parser = argparse.ArgumentParser(description="Pose Estimation Benchmark (COCO Keypoints)")
    parser.add_argument("--model", default="yolo26_pose",
                        help="Model to benchmark (use yolo_pose or yolo26_pose)")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "POSE ESTIMATION COMPARISON")


if __name__ == "__main__":
    main()
