#!/usr/bin/env python3
"""
Pointing benchmark for LocateAnything using COCO keypoints.
Asks the model to point to each keypoint and checks accuracy.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    RESULTS_DIR, COCO_DIR, PROJECT_DIR,
    MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY,
    print_comparison, save_stats,
)

POINTING_MODELS = {"locate_anything", "locate_anything_trt"}

COCO_KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


def parse_pointing_output(text):
    pt = None
    m = re.search(r'<box>(.+?)</box>', text, re.IGNORECASE)
    if m:
        coords = [float(p) for p in re.findall(r'[\d.]+', m.group(1))]
        if len(coords) == 2:
            pt = (coords[0], coords[1])
    return pt


def normalized_distance(px, py, gx, gy, ow, oh):
    dx = (px / 1000 * ow - gx) / ow
    dy = (py / 1000 * oh - gy) / oh
    return (dx ** 2 + dy ** 2) ** 0.5


def benchmark(model_name, max_images=100, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")
    if mn not in POINTING_MODELS:
        raise ValueError(f"Model {mn!r} does not support pointing. Use --model locate_anything or locate_anything_trt")

    display = MODEL_DISPLAY.get(mn, mn)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Pointing Benchmark: {display}")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    worker = obj

    ap = COCO_DIR / "annotations" / "person_keypoints_val2017.json"
    if not ap.exists():
        print(f"Error: keypoints annotations not found at {ap}")
        return None
    coco_gt = COCO(ap)
    img_ids = coco_gt.getImgIds()
    if max_images:
        img_ids = img_ids[:max_images]
    img_dir = COCO_DIR / "val2017"

    total_time = 0.0
    total_kpts = 0
    acc_005 = 0
    acc_010 = 0
    processed = 0

    pbar = tqdm(
        total=len(img_ids), unit="img", desc=f"{mn:>16}",
        bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    pbar.set_postfix_str("avg=-  fps=-  kpts=0")

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            pbar.update(1)
            continue

        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        if not anns:
            pbar.update(1)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            pbar.update(1)
            continue

        ow, oh = image.size
        frame_time = 0.0
        frame_kpts = 0
        frame_acc_005 = 0
        frame_acc_010 = 0

        for ann in anns:
            kpts = ann.get("keypoints", [])
            for kid in range(17):
                kx = kpts[kid * 3]
                ky = kpts[kid * 3 + 1]
                kv = kpts[kid * 3 + 2]
                if kv < 1:
                    continue
                kp_name = COCO_KEYPOINT_NAMES.get(kid, f"keypoint_{kid}")
                prompt = f"Point to: {kp_name}."

                t0 = time.perf_counter()
                try:
                    raw = worker.predict(image, prompt, max_new_tokens=64,
                                         temperature=0.1, generation_mode="fast")
                except Exception:
                    continue
                elapsed = time.perf_counter() - t0
                frame_time += elapsed

                pt = parse_pointing_output(raw)
                if pt is not None:
                    nd = normalized_distance(pt[0], pt[1], kx, ky, ow, oh)
                    if nd <= 0.05:
                        frame_acc_005 += 1
                    if nd <= 0.10:
                        frame_acc_010 += 1
                frame_kpts += 1

        if frame_kpts > 0:
            total_time += frame_time
            total_kpts += frame_kpts
            acc_005 += frame_acc_005
            acc_010 += frame_acc_010
            processed += 1

        pbar.update(1)

    pbar.close()

    if processed == 0 or total_kpts == 0:
        print("  No keypoints processed!")
        return None

    avg_time = total_time / total_kpts
    fps = processed / total_time if total_time > 0 else 0

    stats = {
        "model": display,
        "model_key": mn,
        "task": "pointing",
        "dataset": "coco_keypoints",
        "images": processed,
        "total_keypoints": total_kpts,
        "acc@0.05": round(acc_005 / total_kpts, 4),
        "acc@0.10": round(acc_010 / total_kpts, 4),
        "total_inference_time_s": round(total_time, 3),
        "avg_inference_ms": round(avg_time * 1000, 1),
        "fps": round(fps, 2),
    }

    save_stats(stats, f"{mn}_pointing")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Pointing Benchmark (LocateAnything)")
    parser.add_argument("--model", default="locate_anything",
                        choices=sorted(POINTING_MODELS),
                        help="Model to benchmark")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "POINTING BENCHMARK COMPARISON")


if __name__ == "__main__":
    main()
