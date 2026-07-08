#!/usr/bin/env python3
"""
OCR benchmark for LocateAnything using synthetic text overlays on COCO images.
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    RESULTS_DIR, COCO_DIR, PROJECT_DIR,
    MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY,
    print_comparison, save_stats, build_prompt,
)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
if not Path(FONT_PATH).exists():
    FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

OCR_MODELS = {"locate_anything", "locate_anything_trt"}

WORD_LIST = [
    "hello", "world", "vision", "language", "model", "object", "detection",
    "tracking", "segmentation", "ocr", "text", "benchmark", "locate", "point",
    "coco", "dataset", "accuracy", "speed", "image", "recognition",
]


def pick_random_words(font, max_w, max_h, count=5):
    placed = []
    for _ in range(count):
        word = random.choice(WORD_LIST)
        for attempt in range(20):
            x = random.randint(10, max_w - 60)
            y = random.randint(10, max_h - 20)
            bb = font.getbbox(word)
            tw = bb[2] - bb[0]
            th = bb[3] - bb[1]
            if x + tw > max_w or y + th > max_h:
                continue
            overlap = False
            for (ex, ey, ew, eh, _) in placed:
                if not (x + tw < ex or x > ex + ew or y + th < ey or y > ey + eh):
                    overlap = True
                    break
            if not overlap:
                placed.append((x, y, tw, th, word))
                break
    return placed


def draw_synthetic_text(image, num_words=5):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH, 18)
    except Exception:
        font = ImageFont.load_default()
    w, h = image.size
    regions = pick_random_words(font, w, h, count=num_words)
    text_regions = []
    for (x, y, tw, th, word) in regions:
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        draw.text((x, y), word, fill=color, font=font)
        text_regions.append({
            "word": word,
            "bbox": [x, y, x + tw, y + th],
        })
    return image, text_regions


def parse_la_ocr_boxes(text):
    boxes = []
    for m in re.finditer(r'<box>(.+?)</box>', text, re.IGNORECASE):
        coords = [float(p) for p in re.findall(r'[\d.]+', m.group(1))]
        if len(coords) == 4:
            boxes.append(coords)
        elif len(coords) == 2:
            boxes.append(coords)
    return boxes


def scale_la_boxes(boxes, ow, oh):
    return [[x / 1000 * ow, y / 1000 * oh, x2 / 1000 * ow, y2 / 1000 * oh]
            if len(b) == 4
            else [x / 1000 * ow, y / 1000 * oh]
            for b in boxes
            for (x, y, x2, y2) in [b if len(b) == 4 else (b[0], b[1], b[0], b[1])]]


def iou_2d(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def point_in_bbox(pt, bbox):
    return bbox[0] <= pt[0] <= bbox[2] and bbox[1] <= pt[1] <= bbox[3]


def is_text_detected(det_boxes, gt_region, ow, oh, iou_thresh=0.3, pt_thresh=0.05):
    for db in det_boxes:
        if len(db) == 4:
            scaled = [db[0] / 1000 * ow, db[1] / 1000 * oh, db[2] / 1000 * ow, db[3] / 1000 * oh]
            if iou_2d(scaled, gt_region["bbox"]) >= iou_thresh:
                return True
        elif len(db) == 2:
            sx, sy = db[0] / 1000 * ow, db[1] / 1000 * oh
            gb = gt_region["bbox"]
            gw, gh = gb[2] - gb[0], gb[3] - gb[1]
            if abs(sx - (gb[0] + gb[2]) / 2) <= pt_thresh * ow and abs(sy - (gb[1] + gb[3]) / 2) <= pt_thresh * oh:
                return True
    return False


def benchmark(model_name, max_images=100, words_per_image=5, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")
    if mn not in OCR_MODELS:
        raise ValueError(f"Model {mn!r} does not support OCR. Use --model locate_anything or locate_anything_trt")

    display = MODEL_DISPLAY.get(mn, mn)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"OCR Benchmark: {display}")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    worker = obj

    img_dir = COCO_DIR / "val2017"
    all_images = sorted(img_dir.glob("*.jpg"))
    if max_images:
        all_images = all_images[:max_images]

    total_time = 0.0
    total_gt = 0
    total_detected = 0
    processed = 0

    pbar = tqdm(
        total=len(all_images), unit="img", desc=f"{mn:>16}",
        bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    pbar.set_postfix_str("avg=-  fps=-  det=0/0")

    prompt_text = build_prompt("Detect all the text in box format.", mn)

    for img_path in all_images:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            pbar.update(1)
            continue

        ow, oh = image.size
        synth_img, text_regions = draw_synthetic_text(image.copy(), num_words=words_per_image)

        t0 = time.perf_counter()
        try:
            raw = worker.predict(synth_img, prompt_text, max_new_tokens=512,
                                 temperature=0.1, generation_mode="fast")
        except Exception:
            pbar.update(1)
            continue
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        det_boxes = parse_la_ocr_boxes(raw)
        total_gt += len(text_regions)
        for region in text_regions:
            if is_text_detected(det_boxes, region, ow, oh):
                total_detected += 1

        processed += 1
        pbar.update(1)

    pbar.close()

    if processed == 0:
        print("  No images processed!")
        return None

    avg_time = total_time / processed
    fps = processed / total_time if total_time > 0 else 0
    detection_rate = total_detected / total_gt if total_gt > 0 else 0

    stats = {
        "model": display,
        "model_key": mn,
        "task": "ocr",
        "dataset": "synthetic_coco",
        "images": processed,
        "total_gt_text_regions": total_gt,
        "total_detected": total_detected,
        "detection_rate": round(detection_rate, 4),
        "total_inference_time_s": round(total_time, 3),
        "avg_inference_ms": round(avg_time * 1000, 1),
        "fps": round(fps, 2),
    }

    save_stats(stats, f"{mn}_ocr")
    return stats


def main():
    parser = argparse.ArgumentParser(description="OCR Benchmark (LocateAnything)")
    parser.add_argument("--model", default="locate_anything",
                        choices=sorted(OCR_MODELS),
                        help="Model to benchmark")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--words-per-image", type=int, default=5,
                        help="Number of synthetic words to overlay per image")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark(args.model, max_images=args.max_images,
                      words_per_image=args.words_per_image, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "OCR BENCHMARK COMPARISON")


if __name__ == "__main__":
    main()
