#!/usr/bin/env python3
"""Zero-shot object detection benchmark with standard mAP metrics on COCO."""

import argparse
import json
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    RESULTS_DIR, COCO_DIR, PROJECT_DIR,
    MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY,
    parse_box_tags, parse_json_detections, extract_narrative_boxes,
    parse_loc_tokens, auto_scale_boxes,
    scale_la, scale_qwen, scale_florence2,
    COCO_CAT_NAME_TO_ID,
    save_stats, print_comparison,
)

# ── Models that can do zero-shot detection ────────────────────────────
OBJDET_MODELS = {
    "locate_anything", "locate_anything_trt",
    "florence2", "paligemma", "llama_vision", "phi_vision",
    "phi4_multimodal", "cosmos_nemotron",
    "qwen3_native", "qwen3_thinking",
    "yolo_world", "yolo_worlds", "yolo_worldm", "yolo_worldl",
    "yoloe", "yoloe_11l", "yoloe_26m", "yoloe_26n",
    "diffusion_gemma", "diffusion_gemma_yolo",
    "diffusion_gemma_yolo_pose", "diffusion_gemma_yolo_obb",
    "diffusion_gemma_siglip2", "diffusion_gemma_moonvit",
    "llava_v16_mistral", "llava_onevision",
    "llava_next_video_7b", "llava_next_video_34b", "phi3_vision",
}

LLAVA_VENV_PY = PROJECT_DIR / "Llava" / ".venv" / "bin" / "python"


def zeroshot_od_prompt(category_name):
    return (
        f"Detect all instances of '{category_name}' in this image. "
        f"Return a JSON array where each entry has "
        f"'bbox_2d' = [x1,y1,x2,y2] (pixel coordinates, top-left/bottom-right) "
        f"and 'label' = '{category_name}' and 'confidence' = float 0-1. "
        f"If none exist, return []."
    )


def parse_any_boxes(text):
    boxes = parse_loc_tokens(text)
    if not boxes:
        boxes = parse_box_tags(text)
    if not boxes:
        boxes = parse_json_detections(text)
    if not boxes:
        boxes = extract_narrative_boxes(text)
    return boxes


def parse_detections_with_conf(text):
    """Parse detections returning list of (bbox, confidence) tuples."""
    import re as _re
    text_lower = text.lower()
    if _re.search(r'no\s+(?:object|person|instances|people)', text_lower):
        if not _re.search(r'\[.*?(?:\d|bbox|label)', text_lower):
            return []

    match = _re.search(r'```(?:json)?\s*(.*?)\s*```', text, _re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        depth = 0
        start = None
        raw = None
        for i, c in enumerate(text):
            if c == '[':
                if start is None:
                    start = i
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0 and start is not None:
                    raw = text[start:i + 1]
                    break
        if raw is None:
            return []

    try:
        dets = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(dets, list):
        return []

    out = []
    for d in dets:
        if not isinstance(d, dict):
            continue
        bbox = d.get("bbox_2d") or d.get("bbox") or d.get("box")
        if not bbox or len(bbox) != 4:
            continue
        conf = float(d.get("confidence", d.get("score", 0.5)))
        out.append(([float(v) for v in bbox], conf))
    return out


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


def compute_ap(recalls, precisions):
    """Compute Average Precision using all-point interpolation (VOC style)."""
    # Ensure sorted by recall ascending
    idx = np.argsort(recalls)
    recalls = np.concatenate(([0.0], recalls[idx], [1.0]))
    precisions = np.concatenate(([0.0], precisions[idx], [0.0]))
    # Envelope: take max precision from right
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    # Find unique recall points and compute area
    iou_list = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[iou_list + 1] - recalls[iou_list]) * precisions[iou_list + 1])
    return ap


def compute_map(all_predictions, all_ground_truths, iou_threshold=0.5):
    """Compute mAP@IoU across all categories.
    
    all_predictions: dict[cat_id] -> list of (img_id, bbox, confidence)
    all_ground_truths: dict[cat_id] -> dict[img_id] -> list of bbox
    """
    aps = []
    for cat_id in all_ground_truths:
        gt_per_cat = all_ground_truths[cat_id]
        pred_per_cat = all_predictions.get(cat_id, [])

        # Sort predictions by confidence descending
        pred_per_cat.sort(key=lambda x: -x[2])

        total_gt = sum(len(v) for v in gt_per_cat.values())
        if total_gt == 0:
            continue

        tp = np.zeros(len(pred_per_cat))
        fp = np.zeros(len(pred_per_cat))
        matched = {img_id: set() for img_id in gt_per_cat}

        for i, (img_id, pred_bbox, _conf) in enumerate(pred_per_cat):
            gt_boxes = gt_per_cat.get(img_id, [])
            best_iou = 0
            best_idx = -1
            for j, gt_b in enumerate(gt_boxes):
                if j in matched[img_id]:
                    continue
                iou = compute_iou(pred_bbox, gt_b)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_threshold and best_idx >= 0:
                tp[i] = 1
                matched[img_id].add(best_idx)
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / total_gt
        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


def benchmark_zeroshot_od(model_name, max_images=100, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}")
    if mn not in OBJDET_MODELS:
        raise ValueError(f"Model {mn!r} not in OBJDET_MODELS. Choose: {sorted(OBJDET_MODELS)}")

    is_f2 = mn == "florence2"
    is_pg = mn == "paligemma"
    is_ll = mn == "llama_vision"
    is_ph = mn == "phi_vision"
    is_p4 = mn == "phi4_multimodal"
    is_cm = mn == "cosmos_nemotron"
    is_q3 = mn == "qwen3_native"
    is_th = mn == "qwen3_thinking"
    is_la = mn in ("locate_anything", "locate_anything_trt")
    is_yolo_world = mn.startswith("yolo_world")
    is_yoloe = mn.startswith("yoloe")
    is_dg = mn.startswith("diffusion_gemma")
    is_llava = mn.startswith("llava") or mn == "phi3_vision"

    display = MODEL_DISPLAY.get(mn, mn)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Zero-Shot Object Detection (mAP): {display}")
        print(f"{'=' * 60}")

    obj, extra = MODEL_LOADERS[mn]()
    if is_th:
        detector = obj
    elif is_la:
        la_worker, la_mod = obj
    elif is_q3:
        processor, model = obj
    elif is_yolo_world or is_yoloe:
        model = obj
    elif is_p4:
        model, processor = obj
    elif is_llava or is_dg:
        pass
    else:
        model, processor = obj

    ap_path = COCO_DIR / "annotations" / "instances_val2017.json"
    if not ap_path.exists():
        print(f"Error: COCO annotations not found at {ap_path}")
        return None
    coco_gt = COCO(ap_path)
    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    img_ids = sorted(coco_gt.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]
    img_dir = COCO_DIR / "val2017"

    # Collect per-image, per-category predictions and GT
    # predictions[cat_id] = list of (img_id, [x1,y1,x2,y2], confidence)
    predictions = defaultdict(list)
    ground_truths = defaultdict(lambda: defaultdict(list))
    total_gt_objects = 0
    total_detected = 0
    times = []
    skipped = 0
    n_images = 0

    yolo_cached_results = {}

    pbar = tqdm(total=len(img_ids), unit="img", desc=f"{mn:>20}",
                bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    pbar.set_postfix_str("mAP50=- mAP50:95=-")

    def _run_inference(image, img_path, w, h, cat_name):
        prompt = zeroshot_od_prompt(cat_name)
        dets = []  # list of ([x1,y1,x2,y2], conf)

        if is_f2:
            inputs = processor(text="<OD>", images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
            with torch.no_grad():
                out = model.generate(input_ids=inputs["input_ids"],
                                     pixel_values=inputs["pixel_values"],
                                     max_new_tokens=256, num_beams=1)
            text = processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(text, task="<OD>", image_size=(w, h))
            od = parsed.get("<OD>", {})
            raw_boxes = od.get("bboxes", [])
            labels = od.get("labels", [])
            for rb, lbl in zip(raw_boxes, labels):
                if cat_name.lower() in lbl.lower():
                    dets.append((rb, 0.9))

        elif is_pg:
            inputs = processor(image, f"detect {cat_name}", return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128)
            text = processor.decode(out[0], skip_special_tokens=True)
            raw = parse_loc_tokens(text)
            for box in raw:
                scaled = [box[0]/1024*w, box[1]/1024*h, box[2]/1024*w, box[3]/1024*h]
                dets.append((scaled, 0.85))

        elif is_p4:
            f4_prompt = (
                f"<|user|><|image_1|>{prompt}<|end|><|assistant|>"
            )
            inputs = processor(text=f4_prompt, images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, num_logits_to_keep=1)
            text = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            for box in parse_any_boxes(text):
                dets.append((auto_scale_boxes([box], w, h)[0], 0.8))

        elif is_ll:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}
            ]}]
            chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=chat, images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, temperature=0.1, max_new_tokens=256)
            text = processor.decode(out[0], skip_special_tokens=True)
            if chat in text:
                text = text[len(chat):].strip()
            for box in parse_any_boxes(text):
                dets.append((auto_scale_boxes([box], w, h)[0], 0.8))

        elif is_ph:
            phi_prompt = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
            inputs = processor(phi_prompt, image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
            text = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                               skip_special_tokens=True)
            for box in parse_any_boxes(text):
                dets.append((auto_scale_boxes([box], w, h)[0], 0.8))

        elif is_q3:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}]
            chat = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            inputs = processor(images=image, text=chat, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, temperature=0.1, max_new_tokens=256)
            text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
            for box in parse_any_boxes(text):
                dets.append((scale_qwen([box], w, h)[0], 0.8))

        elif is_th:
            result = detector.detect(image, prompt)
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        bbox = item.get("bbox_2d", item.get("bbox", []))
                        conf = float(item.get("confidence", 0.8))
                        if bbox and len(bbox) == 4:
                            dets.append((auto_scale_boxes([bbox], w, h)[0], conf))

        elif is_cm:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256)
            text = processor.decode(out[0][inputs["input_ids"].shape[-1]:],
                                    skip_special_tokens=True)
            for box in parse_any_boxes(text):
                dets.append((scale_qwen([box], w, h)[0], 0.8))

        elif is_la:
            if is_la:
                bboxes, scores, labels = la_worker.predict(str(img_path), cat_name)
                for b, s in zip(bboxes, scores):
                    dets.append((b, float(s)))

        elif is_yolo_world or is_yoloe:
            cached = yolo_cached_results.get(cat_name, [])
            for b in cached:
                dets.append((b, 0.85))

        elif is_dg:
            run_py = PROJECT_DIR / "diffusion_gemma_vl" / "run.py"
            dg_args = [sys.executable, str(run_py), "--image", str(img_path),
                       "--task", "grounding", "--prompt", prompt, "--n-predict", "128"]
            if "siglip2" in mn:
                dg_args += ["--encoder", "siglip2"]
            elif "moonvit" in mn:
                dg_args += ["--encoder", "moonvit"]
            sub_r = subprocess.run(dg_args, capture_output=True, text=True, timeout=300)
            text = sub_r.stdout.strip()
            for box in parse_any_boxes(text):
                dets.append((box, 0.7))

        elif is_llava:
            run_py = PROJECT_DIR / "Llava" / "run.py"
            llava_key_map = {
                "llava_v16_mistral": "llava-v1.6-mistral",
                "llava_onevision": "llava-onevision",
                "llava_next_video_7b": "llava-next-video-7b",
                "llava_next_video_34b": "llava-next-video-34b",
                "phi3_vision": "phi-3-vision",
            }
            lk = llava_key_map.get(mn, mn)
            llava_args = [str(LLAVA_VENV_PY), str(run_py), "--model", lk,
                          "--image", str(img_path), "--task", "vqa",
                          "--prompt", prompt, "--max-new-tokens", "256"]
            if mn == "llava_next_video_34b":
                llava_args += ["--quantize"]
            sub_r = subprocess.run(llava_args, capture_output=True, text=True, timeout=1800)
            try:
                data = json.loads(sub_r.stdout)
                text = data.get("response", "")
            except (json.JSONDecodeError, KeyError):
                text = sub_r.stdout
            for box in parse_any_boxes(text):
                dets.append((box, 0.7))

        return dets

    for img_id in img_ids:
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
        cat_counts = Counter(a["category_id"] for a in anns)
        cat_ids_present = list(cat_counts.keys())

        # Record ground truth
        for cat_id in cat_ids_present:
            gt_boxes = []
            for a in anns:
                if a["category_id"] == cat_id:
                    x, y, w, h = a["bbox"]
                    gt_boxes.append([x, y, x + w, y + h])
            ground_truths[cat_id][img_id] = gt_boxes
            total_gt_objects += len(gt_boxes)

        # YOLO: run once for all categories
        if is_yolo_world or is_yoloe:
            t0 = time.perf_counter()
            all_cat_names = [cat_id_to_name[cid] for cid in cat_ids_present]
            yolo_results = model(str(img_path), conf=0.25, iou=0.45, verbose=False)
            yolo_time = time.perf_counter() - t0
            yolo_cached_results.clear()
            if yolo_results and yolo_results[0].boxes is not None:
                boxes_xyxy = yolo_results[0].boxes.xyxy.cpu().tolist()
                boxes_cls = yolo_results[0].boxes.cls.cpu().tolist()
                yolo_names_list = [model.names[i] for i in range(len(model.names))]
                name_to_idx = {n: i for i, n in enumerate(yolo_names_list)}
                for name in all_cat_names:
                    target_idx = name_to_idx.get(name, -1)
                    yolo_cached_results[name] = [
                        b for b, c in zip(boxes_xyxy, boxes_cls) if int(c) == target_idx
                    ]

        # Per-category inference
        for cat_id in cat_ids_present:
            cat_name = cat_id_to_name[cat_id]

            t0 = time.perf_counter()
            try:
                dets = _run_inference(image, img_path, ow, oh, cat_name)
            except Exception as e:
                tqdm.write(f"  Error on {img_info['file_name']} ({cat_name}): {e}")
                dets = []
            elapsed = time.perf_counter() - t0

            if not (is_yolo_world or is_yoloe):
                times.append(elapsed)

            for bbox, conf in dets:
                predictions[cat_id].append((img_id, bbox, conf))
                total_detected += 1

        if is_yolo_world or is_yoloe:
            if cat_ids_present:
                times.append(yolo_time)

        n_images += 1
        total_sec = sum(times)
        n_inf = len(times)
        avg_ms = total_sec / max(n_inf, 1) * 1000
        fps_now = n_inf / max(total_sec, 1)

        # Compute running mAP@50 for display
        if n_images % 5 == 0 or n_images == len(img_ids):
            m50 = compute_map(predictions, ground_truths, iou_threshold=0.5)
            pbar.set_postfix_str(f"mAP50={m50:.3f}  fps={fps_now:.1f}")

        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    # Final mAP computation
    mAP50 = compute_map(predictions, ground_truths, iou_threshold=0.5)
    mAP50_95 = np.mean([
        compute_map(predictions, ground_truths, iou_threshold=t)
        for t in np.arange(0.5, 1.0, 0.05)
    ])

    n_inferences = len(times)
    total_sec = sum(times)
    fps = n_inferences / max(total_sec, 1)
    avg_ms = total_sec / max(n_inferences, 1) * 1000

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Zero-Shot OD Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Images:       {n_images}")
        print(f"  Skipped:      {skipped}")
        print(f"  Inferences:   {n_inferences}")
        print(f"  Total time:   {total_sec:.1f}s")
        print(f"  Avg ms/inf:   {avg_ms:.1f}")
        print(f"  FPS:          {fps:.2f}")
        print(f"  GT objects:   {total_gt_objects}")
        print(f"  Detected:     {total_detected}")
        print(f"  mAP@50:       {mAP50:.4f}")
        print(f"  mAP@50:95:    {mAP50_95:.4f}")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    stats = {
        "model": display,
        "model_key": mn,
        "task": "object_detection",
        "dataset": "coco",
        "images": n_images,
        "skipped": skipped,
        "total_inference_time_s": round(total_sec, 3),
        "avg_inference_ms": round(avg_ms, 2),
        "fps": round(fps, 2),
        "total_gt": total_gt_objects,
        "total_detected": total_detected,
        "mAP@50": round(mAP50, 4),
        "mAP@50:95": round(mAP50_95, 4),
    }

    save_stats(stats, f"{safe_name}_od_map")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot OD Benchmark (COCO mAP)")
    all_choices = sorted(set(list(MODEL_LOADERS.keys()) + list(MODEL_ALIASES.keys())))
    parser.add_argument("--model", choices=all_choices, required=True)
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_zeroshot_od(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "ZERO-SHOT OD COMPARISON")


if __name__ == "__main__":
    main()
