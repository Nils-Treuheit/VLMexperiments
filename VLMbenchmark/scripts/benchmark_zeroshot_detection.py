#!/usr/bin/env python3
"""Zero-shot detection benchmark — every VLM detects objects via NL prompts."""

import argparse
import json
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
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

ZEROSHOT_DETECTION_MODELS = {
    "florence2", "paligemma", "llama_vision", "phi_vision", "cosmos_nemotron",
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


def zeroshot_prompt(category_name):
    return (
        f"Output a JSON list of ALL '{category_name}' objects in this image. "
        f"Each entry must have 'bbox_2d' = [x1,y1,x2,y2] and 'label' = '{category_name}'. "
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


def benchmark_zeroshot_detection(model_name, max_images=100, max_categories=None, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}")
    if mn not in ZEROSHOT_DETECTION_MODELS:
        raise ValueError(f"Model {mn!r} not in zero-shot detection list. Choose from: {ZEROSHOT_DETECTION_MODELS}")

    is_f2 = mn == "florence2"
    is_pg = mn == "paligemma"
    is_ll = mn == "llama_vision"
    is_ph = mn == "phi_vision"
    is_cm = mn == "cosmos_nemotron"
    is_q3 = mn == "qwen3_native"
    is_th = mn == "qwen3_thinking"
    is_yolo_world = mn.startswith("yolo_world")
    is_yoloe = mn.startswith("yoloe")
    is_dg = mn.startswith("diffusion_gemma")
    is_llava = mn.startswith("llava") or mn == "phi3_vision"

    display = MODEL_DISPLAY.get(mn, mn)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Zero-shot Detection: {display}")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    if is_th:
        detector = obj
    elif is_q3:
        processor, model = obj
    elif is_yolo_world or is_yoloe:
        model = obj
    elif is_llava or is_dg:
        pass
    else:
        model, processor = obj

    ap = COCO_DIR / "annotations" / "instances_val2017.json"
    if not ap.exists():
        print(f"Error: COCO annotations not found at {ap}")
        return None
    coco_gt = COCO(ap)
    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    img_ids = coco_gt.getImgIds()
    if max_images:
        img_ids = img_ids[:max_images]
    img_dir = COCO_DIR / "val2017"

    yolo_cached_results = {}

    times = []
    n_images_processed = 0
    total_gt_all = 0
    total_b3_correct = 0
    total_pass_correct = [0, 0, 0]
    skipped = 0

    pbar = tqdm(total=len(img_ids), unit="img", desc=f"{mn:>16}",
                bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    pbar.set_postfix_str("avg=-  fps=-  b3=-")

    def _count_correct(det_boxes, gt_boxes_list):
        matched = set()
        count = 0
        for b in det_boxes:
            bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
            for gi, gtb in enumerate(gt_boxes_list):
                if gi in matched:
                    continue
                gx1, gy1, gw, gh = gtb
                gx2, gy2 = gx1 + gw, gy1 + gh
                ix1 = max(bx1, gx1); iy1 = max(by1, gy1)
                ix2 = min(bx2, gx2); iy2 = min(by2, gy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area_b = (bx2 - bx1) * (by2 - by1)
                    area_gt = gw * gh
                    iou = inter / (area_b + area_gt - inter + 1e-8)
                    if iou >= 0.5:
                        count += 1
                        matched.add(gi)
                        break
        return count

    def _run_single_pass(img, img_path, w, h, cat_name):
        """Run one inference pass for a single category; returns list of [x1,y1,x2,y2]."""
        prompt = zeroshot_prompt(cat_name)
        boxes = []

        if is_f2:
            inputs = processor(text="<OD>", images=img, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256, num_beams=1,
                )
            text = processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(text, task="<OD>", image_size=(w, h))
            od = parsed.get("<OD>", {})
            raw_boxes = od.get("bboxes", [])
            labels = od.get("labels", [])
            for rb, lbl in zip(raw_boxes, labels):
                if cat_name.lower() in lbl.lower():
                    boxes.append(rb)

        elif is_pg:
            pg_prompt = f"detect {cat_name}"
            inputs = processor(img, pg_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128)
            text = processor.decode(out[0], skip_special_tokens=True)
            raw = parse_loc_tokens(text)
            boxes = [[x / 1024 * w, y / 1024 * h, x2 / 1024 * w, y2 / 1024 * h]
                     for x, y, x2, y2 in raw]

        elif is_ll:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}]
            chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=chat, images=img, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, temperature=0.1, max_new_tokens=256)
            text = processor.decode(out[0], skip_special_tokens=True)
            if chat in text:
                text = text[len(chat):].strip()
            boxes = auto_scale_boxes(parse_any_boxes(text), w, h)

        elif is_ph:
            phi_prompt = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
            inputs = processor(phi_prompt, img, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
            text = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            boxes = auto_scale_boxes(parse_any_boxes(text), w, h)

        elif is_q3:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]}]
            chat = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            inputs = processor(images=img, text=chat, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, temperature=0.1, max_new_tokens=256)
            text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            boxes = scale_qwen(parse_any_boxes(text), w, h)

        elif is_th:
            text = detector.describe(img, prompt)
            boxes = auto_scale_boxes(parse_any_boxes(text), w, h)

        elif is_cm:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256)
            text = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            boxes = scale_qwen(parse_any_boxes(text), w, h)

        elif is_yolo_world or is_yoloe:
            boxes = yolo_cached_results.get(cat_name, [])

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
            boxes = parse_any_boxes(text)

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
            boxes = parse_any_boxes(text)

        return boxes

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

        # Collect categories present in this image
        cat_counts = Counter(a["category_id"] for a in anns)
        cat_ids_present = list(cat_counts.keys())
        if max_categories:
            cat_ids_present = cat_ids_present[:max_categories]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            pbar.update(1)
            continue

        ow, oh = image.size

        if is_yolo_world or is_yoloe:
            t0 = time.perf_counter()
            all_cat_names = [cat_id_to_name[cid] for cid in cat_ids_present]
            yolo_names_list = [model.names[i] for i in range(len(model.names))]
            name_to_idx = {n: i for i, n in enumerate(yolo_names_list)}
            yolo_results = model(str(img_path), conf=0.25, iou=0.45, verbose=False)
            yolo_infer_time = (time.perf_counter() - t0) / max(len(cat_ids_present), 1)
            yolo_cached_results.clear()
            if yolo_results and yolo_results[0].boxes is not None:
                boxes_xyxy = yolo_results[0].boxes.xyxy.cpu().tolist()
                boxes_cls = yolo_results[0].boxes.cls.cpu().tolist()
                for name in all_cat_names:
                    target_idx = name_to_idx.get(name, -1)
                    yolo_cached_results[name] = [
                        b for b, c in zip(boxes_xyxy, boxes_cls) if int(c) == target_idx
                    ]

        for cat_id in cat_ids_present:
            cat_name = cat_id_to_name[cat_id]
            gt_boxes = [a["bbox"] for a in anns if a["category_id"] == cat_id]
            total_gt_all += len(gt_boxes)

            passes_boxes = []

            for pass_idx in range(3):
                t0 = time.perf_counter()
                try:
                    boxes = _run_single_pass(image, img_path, ow, oh, cat_name)
                except Exception as e:
                    tqdm.write(f"  Pass {pass_idx + 1} error on {img_info['file_name']} ({cat_name}): {e}")
                    boxes = []
                elapsed = time.perf_counter() - t0
                if pass_idx == 0 and (is_yolo_world or is_yoloe):
                    elapsed += yolo_infer_time
                times.append(elapsed)
                passes_boxes.append(boxes)

            for pi, pboxes in enumerate(passes_boxes):
                total_pass_correct[pi] += _count_correct(pboxes, gt_boxes)

            b3_correct = _count_correct(
                [b for pb in passes_boxes for b in pb], gt_boxes
            )
            total_b3_correct += b3_correct

        n_images_processed += 1
        total_sec = sum(times)
        n_inferences_so_far = len(times)
        avg_ms = total_sec / max(n_inferences_so_far, 1) * 1000
        fps_now = n_inferences_so_far / max(total_sec, 1)
        b3_acc_now = total_b3_correct / max(total_gt_all, 1)
        pbar.set_postfix_str(f"avg={avg_ms:.0f}ms  fps={fps_now:.2f}  b3={b3_acc_now:.3f}")
        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    n_inferences = len(times)
    total_sec = sum(times)
    fps = n_inferences / max(total_sec, 1)
    avg_ms_per_infer = total_sec / max(n_inferences, 1) * 1000
    acc_50_b3 = total_b3_correct / max(total_gt_all, 1)
    acc_50_avg = sum(total_pass_correct) / max(total_gt_all * 3, 1)
    acc_50_pass = [total_pass_correct[pi] / max(total_gt_all, 1) for pi in range(3)]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Zero-shot Detection Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Images processed:   {n_images_processed}")
        print(f"  Inferences total:   {n_inferences}")
        print(f"  Passes per cat:     3")
        print(f"  Skipped:            {skipped}")
        print(f"  Total inference:    {total_sec:.2f}s")
        print(f"  Avg per inference:  {avg_ms_per_infer:.1f}ms")
        print(f"  Total GT objects:   {total_gt_all}")
        print(f"  B3 Correct@IoU50:   {total_b3_correct}")
        print(f"  B3 Acc@50:          {acc_50_b3:.4f}")
        print(f"  AVG Acc@50:         {acc_50_avg:.4f}")
        print(f"  Pass Acc@50:        {[f'{a:.4f}' for a in acc_50_pass]}")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    stats = {
        "model": display,
        "model_key": mn,
        "task": "zeroshot_detection",
        "dataset": "coco",
        "images": n_images_processed,
        "passes": 3,
        "skipped": skipped,
        "total_inference_time_s": round(total_sec, 3),
        "avg_inference_ms": round(avg_ms_per_infer, 2),
        "fps": round(fps, 2),
        "total_gt": total_gt_all,
        "total_correct": total_b3_correct,
        "acc@50": round(acc_50_b3, 4),
        "acc@50_b3": round(acc_50_b3, 4),
        "acc@50_avg": round(acc_50_avg, 4),
        "acc@50_pass": [round(a, 4) for a in acc_50_pass],
    }

    save_stats(stats, f"{safe_name}_zeroshot_detection")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Detection Benchmark (COCO)")
    all_choices = [m for m in list(MODEL_LOADERS) + list(MODEL_ALIASES)
                   if MODEL_ALIASES.get(m, m) in ZEROSHOT_DETECTION_MODELS
                   or m in ZEROSHOT_DETECTION_MODELS]
    parser.add_argument("--model", choices=all_choices, default="paligemma")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--max-categories", type=int, default=None,
                        help="Max categories per image (default: all)")
    parser.add_argument("--samples-file", type=str, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_zeroshot_detection(
        args.model, max_images=args.max_images,
        max_categories=args.max_categories, verbose=True,
    )
    if stats:
        print_comparison({args.model: stats}, "ZERO-SHOT DETECTION COMPARISON — COCO")


if __name__ == "__main__":
    main()
