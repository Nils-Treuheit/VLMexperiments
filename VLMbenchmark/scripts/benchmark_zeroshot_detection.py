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
    scale_la, scale_qwen, scale_thinking, scale_florence2,
    COCO_CAT_NAME_TO_ID,
    save_stats, print_comparison,
)

ZEROSHOT_DETECTION_MODELS = {
    "florence2", "paligemma", "llama_vision", "phi_vision", "cosmos_nemotron",
    "qwen3_native", "qwen3_thinking",
    "diffusion_gemma", "diffusion_gemma_yolo",
    "diffusion_gemma_yolo_pose", "diffusion_gemma_yolo_obb",
    "diffusion_gemma_siglip2", "diffusion_gemma_moonvit",
    "llava_v16_mistral", "llava_onevision",
    "llava_next_video_7b", "llava_next_video_34b", "phi3_vision",
}

LLAVA_VENV_PY = PROJECT_DIR / "Llava" / ".venv" / "bin" / "python"


def zeroshot_prompt(category_name):
    return (
        f"Detect all instances of '{category_name}' in this image. "
        f"Output a JSON list of objects, each with 'bbox_2d' (4 numbers: x1,y1,x2,y2) and 'label' (the category name). "
        f"Only include '{category_name}' objects. If none exist, return an empty list."
    )


def parse_any_boxes(text):
    boxes = parse_json_detections(text)
    if not boxes:
        boxes = parse_box_tags(text)
    if not boxes:
        boxes = extract_narrative_boxes(text)
    return boxes


def benchmark_zeroshot_detection(model_name, max_images=100, verbose=True):
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

    results = []
    times = []
    total_gt = 0
    total_correct = 0
    skipped = 0

    pbar = tqdm(total=len(img_ids), unit="img", desc=f"{mn:>16}",
                bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    pbar.set_postfix_str("avg=-  fps=-  acc=-")

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

        primary_cat_id = Counter(a["category_id"] for a in anns).most_common(1)[0][0]
        primary_cat_name = cat_id_to_name[primary_cat_id]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            pbar.update(1)
            continue

        ow, oh = image.size
        prompt = zeroshot_prompt(primary_cat_name)

        t0 = time.perf_counter()
        try:
            if is_f2:
                f2_prompt = "<OD>"
                inputs = processor(text=f2_prompt, images=image, return_tensors="pt")
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
                parsed = processor.post_process_generation(text, task="<OD>", image_size=(ow, oh))
                raw_boxes = parsed.get("bboxes", [])
                labels = parsed.get("labels", [])
                boxes = []
                for rb, lbl in zip(raw_boxes, labels):
                    if lbl.lower().strip() == primary_cat_name.lower().strip():
                        bx = scale_florence2([rb], ow, oh)[0]
                        boxes.append(bx)
            elif is_pg:
                pgi_prompt = f"detect {primary_cat_name}"
                inputs = processor(image, pgi_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=128)
                text = processor.decode(out[0], skip_special_tokens=True)
                boxes = parse_any_boxes(text)
            elif is_ll:
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}]
                chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = processor(text=chat, images=image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, temperature=0.1, max_new_tokens=256)
                text = processor.decode(out[0], skip_special_tokens=True)
                if chat in text:
                    text = text[len(chat):].strip()
                boxes = parse_any_boxes(text)
            elif is_ph:
                phi_prompt = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
                inputs = processor(phi_prompt, image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, use_cache=False)
                text = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                boxes = parse_any_boxes(text)
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
                text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                boxes = scale_qwen(parse_any_boxes(text), ow, oh)
            elif is_th:
                text = detector.describe(image, prompt)
                boxes = scale_thinking(parse_any_boxes(text), ow, oh)
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
                text = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                boxes = scale_qwen(parse_any_boxes(text), ow, oh)
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
            else:
                boxes = []

            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except Exception as e:
            tqdm.write(f"  Error on {img_info['file_name']}: {e}")
            skipped += 1
            pbar.update(1)
            continue

        gt_boxes = [a["bbox"] for a in anns if a["category_id"] == primary_cat_id]
        total_gt += len(gt_boxes)

        if boxes and gt_boxes:
            for b in boxes:
                bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
                for gtb in gt_boxes:
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
                            total_correct += 1
                            break

        avg_ms = sum(times) / len(times) * 1000
        fps_now = len(times) / sum(times)
        acc_now = total_correct / max(total_gt, 1)
        pbar.set_postfix_str(f"avg={avg_ms:.0f}ms  fps={fps_now:.2f}  acc={acc_now:.3f}")
        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    avg_time = sum(times) / len(times)
    fps = len(times) / sum(times)
    acc_50 = total_correct / max(total_gt, 1)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Zero-shot Detection Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Images processed:  {len(times)}")
        print(f"  Skipped:           {skipped}")
        print(f"  Total inference:   {sum(times):.2f}s")
        print(f"  Avg per image:     {avg_time * 1000:.1f}ms")
        print(f"  FPS:               {fps:.2f}")
        print(f"  Total GT objects:  {total_gt}")
        print(f"  Correct@IoU50:     {total_correct}")
        print(f"  Acc@50:            {acc_50:.4f}")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    stats = {
        "model": display,
        "model_key": mn,
        "task": "zeroshot_detection",
        "dataset": "coco",
        "images": len(times),
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "total_gt": total_gt,
        "total_correct": total_correct,
        "acc@50": round(acc_50, 4),
    }

    save_stats(stats, f"{safe_name}_zeroshot_detection")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Detection Benchmark (COCO)")
    all_choices = [m for m in list(MODEL_LOADERS) + list(MODEL_ALIASES)
                   if MODEL_ALIASES.get(m, m) in ZEROSHOT_DETECTION_MODELS
                   or m in ZEROSHOT_DETECTION_MODELS]
    parser.add_argument("--model", choices=all_choices, default="qwen3_native")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--samples-file", type=str, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_zeroshot_detection(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "ZERO-SHOT DETECTION COMPARISON — COCO")


if __name__ == "__main__":
    main()
