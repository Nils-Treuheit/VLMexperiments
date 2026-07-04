import argparse
import json
import time
import warnings
from collections import Counter

import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from common import (
    RESULTS_DIR, COCO_DIR,
    MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY,
    parse_box_tags, parse_json_detections, extract_narrative_boxes,
    scale_la, scale_qwen, scale_thinking, scale_florence2,
    build_prompt, print_comparison, save_stats,
)

GROUNDING_MODELS = {"locate_anything", "qwen3_native", "qwen3_thinking", "florence2"}


def benchmark_grounding(model_name, max_images=100, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")
    if mn not in GROUNDING_MODELS:
        raise ValueError(f"Model {mn!r} does not support grounding. Choose from: {GROUNDING_MODELS}")

    is_la = mn == "locate_anything"
    is_q3 = mn == "qwen3_native"
    is_th = mn == "qwen3_thinking"
    is_f2 = mn == "florence2"

    display = MODEL_DISPLAY.get(mn, mn)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Phrase Grounding: {display}")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    if is_la:
        worker = obj
    elif is_th:
        detector = obj
    elif is_f2:
        model, processor = obj
    else:
        processor, model = obj

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
    total_gt_phrases = 0
    total_correct = 0
    skipped = 0

    pbar = tqdm(
        total=len(img_ids), unit="img", desc=f"{mn:>16}",
        bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
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
        prompt = build_prompt(primary_cat_name, mn)

        t0 = time.perf_counter()
        try:
            if is_la:
                text = worker.predict(image, prompt, max_new_tokens=512,
                                      temperature=0.1, generation_mode="fast")
                boxes = scale_la(parse_box_tags(text), ow, oh)
            elif is_f2:
                f2_prompt = f"<REFERRING_EXPRESSION_SEGMENTATION>\n{primary_cat_name}"
                inputs = processor(text=f2_prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
                with torch.no_grad():
                    out = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512, num_beams=3,
                    )
                text = processor.batch_decode(out, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(
                    text, task=f2_prompt, image_size=(ow, oh)
                )
                poly_boxes = parsed.get("polygons", [])
                labels = parsed.get("labels", [])
                boxes = []
                for pb in poly_boxes:
                    xs = [p[0] for p in pb]
                    ys = [p[1] for p in pb]
                    if xs and ys:
                        x1 = min(xs) / 999 * ow
                        y1 = min(ys) / 999 * oh
                        x2 = max(xs) / 999 * ow
                        y2 = max(ys) / 999 * oh
                        boxes.append([x1, y1, x2, y2])
            elif is_q3:
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]
                text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
                inputs = processor(images=image, text=text, padding=True, return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                          for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=256,
                                         do_sample=False, temperature=0.1, top_p=0.9)
                text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
                boxes = scale_qwen(parse_box_tags(text), ow, oh)
            elif is_th:
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]
                inputs = detector.processor.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ).to(detector.model.device)
                with torch.inference_mode():
                    out = detector.model.generate(
                        **inputs, max_new_tokens=128,
                        temperature=0.1, do_sample=False,
                    )
                decoded = detector.processor.batch_decode(
                    out[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0]
                raw_boxes = parse_json_detections(decoded, target_label=primary_cat_name)
                if not raw_boxes:
                    raw_boxes = parse_json_detections(decoded)
                if not raw_boxes:
                    raw_boxes = parse_box_tags(decoded)
                if not raw_boxes:
                    raw_boxes = extract_narrative_boxes(decoded)
                boxes = scale_thinking(raw_boxes, ow, oh)

            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except Exception as e:
            tqdm.write(f"  Error on {img_info['file_name']}: {e}")
            skipped += 1
            pbar.update(1)
            continue

        gt_boxes = [a["bbox"] for a in anns if a["category_id"] == primary_cat_id]
        total_gt_phrases += len(gt_boxes)

        if boxes and gt_boxes:
            b1 = [boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]]
            for gtb in gt_boxes:
                gx1, gy1, gw, gh = gtb
                gx2, gy2 = gx1 + gw, gy1 + gh
                ix1 = max(b1[0], gx1); iy1 = max(b1[1], gy1)
                ix2 = min(b1[2], gx2); iy2 = min(b1[3], gy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area_b = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    area_gt = gw * gh
                    iou = inter / (area_b + area_gt - inter + 1e-8)
                    if iou >= 0.5:
                        total_correct += 1
                        break

        avg_ms = sum(times) / len(times) * 1000
        fps_now = len(times) / sum(times)
        acc_now = total_correct / max(total_gt_phrases, 1)
        pbar.set_postfix_str(f"avg={avg_ms:.0f}ms  fps={fps_now:.2f}  acc={acc_now:.3f}")
        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    avg_time = sum(times) / len(times)
    fps = len(times) / sum(times)
    acc_50 = total_correct / max(total_gt_phrases, 1)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Speed Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Images processed:  {len(times)}")
        print(f"  Skipped:           {skipped}")
        print(f"  Total inference:   {sum(times):.2f}s")
        print(f"  Avg per image:     {avg_time * 1000:.1f}ms")
        print(f"  FPS:               {fps:.2f}")
        print(f"  Total GT phrases:  {total_gt_phrases}")
        print(f"  Correct@IoU50:     {total_correct}")
        print(f"  Acc@50:            {acc_50:.4f}")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    stats = {
        "model": display,
        "model_key": mn,
        "task": "grounding",
        "dataset": "coco",
        "images": len(times),
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "total_gt": total_gt_phrases,
        "total_correct": total_correct,
        "acc@50": round(acc_50, 4),
    }

    save_stats(stats, f"{safe_name}_grounding")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Phrase Grounding Benchmark (COCO)")
    all_choices = [m for m in list(MODEL_LOADERS) + list(MODEL_ALIASES)
                   if MODEL_ALIASES.get(m, m) in GROUNDING_MODELS or m in GROUNDING_MODELS]
    parser.add_argument("--model", choices=all_choices, default="locate_anything")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_grounding(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "PHRASE GROUNDING COMPARISON — COCO")


if __name__ == "__main__":
    main()
