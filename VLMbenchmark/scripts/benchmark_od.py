import argparse
import json
import os
import time
import warnings
from collections import Counter

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from common import (
    RESULTS_DIR, COCO_DIR, DOTA_DIR,
    MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY,
    parse_box_tags, parse_json_detections, extract_narrative_boxes,
    scale_la, scale_qwen, scale_thinking, scale_florence2,
    COCO_CAT_NAME_TO_ID,
    load_dota_coco_gt, build_prompt, print_comparison, save_stats,
)


def benchmark(model_name, dataset="coco", max_images=100, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")

    is_la = mn in ("locate_anything", "locate_anything_trt")
    is_q3 = mn == "qwen3_native"
    is_th = mn == "qwen3_thinking"
    is_yolo = mn.startswith("yolo")
    is_f2 = mn == "florence2"
    is_pg = mn == "paligemma"

    if dataset not in ("coco", "dota"):
        raise ValueError(f"Unknown dataset {dataset!r}")

    display = MODEL_DISPLAY.get(mn, mn)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Object Detection: {display}  |  Dataset: {dataset}")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    if is_la:
        worker = obj
    elif is_th:
        detector = obj
    elif is_yolo:
        model = obj
    elif is_f2:
        model, processor = obj
    elif is_pg:
        model, processor = obj
    else:
        processor, model = obj

    dota_gt_dict = None
    if dataset == "coco":
        ap = COCO_DIR / "annotations" / "instances_val2017.json"
        if not ap.exists():
            print(f"Error: COCO annotations not found at {ap}")
            return None
        coco_gt = COCO(ap)
        cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
        if is_yolo:
            coco_name_to_id = {c["name"]: c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
        img_ids = coco_gt.getImgIds()
        if max_images:
            img_ids = img_ids[:max_images]
        img_dir = COCO_DIR / "val2017"
    else:
        dota_gt_dict = load_dota_coco_gt(DOTA_DIR, max_images=max_images)
        if dota_gt_dict is None or not dota_gt_dict["images"]:
            print(f"Error: DOTA data not found at {DOTA_DIR}")
            return None
        gt_path = RESULTS_DIR / "dota_gt_temp.json"
        with open(gt_path, "w") as f:
            json.dump(dota_gt_dict, f)
        coco_gt = COCO(str(gt_path))
        cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
        img_ids = [im["id"] for im in dota_gt_dict["images"]]
        img_dir = DOTA_DIR / "images"

    results = []
    times = []
    total_gt = 0
    total_det = 0
    skipped = 0

    pbar = tqdm(
        total=len(img_ids), unit="img", desc=f"{mn:>16}",
        bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    pbar.set_postfix_str("avg=-  fps=-  det=0/0")

    for idx, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            skipped += 1
            pbar.update(1)
            continue

        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        if not anns and not is_yolo:
            skipped += 1
            pbar.update(1)
            continue

        primary_cat_id = Counter(a["category_id"] for a in anns).most_common(1)[0][0] if anns else 0
        primary_cat_name = cat_id_to_name.get(primary_cat_id, "object")

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
            elif is_f2:
                prompt = f"<OD>"
                inputs = processor(text=prompt, images=image, return_tensors="pt")
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
                parsed = processor.post_process_generation(text, task="<OD>", image_size=(ow, oh))
                raw_boxes = parsed.get("bboxes", [])
                labels = parsed.get("labels", [])
                boxes = scale_florence2(raw_boxes, ow, oh)
                for bx, lbl in zip(raw_boxes, labels):
                    x1, y1, x2, y2 = scale_florence2([bx], ow, oh)[0]
                    cid = COCO_CAT_NAME_TO_ID.get(lbl, primary_cat_id)
                    results.append({
                        "image_id": img_id,
                        "category_id": cid,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": 0.9,
                    })
            elif is_pg:
                prompt = f"detect {primary_cat_name}"
                inputs = processor(image, prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=128)
                text = processor.decode(out[0], skip_special_tokens=True)
                boxes = scale_qwen(parse_box_tags(text), ow, oh)
            elif is_yolo:
                yolo_out = model(image, verbose=False)[0]
                boxes = []
                if yolo_out.boxes is not None:
                    xyxy = yolo_out.boxes.xyxy.cpu().numpy()
                    scores = yolo_out.boxes.conf.cpu().numpy()
                    cls = yolo_out.boxes.cls.cpu().numpy().astype(int)
                    for i in range(len(cls)):
                        if scores[i] < 0.25:
                            continue
                        coco_id = coco_name_to_id.get(model.names[cls[i]])
                        if coco_id is None:
                            continue
                        x1, y1, x2, y2 = xyxy[i]
                        results.append({
                            "image_id": img_id,
                            "category_id": coco_id,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(scores[i]),
                        })
                        boxes.append([float(x1), float(y1), float(x2), float(y2)])

            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except Exception as e:
            tqdm.write(f"  Error on {img_info['file_name']}: {e}")
            skipped += 1
            pbar.update(1)
            continue

        total_gt += len(anns)
        total_det += len(boxes)

        if not is_yolo and not is_f2:
            for x1, y1, x2, y2 in boxes:
                results.append({
                    "image_id": img_id,
                    "category_id": primary_cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": 0.9,
                })

        avg_ms = sum(times) / len(times) * 1000
        fps_now = len(times) / sum(times)
        pbar.set_postfix_str(f"avg={avg_ms:.0f}ms  fps={fps_now:.2f}  det={total_det}/{total_gt}")
        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    avg_time = sum(times) / len(times)
    fps = len(times) / sum(times)

    if verbose:
        display = MODEL_DISPLAY.get(mn, mn)
        print(f"\n{'=' * 60}")
        print(f"Speed Results: {display} on {dataset}")
        print(f"{'=' * 60}")
        print(f"  Images processed:  {len(times)}")
        print(f"  Skipped:           {skipped}")
        print(f"  Total inference:   {sum(times):.2f}s")
        print(f"  Avg per image:     {avg_time * 1000:.1f}ms")
        print(f"  FPS:               {fps:.2f}")
        print(f"  Total GT objects:  {total_gt}")
        print(f"  Detected objects:  {total_det}")

    mAP_50_95 = mAP_50 = None
    if results:
        safe_name = mn.replace("/", "_").replace(" ", "_")
        rp = RESULTS_DIR / f"{safe_name}_{dataset}_results.json"
        with open(rp, "w") as f:
            json.dump(results, f)

        try:
            coco_dt = coco_gt.loadRes(results)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.params.imgIds = list(set(r["image_id"] for r in results))
            ev.evaluate()
            ev.accumulate()
            if verbose:
                ev.summarize()
            mAP_50_95 = float(ev.stats[0])
            mAP_50 = float(ev.stats[1])
            if verbose:
                print(f"\n  mAP@50:95 = {mAP_50_95:.4f}")
                print(f"  mAP@50    = {mAP_50:.4f}")
        except Exception as e:
            print(f"  mAP computation skipped: {e}")

    stats = {
        "model": display if verbose else mn,
        "model_key": mn,
        "dataset": dataset,
        "task": "object_detection",
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

    save_stats(stats, f"{safe_name}_{dataset}_od")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Object Detection Benchmark (COCO/DOTA)")
    all_choices = list(MODEL_LOADERS) + list(MODEL_ALIASES)
    parser.add_argument("--model", choices=all_choices, default="locate_anything")
    parser.add_argument("--dataset", choices=["coco", "dota"], default="coco")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark(args.model, dataset=args.dataset, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats},
                         f"OBJECT DETECTION COMPARISON — {args.dataset}")


if __name__ == "__main__":
    main()
