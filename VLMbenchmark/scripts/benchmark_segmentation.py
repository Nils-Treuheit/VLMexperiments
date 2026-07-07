import argparse
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from tqdm import tqdm

from common import (
    RESULTS_DIR, COCO_DIR,
    MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY,
    parse_box_tags, scale_la, scale_florence2,
    print_comparison, save_stats,
)

SEGMENTATION_MODELS = {"locate_anything", "locate_anything_trt", "florence2"}


def decode_gt_mask(segmentation, h, w):
    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = maskUtils.frPyObjects([segmentation], h, w)[0]
        else:
            rle = segmentation
        return maskUtils.decode(rle).astype(np.uint8)
    elif isinstance(segmentation, list):
        if not segmentation or not segmentation[0]:
            return np.zeros((h, w), dtype=np.uint8)
        rles = maskUtils.frPyObjects(segmentation, h, w)
        decoded = maskUtils.decode(rles)
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        return decoded.astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def polygons_to_mask(polygons, h, w):
    if not polygons:
        return np.zeros((h, w), dtype=np.uint8)
    coco_polys = []
    for poly in polygons:
        flat = []
        for pt in poly:
            flat.append(pt[0] / 999.0 * w)
            flat.append(pt[1] / 999.0 * h)
        coco_polys.append(flat)
    try:
        rles = maskUtils.frPyObjects(coco_polys, h, w)
        decoded = maskUtils.decode(rles)
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        return decoded.astype(np.uint8)
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)


def compute_mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0


def compute_seg_metrics(pred_masks, gt_masks, iou_thresh=0.5):
    if not pred_masks and not gt_masks:
        return 0.0, 0.0, 0, 0, 0
    if not pred_masks:
        return 0.0, 0.0, 0, 0, len(gt_masks)
    if not gt_masks:
        return 0.0, 0.0, 0, len(pred_masks), 0

    ious = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            ious[i, j] = compute_mask_iou(pm, gm)

    matched_pred = set()
    matched_gt = set()
    pairs = []
    for i in range(len(pred_masks)):
        for j in range(len(gt_masks)):
            iou_val = ious[i, j]
            if iou_val >= iou_thresh:
                pairs.append((iou_val, i, j))
    pairs.sort(reverse=True)

    tp = 0
    for iou_val, i, j in pairs:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        tp += 1

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp

    pq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0

    matched_ious = [ious[i, j] for i, j in zip(matched_pred, matched_gt)]
    miou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return pq, miou, tp, fp, fn


def benchmark_segmentation(model_name, max_images=100, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")
    if mn not in SEGMENTATION_MODELS:
        raise ValueError(
            f"Model {mn!r} does not support segmentation. "
            f"Choose from: {SEGMENTATION_MODELS}"
        )

    is_la = mn in ("locate_anything", "locate_anything_trt")
    is_f2 = mn == "florence2"

    display = MODEL_DISPLAY.get(mn, mn)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Segmentation: {display}  |  Dataset: COCO")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    if is_la:
        worker = obj
    elif is_f2:
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

    times = []
    total_gt = 0
    total_det = 0
    skipped = 0

    all_pq = []
    all_miou = []
    class_stats = defaultdict(lambda: {"pq": [], "miou": []})

    pbar = tqdm(
        total=len(img_ids), unit="img", desc=f"{mn:>16}",
        bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    pbar.set_postfix_str("avg=-  fps=-  PQ=-")

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

        h, w = img_info["height"], img_info["width"]
        gt_masks = []
        gt_cat_ids = []
        for ann in anns:
            mask = decode_gt_mask(ann["segmentation"], h, w)
            gt_masks.append(mask)
            gt_cat_ids.append(ann["category_id"])

        primary_cat_id = Counter(a["category_id"] for a in anns).most_common(1)[0][0]
        primary_cat_name = cat_id_to_name[primary_cat_id]

        gt_primary_masks = [m for m, cid in zip(gt_masks, gt_cat_ids) if cid == primary_cat_id]
        total_gt += len(gt_primary_masks)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            pbar.update(1)
            continue

        ow, oh = image.size

        t0 = time.perf_counter()
        try:
            if is_la:
                prompt = primary_cat_name
                text = worker.predict(
                    image, prompt, max_new_tokens=512,
                    temperature=0.1, generation_mode="fast",
                )
                boxes = scale_la(parse_box_tags(text), ow, oh)
                pred_masks = []
                for x1, y1, x2, y2 in boxes:
                    mask = np.zeros((oh, ow), dtype=np.uint8)
                    x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
                    x2i, y2i = min(ow, int(round(x2))), min(oh, int(round(y2)))
                    if x2i > x1i and y2i > y1i:
                        mask[y1i:y2i, x1i:x2i] = 1
                    pred_masks.append(mask)

            elif is_f2:
                prompt = f"<OPEN_VOCABULARY_DETECTION>\n{primary_cat_name}"
                inputs = processor(text=prompt, images=image, return_tensors="pt")
                inputs = {
                    k: v.to(model.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
                with torch.no_grad():
                    out = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024, num_beams=3,
                    )
                text = processor.batch_decode(out, skip_special_tokens=False)[0]
                task_tag = "<OPEN_VOCABULARY_DETECTION>"
                parsed = processor.post_process_generation(
                    text, task=task_tag, image_size=(ow, oh)
                )
                result = parsed.get(task_tag, parsed)
                if isinstance(result, dict):
                    polys = result.get("polygons", [])
                    poly_labels = result.get("polygons_labels", [])
                    if not polys:
                        bboxes = result.get("bboxes", [])
                        bbox_labels = result.get("bboxes_labels", [])
                        pred_masks = []
                        for bx in bboxes:
                            x1, y1, x2, y2 = scale_florence2([bx], ow, oh)[0]
                            mask = np.zeros((oh, ow), dtype=np.uint8)
                            x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
                            x2i, y2i = min(ow, int(round(x2))), min(oh, int(round(y2)))
                            if x2i > x1i and y2i > y1i:
                                mask[y1i:y2i, x1i:x2i] = 1
                            pred_masks.append(mask)
                    else:
                        pred_masks = []
                        for p in polys:
                            mask = polygons_to_mask(p, oh, ow)
                            pred_masks.append(mask)
                else:
                    pred_masks = []

            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except Exception as e:
            tqdm.write(f"  Error on {img_info['file_name']}: {e}")
            import traceback
            tqdm.write(traceback.format_exc())
            skipped += 1
            pbar.update(1)
            continue

        total_det += len(pred_masks)

        if gt_primary_masks and pred_masks:
            pq, miou, tp, fp, fn = compute_seg_metrics(pred_masks, gt_primary_masks)
            all_pq.append(pq)
            all_miou.append(miou)
            class_stats[primary_cat_name]["pq"].append(pq)
            class_stats[primary_cat_name]["miou"].append(miou)
        elif gt_primary_masks and not pred_masks:
            all_pq.append(0.0)
            all_miou.append(0.0)
            class_stats[primary_cat_name]["pq"].append(0.0)
            class_stats[primary_cat_name]["miou"].append(0.0)

        avg_ms = sum(times) / len(times) * 1000
        fps_now = len(times) / sum(times)
        avg_pq = float(np.mean(all_pq)) if all_pq else 0.0
        pbar.set_postfix_str(f"avg={avg_ms:.0f}ms  fps={fps_now:.2f}  PQ={avg_pq:.3f}")
        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    avg_time = sum(times) / len(times)
    fps = len(times) / sum(times)
    overall_miou = float(np.mean(all_miou)) if all_miou else None
    overall_pq = float(np.mean(all_pq)) if all_pq else None

    per_class = {}
    for cname, st in class_stats.items():
        per_class[cname] = {
            "pq": round(float(np.mean(st["pq"])), 4),
            "miou": round(float(np.mean(st["miou"])), 4),
            "count": len(st["pq"]),
        }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Segmentation Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Images processed:    {len(times)}")
        print(f"  Skipped:             {skipped}")
        print(f"  Total inference:     {sum(times):.2f}s")
        print(f"  Avg per image:       {avg_time * 1000:.1f}ms")
        print(f"  FPS:                 {fps:.2f}")
        print(f"  Total GT masks:      {total_gt}")
        print(f"  Predicted masks:     {total_det}")
        print(f"  PQ:                  {overall_pq:.4f}" if overall_pq is not None else "  PQ:                  N/A")
        print(f"  mIoU:                {overall_miou:.4f}" if overall_miou is not None else "  mIoU:                N/A")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    stats = {
        "model": display,
        "model_key": mn,
        "task": "segmentation",
        "dataset": "coco",
        "images": len(times),
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "total_gt_masks": total_gt,
        "total_predicted_masks": total_det,
        "PQ": round(overall_pq, 4) if overall_pq is not None else None,
        "mIoU": round(overall_miou, 4) if overall_miou is not None else None,
        "per_class_metrics": per_class,
    }

    save_stats(stats, f"{safe_name}_segmentation")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Segmentation Benchmark (COCO)")
    all_choices = [
        m for m in list(MODEL_LOADERS) + list(MODEL_ALIASES)
        if MODEL_ALIASES.get(m, m) in SEGMENTATION_MODELS or m in SEGMENTATION_MODELS
    ]
    parser.add_argument("--model", choices=all_choices, default="florence2")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_segmentation(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "SEGMENTATION COMPARISON — COCO")


if __name__ == "__main__":
    main()
