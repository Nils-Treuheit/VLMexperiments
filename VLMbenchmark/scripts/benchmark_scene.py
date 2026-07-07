#!/usr/bin/env python3
"""Semantic Scene Analysis Benchmark — COCO val2017.

Tests each VLM's ability to understand scenes: classify indoor/outdoor,
recognize objects, and describe spatial relationships.
"""

import argparse
import json
import subprocess
import sys
import time
import warnings

import torch
from PIL import Image
from tqdm import tqdm

from common import (
    RESULTS_DIR, COCO_DIR, PROJECT_DIR,
    MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY,
    load_coco_captions,
    print_comparison, save_stats,
)

SCENE_MODELS = {
    "florence2", "paligemma", "llama_vision", "phi_vision",
    "cosmos_nemotron", "qwen3_native", "qwen3_thinking",
    "diffusion_gemma", "diffusion_gemma_yolo", "diffusion_gemma_yolo_pose",
    "diffusion_gemma_yolo_obb", "diffusion_gemma_siglip2", "diffusion_gemma_moonvit",
    "siglip2", "moonvit", "dinov3", "dinotool",
}

# Keyword banks for indoor/outdoor classification
_INDOOR_KW = {
    "indoor", "kitchen", "bedroom", "bathroom", "living room", "dining room",
    "office", "inside", "interior", "room", "house", "apartment", "restaurant",
    "store", "shop", "museum", "gym", "basement", "garage", "hall", "corridor",
    "hotel", "classroom", "library", "theater", "cinema", "church",
    "warehouse", "workshop", "studio", "laboratory", "lobby",
}

_OUTDOOR_KW = {
    "outdoor", "outside", "street", "road", "park", "beach", "field", "mountain",
    "forest", "garden", "city", "sky", "nature", "landscape", "sea", "ocean",
    "lake", "river", "sidewalk", "parking", "playground", "farm", "zoo", "trail",
    "countryside", "yard", "harbor", "coast", "shore", "desert", "snow",
    "highway", "bridge", "stadium", "track", "court", "pool",
}

_INDOOR_CAT_IDS = frozenset({
    27, 28, 31, 32, 33, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86,
    87, 88, 89, 90,
})

_OUTDOOR_CAT_IDS = frozenset({
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20,
    21, 22, 23, 24, 25, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
})

SCENE_PROMPT = (
    "Analyze this scene in detail. Describe the type of location or scene. "
    "List every object you can see in the image. "
    "Describe the spatial relationships between the main objects. "
    "State whether this is an indoor or outdoor scene and what time of day it appears to be."
)

FLORENCE_PROMPT = "<DETAILED_CAPTION>"


def load_coco_instances(max_images=None):
    """Load COCO instance annotations: image_id -> (category names, category ids)."""
    ap = COCO_DIR / "annotations" / "instances_val2017.json"
    if not ap.exists():
        print(f"  [WARN] COCO instances not found at {ap}")
        return None, None
    with open(ap) as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}

    img_id_to_objects = {}
    img_id_to_cat_ids = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in img_id_to_objects:
            img_id_to_objects[iid] = set()
            img_id_to_cat_ids[iid] = set()
        img_id_to_objects[iid].add(cat_id_to_name[ann["category_id"]])
        img_id_to_cat_ids[iid].add(ann["category_id"])

    all_ids = sorted(img_id_to_objects)
    if max_images and max_images < len(all_ids):
        all_ids = all_ids[:max_images]

    return all_ids, (img_id_to_objects, img_id_to_cat_ids)


def classify_gt_scene(captions, cat_ids):
    """Determine ground-truth indoor/outdoor from captions, falling back to object categories."""
    text = " ".join(captions).lower()

    indoor_score = sum(1 for kw in _INDOOR_KW if kw in text)
    outdoor_score = sum(1 for kw in _OUTDOOR_KW if kw in text)

    if indoor_score > outdoor_score:
        return "indoor"
    if outdoor_score > indoor_score:
        return "outdoor"

    if cat_ids:
        indoor_cat_score = sum(1 for c in cat_ids if c in _INDOOR_CAT_IDS)
        outdoor_cat_score = sum(1 for c in cat_ids if c in _OUTDOOR_CAT_IDS)
        if indoor_cat_score > outdoor_cat_score:
            return "indoor"
        if outdoor_cat_score > indoor_cat_score:
            return "outdoor"

    return None


def parse_pred_scene(description):
    """Parse model description for predicted indoor/outdoor."""
    if not description:
        return None
    text = description.lower()
    indoor_score = sum(1 for kw in _INDOOR_KW if kw in text)
    outdoor_score = sum(1 for kw in _OUTDOOR_KW if kw in text)
    if indoor_score > outdoor_score:
        return "indoor"
    if outdoor_score > indoor_score:
        return "outdoor"
    return None


def compute_object_recall(description, gt_objects):
    """Fraction of ground-truth object names mentioned in the description."""
    if not gt_objects:
        return 0.0, 0
    text = description.lower()
    found = 0
    for obj in gt_objects:
        if obj.lower() in text:
            found += 1
    return found / len(gt_objects), len(gt_objects)


def benchmark_scene(model_name, max_images=100, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")
    if mn not in SCENE_MODELS:
        raise ValueError(
            f"Model {mn!r} does not support scene analysis. Choose from: {SCENE_MODELS}"
        )

    is_f2 = mn == "florence2"
    is_pg = mn == "paligemma"
    is_ll = mn == "llama_vision"
    is_ph = mn == "phi_vision"
    is_cm = mn == "cosmos_nemotron"
    is_q3 = mn == "qwen3_native"
    is_th = mn == "qwen3_thinking"
    is_dg = mn.startswith("diffusion_gemma")
    is_s2 = mn == "siglip2"
    is_mv = mn == "moonvit"
    is_dv = mn == "dinov3"
    is_dt = mn == "dinotool"

    display = MODEL_DISPLAY.get(mn, mn)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Scene Analysis: {display}")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    if is_th:
        detector = obj
    elif is_q3:
        processor, model = obj
    elif is_dg or is_s2 or is_mv or is_dv or is_dt:
        pass
    else:
        model, processor = obj

    img_ids, (img_infos, img_id_to_captions) = load_coco_captions(max_images=max_images)
    if img_ids is None:
        return None

    _, (img_id_to_objects, img_id_to_cat_ids) = load_coco_instances(max_images=max_images)
    if img_id_to_objects is None:
        img_id_to_objects = {}
        img_id_to_cat_ids = {}

    img_dir = COCO_DIR / "val2017"

    times = []
    skipped = 0

    scene_correct = 0
    scene_total = 0
    total_recalled = 0
    total_gt_objects = 0
    desc_word_counts = []

    pbar = tqdm(
        total=len(img_ids), unit="img", desc=f"{mn:>16}",
        bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    pbar.set_postfix_str("avg=-  fps=-")

    for idx, img_id in enumerate(img_ids):
        img_info = img_infos[img_id]
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
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
            if is_f2:
                inputs = processor(text=FLORENCE_PROMPT, images=image, return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                          for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
                with torch.no_grad():
                    out = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=256, num_beams=3,
                    )
                text = processor.batch_decode(out, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(
                    text, task=FLORENCE_PROMPT, image_size=(ow, oh)
                )
                description = parsed.get(FLORENCE_PROMPT, text)
            elif is_pg:
                inputs = processor(image, SCENE_PROMPT, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=128)
                description = processor.decode(out[0], skip_special_tokens=True)
            elif is_ll:
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": SCENE_PROMPT}
                ]}]
                prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, temperature=0.7, top_p=0.9, max_new_tokens=256
                    )
                description = processor.decode(out[0], skip_special_tokens=True)
                if prompt in description:
                    description = description[len(prompt):].strip()
            elif is_ph:
                prompt_text = (
                    f"<|user|>\n<|image_1|>\n{SCENE_PROMPT}<|end|>\n<|assistant|>\n"
                )
                inputs = processor(prompt_text, image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=256, use_cache=False)
                description = processor.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
            elif is_q3:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": SCENE_PROMPT}
                ]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
                inputs = processor(images=image, text=text, padding=True, return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                          for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=256,
                        do_sample=False, temperature=0.1
                    )
                description = processor.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
            elif is_th:
                description = detector.describe(image, SCENE_PROMPT)
            elif is_cm:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": SCENE_PROMPT}
                ]}]
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=256)
                description = processor.decode(
                    out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
                )
            elif is_dg:
                run_py = PROJECT_DIR / "diffusion_gemma_vl" / "run.py"
                dg_args = [
                    sys.executable, str(run_py), "--image", str(img_path),
                    "--task", "caption", "--n-predict", "256",
                ]
                if "siglip2" in mn:
                    dg_args += ["--encoder", "siglip2"]
                elif "moonvit" in mn:
                    dg_args += ["--encoder", "moonvit"]
                elif "pose" in mn and "obb" in mn:
                    dg_args += ["--yolo-tasks", "aabb,pose,obb"]
                elif "pose" in mn:
                    dg_args += ["--yolo-tasks", "aabb,pose"]
                elif "obb" in mn:
                    dg_args += ["--yolo-tasks", "aabb,pose,obb"]
                result = subprocess.run(
                    dg_args, capture_output=True, text=True, timeout=300,
                )
                description = result.stdout.strip()
            elif is_s2:
                run_py = PROJECT_DIR / "siglip2" / "run.py"
                result = subprocess.run(
                    [sys.executable, str(run_py), "--image", str(img_path),
                     "--task", "describe"],
                    capture_output=True, text=True, timeout=120,
                )
                try:
                    data = json.loads(result.stdout)
                    description = data.get("description_text", result.stdout)
                except json.JSONDecodeError:
                    description = result.stdout
            elif is_mv:
                run_py = PROJECT_DIR / "moonvit" / "run.py"
                result = subprocess.run(
                    [sys.executable, str(run_py), "--image", str(img_path),
                     "--task", "describe"],
                    capture_output=True, text=True, timeout=120,
                )
                try:
                    data = json.loads(result.stdout)
                    description = data.get("description_text", result.stdout)
                except json.JSONDecodeError:
                    description = result.stdout
            elif is_dv:
                run_py = PROJECT_DIR / "dinov3" / "run.py"
                result = subprocess.run(
                    [sys.executable, str(run_py), "--image", str(img_path),
                     "--task", "describe"],
                    capture_output=True, text=True, timeout=120,
                )
                try:
                    data = json.loads(result.stdout)
                    description = data.get("description_text", result.stdout)
                except json.JSONDecodeError:
                    description = result.stdout
            elif is_dt:
                run_py = PROJECT_DIR / "DINOtool" / "run.py"
                venv_py = str(PROJECT_DIR / "DINOtool" / ".venv" / "bin" / "python")
                result = subprocess.run(
                    [venv_py, str(run_py), "--image", str(img_path),
                     "--task", "describe", "--model", "dinov2-s"],
                    capture_output=True, text=True, timeout=120,
                )
                try:
                    data = json.loads(result.stdout)
                    description = data.get("description_text", result.stdout)
                except json.JSONDecodeError:
                    description = result.stdout

            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except Exception as e:
            tqdm.write(f"  Error on {img_info['file_name']}: {e}")
            skipped += 1
            pbar.update(1)
            continue

        if not description or not description.strip():
            skipped += 1
            pbar.update(1)
            continue

        description = description.strip()

        # Scene type accuracy (indoor/outdoor)
        captions = img_id_to_captions.get(img_id, [])
        cat_ids = img_id_to_cat_ids.get(img_id, set())
        gt_scene = classify_gt_scene(captions, cat_ids)
        pred_scene = parse_pred_scene(description)
        if gt_scene is not None and pred_scene is not None:
            scene_total += 1
            if pred_scene == gt_scene:
                scene_correct += 1

        # Object recall
        gt_objects = img_id_to_objects.get(img_id, set())
        recall, n_obj = compute_object_recall(description, gt_objects)
        total_recalled += recall * n_obj if n_obj > 0 else 0
        total_gt_objects += n_obj

        # Description quality proxy
        desc_word_counts.append(len(description.split()))

        avg_ms = sum(times) / len(times) * 1000
        fps_now = len(times) / sum(times)
        pbar.set_postfix_str(f"avg={avg_ms:.0f}ms  fps={fps_now:.2f}")
        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    avg_time = sum(times) / len(times)
    fps = len(times) / sum(times)
    avg_desc_len = sum(desc_word_counts) / len(desc_word_counts) if desc_word_counts else 0
    scene_type_acc = scene_correct / scene_total if scene_total > 0 else 0.0
    object_recall = total_recalled / total_gt_objects if total_gt_objects > 0 else 0.0

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Scene Analysis Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Images processed:  {len(times)}")
        print(f"  Skipped:           {skipped}")
        print(f"  FPS:               {fps:.2f}")
        print(f"  Avg inference:     {avg_time * 1000:.1f}ms")
        print(f"  Scene type acc:    {scene_type_acc:.4f}  ({scene_correct}/{scene_total})")
        print(f"  Object recall:     {object_recall:.4f}  ({total_recalled:.0f}/{total_gt_objects})")
        print(f"  Avg desc length:   {avg_desc_len:.1f} words")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    stats = {
        "model": display,
        "model_key": mn,
        "task": "scene_analysis",
        "dataset": "COCO",
        "images": len(times),
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "scene_type_accuracy": round(scene_type_acc, 4),
        "scene_type_correct": scene_correct,
        "scene_type_total": scene_total,
        "object_recall": round(object_recall, 4),
        "objects_recalled": round(total_recalled),
        "objects_total": total_gt_objects,
        "description_quality": round(avg_desc_len, 1),
    }

    save_stats(stats, f"{safe_name}_scene")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Scene Analysis Benchmark (COCO val2017)"
    )
    all_choices = sorted({
        m for m in list(MODEL_LOADERS) + list(MODEL_ALIASES)
        if MODEL_ALIASES.get(m, m) in SCENE_MODELS or m in SCENE_MODELS
    })
    parser.add_argument("--model", choices=all_choices, default="florence2")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_scene(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "SCENE ANALYSIS COMPARISON — COCO")


if __name__ == "__main__":
    main()
