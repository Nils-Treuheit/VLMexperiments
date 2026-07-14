import argparse
import json
import re
import subprocess
import sys
import time

import torch
from PIL import Image
from tqdm import tqdm

from common import (
    RESULTS_DIR, COCO_DIR, PROJECT_DIR,
    MODEL_LOADERS, MODEL_ALIASES, MODEL_DISPLAY,
    print_comparison, save_stats,
)

LLAVA_VENV_PY = PROJECT_DIR / "Llava" / ".venv" / "bin" / "python"

COUNTING_MODELS = {"florence2", "paligemma", "llama_vision", "phi_vision", "cosmos_nemotron",
                   "qwen3_native", "qwen3_thinking",
                   "diffusion_gemma",
                   "llava_v16_mistral", "llava_onevision", "llava_next_video_7b",
                   "llava_next_video_34b", "phi3_vision"}

DIFFICULT_CATEGORIES = ["person", "car", "chair", "book", "bottle", "cup", "bird", "dog", "cat"]


def extract_count(text):
    nums = re.findall(r'\b(\d+)\b', text)
    for n in nums:
        val = int(n)
        if 0 <= val <= 500:
            return val
    text_lower = text.lower().strip()
    word_map = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "no": 0, "none": 0}
    for word, val in word_map.items():
        if text_lower.startswith(word) or text_lower == word:
            return val
    return -1


def benchmark_counting(model_name, max_images=100, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}")
    if mn not in COUNTING_MODELS:
        raise ValueError(f"Model {mn!r} does not support counting. Choose from: {COUNTING_MODELS}")

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
        print(f"Object Counting: {display}")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    if is_th:
        detector = obj
    elif is_q3:
        processor, model = obj
    elif is_dg or is_llava:
        pass
    else:
        model, processor = obj

    ann_path = COCO_DIR / "annotations" / "instances_val2017.json"
    if not ann_path.exists():
        print(f"Error: COCO instances not found at {ann_path}")
        return None
    with open(ann_path) as f:
        instances = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in instances["categories"]}
    img_id_to_counts = {}
    for ann in instances["annotations"]:
        iid = ann["image_id"]
        if iid not in img_id_to_counts:
            img_id_to_counts[iid] = {}
        cat_name = cat_id_to_name[ann["category_id"]]
        img_id_to_counts[iid][cat_name] = img_id_to_counts[iid].get(cat_name, 0) + 1

    img_infos = {im["id"]: im for im in instances["images"]}
    img_ids = sorted(img_id_to_counts.keys())[:max_images]
    img_dir = COCO_DIR / "val2017"

    times = []
    skipped = 0
    abs_errors = []
    correct_exact = 0
    total_questions = 0

    pbar = tqdm(total=len(img_ids), unit="img", desc=f"{mn:>16}")
    pbar.set_postfix_str("avg=- mae=-")

    for img_id in img_ids:
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

        gt_counts = img_id_to_counts.get(img_id, {})
        present = [c for c, v in gt_counts.items() if v > 0]
        if not present:
            pbar.update(1)
            continue
        present = present[:3]

        for cat in present:
            gt = gt_counts[cat]
            ow, oh = image.size
            t0 = time.perf_counter()
            try:
                if is_f2:
                    prompt = f"<VQA>\nHow many {cat} are in this image?"
                    img_input = processor(text=prompt, images=image, return_tensors="pt")
                    img_input = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in img_input.items()}
                    if "pixel_values" in img_input:
                        img_input["pixel_values"] = img_input["pixel_values"].to(dtype=model.dtype)
                    with torch.no_grad():
                        out = model.generate(
                            input_ids=img_input["input_ids"],
                            pixel_values=img_input["pixel_values"],
                            max_new_tokens=64, num_beams=1,
                        )
                    text = processor.batch_decode(out, skip_special_tokens=False)[0]
                    parsed = processor.post_process_generation(text, task="<VQA>", image_size=(ow, oh))
                    answer = parsed.get("<VQA>", text).strip()
                elif is_pg:
                    prompt = f"answer en How many {cat} are in this image?"
                    inputs = processor(image, prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=10)
                    answer = processor.decode(out[0], skip_special_tokens=True).strip()
                elif is_ll:
                    messages = [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"How many {cat} are in this image? Answer with just a number."}
                    ]}]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, temperature=0.1, max_new_tokens=10)
                    answer = processor.decode(out[0], skip_special_tokens=True)
                    if prompt in answer:
                        answer = answer[len(prompt):].strip()
                elif is_ph:
                    prompt = f"<|user|>\n<|image_1|>\nHow many {cat} are in this image? Just the number.<|end|>\n<|assistant|>\n"
                    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=10, temperature=0.1, use_cache=False)
                    answer = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                elif is_q3:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"How many {cat} are in this image? Answer with just a number."}
                    ]}]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                    inputs = processor(images=image, text=text, padding=True, return_tensors="pt")
                    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    with torch.no_grad():
                        out = model.generate(**inputs, temperature=0.1, max_new_tokens=10)
                    answer = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                elif is_th:
                    answer = detector.describe(
                        image, f"How many {cat} are in this image? Answer with just a number."
                    ).strip()
                elif is_cm:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"How many {cat} are in this image? Answer with just a number."}
                    ]}]
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors="pt",
                    ).to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=10)
                    answer = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                elif is_dg:
                    run_py = PROJECT_DIR / "diffusion_gemma_vl" / "run.py"
                    dg_args = [sys.executable, str(run_py), "--image", str(img_path),
                               "--task", "vqa", "--prompt", f"How many {cat} are in this image?",
                               "--n-predict", "10"]
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
                    result = subprocess.run(dg_args, capture_output=True, text=True, timeout=300)
                    answer = result.stdout.strip()
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
                                  "--prompt", f"How many {cat} are in this image? Just the number.",
                                  "--max-new-tokens", "10"]
                    if mn == "llava_next_video_34b":
                        llava_args += ["--quantize"]
                    result = subprocess.run(llava_args, capture_output=True, text=True, timeout=1800)
                    try:
                        data = json.loads(result.stdout)
                        answer = data.get("response", "").strip()
                    except (json.JSONDecodeError, KeyError):
                        answer = result.stdout.strip()

                elapsed = time.perf_counter() - t0
                times.append(elapsed)
            except Exception as e:
                tqdm.write(f"  Error on {img_info['file_name']} (count {cat}): {e}")
                skipped += 1
                continue

            pred = extract_count(answer)
            if pred >= 0:
                abs_errors.append(abs(pred - gt))
                if pred == gt:
                    correct_exact += 1
            total_questions += 1

        avg_ms = sum(times) / len(times) * 1000 if times else 0
        mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0
        pbar.set_postfix_str(f"avg={avg_ms:.0f}ms mae={mae:.2f}")
        pbar.update(1)

    pbar.close()
    if not times:
        print("  No images processed!")
        return None

    avg_time = sum(times) / len(times)
    fps = len(times) / sum(times)
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0
    rmse = (sum(e ** 2 for e in abs_errors) / len(abs_errors)) ** 0.5 if abs_errors else 0
    exact_acc = correct_exact / total_questions if total_questions > 0 else 0

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Object Counting Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Questions:           {total_questions}")
        print(f"  Skipped:             {skipped}")
        print(f"  FPS:                 {fps:.2f}")
        print(f"  MAE:                 {mae:.2f}")
        print(f"  RMSE:                {rmse:.2f}")
        print(f"  Exact accuracy:      {exact_acc:.4f}")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    stats = {
        "model": display,
        "model_key": mn,
        "task": "object_counting",
        "dataset": "coco",
        "questions": total_questions,
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "exact_accuracy": round(exact_acc, 4),
        "correct_exact": correct_exact,
    }
    save_stats(stats, f"{safe_name}_counting")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Object Counting Benchmark (COCO instances)")
    all_choices = [m for m in list(MODEL_LOADERS) + list(MODEL_ALIASES)
                   if MODEL_ALIASES.get(m, m) in COUNTING_MODELS or m in COUNTING_MODELS]
    parser.add_argument("--model", choices=all_choices, default="florence2")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--samples-file", type=str, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_counting(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "OBJECT COUNTING — COCO")


if __name__ == "__main__":
    main()
