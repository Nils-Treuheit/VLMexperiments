import argparse
import json
import random
import re
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
    print_comparison, save_stats,
)

VQA_MODELS = {"florence2", "paligemma", "llama_vision", "phi_vision", "cosmos_nemotron",
              "qwen3_native", "qwen3_thinking",
              "diffusion_gemma", "diffusion_gemma_yolo", "diffusion_gemma_yolo_pose",
              "diffusion_gemma_yolo_obb", "diffusion_gemma_siglip2", "diffusion_gemma_moonvit"}

VQA_QUESTIONS = [
    ("Is there a person in this image?", ["yes", "no"]),
    ("How many people are in this image?", ["1", "2", "3", "4", "5", "more than 5"]),
    ("Is this image taken indoors or outdoors?", ["indoors", "outdoors"]),
    ("What time of day is it?", ["day", "night", "morning", "evening"]),
    ("Does this image contain a vehicle?", ["yes", "no"]),
    ("Is there an animal in this image?", ["yes", "no"]),
    ("Is this a photograph or a painting?", ["photograph", "painting", "drawing"]),
    ("Is the image in color?", ["yes", "no"]),
    ("Is there text in this image?", ["yes", "no"]),
    ("Is this image blurry?", ["yes", "no"]),
]


VQA_GROUND_TRUTH_TEMPLATES = {
    "person": [("Is there a person in this image?", "yes")],
    "no person": [("Is there a person in this image?", "no")],
    "vehicle": [("Does this image contain a vehicle?", "yes")],
    "no vehicle": [("Does this image contain a vehicle?", "no")],
    "animal": [("Is there an animal in this image?", "yes")],
    "no animal": [("Is there an animal in this image?", "no")],
}


def extract_vlm_answer(text, choices):
    text_lower = text.lower().strip()
    for c in choices:
        cl = c.lower()
        if cl in text_lower:
            return c
    for c in choices:
        cl = c.lower()
        if re.search(rf'\b{re.escape(cl)}\b', text_lower):
            return c
    return text.split(".")[0].split("\n")[0].strip()


def benchmark_vqa(model_name, max_questions=200, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")
    if mn not in VQA_MODELS:
        raise ValueError(f"Model {mn!r} does not support VQA. Choose from: {VQA_MODELS}")

    is_f2 = mn == "florence2"
    is_pg = mn == "paligemma"
    is_ll = mn == "llama_vision"
    is_ph = mn == "phi_vision"
    is_cm = mn == "cosmos_nemotron"
    is_q3 = mn == "qwen3_native"
    is_th = mn == "qwen3_thinking"
    is_dg = mn.startswith("diffusion_gemma")

    display = MODEL_DISPLAY.get(mn, mn)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"VQA Benchmark: {display}")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    if is_th:
        detector = obj
    elif is_q3:
        processor, model = obj
    elif is_dg:
        pass  # subprocess-based; model/processor unused
    else:
        model, processor = obj

    ann_path = COCO_DIR / "annotations" / "captions_val2017.json"
    if not ann_path.exists():
        print(f"Error: COCO captions not found at {ann_path}")
        return None

    with open(ann_path) as f:
        coco_data = json.load(f)

    img_dir = COCO_DIR / "val2017"
    img_infos = {im["id"]: im for im in coco_data["images"]}
    img_ids = sorted(img_infos.keys())
    random.shuffle(img_ids)

    questions_per_image = 2
    needed_images = (max_questions + questions_per_image - 1) // questions_per_image
    img_ids = img_ids[:needed_images]

    results_list = []
    times = []
    correct = 0
    total = 0
    skipped = 0

    pbar = tqdm(total=len(img_ids), unit="img", desc=f"{mn:>16}")
    pbar.set_postfix_str("avg=- acc=-")

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

        ow, oh = image.size
        sampled_qs = random.sample(VQA_QUESTIONS, min(questions_per_image, len(VQA_QUESTIONS)))

        for question, choices in sampled_qs:
            t0 = time.perf_counter()
            try:
                if is_f2:
                    prompt = f"<VQA>\n{question}"
                    inputs = processor(text=prompt, images=image, return_tensors="pt")
                    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
                    with torch.no_grad():
                        out = model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=32, num_beams=1,
                        )
                    text = processor.batch_decode(out, skip_special_tokens=False)[0]
                    parsed = processor.post_process_generation(text, task="<VQA>", image_size=(ow, oh))
                    answer = parsed.get("<VQA>", text).strip()
                elif is_pg:
                    prompt = f"answer en {question}"
                    inputs = processor(image, prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=20)
                    answer = processor.decode(out[0], skip_special_tokens=True).strip().lower()
                elif is_ll:
                    messages = [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Answer the following question with one word: {question}"}
                    ]}]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, temperature=0.1, max_new_tokens=20)
                    answer = processor.decode(out[0], skip_special_tokens=True)
                    if prompt in answer:
                        answer = answer[len(prompt):].strip()
                elif is_ph:
                    prompt = f"<|user|>\n<|image_1|>\nAnswer briefly: {question}<|end|>\n<|assistant|>\n"
                    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=32, temperature=0.1, use_cache=False)
                    answer = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                elif is_q3:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"Answer in one word: {question}"}
                    ]}]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                    inputs = processor(images=image, text=text, padding=True, return_tensors="pt")
                    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    with torch.no_grad():
                        out = model.generate(**inputs, temperature=0.1, max_new_tokens=20)
                    answer = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True).strip()
                elif is_th:
                    answer = detector.describe(
                        image, f"Answer in one word: {question}"
                    ).strip()
                elif is_cm:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"Answer in one word: {question}"}
                    ]}]
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors="pt",
                    ).to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=20)
                    answer = processor.decode(
                        out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
                    )
                elif is_dg:
                    run_py = PROJECT_DIR / "diffusion_gemma_vl" / "run.py"
                    dg_args = [sys.executable, str(run_py), "--image", str(img_path),
                               "--task", "vqa", "--prompt", question, "--n-predict", "64"]
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
                    answer = result.stdout.strip()

                elapsed = time.perf_counter() - t0
                times.append(elapsed)
            except Exception as e:
                tqdm.write(f"  Error on {img_info['file_name']}: {e}")
                skipped += 1
                continue

            predicted = extract_vlm_answer(answer, choices)
            if predicted in choices:
                is_correct = predicted == choices[0]
            else:
                is_correct = False

            if is_correct:
                correct += 1
            total += 1
            results_list.append({
                "image_id": img_id,
                "question": question,
                "expected": choices[0],
                "answer": answer.strip(),
                "predicted": predicted,
                "correct": is_correct,
            })

        avg_ms = sum(times) / len(times) * 1000
        acc = correct / total if total > 0 else 0
        pbar.set_postfix_str(f"avg={avg_ms:.0f}ms acc={acc:.3f}")
        pbar.update(1)

    pbar.close()

    if not times:
        print("  No images processed!")
        return None

    avg_time = sum(times) / len(times)
    fps = len(times) / sum(times)
    accuracy = correct / total if total > 0 else 0

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"VQA Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Questions answered:  {total}")
        print(f"  Images skipped:      {skipped}")
        print(f"  Total inference:     {sum(times):.2f}s")
        print(f"  Avg per question:    {avg_time * 1000:.1f}ms")
        print(f"  FPS:                 {fps:.2f}")
        print(f"  Accuracy:            {accuracy:.4f} ({correct}/{total})")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    stats = {
        "model": display,
        "model_key": mn,
        "task": "vqa",
        "dataset": "coco",
        "questions": total,
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "accuracy": round(accuracy, 4),
    }

    save_stats(stats, f"{safe_name}_vqa")
    return stats


def main():
    parser = argparse.ArgumentParser(description="VQA Benchmark (COCO + templated questions)")
    all_choices = [m for m in list(MODEL_LOADERS) + list(MODEL_ALIASES)
                   if MODEL_ALIASES.get(m, m) in VQA_MODELS or m in VQA_MODELS]
    parser.add_argument("--model", choices=all_choices, default="florence2")
    parser.add_argument("--max-questions", type=int, default=200,
                        help="Total number of VQA questions to evaluate")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_vqa(args.model, max_questions=args.max_questions, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "VQA COMPARISON — COCO")


if __name__ == "__main__":
    main()
