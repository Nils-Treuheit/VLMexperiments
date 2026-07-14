import argparse
import json
import random
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

DOC_MODELS = {"florence2", "paligemma", "llama_vision", "phi_vision", "cosmos_nemotron",
              "qwen3_native", "qwen3_thinking",
              "diffusion_gemma",
              "llava_v16_mistral", "llava_onevision", "llava_next_video_7b",
              "llava_next_video_34b", "phi3_vision"}

DOC_QUESTIONS = [
    ("Does this image appear to be a document, form, or printed page?", ["yes", "no"]),
    ("Is there handwritten content in this image?", ["yes", "no"]),
    ("Does the image contain a table, chart, or structured data?", ["yes", "no"]),
    ("Is there printed text visible in this image?", ["yes", "no"]),
    ("Does this image contain a list or bullet points?", ["yes", "no"]),
    ("Is this a photograph of a natural scene rather than a document?", ["no", "yes"]),
    ("Does the image contain labels or annotations?", ["yes", "no"]),
    ("Is there a title or heading visible?", ["yes", "no"]),
    ("Does this image contain a form with fields to fill in?", ["yes", "no"]),
    ("Does the image show a receipt, invoice, or bill?", ["yes", "no"]),
]


def extract_doc_answer(text, choices):
    text_lower = text.lower().strip()
    for c in choices:
        cl = c.lower()
        if re.search(rf'\b{re.escape(cl)}\b', text_lower):
            return c
    return text.split(".")[0].split("\n")[0].strip()


def benchmark_doc_understanding(model_name, max_questions=200, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}")
    if mn not in DOC_MODELS:
        raise ValueError(f"Model {mn!r} does not support document understanding. Choose from: {DOC_MODELS}")

    is_f2 = mn == "florence2"
    is_pg = mn == "paligemma"
    is_ll = mn == "llama_vision"
    is_ph = mn == "phi_vision"
    is_cm = mn == "cosmos_nemotron"
    is_q3 = mn == "qwen3_native"
    is_th = mn == "qwen3_thinking"
    is_llava = mn.startswith("llava") or mn == "phi3_vision"
    is_dg = mn.startswith("diffusion_gemma")

    display = MODEL_DISPLAY.get(mn, mn)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Document Understanding: {display}")
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

    ann_path = COCO_DIR / "annotations" / "captions_val2017.json"
    if not ann_path.exists():
        print(f"Error: COCO captions not found at {ann_path}")
        return None
    with open(ann_path) as f:
        coco_data = json.load(f)

    img_dir = COCO_DIR / "val2017"
    img_infos = {im["id"]: im for im in coco_data["images"]}
    rng = random.Random(42)
    img_ids = sorted(img_infos.keys())
    rng.shuffle(img_ids)

    q_per_img = 2
    needed = (max_questions + q_per_img - 1) // q_per_img
    img_ids = img_ids[:needed]

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
        sampled_qs = rng.sample(DOC_QUESTIONS, min(q_per_img, len(DOC_QUESTIONS)))

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
                        {"type": "text", "text": f"Answer the question: {question}"}
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
                    answer = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                elif is_th:
                    answer = detector.describe(image, f"Answer in one word: {question}").strip()
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
                    answer = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                elif is_dg:
                    run_py = PROJECT_DIR / "diffusion_gemma_vl" / "run.py"
                    dg_args = [sys.executable, str(run_py), "--image", str(img_path),
                               "--task", "vqa", "--prompt", f"Answer in one word: {question}",
                               "--n-predict", "64"]
                    if "siglip2" in mn:
                        dg_args += ["--encoder", "siglip2"]
                    elif "moonvit" in mn:
                        dg_args += ["--encoder", "moonvit"]
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
                                  "--prompt", f"Answer in one word: {question}",
                                  "--max-new-tokens", "20"]
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
                tqdm.write(f"  Error on {img_info['file_name']}: {e}")
                skipped += 1
                continue

            predicted = extract_doc_answer(answer, choices)
            is_correct = predicted in choices and predicted == choices[0]
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

        avg_ms = sum(times) / len(times) * 1000 if times else 0
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
        print(f"Document Understanding Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Questions answered:  {total}")
        print(f"  Skipped:             {skipped}")
        print(f"  FPS:                 {fps:.2f}")
        print(f"  Accuracy:            {accuracy:.4f} ({correct}/{total})")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    stats = {
        "model": display,
        "model_key": mn,
        "task": "doc_understanding",
        "dataset": "coco",
        "questions": total,
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "accuracy": round(accuracy, 4),
    }
    save_stats(stats, f"{safe_name}_doc_understanding")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Document Understanding Benchmark (COCO)")
    all_choices = [m for m in list(MODEL_LOADERS) + list(MODEL_ALIASES)
                   if MODEL_ALIASES.get(m, m) in DOC_MODELS or m in DOC_MODELS]
    parser.add_argument("--model", choices=all_choices, default="florence2")
    parser.add_argument("--max-questions", type=int, default=200)
    parser.add_argument("--samples-file", type=str, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_doc_understanding(args.model, max_questions=args.max_questions, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "DOCUMENT UNDERSTANDING — COCO")


if __name__ == "__main__":
    main()
