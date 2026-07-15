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
    bleu_score, rouge_l, cider,
    print_comparison, save_stats,
)

CAPTION_MODELS = {"florence2", "paligemma", "llama_vision", "phi_vision", "cosmos_nemotron",
                  "qwen3_native", "qwen3_thinking",
                  "diffusion_gemma", "diffusion_gemma_yolo", "diffusion_gemma_yolo_pose",
                  "diffusion_gemma_yolo_obb", "diffusion_gemma_siglip2", "diffusion_gemma_moonvit",
                  "siglip2", "moonvit", "dinov3", "dinotool",
                  "llava_v16_mistral", "llava_onevision", "llava_next_video_7b",
                  "llava_next_video_34b", "phi3_vision"}


def benchmark_caption(model_name, max_images=100, verbose=True):
    mn = MODEL_ALIASES.get(model_name, model_name)
    if mn not in MODEL_LOADERS:
        raise ValueError(f"Unknown model {mn!r}. Choices: {list(MODEL_LOADERS)}")
    if mn not in CAPTION_MODELS:
        raise ValueError(f"Model {mn!r} does not support captioning. Choose from: {CAPTION_MODELS}")

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
    is_llava = mn.startswith("llava") or mn == "phi3_vision"

    display = MODEL_DISPLAY.get(mn, mn)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Image Captioning: {display}")
        print(f"{'=' * 60}")

    obj, _ = MODEL_LOADERS[mn]()
    if is_th:
        detector = obj
    elif is_q3:
        processor, model = obj
    elif is_s2:
        sys.path.insert(0, str(PROJECT_DIR / "siglip2"))
        import run as siglip2_run
        from transformers import AutoModel, AutoProcessor
        s2_model = AutoModel.from_pretrained(
            "google/siglip2-base-patch16-224",
            torch_dtype=torch.float16, device_map="cuda",
            attn_implementation="sdpa",
        ).eval()
        s2_processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    elif is_dg or is_mv or is_dv or is_dt or is_llava:
        pass  # subprocess-based; model/processor unused
    else:
        model, processor = obj

    img_ids, (img_infos, img_id_to_captions) = load_coco_captions(max_images=max_images)
    if img_ids is None:
        return None

    img_dir = COCO_DIR / "val2017"
    results_list = []
    times = []
    skipped = 0

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
                prompt = "<DETAILED_CAPTION>"
                inputs = processor(text=prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
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
                    text, task=prompt, image_size=(ow, oh)
                )
                caption = parsed.get("<DETAILED_CAPTION>", text)
            elif is_pg:
                prompt = "caption en"
                inputs = processor(image, prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=50)
                caption = processor.decode(out[0], skip_special_tokens=True)
            elif is_ll:
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in detail."}
                ]}]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, temperature=0.7, top_p=0.9, max_new_tokens=256)
                caption = processor.decode(out[0], skip_special_tokens=True)
                if prompt in caption:
                    caption = caption[len(prompt):].strip()
            elif is_ph:
                prompt = "<|user|>\n<|image_1|>\nDescribe this image in detail.<|end|>\n<|assistant|>\n"
                inputs = processor(prompt, image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=200, use_cache=False)
                caption = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            elif is_q3:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in detail."}
                ]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
                inputs = processor(images=image, text=text, padding=True, return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=256,
                                         do_sample=False, temperature=0.1)
                caption = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
            elif is_th:
                caption = detector.describe(image, "Describe this image in detail.")
            elif is_cm:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this scene in detail."}
                ]}]
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=256)
                caption = processor.decode(
                    out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
                )
            elif is_dg:
                run_py = PROJECT_DIR / "diffusion_gemma_vl" / "run.py"
                dg_args = [sys.executable, str(run_py), "--image", str(img_path),
                           "--task", "caption", "--n-predict", "256"]
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
                caption = result.stdout.strip()
            elif is_s2:
                desc = siglip2_run.describe(s2_model, s2_processor, image, top_k=8)
                text_parts = []
                obj_lines = [f"{d['label']} ({d['probability']:.1%})" for d in desc if d.get("category") == "object"]
                scene_lines = [f"{d['label']} ({d['probability']:.1%})" for d in desc if d.get("category") == "scene"]
                attr_lines = [f"{d['label']} ({d['probability']:.1%})" for d in desc if d.get("category") == "attribute"]
                if obj_lines:
                    text_parts.append("Objects detected: " + ", ".join(obj_lines[:6]) + ".")
                if scene_lines:
                    text_parts.append("Scene: " + scene_lines[0] + ".")
                if attr_lines:
                    text_parts.append("Attributes: " + ", ".join(attr_lines[:4]) + ".")
                caption = " ".join(text_parts) if text_parts else ", ".join(d["label"] for d in desc[:5])
            elif is_mv:
                run_py = PROJECT_DIR / "moonvit" / "run.py"
                result = subprocess.run(
                    [sys.executable, str(run_py), "--image", str(img_path),
                     "--task", "describe"],
                    capture_output=True, text=True, timeout=120,
                )
                try:
                    data = json.loads(result.stdout)
                    caption = data.get("description_text", result.stdout)
                except json.JSONDecodeError:
                    caption = result.stdout
            elif is_dv:
                run_py = PROJECT_DIR / "dinov3" / "run.py"
                result = subprocess.run(
                    [sys.executable, str(run_py), "--image", str(img_path),
                     "--task", "describe"],
                    capture_output=True, text=True, timeout=120,
                )
                try:
                    data = json.loads(result.stdout)
                    caption = data.get("description_text", result.stdout)
                except json.JSONDecodeError:
                    caption = result.stdout
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
                    caption = data.get("description_text", result.stdout)
                except json.JSONDecodeError:
                    caption = result.stdout
            elif is_llava:
                run_py = PROJECT_DIR / "Llava" / "run.py"
                llava_vpy = str(PROJECT_DIR / "Llava" / ".venv" / "bin" / "python")
                llava_key_map = {
                    "llava_v16_mistral": "llava-v1.6-mistral",
                    "llava_onevision": "llava-onevision",
                    "llava_next_video_7b": "llava-next-video-7b",
                    "llava_next_video_34b": "llava-next-video-34b",
                    "phi3_vision": "phi-3-vision",
                }
                lk = llava_key_map.get(mn, mn)
                llava_args = [llava_vpy, str(run_py), "--model", lk,
                              "--image", str(img_path), "--task", "caption",
                              "--max-new-tokens", "256"]
                if mn == "llava_next_video_34b":
                    llava_args += ["--quantize"]
                result = subprocess.run(llava_args, capture_output=True, text=True, timeout=1800)
                try:
                    data = json.loads(result.stdout)
                    caption = data.get("response", "")
                except (json.JSONDecodeError, KeyError):
                    caption = result.stdout[:500]

            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except Exception as e:
            tqdm.write(f"  Error on {img_info['file_name']}: {e}")
            skipped += 1
            pbar.update(1)
            continue

        gts = img_id_to_captions.get(img_id, [])
        if caption and gts:
            results_list.append({
                "image_id": img_id,
                "caption": caption,
                "references": gts,
            })

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

    all_bleu = []
    all_rouge = []
    all_cider = []
    for r in results_list:
        all_bleu.append(bleu_score(r["caption"], r["references"]))
        all_rouge.append(rouge_l(r["caption"], r["references"]))
        all_cider.append(cider(r["caption"], r["references"]))

    avg_bleu4 = sum(all_bleu) / len(all_bleu) if all_bleu else 0
    avg_rouge_l = sum(all_rouge) / len(all_rouge) if all_rouge else 0
    avg_cider = sum(all_cider) / len(all_cider) if all_cider else 0

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Captioning Results: {display}")
        print(f"{'=' * 60}")
        print(f"  Images processed:  {len(times)}")
        print(f"  Skipped:           {skipped}")
        print(f"  Total inference:   {sum(times):.2f}s")
        print(f"  Avg per image:     {avg_time * 1000:.1f}ms")
        print(f"  FPS:               {fps:.2f}")
        print(f"  BLEU-4:            {avg_bleu4:.4f}")
        print(f"  ROUGE-L:           {avg_rouge_l:.4f}")
        print(f"  CIDEr:             {avg_cider:.4f}")

    safe_name = mn.replace("/", "_").replace(" ", "_")
    stats = {
        "model": display,
        "model_key": mn,
        "task": "captioning",
        "dataset": "coco",
        "images": len(times),
        "skipped": skipped,
        "total_inference_time_s": round(sum(times), 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
        "bleu_4": round(avg_bleu4, 4),
        "rouge_l": round(avg_rouge_l, 4),
        "cider": round(avg_cider, 4),
    }

    save_stats(stats, f"{safe_name}_caption")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Image Captioning Benchmark (COCO Captions)")
    all_choices = [m for m in list(MODEL_LOADERS) + list(MODEL_ALIASES)
                   if MODEL_ALIASES.get(m, m) in CAPTION_MODELS or m in CAPTION_MODELS]
    parser.add_argument("--model", choices=all_choices, default="florence2")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--samples-file", type=str, default=None, help="Path to samples file (unused, for compatibility)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = benchmark_caption(args.model, max_images=args.max_images, verbose=True)
    if stats:
        print_comparison({args.model: stats}, "IMAGE CAPTIONING COMPARISON — COCO")


if __name__ == "__main__":
    main()
