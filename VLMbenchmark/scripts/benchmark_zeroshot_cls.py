#!/usr/bin/env python3
"""Zero-shot image classification benchmark on Tiny ImageNet (200 classes).

Supports both vision-encoder models (via text-image similarity) and VLMs
(via text-prompt classification).
"""

import argparse
import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_DIR, RESULTS_DIR, PROJECT_DIR, save_stats, print_comparison

TINY_IMAGENET_DIR = Path("/mnt/HDD1/Project_Data/public_datasets/tiny-imagenet-200")

# Vision encoder models (subprocess to their venvs)
VISION_ENCODER_MODELS = {"dinotool", "dinov3", "siglip2", "moonvit"}

# VLM models (loaded via common.py MODEL_LOADERS)
VLM_MODELS = {
    "florence2", "paligemma", "llama_vision", "phi_vision", "phi4_multimodal",
    "cosmos_nemotron", "qwen3_native", "qwen3_thinking",
    "llava_v16_mistral", "llava_onevision",
    "llava_next_video_7b", "llava_next_video_34b", "phi3_vision",
}

ALL_CLS_MODELS = VISION_ENCODER_MODELS | VLM_MODELS

MODEL_VENV = {
    "dinotool": PROJECT_DIR / "DINOtool" / ".venv" / "bin" / "python",
    "dinov3": PROJECT_DIR / "dinov3" / ".venv" / "bin" / "python",
    "siglip2": PROJECT_DIR / "siglip2" / ".venv" / "bin" / "python",
    "moonvit": PROJECT_DIR / "moonvit" / ".venv" / "bin" / "python",
}


def load_tiny_imagenet_labels():
    with open(TINY_IMAGENET_DIR / "wnids.txt") as f:
        allowed = set(line.strip() for line in f)
    id2name = {}
    with open(TINY_IMAGENET_DIR / "words.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] in allowed:
                id2name[parts[0]] = parts[1].split(",")[0].strip()
    sorted_ids = sorted(id2name.keys())
    all_labels = [id2name[cid] for cid in sorted_ids]
    gt_map = {}
    with open(TINY_IMAGENET_DIR / "val" / "val_annotations.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                gt_map[parts[0]] = id2name.get(parts[1], parts[1])
    return all_labels, gt_map


def run_vision_encoder(model_key, all_labels, gt_map, max_images=50):
    """Run classification for vision encoder models via subprocess."""
    val_dir = TINY_IMAGENET_DIR / "val" / "images"
    all_files = sorted([p.name for p in val_dir.glob("*.JPEG")])
    if max_images:
        all_files = all_files[:max_images]

    labels_json = json.dumps(all_labels)
    paths_json = json.dumps(all_files)
    gt_json = json.dumps(gt_map)
    prompts_json = json.dumps([f"This is a photo of {l}." for l in all_labels])

    script = f"""
import json, sys, time, warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from pathlib import Path
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
val_dir = Path('{TINY_IMAGENET_DIR}/val/images')
all_labels = {labels_json}
image_files = {paths_json}
gt_map = {gt_json}
prompts = {prompts_json}
"""

    if model_key == "siglip2":
        script += """
from transformers import AutoModel, AutoProcessor, AutoTokenizer
model = AutoModel.from_pretrained('google/siglip2-base-patch16-224', torch_dtype=dtype,
                                   device_map=device, attn_implementation='sdpa').eval()
processor = AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')
tokenizer = AutoTokenizer.from_pretrained('google/siglip2-base-patch16-224')

batch_size = 32
text_embs_list = []
for i in range(0, len(prompts), batch_size):
    bt = prompts[i:i+batch_size]
    inp = tokenizer(bt, padding='max_length', max_length=64, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.text_model(**inp)
    emb = out.pooler_output if (hasattr(out, 'pooler_output') and out.pooler_output is not None) else out.last_hidden_state.mean(dim=1)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    text_embs_list.append(emb)
text_embs = torch.cat(text_embs_list, dim=0)

results = []
times = []
for fname in image_files:
    t0 = time.time()
    img = Image.open(val_dir / fname).convert('RGB')
    inp = processor(images=img, return_tensors='pt')
    pixel_values = inp['pixel_values'].to(device=device, dtype=dtype)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=pixel_values)
    img_feat = vision_out.pooler_output if (hasattr(vision_out, 'pooler_output') and vision_out.pooler_output is not None) else vision_out.last_hidden_state[:, 0, :]
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    logits = (img_feat @ text_embs.T)
    probs = torch.sigmoid(logits)
    top_probs, top_idx = probs[0].topk(5)
    preds = [{'label': all_labels[i], 'probability': round(float(p), 4)}
             for p, i in zip(top_probs.tolist(), top_idx.tolist())]
    elapsed = time.time() - t0
    times.append(elapsed)
    results.append({'file': fname, 'predictions': preds, 'gt': gt_map.get(fname, '')})
print(json.dumps({'results': results, 'total_time': round(sum(times), 3)}))
"""
    elif model_key in ("dinov3", "moonvit"):
        if model_key == "dinov3":
            embedding_code = """
    img_emb = (out.pooler_output if (hasattr(out, 'pooler_output') and out.pooler_output is not None)
               else out.last_hidden_state[:, 0, :])"""
            model_code = """
vis_model = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m',
                                       torch_dtype=dtype, attn_implementation='sdpa').eval().to(device)
processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
text_enc = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2',
                                      torch_dtype=dtype, trust_remote_code=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"""
        else:
            embedding_code = """
    features = out
    pooled = [f.mean(dim=0) for f in features]
    img_emb = torch.stack(pooled).mean(dim=0)
    if img_emb.dim() == 2:
        img_emb = img_emb.mean(dim=0)"""
            model_code = """
from transformers.modeling_utils import PreTrainedModel
orig_move = PreTrainedModel._move_missing_keys_from_meta_to_device
def _patched_move(self, *args, **kwargs):
    if not hasattr(self, 'all_tied_weights_keys') or not isinstance(self.all_tied_weights_keys, dict):
        object.__setattr__(self, 'all_tied_weights_keys', {})
    return orig_move(self, *args, **kwargs)
PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move

vis_model = AutoModel.from_pretrained('moonshotai/MoonViT-SO-400M',
                                       trust_remote_code=True, low_cpu_mem_usage=False).eval().to(device)
processor = AutoImageProcessor.from_pretrained('moonshotai/MoonViT-SO-400M', trust_remote_code=True)
text_enc = AutoModel.from_pretrained('google/siglip2-so400m-patch14-384',
                                      torch_dtype=dtype).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained('google/siglip2-so400m-patch14-384')"""
        script += f"""
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
{model_code}
batch_size = 64
all_text_embs = []
for i in range(0, len(prompts), batch_size):
    bt = prompts[i:i+batch_size]
    inp = tokenizer(bt, padding='max_length', max_length=64, return_tensors='pt', truncation=True).to(device)
    with torch.no_grad():
        if hasattr(text_enc, 'text_model'):
            out = text_enc.text_model(**inp)
        else:
            out = text_enc(**inp)
    emb = (out.pooler_output if (hasattr(out, 'pooler_output') and out.pooler_output is not None)
           else out.last_hidden_state.mean(dim=1))
    emb = F.normalize(emb, dim=-1)
    all_text_embs.append(emb)
text_embs = torch.cat(all_text_embs, dim=0)
results = []
times = []
for fname in image_files:
    t0 = time.time()
    img = Image.open(val_dir / fname).convert('RGB')
    inp = processor(images=img, return_tensors='pt').to(device=device)
    pixel_values = inp.pixel_values.to(dtype=next(vis_model.parameters()).dtype) if hasattr(inp, 'image_grid_hws') else inp.pixel_values.to(device=device, dtype=dtype)
    image_grid_hws = inp.image_grid_hws if hasattr(inp, 'image_grid_hws') else None
    with torch.no_grad():
        out = vis_model(pixel_values, image_grid_hws) if image_grid_hws is not None else vis_model(pixel_values=pixel_values)
    {embedding_code}
    img_emb = F.normalize(img_emb, dim=-1)
    if img_emb.dim() == 1:
        img_emb = img_emb.unsqueeze(0)
    img_emb = img_emb.to(dtype=text_embs.dtype)
    sims = (img_emb @ text_embs.T).squeeze(0)
    top_sims, top_idx = sims.topk(5)
    preds = [{{'label': all_labels[i], 'similarity': round(float(s), 4)}}
             for s, i in zip(top_sims.tolist(), top_idx.tolist())]
    elapsed = time.time() - t0
    times.append(elapsed)
    results.append({{'file': fname, 'predictions': preds, 'gt': gt_map.get(fname, '')}})
print(json.dumps({{'results': results, 'total_time': round(sum(times), 3)}}))
"""
    elif model_key == "dinotool":
        script += """
sys.path.insert(0, '/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/DINOtool')
from dinotool_wrapper import DINoToolWorker
worker = DINoToolWorker(model_name='dinov2_vits14_reg', device=None,
                        label_overrides=all_labels, prompt_template='This is a photo of {label}.')
results = []
times = []
for fname in image_files:
    t0 = time.time()
    desc_text, preds = worker.describe(str(val_dir / fname), top_k=5)
    elapsed = time.time() - t0
    times.append(elapsed)
    results.append({'file': fname, 'predictions': preds, 'gt': gt_map.get(fname, '')})
print(json.dumps({'results': results, 'total_time': round(sum(times), 3)}))
"""
    return script


EMBEDDINGS_FILE = RESULTS_DIR / "tiny_imagenet_label_embeddings.npz"


def load_label_embeddings():
    """Load pre-computed Tiny ImageNet label embeddings (MiniLM-L6-v2)."""
    data = np.load(str(EMBEDDINGS_FILE), allow_pickle=True)
    return list(data["labels"]), data["embeddings"]  # labels list, (200, 384) array


def embed_texts(texts, tokenizer, model, device, batch_size=32):
    """Embed a list of texts using MiniLM-L6-v2."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb)
    return torch.cat(all_embs, dim=0)


def run_vlm_classification(model_key, all_labels, gt_map, max_images=50):
    """Run classification for VLM models: VLM describes image → MiniLM embeds → cosine match."""
    from common import MODEL_LOADERS, MODEL_ALIASES

    mn = MODEL_ALIASES.get(model_key, model_key)
    loader = MODEL_LOADERS.get(mn)
    if not loader:
        return None

    is_q3 = mn == "qwen3_native"
    is_th = mn == "qwen3_thinking"
    is_f2 = mn == "florence2"
    is_pg = mn == "paligemma"
    is_p4 = mn == "phi4_multimodal"
    is_ll = mn == "llama_vision"
    is_ph = mn == "phi_vision"
    is_cm = mn == "cosmos_nemotron"
    is_llava = mn.startswith("llava") or mn == "phi3_vision"

    if is_th:
        detector = loader()[0]
    elif is_q3:
        processor, model = loader()[0]
    elif is_p4:
        model, processor = loader()[0]
    elif is_llava:
        return None
    else:
        model, processor = loader()[0]

    import torch
    from PIL import Image

    # Load pre-computed label embeddings
    label_names, label_embs_np = load_label_embeddings()
    label_embs = torch.tensor(label_embs_np).to(model.device)

    # Load MiniLM text encoder for embedding VLM descriptions
    print("  Loading MiniLM text encoder for semantic matching...")
    from transformers import AutoModel as _AM, AutoTokenizer as _AT
    mm_tokenizer = _AT.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    mm_model = _AM.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(model.device).eval()

    val_dir = TINY_IMAGENET_DIR / "val" / "images"
    all_files = sorted([p.name for p in val_dir.glob("*.JPEG")])
    if max_images:
        all_files = all_files[:max_images]

    cls_prompt = (
        "Identify the specific species, breed, or type of object shown. "
        "Be as specific as possible. Reply with just the name."
    )

    results = []
    times = []

    for fname in all_files:
        img_path = val_dir / fname
        gt_label = gt_map.get(fname, "")

        t0 = time.time()
        try:
            image = Image.open(img_path).convert("RGB")
            if is_th:
                text = detector.classify(image, all_labels)
            elif is_q3:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": cls_prompt}
                ]}]
                chat = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
                inputs = processor(images=image, text=chat, padding=True,
                                   return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                          for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=32,
                                         do_sample=False)
                text = processor.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
            elif is_p4:
                image = image.resize((224, 224), Image.BICUBIC)
                multi_prompt = (
                    "What are 5 possible things this image could show? "
                    "List them separated by commas, most likely first."
                )
                f4_prompt = f"<|user|><|image_1|>{multi_prompt}<|end|><|assistant|>"
                inputs = processor(text=f4_prompt, images=image,
                                   return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=64,
                                         do_sample=False, num_logits_to_keep=1)
                text = processor.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
            elif is_f2:
                f2_prompt = f"<CAPTION>{cls_prompt}</CAPTION>"
                inputs = processor(text=f2_prompt, images=image,
                                   return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                          for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(input_ids=inputs["input_ids"],
                                         pixel_values=inputs["pixel_values"],
                                         max_new_tokens=32, num_beams=1)
                text = processor.batch_decode(out, skip_special_tokens=True)[0]
            elif is_pg:
                pg_prompt = f"<image>Describe this image in one word:"
                inputs = processor(image, pg_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=16)
                text = processor.decode(out[0], skip_special_tokens=True)
            elif is_cm:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": cls_prompt}
                ]}]
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=32)
                text = processor.decode(out[0][inputs["input_ids"].shape[-1]:],
                                        skip_special_tokens=True)
            elif is_ll:
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": cls_prompt}
                ]}]
                chat = processor.apply_chat_template(messages,
                                                     add_generation_prompt=True,
                                                     tokenize=False)
                inputs = processor(text=chat, images=image,
                                   return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=32,
                                         do_sample=False)
                text = processor.decode(out[0], skip_special_tokens=True)
                if chat in text:
                    text = text[len(chat):].strip()
            elif is_ph:
                phi_prompt = f"<|user|>\n<|image_1|>\n{cls_prompt}<|end|>\n<|assistant|>\n"
                inputs = processor(phi_prompt, image,
                                   return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=32,
                                         do_sample=False)
                text = processor.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
            else:
                text = ""

            text = text.strip()
            # Clean VLM output: take first line, strip trailing garbage
            text = text.split('\n')[0].strip().rstrip('.').strip()
            text = re.sub(r'\s*[<{].*', '', text).strip()
            text = re.sub(r'\s*\|.*', '', text).strip()
            text = text.replace('<|im_start|>', '').strip()

            # Semantic matching: embed VLM output + all labels, find best cosine match
            query_emb = embed_texts([text], mm_tokenizer, mm_model, model.device)
            sims = (query_emb @ label_embs.T).squeeze(0)
            top5_idx = torch.topk(sims, 5).indices.cpu().numpy()
            top5_labels = [(label_names[i], round(float(sims[i]), 4)) for i in top5_idx]
            best_label = top5_labels[0][0]

        except Exception as e:
            text = f"ERROR: {e}"
            best_label = text
            top5_labels = []

        elapsed = time.time() - t0
        times.append(elapsed)
        results.append({
            "file": fname,
            "prediction": best_label,
            "raw_text": text[:200],
            "gt": gt_label,
            "top5": top5_labels[:5] if top5_labels else [],
        })

        if len(results) % 10 == 0:
            correct = sum(1 for r in results if r["prediction"] == r["gt"])
            top5_correct = sum(1 for r in results
                               if any(t[0] == r["gt"] for t in r.get("top5", [])))
            print(f"  [{len(results)}/{len(all_files)}] "
                  f"top1={correct/len(results):.1%} "
                  f"top5={top5_correct/len(results):.1%} "
                  f"avg={sum(times)/len(times)*1000:.0f}ms",
                  flush=True)

    return results, sum(times)


def compute_metrics(results):
    total = len(results)
    top1 = sum(1 for r in results if r.get("prediction", "").lower().strip()
               == r.get("gt", "").lower().strip())
    return {
        "top1_accuracy": round(top1 / total, 4) if total else 0,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Classification (Tiny ImageNet)")
    all_choices = sorted(ALL_CLS_MODELS)
    parser.add_argument("--model", choices=all_choices, required=True)
    parser.add_argument("--max-images", type=int, default=50)
    args = parser.parse_args()

    mn = args.model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_labels, gt_map = load_tiny_imagenet_labels()

    print(f"\n{'=' * 60}")
    print(f"Zero-Shot Classification: {mn}")
    print(f"{'=' * 60}")
    print(f"  Dataset:  Tiny ImageNet ({len(all_labels)} classes)")
    print(f"  Images:   {args.max_images}")

    t0 = time.time()

    if mn in VISION_ENCODER_MODELS:
        venv_py = MODEL_VENV.get(mn)
        if not venv_py or not venv_py.exists():
            print(f"  Error: venv not found for {mn}")
            sys.exit(1)
        script = run_vision_encoder(mn, all_labels, gt_map, args.max_images)

        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        result = subprocess.run(
            [str(venv_py), script_path],
            capture_output=True, text=True, timeout=600,
        )
        Path(script_path).unlink(missing_ok=True)

        if result.returncode != 0:
            print(f"  Error (code {result.returncode})")
            print(f"  {result.stderr[-2000:]}")
            sys.exit(1)

        data = json.loads(result.stdout)
        results_list = data["results"]
        total_time = data["total_time"]

        # Compute metrics
        top1_correct = 0
        top5_correct = 0
        for r in results_list:
            gt_name = r.get("gt", "").lower().strip()
            preds = r.get("predictions", [])
            top1 = (preds[0]["label"] if preds else "").lower().strip() == gt_name
            top5 = any(p.get("label", "").lower().strip() == gt_name for p in preds[:5])
            if top1:
                top1_correct += 1
            if top5:
                top5_correct += 1

        total = len(results_list)
        acc_top1 = top1_correct / total if total > 0 else 0
        acc_top5 = top5_correct / total if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0
        fps = total / total_time if total_time > 0 else 0

    elif mn in VLM_MODELS:
        results_list, total_time = run_vlm_classification(
            mn, all_labels, gt_map, args.max_images,
        )
        if results_list is None:
            print(f"  Model {mn} not supported for VLM classification")
            sys.exit(1)

        total = len(results_list)
        top1_correct = sum(1 for r in results_list
                           if r["prediction"].lower().strip() == r["gt"].lower().strip())
        top5_correct = sum(1 for r in results_list
                           if any(t[0].lower().strip() == r["gt"].lower().strip()
                                  for t in r.get("top5", [])))
        acc_top1 = top1_correct / total if total > 0 else 0
        acc_top5 = top5_correct / total if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0
        fps = total / total_time if total_time > 0 else 0
    else:
        print(f"  Unknown model type for {mn}")
        sys.exit(1)

    elapsed = time.time() - t0

    print(f"\n  Top-1 Accuracy: {acc_top1:.2%} ({top1_correct}/{total})")
    if acc_top5 > 0:
        print(f"  Top-5 Accuracy: {acc_top5:.2%}")
    print(f"  Avg per image:  {avg_time*1000:.0f}ms")
    print(f"  FPS:            {fps:.2f}")

    stats = {
        "model": mn,
        "model_key": mn,
        "task": "classification",
        "dataset": "tiny-imagenet-200",
        "images": total,
        "top1_accuracy": round(acc_top1, 4),
        "top5_accuracy": round(acc_top5, 4),
        "total_inference_time_s": round(total_time, 2),
        "avg_inference_ms": round(avg_time * 1000, 1),
        "fps": round(fps, 2),
    }

    save_stats(stats, f"{mn}_zeroshot_cls")
    return stats


if __name__ == "__main__":
    main()
