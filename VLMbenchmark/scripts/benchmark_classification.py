#!/usr/bin/env python3
"""
Zero-shot classification benchmark using vision encoder models.
Evaluates on Tiny ImageNet (200 classes).

Runs all images in a single model-load + text-pre-encode cycle.
Each model has an inner script that handles loading and inference.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_DIR, RESULTS_DIR, PROJECT_DIR, save_stats

TINY_IMAGENET_DIR = Path("/mnt/HDD1/Project_Data/public_datasets/tiny-imagenet-200")
CLASSIFICATION_MODELS = {"dinotool", "dinov3", "siglip2", "moonvit"}

MODEL_VENV = {
    "dinotool": PROJECT_DIR / "DINOtool" / ".venv" / "bin" / "python",
    "dinov3": PROJECT_DIR / "dinov3" / ".venv" / "bin" / "python",
    "siglip2": PROJECT_DIR / "siglip2" / ".venv" / "bin" / "python",
    "moonvit": PROJECT_DIR / "moonvit" / ".venv" / "bin" / "python",
}


def make_inline_script(model_key, image_paths, max_images):
    """Build a single inline Python script that loads the model once and classifies all images."""

    # Get labels
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
    # Ground truth map: filename -> class_name
    gt_map = {}
    with open(TINY_IMAGENET_DIR / "val" / "val_annotations.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                gt_map[parts[0]] = id2name.get(parts[1], parts[1])

    # Only process files that exist
    val_dir = TINY_IMAGENET_DIR / "val" / "images"
    valid_paths = [p for p in image_paths if (val_dir / p).exists()]
    if max_images:
        valid_paths = valid_paths[:max_images]

    labels_json = json.dumps(all_labels)
    paths_json = json.dumps(valid_paths)

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
gt_map = {json.dumps(gt_map)}
prompts = ['This is a photo of ' + l + '.' for l in all_labels]
"""

    if model_key == "siglip2":
        script += """
from transformers import AutoModel, AutoProcessor, AutoTokenizer
model = AutoModel.from_pretrained('google/siglip2-base-patch16-224', torch_dtype=dtype,
                                   device_map=device, attn_implementation='sdpa').eval()
processor = AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')
tokenizer = AutoTokenizer.from_pretrained('google/siglip2-base-patch16-224')

# Pre-encode text labels using text_model
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
temperature = model.logit_scale.exp().item() if hasattr(model, 'logit_scale') else 1.0

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
    logits = (img_feat @ text_embs.T) * temperature
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
        else:
            embedding_code = """
    features = out
    pooled = [f.mean(dim=0) for f in features]
    img_emb = torch.stack(pooled).mean(dim=0)
    if img_emb.dim() == 2:
        img_emb = img_emb.mean(dim=0)"""
        if model_key == "dinov3":
            model_code = """
vis_model = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m',
                                       torch_dtype=dtype, attn_implementation='sdpa').eval().to(device)
processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
text_enc = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2',
                                      torch_dtype=dtype, trust_remote_code=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"""
        else:
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
    inp = processor(images=img, return_tensors='pt')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="siglip2",
                        choices=sorted(CLASSIFICATION_MODELS))
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--samples-file", type=str, default=None, help="Path to samples file (unused, for compatibility)")
    args = parser.parse_args()

    # List all Tiny ImageNet val images
    val_dir = TINY_IMAGENET_DIR / "val" / "images"
    all_files = sorted([p.name for p in val_dir.glob("*.JPEG")])
    if args.max_images:
        all_files = all_files[:args.max_images]

    print(f"\nClassification Benchmark: {args.model}")
    print(f"  Images: {len(all_files)}")
    print(f"  Dataset: Tiny ImageNet (200 classes)")

    script = make_inline_script(args.model, all_files, args.max_images)
    venv_py = MODEL_VENV[args.model]

    import tempfile
    t0 = time.time()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    result = subprocess.run(
        [str(venv_py), script_path],
        capture_output=True, text=True, timeout=600,
    )
    Path(script_path).unlink(missing_ok=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  Error (code {result.returncode})")
        err = result.stderr[-2000:]
        print(f"  {err}")
        sys.exit(1)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"  JSON error: {e}")
        print(f"  stdout[:500]: {result.stdout[:500]}")
        print(f"  stderr[:500]: {result.stderr[:500]}")
        sys.exit(1)

    results_list = data.get("results", [])
    total_time = data.get("total_time", 0)

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

    print(f"  [{total}/{total}] top1={acc_top1:.2%} top5={acc_top5:.2%} "
          f"avg={avg_time*1000:.0f}ms/img total={total_time:.1f}s")

    stats = {
        "model": f"Tiny ImageNet Classification ({args.model})",
        "model_key": args.model,
        "task": "classification",
        "dataset": "tiny-imagenet-200",
        "images": total,
        "top1_accuracy": round(acc_top1, 4),
        "top5_accuracy": round(acc_top5, 4),
        "total_inference_time_s": round(total_time, 2),
        "avg_inference_ms": round(avg_time * 1000, 1),
        "fps": round(fps, 2),
    }

    save_stats(stats, f"{args.model}_classification")

    print(f"\nClassification Results: {args.model}")
    print(f"  Top-1 Accuracy: {acc_top1:.2%} ({top1_correct}/{total})")
    print(f"  Top-5 Accuracy: {acc_top5:.2%} ({top5_correct}/{total})")
    print(f"  Avg per image: {avg_time*1000:.0f}ms")
    print(f"  FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
