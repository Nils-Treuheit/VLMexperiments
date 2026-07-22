#!/usr/bin/env python3
"""Visual Embedding Quality (VEQ) benchmark.

Evaluates image embeddings on:
  1. Image Retrieval (Recall@1/5/10, mAP)
  2. k-NN Classification (Top-1, Top-5)
  3. Clustering (NMI, ARI, Silhouette)

Uses COCO val2017 for evaluation. Embeddings are extracted from vision
encoder models or VLM vision backbones.
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_DIR, RESULTS_DIR, PROJECT_DIR, COCO_DIR, save_stats

# Models that produce usable image embeddings
VEQ_MODELS = {"siglip2", "dinov3", "moonvit", "dinotool"}

MODEL_VENV = {
    "siglip2": PROJECT_DIR / "siglip2" / ".venv" / "bin" / "python",
    "dinov3": PROJECT_DIR / "dinov3" / ".venv" / "bin" / "python",
    "moonvit": PROJECT_DIR / "moonvit" / ".venv" / "bin" / "python",
    "dinotool": PROJECT_DIR / "DINOtool" / ".venv" / "bin" / "python",
}


def make_embedding_script(model_key, image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])

    if model_key == "siglip2":
        return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
model = AutoModel.from_pretrained('google/siglip2-base-patch16-224', torch_dtype=dtype,
                                   device_map=device, attn_implementation='sdpa').eval()
processor = AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')
image_paths = {paths_json}
results = []
times = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.vision_model(**inputs)
    emb = outputs.pooler_output if (hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None) else outputs.last_hidden_state.mean(dim=1)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    elapsed = time.time() - t0
    times.append(elapsed)
    buf = io.BytesIO()
    np.save(buf, emb.cpu().float().numpy())
    results.append({{'file': path, 'emb': base64.b64encode(buf.getvalue()).decode(),
                     'shape': list(emb.shape), 'time': round(elapsed, 4)}})
print(json.dumps({{'results': results, 'total_time': round(sum(times), 3),
                    'dim': results[0]['shape'][-1] if results else 0,
                    'times': [round(t, 4) for t in times]}}))
"""
    elif model_key == "dinov3":
        return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
model = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m',
                                   torch_dtype=dtype, attn_implementation='sdpa').eval().to(device)
processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
image_paths = {paths_json}
results = []
times = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to(device=device, dtype=dtype)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.pooler_output if (hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None) else outputs.last_hidden_state[:, 0, :]
    elapsed = time.time() - t0
    times.append(elapsed)
    buf = io.BytesIO()
    np.save(buf, emb.cpu().float().numpy())
    results.append({{'file': path, 'emb': base64.b64encode(buf.getvalue()).decode(),
                     'shape': list(emb.shape), 'time': round(elapsed, 4)}})
print(json.dumps({{'results': results, 'total_time': round(sum(times), 3),
                    'dim': results[0]['shape'][-1] if results else 0,
                    'times': [round(t, 4) for t in times]}}))
"""
    elif model_key == "moonvit":
        return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from transformers.modeling_utils import PreTrainedModel as _PTM
if not hasattr(_PTM, 'all_tied_weights_keys'):
    def _get_all_tied_weights_keys(self):
        if hasattr(self, '_dg_all_tied_weights_keys'):
            return self._dg_all_tied_weights_keys
        keys = self._tied_weights_keys
        if keys is None:
            return {{}}
        if isinstance(keys, dict):
            return keys
        if isinstance(keys, str):
            return {{keys: keys}}
        return {{k: k for k in keys}}
    def _set_all_tied_weights_keys(self, value):
        self._dg_all_tied_weights_keys = value if value is not None else {{}}
    _PTM.all_tied_weights_keys = property(_get_all_tied_weights_keys, _set_all_tied_weights_keys)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
model = AutoModel.from_pretrained('moonshotai/MoonViT-SO-400M', trust_remote_code=True,
                                   torch_dtype=dtype, low_cpu_mem_usage=False).eval().to(device)
processor = AutoImageProcessor.from_pretrained('moonshotai/MoonViT-SO-400M', trust_remote_code=True)
image_paths = {paths_json}
results = []
times = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    inputs = processor(img, return_tensors='pt').to(device=device, dtype=dtype)
    with torch.no_grad():
        features = model(inputs.pixel_values, inputs.image_grid_hws)
    pooled = [f.mean(dim=0) for f in features]
    emb = torch.stack(pooled).mean(dim=0)
    if emb.dim() == 2:
        emb = emb.mean(dim=0)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    elapsed = time.time() - t0
    times.append(elapsed)
    buf = io.BytesIO()
    np.save(buf, emb.cpu().float().numpy())
    results.append({{'file': path, 'emb': base64.b64encode(buf.getvalue()).decode(),
                     'shape': list(emb.shape), 'time': round(elapsed, 4)}})
print(json.dumps({{'results': results, 'total_time': round(sum(times), 3),
                    'dim': results[0]['shape'][-1] if results else 0,
                    'times': [round(t, 4) for t in times]}}))
"""
    elif model_key == "dinotool":
        return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
sys.path.insert(0, '/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/DINOtool')
from dinotool_wrapper import DINoToolWorker
worker = DINoToolWorker(model_name='dinov2_vits14_reg', device=None)
vision_model = worker.load_vision_model()
transform = vision_model.get_transform((224, 224)).transform
image_paths = {paths_json}
results = []
times = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        cls_token = vision_model(img_tensor, features="frame")
        cls_token = F.normalize(cls_token, dim=-1)
    elapsed = time.time() - t0
    times.append(elapsed)
    buf = io.BytesIO()
    np.save(buf, cls_token.cpu().float().numpy())
    results.append({{'file': path, 'emb': base64.b64encode(buf.getvalue()).decode(),
                     'shape': list(cls_token.shape), 'time': round(elapsed, 4)}})
print(json.dumps({{'results': results, 'total_time': round(sum(times), 3),
                    'dim': results[0]['shape'][-1] if results else 0,
                    'times': [round(t, 4) for t in times]}}))
"""
    return ""


def load_coco_category_labels(max_images=200):
    """Load COCO images with their category labels for k-NN and clustering."""
    from pycocotools.coco import COCO
    ap = COCO_DIR / "annotations" / "instances_val2017.json"
    if not ap.exists():
        return None, None, None
    coco = COCO(ap)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    img_ids = sorted(coco.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]

    img_to_cats = {}
    img_to_path = {}
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        img_to_path[img_id] = COCO_DIR / "val2017" / info["file_name"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        cats = list(set(a["category_id"] for a in anns))
        img_to_cats[img_id] = cats

    return img_ids, img_to_cats, img_to_path, cat_id_to_name


def compute_retrieval_metrics(embeddings, labels, k_values=(1, 5, 10)):
    """Compute image retrieval metrics using cosine similarity."""
    n = len(embeddings)
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)

    # For each query, find nearest neighbors
    recall_at_k = {k: 0 for k in k_values}
    total = 0

    for i in range(n):
        ranked = np.argsort(-sims[i])
        total += 1
        for k in k_values:
            retrieved_labels = [labels[j] for j in ranked[:k]]
            if labels[i] in retrieved_labels:
                recall_at_k[k] += 1

    metrics = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = recall_at_k[k] / total if total else 0

    # mAP
    aps = []
    for i in range(n):
        ranked = np.argsort(-sims[i])
        relevant = [1 if labels[j] == labels[i] else 0 for j in ranked]
        if sum(relevant) == 0:
            continue
        precisions = []
        hits = 0
        for j, rel in enumerate(relevant):
            if rel:
                hits += 1
                precisions.append(hits / (j + 1))
        aps.append(np.mean(precisions))
    metrics["mAP"] = np.mean(aps) if aps else 0
    return metrics


def compute_knn_metrics(embeddings, train_labels, test_embeddings, test_labels,
                         train_labels_for_knn=None, k=5):
    """Compute k-NN classification accuracy."""
    if train_labels_for_knn is None:
        train_labels_for_knn = train_labels
    sims = test_embeddings @ train_embeddings.T
    n_test = len(test_embeddings)
    top1_correct = 0
    top5_correct = 0

    for i in range(n_test):
        ranked = np.argsort(-sims[i])
        predicted_labels = [train_labels_for_knn[j] for j in ranked[:k]]
        if predicted_labels[0] == test_labels[i]:
            top1_correct += 1
        if test_labels[i] in predicted_labels:
            top5_correct += 1

    return {
        "top1": top1_correct / n_test if n_test else 0,
        "top5": top5_correct / n_test if n_test else 0,
    }


def compute_clustering_metrics(embeddings, labels):
    """Compute NMI and ARI for clustering quality."""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.cluster import KMeans

    n_classes = len(set(labels))
    if n_classes < 2 or n_classes >= len(labels):
        return {"NMI": 0, "ARI": 0}

    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings)

    nmi = normalized_mutual_info_score(labels, pred_labels)
    ari = adjusted_rand_score(labels, pred_labels)
    return {"NMI": nmi, "ARI": ari}


def main():
    parser = argparse.ArgumentParser(description="VEQ Benchmark")
    all_choices = sorted(VEQ_MODELS)
    parser.add_argument("--model", choices=all_choices, required=True)
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()

    mn = args.model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Visual Embedding Quality: {mn}")
    print(f"{'=' * 60}")

    # Extract embeddings
    print(f"  Extracting embeddings for {args.max_images} COCO images...")
    val_dir = COCO_DIR / "val2017"
    all_files = sorted(val_dir.glob("*.jpg"))
    if args.max_images:
        all_files = all_files[:args.max_images]

    venv_py = MODEL_VENV.get(mn)
    if not venv_py or not venv_py.exists():
        print(f"  Error: venv not found for {mn}")
        sys.exit(1)

    script = make_embedding_script(mn, all_files)
    t0 = time.time()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    result = subprocess.run(
        [str(venv_py), script_path],
        capture_output=True, text=True, timeout=600,
    )
    Path(script_path).unlink(missing_ok=True)
    embed_time = time.time() - t0

    if result.returncode != 0:
        print(f"  Error (code {result.returncode})")
        print(f"  {result.stderr[-2000:]}")
        sys.exit(1)

    data = json.loads(result.stdout)
    results_list = data["results"]
    emb_dim = data["dim"]
    total_time = data["total_time"]

    print(f"  Extracted {len(results_list)} embeddings (dim={emb_dim}) in {total_time:.1f}s")

    # Build embedding matrix
    embeddings = []
    emb_files = []
    for r in results_list:
        import io as _io
        raw = base64.b64decode(r["emb"])
        emb = np.load(_io.BytesIO(raw))
        embeddings.append(emb.flatten())
        emb_files.append(Path(r["file"]).stem)

    embeddings = np.array(embeddings, dtype=np.float32)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Load COCO category labels for each image
    coco_data = load_coco_category_labels(args.max_images)
    if coco_data is None:
        print("  Error: COCO data not found")
        sys.exit(1)

    img_ids, img_to_cats, img_to_path, cat_id_to_name = coco_data

    # Map file stems to categories (use primary category)
    labels = []
    for img_id in img_ids:
        path = img_to_path[img_id]
        stem = path.stem
        cats = img_to_cats[img_id]
        labels.append(cat_id_to_name.get(cats[0], "unknown") if cats else "unknown")

    labels = labels[:len(embeddings)]

    # ── Retrieval metrics ──────────────────────────────────────────
    print("  Computing retrieval metrics...")
    retrieval = compute_retrieval_metrics(embeddings, labels)

    # ── Clustering metrics ──────────────────────────────────────────
    print("  Computing clustering metrics...")
    clustering = compute_clustering_metrics(embeddings, labels)

    # ── Summary ─────────────────────────────────────────────────────
    fps = len(embeddings) / total_time if total_time > 0 else 0
    avg_ms = total_time / len(embeddings) * 1000 if embeddings.size else 0

    print(f"\n  {'Metric':<20} {'Value':>10}")
    print(f"  {'─'*20} {'─'*10}")
    for k, v in retrieval.items():
        print(f"  {k:<20} {v:>10.4f}")
    for k, v in clustering.items():
        print(f"  {k:<20} {v:>10.4f}")
    print(f"  {'Embedding dim':<20} {emb_dim:>10}")
    print(f"  {'FPS':<20} {fps:>10.2f}")
    print(f"  {'ms/image':<20} {avg_ms:>10.1f}")

    stats = {
        "model": mn,
        "model_key": mn,
        "task": "veq",
        "dataset": "coco",
        "images": len(embeddings),
        "embedding_dim": emb_dim,
        "total_inference_time_s": round(total_time, 3),
        "avg_inference_ms": round(avg_ms, 2),
        "fps": round(fps, 2),
        **{f"retrieval_{k}": round(v, 4) for k, v in retrieval.items()},
        **{f"clustering_{k}": round(v, 4) for k, v in clustering.items()},
    }

    save_stats(stats, f"{mn}_veq")
    return stats


if __name__ == "__main__":
    main()
