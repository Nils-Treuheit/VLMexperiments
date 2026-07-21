#!/usr/bin/env python3
"""Embedding extraction benchmark for vision encoder models on COCO."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_DIR, RESULTS_DIR, PROJECT_DIR, COCO_DIR, save_stats, print_comparison

EMBEDDING_MODELS = {"siglip2", "dinov3", "moonvit", "dinotool"}

MODEL_VENV = {
    "siglip2": PROJECT_DIR / "siglip2" / ".venv" / "bin" / "python",
    "dinov3": PROJECT_DIR / "dinov3" / ".venv" / "bin" / "python",
    "moonvit": PROJECT_DIR / "moonvit" / ".venv" / "bin" / "python",
    "dinotool": PROJECT_DIR / "DINOtool" / ".venv" / "bin" / "python",
}


def make_inline_script(model_key, image_paths):
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
    elapsed = time.time() - t0
    times.append(elapsed)
    buf = io.BytesIO()
    np.save(buf, emb.cpu().numpy())
    results.append({{'file': path, 'embedding_b64': base64.b64encode(buf.getvalue()).decode(),
                     'shape': list(emb.shape), 'time': round(elapsed, 4)}})
print(json.dumps({{'results': results, 'total_time': round(sum(times), 3),
                    'dim': results[0]['shape'][-1] if results else 0,
                    'per_image_times': [round(t, 4) for t in times]}}))
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
    if hasattr(outputs, 'pooler_output'):
        emb = outputs.pooler_output
    else:
        emb = outputs.last_hidden_state[:, 0, :]
    elapsed = time.time() - t0
    times.append(elapsed)
    buf = io.BytesIO()
    np.save(buf, emb.cpu().numpy())
    results.append({{'file': path, 'embedding_b64': base64.b64encode(buf.getvalue()).decode(),
                     'shape': list(emb.shape), 'time': round(elapsed, 4)}})
print(json.dumps({{'results': results, 'total_time': round(sum(times), 3),
                    'dim': results[0]['shape'][-1] if results else 0,
                    'per_image_times': [round(t, 4) for t in times]}}))
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
    elapsed = time.time() - t0
    times.append(elapsed)
    buf = io.BytesIO()
    np.save(buf, emb.cpu().numpy())
    results.append({{'file': path, 'embedding_b64': base64.b64encode(buf.getvalue()).decode(),
                     'shape': list(emb.shape), 'time': round(elapsed, 4)}})
print(json.dumps({{'results': results, 'total_time': round(sum(times), 3),
                    'dim': results[0]['shape'][-1] if results else 0,
                    'per_image_times': [round(t, 4) for t in times]}}))
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
    np.save(buf, cls_token.cpu().numpy())
    results.append({{'file': path, 'embedding_b64': base64.b64encode(buf.getvalue()).decode(),
                     'shape': list(cls_token.shape), 'time': round(elapsed, 4)}})
print(json.dumps({{'results': results, 'total_time': round(sum(times), 3),
                    'dim': results[0]['shape'][-1] if results else 0,
                    'per_image_times': [round(t, 4) for t in times]}}))
"""

    return ""


def main():
    parser = argparse.ArgumentParser(description="Embedding Extraction Benchmark (COCO)")
    parser.add_argument("--model", type=str, default="siglip2", choices=sorted(EMBEDDING_MODELS))
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--samples-file", type=str, default=None)
    args = parser.parse_args()

    mn = args.model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = COCO_DIR / "val2017"
    all_files = sorted(img_dir.glob("*.jpg"))
    if args.max_images:
        all_files = all_files[:args.max_images]

    print(f"\nEmbedding Benchmark: {mn}")
    print(f"  Images: {len(all_files)}")

    script = make_inline_script(mn, all_files)
    venv_py = MODEL_VENV[mn]

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
        print(f"  stderr: {result.stderr[-2000:]}")
        sys.exit(1)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"  JSON error: {e}")
        print(f"  stdout[:1000]: {result.stdout[:1000]}")
        sys.exit(1)

    results_list = data.get("results", [])
    total_time = data.get("total_time", 0)
    emb_dim = data.get("dim", 0)
    per_image_times = data.get("per_image_times", [])

    total = len(results_list)
    avg_time = total_time / total if total > 0 else 0
    fps = total / total_time if total_time > 0 else 0

    print(f"  [{total}/{total}] dim={emb_dim} avg={avg_time*1000:.1f}ms total={total_time:.2f}s fps={fps:.2f}")

    stats = {
        "model": f"{mn} (Embedding)",
        "model_key": mn,
        "task": "embedding",
        "dataset": "coco",
        "images": total,
        "embedding_dim": emb_dim,
        "total_inference_time_s": round(total_time, 3),
        "avg_inference_ms": round(avg_time * 1000, 2),
        "fps": round(fps, 2),
    }

    save_stats(stats, f"{mn}_embedding")

    if per_image_times:
        print(f"  Time stats: min={min(per_image_times)*1000:.1f}ms "
              f"max={max(per_image_times)*1000:.1f}ms "
              f"median={sorted(per_image_times)[len(per_image_times)//2]*1000:.1f}ms")

    print(f"\nEmbedding Results: {mn}")
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Avg per image: {avg_time*1000:.1f}ms")
    print(f"  FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
