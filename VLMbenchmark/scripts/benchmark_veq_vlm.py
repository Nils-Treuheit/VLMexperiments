#!/usr/bin/env python3
"""VLM Visual Embedding Quality (VEQ) benchmark.

Evaluates VLM visual embeddings on three dimensions:
  1. Caption Consistency — full-VLM caption vs COCO references (BLEU-4, ROUGE-L, CIDEr)
  2. Semantic Clustering — intra/inter-class distance, silhouette, NMI, ARI
  3. Token Economy — number of visual tokens per image

Each VLM runs in its own subprocess/venv. The script extracts:
  - raw vision embeddings (before LLM projection)
  - projected embeddings (after vision-to-LLM projection, if accessible)
  - generated caption
  - visual token count

Uses COCO val2017 for evaluation.
"""

import argparse
import base64
import io
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    BASE_DIR, RESULTS_DIR, PROJECT_DIR, COCO_DIR,
    save_stats, bleu_score, rouge_l, cider,
)

# ── All VLMs + vision encoders we test ────────────────────────────────────
VEQ_VLM_MODELS = {
    "siglip2", "dinov3", "moonvit", "dinotool",
    "paligemma", "florence2", "cosmos_nemotron", "llama_vision",
    "qwen3_native", "phi_vision", "phi4_multimodal",
}

MODEL_VENV = {
    "siglip2": PROJECT_DIR / "siglip2" / ".venv" / "bin" / "python",
    "dinov3": PROJECT_DIR / "dinov3" / ".venv" / "bin" / "python",
    "moonvit": PROJECT_DIR / "moonvit" / ".venv" / "bin" / "python",
    "dinotool": PROJECT_DIR / "DINOtool" / ".venv" / "bin" / "python",
    "paligemma": PROJECT_DIR / "paligemma" / ".venv" / "bin" / "python",
    "florence2": PROJECT_DIR / "florence-2" / ".venv" / "bin" / "python",
    "cosmos_nemotron": PROJECT_DIR / "cosmos-nemotron" / ".venv" / "bin" / "python",
    "llama_vision": PROJECT_DIR / "llama-vision" / ".venv" / "bin" / "python",
    "qwen3_native": PROJECT_DIR / "qwen3-vl_instruct" / ".venv" / "bin" / "python",
    "phi_vision": PROJECT_DIR / "phi-vision" / ".venv" / "bin" / "python",
    "phi4_multimodal": PROJECT_DIR / "phi-4_multimodal" / ".venv" / "bin" / "python",
}


# ── Subprocess scripts per model ──────────────────────────────────────────
# Each script: loads model, extracts raw+projected embeddings, generates
# caption, counts tokens. Returns JSON to stdout.

def _make_siglip2_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
    return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
model = AutoModel.from_pretrained('google/siglip2-base-patch16-224',
    torch_dtype=dtype, device_map=device, attn_implementation='sdpa').eval()
processor = AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')
image_paths = {paths_json}
results = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.vision_model(**inputs)
    raw = out.pooler_output if (hasattr(out, 'pooler_output') and out.pooler_output is not None) else out.last_hidden_state.mean(dim=1)
    raw = raw / raw.norm(dim=-1, keepdim=True)
    elapsed = time.time() - t0
    buf = io.BytesIO(); np.save(buf, raw.cpu().float().numpy())
    results.append({{'file': path, 'raw_emb': base64.b64encode(buf.getvalue()).decode(),
                     'raw_dim': list(raw.shape)[-1], 'token_count': int(raw.shape[1]),
                     'proj_token_count': 0, 'time': round(elapsed, 4),
                     'caption': '', 'has_proj': False}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


def _make_dinov3_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
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
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to(device=device, dtype=dtype)
    with torch.no_grad():
        out = model(**inputs)
    raw = out.pooler_output if (hasattr(out, 'pooler_output') and out.pooler_output is not None) else out.last_hidden_state[:, 0, :]
    elapsed = time.time() - t0
    buf = io.BytesIO(); np.save(buf, raw.cpu().float().numpy())
    results.append({{'file': path, 'raw_emb': base64.b64encode(buf.getvalue()).decode(),
                     'raw_dim': list(raw.shape)[-1], 'token_count': int(inputs['pixel_values'].shape[2] // 16 * inputs['pixel_values'].shape[3] // 16),
                     'proj_token_count': 0, 'time': round(elapsed, 4),
                     'caption': '', 'has_proj': False}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


def _make_moonvit_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
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
        if hasattr(self, '_dg_all_tied_weights_keys'): return self._dg_all_tied_weights_keys
        keys = self._tied_weights_keys
        if keys is None: return {{}}
        if isinstance(keys, dict): return keys
        if isinstance(keys, str): return {{keys: keys}}
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
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    inputs = processor(img, return_tensors='pt').to(device=device, dtype=dtype)
    with torch.no_grad():
        features = model(inputs.pixel_values, inputs.image_grid_hws)
    pooled = [f.mean(dim=0) for f in features]
    emb = torch.stack(pooled).mean(dim=0)
    if emb.dim() == 2: emb = emb.mean(dim=0)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    elapsed = time.time() - t0
    buf = io.BytesIO(); np.save(buf, emb.cpu().float().numpy())
    results.append({{'file': path, 'raw_emb': base64.b64encode(buf.getvalue()).decode(),
                     'raw_dim': list(emb.shape)[-1], 'token_count': int(inputs['pixel_values'].shape[1]),
                     'proj_token_count': 0, 'time': round(elapsed, 4),
                     'caption': '', 'has_proj': False}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


def _make_dinotool_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
    return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image
sys.path.insert(0, '/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/DINOtool')
from dinotool_wrapper import DINoToolWorker
worker = DINoToolWorker(model_name='dinov2_vits14_reg', device=None)
vision_model = worker.load_vision_model()
transform = vision_model.get_transform((224, 224)).transform
image_paths = {paths_json}
results = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        cls_token = vision_model(img_tensor, features="frame")
        cls_token = F.normalize(cls_token, dim=-1)
    elapsed = time.time() - t0
    buf = io.BytesIO(); np.save(buf, cls_token.cpu().float().numpy())
    results.append({{'file': path, 'raw_emb': base64.b64encode(buf.getvalue()).decode(),
                     'raw_dim': list(cls_token.shape)[-1], 'token_count': 1,
                     'proj_token_count': 0, 'time': round(elapsed, 4),
                     'caption': '', 'has_proj': False}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


def _make_paligemma_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
    return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16
model = PaliGemmaForConditionalGeneration.from_pretrained(
    'google/paligemma2-3b-mix-224', torch_dtype=dtype, device_map=device).eval()
processor = AutoProcessor.from_pretrained('google/paligemma2-3b-mix-224')
image_paths = {paths_json}
results = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        raw_out = model.model.vision_tower(**inputs)
        raw = raw_out.last_hidden_state
        proj = model.model.multi_modal_projector(raw)
    raw_emb = raw.mean(dim=1)
    raw_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
    proj_emb = proj.mean(dim=1)
    proj_emb = proj_emb / proj_emb.norm(dim=-1, keepdim=True)
    cap_inputs = processor(images=img, text="caption en", return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**cap_inputs, max_new_tokens=64, do_sample=False)
    caption = processor.decode(out[0][cap_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - t0
    buf_r = io.BytesIO(); np.save(buf_r, raw_emb.cpu().float().numpy())
    buf_p = io.BytesIO(); np.save(buf_p, proj_emb.cpu().float().numpy())
    results.append({{'file': path,
                     'raw_emb': base64.b64encode(buf_r.getvalue()).decode(),
                     'proj_emb': base64.b64encode(buf_p.getvalue()).decode(),
                     'raw_dim': list(raw_emb.shape)[-1],
                     'proj_dim': list(proj_emb.shape)[-1],
                     'token_count': int(raw.shape[1]),
                     'proj_token_count': int(proj.shape[1]),
                     'time': round(elapsed, 4),
                     'caption': caption.strip(),
                     'has_proj': True}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


def _make_florence2_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
    return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16
model_id = 'microsoft/Florence-2-large-ft'
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
    trust_remote_code=True, device_map=device).eval()
image_paths = {paths_json}
results = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    pixel_values = processor(images=img, return_tensors='pt').pixel_values.to(device)
    with torch.no_grad():
        raw_out = model.model.vision_tower(pixel_values)
        raw = raw_out.last_hidden_state
        if raw.dim() == 4:
            raw = raw.flatten(2).permute(0, 2, 1)
        proj = model.model.multi_modal_projector(raw)
    raw_emb = raw.mean(dim=1)
    raw_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
    proj_emb = proj.mean(dim=1)
    proj_emb = proj_emb / proj_emb.norm(dim=-1, keepdim=True)
    cap_inputs = processor(text="<DETAILED_CAPTION>", images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(input_ids=cap_inputs['input_ids'],
                             pixel_values=cap_inputs['pixel_values'],
                             max_new_tokens=64, num_beams=1)
    caption = processor.batch_decode(out, skip_special_tokens=True)[0]
    elapsed = time.time() - t0
    buf_r = io.BytesIO(); np.save(buf_r, raw_emb.cpu().float().numpy())
    buf_p = io.BytesIO(); np.save(buf_p, proj_emb.cpu().float().numpy())
    results.append({{'file': path,
                     'raw_emb': base64.b64encode(buf_r.getvalue()).decode(),
                     'proj_emb': base64.b64encode(buf_p.getvalue()).decode(),
                     'raw_dim': list(raw_emb.shape)[-1],
                     'proj_dim': list(proj_emb.shape)[-1],
                     'token_count': int(raw.shape[1]),
                     'proj_token_count': int(proj.shape[1]),
                     'time': round(elapsed, 4),
                     'caption': caption.strip(),
                     'has_proj': True}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


def _make_cosmos_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
    return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16
model = AutoModelForMultimodalLM.from_pretrained('nvidia/Cosmos-Reason1-7B',
    torch_dtype=dtype, device_map=device, attn_implementation='sdpa').eval()
processor = AutoProcessor.from_pretrained('nvidia/Cosmos-Reason1-7B')
image_paths = {paths_json}
results = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    messages = [{{"role": "user", "content": [{{"type": "image", "image": img}}, {{"type": "text", "text": "Describe this image briefly."}}]}}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors='pt').to(device)
    with torch.no_grad():
        raw = model.model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
    proj = raw.mean(dim=1) if raw.dim() == 3 else raw
    proj = proj / proj.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    caption = processor.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    elapsed = time.time() - t0
    buf_p = io.BytesIO(); np.save(buf_p, proj.cpu().float().numpy())
    results.append({{'file': path,
                     'raw_emb': base64.b64encode(buf_p.getvalue()).decode(),
                     'proj_emb': base64.b64encode(buf_p.getvalue()).decode(),
                     'raw_dim': list(proj.shape)[-1],
                     'proj_dim': list(proj.shape)[-1],
                     'token_count': int(raw.shape[1]) if raw.dim() == 3 else int(raw.numel() // raw.shape[-1]),
                     'proj_token_count': int(proj.shape[1]) if proj.dim() == 3 else 1,
                     'time': round(elapsed, 4),
                     'caption': caption.strip(),
                     'has_proj': False}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


def _make_llama_vision_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
    return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16
model_name = 'unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit'
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForMultimodalLM.from_pretrained(model_name,
    device_map=device, torch_dtype=dtype).eval()
image_paths = {paths_json}
results = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    messages = [{{"role": "user", "content": [{{"type": "image"}}, {{"type": "text", "text": "Describe this image briefly."}}]}}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=chat, images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        raw_out = model.model.vision_model(inputs['pixel_values'],
            aspect_ratio_ids=inputs.get('aspect_ratio_ids'),
            aspect_ratio_mask=inputs.get('aspect_ratio_mask'))
        raw = raw_out.last_hidden_state
        proj = model.model.multi_modal_projector(raw)
    raw_emb = raw.mean(dim=1)
    raw_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
    proj_emb = proj.mean(dim=1)
    proj_emb = proj_emb / proj_emb.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    text = processor.decode(out[0], skip_special_tokens=True)
    caption = text[len(chat):].strip() if chat in text else text.strip()
    elapsed = time.time() - t0
    buf_r = io.BytesIO(); np.save(buf_r, raw_emb.cpu().float().numpy())
    buf_p = io.BytesIO(); np.save(buf_p, proj_emb.cpu().float().numpy())
    results.append({{'file': path,
                     'raw_emb': base64.b64encode(buf_r.getvalue()).decode(),
                     'proj_emb': base64.b64encode(buf_p.getvalue()).decode(),
                     'raw_dim': list(raw_emb.shape)[-1],
                     'proj_dim': list(proj_emb.shape)[-1],
                     'token_count': int(raw.shape[1]),
                     'proj_token_count': int(proj.shape[1]),
                     'time': round(elapsed, 4),
                     'caption': caption.strip(),
                     'has_proj': True}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


def _make_qwen3_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
    return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
sys.path.insert(0, str(Path('/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/qwen3-vl_instruct')))
from pathlib import Path
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16
model_dir = str(Path('/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/qwen3-vl_instruct/model_vl'))
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir,
    torch_dtype=dtype, device_map=device, attn_implementation='sdpa', trust_remote_code=True).eval()
image_paths = {paths_json}
results = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    messages = [{{"role": "user", "content": [{{"type": "image", "image": img}}, {{"type": "text", "text": "Describe this image briefly."}}]}}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=img, text=chat, padding=True, return_tensors='pt')
    inputs = {{k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}}
    with torch.no_grad():
        vis_out = model.model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
    proj = vis_out.mean(dim=1) if vis_out.dim() == 3 else vis_out
    proj = proj / proj.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    text = processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - t0
    ntok = int(vis_out.shape[1]) if vis_out.dim() == 3 else int(vis_out.numel() // vis_out.shape[-1])
    buf_p = io.BytesIO(); np.save(buf_p, proj.cpu().float().numpy())
    results.append({{'file': path,
                     'raw_emb': base64.b64encode(buf_p.getvalue()).decode(),
                     'proj_emb': base64.b64encode(buf_p.getvalue()).decode(),
                     'raw_dim': list(proj.shape)[-1],
                     'proj_dim': list(proj.shape)[-1],
                     'token_count': ntok,
                     'proj_token_count': int(proj.shape[1]) if proj.dim() == 3 else 1,
                     'time': round(elapsed, 4),
                     'caption': text.strip(),
                     'has_proj': False}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


def _make_phi_vision_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
    return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'microsoft/Phi-3.5-vision-instruct'
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config._attn_implementation = 'eager'
model = AutoModelForCausalLM.from_pretrained(model_id, config=config,
    trust_remote_code=True, torch_dtype='auto', device_map=device).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
image_paths = {paths_json}
results = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    try:
        vt = model.vision_tower if hasattr(model, 'vision_tower') else model.model.vision_tower
    except:
        vt = None
    if vt is not None:
        pixel_values = processor(images=img, return_tensors='pt').pixel_values.to(device)
        with torch.no_grad():
            raw_out = vt(pixel_values)
            raw = raw_out.last_hidden_state if hasattr(raw_out, 'last_hidden_state') else raw_out
            if raw.dim() == 4: raw = raw.flatten(2).permute(0, 2, 1)
            try:
                proj = model.multi_modal_projector(raw) if hasattr(model, 'multi_modal_projector') else raw
            except:
                proj = raw
        raw_emb = raw.mean(dim=1)
        raw_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
        proj_emb = proj.mean(dim=1)
        proj_emb = proj_emb / proj_emb.norm(dim=-1, keepdim=True)
    else:
        raw_emb = torch.zeros(1, 1024, device=device)
        proj_emb = raw_emb
    phi_prompt = f"<|user|>\\n<|image_1|>\\nDescribe this image briefly.<|end|>\\n<|assistant|>\\n"
    inputs = processor(phi_prompt, img, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    caption = processor.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - t0
    buf_r = io.BytesIO(); np.save(buf_r, raw_emb.cpu().float().numpy())
    buf_p = io.BytesIO(); np.save(buf_p, proj_emb.cpu().float().numpy())
    results.append({{'file': path,
                     'raw_emb': base64.b64encode(buf_r.getvalue()).decode(),
                     'proj_emb': base64.b64encode(buf_p.getvalue()).decode(),
                     'raw_dim': list(raw_emb.shape)[-1],
                     'proj_dim': list(proj_emb.shape)[-1],
                     'token_count': int(raw.shape[1]) if raw.dim() == 3 else 1,
                     'proj_token_count': int(proj.shape[1]) if proj.dim() == 3 else 1,
                     'time': round(elapsed, 4),
                     'caption': caption.strip(),
                     'has_proj': True}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


def _make_phi4_script(image_paths):
    paths_json = json.dumps([str(p) for p in image_paths])
    return f"""
import json, sys, time, warnings, base64, io
warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'microsoft/Phi-4-multimodal-instruct'
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
    device_map=device, trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
image_paths = {paths_json}
results = []
for path in image_paths:
    t0 = time.time()
    img = Image.open(path).convert('RGB')
    try:
        ve = model.model.vision_encoder
        with torch.no_grad():
            pixel_values = processor(images=img, return_tensors='pt').pixel_values.to(device)
            raw = ve(pixel_values)
            if hasattr(raw, 'last_hidden_state'):
                raw = raw.last_hidden_state
            elif isinstance(raw, torch.Tensor) and raw.dim() == 4:
                raw = raw.flatten(2).permute(0, 2, 1)
            try:
                proj = model.model.multi_modal_projector(raw)
            except:
                proj = raw
        raw_emb = raw.mean(dim=1) if raw.dim() == 3 else raw.reshape(1, -1)
        raw_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
        proj_emb = proj.mean(dim=1) if proj.dim() == 3 else proj.reshape(1, -1)
        proj_emb = proj_emb / proj_emb.norm(dim=-1, keepdim=True)
    except Exception as e:
        raw_emb = torch.zeros(1, 1152, device=device)
        proj_emb = raw_emb
    f4_prompt = f"<|user|><|image_1|>Describe this image briefly.<|end|><|assistant|>"
    inputs = processor(text=f4_prompt, images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_logits_to_keep=1)
    caption = processor.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - t0
    buf_r = io.BytesIO(); np.save(buf_r, raw_emb.cpu().float().numpy())
    buf_p = io.BytesIO(); np.save(buf_p, proj_emb.cpu().float().numpy())
    results.append({{'file': path,
                     'raw_emb': base64.b64encode(buf_r.getvalue()).decode(),
                     'proj_emb': base64.b64encode(buf_p.getvalue()).decode(),
                     'raw_dim': list(raw_emb.shape)[-1],
                     'proj_dim': list(proj_emb.shape)[-1],
                     'token_count': int(raw.shape[1]) if raw.dim() == 3 else 1,
                     'proj_token_count': int(proj.shape[1]) if proj.dim() == 3 else 1,
                     'time': round(elapsed, 4),
                     'caption': caption.strip(),
                     'has_proj': True}})
print(json.dumps({{'results': results, 'total_time': round(sum(r['time'] for r in results), 3)}}))
"""


SCRIPT_BUILDERS = {
    "siglip2": _make_siglip2_script,
    "dinov3": _make_dinov3_script,
    "moonvit": _make_moonvit_script,
    "dinotool": _make_dinotool_script,
    "paligemma": _make_paligemma_script,
    "florence2": _make_florence2_script,
    "cosmos_nemotron": _make_cosmos_script,
    "llama_vision": _make_llama_vision_script,
    "qwen3_native": _make_qwen3_script,
    "phi_vision": _make_phi_vision_script,
    "phi4_multimodal": _make_phi4_script,
}


# ── Data loading ──────────────────────────────────────────────────────────

def load_coco_veq_data(max_images=200, min_per_category=5):
    """Load COCO images, captions, and category labels for VEQ evaluation."""
    from pycocotools.coco import COCO

    inst_ap = COCO_DIR / "annotations" / "instances_val2017.json"
    cap_ap = COCO_DIR / "annotations" / "captions_val2017.json"
    if not inst_ap.exists() or not cap_ap.exists():
        return None

    coco_inst = COCO(inst_ap)
    coco_cap = COCO(cap_ap)

    cat_id_to_name = {c["id"]: c["name"] for c in coco_inst.loadCats(coco_inst.getCatIds())}

    # Count images per category
    cat_counts = defaultdict(int)
    img_to_cats = {}
    for img_id in coco_inst.getImgIds():
        anns = coco_inst.loadAnns(coco_inst.getAnnIds(imgIds=img_id))
        cats = list(set(a["category_id"] for a in anns))
        img_to_cats[img_id] = cats
        for c in cats:
            cat_counts[c] += 1

    # Filter to categories with enough images
    valid_cats = {c for c, n in cat_counts.items() if n >= min_per_category}

    # Select balanced image set: sample up to 20 images per category
    selected_imgs = set()
    for cat_id in valid_cats:
        img_ids_for_cat = [iid for iid, cats in img_to_cats.items() if cat_id in cats]
        selected_imgs.update(img_ids_for_cat[:20])

    selected_imgs = sorted(selected_imgs)
    if max_images:
        selected_imgs = selected_imgs[:max_images]

    # Load captions
    img_id_to_captions = defaultdict(list)
    for ann in coco_cap.loadAnns(coco_cap.getAnnIds()):
        img_id_to_captions[ann["image_id"]].append(ann["caption"])

    # Build result
    images = []
    for img_id in selected_imgs:
        info = coco_inst.loadImgs(img_id)[0]
        cats = img_to_cats.get(img_id, [])
        primary_cat = cat_id_to_name.get(cats[0], "unknown") if cats else "unknown"
        images.append({
            "id": img_id,
            "path": str(COCO_DIR / "val2017" / info["file_name"]),
            "file_name": info["file_name"],
            "category": primary_cat,
            "captions": img_id_to_captions.get(img_id, []),
        })

    return {
        "images": images,
        "categories": {cat_id_to_name[c]: c for c in valid_cats},
    }


# ── Metric computation ───────────────────────────────────────────────────

def compute_caption_consistency(captions, all_captions):
    """Compute BLEU-4, ROUGE-L, CIDEr for generated captions vs COCO refs."""
    bleu4_scores = []
    rouge_scores = []
    cider_scores = []

    for gen_cap, refs in zip(captions, all_captions):
        if not gen_cap or not refs:
            bleu4_scores.append(0)
            rouge_scores.append(0)
            cider_scores.append(0)
            continue
        bleu4_scores.append(bleu_score(gen_cap, refs))
        rouge_scores.append(rouge_l(gen_cap, refs))
        cider_scores.append(cider(gen_cap, refs))

    return {
        "bleu4": round(np.mean(bleu4_scores), 4) if bleu4_scores else 0,
        "rouge_l": round(np.mean(rouge_scores), 4) if rouge_scores else 0,
        "cider": round(np.mean(cider_scores), 4) if cider_scores else 0,
        "avg_caption_len": round(np.mean([len(c.split()) for c in captions if c]), 1) if captions else 0,
    }


def compute_semantic_clustering(embeddings, labels, max_for_silhouette=500):
    """Compute intra/inter-class distance, silhouette, NMI, ARI."""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.cluster import KMeans

    n = len(embeddings)
    unique_labels = list(set(labels))
    n_classes = len(unique_labels)

    if n_classes < 2 or n < n_classes:
        return {
            "intra_class_dist": 0, "inter_class_dist": 0,
            "intra_inter_ratio": 0, "silhouette": 0,
            "NMI": 0, "ARI": 0,
        }

    # Compute pairwise cosine distances
    sims = embeddings @ embeddings.T

    # Intra-class distance: mean cosine distance between same-class pairs
    intra_dists = []
    inter_dists = []
    label_to_idx = defaultdict(list)
    for i, lab in enumerate(labels):
        label_to_idx[lab].append(i)

    for lab, idxs in label_to_idx.items():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, min(len(idxs), 50)):
                intra_dists.append(1.0 - sims[idxs[a], idxs[b]])

    # Inter-class: sample random pairs from different classes
    rng = np.random.RandomState(42)
    inter_count = 0
    for _ in range(min(n * 10, 50000)):
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if labels[i] != labels[j]:
            inter_dists.append(1.0 - sims[i, j])
            inter_count += 1
            if inter_count >= 5000:
                break

    intra_mean = np.mean(intra_dists) if intra_dists else 0
    inter_mean = np.mean(inter_dists) if inter_dists else 1
    ratio = intra_mean / inter_mean if inter_mean > 0 else 0

    # Silhouette (on subset if too large)
    sil_score = 0
    if n <= max_for_silhouette:
        from sklearn.metrics import silhouette_score as _sil
        sil_score = _sil(embeddings, labels)
    else:
        idx = rng.choice(n, max_for_silhouette, replace=False)
        from sklearn.metrics import silhouette_score as _sil
        sil_score = _sil(embeddings[idx], [labels[i] for i in idx])

    # NMI, ARI via KMeans
    n_init = 10
    if n_classes >= n:
        nmi = ari = 0
    else:
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=n_init)
        pred = kmeans.fit_predict(embeddings)
        nmi = normalized_mutual_info_score(labels, pred)
        ari = adjusted_rand_score(labels, pred)

    return {
        "intra_class_dist": round(float(intra_mean), 4),
        "inter_class_dist": round(float(inter_mean), 4),
        "intra_inter_ratio": round(float(ratio), 4),
        "silhouette": round(float(sil_score), 4),
        "NMI": round(float(nmi), 4),
        "ARI": round(float(ari), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLM Visual Embedding Quality Benchmark")
    all_choices = sorted(VEQ_VLM_MODELS)
    parser.add_argument("--model", choices=all_choices, required=True)
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()

    mn = args.model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"VLM VEQ: {mn}")
    print(f"{'=' * 60}")

    # Load COCO data
    print("  Loading COCO data...")
    coco_data = load_coco_veq_data(args.max_images)
    if coco_data is None:
        print("  Error: COCO data not found")
        sys.exit(1)

    images = coco_data["images"]
    print(f"  {len(images)} images, {len(coco_data['categories'])} categories")

    image_paths = [img["path"] for img in images]
    categories = [img["category"] for img in images]
    all_captions = [img["captions"] for img in images]

    # Build and run subprocess script
    builder = SCRIPT_BUILDERS.get(mn)
    if not builder:
        print(f"  Error: no script builder for {mn}")
        sys.exit(1)

    venv_py = MODEL_VENV.get(mn)
    if not venv_py or not venv_py.exists():
        print(f"  Error: venv not found for {mn} at {venv_py}")
        sys.exit(1)

    script = builder(image_paths)
    print(f"  Running in {venv_py.parent.parent.name}...")

    t0 = time.time()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    timeout = 1200 if mn in ("llama_vision", "phi4_multimodal", "phi_vision") else 600
    try:
        result = subprocess.run(
            [str(venv_py), script_path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  Error: subprocess timed out after {timeout}s")
        sys.exit(1)
    finally:
        Path(script_path).unlink(missing_ok=True)

    wall_time = time.time() - t0

    if result.returncode != 0:
        print(f"  Error (code {result.returncode})")
        print(f"  stderr: {result.stderr[-3000:]}")
        sys.exit(1)

    data = json.loads(result.stdout)
    results_list = data["results"]
    total_time = data["total_time"]

    print(f"  Processed {len(results_list)} images in {wall_time:.1f}s")

    # Parse embeddings
    raw_embeddings = []
    proj_embeddings = []
    captions = []
    token_counts = []
    has_proj = False

    for r in results_list:
        raw_emb = np.load(io.BytesIO(base64.b64decode(r["raw_emb"]))).flatten()
        raw_embeddings.append(raw_emb)
        if r.get("has_proj") and r.get("proj_emb"):
            proj_emb = np.load(io.BytesIO(base64.b64decode(r["proj_emb"]))).flatten()
            proj_embeddings.append(proj_emb)
            has_proj = True
        captions.append(r.get("caption", ""))
        token_counts.append(r.get("token_count", 0))

    raw_embeddings = np.array(raw_embeddings, dtype=np.float32)
    raw_embeddings = raw_embeddings / (np.linalg.norm(raw_embeddings, axis=1, keepdims=True) + 1e-8)

    if has_proj:
        proj_embeddings = np.array(proj_embeddings, dtype=np.float32)
        proj_embeddings = proj_embeddings / (np.linalg.norm(proj_embeddings, axis=1, keepdims=True) + 1e-8)

    # ── Test 1: Caption Consistency ──────────────────────────────────
    print("  Computing caption consistency...")
    gen_caps = [c for c in captions if c]
    ref_caps = [all_captions[i] for i, c in enumerate(captions) if c]
    if gen_caps:
        caption_metrics = compute_caption_consistency(gen_caps, ref_caps)
    else:
        caption_metrics = {"bleu4": 0, "rouge_l": 0, "cider": 0, "avg_caption_len": 0}

    # ── Test 2: Semantic Clustering (raw embeddings) ────────────────
    print("  Computing semantic clustering (raw embeddings)...")
    clustering_raw = compute_semantic_clustering(raw_embeddings, categories)

    # ── Test 2b: Semantic Clustering (projected embeddings) ─────────
    clustering_proj = None
    if has_proj:
        print("  Computing semantic clustering (projected embeddings)...")
        clustering_proj = compute_semantic_clustering(proj_embeddings, categories)

    # ── Test 3: Token Economy ───────────────────────────────────────
    avg_tokens = int(np.mean(token_counts)) if token_counts else 0
    total_tokens = int(np.sum(token_counts)) if token_counts else 0

    # ── Summary ──────────────────────────────────────────────────────
    fps = len(results_list) / total_time if total_time > 0 else 0
    avg_ms = total_time / len(results_list) * 1000 if results_list else 0
    raw_dim = int(raw_embeddings.shape[1]) if raw_embeddings.ndim == 2 else 0
    proj_dim = int(proj_embeddings.shape[1]) if has_proj and proj_embeddings.ndim == 2 else 0

    print(f"\n  {'='*50}")
    print(f"  Caption Consistency (vs COCO refs)")
    print(f"  {'='*50}")
    print(f"  {'BLEU-4':<20} {caption_metrics['bleu4']:>10.4f}")
    print(f"  {'ROUGE-L':<20} {caption_metrics['rouge_l']:>10.4f}")
    print(f"  {'CIDEr':<20} {caption_metrics['cider']:>10.4f}")
    print(f"  {'Avg caption len':<20} {caption_metrics['avg_caption_len']:>10.1f} words")

    print(f"\n  {'='*50}")
    print(f"  Semantic Clustering (raw embeddings)")
    print(f"  {'='*50}")
    for k, v in clustering_raw.items():
        print(f"  {k:<25} {v:>10.4f}")

    if clustering_proj:
        print(f"\n  {'='*50}")
        print(f"  Semantic Clustering (projected embeddings)")
        print(f"  {'='*50}")
        for k, v in clustering_proj.items():
            print(f"  {k:<25} {v:>10.4f}")

    print(f"\n  {'='*50}")
    print(f"  Token Economy")
    print(f"  {'='*50}")
    print(f"  {'Raw dim':<20} {raw_dim:>10}")
    if proj_dim:
        print(f"  {'Proj dim':<20} {proj_dim:>10}")
    print(f"  {'Avg tokens/img':<20} {avg_tokens:>10}")
    print(f"  {'Total tokens':<20} {total_tokens:>10}")
    print(f"  {'FPS':<20} {fps:>10.2f}")
    print(f"  {'ms/image':<20} {avg_ms:>10.1f}")

    # ── Save stats ──────────────────────────────────────────────────
    stats = {
        "model": mn,
        "model_key": mn,
        "task": "veq_vlm",
        "dataset": "coco",
        "images": len(results_list),
        "raw_embedding_dim": raw_dim,
        "proj_embedding_dim": proj_dim,
        "total_inference_time_s": round(total_time, 3),
        "avg_inference_ms": round(avg_ms, 2),
        "fps": round(fps, 2),
        "avg_tokens_per_image": avg_tokens,
        "total_tokens": total_tokens,
        # Caption consistency
        "caption_bleu4": caption_metrics["bleu4"],
        "caption_rouge_l": caption_metrics["rouge_l"],
        "caption_cider": caption_metrics["cider"],
        "caption_avg_len": caption_metrics["avg_caption_len"],
        # Clustering (raw)
        "raw_intra_class_dist": clustering_raw["intra_class_dist"],
        "raw_inter_class_dist": clustering_raw["inter_class_dist"],
        "raw_intra_inter_ratio": clustering_raw["intra_inter_ratio"],
        "raw_silhouette": clustering_raw["silhouette"],
        "raw_NMI": clustering_raw["NMI"],
        "raw_ARI": clustering_raw["ARI"],
    }
    if clustering_proj:
        stats.update({
            "proj_intra_class_dist": clustering_proj["intra_class_dist"],
            "proj_inter_class_dist": clustering_proj["inter_class_dist"],
            "proj_intra_inter_ratio": clustering_proj["intra_inter_ratio"],
            "proj_silhouette": clustering_proj["silhouette"],
            "proj_NMI": clustering_proj["NMI"],
            "proj_ARI": clustering_proj["ARI"],
        })

    save_stats(stats, f"{mn}_veq_vlm")
    return stats


if __name__ == "__main__":
    main()
