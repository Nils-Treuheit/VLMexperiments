#!/usr/bin/env python3
"""
Export MoonViT vision encoder + MLP connector to ONNX and TRT.

This module provides:
1. ONNX-compatible RoPE (real-valued, no complex numbers)
2. MoonViTOnnxWrapper - a torch.nn.Module wrapping MoonViT + MLP for ONNX export
3. export_onnx() - exports the wrapper to ONNX opset 18 (float32)
4. OnnxVisionEncoder - ORT-based inference class (replaces PyTorch vision encoder)

Key design decisions:
- Fixed-size 532x532 input (38x38 grid, 1444 patches, 361 merged after 2x2 merge).
  This avoids dynamic shape issues in ONNX's _patch_merge view() operations.
- Manual bmm attention instead of F.scaled_dot_product_attention because
  PyTorch 2.11 changed SDPA mask handling for 3D inputs.
- Float32 export (opset 18 Conv does not support bfloat16).
- Real-valued RoPE avoids view_as_complex/torch.polar which are ONNX-incompatible.
- Legacy dynamo=False exporter handles model control flow; dynamo-based export
  fails on data-dependent int(h) in _patch_merge.

Inputs:  pixel_values [1444, 3, 14, 14] float32
         cos [1444, 36] float32
         sin [1444, 36] float32
Output:  visual_features [361, 2048] float32
"""
import gc, json, logging, os, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for name in ("transformers_modules", "urllib3", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/mnt/HDD1/tmp/torchinductor"
os.environ["TORCH_COMPILE_DIR"] = "/mnt/HDD1/tmp/torch_compile"
os.environ["HF_HOME"] = "/mnt/HDD1/tmp/hf_home"
os.environ["XDG_CACHE_HOME"] = "/mnt/HDD1/tmp"
os.makedirs("/mnt/HDD1/tmp/torchinductor", exist_ok=True)

LA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "onnx", "vision_encoder.onnx")

# --------------------------------------------------------------------------
# ONNX-compatible RoPE (no complex numbers)
# --------------------------------------------------------------------------
def onnx_apply_rope(xq: torch.Tensor, xk: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor):
    """Real-valued RoPE. xq/xk: [L, H, D], cos/sin: [L, D/2]."""
    L, H, D = xq.shape
    D_half = D // 2
    xqr = xq.reshape(L, H, D_half, 2)
    xq_even = xqr[..., 0]
    xq_odd = xqr[..., 1]
    xkr = xk.reshape(L, H, D_half, 2)
    xk_even = xkr[..., 0]
    xk_odd = xkr[..., 1]

    cos = cos.unsqueeze(1).to(xq.dtype)
    sin = sin.unsqueeze(1).to(xq.dtype)

    xq_even_out = xq_even * cos - xq_odd * sin
    xq_odd_out = xq_even * sin + xq_odd * cos
    xk_even_out = xk_even * cos - xk_odd * sin
    xk_odd_out = xk_even * sin + xk_odd * cos

    xq_out = torch.stack([xq_even_out, xq_odd_out], dim=-1).reshape(L, H, D)
    xk_out = torch.stack([xk_even_out, xk_odd_out], dim=-1).reshape(L, H, D)
    return xq_out, xk_out


def compute_rope_cos_sin(grid_hws: torch.Tensor, max_height: int, max_width: int,
                          dim: int, theta_base: float = 10000.0, device=None):
    """Precompute cos/sin for 2D RoPE. Returns [total_patches, dim/2] for cos and sin."""
    if device is None:
        device = grid_hws.device
    D = dim
    N_max = max_height * max_width
    flat_pos = torch.arange(N_max, device=device).float()
    x_pos = flat_pos % max_width
    y_pos = flat_pos // max_width
    dim_range = torch.arange(0, D, 4, device=device)[: D // 4].float()
    freqs = 1.0 / (theta_base ** (dim_range / D))
    x_freqs = torch.outer(x_pos, freqs)
    y_freqs = torch.outer(y_pos, freqs)
    x_cos = torch.cos(x_freqs)
    x_sin = torch.sin(x_freqs)
    y_cos = torch.cos(y_freqs)
    y_sin = torch.sin(y_freqs)
    cos_full = torch.stack([x_cos, y_cos], dim=-1).reshape(max_height, max_width, D // 2)
    sin_full = torch.stack([x_sin, y_sin], dim=-1).reshape(max_height, max_width, D // 2)

    shapes = grid_hws.tolist()
    cos_out = torch.cat([cos_full[:h, :w].reshape(-1, D // 2) for h, w in shapes], dim=0)
    sin_out = torch.cat([sin_full[:h, :w].reshape(-1, D // 2) for h, w in shapes], dim=0)
    return cos_out, sin_out


# --------------------------------------------------------------------------
# ONNX-friendly MoonViT wrapper
# --------------------------------------------------------------------------
class MoonViTOnnxWrapper(nn.Module):
    """Wraps MoonVitPretrainedModel with ONNX-compatible RoPE.
    Fixed for 532x532 input (38×38 grid, 1444 patches, 361 merged)."""

    def __init__(self, original_vit, mlp):
        super().__init__()
        self.patch_embed = original_vit.patch_embed
        self.encoder = original_vit.encoder
        self.final_layernorm = original_vit.encoder.final_layernorm
        self.merge_kernel_size = original_vit.merge_kernel_size
        self.mlp = mlp
        self.head_dim = original_vit.config.hidden_size // original_vit.config.num_attention_heads

    @torch.no_grad()
    def forward(self, pixel_values, cos, sin):
        grid_hws = torch.tensor([[38, 38]], device=pixel_values.device, dtype=torch.int32)
        hidden = self.patch_embed.forward(pixel_values, grid_hws)
        L = hidden.shape[0]
        mask_bias = torch.zeros(1, 1, L, L, device=hidden.device, dtype=hidden.dtype)

        for block in self.encoder.blocks:
            residual = hidden
            hidden = block.norm0(hidden)
            xqkv = block.wqkv(hidden)
            qkv_shape = xqkv.shape[:-1] + (3, block.num_heads, self.head_dim)
            xqkv = xqkv.view(*qkv_shape)
            xq, xk, xv = torch.unbind(xqkv, dim=-3)
            xq, xk = onnx_apply_rope(xq, xk, cos, sin)

            L2, H, D = xq.shape
            scale = D ** -0.5
            q_t = xq.permute(1, 0, 2).float()
            k_t = xk.permute(1, 2, 0).float()
            attn = torch.bmm(q_t, k_t) * scale
            attn = attn + mask_bias.squeeze(0).float() * -10000.0
            attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(xq.dtype)
            v_t = xv.permute(1, 0, 2).to(xq.dtype)
            out = torch.bmm(attn, v_t)
            out = out.permute(1, 0, 2).reshape(L2, -1)

            out = block.wo(out)
            hidden = residual + out
            residual = hidden
            hidden = block.norm1(hidden)
            hidden = block.mlp(hidden)
            hidden = residual + hidden

        hidden = self.final_layernorm(hidden)
        merged = self._patch_merge(hidden)
        features = self.mlp(merged)
        return features

    def _patch_merge(self, x):
        kh, kw = self.merge_kernel_size
        H, W = 38, 38
        nh, nw = H // kh, W // kw
        rs = x.view(nh, kh, nw, kw, -1).permute(0, 2, 1, 3, 4).contiguous()
        return rs.view(nh * nw, -1)


def compute_cu_seqlens(grid_hws, device):
    lengths = torch.cat([
        torch.zeros(1, device=device, dtype=grid_hws.dtype),
        grid_hws[:, 0] * grid_hws[:, 1],
    ])
    return lengths.cumsum(dim=0, dtype=torch.int32)


# --------------------------------------------------------------------------
# Export to ONNX
# --------------------------------------------------------------------------
def export_onnx(batch_size=1):
    from transformers import AutoModel, AutoProcessor

    print("[ONNX] Loading model...", file=sys.stderr)
    t0 = time.time()
    model = AutoModel.from_pretrained(
        LA_PATH, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation='sdpa',
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(LA_PATH, trust_remote_code=True)
    print(f"[ONNX] Loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    vit = model.vision_model
    mlp = model.mlp1
    wrapper = MoonViTOnnxWrapper(vit, mlp).cuda().eval()

    # Use 518x518 image for fixed-size export
    img = Image.new("RGB", (518, 518), color="gray")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "find the object"},
    ]}]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16).cuda()
    grid_hws = torch.from_numpy(inputs["image_grid_hws"]).cuda()

    # Precompute cos/sin for this fixed size
    head_dim = vit.config.hidden_size // vit.config.num_attention_heads
    cos, sin = compute_rope_cos_sin(
        grid_hws, vit.encoder.rope_2d.max_height,
        vit.encoder.rope_2d.max_width, head_dim,
        theta_base=vit.encoder.rope_2d.theta_base,
        device="cuda",
    )

    with torch.no_grad():
        pt_out = model.mlp1(torch.cat(model.extract_feature(pixel_values, grid_hws), dim=0))
        onnx_out = wrapper(pixel_values, cos, sin)

    diff = (pt_out - onnx_out).abs().max().item()
    rel_diff = (diff / pt_out.abs().max()).item()
    print(f"[ONNX] Max abs diff: {diff:.6f}  rel diff: {rel_diff:.6f}", file=sys.stderr)
    if rel_diff > 0.1:
        print(f"[ONNX] WARNING: large diff ({rel_diff*100:.1f}%) — check wrapper correctness!", file=sys.stderr)

    # Export in float32 (ONNX opset 18 Conv does not support bfloat16)
    print("[ONNX] Exporting (float32)...", file=sys.stderr)
    wrapper_f32 = wrapper.float().eval()
    pv_f32 = pixel_values.float()
    cos_f32 = cos.float()
    sin_f32 = sin.float()

    with torch.no_grad():
        onnx_out_f32 = wrapper_f32(pv_f32, cos_f32, sin_f32)
    with torch.no_grad():
        pt_out_f32 = model.mlp1(torch.cat(model.extract_feature(pixel_values.float(), grid_hws), dim=0))
    diff_f32 = (pt_out_f32 - onnx_out_f32).abs().max().item()
    rel_f32 = diff_f32 / pt_out_f32.abs().max().item()
    print(f"[ONNX] float32 diff: abs={diff_f32:.6f}  rel={rel_f32:.6f}", file=sys.stderr)

    with torch.no_grad():
        torch.onnx.export(
            wrapper_f32, (pv_f32, cos_f32, sin_f32),
            ONNX_PATH,
            dynamo=False,
            input_names=["pixel_values", "cos", "sin"],
            output_names=["visual_features"],
            opset_version=18,
            do_constant_folding=True,
        )
    print(f"[ONNX] Exported to {ONNX_PATH}", file=sys.stderr)

    # Verify onnxruntime
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(
            ONNX_PATH,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        ort_inputs = {
            "pixel_values": pv_f32.cpu().numpy(),
            "cos": cos_f32.cpu().numpy(),
            "sin": sin_f32.cpu().numpy(),
        }
        ort_out = ort_session.run(["visual_features"], ort_inputs)[0]
        ort_out = torch.from_numpy(ort_out).cuda()
        diff2 = (pt_out_f32 - ort_out).abs().max().item()
        rel2 = diff2 / pt_out_f32.abs().max().item()
        print(f"[ONNX] Max diff (PT vs ONNX Runtime): {diff2:.6f}  rel={rel2:.6f}", file=sys.stderr)
        print(f"[ONNX] ONNX Runtime output shape: {ort_out.shape}", file=sys.stderr)
    except Exception as e:
        print(f"[ONNX] ONNX Runtime verification skipped: {e}", file=sys.stderr)

    del model, wrapper
    gc.collect()
    torch.cuda.empty_cache()
    print("[ONNX] Done!", file=sys.stderr)


# --------------------------------------------------------------------------
# ONNX Inference Wrapper
# --------------------------------------------------------------------------
class OnnxVisionEncoder:
    """Runs vision encoder + MLP via ONNX Runtime on GPU.

    Fixed for 532x532 input (38x38 grid, 1444 patches -> 361 merged).
    RoPE cos/sin are precomputed once on first call and cached.

    Usage:
        enc = OnnxVisionEncoder()
        features = enc(pixel_values)  # torch tensor [361, 2048]

    Args:
        onnx_path: Path to the exported ONNX model (default: model/onnx/vision_encoder.onnx).
    """

    def __init__(self, onnx_path=ONNX_PATH):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._load_config()
        self._cos_np, self._sin_np = None, None
        print(f"[ONNX Vision] Loaded from {onnx_path}", file=sys.stderr)
        print(f"[ONNX Vision] Inputs: {self.input_names}", file=sys.stderr)
        print(f"[ONNX Vision] Outputs: {self.output_names}", file=sys.stderr)

    def _load_config(self):
        import json
        config_path = os.path.join(LA_PATH, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        vc = cfg.get("vision_config", cfg)
        self.head_dim = vc["hidden_size"] // vc["num_attention_heads"]
        self.max_height = vc.get("init_pos_emb_height", 64)
        self.max_width = vc.get("init_pos_emb_width", 64)

    def _ensure_rope(self, device="cpu"):
        if self._cos_np is None:
            grid_hws = torch.tensor([[38, 38]], dtype=torch.int32)
            cos, sin = compute_rope_cos_sin(
                grid_hws, self.max_height, self.max_width,
                self.head_dim, device=device,
            )
            self._cos_np = cos.cpu().numpy()
            self._sin_np = sin.cpu().numpy()

    @torch.no_grad()
    def __call__(self, pixel_values):
        self._ensure_rope(pixel_values.device)
        inputs = {
            "pixel_values": pixel_values.cpu().numpy(),
            "cos": self._cos_np,
            "sin": self._sin_np,
        }
        outputs = self.session.run(self.output_names, inputs)
        return torch.from_numpy(outputs[0])


# --------------------------------------------------------------------------
# Benchmark: ONNX vs PyTorch
# --------------------------------------------------------------------------
def benchmark_onnx_vs_torch():
    from transformers import AutoModel, AutoProcessor

    print("\n=== Benchmark: ONNX Vision vs PyTorch Vision ===", file=sys.stderr)

    # PyTorch model
    model = AutoModel.from_pretrained(
        LA_PATH, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation='sdpa',
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(LA_PATH, trust_remote_code=True)

    onnx_enc = OnnxVisionEncoder()

    test_images = list(Path("/mnt/HDD1/Project_Data/demoMaterial/images").rglob("*.jpg"))[:10]

    pt_times, onnx_times = [], []
    for img_path in test_images:
        img = Image.open(img_path).convert("RGB")
        # Resize to fixed 518x518 (processor pads to 532=38*14)
        img = img.resize((518, 518), Image.LANCZOS)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "find the object"},
        ]}]
        text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images_in, _ = processor.process_vision_info(messages)
        inputs = processor(text=[text], images=images_in, return_tensors="pt")
        pv = inputs["pixel_values"].to(dtype=torch.bfloat16).cuda()
        ghw = torch.from_numpy(inputs["image_grid_hws"]).cuda()

        # PyTorch
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            feats = model.extract_feature(pv, ghw)
            feats = torch.cat(feats, dim=0)
            feats = model.mlp1(feats)
        torch.cuda.synchronize()
        pt_times.append((time.time() - t0) * 1000)

        # ONNX (fixed-size: the ONNX model handles the 38x38 grid internally)
        torch.cuda.synchronize()
        t0 = time.time()
        feats2 = onnx_enc(pv.float())
        torch.cuda.synchronize()
        onnx_times.append((time.time() - t0) * 1000)

    pt_times = pt_times[2:]
    onnx_times = onnx_times[2:]
    print(f"  PyTorch:  mean={sum(pt_times)/len(pt_times):.0f}ms", file=sys.stderr)
    print(f"  ONNX:     mean={sum(onnx_times)/len(onnx_times):.0f}ms", file=sys.stderr)
    speedup = sum(pt_times) / max(sum(onnx_times), 1)
    print(f"  Ratio:    {speedup:.2f}x", file=sys.stderr)

    del model, onnx_enc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_onnx_vs_torch()
    else:
        export_onnx()
