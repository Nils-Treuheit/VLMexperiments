# LocateAnything — Developer Guide

## Project Structure

```
locate_anything/
├── infer.py                     # Basic inference
├── optimized_infer.py           # torch.compile version
├── unified_engine.py            # YOLO + SigLIP + LA hybrid engine
├── export_onnx_vision.py        # ONNX export + OnnxVisionEncoder class
├── benchmark_all.py             # Comprehensive benchmark suite
├── model/
│   ├── onnx/                    # ONNX model + external weight files
│   │   └── vision_encoder.onnx  # MoonViT + MLP in ONNX format
│   ├── config.json
│   ├── *.safetensors            # Model weights
│   └── ...
├── requirements.txt
├── README.md
└── AGENTS.md
```

## Key Files

### `export_onnx_vision.py`
Exports the MoonViT vision encoder + MLP to ONNX opset 18. Key components:
- `onnx_apply_rope()` — real-valued RoPE (no complex numbers, ONNX-compatible)
- `compute_rope_cos_sin()` — precompute 2D RoPE for arbitrary grid sizes
- `MoonViTOnnxWrapper` — wraps MoonViT with manual bmm attention (avoiding SDPA issues)
- `export_onnx()` — exports to ONNX with verification
- `OnnxVisionEncoder` — inference class using ONNX Runtime CUDA EP

### `unified_engine.py`
Three-model hybrid with auto-routing:
- `LocateAnythingEngine` — VLM for visual grounding
- `YOLODetector` — YOLO26 for fast detection (~17ms)
- `SigLIPSceneDescriber` — SigLIP2 for scene understanding (~15ms)
- `UnifiedEngine` — routes queries to the best model based on regex patterns

### `benchmark_all.py`
Tests all models across task categories:
- YOLO26 detection
- SigLIP2 scene description
- LA grounding (8 task types: simple, positional, attribute, relational, detection, pose, counting, scene)
- ONNX vs PyTorch vision encoder
- torch.compile comparison
- Hybrid vs LA comparison

## Performance (RTX 5090)

| Model | Task | Mean (ms) |
|-------|------|-----------|
| YOLO26 | Detection | 16.8 |
| SigLIP2 | Scene description | 15.4 |
| Vision (PyTorch bf16) | Encode | 94.2 |
| Vision (ONNX CUDA fp32) | Encode | 72.7 |
| Vision (TRT fp16) | Encode | 16.2 |
| LA | Grounding (simple) | 196.1 |
| LA (compiled) | Grounding | 136-169 |
| Hybrid | Detect+describe | 56.7 |

## Optimization Tips
- Set `max_new_tokens=32` or lower for simple detection tasks
- Use `attn_implementation='sdpa'` to avoid magi fallback overhead
- torch.compile: use `max-autotune-no-cudagraphs` mode
- For torch.compile: set `TORCHINDUCTOR_CUDAGRAPHS=0` and `torch._inductor.config.triton.cudagraphs = False`
- To reduce graph breaks: `torch._dynamo.config.capture_scalar_outputs = True`
- Redirect caches to fast SSD: `TORCHINDUCTOR_CACHE_DIR`, `HF_HOME`, `XDG_CACHE_HOME`

## TensorRT Integration
See `/mnt/HDD1/Project_Code/vlm_det_test/LocateAnything_TensorRT/` for TRT acceleration.
The vision encoder (MoonViT + MLP) runs at 16ms in TRT fp16 vs 94ms PyTorch.
