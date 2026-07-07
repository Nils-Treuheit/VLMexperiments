# LocateAnything TensorRT

TensorRT-accelerated vision encoder for the LocateAnything-3B visual grounding model. Converts the MoonViT vision encoder + MLP connector (originally ~94ms in PyTorch) into an optimized TensorRT engine running at **~16ms** (5.8× speedup).

## Architecture

```
onnx/vision_encoder.onnx  ──►  ONNX Runtime  ──►  TensorRT EP  ──►  engine
  (MoonViT + MLP,               (TensorrtExecutionProvider)         .engine file
   fixed 532×532)
```

**Pipeline:**
1. PyTorch MoonViT + MLP → exported to ONNX via `export_onnx_vision.py`
2. ONNX Runtime loads the `.onnx` with `TensorrtExecutionProvider`
3. TRT builds an optimized engine (cached to disk for reuse)
4. Inference runs entirely through TensorRT (FP16 or FP32)

## Setup

```bash
source setup_env.sh    # activates venv + sets LD_LIBRARY_PATH
```

The venv needs the TRT libraries in `LD_LIBRARY_PATH` at runtime:

```bash
export LD_LIBRARY_PATH="$VENV_DIR/.../tensorrt_libs:$HOME/.local/.../nvidia/cudnn/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
```

## Files

| File | Purpose |
|------|---------|
| `setup_env.sh` | Source this to activate venv + set `LD_LIBRARY_PATH` |
| `convert_onnx_to_trt.py` | Build & benchmark TRT engine from ONNX |
| `trt_vision_encoder.py` | Reusable `TrtVisionEncoder` class |
| `engines/` | Cached TRT engine files (built, not regenerated) |
| `requirements.txt` | Full pip dependency freeze |

## Usage

### Build engine and benchmark

```bash
source setup_env.sh
python convert_onnx_to_trt.py          # FP16 (default, faster)
python convert_onnx_to_trt.py --fp32   # FP32 (reference)
```

First run builds the engine (~1 min, GPU-dependent). Subsequent runs load cached engine.

### Programmatic usage

```python
from trt_vision_encoder import TrtVisionEncoder
import torch

encoder = TrtVisionEncoder(fp16=True)    # loads or builds engine

# Run inference (accepts numpy or torch tensors)
features = encoder(pixel_values, cos, sin)
# features: numpy array [361, 2048] float32

# Benchmark
encoder.benchmark()
```

## Engines

Pre-built engines in `engines/`:

| Engine | Size | Precision | Latency |
|--------|------|-----------|---------|
| `*_fp16_sm120.engine` | 828 MB | FP16 | **16.2 ms** |
| `*_sm120.engine` | 1.6 GB | FP32 | 49.6 ms |

Engines are specific to the GPU architecture (SM120 = Blackwell RTX 5090). Rebuild required for different GPUs.

## Benchmark Results

### Vision Encoder (MoonViT + MLP, 532×532 → [361, 2048])

| Method | Mean (ms) | Min (ms) | Max (ms) | vs PyTorch |
|--------|-----------|----------|----------|------------|
| **PyTorch bf16** | 94.2 | 58.1 | 157.1 | 1× |
| **ONNX CUDA EP fp32** | 72.7 | 58.9 | 98.1 | 1.3× |
| **TRT EP fp32** | 49.6 | 42.2 | 68.6 | 1.9× |
| **TRT EP fp16** | **16.2** | **10.0** | **26.0** | **5.8×** |

### Full Model Comparison (incl. LLM decode)

| Model | Task | Mean (ms) |
|-------|------|-----------|
| YOLO26 | Detection | 16.8 |
| SigLIP2 | Scene description | 15.4 |
| LocateAnything (LA) | Grounding | 160–255 |
| **LA + ONNX vision** | Grounding | ~140 |
| **LA + TRT vision** | Grounding | ~120 (projected) |
| Hybrid (YOLO+SigLIP) | Detect+describe | 56.7 |

## Notes

- Only the vision encoder (MoonViT + MLP) runs in TensorRT. The LLM decoder (Qwen2) remains in PyTorch (compiled or eager).
- The ONNX model is fixed-size for 532×532 input (38×38 grid, 1444 patches → 361 merged after 2×2 kernel).
- TRT engine build requires ~300 MB extra GPU memory. Runtime uses <100 MB.
- `do_constant_folding=True` in ONNX export causes external weight files to be named by initializer. Re-export if weights are missing.
- Tested on RTX 5090 (Blackwell, SM120, CUDA 12.8). Requires TensorRT 10.x + ONNX Runtime 1.23+.
