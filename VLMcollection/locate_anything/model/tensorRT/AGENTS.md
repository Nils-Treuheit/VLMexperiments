# LocateAnything TensorRT — Developer Guide

## Setup
- uv venv at `.venv/`
- Activate: `source setup_env.sh`
- `LD_LIBRARY_PATH` must include: `venv/.../tensorrt_libs`, `~/.local/.../nvidia/cudnn/lib`, `/usr/local/cuda-12.8/lib64`

## Installed Packages
- `tensorrt-cu12==10.16.1.11` (TRT 10.16, provides `libnvinfer.so.10`)
- `tensorrt-cu12-bindings==10.16.1.11`
- `tensorrt-cu12-libs==10.16.1.11`
- `onnxruntime-gpu==1.23.2`

## Files
| File | What |
|------|------|
| `setup_env.sh` | Source this to activate env + set LD path |
| `convert_onnx_to_trt.py` | Build TRT engine from ONNX + benchmark. Run with `--fp32` for reference. |
| `trt_vision_encoder.py` | `TrtVisionEncoder` class. Import from external code. |
| `engines/` | Cached TRT engines (fp16: 828MB, fp32: 1.6GB) |
| `requirements.txt` | Full pip freeze |

## Performance (Vision Encoder, RTX 5090)

| Method | Mean | Min | Max | vs PT |
|--------|------|-----|-----|-------|
| PyTorch bf16 | 94.2ms | 58.1ms | 157.1ms | 1x |
| ONNX CUDA fp32 | 72.7ms | 58.9ms | 98.1ms | 1.3x |
| **TRT EP fp16** | **16.2ms** | **10.0ms** | **26.0ms** | **5.8x** |
| TRT EP fp32 | 49.6ms | 42.2ms | 68.6ms | 1.9x |

## Notes
- TRT engine is for SM120 (Blackwell RTX 5090). Rebuild needed for other GPUs.
- Engine build needs ~300MB extra GPU memory. Runtime <100MB.
- ONNX model is fixed-size 532x532. The process: image resized to 518x518, processor pads to 532x532 (38x38 patches of 14x14).
- Input: pixel_values [1444, 3, 14, 14], cos [1444, 36], sin [1444, 36]. Output: [361, 2048].
- Only vision encoder is in TRT; LLM decoder stays in PyTorch.
- The `do_constant_folding=True` ONNX export creates external weight files named by initializer name.

## Common Issues
- **libnvinfer.so.10 not found**: `LD_LIBRARY_PATH` not set. Run `source setup_env.sh`.
- **libcudnn.so.9 not found**: Same fix as above (add nvidia/cudnn/lib path).
- **OOM during engine build**: Other processes using GPU. Wait or kill them.
- **Incorrect CUDA version**: TRT 10.x works with CUDA 12.x. Verify with `python3 -c "import tensorrt; print(tensorrt.__version__)"`.
