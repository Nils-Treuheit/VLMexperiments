# LocateAnything-3B

NVIDIA's [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) — a 3B-parameter vision-language model for visual grounding. Given an image and a text query, it returns bounding boxes or points for the described objects.

## Installation

```bash
pip install -r requirements.txt
huggingface-cli download nvidia/LocateAnything-3B --local-dir model
python infer.py path/to/image.jpg "person"
```

## Usage

### Detection

```bash
python infer.py ~/image.jpg "dog"
python infer.py ~/image.jpg "find the red car"
python infer.py ~/image.jpg "person</c>car</c>bicycle"   # chained categories
python infer.py ~/image.jpg "dog" -o output.jpg           # save annotated
python infer.py ~/image.jpg "dog" --json                  # machine-readable
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `hybrid` | Generation mode: `fast`, `hybrid`, or `slow` |
| `--output` / `-o` | None | Save annotated image to path |
| `--json` | off | Output machine-readable JSON |
| `--max-tokens` | 2048 | Maximum new tokens to generate |
| `--temperature` | 0.7 | Sampling temperature (0 = greedy) |
| `--device` | auto | Override device (`cpu`, `cuda:0`, etc.) |

### Prompt Templates

| Task | Example query |
|------|--------------|
| Multi-class detection | `"person</c>car</c>dog"` |
| Referring expression | `"find the woman in the red dress"` |
| Single instance | `"Locate a single instance that matches the following description: stop sign."` |
| Text detection | `"Detect all the text in box format."` |
| Pointing | `"Point to: the entrance."` |

## Optimized Execution

### 1. ONNX Vision Encoder (`export_onnx_vision.py`)

Exports the MoonViT vision encoder + MLP connector to ONNX for faster inference.

```bash
python export_onnx_vision.py                              # export ONNX model
python export_onnx_vision.py benchmark                    # benchmark vs PyTorch
```

**Architecture:**
- ONNX model: `model/onnx/vision_encoder.onnx` (274 KB graph + external weight files)
- Fixed-size for 532×532 input → 38×38 grid → 1444 patches → 361 merged
- Opset 18, float32 (Conv does not support bfloat16 in ONNX)
- Manual bmm attention (PyTorch 2.11 SDPA mask handling incompatible)
- Real-valued RoPE (no `view_as_complex` / `torch.polar`)

**Inputs:** `pixel_values` [1444, 3, 14, 14], `cos` [1444, 36], `sin` [1444, 36]
**Output:** `visual_features` [361, 2048]

**Class:** `OnnxVisionEncoder` — wraps ONNX Runtime session, precomputes RoPE cos/sin internally.

```python
from export_onnx_vision import OnnxVisionEncoder
enc = OnnxVisionEncoder()
features = enc(pixel_values)   # torch tensor [361, 2048]
```

### 2. TensorRT Acceleration (`infer_trt.py`)

Converts the MoonViT vision encoder + MLP connector to TensorRT via ONNX Runtime's
`TensorrtExecutionProvider`. Runs the vision encoder at ~10ms (9.4× vs PyTorch).

```bash
# Ensure LD_LIBRARY_PATH includes TensorRT libs first:
export LD_LIBRARY_PATH="$PWD/model/tensorRT/.venv/lib/python3.10/site-packages/tensorrt_libs:$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"

python infer_trt.py path/to/image.jpg "find the red car"
```

**Class:** `LocateAnythingWorkerTRT` — same API as `LocateAnythingWorker`, monkey-patches
`extract_feature` with `TrtVisionEncoder` and sets `mlp1 = nn.Identity()`.

**Architecture:**
- ONNX model: `model/onnx/vision_encoder.onnx` (274 KB graph + external weight files)
- TRT engines: `model/tensorRT/engines/` (built and cached on first use, ~828 MB for fp16)
- Fixed-size for 532×532 input → 38×38 grid → 1444 patches → 361 merged tokens
- Only the vision encoder runs in TRT; the Qwen2 LLM decoder remains in PyTorch

**Limitation:** The ONNX model is fixed for 532×532 input. Images are resized to 518×518
before processing (padded to 532×532 by the processor). Variable-size images fall
back to PyTorch encoder (not yet implemented).

**TRT conversion tools** are at `model/tensorRT/`:
```bash
cd model/tensorRT
source setup_env.sh
python convert_onnx_to_trt.py           # Build fp16 engine (~16ms)
python convert_onnx_to_trt.py --fp32    # Build fp32 engine (~50ms)
```

### 3. torch.compile

The LLM decoder can be compiled with `torch.compile` for ~10–20% speedup. Use `max-autotune-no-cudagraphs` mode to avoid CUDAGraph crashes.

### 4. Unified Engine (`unified_engine.py`)

Hybrid engine combining YOLO26 (detection), SigLIP2 (scene), and LocateAnything (grounding) with automatic query routing.

```bash
python unified_engine.py image.jpg "find the red car" --output result.jpg
python unified_engine.py --benchmark
```

## Benchmark Results

### Vision Encoder Only (532×532 → [361, 2048])

| Method | Mean (ms) | vs PyTorch |
|--------|-----------|------------|
| PyTorch bf16 | 94.2 | 1× |
| ONNX CUDA EP fp32 | 72.7 | 1.3× |
| **TRT EP fp16** | **9.6** | **9.8×** |

### Full Model (incl. LLM decode, RTX 5090)

| Model | Task | Mean (ms) |
|-------|------|-----------|
| YOLO26 | Detection | 16.8 |
| SigLIP2 | Scene description | 15.4 |
| LocateAnything | Visual grounding | 160–255 |
| LA (compiled) | Visual grounding | 136–169 |
| **LA + TRT vision** | **OD (COCO)** | **182** |
| **LA + TRT vision** | **Grounding** | **240** |
| **LA + TRT vision** | **Segmentation** | **239** |
| Hybrid (YOLO+SigLIP) | Detect + describe | 56.7 |

### Task Breakdown (LocateAnything)

| Task | Mean (ms) |
|------|-----------|
| Grounding (simple) | 196.1 |
| Grounding (positional) | 207.5 |
| Grounding (attribute) | 175.5 |
| Grounding (relational) | 172.7 |
| Detection (label only) | 210.5 |
| Pose estimation | 194.3 |
| Counting | 209.4 |
| Scene understanding | 174.0 |

## Project Structure

```
locate_anything/
├── infer.py                     # Basic inference script
├── infer_trt.py                 # TensorRT-accelerated inference
├── optimized_infer.py           # torch.compile version
├── unified_engine.py            # YOLO + SigLIP + LA hybrid
├── export_onnx_vision.py        # ONNX export + OnnxVisionEncoder
├── benchmark_all.py             # Comprehensive benchmark
├── model/
│   ├── onnx/                    # ONNX model + external weight files
│   ├── tensorRT/                # TRT conversion tools + engines
│   │   ├── trt_vision_encoder.py
│   │   ├── convert_onnx_to_trt.py
│   │   ├── engines/             # *.engine files (cached)
│   │   └── setup_env.sh
│   ├── config.json              # Model configuration
│   ├── *.safetensors            # Model weights (~7.8 GB)
│   └── ...
├── requirements.txt
└── README.md
```

## Notes

- Model licensed under **NVIDIA AI Foundation Models Community License** — non-commercial / research use only.
- The model supports images up to ~2.5K resolution and text prompts up to ~24K tokens.
- Output format: `<box>x1,y1,x2,y2</box>` for bounding boxes, `<box>x,y</box>` for points (coordinates normalized 0–1).
- TRT engines and conversion tools are in `model/tensorRT/`. Requires `LD_LIBRARY_PATH` to include TensorRT libs at runtime.
