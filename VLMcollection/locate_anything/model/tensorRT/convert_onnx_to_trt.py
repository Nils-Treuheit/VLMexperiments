"""Build TensorRT engine from ONNX and benchmark.

This script loads the exported ONNX vision encoder through ONNX Runtime's
TensorRT execution provider. On first run it builds the TRT engine and caches
it to disk; subsequent runs load the cached engine.

Usage:
    python convert_onnx_to_trt.py          # FP16 (default, ~16ms)
    python convert_onnx_to_trt.py --fp32   # FP32 (~50ms)

Requires:
    - tensorrt-cu12 (TRT 10.x, installed in venv via uv)
    - onnxruntime-gpu (1.23+)
    - LD_LIBRARY_PATH pointing to tensorrt_libs/ and nvidia/cudnn/lib/,
      as set by setup_env.sh
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

_LA_MODEL_DIR = Path(__file__).resolve().parent.parent  # model/
ENGINES_DIR = str(Path(__file__).resolve().parent / "engines")
ONNX_PATH = str(_LA_MODEL_DIR / "onnx" / "vision_encoder.onnx")


def get_trt_session(onnx_path: str = ONNX_PATH, fp16: bool = True, workspace_gb: int = 4) -> ort.InferenceSession:
    """Create an ONNX Runtime session with TensorRT EP.

    Args:
        onnx_path: Path to the ONNX model file.
        fp16: Enable FP16 tensor cores (faster but may discard precision).
        workspace_gb: Max GPU memory for TensorRT engine optimization.

    Returns:
        ORT InferenceSession with TensorrtExecutionProvider and CUDA fallback.
    """
    os.makedirs(ENGINES_DIR, exist_ok=True)

    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    trt_opts = {
        'trt_fp16_enable': str(fp16).lower(),
        'trt_engine_cache_enable': 'true',
        'trt_engine_cache_path': ENGINES_DIR,
        'trt_builder_optimization_level': '3',
        'trt_max_workspace_size': str(workspace_gb * 1073741824),
    }

    return ort.InferenceSession(
        onnx_path, sess_options=opts,
        providers=[('TensorrtExecutionProvider', trt_opts), 'CUDAExecutionProvider'],
    )


def benchmark(sess: ort.InferenceSession, n_warmup: int = 10, n_iter: int = 100) -> list:
    """Benchmark TRT inference latency with random input data.

    Input dimensions: [1444, 3, 14, 14] for pixel_values,
    [1444, 36] for cos and sin (38x38 grid, head_dim/2 = 1152/32 = 36).

    Args:
        sess: ORT session with TRT EP.
        n_warmup: Iterations to exclude from timing.
        n_iter: Timed iterations.

    Returns:
        List of per-iteration latencies in milliseconds.
    """
    rng = np.random.default_rng(42)
    data = {
        'pixel_values': rng.standard_normal((1444, 3, 14, 14)).astype(np.float32),
        'cos': rng.standard_normal((1444, 36)).astype(np.float32),
        'sin': rng.standard_normal((1444, 36)).astype(np.float32),
    }

    for _ in range(n_warmup):
        sess.run(['visual_features'], data)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sess.run(['visual_features'], data)
        times.append((time.perf_counter() - t0) * 1000)

    print(f'TRT EP ({\"fp16\" if \"fp16\" in str(sess.get_providers()) else \"fp32\"}): '
          f'mean={np.mean(times):.1f}ms  min={np.min(times):.1f}ms  '
          f'max={np.max(times):.1f}ms  median={np.median(times):.1f}ms')
    return times


if __name__ == '__main__':
    fp16 = '--fp32' not in sys.argv
    print(f'Building TRT engine (fp16={fp16})...')
    sess = get_trt_session(fp16=fp16)
    print(f'Providers: {sess.get_providers()}')
    print(f'Inputs: {[(i.name, i.shape) for i in sess.get_inputs()]}')
    print(f'Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}')
    benchmark(sess)
