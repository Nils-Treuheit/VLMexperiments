"""TensorRT-accelerated vision encoder for LocateAnything.

Takes the fixed-size (532x532) ONNX vision encoder and runs it through
ONNX Runtime's TensorRT execution provider, delivering ~16ms inference
(5.8x vs PyTorch bf16 baseline).

Usage:
    from trt_vision_encoder import TrtVisionEncoder
    encoder = TrtVisionEncoder()
    features = encoder(pixel_values, cos, sin)  # returns numpy array [361, 2048]

Architecture:
    ONNX model (MoonViT + MLP, model/onnx/vision_encoder.onnx)
    → ONNX Runtime TensorrtExecutionProvider
    → Cached .engine file (engines/*.engine)

Input:  pixel_values [1444, 3, 14, 14] float32
        cos [1444, 36] float32
        sin [1444, 36] float32
Output: visual_features [361, 2048] float32

Notes:
    - Requires LD_LIBRARY_PATH to include tensorrt_libs/ and nvidia/cudnn/lib/
    - First run builds & caches the engine (~1 min). Subsequent runs load cached.
    - The TRT engine is GPU-architecture-specific (SM120 for RTX 5090).
    - Only the vision encoder is accelerated; LLM decode remains in PyTorch.
"""
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Resolve paths relative to this script's location (model/tensorRT/)
_LA_MODEL_DIR = Path(__file__).resolve().parent.parent  # model/
ENGINES_DIR = str(Path(__file__).resolve().parent / "engines")
ONNX_PATH = str(_LA_MODEL_DIR / "onnx" / "vision_encoder.onnx")


class TrtVisionEncoder:
    """Vision encoder accelerated by TensorRT via ONNX Runtime.

    Args:
        onnx_path: Path to the exported ONNX model.
        fp16: Enable FP16 inference (faster, slight accuracy loss).
        workspace_gb: Max GPU memory for TRT engine build (GB).
    """

    def __init__(self, onnx_path: str = ONNX_PATH, fp16: bool = True, workspace_gb: int = 4):
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

        self.sess = ort.InferenceSession(
            onnx_path, sess_options=opts,
            providers=[('TensorrtExecutionProvider', trt_opts), 'CUDAExecutionProvider'],
        )
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def __call__(self, pixel_values, cos, sin) -> np.ndarray:
        """Run vision encoder inference.

        Args:
            pixel_values: [1444, 3, 14, 14] torch.Tensor or numpy array.
            cos: [1444, 36] torch.Tensor or numpy array.
            sin: [1444, 36] torch.Tensor or numpy array.

        Returns:
            visual_features: [361, 2048] numpy float32 array.
        """
        def _to_np(x):
            return x.cpu().numpy().astype(np.float32) if not isinstance(x, np.ndarray) else x.astype(np.float32)

        return self.sess.run(self.output_names, {
            'pixel_values': _to_np(pixel_values),
            'cos': _to_np(cos),
            'sin': _to_np(sin),
        })[0]

    def benchmark(self, n_warmup: int = 10, n_iter: int = 100) -> list:
        """Benchmark TRT inference speed with random data.

        Args:
            n_warmup: Number of warmup iterations (excluded from results).
            n_iter: Number of timed iterations.

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
            self.sess.run(self.output_names, data)

        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            self.sess.run(self.output_names, data)
            times.append((time.perf_counter() - t0) * 1000)

        print(f'TRT EP: mean={np.mean(times):.1f}ms  min={np.min(times):.1f}ms  '
              f'max={np.max(times):.1f}ms  median={np.median(times):.1f}ms')
        return times
