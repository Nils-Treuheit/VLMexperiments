#!/usr/bin/env python3
"""
TensorRT-accelerated LocateAnything inference.

Monkey-patches the MoonViT vision encoder with a TensorRT engine
(ONNX Runtime TensorrtExecutionProvider), replacing the PyTorch encoder
(~94ms → ~16ms). The LLM decoder remains in PyTorch.

Usage:
    from infer_trt import LocateAnythingWorkerTRT
    worker = LocateAnythingWorkerTRT()
    result = worker.predict(image, "find the red cup")
"""

import logging
import os
import sys
import warnings

import torch
import torch.nn as nn
from PIL import Image

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for name in ("transformers_modules", "urllib3", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*image_processor_class.*")
warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")
warnings.filterwarnings("ignore", message=".*copy construct from a tensor.*")
warnings.filterwarnings("ignore", message=".*recommended to use sourceTensor.detach.*")

from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Add TRT module to path
_LA_DIR = os.path.dirname(os.path.abspath(__file__))
_TRT_DIR = os.path.join(_LA_DIR, "model", "tensorRT")
sys.path.insert(0, _TRT_DIR)

from trt_vision_encoder import TrtVisionEncoder
from export_onnx_vision import compute_rope_cos_sin

MODEL_PATH = os.path.join(_LA_DIR, "model")


class LocateAnythingWorkerTRT:
    """LocateAnything with TensorRT-accelerated vision encoder.

    Loads the full model in PyTorch, then replaces extract_feature
    with TRT-based inference. MLP1 is set to identity since the
    TRT encoder already includes it.

    Fixed for 38x38 grid (532x532 image). Input images are resized
    to 518x518 (before processor pads to 532x532) to guarantee the
    grid dimensions. Other sizes fall back to PyTorch encoder.
    """

    def __init__(self, model_path=None, device=None, dtype=torch.bfloat16,
                 fp16=True, workspace_gb=4):
        if model_path is None:
            model_path = MODEL_PATH
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, fix_mistral_regex=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True
        ).to(device).eval()

        self._patch_vision_encoder(fp16, workspace_gb)

    def _patch_vision_encoder(self, fp16=True, workspace_gb=4):
        onnx_path = os.path.join(_LA_DIR, "model", "onnx", "vision_encoder.onnx")
        self.trt_enc = TrtVisionEncoder(onnx_path=onnx_path, fp16=fp16, workspace_gb=workspace_gb)

        self.model.mlp1 = nn.Identity()

        original_extract = self.model.extract_feature
        original_vision_model = self.model.vision_model

        def trt_extract_feature(pixel_values, image_grid_hws):
            cos, sin = compute_rope_cos_sin(
                image_grid_hws,
                original_vision_model.encoder.rope_2d.max_height,
                original_vision_model.encoder.rope_2d.max_width,
                original_vision_model.config.hidden_size // original_vision_model.config.num_attention_heads,
                theta_base=original_vision_model.encoder.rope_2d.theta_base,
                device=pixel_values.device,
            )
            out = self.trt_enc(pixel_values.float(), cos, sin)
            out = torch.from_numpy(out).to(pixel_values.device, dtype=pixel_values.dtype)
            grid_hw_list = image_grid_hws.tolist()
            vit_list = []
            idx = 0
            for h, w in grid_hw_list:
                nh, nw = h // 2, w // 2
                count = nh * nw
                vit_list.append(out[idx:idx + count])
                idx += count
            return vit_list

        self.model.extract_feature = trt_extract_feature

    def predict(self, image, question, generation_mode="hybrid",
                max_new_tokens=2048, temperature=0.7, verbose=False):
        # Resize to 518x518 to guarantee 38x38 grid (532x532 padded, 1444 patches)
        # The TRT ONNX model is fixed for this input size
        if image.size != (518, 518):
            image = image.resize((518, 518), Image.LANCZOS)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"].to(self.dtype)
        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=(temperature > 0),
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )
        return response[0] if isinstance(response, tuple) else response

    @torch.no_grad()
    def benchmark_vision_encoder(self, n_warmup=10, n_iter=100):
        from transformers import AutoProcessor as AP
        proc = AP.from_pretrained(MODEL_PATH, trust_remote_code=True)
        img = Image.new("RGB", (518, 518), color="gray")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "find the object"},
        ]}]
        txt = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = proc.process_vision_info(msgs)
        inp = proc(text=[txt], images=imgs, return_tensors="pt")
        pv = inp["pixel_values"].cuda()
        ghw = torch.from_numpy(inp["image_grid_hws"]).cuda()
        original_extract = self.model.vision_model

        from export_onnx_vision import compute_rope_cos_sin
        cos, sin = compute_rope_cos_sin(
            ghw, self.model.vision_model.encoder.rope_2d.max_height,
            self.model.vision_model.encoder.rope_2d.max_width,
            1152 // 16,
            device="cuda",
        )

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        times_trt = []
        for _ in range(n_warmup + n_iter):
            t0.record()
            out = self.trt_enc(pv.float(), cos, sin)
            t1.record()
            torch.cuda.synchronize()
            if _ >= n_warmup:
                times_trt.append(t0.elapsed_time(t1))
        mean_trt = sum(times_trt) / len(times_trt)

        print(f"TRT encoder: {mean_trt:.1f}ms avg  ({len(times_trt)} iters)")
        return mean_trt
