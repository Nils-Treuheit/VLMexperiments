#!/usr/bin/env python3
"""
Optimized LocateAnything inference engine.
Three-tier: fast (compiled), full, and hybrid YOLO+LocateAnything integration.
"""
import argparse
import gc
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for name in ("transformers_modules", "urllib3", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


class LocateAnythingOptimized:
    """Optimized LocateAnything with torch.compile support."""
    
    def __init__(self, model_path=None, device=None, dtype=torch.bfloat16,
                 compile_llm=True, compile_vision=False, compile_mlp=True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        model_path = model_path or MODEL_PATH
        
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, fix_mistral_regex=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True,
            attn_implementation='sdpa',
        ).to(device).eval()
        print(f"[LA] Loaded in {time.time()-t0:.1f}s", file=sys.stderr)
        
        if compile_llm and torch.cuda.is_available() and hasattr(torch, 'compile'):
            self._compile_llm()
        if compile_mlp and torch.cuda.is_available() and hasattr(torch, 'compile'):
            self._compile_mlp()
    
    def _compile_llm(self):
        print("[LA] Compiling LLM decoder...", file=sys.stderr)
        t0 = time.time()
        try:
            import os
            os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'
            torch._inductor.config.triton.cudagraphs = False
            self.model.language_model = torch.compile(
                self.model.language_model, mode="max-autotune-no-cudagraphs",
                fullgraph=False,
            )
            print(f"[LA] LLM compiled in {time.time()-t0:.0f}s", file=sys.stderr)
        except Exception as e:
            print(f"[LA] LLM compile failed: {e}", file=sys.stderr)
    
    def _compile_mlp(self):
        try:
            self.model.mlp1 = torch.compile(
                self.model.mlp1, mode="reduce-overhead", fullgraph=True,
            )
        except Exception:
            pass
    
    @torch.no_grad()
    def predict(self, image, query, generation_mode="fast", max_new_tokens=64,
                temperature=0, verbose=False, **kwargs):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
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
    
    def extract_vision_features(self, image):
        """Run vision encoder only (for fast path)."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "describe"},
        ]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"].to(self.dtype)
        return self.model.extract_feature(pixel_values, inputs.get("image_grid_hws", None))
    
    def warmup(self, image, query="find the object", n=2):
        for _ in range(n):
            _ = self.predict(image, query, max_new_tokens=16, temperature=0)


def parse_boxes(text):
    boxes = []
    for match in re.finditer(r'<box>(.+?)</box>', text):
        content = match.group(1).strip()
        coords = [float(p) for p in re.findall(r'[\d.]+', content)]
        if len(coords) in (2, 4):
            boxes.append(coords)
    return boxes


def draw_boxes(image, boxes, output_path):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        if len(box) == 4:
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1] - 12), f"{box[0]:.0f},{box[1]:.0f}", fill="red")
        elif len(box) == 2:
            r = 5
            draw.ellipse([box[0]-r, box[1]-r, box[0]+r, box[1]+r], fill="red")
    image.save(output_path)


def benchmark_warm():
    """Run comprehensive benchmark on test images."""
    test_images = []
    demo_dir = Path("/mnt/HDD1/Project_Data/demoMaterial/images")
    if demo_dir.exists():
        for subdir in demo_dir.iterdir():
            if subdir.is_dir():
                test_images.extend(subdir.glob("*.jpg"))
    coco_val = Path("/mnt/HDD1/Project_Data/public_datasets/coco/val2017")
    if coco_val.exists():
        test_images.extend(list(coco_val.glob("*.jpg"))[:50])
    
    if not test_images:
        print("No test images found!", file=sys.stderr)
        return
    
    print(f"Found {len(test_images)} test images", file=sys.stderr)
    queries = ["find the person", "find the car", "find the dog", "what objects are in this image"]
    
    # Test without compile
    print("\n--- Baseline (no compile) ---", file=sys.stderr)
    la_base = LocateAnythingOptimized(compile_llm=False, compile_mlp=False)
    img = Image.open(test_images[0]).convert("RGB")
    la_base.warmup(img)
    
    baseline_times = []
    for img_path in test_images[:10]:
        img = Image.open(img_path).convert("RGB")
        q = queries[hash(str(img_path)) % len(queries)]
        torch.cuda.synchronize()
        t0 = time.time()
        r = la_base.predict(img, q, max_new_tokens=32, temperature=0)
        torch.cuda.synchronize()
        baseline_times.append((time.time() - t0) * 1000)
    baseline_times = baseline_times[2:]
    mean_b = sum(baseline_times) / len(baseline_times)
    print(f"Baseline:  mean={mean_b:.0f}ms  min={min(baseline_times):.0f}ms", file=sys.stderr)
    del la_base; gc.collect(); torch.cuda.empty_cache()
    
    # Test with compile
    print("\n--- With torch.compile ---", file=sys.stderr)
    la_opt = LocateAnythingOptimized(compile_llm=True, compile_mlp=True)
    img = Image.open(test_images[0]).convert("RGB")
    la_opt.warmup(img)
    
    opt_times = []
    for img_path in test_images[:10]:
        img = Image.open(img_path).convert("RGB")
        q = queries[hash(str(img_path)) % len(queries)]
        torch.cuda.synchronize()
        t0 = time.time()
        r = la_opt.predict(img, q, max_new_tokens=32, temperature=0)
        torch.cuda.synchronize()
        opt_times.append((time.time() - t0) * 1000)
    opt_times = opt_times[2:]
    mean_o = sum(opt_times) / len(opt_times)
    print(f"Optimized: mean={mean_o:.0f}ms  min={min(opt_times):.0f}ms", file=sys.stderr)
    
    speedup = mean_b / mean_o if mean_o > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x", file=sys.stderr)
    
    # Test YOLO26
    print("\n--- YOLO26 benchmark ---", file=sys.stderr)
    try:
        sys.path.insert(0, '/mnt/HDD1/Project_Code/vlm_det_test/yolo11-26')
        from ultralytics import YOLO
        yolo = YOLO(str(Path("/mnt/HDD1/Project_Code/vlm_det_test/yolo11-26/models/yolo26m.pt")))
        yolo_times = []
        for img_path in test_images[:10]:
            torch.cuda.synchronize()
            t0 = time.time()
            _ = yolo.predict(source=str(img_path), imgsz=640, conf=0.25, iou=0.45, device='0', verbose=False)
            torch.cuda.synchronize()
            yolo_times.append((time.time() - t0) * 1000)
        yolo_times = yolo_times[2:]
        mean_y = sum(yolo_times) / len(yolo_times)
        print(f"YOLO26:    mean={mean_y:.0f}ms  min={min(yolo_times):.0f}ms", file=sys.stderr)
    except Exception as e:
        print(f"YOLO26 benchmark failed: {e}", file=sys.stderr)
    
    # Test SigLIP2
    print("\n--- SigLIP2 benchmark ---", file=sys.stderr)
    try:
        sys.path.insert(0, '/mnt/HDD1/Project_Code/vlm_det_test/siglip2')
        from transformers import AutoModel as HFAutoModel, AutoProcessor as HFAutoProcessor
        siglip = HFAutoModel.from_pretrained("google/siglip2-base-patch16-224",
            torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa").eval()
        siglip_proc = HFAutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        siglip_times = []
        for img_path in test_images[:10]:
            img = Image.open(img_path).convert("RGB")
            inputs = siglip_proc(images=img, return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                _ = siglip(**inputs)
            torch.cuda.synchronize()
            siglip_times.append((time.time() - t0) * 1000)
        siglip_times = siglip_times[2:]
        mean_s = sum(siglip_times) / len(siglip_times)
        print(f"SigLIP2:   mean={mean_s:.0f}ms  min={min(siglip_times):.0f}ms", file=sys.stderr)
    except Exception as e:
        print(f"SigLIP2 benchmark failed: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Optimized LocateAnything")
    parser.add_argument("image", nargs="?", help="Input image")
    parser.add_argument("query", nargs="?", help="Text query")
    parser.add_argument("--mode", choices=["fast", "hybrid", "slow"], default="fast")
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_warm()
        return
    
    if not args.image or not args.query:
        parser.print_help()
        return
    
    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    
    la = LocateAnythingOptimized(compile_llm=not args.no_compile, device=args.device)
    img = Image.open(args.image).convert("RGB")
    
    torch.cuda.synchronize()
    t0 = time.time()
    text = la.predict(img, args.query, generation_mode=args.mode,
                      max_new_tokens=args.max_tokens, temperature=args.temperature)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) * 1000
    
    boxes = parse_boxes(text)
    
    if args.json:
        print(json.dumps({"text": text, "boxes": boxes, "time_ms": round(elapsed, 1)}))
    else:
        print(text)
        print(f"\nTime: {elapsed:.0f}ms", file=sys.stderr)
        if boxes:
            print(f"Detected {len(boxes)} object(s)", file=sys.stderr)
    
    if args.output and boxes:
        draw_boxes(img.copy(), boxes, args.output)
        print(f"Output saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
