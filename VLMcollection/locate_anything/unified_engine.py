#!/usr/bin/env python3
"""
Unified Inference Engine: LocateAnything + YOLO26 + SigLIP2

Provides a single interface for:
- Fast object detection (YOLO26 backbone, millisecond range)
- Visual grounding (LocateAnything VLM, ~100ms)
- Scene understanding (SigLIP2 classifier, ~15ms)
- Unified routing: auto-selects best model per query

Usage:
  python unified_engine.py image.jpg "find the red car" --output result.jpg
  python unified_engine.py image.jpg --task detect --json
  python unified_engine.py --benchmark
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
from typing import List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

for name in ("transformers_modules", "urllib3", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

LA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
YOLO_PATH = "/mnt/HDD1/Project_Code/vlm_det_test/yolo11-26/models/yolo26m.pt"
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
SCENE_LABELS = [
    "indoor scene", "outdoor scene", "city street", "park", "beach", "mountain",
    "forest", "office", "kitchen", "bedroom", "living room", "bathroom", "classroom",
    "restaurant", "store", "hospital", "airport", "stadium", "farm", "desert",
]


# ---------------------------------------------------------------------------
# Core: LocateAnything VLM (optimized)
# ---------------------------------------------------------------------------
class LocateAnythingEngine:
    """Optimized LocateAnything VLM for visual grounding."""

    def __init__(self, compile_llm=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self._load(compile_llm)

    def _load(self, compile_llm):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            LA_PATH, trust_remote_code=True, fix_mistral_regex=True)
        self.processor = AutoProcessor.from_pretrained(
            LA_PATH, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            LA_PATH, dtype=self.dtype, trust_remote_code=True,
            attn_implementation='sdpa',
        ).to(self.device).eval()
        print(f"[LA] Loaded ({time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/1024**2:.0f}MB)",
              file=sys.stderr)
        if compile_llm and hasattr(torch, 'compile'):
            try:
                import os
                os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'
                torch._inductor.config.triton.cudagraphs = False
                self.model.language_model = torch.compile(
                    self.model.language_model, mode="max-autotune-no-cudagraphs",
                    fullgraph=False)
            except Exception as e:
                print(f"[LA] Compile warning: {e}", file=sys.stderr)

    @torch.no_grad()
    def ground(self, image: Image.Image, query: str,
               max_new_tokens=64, temperature=0, verbose=False) -> str:
        """Visual grounding: locate objects described in query."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query},
        ]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"].to(self.dtype)

        response = self.model.generate(
            pixel_values=pixel_values, input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer, max_new_tokens=max_new_tokens,
            use_cache=True, generation_mode="fast",
            temperature=temperature, do_sample=(temperature > 0),
            top_p=0.9, repetition_penalty=1.1, verbose=verbose,
        )
        return response[0] if isinstance(response, tuple) else response


# ---------------------------------------------------------------------------
# Fast detector: YOLO26
# ---------------------------------------------------------------------------
class YOLODetector:
    """YOLO26 for millisecond-range object detection."""

    def __init__(self):
        sys.path.insert(0, '/mnt/HDD1/Project_Code/vlm_det_test/yolo11-26')
        from ultralytics import YOLO
        t0 = time.time()
        self.model = YOLO(YOLO_PATH)
        print(f"[YOLO] Loaded ({time.time()-t0:.1f}s)", file=sys.stderr)

    @torch.no_grad()
    def detect(self, image: Image.Image, conf=0.25, iou=0.45,
               classes=None) -> List[dict]:
        """Run YOLO detection. Returns list of {bbox, confidence, class_id, class_name}."""
        results = self.model.predict(
            source=image, imgsz=640, conf=conf, iou=iou,
            device='0', verbose=False, classes=classes,
        )[0]
        dets = []
        if results.boxes is not None:
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                dets.append({
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "class_name": COCO_LABELS[int(cls)] if int(cls) < len(COCO_LABELS) else str(int(cls)),
                })
        return dets


# ---------------------------------------------------------------------------
# Scene describer: SigLIP2
# ---------------------------------------------------------------------------
class SigLIPSceneDescriber:
    """SigLIP2 for scene understanding and attribute classification."""

    def __init__(self, model_name="google/siglip2-base-patch16-224"):
        from transformers import AutoModel, AutoProcessor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map=self.device, attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Build label texts
        self.all_labels = COCO_LABELS + SCENE_LABELS + [
            "daytime", "nighttime", "sunny", "rainy", "snowy", "dark", "bright",
            "crowded", "empty", "natural lighting", "artificial lighting",
            "close-up shot", "wide angle shot", "blurry", "sharp",
        ]
        self.label_texts = [f"This is a photo of {l}." for l in self.all_labels]
        print(f"[SigLIP] Loaded ({time.time()-t0:.1f}s)", file=sys.stderr)

    @torch.no_grad()
    def describe(self, image: Image.Image, top_k=8) -> List[dict]:
        """Return top-k predictions with labels, probabilities, and categories."""
        inputs = self.processor(
            text=self.label_texts, images=image,
            padding="max_length", max_length=64, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        outputs = self.model(**inputs)
        probs = torch.sigmoid(outputs.logits_per_image)
        top_probs, top_indices = probs[0].topk(top_k)

        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            label = self.all_labels[idx]
            category = ("object" if label in COCO_LABELS
                       else "scene" if label in SCENE_LABELS
                       else "attribute")
            results.append({"label": label, "probability": round(prob, 4), "category": category})
        return results

    def describe_text(self, image: Image.Image, top_k=8) -> str:
        """Return human-readable description text."""
        preds = self.describe(image, top_k)
        obj_lines, scene_lines, attr_lines = [], [], []
        for p in preds:
            if p["category"] == "object":
                obj_lines.append(f"{p['label']} ({p['probability']:.1%})")
            elif p["category"] == "scene":
                scene_lines.append(f"{p['label']} ({p['probability']:.1%})")
            else:
                attr_lines.append(f"{p['label']} ({p['probability']:.1%})")
        parts = []
        if obj_lines:
            parts.append("Objects: " + ", ".join(obj_lines[:6]) + ".")
        if scene_lines:
            parts.append("Scene: " + scene_lines[0] + ".")
        if attr_lines:
            parts.append("Attributes: " + ", ".join(attr_lines[:4]) + ".")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Unified Engine
# ---------------------------------------------------------------------------
class UnifiedEngine:
    """Single interface to all three models with automatic query routing."""

    DETECT_PATTERNS = [
        r'^(find|detect|locate|spot|where is|show me)\s',
        r'^(count|how many)\s',
    ]
    GROUND_PATTERNS = [
        r'(next to|beside|behind|in front of|on top of|under|below)',
        r'(left|right|top|bottom|corner)',
        r'(red|blue|green|yellow|white|black|large|small)',
        r'(and|with|holding|wearing|carrying)',
    ]
    SCENE_PATTERNS = [
        r'^(describe|what|scene|weather|time|attribute)',
        r'(indoor|outdoor|inside|outside)',
        r'(day|night|sunny|rainy|dark|bright)',
    ]

    def __init__(self, use_yolo=True, use_siglip=True, compile_llm=True):
        self.la = None
        self.yolo = None
        self.siglip = None
        self.use_yolo = use_yolo
        self.use_siglip = use_siglip
        self.compile_llm = compile_llm
        # Lazy load on first use

    def _get_la(self):
        if self.la is None:
            self.la = LocateAnythingEngine(compile_llm=self.compile_llm)
        return self.la

    def _get_yolo(self):
        if self.yolo is None and self.use_yolo:
            self.yolo = YOLODetector()
        return self.yolo

    def _get_siglip(self):
        if self.siglip is None and self.use_siglip:
            self.siglip = SigLIPSceneDescriber()
        return self.siglip

    def predict(self, image: Image.Image, query: str, **kwargs) -> dict:
        """Run inference with auto-routing. Returns dict with results and timing."""
        ql = query.lower()

        # Determine task type
        is_detect = any(re.match(p, ql) for p in self.DETECT_PATTERNS)
        is_scene = any(re.match(p, ql) for p in self.SCENE_PATTERNS)
        is_ground = any(re.search(p, ql) for p in self.GROUND_PATTERNS)
        is_simple_obj = ql.strip() in [l.lower() for l in COCO_LABELS]

        result = {"query": query}
        timings = {}

        # Route: Fast detection (YOLO)
        if is_detect or is_simple_obj:
            yolo = self._get_yolo()
            if yolo:
                t0 = time.time()
                dets = yolo.detect(image)
                timings["yolo"] = (time.time() - t0) * 1000
                result["detections"] = dets
                result["method"] = "yolo"

        # Route: Scene description (SigLIP2)
        if is_scene:
            siglip = self._get_siglip()
            if siglip:
                t0 = time.time()
                result["scene_description"] = siglip.describe_text(image)
                result["scene_predictions"] = siglip.describe(image)
                timings["siglip"] = (time.time() - t0) * 1000
                if "method" not in result:
                    result["method"] = "siglip"

        # Route: Visual grounding (LocateAnything)
        if is_ground or (not is_detect and not is_scene):
            la = self._get_la()
            t0 = time.time()
            result["grounding"] = la.ground(image, query, **kwargs)
            timings["la"] = (time.time() - t0) * 1000
            if "method" not in result:
                result["method"] = "locateanything"

        # Default: if nothing matched, use LA
        if not result.get("grounding") and not result.get("detections"):
            la = self._get_la()
            t0 = time.time()
            result["grounding"] = la.ground(image, query, **kwargs)
            timings["la"] = (time.time() - t0) * 1000
            result["method"] = "locateanything"

        result["timings_ms"] = timings
        result["total_ms"] = round(sum(timings.values()), 1)
        return result

    def warmup(self):
        """Warm up all models."""
        img = Image.new("RGB", (224, 224), color="gray")
        if self.use_yolo:
            yolo = self._get_yolo()
            if yolo:
                yolo.detect(img)
        if self.use_siglip:
            siglip = self._get_siglip()
            if siglip:
                siglip.describe(img)
        la = self._get_la()
        la.ground(img, "find the object", max_new_tokens=8)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    """Run comprehensive benchmark on demo/COCO images."""
    print("=" * 60, file=sys.stderr)
    print("UNIFIED ENGINE BENCHMARK", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Find test images
    test_images = []
    for d in [
        "/mnt/HDD1/Project_Data/demoMaterial/images",
        "/mnt/HDD1/Project_Data/public_datasets/coco/val2017",
    ]:
        if os.path.isdir(d):
            if d.endswith("val2017"):
                test_images.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg")][:50])
            else:
                for root, _, files in os.walk(d):
                    test_images.extend([os.path.join(root, f) for f in files if f.endswith(".jpg")])

    print(f"Found {len(test_images)} test images", file=sys.stderr)

    queries = {
        "detect": ["find the person", "detect the car", "find the dog"],
        "ground": ["find the bus next to the building", "person on the left"],
        "scene": ["describe this scene", "what is the weather"],
    }

    # Test YOLO only
    print("\n--- YOLO26 Detection ---", file=sys.stderr)
    yolo = YOLODetector()
    yolo_times = []
    for img_path in test_images[:30]:
        img = Image.open(img_path).convert("RGB")
        torch.cuda.synchronize()
        t0 = time.time()
        dets = yolo.detect(img)
        torch.cuda.synchronize()
        yolo_times.append((time.time() - t0) * 1000)
    yolo_times = yolo_times[5:]
    print(f"  Mean: {sum(yolo_times)/len(yolo_times):.0f}ms  Min: {min(yolo_times):.0f}ms  Max: {max(yolo_times):.0f}ms",
          file=sys.stderr)

    # Test SigLIP2 only
    print("\n--- SigLIP2 Scene Description ---", file=sys.stderr)
    siglip = SigLIPSceneDescriber()
    siglip_times = []
    for img_path in test_images[:30]:
        img = Image.open(img_path).convert("RGB")
        torch.cuda.synchronize()
        t0 = time.time()
        desc = siglip.describe(img)
        torch.cuda.synchronize()
        siglip_times.append((time.time() - t0) * 1000)
    siglip_times = siglip_times[5:]
    print(f"  Mean: {sum(siglip_times)/len(siglip_times):.0f}ms  Min: {min(siglip_times):.0f}ms  Max: {max(siglip_times):.0f}ms",
          file=sys.stderr)

    # Test LocateAnything grounding
    print("\n--- LocateAnything Visual Grounding ---", file=sys.stderr)
    la = LocateAnythingEngine(compile_llm=False)
    la_times = []
    for img_path in test_images[:20]:
        img = Image.open(img_path).convert("RGB")
        q = queries["detect"][hash(img_path) % len(queries["detect"])]
        torch.cuda.synchronize()
        t0 = time.time()
        result = la.ground(img, q, max_new_tokens=32, temperature=0)
        torch.cuda.synchronize()
        la_times.append((time.time() - t0) * 1000)
    la_times = la_times[2:]
    print(f"  Mean: {sum(la_times)/len(la_times):.0f}ms  Min: {min(la_times):.0f}ms  Max: {max(la_times):.0f}ms",
          file=sys.stderr)

    # Test unified engine
    print("\n--- Unified Engine (auto-route) ---", file=sys.stderr)
    engine = UnifiedEngine(compile_llm=False)
    engine_times = []
    for img_path in test_images[:30]:
        img = Image.open(img_path).convert("RGB")
        q = (queries["detect"][hash(img_path) % len(queries["detect"])] if hash(img_path) % 3 != 0
             else queries["scene"][hash(img_path) % len(queries["scene"])])
        torch.cuda.synchronize()
        t0 = time.time()
        result = engine.predict(img, q, max_new_tokens=32, temperature=0)
        torch.cuda.synchronize()
        engine_times.append(result["total_ms"])
    engine_times = engine_times[5:]
    print(f"  Mean: {sum(engine_times)/len(engine_times):.0f}ms  Min: {min(engine_times):.0f}ms  Max: {max(engine_times):.0f}ms",
          file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  YOLO26 detection:          {sum(yolo_times)/len(yolo_times):.0f}ms", file=sys.stderr)
    print(f"  SigLIP2 scene description: {sum(siglip_times)/len(siglip_times):.0f}ms", file=sys.stderr)
    print(f"  LocateAnything grounding:  {sum(la_times)/len(la_times):.0f}ms", file=sys.stderr)
    print(f"  Unified (auto-routed):     {sum(engine_times)/len(engine_times):.0f}ms", file=sys.stderr)
    print(f"  VRAM usage:                {torch.cuda.max_memory_allocated()/1024**2:.0f}MB", file=sys.stderr)


def parse_boxes(text: str) -> list:
    boxes = []
    for match in re.finditer(r'<box>(.+?)</box>', text):
        content = match.group(1).strip()
        coords = [float(p) for p in re.findall(r'[\d.]+', content)]
        if len(coords) in (2, 4):
            boxes.append(coords)
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Unified LocateAnything Engine")
    parser.add_argument("image", nargs="?", help="Input image path")
    parser.add_argument("query", nargs="?", help="Text query")
    parser.add_argument("--task", choices=["auto", "detect", "ground", "describe"], default="auto")
    parser.add_argument("--output", "-o", help="Output image with drawn boxes")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO")
    parser.add_argument("--no-siglip", action="store_true", help="Disable SigLIP")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    if not args.image:
        parser.print_help()
        return

    engine = UnifiedEngine(use_yolo=not args.no_yolo, use_siglip=not args.no_siglip, compile_llm=False)
    engine.warmup()

    img = Image.open(args.image).convert("RGB")
    result = engine.predict(img, args.query, max_new_tokens=args.max_tokens, temperature=0)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        text = result.get("grounding", "") or json.dumps(result.get("detections", []))
        print(text)
        print(f"\nMethod: {result['method']}", file=sys.stderr)
        print(f"Time: {result['total_ms']:.0f}ms", file=sys.stderr)
        for name, t in result['timings_ms'].items():
            print(f"  {name}: {t:.0f}ms", file=sys.stderr)

    # Draw boxes if requested
    if args.output:
        boxes = []
        if "detections" in result:
            for d in result["detections"]:
                boxes.append(d["bbox"])
        elif "grounding" in result:
            boxes = parse_boxes(result["grounding"])
        if boxes:
            draw = ImageDraw.Draw(img)
            for box in boxes:
                if len(box) == 4:
                    draw.rectangle(box, outline="red", width=3)
            img.save(args.output)
            print(f"Saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
