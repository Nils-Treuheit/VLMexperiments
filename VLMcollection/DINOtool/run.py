"""
DINOtool – feature extraction and zero-shot description.

Uses the `dinotool` package to load DINOv2/v3, SigLIP, RADIO, TIPSv2,
and other ViT models for feature extraction and zero-shot classification.

Modes:
  --task describe      Zero-shot object/scene/attribute classification
  --task encode        Extract CLS token embedding as base64 numpy
  --task features      Extract local patch features + PCA (like CLI)
  --task all           Run all modes

Usage:
  python run.py --image path/to/img.jpg --task describe
  python run.py --image path/to/img.jpg --task encode
  python run.py --image path/to/img.jpg --task features
  python run.py --image path/to/img.jpg --task describe --model dinov3-b
  python run.py --image path/to/img.jpg --task describe --model siglip2

Supported model shortcuts (--model):
  dinov2-s, dinov2-b, dinov2-l, dinov2-g        (DINOv2)
  dinov3-s, dinov3-b, dinov3-l, dinov3-hplus     (DINOv3)
  siglip1, siglip2, siglip2-b16-256, siglip2-b16-512  (SigLIP)
  clip                                             (OpenCLIP)
  radio-b, radio-l, radio-h, radio-g              (RADIO)
  tipsv2-b, tipsv2-l, tipsv2-so400m, tipsv2-g     (TIPSv2)
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from dinotool_wrapper import DINoToolWorker

MODEL_SHORTCUTS = list(DINoToolWorker.available_models())


def main():
    parser = argparse.ArgumentParser(description="DINOtool – ViT feature extraction & description")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--task", type=str, default="describe",
                        choices=["describe", "encode", "features", "all"],
                        help="Task type (default: describe)")
    parser.add_argument("--model", type=str, default="dinov2_vits14_reg",
                        choices=MODEL_SHORTCUTS,
                        help="Model shortcut (default: dinov2_vits14_reg)")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k labels for describe")
    parser.add_argument("--labels-file", type=str, default=None,
                        help="JSON file with custom labels (format: {\"labels\": [...], \"prompt_template\": \"...\"})")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda)")

    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(json.dumps({"error": f"Image not found: {img_path}"}))
        sys.exit(1)

    label_overrides = None
    prompt_template = None
    if args.labels_file:
        with open(args.labels_file) as f:
            ldata = json.load(f)
        label_overrides = ldata["labels"]
        prompt_template = ldata.get("prompt_template")

    worker = DINoToolWorker(model_name=args.model, device=args.device,
                            label_overrides=label_overrides, prompt_template=prompt_template)
    result = {"model": args.model, "image": str(img_path)}

    t0 = time.time()

    if args.task in ("describe", "all"):
        desc_text, predictions = worker.describe(str(img_path), top_k=args.top_k)
        result["description_text"] = desc_text
        result["predictions"] = predictions

    if args.task in ("encode", "all"):
        emb = worker.encode(str(img_path))
        buf = io.BytesIO()
        np.save(buf, emb)
        result["embedding_b64"] = base64.b64encode(buf.getvalue()).decode()
        result["embedding_shape"] = list(emb.shape)
        result["embedding_dtype"] = str(emb.dtype)

    if args.task in ("features", "all"):
        feat_info = worker.extract_features(str(img_path))
        result["features"] = feat_info

    result["time_s"] = round(time.time() - t0, 3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
