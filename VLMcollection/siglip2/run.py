"""
SigLIP2 Vision Encoder – standalone feature extraction and scene description.

Modes:
  --task encode      Extract vision embeddings as JSON (numpy base64-encoded)
  --task describe    Zero-shot classification -> structured text description
  --task encode+describe  Both

Usage:
  python run.py --image path/to/img.jpg --task describe
  python run.py --image path/to/img.jpg --task encode
  python run.py --image path/to/img.jpg --task describe --top-k 10
  python run.py --image path/to/img.jpg --task encode --model google/siglip2-large-patch16-384
"""

import argparse
import base64
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

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

ATTRIBUTE_LABELS = [
    "daytime", "nighttime", "sunny", "rainy", "snowy", "dark", "bright",
    "crowded", "empty", "natural lighting", "artificial lighting",
    "close-up shot", "wide angle shot", "blurry", "sharp",
]

ALL_LABELS = COCO_LABELS + SCENE_LABELS + ATTRIBUTE_LABELS


def describe(model, processor, image, top_k=8):
    candidate_texts = [f"This is a photo of {l}." for l in ALL_LABELS]
    inputs = processor(
        text=candidate_texts, images=image,
        padding="max_length", max_length=64, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits_per_image
    probs = torch.sigmoid(logits)
    top_probs, top_indices = probs[0].topk(top_k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = ALL_LABELS[idx]
        category = (
            "object" if label in COCO_LABELS
            else "scene" if label in SCENE_LABELS
            else "attribute"
        )
        results.append({"label": label, "probability": round(prob, 4), "category": category})
    return results


def encode(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    if hasattr(outputs, "pooler_output"):
        emb = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state"):
        emb = outputs.last_hidden_state.mean(dim=1)
    else:
        emb = outputs.logits_per_image if hasattr(outputs, "logits_per_image") else None
    if emb is None:
        raise RuntimeError(f"Cannot extract embedding from model output: {type(outputs)}")
    return emb.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="SigLIP2 Vision Encoder")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--task", type=str, default="describe",
        choices=["describe", "encode", "encode+describe"],
        help="Task type (default: describe)",
    )
    parser.add_argument(
        "--model", type=str, default="google/siglip2-base-patch16-224",
        help="SigLIP2 model variant (default: google/siglip2-base-patch16-224)",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top-k labels for describe")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. 'cpu', 'cuda')")

    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(json.dumps({"error": f"Image not found: {img_path}"}))
        sys.exit(1)

    image = Image.open(img_path).convert("RGB")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    warnings.filterwarnings("ignore", message=".*torch_dtype.*")

    model = AutoModel.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    result = {"model": args.model, "image": str(img_path)}

    if args.task in ("describe", "encode+describe"):
        desc = describe(model, processor, image, top_k=args.top_k)
        lines = []
        obj_lines = []
        scene_lines = []
        attr_lines = []
        for d in desc:
            if d["category"] == "object":
                obj_lines.append(f"{d['label']} ({d['probability']:.1%})")
            elif d["category"] == "scene":
                scene_lines.append(f"{d['label']} ({d['probability']:.1%})")
            else:
                attr_lines.append(f"{d['label']} ({d['probability']:.1%})")
        text_parts = []
        if obj_lines:
            text_parts.append("Objects detected: " + ", ".join(obj_lines[:6]) + ".")
        if scene_lines:
            text_parts.append("Scene: " + scene_lines[0] + ".")
        if attr_lines:
            text_parts.append("Attributes: " + ", ".join(attr_lines[:4]) + ".")
        result["description_text"] = " ".join(text_parts)
        result["predictions"] = desc

    if args.task in ("encode", "encode+describe"):
        emb = encode(model, processor, image)
        buf = io.BytesIO()
        np.save(buf, emb)
        result["embedding_b64"] = base64.b64encode(buf.getvalue()).decode()
        result["embedding_shape"] = list(emb.shape)
        result["embedding_dtype"] = str(emb.dtype)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
