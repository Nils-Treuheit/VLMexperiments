"""
DINOv3 Vision Encoder – standalone feature extraction and scene description.

DINOv3 is a self-supervised vision foundation model (Meta AI) producing
high-quality dense features. Available as ViT and ConvNeXt variants.

Modes:
  --task encode      Extract DINOv3 vision embeddings as JSON
  --task describe    Zero-shot classification -> structured text description
                     (uses SigLIP2 text encoder for candidate labels)
  --task encode+describe  Both

Usage:
  python run.py --image path/to/img.jpg --task describe
  python run.py --image path/to/img.jpg --task encode
  python run.py --image path/to/img.jpg --task encode --model facebook/dinov3-vitb16-pretrain-lvd1689m
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
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

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
LABEL_PROMPTS = [f"This is a photo of {l}." for l in ALL_LABELS]


def describe(vision_model, processor, text_model, text_tokenizer, image, top_k=8, device="cpu"):
    inputs = processor(images=image, return_tensors="pt").to(device=device, dtype=vision_model.dtype)
    with torch.no_grad():
        outputs = vision_model(**inputs)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        img_emb = outputs.pooler_output
    else:
        img_emb = outputs.last_hidden_state[:, 0, :]
    img_emb = F.normalize(img_emb, dim=-1)

    text_inputs = text_tokenizer(
        LABEL_PROMPTS, padding="max_length", max_length=64,
        return_tensors="pt", truncation=True,
    ).to(device)
    with torch.no_grad():
        text_outputs = text_model.text_model(**text_inputs)
        if hasattr(text_outputs, "pooler_output"):
            text_embs = text_outputs.pooler_output
        else:
            text_embs = text_outputs.last_hidden_state.mean(dim=1)
        text_embs = F.normalize(text_embs, dim=-1)

    sims = (img_emb @ text_embs.T).squeeze(0)
    top_sims, top_indices = sims.topk(top_k)
    results = []
    for sim, idx in zip(top_sims.tolist(), top_indices.tolist()):
        label = ALL_LABELS[idx]
        category = (
            "object" if label in COCO_LABELS
            else "scene" if label in SCENE_LABELS
            else "attribute"
        )
        results.append({"label": label, "similarity": round(sim, 4), "category": category})
    return results


def encode(vision_model, processor, image, device="cpu"):
    inputs = processor(images=image, return_tensors="pt").to(device=device, dtype=vision_model.dtype)
    with torch.no_grad():
        outputs = vision_model(**inputs)
    return outputs


def _tensor_to_b64(t):
    buf = io.BytesIO()
    np.save(buf, t.cpu().numpy())
    return base64.b64encode(buf.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description="DINOv3 Vision Encoder")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--task", type=str, default="describe",
        choices=["describe", "encode", "encode+describe"],
        help="Task type (default: describe)",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top-k labels for describe")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument(
        "--model", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="DINOv3 model variant (default: facebook/dinov3-vits16-pretrain-lvd1689m)",
    )
    parser.add_argument(
        "--text-model", type=str, default="google/siglip2-so400m-patch14-384",
        help="SigLIP2 model for text encoding (default: google/siglip2-so400m-patch14-384)",
    )

    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(json.dumps({"error": f"Image not found: {img_path}"}))
        sys.exit(1)

    image = Image.open(img_path).convert("RGB")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    warnings.filterwarnings("ignore", message=".*torch_dtype.*")

    print(f"Loading DINOv3 vision encoder ({args.model})...", file=sys.stderr)
    vision_model = AutoModel.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa",
    ).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(args.model)

    result = {"model": args.model, "image": str(img_path)}

    if args.task in ("describe", "encode+describe"):
        print(f"Loading SigLIP2 text encoder ({args.text_model})...", file=sys.stderr)
        if args.text_model == "google/siglip2-so400m-patch14-384":
            attn_impl = None
        else:
            attn_impl = "sdpa"
        text_model = AutoModel.from_pretrained(
            args.text_model, torch_dtype=dtype,
            attn_implementation=attn_impl,
        ).eval().to(device)
        text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)

        desc = describe(vision_model, processor, text_model, text_tokenizer,
                        image, top_k=args.top_k, device=device)
        lines = []
        obj_lines = []
        scene_lines = []
        attr_lines = []
        for d in desc:
            if d["category"] == "object":
                obj_lines.append(f"{d['label']} ({d['similarity']:.1%})")
            elif d["category"] == "scene":
                scene_lines.append(f"{d['label']} ({d['similarity']:.1%})")
            else:
                attr_lines.append(f"{d['label']} ({d['similarity']:.1%})")
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
        outputs = encode(vision_model, processor, image, device=device)
        hidden = outputs.last_hidden_state
        result["embedding_b64"] = _tensor_to_b64(hidden)
        result["embedding_shape"] = list(hidden.shape)
        result["embedding_dtype"] = str(hidden.dtype)
        if hasattr(outputs, "pooler_output"):
            result["pooler_b64"] = _tensor_to_b64(outputs.pooler_output)
            result["pooler_shape"] = list(outputs.pooler_output.shape)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
