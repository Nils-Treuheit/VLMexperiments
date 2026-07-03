"""
MoonViT Vision Encoder – standalone feature extraction and scene description.

MoonViT is a native-resolution vision encoder (Moonshot AI / Kimi-VL),
initialized from SigLIP-SO-400M. Supports dynamic image resolutions.

Modes:
  --task encode      Extract MoonViT patch features as JSON
  --task describe    Zero-shot classification -> structured text description
                     (uses SigLIP2 text encoder for candidate labels)
  --task encode+describe  Both

Usage:
  python run.py --image path/to/img.jpg --task describe
  python run.py --image path/to/img.jpg --task encode
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
from transformers import AutoModel, AutoProcessor, AutoTokenizer

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


def get_global_embedding(model, pixel_values, image_grid_hws):
    features = model(pixel_values, image_grid_hws)
    pooled = []
    for f in features:
        f = f.mean(dim=0)
        pooled.append(f)
    result = torch.stack(pooled).mean(dim=0)
    if result.dim() == 2:
        result = result.mean(dim=0)
    return result


def describe(vision_model, text_model, text_tokenizer, image, top_k=8, device="cuda"):
    processor = AutoImageProcessor.from_pretrained(
        "moonshotai/MoonViT-SO-400M", trust_remote_code=True,
    )
    inputs = processor(image, return_tensors="pt").to(device=device, dtype=vision_model.dtype)
    with torch.no_grad():
        img_emb = get_global_embedding(vision_model, inputs.pixel_values, inputs.image_grid_hws)
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


def encode(vision_model, image, device="cuda"):
    processor = AutoImageProcessor.from_pretrained(
        "moonshotai/MoonViT-SO-400M", trust_remote_code=True,
    )
    inputs = processor(image, return_tensors="pt").to(device=device, dtype=vision_model.dtype)
    with torch.no_grad():
        features = vision_model(inputs.pixel_values, inputs.image_grid_hws)
    feat_list = []
    for i, f in enumerate(features):
        feat_list.append({"index": i, "shape": list(f.shape), "values_b64": _tensor_to_b64(f)})
    global_emb = get_global_embedding(vision_model, inputs.pixel_values, inputs.image_grid_hws)
    return {
        "patch_features": feat_list,
        "embedding_b64": _tensor_to_b64(global_emb),
        "embedding_shape": list(global_emb.shape),
        "embedding_dtype": str(global_emb.dtype),
    }


def _tensor_to_b64(t):
    buf = io.BytesIO()
    np.save(buf, t.cpu().numpy())
    return base64.b64encode(buf.getvalue()).decode()


AutoImageProcessor = None

# Monkey-patch PreTrainedModel for transformers 5.x dev compat:
# _move_missing_keys_from_meta_to_device calls self.all_tied_weights_keys.keys()
# but the dev version (5.13.0.dev0) hasn't added this property to all models yet.
# We add a read/write property so post_init can assign it and other code can read it.
from transformers.modeling_utils import PreTrainedModel as _PTM
if not hasattr(_PTM, 'all_tied_weights_keys'):
    def _get_all_tied_weights_keys(self):
        if hasattr(self, '_dg_all_tied_weights_keys'):
            return self._dg_all_tied_weights_keys
        keys = self._tied_weights_keys
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        if isinstance(keys, str):
            return {keys}
        return {k: k for k in keys}

    def _set_all_tied_weights_keys(self, value):
        self._dg_all_tied_weights_keys = value if value is not None else {}

    _PTM.all_tied_weights_keys = property(_get_all_tied_weights_keys, _set_all_tied_weights_keys)


def main():
    global AutoImageProcessor
    from transformers import AutoImageProcessor

    parser = argparse.ArgumentParser(description="MoonViT Vision Encoder")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--task", type=str, default="describe",
        choices=["describe", "encode", "encode+describe"],
        help="Task type (default: describe)",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top-k labels for describe")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument(
        "--text-model", type=str, default="google/siglip2-so400m-patch14-384",
        help="SigLIP2 model for text encoding (default: google/siglip2-so400m-patch14-384, must match MoonViT's 1152-dim embedding)",
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

    print(f"Loading MoonViT vision encoder...", file=sys.stderr)
    vision_model = AutoModel.from_pretrained(
        "moonshotai/MoonViT-SO-400M",
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    ).eval().to(device)

    result = {"model": "moonshotai/MoonViT-SO-400M", "image": str(img_path)}

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

        desc = describe(vision_model, text_model, text_tokenizer, image, top_k=args.top_k, device=device)
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
        enc = encode(vision_model, image, device=device)
        result["patch_features"] = enc["patch_features"]
        result["embedding_b64"] = enc["embedding_b64"]
        result["embedding_shape"] = enc["embedding_shape"]
        result["embedding_dtype"] = enc["embedding_dtype"]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
