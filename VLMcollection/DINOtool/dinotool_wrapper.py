"""
DINOtool wrapper – unified API for feature extraction and zero-shot description.

Uses `dinotool` for vision features + `sentence-transformers` for text features.
"""

import base64
import io
import json
import os
import warnings
from pathlib import Path

# Disable xformers on Blackwell GPUs (compute capability ≥ 10.0)
# where pre-built xformers kernels are not yet available.
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer

from dinotool import DinoToolModel

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
DEFAULT_LABEL_PROMPTS = [f"This is a photo of {l}." for l in ALL_LABELS]


def _build_label_prompts(labels, tmpl=None):
    tmpl = tmpl or "This is a photo of {label}."
    return [tmpl.replace("{label}", l) for l in labels]


class DINoToolWorker:
    def __init__(self, model_name="dinov2_vits14_reg", device=None, text_model_name="all-MiniLM-L6-v2",
                 label_overrides=None, prompt_template=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model_name = text_model_name
        self._vision_model = None
        self._text_model = None
        self._label_embs = None
        self._label_overrides = label_overrides
        self._prompt_template = prompt_template

    def load_vision_model(self):
        if self._vision_model is not None:
            return self._vision_model
        warnings.filterwarnings("ignore", message=".*torch_dtype.*")
        self._vision_model = DinoToolModel(
            model_name=self.model_name,
            device=self.device,
            verbose=False,
        )
        return self._vision_model

    def load_text_model(self):
        if self._text_model is not None:
            return self._text_model
        self._text_model = SentenceTransformer(self.text_model_name, device=self.device)
        return self._text_model

    def get_label_embeddings(self):
        if self._label_embs is not None:
            return self._label_embs
        labels = self._label_overrides or ALL_LABELS
        label_prompts = _build_label_prompts(labels, self._prompt_template)
        text_model = self.load_text_model()
        embs = text_model.encode(label_prompts, convert_to_tensor=True, normalize_embeddings=True)
        self._label_embs = embs
        self._labels = labels
        return embs

    def _current_labels(self):
        return self._label_overrides or ALL_LABELS

    def describe(self, image_path, top_k=8):
        vision_model = self.load_vision_model()
        image = Image.open(image_path).convert("RGB")
        transform = vision_model.get_transform((224, 224)).transform
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            cls_token = vision_model(img_tensor, features="frame")
            cls_token = F.normalize(cls_token, dim=-1)

        text_embs = self.get_label_embeddings()
        labels = self._current_labels()
        cls_token = cls_token.to(text_embs.device)
        sims = (cls_token @ text_embs.T).squeeze(0)

        top_sims, top_indices = sims.topk(top_k)
        results = []
        for sim, idx in zip(top_sims.tolist(), top_indices.tolist()):
            label = labels[idx]
            category = "custom"
            if self._label_overrides is None:
                category = (
                    "object" if label in COCO_LABELS
                    else "scene" if label in SCENE_LABELS
                    else "attribute"
                )
            results.append({"label": label, "similarity": round(sim, 4), "category": category})

        if self._label_overrides is None:
            obj_lines = [f"{d['label']} ({d['similarity']:.1%})" for d in results if d["category"] == "object"]
            scene_lines = [f"{d['label']} ({d['similarity']:.1%})" for d in results if d["category"] == "scene"]
            attr_lines = [f"{d['label']} ({d['similarity']:.1%})" for d in results if d["category"] == "attribute"]
            text_parts = []
            if obj_lines:
                text_parts.append("Objects detected: " + ", ".join(obj_lines[:6]) + ".")
            if scene_lines:
                text_parts.append("Scene: " + scene_lines[0] + ".")
            if attr_lines:
                text_parts.append("Attributes: " + ", ".join(attr_lines[:4]) + ".")
            description_text = " ".join(text_parts)
        else:
            description_text = ", ".join(d["label"] for d in results[:5])

        return description_text, results

    def encode(self, image_path):
        vision_model = self.load_vision_model()
        image = Image.open(image_path).convert("RGB")
        transform = vision_model.get_transform((224, 224)).transform
        img_tensor = transform(image).unsqueeze(0)
        cls_token = vision_model(img_tensor, features="frame")
        return cls_token.cpu().numpy()

    def extract_features(self, image_path):
        vision_model = self.load_vision_model()
        image = Image.open(image_path).convert("RGB")
        transform = vision_model.get_transform((224, 224)).transform
        img_tensor = transform(image).unsqueeze(0)
        local_features = vision_model(img_tensor, features="full")
        pca_array = vision_model.pca(local_features, n_components=3)
        return {
            "features_shape": list(local_features.tensor.shape),
            "pca_shape": list(pca_array.shape),
        }

    @staticmethod
    def available_models():
        return list(DinoToolModel.available_models().keys())
