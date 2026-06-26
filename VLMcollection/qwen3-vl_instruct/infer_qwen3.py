#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import warnings

import torch
from PIL import Image, ImageDraw

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers_modules").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*causal_conv1d was requested.*")
warnings.filterwarnings("ignore", message=".*The fast path is not available.*")
warnings.filterwarnings("ignore", message=".*parameters are on the meta device.*")
warnings.filterwarnings("ignore", message=".*copy construct from a tensor.*")
warnings.filterwarnings("ignore", message=".*recommended to use sourceTensor.detach.*")
warnings.filterwarnings("ignore", message=".*_check_is_size.*")
warnings.filterwarnings("ignore", message=".*Python version.*")

from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

logging.getLogger("fla").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_vl")
FALLBACK_PATH = os.path.join(BASE_DIR, "model")


def resolve_model_path():
    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        return MODEL_PATH
    if os.path.exists(FALLBACK_PATH) and os.listdir(FALLBACK_PATH):
        return FALLBACK_PATH
    return MODEL_PATH


def get_model_class(model_path):
    import json
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    arch = config.get("architectures", [""])[0]
    if arch == "Qwen3VLMoeForConditionalGeneration":
        from transformers.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
        return Qwen3VLMoeForConditionalGeneration
    return Qwen3VLForConditionalGeneration


class Qwen3Worker:
    def __init__(self, model_path, device=None):
        self.model, self.processor = load_model(model_path, device=device)
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def predict(self, image, query, max_new_tokens=1024, temperature=0.7,
                top_p=0.8, top_k=20, enable_thinking=False):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        inputs = self.processor(images=image, text=text, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=(temperature > 0),
        )
        output = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return output


def parse_boxes(text):
    boxes = []
    for match in re.finditer(r'<box>(.+?)</box>', text):
        coords = [float(p) for p in re.findall(r'[\d.]+', match.group(1))]
        if len(coords) in (2, 4):
            boxes.append(coords)
    return boxes


def draw_boxes(image, boxes, output_path):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        if len(box) == 4:
            draw.rectangle(box, outline="red", width=3)
        elif len(box) == 2:
            x, y = box
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="red")
    image.save(output_path)


def load_model(model_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model_class = get_model_class(model_path)
    is_fp8 = (model_class.__name__ == "Qwen3VLMoeForConditionalGeneration")

    load_kwargs = {"trust_remote_code": True}

    if device == "cuda":
        import transformers.modeling_utils as mu
        original_warmup = mu.caching_allocator_warmup
        mu.caching_allocator_warmup = lambda *args, **kwargs: None
        load_kwargs["device_map"] = "cuda"
        if not is_fp8:
            load_kwargs["torch_dtype"] = torch.bfloat16
        model = model_class.from_pretrained(model_path, **load_kwargs)
        mu.caching_allocator_warmup = original_warmup
    else:
        load_kwargs["device_map"] = "cpu"
        model = model_class.from_pretrained(model_path, **load_kwargs)

    model.eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL: Multimodal inference on any image"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("query", nargs="?", default="Describe this image in detail.",
                        help="Text query (default: describe the image)")
    parser.add_argument("--output", "-o", help="Output image path (draws boxes if detected)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--device", default=None, help="Device override ('cpu', 'cuda', 'auto')")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    model_path = resolve_model_path()
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"Error: model not found at {model_path}", file=sys.stderr)
        print("Run: hf download Qwen/Qwen3-VL-8B-Instruct --local-dir model_vl", file=sys.stderr)
        sys.exit(1)

    model, processor = load_model(model_path, device=args.device)

    image = Image.open(args.image).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.query},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": not args.no_thinking}
    )

    inputs = processor(images=image, text=text, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=(args.temperature > 0),
        )
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    boxes = parse_boxes(output_text)

    if args.json:
        print(json.dumps({"text": output_text, "boxes": boxes, "image": args.image}))
    else:
        print(output_text)

    if args.output and boxes:
        draw_boxes(image.copy(), boxes, args.output)
        print(f"Output saved to: {args.output}", file=sys.stderr)

    if not args.json:
        if boxes:
            print(f"\nDetected {len(boxes)} object(s)", file=sys.stderr)
        else:
            print("\n(No bounding boxes in response)", file=sys.stderr)


if __name__ == "__main__":
    main()
