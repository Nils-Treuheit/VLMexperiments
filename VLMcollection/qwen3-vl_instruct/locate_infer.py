#!/usr/bin/env python3
"""
LocateAnything architecture adapted for Qwen3.6-35B-A3B backbone.

This is a SCAFFOLD / EXPERIMENTAL script. The existing pretrained
LocateAnything-3B weights were trained with Qwen2.5-3B and are NOT
compatible with Qwen3.6-35B-A3B. This script shows how the architecture
would connect, but needs finetuning to produce meaningful results.

For working inference, use infer_qwen3.py instead.
"""
import argparse
import json
import os
import re
import sys

import torch
from PIL import Image, ImageDraw

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


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


class LocateAnythingQwen3:
    """
    Attempts to use the LocateAnything parallel box decoding architecture
    with Qwen3.6-35B-A3B as the language backbone.

    Currently this is experimental scaffolding. The PBD heads, MTP decoder,
    and attention modifications from the original LocateAnything were
    designed for Qwen2.5-3B and need adaptation for Qwen3.6's MoE
    architecture with Gated DeltaNet + Gated Attention.
    """

    def __init__(self, model_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_path = model_path

        from transformers import AutoProcessor
        from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, image, question, max_new_tokens=1024, temperature=0.7, top_p=0.8, top_k=20, enable_thinking=True):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        chat_kwargs = {}
        if enable_thinking:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
        inputs = {
            k: v.to(self.model.device) if hasattr(v, 'to') else v
            for k, v in inputs.items()
        }

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
            skip_special_tokens=True
        )
        return output


def main():
    parser = argparse.ArgumentParser(
        description="[EXPERIMENTAL] LocateAnything architecture + Qwen3.6 backbone"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("query", nargs="?", default="Describe this image in detail.",
                        help="Text query")
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking mode")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"Error: model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print("Loading model (this may take a few minutes)...", file=sys.stderr)
    worker = LocateAnythingQwen3(MODEL_PATH, device=args.device)

    img = Image.open(args.image).convert("RGB")
    print("Running inference...", file=sys.stderr)
    output_text = worker.predict(
        img, args.query,
        enable_thinking=not args.no_thinking,
        temperature=args.temperature,
    )

    boxes = parse_boxes(output_text)

    if args.json:
        print(json.dumps({"text": output_text, "boxes": boxes}))
    else:
        print(output_text)

    if args.output and boxes:
        draw_boxes(img.copy(), boxes, args.output)
        print(f"Output saved to: {args.output}", file=sys.stderr)

    if not args.json:
        print(f"\nDetected {len(boxes)} box(es)", file=sys.stderr)


if __name__ == "__main__":
    print("=" * 60, file=sys.stderr)
    print("EXPERIMENTAL: LocateAnything architecture with Qwen3.6", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)
    print("NOTE: This is a scaffold. The original LocateAnything-3B's", file=sys.stderr)
    print("parallel box decoding heads were trained on Qwen2.5-3B.", file=sys.stderr)
    print("Qwen3.6-35B-A3B has a different architecture (MoE with", file=sys.stderr)
    print("Gated DeltaNet + Gated Attention) and needs its own PBD", file=sys.stderr)
    print("training. This script uses Qwen3.6 natively for now.", file=sys.stderr)
    print(file=sys.stderr)
    main()
