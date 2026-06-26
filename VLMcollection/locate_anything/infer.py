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

for name in ("transformers_modules", "urllib3", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*image_processor_class.*")
warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")
warnings.filterwarnings("ignore", message=".*copy construct from a tensor.*")
warnings.filterwarnings("ignore", message=".*recommended to use sourceTensor.detach.*")

from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


class LocateAnythingWorker:
    def __init__(self, model_path, device=None, dtype=torch.bfloat16):
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

    @torch.no_grad()
    def predict(self, image, question, generation_mode="hybrid", max_new_tokens=2048, temperature=0.7, verbose=False):
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
        output = response[0] if isinstance(response, tuple) else response
        return output


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
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 12), f"{x1:.0f},{y1:.0f}", fill="red")
        elif len(box) == 2:
            x, y = box
            r = 5
            draw.ellipse([x - r, y - r, x + r, y + r], fill="red")
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="LocateAnything-3B: Visual grounding on any image"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("query", help="Text query (e.g. 'person</c>car' or 'find the red cup')")
    parser.add_argument("--mode", choices=["fast", "hybrid", "slow"], default="hybrid", help="Generation mode")
    parser.add_argument("--output", "-o", help="Output image path (draws boxes on image)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", default=None, help="Device override (e.g. 'cpu', 'cuda:0')")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"Error: model not found at {MODEL_PATH}", file=sys.stderr)
        print("Run: hf download nvidia/LocateAnything-3B --local-dir model", file=sys.stderr)
        sys.exit(1)

    print("Loading model...", file=sys.stderr)
    worker = LocateAnythingWorker(MODEL_PATH, device=args.device)

    img = Image.open(args.image).convert("RGB")
    print("Running inference...", file=sys.stderr)
    text = worker.predict(img, args.query, generation_mode=args.mode,
                          max_new_tokens=args.max_tokens, temperature=args.temperature)

    boxes = parse_boxes(text)

    if args.json:
        print(json.dumps({"text": text, "boxes": boxes}))
    else:
        print(text)

    if args.output and boxes:
        draw_boxes(img.copy(), boxes, args.output)
        print(f"Output saved to: {args.output}", file=sys.stderr)

    if not args.json:
        if boxes:
            print(f"\nDetected {len(boxes)} object(s)", file=sys.stderr)
        else:
            print("\nNo bounding boxes detected in response", file=sys.stderr)


if __name__ == "__main__":
    main()
