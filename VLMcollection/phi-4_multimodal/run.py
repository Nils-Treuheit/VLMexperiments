#!/usr/bin/env python3
"""
Phi-4 Multimodal -- Microsoft's multimodal model for vision-language tasks.

Supports image captioning, visual question answering, and document understanding.
Uses the Phi-4-multimodal-instruct model from HuggingFace.

Usage:
  python run.py --image path/to/img.jpg
  python run.py --image path/to/img.jpg --prompt "Describe this image"
  python run.py --image path/to/img.jpg --prompt "What does this chart show?"
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Phi-4 Multimodal Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    warnings.filterwarnings("ignore")
    image = Image.open(img_path).convert("RGB")

    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/Phi-4-multimodal-instruct"
    print(f"Loading {model_id}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    prompt = f"<|user|><|image_1|>{args.prompt}<|end|><|assistant|>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=args.max_tokens, temperature=0.7, num_logits_to_keep=0)

    response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()
