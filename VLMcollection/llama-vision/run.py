import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

CUDA_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "diffusion_gemma_vl", ".venv", "lib", "python3.13",
    "site-packages", "nvidia", "cu13", "lib",
)
if os.path.isdir(CUDA_LIB):
    os.environ.setdefault("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{CUDA_LIB}:{os.environ['LD_LIBRARY_PATH']}"

os.environ["HF_HOME"] = "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface"

model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
DESCRIBE_PROMPT = "Describe this image in one or two sentences. Be concise and factual."

processor = None
model = None


def describe(image_path, prompt=None, max_tokens=256):
    global processor, model
    if prompt is None:
        prompt = DESCRIBE_PROMPT

    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=prompt_text, images=image,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    caption = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return caption.strip()


def load_model():
    global processor, model
    t0 = time.time()
    print("Loading processor...", file=sys.stderr, flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    print("Loading model (bnb-4bit, ~7 GB VRAM)...", file=sys.stderr, flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    return processor, model


def main():
    global model_name
    parser = argparse.ArgumentParser(description="Llama 3.2 Vision (transformers + bnb-4bit)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--task", type=str, default="describe", choices=["describe"],
                        help="Task type (default: describe)")
    parser.add_argument("--model", type=str, default=None, help="Model name or path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    args = parser.parse_args()

    if args.model is not None:
        model_name = args.model

    img_path = Path(args.image)
    if not img_path.exists():
        print(json.dumps({"error": f"Image not found: {img_path}"}))
        sys.exit(1)

    load_model()
    result = {"model": model_name, "image": str(img_path)}

    if args.task == "describe":
        t0 = time.time()
        caption = describe(str(img_path), max_tokens=args.max_tokens)
        result["description_text"] = caption
        result["time_s"] = round(time.time() - t0, 2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
