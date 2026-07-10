import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

MODEL_MAP = {
    "phi-3.5-vision": {
        "hf_id": "microsoft/Phi-3.5-vision-instruct",
        "prompt_fmt": lambda p: f"<|user|>\n<|image_1|>\n{p}<|end|>\n<|assistant|>\n",
    },
    "phi-3-vision-128k": {
        "hf_id": "microsoft/Phi-3-vision-128k-instruct",
        "prompt_fmt": lambda p: f"<|user|>\n<|image_1|>\n{p}<|end|>\n<|assistant|>\n",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Phi-Vision Model Runner")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="phi-3.5-vision",
                        choices=list(MODEL_MAP), help="Model variant")
    parser.add_argument("--task", type=str, default="describe",
                        choices=["describe", "vqa", "caption"], help="Task type")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--n-predict", type=int, default=None, help="Alias for max-new-tokens")
    args = parser.parse_args()

    if args.n_predict:
        args.max_new_tokens = args.n_predict

    img_path = Path(args.image)
    if not img_path.exists():
        print(json.dumps({"error": f"Image not found: {img_path}"}))
        sys.exit(1)

    image = Image.open(img_path).convert("RGB")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_info = MODEL_MAP[args.model]
    hf_id = model_info["hf_id"]

    warnings.filterwarnings("ignore")
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, config=config, trust_remote_code=True,
        torch_dtype="auto", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

    if args.prompt:
        prompt = args.prompt
    elif args.task == "vqa":
        prompt = "Please answer the question based on the image."
    elif args.task == "caption":
        prompt = "Please describe this image in detail."
    else:
        prompt = "What is shown in this image? Describe it in detail."

    prompt_text = model_info["prompt_fmt"](prompt)
    inputs = processor(prompt_text, image, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, use_cache=False)
    response = processor.decode(generate_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - t0

    result = {
        "model": hf_id,
        "model_key": args.model,
        "image": str(img_path),
        "task": args.task,
        "prompt": prompt,
        "response": response.strip(),
        "inference_time_s": round(elapsed, 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
