import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch
from PIL import Image


DESCRIBE_PROMPTS = {
    "vqa": "Please answer the question based on the image.",
    "caption": "Please describe this image in detail.",
}


def get_model_info(model_name):
    info = {
        "llava-onevision": {
            "hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "arch": "LlavaOnevisionForConditionalGeneration",
            "processor_cls": "LlavaOnevisionProcessor",
        },
        "llava-next-video-34b": {
            "hf_id": "llava-hf/LLaVA-NeXT-Video-34B-DPO-hf",
            "arch": "LlavaNextVideoForConditionalGeneration",
            "processor_cls": "LlavaNextVideoProcessor",
        },
        "llava-next-video-7b": {
            "hf_id": "llava-hf/LLaVA-NeXT-Video-7B-hf",
            "arch": "LlavaNextVideoForConditionalGeneration",
            "processor_cls": "LlavaNextVideoProcessor",
        },
        "llava-v1.6-mistral": {
            "hf_id": "llava-hf/llava-v1.6-mistral-7b-hf",
            "arch": "LlavaNextForConditionalGeneration",
            "processor_cls": "LlavaNextProcessor",
        },
        "phi-3-vision": {
            "hf_id": "xtuner/llava-phi-3-mini-hf",
            "arch": "LlavaForConditionalGeneration",
        },
    }
    if model_name not in info:
        raise ValueError(f"Unknown model {model_name!r}. Choices: {list(info)}")
    return info[model_name]


def load_model(model_name, device, quantize_4bit=False, attn_implementation=None):
    info = get_model_info(model_name)
    hf_id = info["hf_id"]
    arch = info["arch"]

    from transformers import AutoProcessor

    ARCH_CLASSES = {
        "LlavaOnevisionForConditionalGeneration": "LlavaOnevisionForConditionalGeneration",
        "LlavaNextVideoForConditionalGeneration": "LlavaNextVideoForConditionalGeneration",
        "LlavaNextForConditionalGeneration": "LlavaNextForConditionalGeneration",
        "LlavaForConditionalGeneration": "LlavaForConditionalGeneration",
    }

    import importlib
    mod = importlib.import_module("transformers")
    model_cls = getattr(mod, ARCH_CLASSES[arch])

    dtype = torch.float16 if device == "cuda" else torch.float32
    kw = {"torch_dtype": dtype, "device_map": device, "trust_remote_code": True}
    if attn_implementation is not None:
        kw["attn_implementation"] = attn_implementation

    if arch == "LlavaForConditionalGeneration":
        kw["low_cpu_mem_usage"] = True

    if quantize_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kw["quantization_config"] = bnb_cfg
        kw.pop("torch_dtype", None)
        kw["device_map"] = device

    model = model_cls.from_pretrained(hf_id, **kw).eval()
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

    if arch == "LlavaForConditionalGeneration":
        if processor.patch_size is None:
            processor.patch_size = 14

    return model, processor


def run_inference(model, processor, image, prompt, model_name, device, max_new_tokens=512):
    info = get_model_info(model_name)
    arch = info["arch"]

    if arch in ("LlavaOnevisionForConditionalGeneration", "LlavaNextVideoForConditionalGeneration", "LlavaNextForConditionalGeneration"):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(device=device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response_text = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    elif arch == "LlavaForConditionalGeneration":
        prompt_text = f"<|user|>\n<image>\n{prompt}<|end|>\n<assistant|>\n"
        inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        inputs = {k: v.to(device=device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response_text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return response_text.strip()


def main():
    parser = argparse.ArgumentParser(description="LLaVA Model Runner")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="llava-v1.6-mistral",
                        choices=["llava-onevision", "llava-next-video-34b", "llava-next-video-7b",
                                 "llava-v1.6-mistral", "phi-3-vision"],
                        help="Model variant")
    parser.add_argument("--task", type=str, default="describe",
                        choices=["describe", "vqa", "caption"],
                        help="Task type")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization (for 34B model)")
    parser.add_argument("--attn-implementation", type=str, default="sdpa",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention implementation (default: sdpa — FA2 ELF mismatch on this system)")
    parser.add_argument("--top-k", type=int, default=5, help="Not used for LLaVA (kept for compatibility)")
    parser.add_argument("--labels-file", type=str, default=None, help="Not used for LLaVA (kept for compat)")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(json.dumps({"error": f"Image not found: {img_path}"}))
        sys.exit(1)

    image = Image.open(img_path).convert("RGB")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    warnings.filterwarnings("ignore", message=".*torch_dtype.*")

    model, processor = load_model(args.model, device, quantize_4bit=args.quantize,
                                   attn_implementation=args.attn_implementation)

    if args.prompt:
        prompt = args.prompt
    elif args.task == "vqa":
        prompt = DESCRIBE_PROMPTS["vqa"]
    elif args.task == "caption":
        prompt = DESCRIBE_PROMPTS["caption"]
    else:
        prompt = "Please describe this image in detail."

    t0 = time.time()
    response_text = run_inference(model, processor, image, prompt, args.model, device,
                                   max_new_tokens=args.max_new_tokens)
    elapsed = time.time() - t0

    result = {
        "model": get_model_info(args.model)["hf_id"],
        "model_key": args.model,
        "image": str(img_path),
        "task": args.task,
        "prompt": prompt,
        "response": response_text,
        "inference_time_s": round(elapsed, 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
