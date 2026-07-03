import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
from PIL import Image

from model_loader import load_model


def load_image(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(path).convert("RGB")


def load_audio(audio_path: str):
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    data, samplerate = sf.read(str(path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, samplerate


def build_prompt(text: str = None, has_image: bool = False, has_audio: bool = False) -> str:
    user_prompt = "<|user|>"
    prompt_suffix = "<|end|>"
    assistant_prompt = "<|assistant|>"

    parts = []
    if has_image:
        parts.append("<|image_1|>")
    if has_audio:
        parts.append("<|audio_1|>")
    if text:
        parts.append(text)

    return f"{user_prompt}{''.join(parts)}{prompt_suffix}{assistant_prompt}"


@torch.no_grad()
def run_inference(
    model,
    processor,
    text: str = None,
    image_path: str = None,
    audio_path: str = None,
    max_new_tokens: int = 1000,
    do_sample: bool = False,
    temperature: float = None,
    top_p: float = None,
):
    has_image = image_path is not None
    has_audio = audio_path is not None
    has_text = bool(text and text.strip())

    if not has_text and not has_audio and not has_image:
        print("Error: At least one of --text, --image, or --audio is required", file=sys.stderr)
        sys.exit(1)

    prompt = build_prompt(text=text, has_image=has_image, has_audio=has_audio)
    print(f"Prompt:\n{prompt}\n")

    images = None
    if has_image:
        images = load_image(image_path)

    audios = None
    if has_audio:
        audio_data, samplerate = load_audio(audio_path)
        audios = [(audio_data, samplerate)]

    inputs = processor(
        text=prompt,
        images=images,
        audios=audios,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_logits_to_keep": 0,
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    generate_ids = model.generate(**inputs, **gen_kwargs)
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


def main():
    parser = argparse.ArgumentParser(
        description="Phi-4-multimodal-instruct inference with text, image, and audio"
    )
    parser.add_argument("--text", type=str, default=None, help="Text prompt")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--sampling", action="store_true", help="Use sampling (default: greedy)")
    parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit quantization")
    args = parser.parse_args()

    model, processor = load_model(
        cache_dir=args.cache_dir or None,
        load_in_4bit=args.load_4bit,
        load_in_8bit=args.load_8bit,
    )

    response = run_inference(
        model=model,
        processor=processor,
        text=args.text,
        image_path=args.image,
        audio_path=args.audio,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sampling,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
