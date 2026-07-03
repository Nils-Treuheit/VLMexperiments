#!/usr/bin/env python3
"""CLI wrapper for Phi-3.5-Vision. Bypasses broken remote code processor.decode()."""
import json
import os
import sys
import torch
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_suppress = StringIO()
with redirect_stdout(_suppress), redirect_stderr(_suppress):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/Phi-3.5-vision-instruct"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, trust_remote_code=True,
        torch_dtype="auto", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # Use the base tokenizer directly (bypass remote code processor)
    tokenizer = processor.tokenizer


def main():
    if len(sys.argv) < 3:
        print("Usage: _phi_wrapper.py <image_path> <prompt>")
        sys.exit(1)

    img_path = sys.argv[1]
    prompt = sys.argv[2]

    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    full = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
    inputs = processor(full, img, return_tensors="pt").to(model.device)
    # Use raw input IDs from processor
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        gids = model.generate(
            **inputs,
            max_new_tokens=500,
            use_cache=False,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    # Decode using base tokenizer (new tokens only)
    response = tokenizer.decode(gids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()
