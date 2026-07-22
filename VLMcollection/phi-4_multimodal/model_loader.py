import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")


def load_model(
    cache_dir: str = None,
    device_map: str = "cuda",
    torch_dtype: str = "auto",
    use_flash_attention: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    if use_flash_attention and torch.cuda.is_available():
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    else:
        attn_impl = "eager"

    quantization_config = None
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        quantization_config = bnb_config

    print(f"Loading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    print(f"Loading model from {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        _attn_implementation=attn_impl,
        quantization_config=quantization_config,
    )

    model.eval()
    print(f"Model loaded on {model.device}")
    return model, processor
