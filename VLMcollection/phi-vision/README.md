# Phi Vision Models

This folder supports Microsoft's Phi vision-language models.

| Model | Size | Context |
|---|---|---|
| `phi-3.5-vision` (default) | Phi-3.5-Vision-Instruct — 4.2B | 128K |
| `phi-3-vision-128k` | Phi-3-Vision-128K-Instruct — 4.2B | 128K |

## Setup

```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -U torch transformers pillow requests accelerate
```

## Usage

```bash
# Default (Phi-3.5-Vision)
python run.py --image path/to/image.jpg

# Phi-3-Vision-128K
python run.py --model phi-3-vision-128k --image path/to/image.jpg

# Custom prompt
python run.py --image path/to/image.jpg --prompt "What text do you see?"
```

## Quantization for Edge

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)
```

## ONNX Runtime (Alternative)

For even faster edge inference, use the ONNX variant:

```bash
pip install onnxruntime-genai-cuda pillow
huggingface-cli download microsoft/Phi-3.5-vision-instruct-onnx --include gpu/gpu-int4-rtn-block-32/* --local-dir ./model
```

## Tested & Working (2026-07-03)

Verified on RTX 5090. Notes:
- Use `config._attn_implementation = "eager"` before loading (Flash Attention 2 not supported)
- Use `use_cache=False` during generation (static cache incompatibility)
- Use `processor.tokenizer.decode()` directly for response (bypasses remote code processor)

Programmatic usage (for benchmark/showcase integration):
```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from PIL import Image

model_id = "microsoft/Phi-3.5-vision-instruct"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config._attn_implementation = "eager"
model = AutoModelForCausalLM.from_pretrained(
    model_id, config=config, trust_remote_code=True,
    torch_dtype="auto", device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

img = Image.open("image.jpg").convert("RGB")
prompt = "<|user|>\n<|image_1|>\nDescribe this image.<|end|>\n<|assistant|>\n"
inputs = processor(prompt, img, return_tensors="pt").to(model.device)
gids = model.generate(**inputs, max_new_tokens=200, use_cache=False, do_sample=False)
response = processor.tokenizer.decode(gids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

## References

- https://huggingface.co/microsoft/Phi-3.5-vision-instruct
- https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
- https://onnxruntime.ai/docs/genai/tutorials/phi3-v.html
