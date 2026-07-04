# Meta Llama 3.2 Vision — Edge VLM

11B multimodal model with strong reasoning. Needs 16GB+ VRAM; use quantization for Jetson Orin 64GB.

## Setup

```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -U torch transformers pillow requests accelerate bitsandbytes
```

**Requires HuggingFace login** (Meta license):

```bash
huggingface-cli login
# Accept terms at: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
```

## Usage

```bash
python run.py
```

## Quantization for Edge

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

## Ollama Alternative (Recommended for Edge)

```bash
ollama pull llama3.2-vision:11b
ollama run llama3.2-vision:11b
```

## Tested & Working (2026-07-03)

Verified on RTX 5090 with `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` (quantized, ~7GB VRAM):
- `python run.py --image <path> --task describe` — scene description

Notes:
- Uses `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` by default (4-bit quantized, fast loading)
- Falls back to `meta-llama/Llama-3.2-11B-Vision-Instruct` if `--model` is specified

Programmatic usage (for benchmark/showcase integration):
```python
from transformers import AutoProcessor, AutoModelForMultimodalLM
from PIL import Image

processor = AutoProcessor.from_pretrained("unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit")
model = AutoModelForMultimodalLM.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    device_map="auto", torch_dtype=torch.bfloat16,
).eval()

img = Image.open("image.jpg").convert("RGB")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
caption = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
```

## References

- https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- https://www.llama.com/docs/how-to-guides/vision-capabilities
