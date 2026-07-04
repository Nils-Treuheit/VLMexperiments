# Google PaliGemma 2 — Edge VLM

3B VLM (SigLIP-400M + Gemma-2B), great for segmentation, navigation, VQA.

## Setup

```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -U torch transformers pillow requests accelerate
```

## Usage

```bash
python run.py
```

Switch to a different task by changing the prompt:
- `"caption en"` — captioning
- `"detect cat"` — object detection
- `"segment"` — segmentation
- `"What is in this image?"` — VQA

## Quantization (Edge)

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-3b-mix-224",
    quantization_config=quant_config,
    device_map="auto",
)
```

## Model Variants

| Model | Res. | Best for |
|-------|------|----------|
| `google/paligemma2-3b-mix-224` | 224px | Out-of-box use |
| `google/paligemma2-3b-pt-224` | 224px | Fine-tuning |
| `google/paligemma2-3b-mix-448` | 448px | Higher detail |

## Tested & Working (2026-07-03)

Gated model — requires `HF_TOKEN` environment variable (set globally). Verified on RTX 5090:
- `"caption en"` — English captioning
- `"What is in this image?"` — VQA
- `"detect <object>"` — detection

Programmatic usage (for benchmark/showcase integration):
```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-3b-mix-224", torch_dtype=torch.bfloat16, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

img = Image.open("image.jpg").convert("RGB")
inputs = processor(img, "caption en", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(output[0], skip_special_tokens=True)
```

## References

- https://huggingface.co/google/paligemma2-3b-mix-224
- https://ai.google.dev/gemma/docs/paligemma
