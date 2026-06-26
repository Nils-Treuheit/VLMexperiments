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

## References

- https://huggingface.co/google/paligemma2-3b-mix-224
- https://ai.google.dev/gemma/docs/paligemma
