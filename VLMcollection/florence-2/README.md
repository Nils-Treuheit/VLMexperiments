# Microsoft Florence-2 — Edge VLM

Ultra-compact VLM (0.23B–0.77B params) for captioning, object detection, grounding, OCR.

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

The script demonstrates:
- `<CAPTION>` — image captioning
- `<DETAILED_CAPTION>` — detailed captioning
- `<OD>` — object detection with bounding boxes
- `<OCR>` — optical character recognition

## Quantization (Edge)

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = Florence2ForConditionalGeneration.from_pretrained(
    "microsoft/Florence-2-base",
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)
```

## Model Variants

| Model | Params | Best for |
|-------|--------|----------|
| `microsoft/Florence-2-base` | 0.23B | Fastest on edge |
| `microsoft/Florence-2-large-ft` | 0.77B | Best accuracy |

## References

- https://huggingface.co/microsoft/Florence-2-large-ft
- https://arxiv.org/abs/2311.06242
