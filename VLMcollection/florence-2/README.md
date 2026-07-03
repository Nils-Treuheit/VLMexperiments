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

## Tested & Working (2026-07-03)

All tasks verified on RTX 5090:
- `<CAPTION>` — concise image captioning
- `<DETAILED_CAPTION>` — detailed description
- `<OD>` — object detection

Notes:
- Uses `AutoModelForCausalLM` with `trust_remote_code=True` (Florence2ForConditionalGeneration not available in transformers 4.47.1)
- Cast `pixel_values` to model dtype (float16) to avoid dtype mismatch

Programmatic usage (for benchmark/showcase integration):
```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large-ft", trust_remote_code=True, torch_dtype=torch.float16
).to("cuda")
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

img = Image.open("image.jpg").convert("RGB")
inputs = processor(text="<CAPTION>", images=img, return_tensors="pt").to("cuda")
inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
gids = model.generate(**inputs, max_new_tokens=200)
result = processor.batch_decode(gids, skip_special_tokens=False)[0]
parsed = processor.post_process_generation(result, task="<CAPTION>", image_size=img.size)
```

## References

- https://huggingface.co/microsoft/Florence-2-large-ft
- https://arxiv.org/abs/2311.06242
