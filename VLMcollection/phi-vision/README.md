# Microsoft Phi-3.5 Vision — Edge VLM

4.2B VLM with 128K context. Excellent for document understanding, charts, general VQA.

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

## References

- https://huggingface.co/microsoft/Phi-3.5-vision-instruct
- https://onnxruntime.ai/docs/genai/tutorials/phi3-v.html
