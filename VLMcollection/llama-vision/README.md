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

## References

- https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- https://www.llama.com/docs/how-to-guides/vision-capabilities
