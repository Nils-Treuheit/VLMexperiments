# NVIDIA Cosmos-Reason1 (Nemotron) — Edge VLM

7B Physical AI reasoning VLM — understands space, time, and physics. Optimized for robotics & autonomous vehicles. Based on Qwen2.5-VL-7B.

**This is the quantized-ready version.** The model supports 4-bit AWQ and 8-bit FP8 quantization for Jetson deployment.

## Setup

```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -U torch transformers pillow requests accelerate bitsandbytes
```

## Usage

```bash
python run.py
```

## Quantization for Edge

### 4-bit (bitsandbytes)

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForMultimodalLM.from_pretrained(
    "nvidia/Cosmos-Reason1-7B",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

### FP8 (NVIDIA TensorRT-LLM)

Uses `llmcompressor` for FP8 quantization:

```bash
pip install vllm llmcompressor qwen-vl-utils
python -c "
from llmcompressor import compress
compress('nvidia/Cosmos-Reason1-7B', 'Cosmos-Reason1-7B-W8A8-FP8')
"
```

### AWQ Quantized Variants (Community)

Check HuggingFace for AWQ-quantized versions:
- `stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ` (Nemotron 3 Nano, 30B MoE, 3.6B active)

## Video Input

The model natively supports video input with temporal reasoning. See the script for video inference examples.

## References

- https://huggingface.co/nvidia/Cosmos-Reason1-7B
- https://github.com/nvidia-cosmos/cosmos-reason1
- https://developer.nvidia.com/blog/visual-language-intelligence-and-edge-ai-2-0/
- https://research.nvidia.com/labs/nemotron/

## Tested & Working (2026-07-04)

Verified on RTX 5090 (32 GB VRAM):
- `python run.py` — single-image description with default prompt
- VQA with custom prompts (`--prompt "What physical interactions do you see?"`)
- Benchmark integration: `benchmark_caption.py --model cosmos_nemotron --max-images 2` — 4.3s/image avg
- Benchmark integration: `benchmark_vqa.py --model cosmos_nemotron --max-questions 8` — works
- VLMshowcase: `vlm-demo vlm cosmos_nemotron img.jpg "Describe this scene"` — works

Notes:
- Model loads slowly (~30-60s for first inference on RTX 5090)
- ~16 GB VRAM at BF16
- Loading from HuggingFace cache on first run may take time
