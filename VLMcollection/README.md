# Edge-Device Compatible Vision-Language Models (VLMs)

Ready-to-run implementations of lightweight VLMs for edge deployment (NVIDIA Jetson, Orin NX/AGX, etc.)

## Models

| Model | Params | Folder | Description |
|-------|--------|--------|-------------|
| **Microsoft Florence-2** | 0.23B–0.77B | `florence-2/` | Captioning, OD, grounding, OCR — extremely fast |
| **Google PaliGemma 2** | 3B | `paligemma/` | SigLIP + Gemma-2B decoder, versatile VLM |
| **Llama 3.2 Vision** | 11B | `llama-vision/` | Strong reasoning, needs 16GB+ VRAM (quantized) |
| **Phi-3.5 Vision** | 4.2B | `phi-vision/` | 128K context, excellent image understanding |
| **NVIDIA Cosmos Reason1** | 7B | `cosmos-nemotron/` | Physical AI reasoning, robotics & AV |

## Quick Start

Each project has its own venv and run script:

```bash
cd <model-folder>
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -U torch transformers pillow requests accelerate bitsandbytes
python run.py
```

## Quantization for Edge

All models support 4/8-bit quantization via `bitsandbytes`:

```python
model = AutoModel.from_pretrained(..., load_in_4bit=True, device_map="auto")
```

For TensorRT-LLM or AWQ quantization, see each project's README for specific instructions.
