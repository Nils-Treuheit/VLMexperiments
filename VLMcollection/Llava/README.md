# LLaVA Models

This folder provides a unified interface for several LLaVA-based vision-language models:

| Key | Model | Size | Architecture |
|---|---|---|---|
| `llava-v1.6-mistral` | [LLaVA-v1.6-Mistral-7B](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | 7B | LlavaNextForConditionalGeneration |
| `llava-onevision` | [LLaVA-OneVision-Qwen2-7B](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) | 7B | LlavaOnevisionForConditionalGeneration |
| `llava-next-video-7b` | [LLaVA-NeXT-Video-7B](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) | 7B | LlavaNextVideoForConditionalGeneration |
| `llava-next-video-34b` | [LLaVA-NeXT-Video-34B-DPO](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-34B-DPO-hf) | 34B | LlavaNextVideoForConditionalGeneration (4-bit quantized) |
| `phi-3-vision` | [LLaVA-Phi-3-Mini](https://huggingface.co/xtuner/llava-phi-3-mini-hf) | 4.2B | LlavaForConditionalGeneration (LLaVA fine-tuned on Phi-3-mini) |

## Setup

The virtual environment is already configured. To install dependencies:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

## Usage

```bash
# Basic image captioning
python run.py --model llava-v1.6-mistral --image path/to/image.jpg --task caption

# VQA with custom prompt
python run.py --model llava-onevision --image path/to/image.jpg --task vqa --prompt "What color is the car?"

# 34B model with 4-bit quantization
python run.py --model llava-next-video-34b --image path/to/image.jpg --task caption --quantize

# All available models
for model in llava-v1.6-mistral llava-onevision llava-next-video-7b phi-3-vision; do
    python run.py --model $model --image path/to/image.jpg --task caption
done
```

## Model details

- **LLaVA-v1.6-Mistral-7B**: LLaVA 1.6 with Mistral-7B language backbone. Good general VLM.
- **LLaVA-OneVision-Qwen2-7B**: Latest LLaVA with Qwen2-7B backbone. Supports multi-image and video.
- **LLaVA-NeXT-Video-7B/34B**: Video-focused LLaVA models with improved temporal understanding.
- **Phi-3-Vision-128K**: Microsoft's 4.2B VLM with 128K context window. Efficient.
- **LLaVA-Phi-3-Mini** (now `phi-3-vision`): LLaVA-v1.5-style model fine-tuned on Phi-3-mini. Uses `<image>` token format (not `<|image_1|>`).

## Architecture

All models use a similar architecture:
1. Vision encoder (CLIP/SigLIP-based) processes images into visual tokens
2. Projector aligns visual tokens with language embedding space
3. Language model (Mistral/Qwen2/Phi-3) generates responses conditioned on visual + text tokens

## Output format

```json
{
  "model": "llava-hf/llava-v1.6-mistral-7b-hf",
  "model_key": "llava-v1.6-mistral",
  "image": "path/to/image.jpg",
  "task": "caption",
  "prompt": "Please describe this image in detail.",
  "response": "A beautiful landscape with mountains...",
  "inference_time_s": 2.345
}
```

## Notes

- The 34B model requires ~20GB GPU memory even with 4-bit quantization.
- All images are converted to RGB internally.
- First run will download model weights from HuggingFace (~15GB for 7B, ~20GB for 34B).
