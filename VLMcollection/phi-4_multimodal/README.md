# Phi-4 Multimodal — Microsoft

Multimodal VLM supporting **text + image + audio** inputs with Flash Attention 2 and 4/8-bit quantization.

- **Model:** `microsoft/Phi-4-multimodal-instruct`
- **Params:** ~14B
- **VRAM:** ~16 GB (BF16), ~8 GB (4-bit)
- **Capabilities:** Description, VQA, document understanding, audio understanding, video frame analysis

## Quick Start

```bash
source .venv/bin/activate

# Image description
python run.py --image img.jpg

# VQA with custom prompt
python run.py --image img.jpg --prompt "What is shown in this image?"

# Full multimodal (text + image + audio)
python inference.py --image img.jpg --audio speech.wav --text "Describe this scene"
```

## Usage

### Single-image inference (`run.py`)

```bash
python run.py --image path/to/img.jpg
python run.py --image path/to/img.jpg --prompt "Read this document" --max-tokens 500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | (required) | Path to input image |
| `--prompt` | `"Describe this image in detail."` | Text prompt |
| `--max-tokens` | `256` | Maximum new tokens |
| `--device` | auto | Override device (`cpu`, `cuda`) |

### Full multimodal inference (`inference.py`)

```bash
# Image + text
python inference.py --image img.jpg --text "What color is the car?"

# Audio + text
python inference.py --audio speech.wav --text "Summarize this"

# Image + audio + text
python inference.py --image img.jpg --audio speech.wav --text "Describe with audio context"

# With quantization
python inference.py --image img.jpg --load-4bit
python inference.py --image img.jpg --load-8bit
```

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | None | Path to image |
| `--audio` | None | Path to audio file |
| `--text` | `""` | Text prompt |
| `--max-new-tokens` | `256` | Maximum new tokens |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-p` | `0.9` | Top-p sampling |
| `--sampling` | off | Enable sampling (default: greedy) |
| `--load-4bit` | off | 4-bit quantization via bitsandbytes |
| `--load-8bit` | off | 8-bit quantization via bitsandbytes |

### Video / webcam pipeline (`video_pipeline.py`)

```bash
# Analyze video frames
python video_pipeline.py --video video.mp4 --text "Describe each frame"

# Live webcam inference
python video_pipeline.py --webcam 0 --text "What do you see?"
```

## Hardware

RTX 5090 (32 GB VRAM). At BF16 the model uses ~16 GB. 4-bit quantization reduces to ~8 GB.

## Dependencies

- `transformers >= 4.48.2` (Phi-4-multimodal support added in 4.48.x)
- `torch`, `accelerate`, `bitsandbytes` for quantization
- `soundfile`, `librosa` for audio
- `opencv-python-headless` for video/webcam

## Files

| File | Purpose |
|------|---------|
| `run.py` | Quick single-image inference |
| `inference.py` | Full multimodal (image + audio + text) |
| `model_loader.py` | Shared model loading with quantization support |
| `video_pipeline.py` | Video frame sampling + webcam inference |

## Notes

- Prompt format: `<|user|><|image_1|>(optional audio)<|audio_1|>text<|end|><|assistant|>`
- The venv's `python` symlink points to system Python 3.13 (works correctly with the venv's packages)
- Model cached at `model_cache/` (auto-downloads on first run)
- Flash Attention 2 enabled by default on CUDA; falls back to `"eager"` if unavailable

## Tested & Working (2026-07-04)

Verified on RTX 5090 (32 GB VRAM):
- `python run.py --image img.jpg` — single-image description with proper prompt format
- `python run.py --image img.jpg --prompt "Read this document"` — VQA with custom prompt
- VLMshowcase: `vlm-demo vlm phi4 img.jpg "Describe this scene"` — works via custom prompt

Notes:
- The default `AutoProcessor.from_pretrained()` chat template has a Jinja2 bug; prompts are constructed manually with special tokens (`<|user|><|image_1|>prompt<|end|><|assistant|>`)
- Loading requires `trust_remote_code=True`
- Set `HF_HOME` to the shared HuggingFace cache before running
