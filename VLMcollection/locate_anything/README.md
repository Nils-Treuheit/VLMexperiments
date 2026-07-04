# LocateAnything-3B

NVIDIA's [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) — a 3B-parameter vision-language model for visual grounding. Given an image and a text query, it returns bounding boxes or points for the described objects.

## Installation

```bash
# 1. Install system dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 2. Download the model weights (~7.8 GB)
huggingface-cli download nvidia/LocateAnything-3B --local-dir model

# 3. Verify it works
python infer.py path/to/image.jpg "person"
```

> **GPU recommended.** The model runs on CPU but will be very slow. NVIDIA GPU with 8GB+ VRAM (e.g. RTX 3070+) required for reasonable performance.

## Usage

### Basic detection

```bash
python infer.py ~/image.jpg "dog"

python infer.py ~/image.jpg "find the red car"

python infer.py ~/image.jpg "person</c>car</c>bicycle"
```

The `<c>` separator chains multiple category names into one query.

### Save annotated output

```bash
python infer.py ~/image.jpg "dog" -o output.jpg
```

This draws the predicted bounding boxes on the image and saves the result.

### JSON output (for scripting)

```bash
python infer.py ~/image.jpg "dog" --json
```

Returns:
```json
{"text": "There is one dog in the image <box>0.15,0.20,0.65,0.85</box>", "boxes": [[0.15,0.20,0.65,0.85]]}
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `hybrid` | Generation mode: `fast`, `hybrid`, or `slow` |
| `--output` / `-o` | None | Save annotated image to path |
| `--json` | off | Output machine-readable JSON |
| `--max-tokens` | 2048 | Maximum new tokens to generate |
| `--temperature` | 0.7 | Sampling temperature (0 = greedy) |
| `--device` | auto | Override device (`cpu`, `cuda:0`, etc.) |

## Prompt templates

| Task | Example query |
|---|---|
| Multi-class detection | `"person</c>car</c>dog"` |
| Referring expression | `"find the woman in the red dress"` |
| Single instance | `"Locate a single instance that matches the following description: stop sign."` |
| Text detection | `"Detect all the text in box format."` |
| Pointing | `"Point to: the entrance."` |

## Aliases (optional)

Add to your `~/.bashrc` for quick access:

```bash
alias locate-anything='python /mnt/HDD1/Project_Code/locate_anything/infer.py'
```

Then use from anywhere:

```bash
locate-anything ~/photo.jpg "cat"
```

## Notes

- Model licensed under **NVIDIA AI Foundation Models Community License** — non-commercial / research use only.
- The model supports images up to ~2.5K resolution and text prompts up to ~24K tokens.
- Output format: `<box>x1,y1,x2,y2</box>` for bounding boxes, `<box>x,y</box>` for points (coordinates normalized 0–1).

## Tested & Working (2026-07-04)

Verified on RTX 5090 (32 GB VRAM):
- `python infer.py img.jpg "person"` — basic detection
- `python infer.py img.jpg "person" --json` — JSON output for scripting
- `python infer.py img.jpg "person</c>car</c>dog"` — multi-class detection
- `python infer.py img.jpg "person" -o output.jpg` — annotated image output
- Benchmark integration: `benchmark_od.py --model locate_anything --max-images 100` — 4.7 FPS, mAP@50:95 0.170
- Benchmark integration: `benchmark_grounding.py --model locate_anything --max-images 100` — works
- VLMshowcase: `vlm-demo detect img.jpg --grounding person car` — works

Notes:
- Model must be downloaded first via `huggingface-cli download nvidia/LocateAnything-3B --local-dir model`
- ~8 GB VRAM at inference
- Three generation modes: `--mode fast` (faster, less accurate), `--mode hybrid` (balanced), `--mode slow` (most accurate)
