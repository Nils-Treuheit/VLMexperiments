# VLMshowcase

Vision-Language Model Capability Demo — runs nine VLM models installed on this system.

```
source .venv/bin/activate
vlm-demo <command> [args]
```

## Quick Start

```bash
# Show available models and demo material
vlm-demo list

# Process a folder of images with batch mode (load model once)
vlm-demo run /path/to/images/ --model yolo26n --batch
vlm-demo run /path/to/images/ --model cosmos_nemotron --batch

# Run any VLM with a custom prompt
vlm-demo vlm florence2 image.jpg "<CAPTION>"
vlm-demo vlm cosmos_nemotron image.jpg "What physical interactions do you see?"

# Batch process multiple images
vlm-demo batch yolo26n img1.jpg img2.jpg
vlm-demo batch cosmos_nemotron img1.jpg img2.jpg --prompt "Describe"

# Side-by-side comparison across all models
vlm-demo compare image.jpg
```

## Models Overview

| Model | Params | Type | VRAM | Load Time | Status |
|-------|--------|------|------|-----------|--------|
| **YOLO11/26** (Ultralytics) | 2.7M–68M | CNN detector | <1 GB | <1s | ✅ Ready |
| **LocateAnything-3B** (NVIDIA) | 3B | Specialized VLM (grounding) | ~8 GB | ~30s | ✅ Ready |
| **Qwen3-VL-8B-Instruct** (Alibaba) | 8.8B | General VLM | ~30 GB | ~15s | ✅ Ready |
| **Qwen3-VL-8B-Thinking** (Unsloth 4-bit) | ~9B | Reasoning VLM | ~6 GB | ~45s | ✅ Ready |
| **Florence-2-large** (Microsoft) | 0.77B | Compact task-prompt VLM | ~3 GB | ~20s | ⚠️ (see notes) |
| **PaliGemma2-3B** (Google) | 3B | Segmentation+detection VLM | ~6 GB | ~10s | ⚠️ HF login required |
| **Cosmos-Reason1-7B** (NVIDIA) | 7B | Physical AI reasoning VLM | ~16 GB | ~11s | ✅ Ready |
| **Phi-3.5-Vision-4B** (Microsoft) | 4.2B | Document/chart VLM (128K ctx) | ~8 GB | ~10s | ⚠️ (see notes) |
| **Llama-3.2-11B-Vision** (Meta) | 11B | Multimodal reasoning VLM | ~16 GB | ~15s | ⚠️ HF login required |
| **DiffusionGemma-26B** (Google) | 26B | Diffusion VLM + YOLO encoder | ~16 GB | ~60s | ⚠️ Needs GGUF |
| **Phi-4-Multimodal** (Microsoft) | — | Next-gen multimodal VLM | ~16 GB | ~15s | ⚠️ (see notes) |
| **DINOv3** (Meta) | 22M–7B | Vision encoder (features) | <1 GB | <5s | ⚠️ HF login |
| **SigLIP2** (Google) | 0.4B | Vision encoder (zero-shot) | <1 GB | <8s | ✅ Ready |
| **MoonViT** (Moonshot AI) | 0.4B | Vision encoder (native res) | <1 GB | <8s | ✅ Ready |

All models pre-installed at `/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/`. Heavy VLMs called via subprocess
through their own venvs; YOLO loaded directly via `ultralytics`.

## Model Status Notes

- **Florence-2**: Uses `AutoModelForCausalLM` with `trust_remote_code=True` (compatible with transformers 5.x).
  All benchmark scripts updated for dtype cast. See `VLMcollection/florence-2/` for standalone usage.
- **PaliGemma2**: Gated model — run `huggingface-cli login` with a token that has accepted the license.
  Works once authenticated (`HF_TOKEN` env var).
- **Llama-3.2-Vision**: Uses 4-bit quantized version (`unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`)
  instead of gated meta-llama original. No login required.
- **Phi-3.5-Vision**: Patched for transformers 5.x — uses `_attn_implementation="eager"` via config
  and `use_cache=False` in `generate()`. Works correctly.

---

## Available Commands

| Command | Description |
|---------|-------------|
| `list` | Show models and demo material |
| `scene <image>` | Full scene analysis (Qwen3 + YOLO) |
| `detect <image>` | Object detection (YOLO + LocateAnything) |
| `pose <image>` | Pose estimation |
| `intent <image>` | Human intent/action/emotion analysis |
| `track <video>` | Multi-object tracking |
| `compare <image>` | All-model side-by-side comparison |
| `vlm <model> <img> <prompt>` | Run any VLM with custom prompt |
| `run <folder>` | Process all images in a folder (--model, --batch) |
| `batch <model> <imgs...>` | Load-once batch processing |
| `obb <image>` | Oriented bounding boxes (aerial) |
| `webcam` | Live YOLO webcam detection |

---

## YOLO11/26

Lightning-fast CNN detection, pose, OBB, and tracking. 18 pre-downloaded weights.

### Detection
```
vlm-demo detect image.jpg --yolo yolo26n yolo26s yolo11m
```

### Pose Estimation
```
vlm-demo pose image.jpg --model yolo26n-pose yolo11n-pose
```

### Oriented Bounding Boxes (DOTA aerial)
```
vlm-demo obb image.png --model yolo26n-obb yolo11n-obb
```

### Multi-Object Tracking
```
vlm-demo track video.mp4 --model yolo26n
```

### Live Webcam
```
vlm-demo webcam --model yolo26n --fps 15
```

### Batch
```
vlm-demo batch yolo26n img1.jpg img2.jpg
vlm-demo batch yolo26n-pose img1.jpg img2.jpg --mode pose
```

---

## LocateAnything-3B

Free-form visual grounding — describe any object in natural language.

### Grounded Detection
```
vlm-demo detect image.jpg --grounding "person" "car" "red cup"
```

### Text Detection
```
vlm-demo detect document.jpg --grounding "Detect all text in box format"
```

### Batch Grounding
```
vlm-demo batch locate_anything img1.jpg img2.jpg --prompt "person" "car"
```

### Multi-class Detection
```
vlm-demo detect scene.jpg --grounding "person</c>car</c>bicycle"
```

---

## Qwen3-VL-8B-Instruct

General-purpose VLM for scene description, OCR, and VQA.

### Scene Description
```
vlm-demo scene image.jpg
```

### Visual Question Answering
```
vlm-demo detect image.jpg --grounding "What color is the car?"
```

### OCR
```
vlm-demo detect receipt.jpg --grounding "Read all the text"
```

### Batch
```
vlm-demo batch qwen3_instruct img1.jpg img2.jpg --mode describe
```

---

## Qwen3-VL-8B-Thinking

Reasoning VLM with chain-of-thought. Best for intent and complex reasoning.

### Reasoning Description
```
vlm-demo scene image.jpg
```

### Human Intent Analysis
```
vlm-demo intent image.jpg --all
vlm-demo intent image.jpg --aspect action|intent|emotion|social
```

### Detection
```
vlm-demo batch qwen3_thinking img.jpg --mode detect
```

### Batch
```
vlm-demo batch qwen3_thinking img1.jpg img2.jpg --mode describe
```

---

## Cosmos-Reason1-7B (NVIDIA)

Physical AI reasoning VLM. Understands space, time, and physics in scenes.

### Physical Scene Description
```
vlm-demo vlm cosmos_nemotron image.jpg "Describe the physical interactions"
vlm-demo vlm cosmos_nemotron image.jpg "What forces are at work here?"
```

### VQA with Physical Reasoning
```
vlm-demo vlm cosmos_nemotron image.jpg "Is this scene physically plausible? Why?"
```

### Folder Batch
```
vlm-demo run /path/to/folder/ --model cosmos_nemotron --batch
```

### Batch
```
vlm-demo batch cosmos_nemotron img1.jpg img2.jpg --prompt "Describe this scene physically"
```

---

## Florence-2-large (Microsoft)

Task-prompt driven compact VLM (<1B params). Fast captioning, detection, OCR.

> **Note**: Currently needs transformers 4.x. Use the original script directly:
> ```bash
> source vlm_det_test/florence-2/.venv/bin/activate
> python vlm_det_test/florence-2/run.py
> ```

### Capabilities (when working)
- `<CAPTION>` — image captioning
- `<DETAILED_CAPTION>` — detailed captioning
- `<OD>` — object detection with bounding boxes
- `<OCR>` — optical character recognition

---

## PaliGemma2-3B (Google)

Segmentation-capable VLM. Prompt-driven task switching.

> **Note**: Gated model. Run `huggingface-cli login` first.

### Capabilities (when authenticated)
- `"caption en"` — captioning
- `"detect cat"` — object detection
- `"segment"` — segmentation
- `"What is in this image?"` — VQA

---

## Phi-3.5-Vision-4B (Microsoft)

Small VLM with 128K context. Excels at document/chart understanding.

### Document Understanding
```
vlm-demo vlm phi_vision document.jpg "Read and summarize this document"
```

### Chart QA
```
vlm-demo vlm phi_vision chart.png "What does this chart show?"
```

### Scene Description
```
vlm-demo vlm phi_vision image.jpg "Describe this scene"
```

---

## Llama-3.2-11B-Vision (Meta)

Strong multimodal reasoning (11B params, gated model).

> **Note**: Requires `huggingface-cli login` and license acceptance.

### Capabilities (when authenticated)
- Scene description and reasoning
- Visual question answering
- Complex multimodal tasks

---

## Folder Batch Processing (`run` command)

Process all images in a folder with a single model call:

```bash
# Process with YOLO (fast)
vlm-demo run /mnt/HDD1/Project_Data/demoMaterial/images/street_scene/ --model yolo26n --batch

# Process with Cosmos-Nemotron (loads once, then processes all images)
vlm-demo run /mnt/HDD1/Project_Data/demoMaterial/images/street_scene/ --model cosmos_nemotron --batch

# Run ALL models on a folder
vlm-demo run /mnt/HDD1/Project_Data/demoMaterial/images/animals/ --model all --batch

# Disable batch mode (load model per image)
vlm-demo run /path/to/folder/ --model yolo26n --no-batch

# Search subdirectories
vlm-demo run /path/to/ --model yolo26n --batch --recurse
```

Output includes per-image inference time and results summary.

---

## Custom VLM Prompt (`vlm` command)

Run any VLM model with a custom prompt in natural language:

```bash
# Florence-2 task prompts
vlm-demo vlm florence2 image.jpg "<CAPTION>"
vlm-demo vlm florence2 image.jpg "<DETAILED_CAPTION>"
vlm-demo vlm florence2 image.jpg "<OD>"

# Cosmos physical reasoning
vlm-demo vlm cosmos_nemotron image.jpg "Analyze the spatial relationships"
vlm-demo vlm cosmos_nemotron image.jpg "What physical laws apply here?"

# Phi-3.5 document understanding
vlm-demo vlm phi_vision image.jpg "Extract and summarize all visible text"
vlm-demo vlm phi_vision chart.png "Interpret this data visualization"
```

---

## Cross-Model Commands

### Side-by-Side Comparison
```
vlm-demo compare image.jpg
```
Runs all compatible models and prints results.

### Full Scene Analysis
```
vlm-demo scene image.jpg
```
Runs Qwen3-Instruct + Qwen3-Thinking + YOLO on one image.

---

## Batch Processing

The `batch` command loads a model **once**, reports load time, then processes each image:

```bash
$ vlm-demo batch yolo26n bus.jpg zidane.jpg
  Model loaded in 0.4s  |  2 image(s)
  [bus.jpg] 5 objects  (0.08s)
  [zidane.jpg] 2 objects  (0.06s)
```

For VLM models (cosmos_nemotron, locate_anything, qwen3_instruct, qwen3_thinking),
a persistent subprocess keeps the model loaded across requests.

---

## Demo Material

Images are organized by theme — no images in the root `images/` folder.

| Category | Path | Images | Description |
|----------|------|--------|-------------|
| Animals | `demoMaterial/images/animals/` | 10 | Multi-object animal scenes |
| Indoor Scenes | `demoMaterial/images/indoor_scenes/` | 10 | Rooms, furniture, people |
| People Actions | `demoMaterial/images/people_actions/` | 16 | Groups, activities, interactions |
| Sports | `demoMaterial/images/sports/` | 10 | Sports equipment, athletes |
| Street Scene | `demoMaterial/images/street_scene/` | 11 | Traffic, pedestrians, vehicles |
| COCO val2017 | `public_datasets/coco/val2017/` | 5000 | General 80-class dataset |
| DOTA v1.0 | `public_datasets/dotav1/images/` | 1869 | Aerial OBB dataset |

| Video | Use Case |
|-------|----------|
| `vtest.avi` | Office scene tracking |
| `walking_people.mp4` | Pedestrian tracking |
| `car_traffic.mp4` | Vehicle detection |
| `people_crossing.mp4` | Crowded intersection |

All outputs: `/tmp/vlm_demo/`

---

## ROS2 Live Webcam

```bash
source ros2_ws/install/setup.bash
ros2 run vlm_webcam_demo vlm_webcam_node --ros-args -p model:=yolo26n -p confidence:=0.25
```

Rebuild: `colcon build --packages-select vlm_webcam_demo` in `ros2_ws/`
