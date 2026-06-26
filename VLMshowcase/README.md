# VLMshowcase

Vision-Language Model Capability Demo — runs four models installed on this system side-by-side.

```
source .venv/bin/activate
vlm-demo <command> [args]
```

## Models Overview

| Model | Params | Type | VRAM | Load Time |
|-------|--------|------|------|-----------|
| **YOLO11/26** (Ultralytics) | 2.7M–68M | CNN detector | <1 GB | <1s |
| **LocateAnything-3B** (NVIDIA) | 3B | Specialized VLM (grounding) | ~8 GB | ~30s |
| **Qwen3-VL-8B-Instruct** (Alibaba) | 8.8B | General VLM | ~30 GB | ~15s |
| **Qwen3-VL-8B-Thinking** (Unsloth 4-bit) | ~9B | Reasoning VLM | ~6 GB | ~45s |

All four models are pre-installed at `/mnt/HDD1/Project_Code/vlm_det_test/`. The showcase calls heavyweight VLMs via subprocess through their own venvs (incompatible `transformers` deps), and loads YOLO directly via `ultralytics`.

---

## YOLO11/26 — Commands

Lightning-fast CNN-based detection, pose, OBB, and tracking. 14 pre-downloaded weights at `vlm_det_test/yolo11-26/models/`.

### Object Detection
```
vlm-demo detect image.jpg --yolo yolo26n yolo26s yolo11m
vlm-demo detect image.jpg --yolo yolo26x
```

### Pose Estimation (17-keypoint COCO skeleton)
```
vlm-demo pose image.jpg --model yolo26n-pose yolo11n-pose
vlm-demo pose image.jpg --model yolo26s-pose
```

### Oriented Bounding Box Detection (DOTA aerial)
```
vlm-demo obb dota_image.png --model yolo26n-obb yolo11n-obb
vlm-demo obb dota_image.png --model yolo26s-obb
```
DOTA images: `/mnt/HDD1/Project_Data/public_datasets/dotav1/images/`

### Multi-Object Tracking
```
vlm-demo track video.mp4 --model yolo26n
vlm-demo track walking_people.mp4 --model yolo11n
```
Demo videos: `/mnt/HDD1/Project_Data/demoMaterial/videos/`

### Live Webcam
```
vlm-demo webcam --model yolo26n --fps 15
vlm-demo webcam --model yolo11n --fps 20
```
Press `q` to quit.

### Batch Processing (load once, process many)
```
vlm-demo batch yolo26n img1.jpg img2.jpg img3.jpg
vlm-demo batch yolo26n-pose img1.jpg img2.jpg --mode pose
```

---

## LocateAnything-3B — Commands

NVIDIA's free-form visual grounding model. No fixed class set — describe any object in natural language.

### Grounded Detection
```
vlm-demo detect image.jpg --grounding "person" "car" "red cup"
```

### Text Detection
```
vlm-demo detect document.jpg --grounding "Detect all text in box format"
```

### Batch Grounding (load once, process many)
```
vlm-demo batch locate_anything img1.jpg img2.jpg --prompt "person" "car"
vlm-demo batch locate_anything *.jpg --prompt "find all animals"
```

### Multi-class Detection (using `<c>` separator)
_(query through `--grounding`)_
```
vlm-demo detect scene.jpg --grounding "person</c>car</c>bicycle"
```

---

## Qwen3-VL-8B-Instruct — Commands

Alibaba's general-purpose VLM. Strong at scene description, OCR, and visual question answering.

### Scene Description
```
vlm-demo scene image.jpg
vlm-demo compare image.jpg
```

### Visual Question Answering
```
vlm-demo detect image.jpg --grounding "What color is the car?"
```
_(Any question works — the model answers in natural language.)_

### Object Detection (via text prompting)
```
vlm-demo scene image.jpg     # includes detection pass
```

### OCR (32 languages)
```
vlm-demo detect receipt.jpg --grounding "Read all the text in this image"
```

### Video Understanding
_(Model supports video natively — feed frames.)_

### Batch Description
```
vlm-demo batch qwen3_instruct img1.jpg img2.jpg --mode describe
vlm-demo batch qwen3_instruct img1.jpg img2.jpg --prompt "Describe" "What objects?"
```

---

## Qwen3-VL-8B-Thinking — Commands

Unsloth 4-bit quantized reasoning VLM. Shows chain-of-thought before answering. Best for intent, action recognition, and complex reasoning.

### Reasoning-Based Scene Description
```
vlm-demo scene image.jpg
vlm-demo compare image.jpg
```

### Human Intent & Action Recognition
```
vlm-demo intent image.jpg --aspect action
vlm-demo intent image.jpg --aspect intent
vlm-demo intent image.jpg --aspect emotion
vlm-demo intent image.jpg --aspect social
vlm-demo intent image.jpg --all              # all four aspects
```

### Object Detection (JSON output)
```
vlm-demo batch qwen3_thinking img.jpg --mode detect
```

### Batch Reasoning
```
vlm-demo batch qwen3_thinking img1.jpg img2.jpg --mode describe \
  --prompt "Analyze this scene" "What is happening?"
```

---

## Cross-Model Commands

### Side-by-Side Comparison
Runs all models on one image and prints results:
```
vlm-demo compare image.jpg
```
This calls Qwen3-Instruct (description), Qwen3-Thinking (reasoning), YOLO (3 sizes), and LocateAnything (4 queries).

### Full Scene Analysis
```
vlm-demo scene image.jpg
```
Runs Qwen3-Instruct (description + detection), Qwen3-Thinking (reasoning description).

---

## Batch Processing Details

The `batch` command loads a model into memory **once**, reports the load time, then processes each image and reports per-image inference time:

```
$ vlm-demo batch yolo26n bus.jpg zidane.jpg
  ...
  Model loaded in 0.4s  |  2 image(s)

  [bus.jpg]  5 objects  (0.08s)
  [zidane.jpg]  2 objects  (0.06s)
```

For VLM models, a persistent subprocess keeps the model loaded:
```
$ vlm-demo batch locate_anything bus.jpg zidane.jpg --prompt person person
  ...
  Model loaded in 30.2s  |  2 image(s)

  [bus.jpg]  (0.5s)
    <ref>person</ref><box>...
  [zidane.jpg]  (0.4s)
    <ref>person</ref><box>...
```

---

## Demo Material

| Source | Path | Contents |
|--------|------|----------|
| COCO val2017 | `/mnt/HDD1/Project_Data/public_datasets/coco/val2017` | 5000 images, 80 classes |
| DOTA v1.0 | `/mnt/HDD1/Project_Data/public_datasets/dotav1/images` | 1869 aerial images, 15 OBB classes |
| Demo images | `/mnt/HDD1/Project_Data/demoMaterial/images/` | bus.jpg, zidane.jpg, person.jpg, COCO samples |
| Demo videos | `/mnt/HDD1/Project_Data/demoMaterial/videos/` | vtest.avi, walking_people.mp4, car_traffic.mp4 |
| YOLO weights | `/mnt/HDD1/Project_Code/vlm_det_test/yolo11-26/models/` | 14 models (detect, pose, OBB) |

Output directory: `/tmp/vlm_demo/`

## ROS2 Live Webcam

A ROS2 Humble node publishes annotated video to `/vlm_detections`:

```bash
source /mnt/HDD1/Project_Code/VLMshowcase/ros2_ws/install/setup.bash
ros2 run vlm_webcam_demo vlm_webcam_node --ros-args -p model:=yolo26n -p confidence:=0.25
```

Rebuild after edits:
```bash
source /opt/ros/humble/setup.bash
cd /mnt/HDD1/Project_Code/VLMshowcase/ros2_ws
colcon build --packages-select vlm_webcam_demo
```
