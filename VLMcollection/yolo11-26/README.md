# YOLO11-26 — Training & Inference

Ultralytics YOLO11 + YOLO26 detection, OBB, and pose with pre-downloaded weights.

## Setup

```bash
source .venv/bin/activate
```

## Pre-downloaded Models

All weights in `models/` — no first-run download needed.

| File | Model | Task |
|------|-------|------|
| `yolo11n.pt` – `yolo11x.pt` | YOLO11 n/s/m/l/x | Detection |
| `yolo11n-pose.pt` / `yolo11s-pose.pt` | YOLO11 n/s | Pose |
| `yolo11n-obb.pt` / `yolo11s-obb.pt` | YOLO11 n/s | OBB |
| `yolo26n.pt` – `yolo26x.pt` | YOLO26 n/s/m/l/x | Detection |
| `yolo26n-pose.pt` / `yolo26s-pose.pt` | YOLO26 n/s | Pose |
| `yolo26n-obb.pt` / `yolo26s-obb.pt` | YOLO26 n/s | OBB |

## Training

```bash
python3 train.py --data data/coco.yaml --model yolo26m --epochs 100
```

Arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | (required) | Path to `data.yaml` |
| `--model` | `yolo26m` | Model size (`yolo26n/s/m/l/x` or `yolo11n/s/m/l/x`) |
| `--epochs` | `100` | Number of epochs |
| `--batch` | `16` | Batch size |
| `--imgsz` | `640` | Input image size |
| `--device` | `0` | GPU device(s) or `cpu` |
| `--lr` | `0.01` | Initial learning rate |
| `--resume` | — | Resume from last checkpoint |
| `--freeze` | `0` | Freeze first N backbone layers |
| `--patience` | `50` | Early stopping patience (0 = disable) |
| `--project` | `runs/` | Output directory |
| `--name` | auto | Experiment subdirectory |

## Inference

```bash
python3 predict.py --model models/yolo26m.pt --source image.jpg --save-txt
```

Arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | `.pt` path or built-in name (`yolo26m`) |
| `--source` | (required) | Image, video, directory, or `0` for webcam |
| `--imgsz` | `640` | Inference size |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--device` | `0` | GPU or `cpu` |
| `--save-txt` | — | Save labels as `.txt` |
| `--max-det` | `300` | Max detections per image |
| `--half` | — | FP16 half-precision |
| `--name` | auto | Output subdirectory under `runs/` |

## Validation

```bash
python3 val.py --model runs/train/exp/weights/best.pt --data data/coco.yaml
```

Arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Trained `.pt` path |
| `--data` | (required) | Path to `data.yaml` |
| `--imgsz` | `640` | Image size |
| `--batch` | `16` | Batch size |
| `--device` | `0` | GPU or `cpu` |
| `--conf` | `0.001` | Confidence threshold |
| `--iou` | `0.6` | NMS IoU for eval |
| `--half` | — | FP16 |
| `--save-json` | — | Save COCO JSON results |
| `--plots` | yes | Generate plots |

## DOTA Label Conversion

Converts DOTA native (oriented) labels to YOLO format:

```bash
python3 scripts/convert_dota.py --obb  # OBB format (default: horizontal bbox)
```

## Data Configs

- `data/coco.yaml` — COCO dataset (ready to use)
- `data/dota.yaml` — DOTA v1 (convert labels first)
- `data_template.yaml` — template for custom datasets

## Benchmark

See `../Benchmark/` for multi-model comparison (YOLO11, YOLO26, Qwen3, LocateAnything).
