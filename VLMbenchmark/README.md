# Benchmark

Multi-task benchmark for visual models on COCO and DOTA-v1.0.

## Tasks

| Script | Task | Dataset | Models |
|--------|------|---------|--------|
| `benchmark_od.py` | Object Detection | COCO val2017 | locate_anything, qwen3_native, qwen3_thinking, yolo26 |
| `benchmark_od.py` | Object Detection | DOTA-v1.0 | locate_anything, qwen3_native, qwen3_thinking |
| `benchmark_pose.py` | Pose Estimation | COCO Keypoints | yolo26_pose |
| `benchmark_obb.py` | Oriented BBox | DOTA-v1.0 | yolo26_obb |
| `benchmark_grounding.py` | Phrase Grounding | COCO val2017 | locate_anything, qwen3_native, qwen3_thinking |
| `benchmark_all.py` | All supported tasks | — | All models |

## Quick Start

Each model must be run with **its own virtual environment**:

### Object Detection (COCO)
```bash
# LocateAnything-3B
/mnt/HDD1/Project_Code/vlm_det_test/locate_anything/.venv/bin/python \
    scripts/benchmark_od.py --model locate_anything --max-images 100

# Qwen3-VL-8B-Instruct
/mnt/HDD1/Project_Code/vlm_det_test/qwen3-vl_instruct/.venv/bin/python \
    scripts/benchmark_od.py --model qwen3_native --max-images 100

# Qwen3-VL-8B-Thinking
/mnt/HDD1/Project_Code/vlm_det_test/qwen3-vl_thinking/.venv/bin/python \
    scripts/benchmark_od.py --model qwen3_thinking --max-images 100

# YOLO11n
/mnt/HDD1/Project_Code/vlm_det_test/yolo11-26/.venv/bin/python \
    scripts/benchmark_od.py --model yolo26 --max-images 100
```

### Pose Estimation
```bash
/mnt/HDD1/Project_Code/vlm_det_test/yolo11-26/.venv/bin/python \
    scripts/benchmark_pose.py --model yolo26_pose --max-images 100
```

### OBB Detection (DOTA)
```bash
/mnt/HDD1/Project_Code/vlm_det_test/yolo11-26/.venv/bin/python \
    scripts/benchmark_obb.py --model yolo26_obb --max-images 100
```

### Run Everything
```bash
/mnt/HDD1/Project_Code/vlm_det_test/yolo11-26/.venv/bin/python \
    scripts/benchmark_all.py --max-images 25
```

## Models

| Key | Model | Tasks | Venv |
|-----|-------|-------|------|
| `locate_anything` | LocateAnything-3B | OD, Grounding | `locate_anything/.venv` |
| `qwen3_native` | Qwen3-VL-8B-Instruct | OD, Grounding, OD(DOTA) | `qwen3-vl_instruct/.venv` |
| `qwen3_thinking` | Qwen3-VL-8B-Thinking | OD, Grounding, OD(DOTA) | `qwen3-vl_thinking/.venv` |
| `yolo26` | YOLO26n (Detect) | OD | `yolo11-26/.venv` |
| `yolo26_pose` | YOLO26n (Pose) | Pose | `yolo11-26/.venv` |
| `yolo26_obb` | YOLO26n (OBB) | OBB | `yolo11-26/.venv` |

## Aliases

`la`=`locate_anything`, `qwen3`=`qwen3_native`, `thinking`=`qwen3_thinking`,
`yolo`/`yolo26`=`yolo26`, `yolo11`=`yolo11`, `yolo_pose`=`yolo26_pose`, `yolo_obb`=`yolo26_obb`

## Datasets

| Dataset | Path | Content |
|---------|------|---------|
| COCO val2017 | `/mnt/HDD1/Project_Data/public_datasets/coco/` | 5000 images, 80 classes |
| COCO Keypoints | `/mnt/HDD1/Project_Data/public_datasets/coco/` | 5000 images, 17 keypoints |
| DOTA-v1.0 | `/mnt/HDD1/Project_Data/public_datasets/dotav1/` | Aerial images, 15 classes |

Download DOTA: `python download_datasets.py --dota 200`

## Results

Stored in `results/` as `{model}_{task}_stats.json`.
