import json
import numpy as np
from pathlib import Path
from .subprocess_utils import run_with_timer
from .config import MODELS, YOLO_MODEL_PATHS, COCO_DIR


def _yolo_infer(model_key, image_path, conf=0.25):
    from ultralytics import YOLO
    pt_path = YOLO_MODEL_PATHS.get(model_key)
    if not pt_path or not pt_path.exists():
        available = [k for k, v in YOLO_MODEL_PATHS.items() if v.exists()]
        raise FileNotFoundError(f"Model '{model_key}' not found. Available: {available}")
    model = YOLO(str(pt_path))
    results = model(str(image_path), conf=conf, verbose=False)
    return results


def detect_yolo(model_key, image_path, conf=0.25):
    results = _yolo_infer(model_key, image_path, conf)
    detections = []
    for r in results:
        if r.boxes is not None:
            for box, cls, conf_val in zip(r.boxes.xyxy.cpu().numpy(),
                                           r.boxes.cls.cpu().numpy(),
                                           r.boxes.conf.cpu().numpy()):
                detections.append({
                    "bbox": box.tolist(),
                    "class_id": int(cls),
                    "class_name": r.names[int(cls)],
                    "confidence": float(conf_val),
                })
    return detections, results


def detect_yolo_multi(model_keys, image_path, conf=0.25):
    all_detections = {}
    for key in model_keys:
        dets, _ = detect_yolo(key, image_path, conf)
        all_detections[key] = dets
    return all_detections


def ground_with_locate_anything(image_path, query):
    cfg = MODELS["locate_anything"]
    out_path = str(Path(image_path).parent / f"_la_{Path(image_path).stem}.jpg")
    result = run_with_timer(
        [cfg["venv_python"], cfg["script"], image_path, query,
         "--json", "--output", out_path],
        timeout=1000, label="Loading LocateAnything",
    )
    try:
        return json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return {"text": result.stdout.strip(), "boxes": []}


def compare_detectors(image_path, yolo_model="yolo26n", queries=None):
    if queries is None:
        queries = ["person", "car", "dog", "chair"]
    results = {}
    yolo_dets, _ = detect_yolo(yolo_model, image_path)
    results[f"YOLO_{yolo_model}"] = yolo_dets
    for q in queries:
        la_result = ground_with_locate_anything(image_path, q)
        results[f"LA_{q}"] = la_result
    return results


def get_coco_sample(sample_id=None):
    import random
    val_dir = COCO_DIR / "val2017"
    images = sorted(val_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {val_dir}")
    if sample_id is not None and sample_id < len(images):
        return images[sample_id]
    return random.choice(images)
