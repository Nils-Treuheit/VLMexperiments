from pathlib import Path
from .config import YOLO_MODEL_PATHS


def detect_obb(model_key, image_path, conf=0.25):
    from ultralytics import YOLO
    pt_path = YOLO_MODEL_PATHS.get(model_key)
    if not pt_path or not pt_path.exists():
        available = [k for k, v in YOLO_MODEL_PATHS.items() if v.exists()]
        raise FileNotFoundError(f"Model '{model_key}' not found. Available: {available}")
    model = YOLO(str(pt_path))
    results = model(str(image_path), conf=conf, verbose=False)
    obb_detections = []
    for r in results:
        if r.obb is not None:
            obb_data = r.obb.data.cpu().numpy()
            for det in obb_data:
                cx, cy, w, h, theta = det[0], det[1], det[2], det[3], det[4]
                cls_id = int(det[6])
                conf_val = float(det[5])
                obb_detections.append({
                    "cx": float(cx), "cy": float(cy),
                    "w": float(w), "h": float(h),
                    "theta": float(theta),
                    "confidence": conf_val,
                    "class_id": cls_id,
                    "class_name": r.names.get(cls_id, str(cls_id)),
                })
    return obb_detections, results
