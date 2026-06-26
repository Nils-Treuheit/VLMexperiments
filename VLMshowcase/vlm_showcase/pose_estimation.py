from pathlib import Path
import numpy as np
from .config import YOLO_MODEL_PATHS


def estimate_pose(model_key, image_path, conf=0.25):
    from ultralytics import YOLO
    pt_path = YOLO_MODEL_PATHS.get(model_key)
    if not pt_path or not pt_path.exists():
        available = [k for k, v in YOLO_MODEL_PATHS.items() if v.exists()]
        raise FileNotFoundError(f"Model '{model_key}' not found. Available: {available}")
    model = YOLO(str(pt_path))
    results = model(str(image_path), conf=conf, verbose=False)
    poses = []
    for r in results:
        if r.keypoints is not None:
            kp_data = r.keypoints.data.cpu().numpy()
            for person_kps in kp_data:
                poses.append(person_kps.tolist())
    return poses, results


def estimate_pose_multi(models, image_path, conf=0.25):
    all_poses = {}
    for key in models:
        poses, _ = estimate_pose(key, image_path, conf)
        all_poses[key] = poses
    return all_poses
