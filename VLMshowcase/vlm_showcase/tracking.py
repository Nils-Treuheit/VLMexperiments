from pathlib import Path
from .config import YOLO_MODEL_PATHS, DEMO_MATERIAL


def track_video(model_key, video_path, conf=0.25, save_path=None):
    from ultralytics import YOLO
    pt_path = YOLO_MODEL_PATHS.get(model_key)
    if not pt_path or not pt_path.exists():
        available = [k for k, v in YOLO_MODEL_PATHS.items() if v.exists()]
        raise FileNotFoundError(f"Model '{model_key}' not found. Available: {available}")
    model = YOLO(str(pt_path))
    results = model.track(
        str(video_path),
        conf=conf,
        persist=True,
        save=True if save_path else False,
        project=str(save_path.parent) if save_path else None,
        name=save_path.stem if save_path else None,
        verbose=False,
    )
    return results


def get_tracking_summary(results):
    track_data = {}
    for r in results:
        if r.boxes is not None and r.boxes.id is not None:
            for box, track_id, cls, conf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.id.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
            ):
                tid = int(track_id)
                cname = r.names[int(cls)]
                if tid not in track_data:
                    track_data[tid] = {
                        "class": cname,
                        "first_frame": r.path,
                        "positions": [],
                    }
                track_data[tid]["positions"].append(box.tolist())
    return track_data
