import json
import subprocess
import time
import sys
from pathlib import Path
from .config import MODELS, YOLO_MODEL_PATHS

VLM_KEYS = {"locate_anything", "qwen3_instruct", "qwen3_thinking"}
SERVER_SCRIPT = str(Path(__file__).parent.parent / "scripts" / "_model_server.py")


def _start_server(model_key, mode="describe"):
    cfg = MODELS[model_key]
    proc = subprocess.Popen(
        [cfg["venv_python"], SERVER_SCRIPT, model_key, mode],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1,
    )
    line = proc.stdout.readline()
    ready = json.loads(line)
    if "error" in ready:
        proc.kill()
        raise RuntimeError(ready["error"])
    return proc, ready["load_time_sec"]


def _server_process(proc, image_path, prompt=None):
    req = json.dumps({"image": str(image_path), "prompt": prompt or ""})
    proc.stdin.write(req + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    resp = json.loads(line)
    return resp["result"], resp["time_sec"]


def _stop_server(proc):
    if proc:
        try:
            proc.stdin.write("\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def _batch_yolo(model_key, images, conf=0.25):
    from ultralytics import YOLO
    pt = YOLO_MODEL_PATHS.get(model_key)
    if not pt or not pt.exists():
        raise FileNotFoundError(f"YOLO model '{model_key}' not found")
    t0 = time.time()
    model = YOLO(str(pt))
    load_time = time.time() - t0
    results = []
    for img in images:
        t1 = time.time()
        r = model(str(img), conf=conf, verbose=False)
        dets = []
        for pred in r:
            if pred.boxes is not None:
                for box, cls, c in zip(
                    pred.boxes.xyxy.cpu().numpy(),
                    pred.boxes.cls.cpu().numpy(),
                    pred.boxes.conf.cpu().numpy(),
                ):
                    dets.append({
                        "bbox": box.tolist(),
                        "class": pred.names[int(cls)],
                        "confidence": float(c),
                    })
        inf_time = time.time() - t1
        results.append({"image": str(img), "detections": dets, "time_sec": round(inf_time, 3)})
    return round(load_time, 2), results


def _batch_yolo_pose(model_key, images, conf=0.25):
    from ultralytics import YOLO
    pt = YOLO_MODEL_PATHS.get(model_key)
    if not pt or not pt.exists():
        raise FileNotFoundError(f"YOLO model '{model_key}' not found")
    t0 = time.time()
    model = YOLO(str(pt))
    load_time = time.time() - t0
    results = []
    for img in images:
        t1 = time.time()
        r = model(str(img), conf=conf, verbose=False)
        poses = []
        dets = []
        for pred in r:
            if pred.keypoints is not None:
                for kps in pred.keypoints.data.cpu().numpy():
                    poses.append(kps.tolist())
            if pred.boxes is not None:
                for box, cls, c in zip(
                    pred.boxes.xyxy.cpu().numpy(),
                    pred.boxes.cls.cpu().numpy(),
                    pred.boxes.conf.cpu().numpy(),
                ):
                    dets.append({
                        "bbox": box.tolist(),
                        "class": pred.names[int(cls)],
                        "confidence": float(c),
                    })
        inf_time = time.time() - t1
        results.append({
            "image": str(img), "detections": dets, "poses": poses,
            "time_sec": round(inf_time, 3),
        })
    return round(load_time, 2), results


def batch_process(model_key, images, prompts=None, conf=0.25, mode=None):
    is_vlm = model_key in VLM_KEYS

    if not is_vlm and mode == "pose":
        return _batch_yolo_pose(model_key, images, conf)

    if not is_vlm:
        return _batch_yolo(model_key, images, conf)

    server_mode = mode or ("detect" if model_key == "locate_anything" else "describe")
    proc = None
    try:
        proc, load_time = _start_server(model_key, server_mode)
        results = []
        for i, img in enumerate(images):
            p = prompts[i] if prompts and i < len(prompts) else None
            result, inf_time = _server_process(proc, img, p)
            results.append({"image": str(img), "result": result, "time_sec": inf_time})
        return load_time, results
    finally:
        _stop_server(proc)
