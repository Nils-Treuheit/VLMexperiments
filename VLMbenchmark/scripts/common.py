import json
import os
import re
import sys
import warnings
from pathlib import Path

from PIL import Image

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = Path("/mnt/HDD1/Project_Data/public_datasets")
COCO_DIR = DATA_DIR / "coco"
DOTA_DIR = DATA_DIR / "dotav1"
PROJECT_DIR = BASE_DIR.parent

DOTA_CATEGORIES = [
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field",
    "roundabout", "harbor", "swimming-pool", "helicopter",
]
DOTA_CAT_NAME_TO_ID = {n: i + 1 for i, n in enumerate(DOTA_CATEGORIES)}


def parse_box_tags(text):
    boxes = []
    for m in re.finditer(r'<box>(.+?)</box>', text, re.IGNORECASE):
        coords = [float(p) for p in re.findall(r'[\d.]+', m.group(1))]
        if len(coords) == 4:
            boxes.append(coords)
    return boxes


def parse_json_detections(text, target_label=None):
    import json as _json
    text_lower = text.lower()
    if re.search(r'no\s+(?:object|person|instances|people|one)|not\s+\w+\s+(?:object|person|instances)',
                 text_lower):
        if not re.search(r'\[.*?(?:\d|bbox|label)', text_lower):
            return []
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        depth = 0
        start = None
        raw = None
        for i, c in enumerate(text):
            if c == '[':
                if start is None:
                    start = i
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0 and start is not None:
                    raw = text[start:i + 1]
                    break
        if raw is None:
            return []
    try:
        dets = _json.loads(raw)
    except _json.JSONDecodeError:
        return []
    if not isinstance(dets, list):
        return []
    out = []
    for d in dets:
        if not isinstance(d, dict):
            continue
        bbox = d.get("bbox_2d") or d.get("bbox") or d.get("box")
        if not bbox or len(bbox) != 4:
            continue
        if target_label is not None:
            lab = str(d.get("label", "")).lower().strip()
            tgt = target_label.lower().strip()
            if lab != tgt and tgt not in lab and lab not in tgt:
                continue
        out.append([float(v) for v in bbox])
    return out


def extract_narrative_boxes(text):
    boxes = []
    seen = set()
    for m in re.finditer(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text):
        key = (m.group(1), m.group(2), m.group(3), m.group(4))
        if key not in seen:
            seen.add(key)
            boxes.append([float(v) for v in key])
    return boxes


def scale_la(boxes, ow, oh):
    return [[x1 / 1000 * ow, y1 / 1000 * oh, x2 / 1000 * ow, y2 / 1000 * oh]
            for x1, y1, x2, y2 in boxes]


def scale_qwen(boxes, ow, oh):
    out = []
    for x1, y1, x2, y2 in boxes:
        mx = max(x1, y1, x2, y2)
        if mx <= 1.0:
            x1 *= ow; y1 *= oh; x2 *= ow; y2 *= oh
        elif mx > max(ow, oh):
            s = max(ow, oh) / mx
            x1 *= s; y1 *= s; x2 *= s; y2 *= s
        out.append([x1, y1, x2, y2])
    return out


def scale_thinking(boxes, ow, oh):
    return scale_la(boxes, ow, oh)


def load_dota_coco_gt(dota_dir, max_images=None):
    images_dir = dota_dir / "images"
    labels_dir = dota_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return None

    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        return None
    if max_images:
        image_files = image_files[:max_images]

    out = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(DOTA_CATEGORIES)],
    }
    aid = 1
    for iid, imp in enumerate(image_files, start=1):
        lp = labels_dir / f"{imp.stem}.txt"
        if not lp.exists():
            continue
        try:
            with Image.open(imp) as im:
                w, h = im.size
        except Exception:
            continue
        out["images"].append({"id": iid, "file_name": imp.name, "width": w, "height": h})
        with open(lp) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "imagesource", "gsd")):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                try:
                    coords = [float(p) for p in parts[:8]]
                    cname = parts[8]
                    diff = int(parts[9]) if len(parts) > 9 else 0
                except (ValueError, IndexError):
                    continue
                if diff != 0 or cname not in DOTA_CAT_NAME_TO_ID:
                    continue
                cat_id = DOTA_CAT_NAME_TO_ID[cname]
                xs, ys = coords[0::2], coords[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                out["annotations"].append({
                    "id": aid, "image_id": iid,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                })
                aid += 1
    return out


def load_la():
    warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")
    warnings.filterwarnings("ignore", message=".*image_processor_class.*")
    sys.path.insert(0, str(PROJECT_DIR / "locate_anything"))
    import infer
    mp = str(PROJECT_DIR / "locate_anything" / "model")
    dev = "cuda" if torch.cuda.is_available() else None
    return infer.LocateAnythingWorker(mp, device=dev), infer


def load_qwen3():
    _logging = __import__("logging")
    for msg in [".*_check_is_size.*", ".*Python version.*", ".*parameters are on the meta device.*",
                ".*causal_conv1d was requested.*", ".*The fast path is not available.*",
                ".*copy construct from a tensor.*", ".*recommended to use sourceTensor.detach.*"]:
        warnings.filterwarnings("ignore", message=msg)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _logging.getLogger("transformers_modules").setLevel(_logging.ERROR)
    _logging.getLogger("fla").setLevel(_logging.ERROR)

    sys.path.insert(0, str(PROJECT_DIR / "qwen3-vl_instruct"))
    from infer_qwen3 import resolve_model_path, load_model as _load_qwen3_model

    model_path = resolve_model_path()
    if not os.path.exists(model_path) or not os.listdir(model_path):
        raise FileNotFoundError(f"Qwen3-VL model not found at {model_path}")

    dev = "cuda" if torch.cuda.is_available() else None
    model, processor = _load_qwen3_model(model_path, device=dev)
    return (processor, model), {}


def load_qwen3_thinking():
    sys.path.insert(0, str(PROJECT_DIR / "qwen3-vl_thinking"))
    from qwen_detector import QwenVLDetector
    return QwenVLDetector(max_seq_length=2048), {}


def load_yolo26(model_name="yolo26n"):
    sys.path.insert(0, str(PROJECT_DIR / "yolo11-26"))
    from ultralytics import YOLO
    # Prefer local pre-downloaded weights; fall back to ultralytics built-in
    models_dir = PROJECT_DIR / "yolo11-26" / "models"
    local = models_dir / f"{model_name}.pt"
    if local.exists():
        return YOLO(str(local)), {}
    return YOLO(model_name), {}


MODEL_LOADERS = {
    "locate_anything": load_la,
    "qwen3_native": load_qwen3,
    "qwen3_thinking": load_qwen3_thinking,
    # YOLO26 detection
    "yolo26": lambda: load_yolo26("yolo26n"),
    "yolo26s": lambda: load_yolo26("yolo26s"),
    "yolo26m": lambda: load_yolo26("yolo26m"),
    "yolo26l": lambda: load_yolo26("yolo26l"),
    "yolo26x": lambda: load_yolo26("yolo26x"),
    # YOLO26 pose
    "yolo26_pose": lambda: load_yolo26("yolo26n-pose"),
    "yolo26s_pose": lambda: load_yolo26("yolo26s-pose"),
    # YOLO26 OBB
    "yolo26_obb": lambda: load_yolo26("yolo26n-obb"),
    "yolo26s_obb": lambda: load_yolo26("yolo26s-obb"),
    # YOLO11 detection (legacy, for comparison)
    "yolo11": lambda: load_yolo26("yolo11n"),
    "yolo11s": lambda: load_yolo26("yolo11s"),
    "yolo11m": lambda: load_yolo26("yolo11m"),
    "yolo11l": lambda: load_yolo26("yolo11l"),
    "yolo11x": lambda: load_yolo26("yolo11x"),
    # YOLO11 pose
    "yolo11_pose": lambda: load_yolo26("yolo11n-pose"),
    "yolo11s_pose": lambda: load_yolo26("yolo11s-pose"),
    # YOLO11 OBB
    "yolo11_obb": lambda: load_yolo26("yolo11n-obb"),
    "yolo11s_obb": lambda: load_yolo26("yolo11s-obb"),
}

MODEL_ALIASES = {
    "la": "locate_anything",
    "qwen3": "qwen3_native",
    "qwen3_vl_instruct": "qwen3_native",
    "qwen3_vl_thinking": "qwen3_thinking",
    "thinking": "qwen3_thinking",
    "yolo": "yolo26",
    "yolo26n": "yolo26",
    "yolo11n": "yolo11",
    "ultralytics": "yolo26",
    "yolo_pose": "yolo26_pose",
    "yolo_obb": "yolo26_obb",
}

MODEL_DISPLAY = {
    "locate_anything": "LocateAnything-3B",
    "qwen3_native": "Qwen3-VL-8B-Instruct",
    "qwen3_thinking": "Qwen3-VL-8B-Thinking",
    "yolo26": "YOLO26n (Detect)",
    "yolo26s": "YOLO26s (Detect)",
    "yolo26m": "YOLO26m (Detect)",
    "yolo26l": "YOLO26l (Detect)",
    "yolo26x": "YOLO26x (Detect)",
    "yolo26_pose": "YOLO26n (Pose)",
    "yolo26s_pose": "YOLO26s (Pose)",
    "yolo26_obb": "YOLO26n (OBB)",
    "yolo26s_obb": "YOLO26s (OBB)",
    "yolo11": "YOLO11n (Detect)",
    "yolo11s": "YOLO11s (Detect)",
    "yolo11m": "YOLO11m (Detect)",
    "yolo11l": "YOLO11l (Detect)",
    "yolo11x": "YOLO11x (Detect)",
    "yolo11_pose": "YOLO11n (Pose)",
    "yolo11s_pose": "YOLO11s (Pose)",
    "yolo11_obb": "YOLO11n (OBB)",
    "yolo11s_obb": "YOLO11s (OBB)",
}


def build_prompt(category_name, model_type):
    if model_type == "locate_anything":
        return category_name
    if model_type == "qwen3_native":
        return (
            f"List ALL instances of '{category_name}' in this image. "
            f"Be exhaustive - do not miss any. Output each as <box>x1,y1,x2,y2</box>. "
            f"Place each box on its own line."
        )
    if model_type == "qwen3_thinking":
        return (
            f"List ALL instances of '{category_name}' in this image. "
            f"Be exhaustive - do not miss any. Output each as <box>x1,y1,x2,y2</box>. "
            f"Place each box on its own line."
        )
    return category_name


TASK_ROWS = {
    "object_detection": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("mAP@50:95", "mAP@50:95", "{:.4f}"),
        ("mAP@50", "mAP@50", "{:.4f}"),
        ("images", "Images processed", "{}"),
        ("total_gt", "Total GT objects", "{}"),
        ("total_detected", "Total detected", "{}"),
    ],
    "pose_estimation": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("AP@50:95_keypoints", "AP@50:95 (keypoints)", "{:.4f}"),
        ("AP@50_keypoints", "AP@50 (keypoints)", "{:.4f}"),
        ("images", "Images processed", "{}"),
        ("total_gt_persons", "Total GT persons", "{}"),
        ("total_detected_persons", "Detected persons", "{}"),
    ],
    "obb_detection": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("mAP@50:95", "mAP@50:95", "{:.4f}"),
        ("mAP@50", "mAP@50", "{:.4f}"),
        ("images", "Images processed", "{}"),
        ("total_gt", "Total GT objects", "{}"),
        ("total_detected", "Total detected", "{}"),
    ],
    "grounding": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("acc@50", "Acc@50", "{:.4f}"),
        ("images", "Images processed", "{}"),
        ("total_gt", "Total GT phrases", "{}"),
        ("total_detected", "Total detected", "{}"),
    ],
}


def print_comparison(all_stats, title="BENCHMARK COMPARISON"):
    models = list(all_stats.keys())
    if not models:
        return

    task = None
    for s in all_stats.values():
        t = s.get("task")
        if t:
            task = t
            break

    rows = TASK_ROWS.get(task, [
        (k, k, "{!s}") for k in next(iter(all_stats.values())).keys()
    ])

    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")

    hdr = f"  {'Metric':<25}"
    for m in models:
        hdr += f"  {m:>20}"
    print(hdr)
    print(f"  {'─' * 25}", end="")
    for _ in models:
        print(f"  {'─' * 20}", end="")
    print()

    for key, label, fmt in rows:
        line = f"  {label:<25}"
        for m in models:
            v = all_stats[m].get(key, "N/A")
            if v is not None:
                try:
                    line += f"  {fmt.format(v):>20}"
                except (ValueError, KeyError):
                    line += f"  {str(v):>20}"
            else:
                line += f"  {'N/A':>20}"
        print(line)


def save_stats(stats, name):
    sp = RESULTS_DIR / f"{name}_stats.json"
    with open(sp, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to: {sp}")
    return sp
