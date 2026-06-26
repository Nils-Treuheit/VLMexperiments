from pathlib import Path

PROJECT_DIR = Path("/mnt/HDD1/Project_Code/vlm_det_test")
SHOWCASE_DIR = Path("/mnt/HDD1/Project_Code/VLMshowcase")
DATA_DIR = Path("/mnt/HDD1/Project_Data")
PUBLIC_DATASETS = DATA_DIR / "public_datasets"
DEMO_MATERIAL = DATA_DIR / "demoMaterial"
COCO_DIR = PUBLIC_DATASETS / "coco"
DOTA_DIR = PUBLIC_DATASETS / "dotav1"
TMP_DIR = Path("/mnt/HDD1/tmp")

MODELS = {
    "locate_anything": {
        "path": PROJECT_DIR / "locate_anything",
        "venv_python": str(PROJECT_DIR / "locate_anything" / ".venv" / "bin" / "python"),
        "script": str(PROJECT_DIR / "locate_anything" / "infer.py"),
        "model_dir": str(PROJECT_DIR / "locate_anything" / "model"),
        "display": "LocateAnything-3B",
        "capabilities": ["grounding", "detection", "text_detection"],
    },
    "qwen3_instruct": {
        "path": PROJECT_DIR / "qwen3-vl_instruct",
        "venv_python": str(PROJECT_DIR / "qwen3-vl_instruct" / ".venv" / "bin" / "python"),
        "script": str(PROJECT_DIR / "qwen3-vl_instruct" / "infer_qwen3.py"),
        "model_dir": str(PROJECT_DIR / "qwen3-vl_instruct" / "model_vl"),
        "display": "Qwen3-VL-8B-Instruct",
        "capabilities": ["description", "vqa", "detection", "ocr", "video"],
    },
    "qwen3_thinking": {
        "path": PROJECT_DIR / "qwen3-vl_thinking",
        "venv_python": str(PROJECT_DIR / "qwen3-vl_thinking" / ".venv" / "bin" / "python"),
        "script_wrapper": str(SHOWCASE_DIR / "scripts" / "_thinking_wrapper.py"),
        "display": "Qwen3-VL-8B-Thinking",
        "capabilities": ["reasoning", "description", "detection", "intent"],
    },
    "yolo": {
        "path": PROJECT_DIR / "yolo11-26",
        "model_dir": str(PROJECT_DIR / "yolo11-26" / "models"),
        "display": "YOLO11/26",
        "capabilities": ["detection", "pose", "obb", "tracking"],
    },
}

YOLO_MODEL_PATHS = {
    "yolo11n": PROJECT_DIR / "yolo11-26" / "models" / "yolo11n.pt",
    "yolo11s": PROJECT_DIR / "yolo11-26" / "models" / "yolo11s.pt",
    "yolo11m": PROJECT_DIR / "yolo11-26" / "models" / "yolo11m.pt",
    "yolo11l": PROJECT_DIR / "yolo11-26" / "models" / "yolo11l.pt",
    "yolo11x": PROJECT_DIR / "yolo11-26" / "models" / "yolo11x.pt",
    "yolo26n": PROJECT_DIR / "yolo11-26" / "models" / "yolo26n.pt",
    "yolo26s": PROJECT_DIR / "yolo11-26" / "models" / "yolo26s.pt",
    "yolo26m": PROJECT_DIR / "yolo11-26" / "models" / "yolo26m.pt",
    "yolo26l": PROJECT_DIR / "yolo11-26" / "models" / "yolo26l.pt",
    "yolo26x": PROJECT_DIR / "yolo11-26" / "models" / "yolo26x.pt",
    "yolo11n-pose": PROJECT_DIR / "yolo11-26" / "models" / "yolo11n-pose.pt",
    "yolo11s-pose": PROJECT_DIR / "yolo11-26" / "models" / "yolo11s-pose.pt",
    "yolo26n-pose": PROJECT_DIR / "yolo11-26" / "models" / "yolo26n-pose.pt",
    "yolo26s-pose": PROJECT_DIR / "yolo11-26" / "models" / "yolo26s-pose.pt",
    "yolo11n-obb": PROJECT_DIR / "yolo11-26" / "models" / "yolo11n-obb.pt",
    "yolo11s-obb": PROJECT_DIR / "yolo11-26" / "models" / "yolo11s-obb.pt",
    "yolo26n-obb": PROJECT_DIR / "yolo11-26" / "models" / "yolo26n-obb.pt",
    "yolo26s-obb": PROJECT_DIR / "yolo11-26" / "models" / "yolo26s-obb.pt",
}

COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
