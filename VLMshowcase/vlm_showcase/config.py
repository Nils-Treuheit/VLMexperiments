from pathlib import Path

PROJECT_DIR = Path("/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection")
SHOWCASE_DIR = Path("/mnt/HDD1/Project_Code/VLMexperiments/VLMshowcase")
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
    "locate_anything_trt": {
        "path": PROJECT_DIR / "locate_anything",
        "venv_python": str(PROJECT_DIR / "locate_anything" / ".venv" / "bin" / "python"),
        "script": str(PROJECT_DIR / "locate_anything" / "infer_trt.py"),
        "model_dir": str(PROJECT_DIR / "locate_anything" / "model"),
        "display": "LocateAnything-3B (TRT)",
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
    "florence2": {
        "path": PROJECT_DIR / "florence-2",
        "venv_python": str(PROJECT_DIR / "florence-2" / ".venv" / "bin" / "python"),
        "display": "Florence-2-large",
        "capabilities": ["caption", "detailed_caption", "detection", "ocr"],
    },
    "paligemma": {
        "path": PROJECT_DIR / "paligemma",
        "venv_python": str(PROJECT_DIR / "paligemma" / ".venv" / "bin" / "python"),
        "display": "PaliGemma2-3B",
        "capabilities": ["caption", "detection", "segmentation", "vqa"],
    },
    "cosmos_nemotron": {
        "path": PROJECT_DIR / "cosmos-nemotron",
        "venv_python": str(PROJECT_DIR / "cosmos-nemotron" / ".venv" / "bin" / "python"),
        "display": "Cosmos-Reason1-7B",
        "capabilities": ["physical_reasoning", "vqa", "scene_description", "video"],
    },
    "phi_vision": {
        "path": PROJECT_DIR / "phi-vision",
        "venv_python": str(PROJECT_DIR / "phi-vision" / ".venv" / "bin" / "python"),
        "display": "Phi-3.5-Vision-4B",
        "capabilities": ["description", "document_understanding", "chart_qa", "vqa"],
    },
    "llama_vision": {
        "path": PROJECT_DIR / "llama-vision",
        "venv_python": str(PROJECT_DIR / "llama-vision" / ".venv" / "bin" / "python"),
        "display": "Llama-3.2-11B-Vision",
        "capabilities": ["description", "reasoning", "vqa"],
    },
    "diffusion_gemma_vl": {
        "path": PROJECT_DIR / "diffusion_gemma_vl",
        "venv_python": str(PROJECT_DIR / "diffusion_gemma_vl" / ".venv" / "bin" / "python"),
        "script": str(PROJECT_DIR / "diffusion_gemma_vl" / "run.py"),
        "display": "DiffusionGemma-26B",
        "capabilities": ["caption", "vqa", "detect", "pose", "obb"],
    },
    "phi4_multimodal": {
        "path": PROJECT_DIR / "phi-4_multimodal",
        "venv_python": str(PROJECT_DIR / "phi-4_multimodal" / ".venv" / "bin" / "python"),
        "script": str(PROJECT_DIR / "phi-4_multimodal" / "run.py"),
        "display": "Phi-4-Multimodal",
        "capabilities": ["description", "document_understanding", "vqa"],
    },
    "dinov3": {
        "path": PROJECT_DIR / "dinov3",
        "venv_python": str(PROJECT_DIR / "dinov3" / ".venv" / "bin" / "python"),
        "script": str(PROJECT_DIR / "dinov3" / "run.py"),
        "display": "DINOv3 (Vision Encoder)",
        "capabilities": ["description", "encode"],
    },
    "siglip2": {
        "path": PROJECT_DIR / "siglip2",
        "venv_python": str(PROJECT_DIR / "siglip2" / ".venv" / "bin" / "python"),
        "script": str(PROJECT_DIR / "siglip2" / "run.py"),
        "display": "SigLIP2 (Vision Encoder)",
        "capabilities": ["description", "encode"],
    },
    "moonvit": {
        "path": PROJECT_DIR / "moonvit",
        "venv_python": str(PROJECT_DIR / "moonvit" / ".venv" / "bin" / "python"),
        "script": str(PROJECT_DIR / "moonvit" / "run.py"),
        "display": "MoonViT (Vision Encoder)",
        "capabilities": ["description", "encode"],
    },
    "dinotool": {
        "path": PROJECT_DIR / "DINOtool",
        "venv_python": str(PROJECT_DIR / "DINOtool" / ".venv" / "bin" / "python"),
        "script": str(PROJECT_DIR / "DINOtool" / "run.py"),
        "display": "DINOtool (Multi-ViT Encoder)",
        "capabilities": ["description", "encode", "features"],
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

VLM_MODEL_KEYS = {
    "locate_anything", "locate_anything_trt", "qwen3_instruct", "qwen3_thinking",
    "florence2", "paligemma", "cosmos_nemotron", "phi_vision", "llama_vision",
    "diffusion_gemma_vl", "phi4_multimodal",
}

VISION_ENCODER_KEYS = {"dinov3", "siglip2", "moonvit", "dinotool"}
