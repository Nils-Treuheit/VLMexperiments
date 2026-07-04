import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

from PIL import Image

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = Path("/mnt/HDD1/Project_Data/public_datasets")
COCO_DIR = DATA_DIR / "coco"
DOTA_DIR = DATA_DIR / "dotav1"
PROJECT_DIR = Path("/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection")

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


def scale_florence2(boxes, ow, oh):
    return [[x1 / 999 * ow, y1 / 999 * oh, x2 / 999 * ow, y2 / 999 * oh]
            for x1, y1, x2, y2 in boxes]


COCO_CAT_NAME_TO_ID = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
    "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
    "fire hydrant": 11, "stop sign": 13, "parking meter": 14, "bench": 15,
    "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20,
    "cow": 21, "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25,
    "backpack": 27, "umbrella": 28, "handbag": 31, "tie": 32, "suitcase": 33,
    "frisbee": 34, "skis": 35, "snowboard": 36, "sports ball": 37, "kite": 38,
    "baseball bat": 39, "baseball glove": 40, "skateboard": 41, "surfboard": 42,
    "tennis racket": 43, "bottle": 44, "wine glass": 46, "cup": 47, "fork": 48,
    "knife": 49, "spoon": 50, "bowl": 51, "banana": 52, "apple": 53,
    "sandwich": 54, "orange": 55, "broccoli": 56, "carrot": 57, "hot dog": 58,
    "pizza": 59, "donut": 60, "cake": 61, "chair": 62, "couch": 63,
    "potted plant": 64, "bed": 65, "dining table": 67, "toilet": 70, "tv": 72,
    "laptop": 73, "mouse": 74, "remote": 75, "keyboard": 76, "cell phone": 77,
    "microwave": 78, "oven": 79, "toaster": 80, "sink": 81, "refrigerator": 82,
    "book": 84, "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88,
    "hair drier": 89, "toothbrush": 90,
}


def tokenize(text):
    return text.lower().split()


def bleu_score(candidate, references, max_n=4):
    c_tok = tokenize(candidate)
    max_n = min(max_n, len(c_tok))
    scores = []
    for n in range(1, max_n + 1):
        c_ngrams = Counter(zip(*[c_tok[i:] for i in range(n)]))
        ref_counts = Counter()
        for ref in references:
            r_tok = tokenize(ref)
            if len(r_tok) < n:
                continue
            r_ngrams = Counter(zip(*[r_tok[i:] for i in range(n)]))
            for ng in r_ngrams:
                ref_counts[ng] = max(ref_counts.get(ng, 0), r_ngrams[ng])
        matches = sum(min(c_ngrams.get(ng, 0), ref_counts.get(ng, 0)) for ng in c_ngrams)
        total = sum(c_ngrams.values())
        precision = matches / total if total > 0 else 0
        scores.append(precision)
    if len(scores) < 4:
        scores += [0] * (4 - len(scores))
    bp = min(1, len(c_tok) / max((min(len(tokenize(r)) for r in references) if references else 1), 1))
    return bp * (scores[0] * scores[1] * scores[2] * scores[3]) ** 0.25 if all(s > 0 for s in scores[:4]) else 0.0


def lcs(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(candidate, references):
    c_tok = tokenize(candidate)
    best = 0
    for ref in references:
        r_tok = tokenize(ref)
        l = lcs(c_tok, r_tok)
        prec = l / len(c_tok) if c_tok else 0
        rec = l / len(r_tok) if r_tok else 0
        f = 2 * prec * rec / (prec + rec + 1e-12)
        best = max(best, f)
    return best


def cider(candidate, references):
    c_tok = tokenize(candidate)
    scores = []
    for n in [1, 2, 3, 4]:
        c_ngrams = Counter(zip(*[c_tok[i:] for i in range(n)]))
        if not c_ngrams:
            scores.append(0)
            continue
        ref_avg = Counter()
        for ref in references:
            r_tok = tokenize(ref)
            if len(r_tok) < n:
                continue
            r_ngrams = Counter(zip(*[r_tok[i:] for i in range(n)]))
            for ng in r_ngrams:
                ref_avg[ng] = ref_avg.get(ng, 0) + r_ngrams[ng]
        for ng in ref_avg:
            ref_avg[ng] /= len(references)
        score = sum(min(c_ngrams.get(ng, 0), ref_avg.get(ng, 0)) for ng in c_ngrams)
        total = sum(c_ngrams.values())
        scores.append(score / total if total > 0 else 0)
    return sum(scores) / len(scores) * 10 if scores else 0


def load_coco_captions(max_images=None):
    ap = COCO_DIR / "annotations" / "captions_val2017.json"
    if not ap.exists():
        print(f"Error: COCO captions not found at {ap}")
        return None, None
    with open(ap) as f:
        data = json.load(f)
    img_id_to_captions = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in img_id_to_captions:
            img_id_to_captions[iid] = []
        img_id_to_captions[iid].append(ann["caption"])
    img_infos = {im["id"]: im for im in data["images"]}
    img_ids = sorted(img_id_to_captions.keys())
    if max_images:
        img_ids = img_ids[:max_images]
    return img_ids, (img_infos, img_id_to_captions)


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


def load_florence2():
    sys.path.insert(0, str(PROJECT_DIR / "florence-2"))
    from transformers import AutoModelForCausalLM, AutoProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large-ft", torch_dtype=dtype, trust_remote_code=True
    ).to(dev)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
    return (model, processor), {}


def load_paligemma():
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma2-3b-mix-224", torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")
    return (model, processor), {}


def load_llama_vision():
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    model = AutoModelForMultimodalLM.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", device_map="auto", torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained("unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit")
    return (model, processor), {}


def load_phi_vision():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    config = AutoConfig.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-vision-instruct", config=config, trust_remote_code=True,
        dtype="auto", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
    return (model, processor), {}


def load_cosmos_nemotron():
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    model = AutoModelForMultimodalLM.from_pretrained(
        "nvidia/Cosmos-Reason1-7B", torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Reason1-7B")
    return (model, processor), {}


def load_diffusion_gemma():
    # DiffusionGemma inference is via subprocess to run.py (YOLO+llama-diffusion-cli).
    # Return a sentinel so benchmark scripts can detect this model type.
    return ("diffusion_gemma",), {}


def load_diffusion_gemma_yolo():
    return ("diffusion_gemma_yolo",), {}


def load_diffusion_gemma_yolo_pose():
    return ("diffusion_gemma_yolo_pose",), {}


def load_diffusion_gemma_yolo_obb():
    return ("diffusion_gemma_yolo_obb",), {}


def load_diffusion_gemma_siglip2():
    return ("diffusion_gemma_siglip2",), {}


def load_diffusion_gemma_moonvit():
    return ("diffusion_gemma_moonvit",), {}


def load_siglip2():
    # SigLIP2 inference is via subprocess to siglip2/run.py.
    return ("siglip2",), {}


def load_moonvit():
    # MoonViT inference is via subprocess to moonvit/run.py.
    return ("moonvit",), {}


def load_dinov3():
    # DINOv3 inference is via subprocess to dinov3/run.py.
    return ("dinov3",), {}


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
    # Edge VLM models (captioning / VQA)
    "florence2": load_florence2,
    "paligemma": load_paligemma,
    "llama_vision": load_llama_vision,
    "phi_vision": load_phi_vision,
    "cosmos_nemotron": load_cosmos_nemotron,
    # DiffusionGemma (YOLO feeder + text-only diffusion model)
    "diffusion_gemma": load_diffusion_gemma,
    "diffusion_gemma_yolo": load_diffusion_gemma_yolo,
    "diffusion_gemma_yolo_pose": load_diffusion_gemma_yolo_pose,
    "diffusion_gemma_yolo_obb": load_diffusion_gemma_yolo_obb,
    "diffusion_gemma_siglip2": load_diffusion_gemma_siglip2,
    "diffusion_gemma_moonvit": load_diffusion_gemma_moonvit,
    # Vision encoders (zero-shot structured description via subprocess)
    "dinov3": load_dinov3,
    "siglip2": load_siglip2,
    "moonvit": load_moonvit,
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
    "f2": "florence2",
    "florence": "florence2",
    "pg": "paligemma",
    "gemma": "paligemma",
    "llama": "llama_vision",
    "llama3": "llama_vision",
    "phi": "phi_vision",
    "phi3": "phi_vision",
    "cosmos": "cosmos_nemotron",
    "nemotron": "cosmos_nemotron",
    "dg": "diffusion_gemma",
    "diffusion_gemma_vl": "diffusion_gemma",
    "dg_yolo": "diffusion_gemma_yolo",
    "dg_pose": "diffusion_gemma_yolo_pose",
    "dg_obb": "diffusion_gemma_yolo_obb",
    "dg_siglip2": "diffusion_gemma_siglip2",
    "dg_moonvit": "diffusion_gemma_moonvit",
    "d3": "dinov3",
    "dino": "dinov3",
    "dinov3": "dinov3",
    "s2": "siglip2",
    "siglip": "siglip2",
    "mv": "moonvit",
    "moon": "moonvit",
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
    "florence2": "Florence-2-large-ft",
    "paligemma": "PaliGemma2-3B-mix",
    "llama_vision": "Llama-3.2-11B-Vision",
    "phi_vision": "Phi-3.5-Vision-4.2B",
    "cosmos_nemotron": "Cosmos-Reason1-7B",
    "diffusion_gemma": "DiffusionGemma-26B (YOLO)",
    "diffusion_gemma_yolo": "DiffusionGemma-26B (YOLO)",
    "diffusion_gemma_yolo_pose": "DiffusionGemma-26B (YOLO+pose)",
    "diffusion_gemma_yolo_obb": "DiffusionGemma-26B (YOLO+pose+obb)",
    "diffusion_gemma_siglip2": "DiffusionGemma-26B (SigLIP2)",
    "diffusion_gemma_moonvit": "DiffusionGemma-26B (MoonViT)",
    "dinov3": "DINOv3 (Zero-shot Description)",
    "siglip2": "SigLIP2 (Zero-shot Description)",
    "moonvit": "MoonViT (Zero-shot Description)",
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
    if model_type == "florence2":
        return f"<OD>{category_name}"
    if model_type == "paligemma":
        return f"detect {category_name}"
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
    "captioning": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("bleu_4", "BLEU-4", "{:.4f}"),
        ("rouge_l", "ROUGE-L", "{:.4f}"),
        ("cider", "CIDEr", "{:.4f}"),
        ("images", "Images processed", "{}"),
    ],
    "vqa": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("accuracy", "Accuracy", "{:.4f}"),
        ("images", "Questions answered", "{}"),
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
