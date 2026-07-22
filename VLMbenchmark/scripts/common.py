import ctypes
import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

from PIL import Image

import torch

# Cache/environment config
os.environ.setdefault("UNSLOTH_COMPILED_CACHE",
    "/mnt/HDD1/unsloth_and_hugging_face_models/unsloth_compiled_cache")
os.environ.setdefault("HF_HOME",
    "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface")
os.environ.setdefault("XFORMERS_DISABLED", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
SAMPLES_DIR = BASE_DIR / "samples"
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


def parse_loc_tokens(text):
    """Parse PaliGemma-style <locXXXX> tokens — groups of 4 as y1,x1,y2,x2, range 0-1023."""
    locs = [int(x) for x in re.findall(r'<loc(\d{4})>', text)]
    boxes = []
    for i in range(0, len(locs), 4):
        if i + 3 < len(locs):
            y1, x1, y2, x2 = locs[i], locs[i+1], locs[i+2], locs[i+3]
            boxes.append([x1, y1, x2, y2])
    return boxes


def auto_scale_boxes(boxes, ow, oh):
    """Detect coordinate range and scale to pixel coords [x1,y1,x2,y2]."""
    if not boxes:
        return boxes
    all_vals = [v for b in boxes for v in b]
    mx = max(all_vals)
    if mx <= 1.0:
        return [[x * ow, y * oh, x2 * ow, y2 * oh] for x, y, x2, y2 in boxes]
    if 900 <= mx <= 1100:
        max_dim = max(ow, oh)
        if mx <= max_dim:
            return boxes
        if mx <= 1024:
            return [[x1 / 1024 * ow, y1 / 1024 * oh, x2 / 1024 * ow, y2 / 1024 * oh]
                    for x1, y1, x2, y2 in boxes]
        return [[x1 / mx * ow, y1 / mx * oh, x2 / mx * ow, y2 / mx * oh]
                for x1, y1, x2, y2 in boxes]
    if mx > max(ow, oh):
        s = max(ow, oh) / mx
        return [[x1 * s, y1 * s, x2 * s, y2 * s] for x1, y1, x2, y2 in boxes]
    return boxes


# Keep legacy scale functions for backwards compat
def scale_la(boxes, ow, oh):
    return [[x1 / 1000 * ow, y1 / 1000 * oh, x2 / 1000 * ow, y2 / 1000 * oh]
            for x1, y1, x2, y2 in boxes]


def scale_qwen(boxes, ow, oh):
    out = []
    for x1, y1, x2, y2 in boxes:
        mx = max(x1, y1, x2, y2)
        if mx <= 1.0:
            x1 *= ow; y1 *= oh; x2 *= ow; y2 *= oh
        elif mx > max(ow, oh) and mx <= 1024:
            x1 = x1 / 999.0 * ow; y1 = y1 / 999.0 * oh
            x2 = x2 / 999.0 * ow; y2 = y2 / 999.0 * oh
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


def load_sample_list(filename):
    """Load a persistent sample list from SAMPLES_DIR. Returns None if not found."""
    fp = SAMPLES_DIR / filename
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return None


def save_sample_list(data, filename):
    """Save a persistent sample list to SAMPLES_DIR."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    fp = SAMPLES_DIR / filename
    with open(fp, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Samples saved to: {fp}")


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


def load_la():
    warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")
    warnings.filterwarnings("ignore", message=".*image_processor_class.*")
    sys.path.insert(0, str(PROJECT_DIR / "locate_anything"))
    import infer
    mp = str(PROJECT_DIR / "locate_anything" / "model")
    dev = "cuda" if torch.cuda.is_available() else None
    return infer.LocateAnythingWorker(mp, device=dev), infer


def _load_trt_libs():
    """Pre-load TensorRT shared libraries so onnxruntime can find them."""
    trt_venv = str(PROJECT_DIR / "locate_anything" / "model" / "tensorRT" / ".venv" / "lib" / "python3.10" / "site-packages")
    candidates = [
        trt_venv + "/tensorrt_libs",
        os.path.expanduser("~/.local/lib/python3.10/site-packages/nvidia/cudnn/lib"),
        "/usr/local/cuda-12.8/lib64",
    ]
    for lib_dir in candidates:
        if os.path.isdir(lib_dir):
            for f in sorted(os.listdir(lib_dir)):
                if f.endswith(".so") or ".so." in f:
                    try:
                        ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass


def load_la_trt():
    _load_trt_libs()
    warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")
    warnings.filterwarnings("ignore", message=".*image_processor_class.*")
    sys.path.insert(0, str(PROJECT_DIR / "locate_anything"))
    from infer_trt import LocateAnythingWorkerTRT
    mp = str(PROJECT_DIR / "locate_anything" / "model")
    return LocateAnythingWorkerTRT(mp), {}


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
    return QwenVLDetector(), {}


def load_phi4_multimodal():
    sys.path.insert(0, str(PROJECT_DIR / "phi-4_multimodal"))
    from model_loader import load_model as _load_phi4
    model, processor = _load_phi4()
    return (model, processor), {}


def load_yolo26(model_name="yolo26n"):
    sys.path.insert(0, str(PROJECT_DIR / "YOLO"))
    from ultralytics import YOLO
    # Prefer local pre-downloaded weights; fall back to ultralytics built-in
    models_dir = PROJECT_DIR / "YOLO" / "models"
    local = models_dir / f"{model_name}.pt"
    if local.exists():
        return YOLO(str(local)), {}
    return YOLO(model_name), {}


def load_yolo_world(model_name="yolov8x-worldv2"):
    sys.path.insert(0, str(PROJECT_DIR / "YOLO"))
    from ultralytics import YOLO
    models_dir = PROJECT_DIR / "YOLO" / "models"
    local = models_dir / f"{model_name}.pt"
    if local.exists():
        model = YOLO(str(local))
    else:
        model = YOLO(model_name)
    model.to("cuda")
    all_names = list(COCO_CAT_NAME_TO_ID.keys())
    model.set_classes(all_names)
    model._yolo_world_all_names = all_names
    return model, {}


def load_yoloe(model_name="yoloe-v8l-seg"):
    sys.path.insert(0, str(PROJECT_DIR / "YOLO"))
    from ultralytics import YOLO
    models_dir = PROJECT_DIR / "YOLO" / "models"
    local = models_dir / f"{model_name}.pt"
    if local.exists():
        model = YOLO(str(local))
    else:
        model = YOLO(model_name)
    model.to("cuda")
    all_names = list(COCO_CAT_NAME_TO_ID.keys())
    model.set_classes(all_names)
    model._yolo_world_all_names = all_names
    return model, {}


def load_florence2():
    _patch_florence_model_class()
    from transformers import AutoModelForCausalLM, AutoProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large-ft", torch_dtype=torch.float16,
        device_map=dev, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
    return (model, processor), {}


def _patch_florence_model_class():
    """Monkey-patch Florence-2 custom modeling to work with transformers >=4.57."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    def _get_cls(ref):
        return get_class_from_dynamic_module(ref, "microsoft/Florence-2-large-ft")

    try:
        cls_main = _get_cls("modeling_florence2.Florence2ForConditionalGeneration")
        cls_main._supports_sdpa = True

        # Fix outer generate(): the original passes **kwargs (including original
        # attention_mask) to language_model.generate(), but the correct merged
        # attention_mask must be substituted.
        _orig_gen = cls_main.generate

        def _patched_gen(self, input_ids=None, inputs_embeds=None, pixel_values=None,
                         **kwargs):
            if inputs_embeds is None:
                if input_ids is not None:
                    inputs_embeds = self.get_input_embeddings()(input_ids)
                if pixel_values is not None:
                    image_features = self._encode_image(pixel_values)
                    merged_embeds, merged_attn = \
                        self._merge_input_ids_with_image_features(
                            image_features, inputs_embeds
                        )
                    inputs_embeds = merged_embeds
                    kwargs["attention_mask"] = merged_attn
            kwargs.pop("pixel_values", None)
            kwargs.pop("input_ids", None)
            return self.language_model.generate(
                input_ids=None, inputs_embeds=inputs_embeds, **kwargs,
            )

        cls_main.generate = _patched_gen

        # Fix decoder forward: handle past_key_values that is (None,) instead of None
        cls_dec = _get_cls("modeling_florence2.Florence2Decoder")
        _orig_dec_fwd = cls_dec.forward

        def _patched_dec_fwd(self, *args, **kw):
            pkv = kw.get("past_key_values")
            if pkv is not None:
                try:
                    _ = pkv[0][0].shape[2]
                except (AttributeError, TypeError, IndexError):
                    kw["past_key_values"] = None
            return _orig_dec_fwd(self, *args, **kw)

        cls_dec.forward = _patched_dec_fwd

        # Fix language_model forward: ensure past_key_values is sanitized
        cls_lm = _get_cls(
            "modeling_florence2.Florence2LanguageForConditionalGeneration"
        )
        _orig_lm_prep = cls_lm.prepare_inputs_for_generation

        def _patched_lm_prep(self, decoder_input_ids, past_key_values=None, **kw):
            if past_key_values is not None:
                try:
                    return _orig_lm_prep(
                        self, decoder_input_ids, past_key_values=past_key_values,
                        **kw,
                    )
                except (AttributeError, TypeError, IndexError):
                    pass
            return {
                "input_ids": None,
                "encoder_outputs": kw.get("encoder_outputs"),
                "past_key_values": None,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": kw.get("attention_mask"),
                "decoder_attention_mask": kw.get("decoder_attention_mask"),
                "head_mask": kw.get("head_mask"),
                "decoder_head_mask": kw.get("decoder_head_mask"),
                "cross_attn_head_mask": kw.get("cross_attn_head_mask"),
                "use_cache": kw.get("use_cache"),
            }

        cls_lm.prepare_inputs_for_generation = _patched_lm_prep
    except Exception as e:
        print(f"  [warn] Florence-2 patch failed: {e}")


def load_paligemma():
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma2-3b-mix-224", torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")
    return (model, processor), {}


def load_llama_vision():
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    model = AutoModelForMultimodalLM.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained("unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit")
    return (model, processor), {}


def load_phi_vision():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    config = AutoConfig.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-vision-instruct", config=config, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
    return (model, processor), {}


def load_cosmos_nemotron():
    from transformers import AutoModelForVision2Seq, AutoProcessor
    model = AutoModelForVision2Seq.from_pretrained(
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


def load_dinotool():
    # DINOtool inference is via subprocess to DINOtool/run.py.
    return ("dinotool",), {}


def load_llava_v16_mistral():
    return ("llava_v16_mistral",), {}


def load_llava_onevision():
    return ("llava_onevision",), {}


def load_llava_next_video_7b():
    return ("llava_next_video_7b",), {}


def load_llava_next_video_34b():
    return ("llava_next_video_34b",), {}


def load_phi3_vision():
    return ("phi3_vision",), {}


MODEL_LOADERS = {
    "locate_anything": load_la,
    "locate_anything_trt": load_la_trt,
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
    # YOLO-World zero-shot detection
    "yolo_world": lambda: load_yolo_world("yolov8x-worldv2"),
    "yolo_worlds": lambda: load_yolo_world("yolov8s-worldv2"),
    "yolo_worldm": lambda: load_yolo_world("yolov8m-worldv2"),
    "yolo_worldl": lambda: load_yolo_world("yolov8l-worldv2"),
    # YOLOE zero-shot detection
    "yoloe": lambda: load_yoloe("yoloe-v8l-seg"),
    "yoloe_11l": lambda: load_yoloe("yoloe-11l-seg"),
    "yoloe_26m": lambda: load_yoloe("yoloe-26m-seg"),
    "yoloe_26n": lambda: load_yoloe("yoloe-26n-seg"),
    # LLaVA models
    "llava_v16_mistral": load_llava_v16_mistral,
    "llava_onevision": load_llava_onevision,
    "llava_next_video_7b": load_llava_next_video_7b,
    "llava_next_video_34b": load_llava_next_video_34b,
    "phi3_vision": load_phi3_vision,
    # Edge VLM models (captioning / VQA)
    "florence2": load_florence2,
    "paligemma": load_paligemma,
    "llama_vision": load_llama_vision,
    "phi_vision": load_phi_vision,
    "cosmos_nemotron": load_cosmos_nemotron,
    "phi4_multimodal": load_phi4_multimodal,
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
    "dinotool": load_dinotool,
}

MODEL_ALIASES = {
    "la": "locate_anything",
    "la_trt": "locate_anything_trt",
    "trt": "locate_anything_trt",
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
    "yolo_world": "yolo_world",
    "world": "yolo_world",
    "yoloe": "yoloe",
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
    "dinotool": "dinotool",
    "dt": "dinotool",
    "llava": "llava_v16_mistral",
    "llava_mistral": "llava_v16_mistral",
    "llava_onevision": "llava_onevision",
    "llava_next7b": "llava_next_video_7b",
    "llava_next34b": "llava_next_video_34b",
    "phi3v": "phi3_vision",
    "s2": "siglip2",
    "siglip": "siglip2",
    "mv": "moonvit",
    "moon": "moonvit",
}

MODEL_DISPLAY = {
    "locate_anything": "LocateAnything-3B",
    "locate_anything_trt": "LocateAnything-3B (TRT)",
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
    "yolo_world": "YOLO-Worldv2-x",
    "yolo_worlds": "YOLO-Worldv2-s",
    "yolo_worldm": "YOLO-Worldv2-m",
    "yolo_worldl": "YOLO-Worldv2-l",
    "yoloe": "YOLOE-v8l",
    "yoloe_11l": "YOLOE-11l",
    "yoloe_26m": "YOLOE-26m",
    "yoloe_26n": "YOLOE-26n",
    "florence2": "Florence-2-large-ft",
    "paligemma": "PaliGemma2-3B-mix",
    "llama_vision": "Llama-3.2-11B-Vision",
    "phi_vision": "Phi-3.5-Vision-4.2B",
    "phi4_multimodal": "Phi-4-Multimodal-14B",
    "cosmos_nemotron": "Cosmos-Reason1-7B",
    # LLaVA models
    "llava_v16_mistral": "LLaVA-v1.6-Mistral-7B",
    "llava_onevision": "LLaVA-OneVision-Qwen2-7B",
    "llava_next_video_7b": "LLaVA-NeXT-Video-7B",
    "llava_next_video_34b": "LLaVA-NeXT-Video-34B",
    "phi3_vision": "LLaVA-Phi-3-Mini-4B",
    "diffusion_gemma": "DiffusionGemma-26B (YOLO)",
    "diffusion_gemma_yolo": "DiffusionGemma-26B (YOLO)",
    "diffusion_gemma_yolo_pose": "DiffusionGemma-26B (YOLO+pose)",
    "diffusion_gemma_yolo_obb": "DiffusionGemma-26B (YOLO+pose+obb)",
    "diffusion_gemma_siglip2": "DiffusionGemma-26B (SigLIP2)",
    "diffusion_gemma_moonvit": "DiffusionGemma-26B (MoonViT)",
    "dinov3": "DINOv3 (Zero-shot Description)",
    "siglip2": "SigLIP2 (Zero-shot Description)",
    "moonvit": "MoonViT (Zero-shot Description)",
    "dinotool": "DINOtool (Zero-shot Description)",
}


def build_prompt(category_name, model_type):
    if model_type in ("locate_anything", "locate_anything_trt"):
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
    "classification": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("top1_accuracy", "Top-1 Accuracy", "{:.4f}"),
        ("top5_accuracy", "Top-5 Accuracy", "{:.4f}"),
        ("images", "Images processed", "{}"),
    ],
    "segmentation": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("PQ", "Panoptic Quality", "{:.4f}"),
        ("mIoU", "mIoU", "{:.4f}"),
        ("images", "Images processed", "{}"),
    ],
    "scene_analysis": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("scene_type_accuracy", "Scene type accuracy", "{:.4f}"),
        ("object_recall", "Object recall", "{:.4f}"),
        ("images", "Images processed", "{}"),
    ],
    "tracking": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("MOTA", "MOTA", "{:.4f}"),
        ("MOTP", "MOTP", "{:.4f}"),
        ("frames", "Frames processed", "{}"),
    ],
    "6d_pose": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("detection_rate", "Detection rate", "{:.4f}"),
        ("images", "Images processed", "{}"),
    ],
    "embedding": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("embedding_dim", "Embedding dim", "{}"),
        ("images", "Images processed", "{}"),
    ],
    "zeroshot_detection": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("acc@50_b3", "B3 Acc@50", "{:.4f}"),
        ("acc@50_avg", "AVG Acc@50", "{:.4f}"),
        ("images", "Images processed", "{}"),
        ("total_gt", "Total GT objects", "{}"),
        ("total_correct", "Correct@IoU50 (B3)", "{}"),
    ],
    "veq": [
        ("fps", "FPS", "{:.2f}"),
        ("avg_inference_ms", "Avg inference (ms)", "{:.1f}"),
        ("embedding_dim", "Embedding dim", "{}"),
        ("retrieval_mAP", "Retrieval mAP", "{:.4f}"),
        ("retrieval_Recall@1", "Recall@1", "{:.4f}"),
        ("retrieval_Recall@5", "Recall@5", "{:.4f}"),
        ("retrieval_Recall@10", "Recall@10", "{:.4f}"),
        ("clustering_NMI", "Clustering NMI", "{:.4f}"),
        ("clustering_ARI", "Clustering ARI", "{:.4f}"),
        ("images", "Images", "{}"),
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
