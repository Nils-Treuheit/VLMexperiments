import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILED_CACHE",
    "/mnt/HDD1/unsloth_and_hugging_face_models/unsloth_compiled_cache")
os.environ.setdefault("HF_HOME",
    "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface")

SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPTS_DIR.parent / "results"
SAMPLES_DIR = SCRIPTS_DIR.parent / "samples"

# Model categories by approximate VRAM usage (smallest first):
#   tiny:    vision encoders, YOLO (< 2 GB)
#   small:   florence2, locate_anything (2-4 GB)
#   medium:  paligemma, phi_vision (4-8 GB)
#   large:   cosmos, llama, qwen3 (8-14 GB)
#   xlarge:  diffusion_gemma, LLaVA, phi4_mm (14-20+ GB)
# Within each category, keep fast models before slow ones.

TASKS = {
    "od": {
        "script": "benchmark_od.py",
        "title": "Object Detection (COCO)",
        "models": [
            "yolo11", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
            "yolo26", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
            "florence2", "paligemma",
            "locate_anything", "locate_anything_trt",
            "qwen3_native", "qwen3_thinking",
        ],
        "dataset": "coco",
    },
    "pose": {
        "script": "benchmark_pose.py",
        "title": "Pose Estimation (COCO Keypoints)",
        "models": ["yolo11_pose", "yolo11s_pose", "yolo26_pose", "yolo26s_pose"],
        "dataset": None,
    },
    "obb": {
        "script": "benchmark_obb.py",
        "title": "OBB Detection (DOTA-v1.0)",
        "models": ["yolo11_obb", "yolo11s_obb", "yolo26_obb", "yolo26s_obb"],
        "dataset": None,
    },
    "grounding": {
        "script": "benchmark_grounding.py",
        "title": "Phrase Grounding (COCO)",
        "models": ["florence2", "locate_anything", "locate_anything_trt",
                    "qwen3_native", "qwen3_thinking"],
        "dataset": None,
    },
    "captioning": {
        "script": "benchmark_caption.py",
        "title": "Image Captioning (COCO Captions)",
        "models": [
            "siglip2", "moonvit", "dinov3", "dinotool",
            "florence2", "paligemma",
            "cosmos_nemotron",
            "llama_vision",
            "qwen3_native", "qwen3_thinking",
            "phi_vision",
            "llava_v16_mistral", "llava_onevision",
            "llava_next_video_7b", "phi3_vision",
            "llava_next_video_34b",
            "diffusion_gemma", "diffusion_gemma_yolo",
            "diffusion_gemma_yolo_pose", "diffusion_gemma_yolo_obb",
            "diffusion_gemma_siglip2", "diffusion_gemma_moonvit",
        ],
        "dataset": None,
    },
    "vqa": {
        "script": "benchmark_vqa.py",
        "title": "Visual Question Answering (COCO)",
        "models": [
            "florence2", "paligemma",
            "cosmos_nemotron",
            "llama_vision",
            "qwen3_native", "qwen3_thinking",
            "phi_vision",
            "llava_v16_mistral", "llava_onevision",
            "llava_next_video_7b", "phi3_vision",
            "llava_next_video_34b",
            "diffusion_gemma", "diffusion_gemma_yolo",
            "diffusion_gemma_yolo_pose", "diffusion_gemma_yolo_obb",
            "diffusion_gemma_siglip2", "diffusion_gemma_moonvit",
        ],
        "dataset": None,
    },
    "classification": {
        "script": "benchmark_classification.py",
        "title": "Zero-Shot Classification (Tiny ImageNet)",
        "models": ["dinotool", "dinov3", "siglip2", "moonvit",
                    "florence2", "paligemma",
                    "locate_anything", "locate_anything_trt",
                    "qwen3_native", "qwen3_thinking"],
        "dataset": None,
    },
    "segmentation": {
        "script": "benchmark_segmentation.py",
        "title": "Segmentation (COCO)",
        "models": ["florence2", "locate_anything", "locate_anything_trt"],
        "dataset": None,
    },
    "scene_analysis": {
        "script": "benchmark_scene.py",
        "title": "Semantic Scene Analysis (COCO)",
        "models": [
            "florence2", "paligemma",
            "cosmos_nemotron",
            "llama_vision",
            "qwen3_native", "qwen3_thinking",
            "phi_vision",
            "llava_v16_mistral", "llava_onevision",
            "llava_next_video_7b", "phi3_vision",
            "llava_next_video_34b",
        ],
        "dataset": None,
    },
    "tracking": {
        "script": "benchmark_tracking.py",
        "title": "Multi-Object Tracking (MOT17)",
        "models": ["yolo11", "yolo11s", "yolo11m", "yolo26", "yolo26s", "yolo26m"],
        "dataset": None,
    },
    "6d_pose": {
        "script": "benchmark_6dpose.py",
        "title": "6D Pose Estimation Detection (Linemod)",
        "models": ["yolo11", "yolo11s", "yolo11m", "yolo26", "yolo26s", "yolo26m"],
        "dataset": None,
    },
    "ocr": {
        "script": "benchmark_ocr.py",
        "title": "OCR / Text Detection (Synthetic COCO)",
        "models": ["florence2", "locate_anything", "locate_anything_trt"],
        "dataset": None,
    },
    "pointing": {
        "script": "benchmark_pointing.py",
        "title": "Pointing / 2D Keypoint Localization (COCO)",
        "models": ["locate_anything", "locate_anything_trt"],
        "dataset": None,
    },
    "counting": {
        "script": "benchmark_counting.py",
        "title": "Object Counting (COCO instances)",
        "models": [
            "florence2", "paligemma",
            "cosmos_nemotron",
            "llama_vision",
            "qwen3_native", "qwen3_thinking",
            "phi_vision",
            "llava_v16_mistral", "llava_onevision",
            "llava_next_video_7b", "phi3_vision",
            "llava_next_video_34b",
            "diffusion_gemma", "diffusion_gemma_yolo",
            "diffusion_gemma_yolo_pose", "diffusion_gemma_yolo_obb",
            "diffusion_gemma_siglip2", "diffusion_gemma_moonvit",
        ],
        "dataset": None,
    },
    "visual_reasoning": {
        "script": "benchmark_visual_reasoning.py",
        "title": "Visual Reasoning (COCO)",
        "models": [
            "florence2", "paligemma",
            "cosmos_nemotron",
            "llama_vision",
            "qwen3_native", "qwen3_thinking",
            "phi_vision",
            "llava_v16_mistral", "llava_onevision",
            "llava_next_video_7b", "phi3_vision",
            "llava_next_video_34b",
        ],
        "dataset": None,
    },
    "docvqa": {
        "script": "benchmark_docvqa.py",
        "title": "Document VQA (COCO)",
        "models": [
            "florence2", "paligemma",
            "cosmos_nemotron",
            "llama_vision",
            "qwen3_native", "qwen3_thinking",
            "phi_vision",
            "llava_v16_mistral", "llava_onevision",
            "llava_next_video_7b", "phi3_vision",
            "llava_next_video_34b",
        ],
        "dataset": None,
    },
    "emotion": {
        "script": "benchmark_emotion.py",
        "title": "Emotion Detection (COCO)",
        "models": [
            "florence2", "paligemma",
            "cosmos_nemotron",
            "llama_vision",
            "qwen3_native", "qwen3_thinking",
            "phi_vision",
            "llava_v16_mistral", "llava_onevision",
            "llava_next_video_7b", "phi3_vision",
            "llava_next_video_34b",
        ],
        "dataset": None,
    },
    "hir": {
        "script": "benchmark_hir.py",
        "title": "Human Intention Recognition (COCO)",
        "models": [
            "florence2", "paligemma",
            "cosmos_nemotron",
            "llama_vision",
            "qwen3_native", "qwen3_thinking",
            "phi_vision",
            "llava_v16_mistral", "llava_onevision",
            "llava_next_video_7b", "phi3_vision",
            "llava_next_video_34b",
        ],
        "dataset": None,
    },
    "doc_understanding": {
        "script": "benchmark_doc_understanding.py",
        "title": "Document Understanding (COCO)",
        "models": [
            "florence2", "paligemma",
            "cosmos_nemotron",
            "llama_vision",
            "qwen3_native", "qwen3_thinking",
            "phi_vision",
            "llava_v16_mistral", "llava_onevision",
            "llava_next_video_7b", "phi3_vision",
            "llava_next_video_34b",
        ],
        "dataset": None,
    },
    "embedding": {
        "script": "benchmark_embedding.py",
        "title": "Embedding Extraction (COCO)",
        "models": ["siglip2", "dinov3", "moonvit", "dinotool"],
        "dataset": None,
    },
    "zeroshot_detection": {
        "script": "benchmark_zeroshot_detection.py",
        "title": "Zero-Shot Detection (COCO)",
        "models": [
            "florence2", "paligemma",
            "cosmos_nemotron",
            "llama_vision",
            "qwen3_native", "qwen3_thinking",
            "phi_vision",
            "llava_v16_mistral", "llava_onevision",
            "llava_next_video_7b", "phi3_vision",
            "llava_next_video_34b",
            "diffusion_gemma", "diffusion_gemma_yolo",
            "diffusion_gemma_yolo_pose", "diffusion_gemma_yolo_obb",
            "diffusion_gemma_siglip2", "diffusion_gemma_moonvit",
        ],
        "dataset": None,
    },
}

COLLECTION_DIR = Path("/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection")

MODEL_VENV = {k: str(COLLECTION_DIR / v / ".venv" / "bin" / "python")
    for k, v in {
        "locate_anything": "locate_anything",
        "locate_anything_trt": "locate_anything",
        "qwen3_native": "qwen3-vl_instruct",
        "qwen3_thinking": "qwen3-vl_thinking",
        "yolo26": "yolo11-26",
        "yolo26s": "yolo11-26",
        "yolo26m": "yolo11-26",
        "yolo26l": "yolo11-26",
        "yolo26x": "yolo11-26",
        "yolo26_pose": "yolo11-26",
        "yolo26s_pose": "yolo11-26",
        "yolo26_obb": "yolo11-26",
        "yolo26s_obb": "yolo11-26",
        "yolo11": "yolo11-26",
        "yolo11s": "yolo11-26",
        "yolo11m": "yolo11-26",
        "yolo11l": "yolo11-26",
        "yolo11x": "yolo11-26",
        "yolo11_pose": "yolo11-26",
        "yolo11s_pose": "yolo11-26",
        "yolo11_obb": "yolo11-26",
        "yolo11s_obb": "yolo11-26",
        "florence2": "florence-2",
        "paligemma": "paligemma",
        "llama_vision": "llama-vision",
        "phi_vision": "phi-vision",
        "cosmos_nemotron": "cosmos-nemotron",
        "diffusion_gemma": "diffusion_gemma_vl",
        "diffusion_gemma_yolo": "diffusion_gemma_vl",
        "diffusion_gemma_yolo_pose": "diffusion_gemma_vl",
        "diffusion_gemma_yolo_obb": "diffusion_gemma_vl",
        "diffusion_gemma_siglip2": "diffusion_gemma_vl",
        "diffusion_gemma_moonvit": "diffusion_gemma_vl",
        "dinov3": "dinov3",
        "siglip2": "siglip2",
        "moonvit": "moonvit",
        "dinotool": "DINOtool",
        "llava_v16_mistral": "Llava",
        "llava_onevision": "Llava",
        "llava_next_video_7b": "Llava",
        "llava_next_video_34b": "Llava",
        "phi3_vision": "Llava",
    }.items()}


def generate_samples(task_key, task, max_images):
    """Pre-generate persistent sample files for reproducibility."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    script = task["script"]
    task_name = script.replace("benchmark_", "").replace(".py", "")

    if task_name in ("caption", "vqa", "scene"):
        sample_file = SAMPLES_DIR / f"coco_{task_name}_samples.json"
    elif task_name == "classification":
        sample_file = SAMPLES_DIR / "tiny_imagenet_val.json"
    elif task_name == "obb":
        sample_file = SAMPLES_DIR / "dota_obb_samples.json"
    elif task_name == "tracking":
        sample_file = SAMPLES_DIR / "mot17_samples.json"
    elif task_name == "6dpose":
        sample_file = SAMPLES_DIR / "linemod_samples.json"
    elif task_name == "ocr":
        sample_file = SAMPLES_DIR / "coco_ocr_samples.json"
    elif task_name == "pointing":
        sample_file = SAMPLES_DIR / "coco_pointing_samples.json"
    elif task_name in ("od", "pose", "grounding", "segmentation"):
        sample_file = SAMPLES_DIR / f"coco_{task_name}_samples.json"
    else:
        sample_file = SAMPLES_DIR / f"{task_name}_samples.json"

    if sample_file.exists():
        print(f"  Samples already exist: {sample_file.name}")
        return str(sample_file)

    print(f"  Generating samples: {sample_file.name}")
    return str(sample_file)


def wait_for_vram(min_free_mib=4096, check_interval=60, max_wait=7200):
    """Wait until at least min_free_mib MiB of VRAM is free. Returns True once available."""
    waited = 0
    while waited < max_wait:
        try:
            import subprocess as _sp
            out = _sp.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            free = int(''.join(c for c in out if c.isdigit()))
            if free >= min_free_mib:
                return True
            print(f"  [WAIT] Free VRAM: {free} MiB < {min_free_mib} MiB needed — waiting {check_interval}s...")
        except Exception:
            pass
        time.sleep(check_interval)
        waited += check_interval
    print(f"  [SKIP] Timed out waiting for VRAM after {max_wait}s")
    return False


def run_model(model, script, max_images, dataset=None, sample_file=None, max_retries=3):
    venv_python = MODEL_VENV.get(model)
    if not venv_python:
        print(f"  [SKIP] No venv configured for {model}")
        return None

    # Determine approximate VRAM needs (MiB free needed before loading)
    vram_map = {
        # tiny: vision encoders, YOLO
        "siglip2": 1024, "moonvit": 1024, "dinov3": 1024, "dinotool": 1024,
        "yolo11": 1024, "yolo11s": 1024, "yolo11m": 1024, "yolo11l": 1024, "yolo11x": 1024,
        "yolo26": 1024, "yolo26s": 1024, "yolo26m": 1024, "yolo26l": 1024, "yolo26x": 1024,
        "yolo11_pose": 1024, "yolo11s_pose": 1024, "yolo26_pose": 1024, "yolo26s_pose": 1024,
        "yolo11_obb": 1024, "yolo11s_obb": 1024, "yolo26_obb": 1024, "yolo26s_obb": 1024,
        # small: florence2, locate_anything
        "florence2": 4096, "locate_anything": 4096, "locate_anything_trt": 4096,
        # medium: paligemma
        "paligemma": 6144,
        # large: cosmos, llama, qwen3, phi_vision
        "cosmos_nemotron": 12288, "llama_vision": 12288,
        "qwen3_native": 12288, "qwen3_thinking": 12288,
        "phi_vision": 12288,
        # xlarge: diffusion_gemma, LLaVA
        "diffusion_gemma": 16384, "diffusion_gemma_yolo": 16384,
        "diffusion_gemma_yolo_pose": 16384, "diffusion_gemma_yolo_obb": 16384,
        "diffusion_gemma_siglip2": 16384, "diffusion_gemma_moonvit": 16384,
        "llava_v16_mistral": 16384, "llava_onevision": 16384,
        "llava_next_video_7b": 16384, "llava_next_video_34b": 20480,
        "phi3_vision": 16384,
    }
    needed = vram_map.get(model, 4096)

    for attempt in range(1, max_retries + 1):
        if not wait_for_vram(min_free_mib=needed):
            print(f"  [SKIP] {model}: insufficient VRAM after waiting")
            return None

        script_path = SCRIPTS_DIR / script
        q_scripts = ("vqa", "emotion", "hir", "visual_reasoning", "docvqa", "doc_understanding")
        if any(s in script for s in q_scripts):
            cmd = [venv_python, str(script_path), "--model", model, "--max-questions", str(max_images * 2)]
        else:
            cmd = [venv_python, str(script_path), "--model", model, "--max-images", str(max_images)]
        if dataset:
            cmd += ["--dataset", dataset]
        if sample_file and "classification" not in script:
            cmd += ["--samples-file", sample_file]

        print(f"\n  ── Running: {model} via {venv_python.split('/')[-3]} (attempt {attempt}) ──")
        sys.stdout.flush()

        # Model timeout: 90 min for slow models (diffusion_gemma), 30 min for normal ones
        mdl_timeout = 5400 if model.startswith("diffusion_gemma") else 1800
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=mdl_timeout)
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {model} exceeded {mdl_timeout}s")
            continue

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Show last 2000 chars of stderr
            err = result.stderr[-3000:]
            if "CUDA out of memory" in err or "out of memory" in err.lower():
                print(f"  [OOM] {model} — waiting for VRAM to clear, retry {attempt}/{max_retries}")
                time.sleep(120)
                continue
            print(err[-2000:] if len(err) > 2000 else err)

        if result.returncode != 0:
            print(f"  [ERROR] {model} failed (code {result.returncode}), retry {attempt}/{max_retries}")
            time.sleep(30)
            continue

        return extract_stats(model, script)

    print(f"  [FAIL] {model} exhausted all {max_retries} retries")
    return None


def extract_stats(model, script):
    prefix = model.replace("/", "_").replace(" ", "_")
    if "caption" in script:
        fp = RESULTS_DIR / f"{prefix}_caption_stats.json"
    elif "vqa" in script:
        fp = RESULTS_DIR / f"{prefix}_vqa_stats.json"
    elif "obb" in script:
        fp = RESULTS_DIR / f"{prefix}_obb_stats.json"
    elif "pose" in script:
        fp = RESULTS_DIR / f"{prefix}_pose_stats.json"
    elif "grounding" in script:
        fp = RESULTS_DIR / f"{prefix}_grounding_stats.json"
    elif "classification" in script:
        fp = RESULTS_DIR / f"{prefix}_classification_stats.json"
    elif "segmentation" in script:
        fp = RESULTS_DIR / f"{prefix}_segmentation_stats.json"
    elif "scene" in script:
        fp = RESULTS_DIR / f"{prefix}_scene_stats.json"
    elif "tracking" in script:
        fp = RESULTS_DIR / f"{prefix}_botsort_tracking_stats.json"
        if not fp.exists():
            fp = RESULTS_DIR / f"{prefix}_bytetrack_tracking_stats.json"
        if not fp.exists():
            fp = RESULTS_DIR / f"{prefix}_tracking_stats.json"
    elif "6dpose" in script:
        fp = RESULTS_DIR / f"{prefix}_linemod_6dpose_stats.json"
    elif "ocr" in script:
        fp = RESULTS_DIR / f"{prefix}_ocr_stats.json"
    elif "pointing" in script:
        fp = RESULTS_DIR / f"{prefix}_pointing_stats.json"
    elif "counting" in script:
        fp = RESULTS_DIR / f"{prefix}_counting_stats.json"
    elif "visual_reasoning" in script:
        fp = RESULTS_DIR / f"{prefix}_visual_reasoning_stats.json"
    elif "docvqa" in script:
        fp = RESULTS_DIR / f"{prefix}_docvqa_stats.json"
    elif "emotion" in script:
        fp = RESULTS_DIR / f"{prefix}_emotion_stats.json"
    elif "hir" in script:
        fp = RESULTS_DIR / f"{prefix}_hir_stats.json"
    elif "doc_understanding" in script:
        fp = RESULTS_DIR / f"{prefix}_doc_understanding_stats.json"
    elif "embedding" in script:
        fp = RESULTS_DIR / f"{prefix}_embedding_stats.json"
    elif "zeroshot_detection" in script:
        fp = RESULTS_DIR / f"{prefix}_zeroshot_detection_stats.json"
    else:
        fp = RESULTS_DIR / f"{prefix}_coco_od_stats.json"
        if fp.exists():
            with open(fp) as f:
                return json.load(f)
        return None

    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--max-images", type=int, default=50,
                        help="Images per model per task (default: 50)")
    parser.add_argument("--tasks", nargs="+",
                        choices=list(TASKS) + ["all"],
                        default=["all"],
                        help="Tasks to run (default: all)")
    parser.add_argument("--no-samples", action="store_true",
                        help="Skip persistent sample generation")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks_to_run = list(TASKS) if "all" in args.tasks else args.tasks

    all_comparisons = {}

    for task_key in tasks_to_run:
        task = TASKS[task_key]
        print(f"\n{'#' * 70}")
        print(f"# TASK: {task['title']}")
        print(f"{'#' * 70}")

        sample_file = generate_samples(task_key, task, args.max_images) if not args.no_samples else None

        task_stats = {}
        for model in task["models"]:
            stats = run_model(model, task["script"], args.max_images, task.get("dataset"), sample_file)
            if stats:
                task_stats[model] = stats

        if task_stats:
            all_comparisons[task_key] = task_stats

    print(f"\n{'=' * 70}")
    print("ALL BENCHMARKS COMPLETE")
    print(f"{'=' * 70}")
    for task_key, stats in all_comparisons.items():
        print(f"\n  {TASKS[task_key]['title']}: {len(stats)} models run")

    summary_path = RESULTS_DIR / "all_benchmarks_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"\n  Full summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
