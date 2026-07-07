import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPTS_DIR.parent / "results"

TASKS = {
    "od": {
        "script": "benchmark_od.py",
        "title": "Object Detection (COCO)",
        "models": ["locate_anything", "locate_anything_trt", "qwen3_native", "qwen3_thinking", "yolo26",
                    "florence2", "paligemma"],
        "dataset": "coco",
    },
    "od_dota": {
        "script": "benchmark_od.py",
        "title": "Object Detection (DOTA)",
        "models": ["locate_anything", "locate_anything_trt", "qwen3_native", "qwen3_thinking"],
        "dataset": "dota",
    },
    "pose": {
        "script": "benchmark_pose.py",
        "title": "Pose Estimation (COCO Keypoints)",
        "models": ["yolo26_pose"],
        "dataset": None,
    },
    "obb": {
        "script": "benchmark_obb.py",
        "title": "OBB Detection (DOTA-v1.0)",
        "models": ["yolo26_obb"],
        "dataset": None,
    },
    "grounding": {
        "script": "benchmark_grounding.py",
        "title": "Phrase Grounding (COCO)",
        "models": ["locate_anything", "locate_anything_trt", "qwen3_native", "qwen3_thinking", "florence2"],
        "dataset": None,
    },
    "captioning": {
        "script": "benchmark_caption.py",
        "title": "Image Captioning (COCO Captions)",
        "models": ["florence2", "paligemma", "llama_vision", "phi_vision", "cosmos_nemotron",
                    "qwen3_native", "qwen3_thinking", "diffusion_gemma",
                    "diffusion_gemma_yolo", "diffusion_gemma_yolo_pose",
                    "diffusion_gemma_yolo_obb", "diffusion_gemma_siglip2",
                    "diffusion_gemma_moonvit",
                    "siglip2", "moonvit", "dinov3", "dinotool"],
        "dataset": None,
    },
    "vqa": {
        "script": "benchmark_vqa.py",
        "title": "Visual Question Answering (COCO)",
        "models": ["florence2", "paligemma", "llama_vision", "phi_vision", "cosmos_nemotron",
                    "qwen3_native", "qwen3_thinking",
                    "diffusion_gemma", "diffusion_gemma_yolo", "diffusion_gemma_yolo_pose",
                    "diffusion_gemma_yolo_obb", "diffusion_gemma_siglip2",
                    "diffusion_gemma_moonvit"],
        "dataset": None,
    },
    "classification": {
        "script": "benchmark_classification.py",
        "title": "Zero-Shot Classification (Tiny ImageNet)",
        "models": ["dinotool", "dinov3", "siglip2", "moonvit"],
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
        "models": ["florence2", "paligemma", "llama_vision", "phi_vision",
                    "cosmos_nemotron", "qwen3_native", "qwen3_thinking"],
        "dataset": None,
    },
    "tracking": {
        "script": "benchmark_tracking.py",
        "title": "Multi-Object Tracking (MOT17)",
        "models": ["yolo26", "yolo11"],
        "dataset": None,
    },
    "6d_pose": {
        "script": "benchmark_6dpose.py",
        "title": "6D Pose Estimation Detection (Linemod)",
        "models": ["yolo26", "yolo11"],
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
        "yolo26_pose": "yolo11-26",
        "yolo26_obb": "yolo11-26",
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
    }.items()}


def run_model(model, script, max_images, dataset=None):
    venv_python = MODEL_VENV.get(model)
    if not venv_python:
        print(f"  [SKIP] No venv configured for {model}")
        return None

    script_path = SCRIPTS_DIR / script
    if "vqa" in script:
        cmd = [venv_python, str(script_path), "--model", model, "--max-questions", str(max_images * 4)]
    else:
        cmd = [venv_python, str(script_path), "--model", model, "--max-images", str(max_images)]
    if dataset:
        cmd += ["--dataset", dataset]

    print(f"\n  ── Running: {model} via {venv_python.split('/')[-3]} ──")
    sys.stdout.flush()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    if result.returncode != 0:
        print(f"  [ERROR] {model} failed (code {result.returncode})")
        return None

    return extract_stats(model, script)


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
        fp = RESULTS_DIR / f"{prefix}_tracking_stats.json"
    elif "6dpose" in script:
        fp = RESULTS_DIR / f"{prefix}_linemod_6dpose_stats.json"
    else:
        for ds in ["coco", "dota"]:
            fp = RESULTS_DIR / f"{prefix}_{ds}_od_stats.json"
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
    parser.add_argument("--max-images", type=int, default=25,
                        help="Images per model per task")
    parser.add_argument("--tasks", nargs="+",
                        choices=list(TASKS) + ["all"],
                        default=["all"],
                        help="Tasks to run (default: all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks_to_run = list(TASKS) if "all" in args.tasks else args.tasks

    all_comparisons = {}

    for task_key in tasks_to_run:
        task = TASKS[task_key]
        print(f"\n{'#' * 70}")
        print(f"# TASK: {task['title']}")
        print(f"{'#' * 70}")

        task_stats = {}
        for model in task["models"]:
            stats = run_model(model, task["script"], args.max_images, task.get("dataset"))
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
