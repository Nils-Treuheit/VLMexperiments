#!/usr/bin/env python3
"""
Comprehensive benchmark across all models and task types.
Tests YOLO26, SigLIP2, LocateAnything on: detection, grounding,
pose estimation, scene understanding, etc.
"""
import gc, json, logging, os, re, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
for name in ("transformers_modules", "urllib3", "huggingface_hub", "ultralytics"):
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Results table
# --------------------------------------------------------------------------
RESULTS = []


def add_result(model, task, img_name, time_ms, notes=""):
    RESULTS.append({
        "model": model, "task": task, "image": img_name,
        "time_ms": round(time_ms, 1), "notes": notes,
    })
    print(f"  [{model:>12}] {task:<22} {img_name:<30} {time_ms:>8.1f}ms  {notes}", file=sys.stderr)


# --------------------------------------------------------------------------
# 1. YOLO26
# --------------------------------------------------------------------------
def bench_yolo(images, n_warmup=3, n_run=5):
    from ultralytics import YOLO
    print("\n=== YOLO26 ===", file=sys.stderr)
    yolo = YOLO(Path(__file__).resolve().parent / ".." / "yolo11-26" / "models" / "yolo26m.pt")
    # Warmup
    for _ in range(n_warmup):
        _ = yolo.predict(source=images[0], imgsz=640, conf=0.25, iou=0.45, device='0', verbose=False)

    for img_path in images[:n_run]:
        torch.cuda.synchronize(); t0 = time.time()
        res = yolo.predict(source=img_path, imgsz=640, conf=0.25, iou=0.45, device='0', verbose=False)
        torch.cuda.synchronize(); elapsed = (time.time() - t0) * 1000
        n_dets = len(res[0].boxes) if res[0].boxes is not None else 0
        add_result("YOLO26", "detection", Path(img_path).name, elapsed, f"{n_dets} objects")
    del yolo; gc.collect(); torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 2. SigLIP2
# --------------------------------------------------------------------------
def bench_siglip(images, n_warmup=3, n_run=5):
    from transformers import AutoModel, AutoProcessor
    print("\n=== SigLIP2 ===", file=sys.stderr)
    siglip = AutoModel.from_pretrained(
        "google/siglip2-base-patch16-224", torch_dtype=torch.float16,
        device_map="cuda", attn_implementation="sdpa",
    ).eval()
    proc = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

    for img_path in images[:n_run]:
        img = Image.open(img_path).convert("RGB")
        for _ in range(n_warmup):
            inputs = proc(text=["description of this image"], images=img,
                          padding="max_length", max_length=64, return_tensors="pt")
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad(): _ = siglip(**inputs)

        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad(): out = siglip(**inputs)
        torch.cuda.synchronize(); elapsed = (time.time() - t0) * 1000
        add_result("SigLIP2", "scene_desc", Path(img_path).name, elapsed)
    del siglip; gc.collect(); torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 3. LocateAnything - Different Task Types
# --------------------------------------------------------------------------
def bench_la_tasks(images, n_run=5):
    from unified_engine import LocateAnythingEngine

    print("\n=== LocateAnything ===", file=sys.stderr)
    la = LocateAnythingEngine(compile_llm=False)

    # Different task categories to test
    task_prompts = {
        "grounding (simple)": [
            "find the person",
            "find the car",
            "find the dog",
            "locate the cat",
            "find the chair",
        ],
        "grounding (positional)": [
            "person on the left",
            "car in the center",
            "object on the right side",
            "person in the foreground",
            "animal in the background",
        ],
        "grounding (attribute)": [
            "find the red object",
            "find the large vehicle",
            "find the white animal",
            "find the small object",
            "person wearing a hat",
        ],
        "grounding (relational)": [
            "person next to the car",
            "dog behind the fence",
            "cup on the table",
            "person holding something",
            "car parked in front of building",
        ],
        "detection (label only)": [
            "person",
            "car",
            "bus",
            "dog",
            "cat",
        ],
        "pose estimation": [
            "person standing",
            "person sitting",
            "person walking",
            "person raising arm",
            "person bending",
        ],
        "counting": [
            "count the number of people",
            "how many cars are there",
            "count the animals",
            "how many chairs",
            "count the vehicles",
        ],
        "scene understanding": [
            "describe this scene",
            "what objects are in this image",
            "what is the setting of this image",
            "is this indoor or outdoor",
            "describe what you see",
        ],
    }

    for task_name, prompts in task_prompts.items():
        for img_path in images[:min(n_run, len(images))]:
            img = Image.open(img_path).convert("RGB")
            q = prompts[hash(str(img_path)) % len(prompts)]
            # Warmup once
            la.ground(Image.new("RGB", (224, 224), "gray"), q, max_new_tokens=16, temperature=0)
            torch.cuda.synchronize(); t0 = time.time()
            resp = la.ground(img, q, max_new_tokens=32, temperature=0)
            torch.cuda.synchronize(); elapsed = (time.time() - t0) * 1000
            n_boxes = len(re.findall(r'<box>', resp))
            n_refs = len(re.findall(r'<ref>', resp))
            notes = f"{n_boxes} boxes" if n_boxes > 0 else f"refs={n_refs}"
            add_result("LocateAnything", task_name, Path(img_path).name, elapsed, notes)

    del la; gc.collect(); torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 4. YOLO + SigLIP Hybrid (vs LA for scene+detect)
# --------------------------------------------------------------------------
def bench_hybrid_vs_la(images, n_run=5):
    from unified_engine import LocateAnythingEngine, YOLODetector, SigLIPSceneDescriber

    print("\n=== Hybrid (YOLO+SigLIP) vs LocateAnything ===", file=sys.stderr)
    la = LocateAnythingEngine(compile_llm=False)
    yolo = YOLODetector()
    siglip = SigLIPSceneDescriber()

    for img_path in images[:n_run]:
        img = Image.open(img_path).convert("RGB")
        q = "find the person in this scene"

        # Hybrid: YOLO + SigLIP
        torch.cuda.synchronize(); t0 = time.time()
        dets = yolo.detect(img)
        desc = siglip.describe_text(img)
        torch.cuda.synchronize(); hybrid_time = (time.time() - t0) * 1000

        # LocateAnything
        la.ground(Image.new("RGB", (224, 224), "gray"), q, max_new_tokens=16, temperature=0)
        torch.cuda.synchronize(); t0 = time.time()
        resp = la.ground(img, q, max_new_tokens=32, temperature=0)
        torch.cuda.synchronize(); la_time = (time.time() - t0) * 1000

        add_result("Hybrid", "detect+describe", Path(img_path).name, hybrid_time,
                   f"{len(dets)} detections")
        add_result("LocateAnything", "detect+describe", Path(img_path).name, la_time)

    del la, yolo, siglip; gc.collect(); torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 5. LocateAnything compiled (fast mode)
# --------------------------------------------------------------------------
def bench_la_compiled(images, n_run=5):
    from unified_engine import LocateAnythingEngine
    print("\n=== LocateAnything (compiled, cudagraphs=False) ===", file=sys.stderr)
    la = LocateAnythingEngine(compile_llm=True)
    warmup_img = Image.new("RGB", (224, 224), "gray")
    for _ in range(2):
        la.ground(warmup_img, "find the car", max_new_tokens=16, temperature=0)

    for img_path in images[:n_run]:
        img = Image.open(img_path).convert("RGB")
        torch.cuda.synchronize(); t0 = time.time()
        resp = la.ground(img, "find the car", max_new_tokens=32, temperature=0)
        torch.cuda.synchronize(); elapsed = (time.time() - t0) * 1000
        n_boxes = len(re.findall(r'<box>', resp))
        add_result("LA (compiled)", "grounding", Path(img_path).name, elapsed, f"{n_boxes} boxes")

    del la; gc.collect(); torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 6. Vision encoder: ONNX vs PyTorch
# --------------------------------------------------------------------------
def bench_vision_encoder(images, n_run=5):
    print("\n=== Vision Encoder: ONNX vs PyTorch ===", file=sys.stderr)
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(
        "/mnt/HDD1/Project_Code/vlm_det_test/locate_anything/model",
        dtype=torch.bfloat16, trust_remote_code=True, attn_implementation='sdpa',
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(
        "/mnt/HDD1/Project_Code/vlm_det_test/locate_anything/model",
        trust_remote_code=True,
    )

    # Try ONNX
    has_onnx = False
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "onnx", "vision_encoder.onnx")
    if os.path.exists(onnx_path):
        try:
            sys.path.insert(0, "/mnt/HDD1/Project_Code/vlm_det_test/locate_anything")
            from export_onnx_vision import OnnxVisionEncoder
            onnx_enc = OnnxVisionEncoder()
            has_onnx = True
        except Exception as e:
            print(f"  ONNX not available: {e}", file=sys.stderr)
    else:
        print(f"  ONNX model not found (still exporting?)", file=sys.stderr)

    for img_path in images[:n_run]:
        img = Image.open(img_path).convert("RGB").resize((518, 518), Image.LANCZOS)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": "find"},
        ]}]
        text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images_in, _ = processor.process_vision_info(messages)
        inputs = processor(text=[text], images=images_in, return_tensors="pt")
        pv = inputs["pixel_values"].to(dtype=torch.bfloat16).cuda()
        ghw = inputs["image_grid_hws"]
        if isinstance(ghw, np.ndarray):
            ghw = torch.from_numpy(ghw).cuda()
        else:
            ghw = ghw.cuda()

        # PyTorch
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            feats = model.extract_feature(pv, ghw)
            feats = torch.cat(feats, dim=0)
            feats = model.mlp1(feats)
        torch.cuda.synchronize(); pt_time = (time.time() - t0) * 1000
        add_result("Vision (PT)", "encode", Path(img_path).name, pt_time)

        # ONNX (fixed-size: ONNX model only handles 38x38 grid)
        if has_onnx:
            torch.cuda.synchronize(); t0 = time.time()
            feats2 = onnx_enc(pv.float())
            torch.cuda.synchronize(); onnx_time = (time.time() - t0) * 1000
            add_result("Vision (ONNX)", "encode", Path(img_path).name, onnx_time)

    del model; gc.collect(); torch.cuda.empty_cache()
    if has_onnx:
        del onnx_enc; gc.collect(); torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("=" * 90, file=sys.stderr)
    print("COMPREHENSIVE BENCHMARK: Detection, Grounding, Scene Understanding", file=sys.stderr)
    print("=" * 90, file=sys.stderr)
    print(f"GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", file=sys.stderr)

    # Collect test images from all categories
    demo_dir = Path("/mnt/HDD1/Project_Data/demoMaterial/images")
    all_images = sorted(demo_dir.rglob("*.jpg"))
    print(f"Found {len(all_images)} demo images", file=sys.stderr)

    # Select diverse images (one from each subdir)
    diverse = []
    for subdir in sorted(demo_dir.iterdir()):
        if subdir.is_dir():
            imgs = sorted(subdir.glob("*.jpg"))[:3]
            diverse.extend(imgs)
    print(f"Selected {len(diverse)} diverse images", file=sys.stderr)

    # ===== RUN BENCHMARKS =====
    torch.cuda.empty_cache()

    # 1. YOLO
    bench_yolo(diverse, n_warmup=3, n_run=min(5, len(diverse)))

    # 2. SigLIP
    bench_siglip(diverse, n_warmup=3, n_run=min(5, len(diverse)))

    # 3. Vision encoder (ONNX vs PT)
    bench_vision_encoder(diverse, n_run=min(3, len(diverse)))

    # 4. LocateAnything tasks
    bench_la_tasks(diverse, n_run=min(3, len(diverse)))

    # 5. Compiled LA
    bench_la_compiled(diverse, n_run=min(3, len(diverse)))

    # 6. Hybrid comparison
    bench_hybrid_vs_la(diverse, n_run=min(3, len(diverse)))

    # ===== SUMMARY =====
    print("\n" + "=" * 90, file=sys.stderr)
    print("SUMMARY TABLE", file=sys.stderr)
    print("=" * 90, file=sys.stderr)
    print(f"{'Model':>15} {'Task':<25} {'Count':>6} {'Mean(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}", file=sys.stderr)
    print("-" * 90, file=sys.stderr)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in RESULTS:
        groups[(r["model"], r["task"])].append(r["time_ms"])

    for (model, task), times in sorted(groups.items()):
        mean_t = sum(times) / len(times)
        print(f"{model:>15} {task:<25} {len(times):>6} {mean_t:>10.1f} {min(times):>10.1f} {max(times):>10.1f}",
              file=sys.stderr)

    print(f"\nVRAM peak: {torch.cuda.max_memory_allocated()/1024**2:.0f}MB", file=sys.stderr)
    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
