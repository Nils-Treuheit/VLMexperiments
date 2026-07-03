#!/usr/bin/env python3
"""
YOLO + VLM Demo: Runs YOLO detection, pose, and OBB simultaneously,
feeds structured visual observations to a Diffusion LLM for semantic understanding.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# --- Paths ---
BASE = Path(__file__).parent
YOLO_DIR = BASE / "yolo11-26"
DIFFUSION_DIR = BASE / "diffusion_gemma_vl"
LLAMA_CLI = DIFFUSION_DIR / "llama.cpp" / "build" / "bin" / "llama-diffusion-cli"
# Try multiple paths for the GGUF model
_CANDIDATE_GGUF = [
    Path.home() / ".cache" / "huggingface" / "hub" / "diffusiongemma_local" / "diffusiongemma-26B-A4B-it-Q8_0.gguf",
    Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--diffusiongemma-26B-A4B-it-GGUF" / "snapshots" / "aab0a2972da0e41310fbcca5ea63fc47eb932a71" / "diffusiongemma-26B-A4B-it-Q8_0.gguf",
]
MODEL_GGUF = None
for p in _CANDIDATE_GGUF:
    if p.exists():
        MODEL_GGUF = p.resolve()
        break
if MODEL_GGUF is None:
    print("Warning: Diffusion Gemma GGUF model not found. Place it in ~/.cache/huggingface/hub/diffusiongemma_local/", file=sys.stderr)
    MODEL_GGUF = _CANDIDATE_GGUF[0]  # fallback for error message

YOLO_WEIGHTS = {
    "detect": YOLO_DIR / "models" / "yolo11m.pt",
    "pose": YOLO_DIR / "models" / "yolo11n-pose.pt",
    "obb": YOLO_DIR / "models" / "yolo11n-obb.pt",
}
VENV_YOLO = YOLO_DIR / ".venv" / "bin" / "python"
VENV_DIFFUSION = DIFFUSION_DIR / ".venv" / "bin" / "python"

# COCO class names (subset for display)
COCO_NAMES = [
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
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (11, 12), (5, 11), (6, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


def run_yolo_detect(model_path, image_path, conf=0.25):
    script = f"""
import json, sys
from ultralytics import YOLO
model = YOLO(r"{model_path}")
results = model(r"{image_path}", conf={conf}, task="detect", verbose=False)[0]
objs = []
if results.boxes is not None:
    for i in range(len(results.boxes)):
        xyxy = results.boxes.xyxy[i].tolist()
        cls = int(results.boxes.cls[i])
        conf_v = float(results.boxes.conf[i])
        label = {json.dumps(COCO_NAMES)}[cls] if cls < len({json.dumps(COCO_NAMES)}) else f"cls_{{cls}}"
        objs.append({{"bbox": [round(x,1) for x in xyxy], "label": label, "conf": round(conf_v,3)}})
print(json.dumps(objs))
"""
    result = subprocess.run([str(VENV_YOLO), "-c", script],
                            capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout.strip())


def run_yolo_pose(model_path, image_path, conf=0.25):
    script = f"""
import json, sys
from ultralytics import YOLO
model = YOLO(r"{model_path}")
results = model(r"{image_path}", conf={conf}, task="pose", verbose=False)[0]
poses = []
if results.keypoints is not None:
    kp_data = results.keypoints.data  # [N, 17, 3]
    for i in range(len(kp_data)):
        kps = []
        for j in range(17):
            x, y, c = kp_data[i][j].tolist()
            kps.append({{"name": {json.dumps(COCO_KEYPOINTS)}[j], "x": round(x,1), "y": round(y,1), "conf": round(c,3)}})
        bbox = results.boxes.xyxy[i].tolist() if results.boxes is not None else []
        poses.append({{"keypoints": kps, "bbox": [round(x,1) for x in bbox]}})
print(json.dumps(poses))
"""
    result = subprocess.run([str(VENV_YOLO), "-c", script],
                            capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout.strip())


def run_yolo_obb(model_path, image_path, conf=0.25):
    script = f"""
import json, sys
from ultralytics import YOLO
model = YOLO(r"{model_path}")
results = model(r"{image_path}", conf={conf}, task="obb", verbose=False)[0]
objs = []
if results.obb is not None:
    cls_names = ["plane", "ship", "storage_tank", "baseball_diamond", "tennis_court",
                 "basketball_court", "ground_track_field", "harbor", "bridge", "large_vehicle",
                 "small_vehicle", "helicopter", "roundabout", "soccer_ball_field", "swimming_pool"]
    for i in range(len(results.obb)):
        xywhr = results.obb.xywhr[i].tolist()
        cls = int(results.obb.cls[i])
        conf_v = float(results.obb.conf[i])
        label = cls_names[cls] if cls < len(cls_names) else f"cls_{{cls}}"
        objs.append({{"xywhr": [round(x,2) for x in xywhr], "label": label, "conf": round(conf_v,3)}})
print(json.dumps(objs))
"""
    result = subprocess.run([str(VENV_YOLO), "-c", script],
                            capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout.strip())


def format_prompt(detections, poses, obb_results, image_size, min_conf=0.3):
    w, h = image_size
    lines = []
    lines.append(f"Image resolution: {w}x{h} pixels")
    lines.append("")

    detections = [d for d in detections if d["conf"] >= min_conf]
    obb_results = [o for o in obb_results if o["conf"] >= min_conf]

    if detections:
        lines.append("=== Objects Detected ===")
        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            rel_w = (x2 - x1) / w * 100
            rel_h = (y2 - y1) / h * 100
            cx = (x1 + x2) / 2 / w * 100
            cy = (y1 + y2) / 2 / h * 100
            lines.append(f"  - {obj['label']} at ({cx:.0f}%, {cy:.0f}%), size {rel_w:.0f}%x{rel_h:.0f}%, confidence {obj['conf']:.0%}")
        lines.append("")

    if poses:
        lines.append("=== Human Poses Detected ===")
        for i, pose in enumerate(poses):
            if not pose["keypoints"]:
                continue
            kp_str = ", ".join(f"{kp['name']}:({kp['x']:.0f},{kp['y']:.0f})" for kp in pose["keypoints"] if kp["conf"] > 0.3)
            lines.append(f"  Person {i+1}: {kp_str}")
        lines.append("")

    if obb_results:
        lines.append("=== Oriented Objects ===")
        for obj in obb_results:
            x, y, w2, h2, angle = obj["xywhr"]
            cx_pct = x / w * 100
            cy_pct = y / h * 100
            angle_deg = angle * 180 / 3.14159
            lines.append(f"  - {obj['label']} at ({cx_pct:.0f}%, {cy_pct:.0f}%), size {w2:.0f}x{h2:.0f}, rotated {angle_deg:.0f}°, confidence {obj['conf']:.0%}")
        lines.append("")

    if not any([detections, poses, obb_results]):
        lines.append("No objects detected in the scene.")
        lines.append("")

    lines.append("=== Analysis ===")
    lines.append("Describe this scene in 2-3 sentences based on the visual data above. Focus on what is happening, spatial arrangement of key elements, and human activity or intent.")

    return "\n".join(lines)


def run_diffusion_gemma(prompt, max_tokens=256, diffusion_steps=64, temperature=0.3):
    ld_library = str(DIFFUSION_DIR / "llama.cpp" / "build" / "bin")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ld_library + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(LLAMA_CLI),
        "-m", str(MODEL_GGUF),
        "-p", prompt,
        "-n", str(max_tokens),
        "--diffusion-steps", str(diffusion_steps),
        "--temp", str(temperature),
        "-t", "8",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    output = result.stdout
    # Extract generated text after the last diffusion step log
    # The CLI prints: "total time: ... throughput: ..."
    # The generated text is before or after the timing info
    idx = output.find("total time:")
    if idx >= 0:
        output = output[:idx]
    # Remove log lines (starting with timestamps or special chars)
    clean = []
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and "." in line[:6]:
            continue
        if line.startswith(("diffusion", "Throughput", "warning", "Warning", "prompt", "total time", "time per step", "in-step")):
            continue
        clean.append(line)
    joined = "\n".join(clean).strip()
    # Remove leading "thought" if present
    if joined.startswith("thought"):
        joined = joined[len("thought"):].strip()
        if joined.startswith("\n*"):
            joined = joined[1:].strip()
    return joined


def draw_visualization(image_path, detections, poses, obb_results, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for obj in detections:
        x1, y1, x2, y2 = obj["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
        draw.text((x1, y1 - 14), f"{obj['label']} {obj['conf']:.0%}", fill="#00FF00")

    for pose in poses:
        if not pose["keypoints"]:
            continue
        pts = {}
        for kp in pose["keypoints"]:
            if kp["conf"] > 0.3:
                pts[kp["name"]] = (kp["x"], kp["y"])
                r = 4
                draw.ellipse([kp["x"] - r, kp["y"] - r, kp["x"] + r, kp["y"] + r], fill="#FF0000")
        for i, j in COCO_SKELETON:
            n1 = COCO_KEYPOINTS[i]
            n2 = COCO_KEYPOINTS[j]
            if n1 in pts and n2 in pts:
                draw.line([pts[n1], pts[n2]], fill="#FF8800", width=3)

    for obj in obb_results:
        x, y, w, h, angle = obj["xywhr"]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        corners = []
        for dx, dy in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]:
            rx = x + dx * cos_a - dy * sin_a
            ry = y + dx * sin_a + dy * cos_a
            corners.append((rx, ry))
        draw.polygon(corners, outline="#FF00FF", width=3)
        draw.text((corners[0][0], corners[0][1] - 14), obj["label"], fill="#FF00FF")

    img.save(output_path)
    print(f"Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO + Diffusion VLM Demo — semantic understanding from visual observations"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", "-o", help="Output annotated image path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens for LLM")
    parser.add_argument("--diffusion-steps", type=int, default=64, help="Diffusion denoising steps")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM sampling temperature")
    parser.add_argument("--json", action="store_true", help="Output JSON for scripting")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    img = Image.open(args.image).convert("RGB")

    print(f"Image: {args.image} ({img.width}x{img.height})")
    print("Running YOLO detection, pose, and OBB...")
    t0 = time.time()

    detections = run_yolo_detect(YOLO_WEIGHTS["detect"], args.image, args.conf)
    poses = run_yolo_pose(YOLO_WEIGHTS["pose"], args.image, args.conf)
    obb_results = run_yolo_obb(YOLO_WEIGHTS["obb"], args.image, args.conf)

    t1 = time.time()
    print(f"  YOLO done in {t1 - t0:.1f}s")
    print(f"  Detections: {len(detections)}, Poses: {len(poses)}, OBB: {len(obb_results)}")

    prompt = format_prompt(detections, poses, obb_results, (img.width, img.height), min_conf=0.3)

    if args.json:
        print(json.dumps({
            "detections": detections,
            "poses": poses,
            "obb": obb_results,
            "prompt": prompt,
        }))
        return

    print("\n" + "=" * 60)
    print("Prompt to Diffusion Gemma:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)

    print("\nRunning Diffusion Gemma LLM...")
    t2 = time.time()
    response = run_diffusion_gemma(prompt, args.max_tokens, args.diffusion_steps, args.temperature)
    t3 = time.time()
    print(f"  LLM done in {t3 - t2:.1f}s")

    print("\n" + "=" * 60)
    print("Semantic Context / Human Intent:")
    print("=" * 60)
    print(response)
    print("=" * 60)

    if args.output:
        draw_visualization(args.image, detections, poses, obb_results, args.output)


if __name__ == "__main__":
    main()
