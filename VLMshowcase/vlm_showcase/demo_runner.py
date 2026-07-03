#!/usr/bin/env python3
"""VLMshowcase — Interactive demo of Vision-Language Model capabilities."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .config import MODELS, COCO_DIR, DEMO_MATERIAL, DOTA_DIR, YOLO_MODEL_PATHS, VLM_MODEL_KEYS
from .subprocess_utils import run_with_timer

MAX_NUM_DET = 25  # 10

def _find_char(text,char,start_pos=0,default=2000,reversed=False):
    idx = text[::-1].find(char, start_pos) if reversed else text.find(char, start_pos)
    return default if idx==-1 else idx

def _get_cutoff(text, soft_lim=1000):
    last_idx_prior_soft_lim = len(text)-min(_find_char(text,'.',reversed=True),
                                            _find_char(text,'\u2026',reversed=True),
                                            _find_char(text,'\n',reversed=True))
    if len(text)<soft_lim: return last_idx_prior_soft_lim+1
    max_id = min(_find_char(text,'.',soft_lim,default=len(text)),
                 _find_char(text,'\u2026',soft_lim,default=len(text)),
                 _find_char(text,'\n',soft_lim,default=len(text)))
    return (max_id if len(text)>max_id else last_idx_prior_soft_lim)+1

def banner():
    print("=" * 96)
    print("  VLMshowcase — Vision-Language Model Capability Demo")
    print("=" * 96)
    for key, cfg in MODELS.items():
        print(f"    \u2022 {cfg['display']:35s} ({', '.join(cfg['capabilities'])})")
    print("=" * 96)


def cmd_list_samples(args):
    banner()
    print("\nDemo material locations:\n")
    coco = COCO_DIR / "val2017"
    if coco.exists():
        print(f"  COCO val2017:    {coco}  ({len(list(coco.glob('*.jpg')))} images)")
    dota = DOTA_DIR / "images"
    if dota.exists():
        print(f"  DOTA v1.0:       {dota}  ({len(list(dota.glob('*.png')))} images)")
    mat = DEMO_MATERIAL
    if mat.exists():
        print(f"\n  Demo images (by category):")
        img_dir = mat / "images"
        if img_dir.exists():
            for d in sorted(img_dir.iterdir()):
                if d.is_dir():
                    files = list(d.glob("*"))
                    print(f"    {d.name:22s} {len(files)} images")
        vid_dir = mat / "videos"
        if vid_dir.exists():
            print(f"\n  Demo videos:")
            for f in sorted(vid_dir.iterdir()):
                print(f"    {f.name}")
    print(f"\n  YOLO models:     {Path(next(iter(YOLO_MODEL_PATHS.values()))).parent}/")
    print(f"  ({sum(1 for p in YOLO_MODEL_PATHS.values() if p.exists())} available)")
    print()
    coco_sample = Path(coco) / "000000000139.jpg"
    if coco_sample.exists():
        print(f"  Quick test image: {coco_sample}")


def cmd_scene_analysis(args):
    from .scene_analysis import analyze_scene
    banner()
    img = _resolve_image(args.image)
    print(f"\n{'=' * 96}")
    print(f"  SCENE ANALYSIS — {img.name}")
    print(f"{'=' * 96}")
    result = analyze_scene(str(img))
    for key, val in result.items():
        text = val if isinstance(val, str) else val.get("text", str(val))
        max_len = _get_cutoff(text,1500)
        print(f"\n  \u2500\u2500 [{key}] \u2500\u2500")
        print(f"  {text[:max_len]}{'\u2026' if len(text) > max_len else ''}")
    print()


def cmd_object_detection(args):
    from .object_detection import detect_yolo, ground_with_locate_anything, detect_yolo_multi
    from .visualization import draw_boxes, save_output
    import cv2

    banner()
    img = _resolve_image(args.image)
    print(f"\n{'=' * 96}")
    print(f"  OBJECT DETECTION — {img.name}")
    print(f"{'=' * 96}")

    img_cv = cv2.imread(str(img))

    if args.yolo:
        for model_key in args.yolo:
            dets, _ = detect_yolo(model_key, str(img), args.confidence)
            print(f"\n  \u2500\u2500 YOLO {model_key} ({len(dets)} objects) \u2500\u2500")
            for d in dets[:min(len(dets),MAX_NUM_DET)]:
                print(f"     {d['class_name']:20s} conf={d['confidence']:.2f}  box={[int(v) for v in d['bbox']]}")
            vis = img_cv.copy()
            draw_boxes(vis, [d["bbox"] for d in dets], [d["class_name"] for d in dets])
            out = save_output(vis, f"/tmp/vlm_demo/yolo_{model_key}_{img.stem}.jpg")
            print(f"     \u2192 {out}")

    if args.grounding:
        for query in args.grounding:
            print(f"\n  \u2500\u2500 LocateAnything: \"{query}\" \u2500\u2500")
            la_result = ground_with_locate_anything(str(img), query)
            text = la_result.get("text", "")
            boxes = la_result.get("boxes", [])
            print(f"     Response: {text[:500]}")
            print(f"     Boxes: {len(boxes)}")
            if boxes:
                vis = img_cv.copy()
                draw_boxes(vis, boxes)
                out = save_output(vis, f"/tmp/vlm_demo/la_{query}_{img.stem}.jpg")
                print(f"     \u2192 {out}")
    print()


def cmd_pose_estimation(args):
    from .pose_estimation import estimate_pose, estimate_pose_multi
    from .visualization import draw_keypoints, save_output, draw_boxes
    import cv2

    banner()
    img = _resolve_image(args.image)
    print(f"\n{'=' * 96}")
    print(f"  POSE ESTIMATION — {img.name}")
    print(f"{'=' * 96}")

    models_to_run = args.model if args.model else ["yolo26n-pose", "yolo11n-pose"]
    img_cv = cv2.imread(str(img))

    for model_key in models_to_run:
        if not YOLO_MODEL_PATHS.get(model_key, Path("/")).exists():
            print(f"  [SKIP] {model_key} \u2014 weights not found")
            continue
        poses, results = estimate_pose(model_key, str(img), args.confidence)
        dets = []
        for r in results:
            if r.boxes is not None:
                for box, conf_val, cls in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                ):
                    dets.append({"bbox": box.tolist(), "class_name": r.names[int(cls)], "confidence": float(conf_val)})
        print(f"\n  \u2500\u2500 {model_key} ({len(poses)} persons, {len(dets)} detections) \u2500\u2500")
        for i, p in enumerate(poses):
            kp_count = sum(1 for kp in p if len(kp) <= 2 or kp[2] > args.confidence)
            print(f"     Person {i+1}: {kp_count} keypoints above conf threshold")
        vis = img_cv.copy()
        draw_boxes(vis, [d["bbox"] for d in dets], [d["class_name"] for d in dets])
        if poses:
            draw_keypoints(vis, [np.array(p) for p in poses], args.confidence)
        out = save_output(vis, f"/tmp/vlm_demo/pose_{model_key}_{img.stem}.jpg")
        print(f"     \u2192 {out}")
    print()


def cmd_human_intent(args):
    from .human_intent import full_intent_analysis, analyze_intent
    banner()
    img = _resolve_image(args.image)
    print(f"\n{'=' * 96}")
    print(f"  HUMAN INTENT ANALYSIS — {img.name}")
    print(f"{'=' * 96}")

    if args.all:
        result = full_intent_analysis(str(img))
        for aspect, text in result.items():
            max_len = _get_cutoff(text,1000)
            print(f"\n  \u2500\u2500 [{aspect}] \u2500\u2500")
            print(f"  {text[:max_len]}{'\u2026' if len(text) > max_len else ''}")
    else:
        text = analyze_intent(str(img), args.aspect)
        print(f"\n  \u2500\u2500 [{args.aspect}] \u2500\u2500")
        print(f"  {text}")
    print()


def cmd_tracking(args):
    from .tracking import track_video, get_tracking_summary
    banner()
    video = _resolve_image(args.video)
    print(f"\n{'=' * 96}")
    print(f"  OBJECT TRACKING — {video.name}")
    print(f"{'=' * 96}")

    model_key = args.model or "yolo26n"
    save_path = Path(f"/tmp/vlm_demo/track_{video.stem}")
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n  Running YOLO {model_key} tracking on {video}...")
    results = track_video(model_key, str(video), args.confidence, save_path)
    summary = get_tracking_summary(results)
    print(f"\n  \u2500\u2500 Tracking Summary ({len(summary)} unique tracks) \u2500\u2500")
    for tid, data in sorted(summary.items()):
        print(f"     Track #{tid}: {data['class']} \u2014 {len(data['positions'])} frames")
    if results:
        print(f"\n  Saved to: {save_path}")
    print()


def cmd_compare(args):
    from .object_detection import detect_yolo, ground_with_locate_anything
    from .scene_analysis import describe_with_qwen3_instruct, describe_with_qwen3_thinking
    from .visualization import draw_boxes, save_output
    import cv2

    banner()
    img = _resolve_image(args.image)
    img_cv = cv2.imread(str(img))
    print(f"\n{'=' * 96}")
    print(f"  MODEL COMPARISON — {img.name}")
    print(f"{'=' * 96}")

    print("\n  \u2500\u2500 Scene Description Comparison \u2500\u2500")
    print("\n  [Qwen3-VL-8B-Instruct]")
    try:
        desc_i = describe_with_qwen3_instruct(str(img))
        max_len = _get_cutoff(desc_i,1000)
        # 800 seems like a little too short
        print(f"  {desc_i[:max_len]}{'[...]' if len(str(desc_i)) > max_len else ''}")
    except Exception as e:
        print(f"  [TIMEOUT/ERROR] {e}")
    print("\n  [Qwen3-VL-8B-Thinking]")
    try:
        desc_t = describe_with_qwen3_thinking(str(img))
        max_len = _get_cutoff(desc_t,1000)
        # 800 seems like a little too short
        print(f"  {desc_t[:max_len]}{'[...]' if len(str(desc_t)) > max_len else ''}")
    except Exception as e:
        print(f"  [TIMEOUT/ERROR] {e}")

    print("\n  \u2500\u2500 Object Detection Comparison \u2500\u2500")
    yolo_models = ["yolo26n", "yolo26s", "yolo11m"]
    for m in yolo_models:
        if YOLO_MODEL_PATHS.get(m, Path("/")).exists():
            dets, _ = detect_yolo(m, str(img), args.confidence)
            print(f"  {m:15s}: {len(dets)} objects")
            vis = img_cv.copy()
            draw_boxes(vis, [d["bbox"] for d in dets], [d["class_name"] for d in dets])
            save_output(vis, f"/tmp/vlm_demo/compare_{m}_{img.stem}.jpg")

    print("\n  \u2500\u2500 Visual Grounding (LocateAnything) \u2500\u2500")
    for q in ["person", "car", "dog", "bicycle"]:
        la = ground_with_locate_anything(str(img), q)
        n_boxes = len(la.get("boxes", []))
        print(f"  '{q:12s}': {n_boxes} instances")
    print()


def cmd_batch(args):
    from .batch_processor import batch_process
    banner()
    model_key = args.model
    images = [_resolve_image(p) for p in args.images]
    prompts = args.prompt or []
    mode = args.mode

    print(f"\n{'=' * 96}")
    print(f"  BATCH PROCESSING — {model_key}  ({len(images)} images)")
    print(f"{'=' * 96}")
    print()
    sys.stderr.write(f"Loading {model_key} ...\n")
    sys.stderr.flush()

    load_time, results = batch_process(model_key, images, prompts, args.confidence, mode)

    print(f"\n  Model loaded in {load_time}s  |  {len(images)} image(s)\n")
    for r in results:
        inf = r.get("time_sec", 0)
        dets = r.get("detections", [])
        result = r.get("result", "")
        if dets:
            print(f"  [{r['image']}]  {len(dets)} objects  ({inf}s)")
            for d in dets[:5]:
                print(f"    {d.get('class', ''):20s} conf={d.get('confidence', 0):.2f}")
        elif result:
            text = result if isinstance(result, str) else str(result)
            max_len = _get_cutoff(text,1000)
            # 600 seemed a bit short
            print(f"  [{r['image']}]  ({inf}s)")
            print(f"    {text[:max_len]}{'[...]' if len(str(text)) > max_len else ''}")
        poses = r.get("poses", [])
        if poses:
            print(f"  [{r['image']}]  {len(poses)} poses  ({inf}s)")
        print(" ")
    print("\n")


def cmd_obb(args):
    from .obb_detection import detect_obb
    from .visualization import draw_boxes, save_output
    import cv2

    banner()
    img = _resolve_image(args.image)
    print(f"\n{'=' * 96}")
    print(f"  ORIENTED BOUNDING BOX DETECTION — {img.name}")
    print(f"{'=' * 96}")

    for model_key in args.model:
        if not YOLO_MODEL_PATHS.get(model_key, Path("/")).exists():
            print(f"  [SKIP] {model_key} \u2014 weights not found")
            continue
        obbs, results = detect_obb(model_key, str(img), args.confidence)
        print(f"\n  \u2500\u2500 {model_key} ({len(obbs)} OBB instances) \u2500\u2500")
        for o in obbs[:min(len(obbs),MAX_NUM_DET)]:
            print(f"     {o['class_name']:20s} conf={o['confidence']:.2f}  "
                  f"cx={o['cx']:.0f} cy={o['cy']:.0f} w={o['w']:.0f} h={o['h']:.0f} \u03b8={o['theta']:.2f}")
        vis = cv2.imread(str(img))
        for o in obbs:
            cx, cy, w, h, theta = o["cx"], o["cy"], o["w"], o["h"], o["theta"]
            rect = ((cx, cy), (w, h), theta * 180 / 3.14159)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            cv2.polylines(vis, [box], True, (0, 255, 0), 2)
            cv2.putText(vis, o["class_name"], tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out = save_output(vis, f"/tmp/vlm_demo/obb_{model_key}_{img.stem}.jpg")
        print(f"     \u2192 {out}")
    print()


def cmd_vlm(args):
    """Run any VLM model with a custom prompt."""
    from .batch_processor import batch_process
    banner()
    model_key = args.model
    img = _resolve_image(args.image)
    prompt = args.prompt

    print(f"\n{'=' * 96}")
    print(f"  VLM: {MODELS[model_key]['display']} — {img.name}")
    print(f"  Prompt: {prompt}")
    print(f"{'=' * 96}")

    sys.stderr.write(f"Loading {model_key} ...\n")
    sys.stderr.flush()
    load_time, results = batch_process(model_key, [img], [prompt], mode=args.mode)
    print(f"\n  Loaded in {load_time}s | Inference: {results[0].get('time_sec', 0)}s\n")
    result = results[0].get("result", results[0].get("detections", ""))
    if isinstance(result, list):
        for d in result[:min(len(result),MAX_NUM_DET)]:
            print(f"    {d.get('class', ''):20s} conf={d.get('confidence', 0):.2f}")
    else:
        max_len = _get_cutoff(result,2000)
        print(f"  {result[:max_len]}{'[...]' if len(str(result)) > max_len else ''}")
    print()


def cmd_run(args):
    """Run a model on all images in a folder."""
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    images = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    if not images:
        print(f"ERROR: No images found in {folder}")
        sys.exit(1)

    if args.recurse:
        images = sorted([
            p for p in folder.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])

    model_key = args.model
    if model_key == "all":
        banner()
        print(f"\n{'=' * 96}")
        print(f"  RUN ALL MODELS ON FOLDER — {folder}")
        print(f"  {len(images)} images found")
        print(f"{'=' * 96}")
        for key in sorted(MODELS.keys()):
            print(f"\n--- {MODELS[key]['display']} ---")
            try:
                _run_on_folder(key, images, args)
            except Exception as e:
                print(f"  [ERROR] {e}")
        return

    if model_key not in MODELS:
        print(f"ERROR: Unknown model '{model_key}'. Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    banner()
    _run_on_folder(model_key, images, args)


def _run_on_folder(model_key, images, args):
    from .batch_processor import batch_process
    prompt = args.prompt
    mode = args.mode
    use_batch = args.batch

    print(f"\n{'=' * 96}")
    print(f"  RUN: {MODELS[model_key]['display']}  |  {len(images)} images")
    if use_batch:
        print(f"  Mode: batch (load once)")
    else:
        print(f"  Mode: single (load per image)")
    print(f"{'=' * 96}")

    if use_batch:
        sys.stderr.write(f"Loading {model_key} ...\n")
        sys.stderr.flush()
        load_time, results = batch_process(model_key, images, [prompt] * len(images), args.confidence, mode)
        print(f"\n  Model loaded in {load_time}s\n")
        for r in results:
            inf = r.get("time_sec", 0)
            res = r.get("result", r.get("detections", ""))
            label = Path(r["image"]).name
            if isinstance(res, list):
                print(f"  {label:30s} {len(res)} detections  ({inf}s)")
            else:
                text = str(res)[:120].replace("\n", " ")
                print(f"  {label:30s} ({inf}s) {text}")
    else:
        for img in images:
            t0 = time.time()
            _, results = batch_process(model_key, [img], [prompt], args.confidence, mode)
            inf = results[0].get("time_sec", time.time() - t0)
            res = results[0].get("result", results[0].get("detections", ""))
            text = str(res)[:200].replace("\n", " ")
            print(f"  {img.name:30s} ({inf}s) {text}")
    print()


def cmd_webcam(args):
    import cv2
    print("Starting webcam demo...")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Could not open webcam")
        sys.exit(1)

    from .object_detection import detect_yolo
    model_key = args.model or "yolo26n"
    fps = args.fps or 30
    delay = int(960 / fps)

    print(f"  Model: {model_key}  FPS cap: {fps}  Press 'q' to quit")
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        dets, _ = detect_yolo(model_key, frame, args.confidence)
        from .visualization import draw_boxes
        frame = draw_boxes(frame, [d["bbox"] for d in dets], [d["class_name"] for d in dets])
        cv2.putText(frame, f"YOLO {model_key}: {len(dets)} objects", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("VLMshowcase \u2014 Webcam", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()


def _resolve_image(path):
    p = Path(path)
    if p.exists():
        return p
    coco = COCO_DIR / "val2017" / p
    if coco.exists():
        return coco
    dota = DOTA_DIR / "images" / p
    if dota.exists():
        return dota
    mat = DEMO_MATERIAL / "images"
    if mat.exists():
        for sub in mat.iterdir():
            candidate = sub / p
            if candidate.exists():
                return candidate
    mat_vid = DEMO_MATERIAL / "videos" / p
    if mat_vid.exists():
        return mat_vid
    coco_samples = sorted((COCO_DIR / "val2017").glob("*.jpg"))
    if coco_samples:
        return coco_samples[0]
    raise FileNotFoundError(f"File not found: {path}")


def main():
    model_choices = sorted(MODELS.keys())

    parser = argparse.ArgumentParser(
        description="VLMshowcase \u2014 Demonstrate VLM capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Commands:\n"
            "  list                        Show models and available material\n"
            "  scene     <image>           Analyze scene (Qwen3 + YOLO)\n"
            "  detect    <image>           Object detection (YOLO + LA)\n"
            "  pose      <image>           Pose estimation\n"
            "  intent    <image>           Human intent analysis\n"
            "  track     <video>           Object tracking\n"
            "  compare   <image>           Compare all models\n"
            "  vlm       <model> <img>     Run any VLM with custom prompt\n"
            "                             <prompt>\n"
            "  run       <folder>          Process all images in a folder\n"
            "  batch     <model> <imgs..>  Load-once batch processing\n"
            "  obb       <image>           Oriented bounding boxes\n"
            "  webcam                      Live webcam YOLO detection\n"
            "\n"
            "Examples:\n"
            "  vlm-demo scene image.jpg\n"
            "  vlm-demo vlm florence2 img.jpg 'describe the scene'\n"
            "  vlm-demo run /path/to/folder --model yolo26n --batch\n"
            "  vlm-demo run /path/to/folder --model all --batch\n"
            "  vlm-demo batch yolo26n img1.jpg img2.jpg\n"
        ),
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available demo material and models")

    p_scene = sub.add_parser("scene", help="Analyze scene (description + detection)")
    p_scene.add_argument("image", help="Image path")

    p_detect = sub.add_parser("detect", help="Object detection")
    p_detect.add_argument("image", help="Image path")
    p_detect.add_argument("--yolo", nargs="+", default=["yolo26n"], help="YOLO model keys")
    p_detect.add_argument("--grounding", nargs="*", default=[], help="Queries for LocateAnything")

    p_pose = sub.add_parser("pose", help="Pose estimation")
    p_pose.add_argument("image", help="Image path")
    p_pose.add_argument("--model", nargs="+", default=["yolo26n-pose", "yolo11n-pose"], help="Pose model keys")

    p_intent = sub.add_parser("intent", help="Human intent analysis")
    p_intent.add_argument("image", help="Image path")
    p_intent.add_argument("--aspect", choices=["action", "intent", "emotion", "social"], default="action")
    p_intent.add_argument("--all", action="store_true", help="Run all aspects")

    p_track = sub.add_parser("track", help="Object tracking (video)")
    p_track.add_argument("video", help="Video path")
    p_track.add_argument("--model", default="yolo26n", help="YOLO model key")

    p_comp = sub.add_parser("compare", help="Compare all models on same image")
    p_comp.add_argument("image", help="Image path")

    p_vlm = sub.add_parser("vlm", help="Run any VLM model with custom prompt")
    p_vlm.add_argument("model", choices=model_choices, help="Model key")
    p_vlm.add_argument("image", help="Image path")
    p_vlm.add_argument("prompt", nargs="?", default="Describe this image in detail.",
                       help="Prompt text")
    p_vlm.add_argument("--mode", default=None, help="Inference mode (describe/detect/ground)")

    p_run = sub.add_parser("run", help="Process all images in a folder")
    p_run.add_argument("folder", help="Folder path containing images")
    p_run.add_argument("--model", default="yolo26n", help="Model key (or 'all')")
    p_run.add_argument("--prompt", default="", help="Custom prompt for VLM models")
    p_run.add_argument("--mode", default=None, help="Inference mode")
    p_run.add_argument("--batch", action="store_true", default=True,
                       help="Use batch mode (load model once)")
    p_run.add_argument("--no-batch", action="store_false", dest="batch",
                       help="Load model per image")
    p_run.add_argument("--recurse", action="store_true", help="Search subdirectories")

    p_batch = sub.add_parser("batch", help="Load model once, process multiple images")
    p_batch.add_argument("model", help="Model key")
    p_batch.add_argument("images", nargs="+", help="One or more image paths")
    p_batch.add_argument("--prompt", nargs="*", default=[], help="Prompt(s) per image")
    p_batch.add_argument("--mode", default=None, help="describe|detect|pose|ground")
    p_batch.add_argument("--confidence", type=float, default=0.25, help="Detection confidence")

    p_obb = sub.add_parser("obb", help="Oriented bounding box detection (aerial)")
    p_obb.add_argument("image", help="DOTA aerial image path")
    p_obb.add_argument("--model", nargs="+", default=["yolo11n-obb", "yolo26n-obb"], help="OBB model keys")

    p_webcam = sub.add_parser("webcam", help="Live webcam detection")
    p_webcam.add_argument("--model", default="yolo26n", help="YOLO model key")
    p_webcam.add_argument("--fps", type=int, default=30, help="Target FPS")

    args = parser.parse_args()

    command_map = {
        "list": cmd_list_samples,
        "scene": cmd_scene_analysis,
        "detect": cmd_object_detection,
        "pose": cmd_pose_estimation,
        "intent": cmd_human_intent,
        "track": cmd_tracking,
        "compare": cmd_compare,
        "vlm": cmd_vlm,
        "run": cmd_run,
        "batch": cmd_batch,
        "obb": cmd_obb,
        "webcam": cmd_webcam,
    }

    fn = command_map.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
