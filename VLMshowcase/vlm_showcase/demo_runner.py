#!/usr/bin/env python3
"""VLMshowcase — Interactive demo of Vision-Language Model capabilities."""
import argparse
import sys
from pathlib import Path

import numpy as np

from .config import MODELS, COCO_DIR, DEMO_MATERIAL, DOTA_DIR, YOLO_MODEL_PATHS


def banner():
    print("=" * 72)
    print("  VLMshowcase — Vision-Language Model Capability Demo")
    print("=" * 72)
    print(f"  Models available:")
    for key, cfg in MODELS.items():
        print(f"    • {cfg['display']:35s} ({', '.join(cfg['capabilities'])})")
    print("=" * 72)


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
        for d in sorted(mat.iterdir()):
            if d.is_dir():
                files = list(d.glob("*"))
                print(f"  {d.name:18s} {d}/  ({len(files)} files)")
            else:
                print(f"  {d.name:18s} {d}")
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
    print(f"\n{'=' * 72}")
    print(f"  SCENE ANALYSIS — {img.name}")
    print(f"{'=' * 72}")
    result = analyze_scene(str(img))
    for key, val in result.items():
        text = val if isinstance(val, str) else val.get("text", str(val))
        print(f"\n  ── [{key}] ──")
        print(f"  {text[:1500]}{'…' if len(text) > 1500 else ''}")
    print()


def cmd_object_detection(args):
    from .object_detection import detect_yolo, ground_with_locate_anything, detect_yolo_multi
    from .visualization import draw_boxes, save_output
    import cv2

    banner()
    img = _resolve_image(args.image)
    print(f"\n{'=' * 72}")
    print(f"  OBJECT DETECTION — {img.name}")
    print(f"{'=' * 72}")

    img_cv = cv2.imread(str(img))

    if args.yolo:
        for model_key in args.yolo:
            dets, _ = detect_yolo(model_key, str(img), args.confidence)
            print(f"\n  ── YOLO {model_key} ({len(dets)} objects) ──")
            for d in dets[:10]:
                print(f"     {d['class_name']:20s} conf={d['confidence']:.2f}  box={[int(v) for v in d['bbox']]}")
            vis = img_cv.copy()
            draw_boxes(vis, [d["bbox"] for d in dets], [d["class_name"] for d in dets])
            out = save_output(vis, f"/tmp/vlm_demo/yolo_{model_key}_{img.stem}.jpg")
            print(f"     → {out}")

    if args.grounding:
        for query in args.grounding:
            print(f"\n  ── LocateAnything: \"{query}\" ──")
            la_result = ground_with_locate_anything(str(img), query)
            text = la_result.get("text", "")
            boxes = la_result.get("boxes", [])
            print(f"     Response: {text[:500]}")
            print(f"     Boxes: {len(boxes)}")
            if boxes:
                vis = img_cv.copy()
                draw_boxes(vis, boxes)
                out = save_output(vis, f"/tmp/vlm_demo/la_{query}_{img.stem}.jpg")
                print(f"     → {out}")
    print()


def cmd_pose_estimation(args):
    from .pose_estimation import estimate_pose, estimate_pose_multi
    from .visualization import draw_keypoints, save_output, draw_boxes
    import cv2

    banner()
    img = _resolve_image(args.image)
    print(f"\n{'=' * 72}")
    print(f"  POSE ESTIMATION — {img.name}")
    print(f"{'=' * 72}")

    models_to_run = args.model if args.model else ["yolo26n-pose", "yolo11n-pose"]
    img_cv = cv2.imread(str(img))

    for model_key in models_to_run:
        if not YOLO_MODEL_PATHS.get(model_key, Path("/")).exists():
            print(f"  [SKIP] {model_key} — weights not found")
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
        print(f"\n  ── {model_key} ({len(poses)} persons, {len(dets)} detections) ──")
        for i, p in enumerate(poses):
            kp_count = sum(1 for kp in p if len(kp) <= 2 or kp[2] > args.confidence)
            print(f"     Person {i+1}: {kp_count} keypoints above conf threshold")
        vis = img_cv.copy()
        draw_boxes(vis, [d["bbox"] for d in dets], [d["class_name"] for d in dets])
        if poses:
            draw_keypoints(vis, [np.array(p) for p in poses], args.confidence)
        out = save_output(vis, f"/tmp/vlm_demo/pose_{model_key}_{img.stem}.jpg")
        print(f"     → {out}")
    print()


def cmd_human_intent(args):
    from .human_intent import full_intent_analysis, analyze_intent
    banner()
    img = _resolve_image(args.image)
    print(f"\n{'=' * 72}")
    print(f"  HUMAN INTENT ANALYSIS — {img.name}")
    print(f"{'=' * 72}")

    if args.all:
        result = full_intent_analysis(str(img))
        for aspect, text in result.items():
            print(f"\n  ── [{aspect}] ──")
            print(f"  {text[:1000]}{'…' if len(text) > 1000 else ''}")
    else:
        text = analyze_intent(str(img), args.aspect)
        print(f"\n  ── [{args.aspect}] ──")
        print(f"  {text}")
    print()


def cmd_tracking(args):
    from .tracking import track_video, get_tracking_summary
    banner()
    video = _resolve_image(args.video)
    print(f"\n{'=' * 72}")
    print(f"  OBJECT TRACKING — {video.name}")
    print(f"{'=' * 72}")

    model_key = args.model or "yolo26n"
    save_path = Path(f"/tmp/vlm_demo/track_{video.stem}")
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n  Running YOLO {model_key} tracking on {video}...")
    results = track_video(model_key, str(video), args.confidence, save_path)
    summary = get_tracking_summary(results)
    print(f"\n  ── Tracking Summary ({len(summary)} unique tracks) ──")
    for tid, data in sorted(summary.items()):
        print(f"     Track #{tid}: {data['class']} — {len(data['positions'])} frames")
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
    print(f"\n{'=' * 72}")
    print(f"  MODEL COMPARISON — {img.name}")
    print(f"{'=' * 72}")

    print("\n  ── Scene Description Comparison ──")
    print("\n  [Qwen3-VL-8B-Instruct]")
    try:
        desc_i = describe_with_qwen3_instruct(str(img))
        print(f"  {desc_i[:800]}")
    except Exception as e:
        print(f"  [TIMEOUT/ERROR] {e}")
    print("\n  [Qwen3-VL-8B-Thinking]")
    try:
        desc_t = describe_with_qwen3_thinking(str(img))
        print(f"  {desc_t[:800]}")
    except Exception as e:
        print(f"  [TIMEOUT/ERROR] {e}")

    print("\n  ── Object Detection Comparison ──")
    yolo_models = ["yolo26n", "yolo26s", "yolo11m"]
    for m in yolo_models:
        if YOLO_MODEL_PATHS.get(m, Path("/")).exists():
            dets, _ = detect_yolo(m, str(img), args.confidence)
            print(f"  {m:15s}: {len(dets)} objects")
            vis = img_cv.copy()
            draw_boxes(vis, [d["bbox"] for d in dets], [d["class_name"] for d in dets])
            save_output(vis, f"/tmp/vlm_demo/compare_{m}_{img.stem}.jpg")

    print("\n  ── Visual Grounding (LocateAnything) ──")
    for q in ["person", "car", "dog", "bicycle"]:
        la = ground_with_locate_anything(str(img), q)
        n_boxes = len(la.get("boxes", []))
        print(f"  '{
q:12s}': {n_boxes} instances")
    print()


def cmd_batch(args):
    from .batch_processor import batch_process, VLM_KEYS
    banner()
    model_key = args.model
    images = [_resolve_image(p) for p in args.images]
    prompts = args.prompt or []
    mode = args.mode

    print(f"\n{'=' * 72}")
    print(f"  BATCH PROCESSING — {model_key}  ({len(images)} images)")
    print(f"{'=' * 72}")
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
            print(f"  [{r['image']}]  ({inf}s)")
            print(f"    {text[:600]}")
        poses = r.get("poses", [])
        if poses:
            print(f"  [{r['image']}]  {len(poses)} poses  ({inf}s)")
    print()


def cmd_obb(args):
    from .obb_detection import detect_obb
    from .visualization import draw_boxes, save_output
    import cv2

    banner()
    img = _resolve_image(args.image)
    print(f"\n{'=' * 72}")
    print(f"  ORIENTED BOUNDING BOX DETECTION — {img.name}")
    print(f"{'=' * 72}")

    for model_key in args.model:
        if not YOLO_MODEL_PATHS.get(model_key, Path("/")).exists():
            print(f"  [SKIP] {model_key} — weights not found")
            continue
        obbs, results = detect_obb(model_key, str(img), args.confidence)
        print(f"\n  ── {model_key} ({len(obbs)} OBB instances) ──")
        for o in obbs[:10]:
            print(f"     {o['class_name']:20s} conf={o['confidence']:.2f}  "
                  f"cx={o['cx']:.0f} cy={o['cy']:.0f} w={o['w']:.0f} h={o['h']:.0f} θ={o['theta']:.2f}")
        vis = cv2.imread(str(img))
        for o in obbs:
            cx, cy, w, h, theta = o["cx"], o["cy"], o["w"], o["h"], o["theta"]
            rect = ((cx, cy), (w, h), theta * 180 / 3.14159)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            cv2.polylines(vis, [box], True, (0, 255, 0), 2)
            cv2.putText(vis, o["class_name"], tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out = save_output(vis, f"/tmp/vlm_demo/obb_{model_key}_{img.stem}.jpg")
        print(f"     → {out}")
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
    fps = args.fps or 10
    delay = int(1000 / fps)

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
        cv2.imshow("VLMshowcase — Webcam", frame)
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
    mat = DEMO_MATERIAL / p
    if mat.exists():
        return mat
    coco_samples = sorted((COCO_DIR / "val2017").glob("*.jpg"))
    if coco_samples:
        return coco_samples[0]
    raise FileNotFoundError(f"Image not found: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="VLMshowcase — Demonstrate VLM capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  vlm-demo scene image.jpg\n"
            "  vlm-demo detect image.jpg --yolo yolo26n yolo26s --grounding person car\n"
            "  vlm-demo pose image.jpg --model yolo26n-pose\n"
            "  vlm-demo intent image.jpg --all\n"
            "  vlm-demo track video.mp4\n"
            "  vlm-demo compare image.jpg\n"
            "  vlm-demo batch yolo26n img1.jpg img2.jpg\n"
            "  vlm-demo batch locate_anything img1.jpg img2.jpg --prompt 'find car' 'find person'\n"
            "  vlm-demo batch qwen3_thinking img1.jpg img2.jpg --mode describe\n"
            "  vlm-demo obb aerial.png --model yolo26n-obb\n"
            "  vlm-demo webcam --model yolo26n --fps 15\n"
            "  vlm-demo list\n"
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

    p_batch = sub.add_parser("batch", help="Load model once, process multiple images")
    p_batch.add_argument("model", help="Model key (yolo26n, locate_anything, qwen3_instruct, qwen3_thinking, yolo26n-pose)")
    p_batch.add_argument("images", nargs="+", help="One or more image paths")
    p_batch.add_argument("--prompt", nargs="*", default=[], help="Prompt(s) per image (reuses last if fewer)")
    p_batch.add_argument("--mode", default=None, help="describe|detect|pose|ground")

    p_obb = sub.add_parser("obb", help="Oriented bounding box detection (aerial images)")
    p_obb.add_argument("image", help="DOTA aerial image path")
    p_obb.add_argument("--model", nargs="+", default=["yolo11n-obb", "yolo26n-obb"], help="OBB model keys")

    p_webcam = sub.add_parser("webcam", help="Live webcam detection")
    p_webcam.add_argument("--model", default="yolo26n", help="YOLO model key")
    p_webcam.add_argument("--fps", type=int, default=10, help="Target FPS")

    args = parser.parse_args()

    command_map = {
        "list": cmd_list_samples,
        "scene": cmd_scene_analysis,
        "detect": cmd_object_detection,
        "pose": cmd_pose_estimation,
        "intent": cmd_human_intent,
        "track": cmd_tracking,
        "compare": cmd_compare,
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
