#!/usr/bin/env python3
"""
Multi-object tracking benchmark on MOT17 using YOLO + built-in tracking.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_DIR, RESULTS_DIR, PROJECT_DIR, save_stats, print_comparison

MOT17_DIR = Path("/mnt/HDD1/Project_Data/public_datasets/mot17/train")
MOT17_MINIMAL_DIR = Path("/mnt/HDD1/Project_Data/public_datasets/mot17_minimal")

TRACKING_MODELS = {"yolo11", "yolo11s", "yolo11m", "yolo26", "yolo26s", "yolo26m"}

MODEL_TO_YOLO_NAME = {
    "yolo11": "yolo11n", "yolo11s": "yolo11s", "yolo11m": "yolo11m",
    "yolo26": "yolo26n", "yolo26s": "yolo26s", "yolo26m": "yolo26m",
}


def find_mot17_sequences():
    """Find available MOT17 sequences."""
    for base_dir in [MOT17_DIR, MOT17_MINIMAL_DIR]:
        if base_dir.exists():
            seqs = sorted([d for d in base_dir.iterdir() if d.is_dir() and "-" in d.name])
            if seqs:
                return seqs, base_dir
    return [], None


def load_mot_gt(seq_dir):
    """Load ground truth tracking data from a MOT17 sequence."""
    gt_file = seq_dir / "gt" / "gt.txt"
    if not gt_file.exists():
        return None
    
    gt_by_frame = defaultdict(list)
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 9:
                frame = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf = float(parts[6])
                class_id = int(parts[7])
                visibility = float(parts[8])
                if class_id == 1:  # person only (MOT17 standard)
                    gt_by_frame[frame].append({
                        "track_id": track_id,
                        "bbox": [x, y, x + w, y + h],
                        "visibility": visibility,
                    })
    
    return dict(gt_by_frame)


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def compute_mot_metrics(gt_by_frame, detections_by_frame, iou_threshold=0.5):
    """
    Simplified MOT evaluation.
    detections_by_frame: dict of frame_num -> list of {track_id, bbox, conf}
    """
    total_gt = 0
    total_fn = 0
    total_fp = 0
    total_idsw = 0
    total_motp = 0.0
    motp_count = 0
    
    # Track ID mapping from previous frame
    prev_matches = {}  # gt_track_id -> det_track_id
    
    for frame_num in sorted(set(list(gt_by_frame.keys()) + list(detections_by_frame.keys()))):
        gt_objs = gt_by_frame.get(frame_num, [])
        det_objs = detections_by_frame.get(frame_num, [])
        
        total_gt += len(gt_objs)
        
        if not gt_objs:
            total_fp += len(det_objs)
            continue
        if not det_objs:
            total_fn += len(gt_objs)
            prev_matches = {}
            continue
        
        # Build IoU matrix between GT and detections
        iou_matrix = np.zeros((len(gt_objs), len(det_objs)))
        for gi, gt in enumerate(gt_objs):
            for di, det in enumerate(det_objs):
                iou_matrix[gi, di] = compute_iou(gt["bbox"], det["bbox"])
        
        # Greedy matching
        matched_gt = set()
        matched_det = set()
        frame_matches = []  # (gt_idx, det_idx)
        
        # Sort by IoU descending
        indices = np.dstack(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape))[0]
        for gi, di in indices:
            if gi in matched_gt or di in matched_det:
                continue
            if iou_matrix[gi, di] >= iou_threshold:
                matched_gt.add(gi)
                matched_det.add(di)
                frame_matches.append((gi, di))
                total_motp += iou_matrix[gi, di]
                motp_count += 1
        
        total_fn += len(gt_objs) - len(matched_gt)
        total_fp += len(det_objs) - len(matched_det)
        
        # Count ID switches
        current_matches = {}
        for gi, di in frame_matches:
            gt_id = gt_objs[gi]["track_id"]
            det_id = det_objs[di]["track_id"]
            current_matches[gt_id] = det_id
            
            if gt_id in prev_matches and prev_matches[gt_id] != det_id:
                total_idsw += 1
        
        prev_matches = current_matches
    
    mota = 1.0 - (total_fn + total_fp + total_idsw) / total_gt if total_gt > 0 else 0
    motp = total_motp / motp_count if motp_count > 0 else 0
    
    return {
        "MOTA": round(mota, 4),
        "MOTP": round(motp, 4),
        "FN": total_fn,
        "FP": total_fp,
        "IDSW": total_idsw,
        "GT": total_gt,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-object tracking benchmark")
    parser.add_argument("--model", type=str, default="yolo26",
                        choices=sorted(TRACKING_MODELS),
                        help="Model to benchmark")
    parser.add_argument("--max-sequences", type=int, default=2,
                        help="Number of MOT17 sequences to evaluate (default: 2)")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Max frames per sequence (default: 100)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Total images per model (mapped to --max-frames, overrides default)")
    parser.add_argument("--tracker", type=str, default="botsort",
                        choices=["botsort", "bytetrack"],
                        help="Tracker type (default: botsort)")
    parser.add_argument("--samples-file", type=str, default=None, help="Path to samples file (unused, for compatibility)")
    args = parser.parse_args()
    if args.max_images is not None:
        args.max_frames = args.max_images // args.max_sequences
    
    # Find MOT17 sequences
    sequences, mot_dir = find_mot17_sequences()
    if not sequences:
        print("Error: MOT17 dataset not found. Download from https://motchallenge.net/")
        print(f"  Looked in: {MOT17_DIR}")
        sys.exit(1)
    
    print(f"\nTracking Benchmark: {args.model}")
    print(f"  MOT dataset: {mot_dir}")
    print(f"  Sequences: {len(sequences)} available, using {args.max_sequences}")
    
    # Load model
    sys.path.insert(0, str(PROJECT_DIR / "yolo11-26"))
    model_name = MODEL_TO_YOLO_NAME.get(args.model, f"{args.model}n")
    from ultralytics import YOLO
    
    models_dir = PROJECT_DIR / "yolo11-26" / "models"
    weight_path = models_dir / f"{model_name}.pt"
    if weight_path.exists():
        model = YOLO(str(weight_path))
    else:
        model = YOLO(model_name)
    
    total_time = 0.0
    total_frames = 0
    all_metrics = []
    
    for seq_idx, seq_dir in enumerate(sequences[:args.max_sequences]):
        seq_name = seq_dir.name
        print(f"\n  Sequence: {seq_name}")
        
        gt = load_mot_gt(seq_dir)
        if gt is None:
            print(f"    No ground truth found, skipping")
            continue
        
        img_dir = seq_dir / "img1"
        img_files = sorted(img_dir.glob("*.jpg"))[:args.max_frames]
        
        if not img_files:
            print(f"    No images found, skipping")
            continue
        
        print(f"    Frames: {len(img_files)}")
        
        frame_detections = {}
        seq_time = 0.0
        
        for frame_idx, img_path in enumerate(img_files):
            frame_num = frame_idx + 1
            t0 = time.time()
            
            tracker_cfg = f"{args.tracker}.yaml"
            results = model.track(str(img_path), persist=True, verbose=False,
                                  tracker=tracker_cfg,
                                  device="cuda" if __import__('torch').cuda.is_available() else "cpu")
            
            elapsed = time.time() - t0
            seq_time += elapsed
            
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id
                confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else [1.0] * len(boxes)
                
                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy().astype(int)
                    for box, tid, conf in zip(boxes, track_ids, confs):
                        detections.append({
                            "track_id": int(tid),
                            "bbox": box.tolist(),
                            "conf": float(conf),
                        })
            
            frame_detections[frame_num] = detections
        
        metrics = compute_mot_metrics(gt, frame_detections)
        metrics["sequence"] = seq_name
        metrics["frames"] = len(img_files)
        metrics["seq_time_s"] = round(seq_time, 2)
        all_metrics.append(metrics)
        
        print(f"    MOTA={metrics['MOTA']:.4f} MOTP={metrics['MOTP']:.4f} "
              f"IDSW={metrics['IDSW']} FN={metrics['FN']} FP={metrics['FP']}")
        
        total_time += seq_time
        total_frames += len(img_files)
    
    # Aggregate metrics
    avg_mota = np.mean([m["MOTA"] for m in all_metrics]) if all_metrics else 0
    avg_motp = np.mean([m["MOTP"] for m in all_metrics]) if all_metrics else 0
    total_gt = sum(m["GT"] for m in all_metrics)
    total_fn = sum(m["FN"] for m in all_metrics)
    total_fp = sum(m["FP"] for m in all_metrics)
    total_idsw = sum(m["IDSW"] for m in all_metrics)
    avg_fps = total_frames / total_time if total_time > 0 else 0
    
    stats = {
        "model": f"MOT17 Tracking ({args.model} + {args.tracker})",
        "model_key": args.model,
        "tracker": args.tracker,
        "task": "tracking",
        "dataset": "MOT17",
        "sequences": len(all_metrics),
        "frames": total_frames,
        "MOTA": round(float(avg_mota), 4),
        "MOTP": round(float(avg_motp), 4),
        "FN": total_fn,
        "FP": total_fp,
        "IDSW": total_idsw,
        "total_gt": total_gt,
        "total_inference_time_s": round(total_time, 2),
        "avg_inference_ms": round(total_time / total_frames * 1000, 1) if total_frames > 0 else 0,
        "fps": round(avg_fps, 2),
    }
    
    save_stats(stats, f"{args.model}_{args.tracker}_tracking")
    
    print(f"\nTracking Results: {args.model}")
    print(f"  MOTA: {avg_mota:.4f}")
    print(f"  MOTP: {avg_motp:.4f}")
    print(f"  FPS: {avg_fps:.2f}")
    print(f"  Frames: {total_frames}")
    
    # Comparison if multiple models
    print_comparison({args.model: stats}, "TRACKING COMPARISON")


if __name__ == "__main__":
    main()
