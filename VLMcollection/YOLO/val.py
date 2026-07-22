#!/usr/bin/env python3
"""Validate a trained YOLO26 model on a test/val dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO26 validation")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained .pt model")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="NMS IoU threshold for evaluation")
    parser.add_argument("--project", type=str, default=None,
                        help="Output directory (default: runs/)")
    parser.add_argument("--name", type=str, default=None,
                        help="Output subdirectory")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16")
    parser.add_argument("--save-json", action="store_true",
                        help="Save COCO JSON results")
    parser.add_argument("--plots", action="store_true", default=True,
                        help="Generate plots")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Data:  {args.data}")
    print(f"Device: {args.device}")
    print()

    model = YOLO(args.model)

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        half=args.half,
        save_json=args.save_json,
        plots=args.plots,
    )

    print(f"\nmAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")


if __name__ == "__main__":
    main()
