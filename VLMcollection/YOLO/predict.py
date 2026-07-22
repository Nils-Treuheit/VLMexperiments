#!/usr/bin/env python3
"""Run YOLO26 inference on images/videos with a trained model."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO26 inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained .pt model or built-in name (yolo26m)")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image, video, directory, or 0 for webcam")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="Device (0, cpu, ...)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save results")
    parser.add_argument("--save-txt", action="store_true", default=False,
                        help="Save labels as .txt files")
    parser.add_argument("--project", type=str, default=None,
                        help="Output directory (default: runs/)")
    parser.add_argument("--name", type=str, default=None, help="Output subdirectory")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Maximum detections per image")
    parser.add_argument("--half", action="store_true", default=False,
                        help="Use FP16 half-precision")
    parser.add_argument("--stream", action="store_true", default=False,
                        help="Stream results (for video/webcam)")
    args = parser.parse_args()

    models_dir = Path(__file__).parent / "models"
    weight_path = Path(args.model)

    if not weight_path.exists():
        builtin = models_dir / f"{args.model}.pt"
        if builtin.exists():
            weight_path = builtin
        else:
            weight_path = args.model  # let ultralytics handle it

    print(f"Model:   {weight_path}")
    print(f"Source:  {args.source}")
    print(f"Device:  {args.device}")
    print(f"Conf:    {args.conf}  IoU: {args.iou}")
    print()

    model = YOLO(str(weight_path))

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        max_det=args.max_det,
        half=args.half,
        stream=args.stream,
    )

    if not args.stream:
        n_detections = sum(len(r.boxes) for r in results if r.boxes is not None)
        n_images = len(results)
        print(f"\nProcessed {n_images} images, {n_detections} detections total")


if __name__ == "__main__":
    main()
