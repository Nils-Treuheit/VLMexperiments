#!/usr/bin/env python3
"""Train a YOLO26 model on a custom detection dataset."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 on custom dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo26m",
                        choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"],
                        help="Model size (default: yolo26m)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device (e.g., '0' for GPU 0, 'cpu', '0,1' for multi-GPU)")
    parser.add_argument("--workers", type=int, default=8, help="Data loading workers")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--project", type=str, default=None,
                        help="Output directory (default: runs/)")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Start from pretrained weights")
    parser.add_argument("--freeze", type=int, default=0,
                        help="Freeze first N backbone layers (0 = no freeze)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (0 = disable)")
    args = parser.parse_args()

    models_dir = Path(__file__).parent / "models"

    if args.pretrained:
        weight_path = models_dir / f"{args.model}.pt"
        if not weight_path.exists():
            weight_path = args.model  # fallback to ultralytics built-in
    else:
        weight_path = None  # train from scratch

    print(f"Using weights: {weight_path or 'from scratch'}")
    print(f"Data config:   {args.data}")
    print(f"Device:        {args.device}")
    print(f"Batch size:    {args.batch}")
    print(f"Image size:    {args.imgsz}")
    print(f"Epochs:        {args.epochs}")
    print()

    model = YOLO(weight_path) if weight_path else YOLO(args.model + ".yaml")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        lr0=args.lr,
        resume=args.resume,
        project=args.project,
        name=args.name,
        freeze=args.freeze,
        patience=args.patience,
        amp=True,
    )

    print(f"\nTraining complete. Best model saved in {results.save_dir}")


if __name__ == "__main__":
    main()
