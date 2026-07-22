#!/usr/bin/env python3
"""Convert DOTA annotations to YOLO format (horizontal or oriented)."""

import argparse
from pathlib import Path

DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]

CLASS_TO_ID = {name: i for i, name in enumerate(DOTA_CLASSES)}


def dota_to_yolo_obb(dota_path, img_w, img_h):
    """Convert a DOTA-format label line to YOLO OBB format."""
    lines = dota_path.read_text().strip().splitlines()
    yolo_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
        cls_name = parts[8]
        if cls_name not in CLASS_TO_ID:
            continue
        cls_id = CLASS_TO_ID[cls_name]
        # Normalize coordinates
        coords = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h,
                  x3 / img_w, y3 / img_h, x4 / img_w, y4 / img_h]
        yolo_lines.append(f"{cls_id} " + " ".join(f"{c:.6f}" for c in coords))
    return yolo_lines


def dota_to_yolo_hbb(dota_path, img_w, img_h):
    """Convert DOTA to YOLO horizontal bbox format."""
    lines = dota_path.read_text().strip().splitlines()
    yolo_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        xs = list(map(float, parts[0:8:2]))
        ys = list(map(float, parts[1:8:2]))
        cls_name = parts[8]
        if cls_name not in CLASS_TO_ID:
            continue
        cls_id = CLASS_TO_ID[cls_name]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2 / img_w
        cy = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return yolo_lines


def main():
    parser = argparse.ArgumentParser(description="Convert DOTA labels to YOLO format")
    parser.add_argument("--dota-dir", type=str,
                        default="/mnt/HDD1/Project_Data/public_datasets/dotav1",
                        help="DOTAv1 root directory")
    parser.add_argument("--img-dir", type=str, default="images",
                        help="Image dir relative to dota-dir")
    parser.add_argument("--label-dir", type=str, default="labels",
                        help="DOTA label dir relative to dota-dir")
    parser.add_argument("--output-dir", type=str, default="labels_yolo",
                        help="Output dir relative to dota-dir")
    parser.add_argument("--obb", action="store_true",
                        help="Output OBB format (default: horizontal bbox)")
    parser.add_argument("--img-size", type=int, nargs=2, default=None,
                        help="Force image W H (skip reading image files)")
    args = parser.parse_args()

    dota_dir = Path(args.dota_dir)
    img_dir = dota_dir / args.img_dir
    lbl_dir = dota_dir / args.label_dir
    out_dir = dota_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    convert_fn = dota_to_yolo_obb if args.obb else dota_to_yolo_hbb
    fmt_name = "OBB" if args.obb else "horizontal bbox"

    lbl_files = sorted(lbl_dir.glob("*.txt"))
    print(f"Converting {len(lbl_files)} labels to YOLO {fmt_name} format...")

    import cv2

    for lbl_path in lbl_files:
        img_path = img_dir / f"{lbl_path.stem}.png"
        if args.img_size:
            img_w, img_h = args.img_size
        elif img_path.exists():
            h, w = cv2.imread(str(img_path)).shape[:2]
            img_w, img_h = w, h
        else:
            print(f"  Skipping {lbl_path.stem}: image not found")
            continue

        yolo_lines = convert_fn(lbl_path, img_w, img_h)
        if yolo_lines:
            out_path = out_dir / f"{lbl_path.stem}.txt"
            out_path.write_text("\n".join(yolo_lines) + "\n")

    print(f"Done. YOLO-format labels in {out_dir}")


if __name__ == "__main__":
    main()
