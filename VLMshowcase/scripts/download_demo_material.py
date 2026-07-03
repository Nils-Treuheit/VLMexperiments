#!/usr/bin/env python3
"""Download / organize demo images and videos for VLMshowcase.

- Organizes COCO images into thematic subfolders (10 per folder)
- Downloads engaging multi-object video clips
- No images placed directly in the root images/ folder
"""
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

DEMO_DIR = Path("/mnt/HDD1/Project_Data/demoMaterial")
COCO_DIR = Path("/mnt/HDD1/Project_Data/public_datasets/coco")
TMP = Path("/mnt/HDD1/tmp")

IMAGES_DIR = DEMO_DIR / "images"
VIDEOS_DIR = DEMO_DIR / "videos"

COCO_IMG = COCO_DIR / "val2017"
COCO_ANN = COCO_DIR / "annotations" / "instances_val2017.json"

# Category -> list of COCO category names to include
FOLDER_CATEGORIES = {
    "animals": ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "indoor_scenes": ["chair", "couch", "bed", "dining table", "tv", "laptop", "book", "clock", "vase", "bottle"],
    "people_actions": ["person"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket"],
    "street_scene": ["car", "motorcycle", "bus", "truck", "traffic light", "stop sign",
                     "parking meter", "bench", "person"],
}

# Additional video URLs with multi-object scenes
VIDEOS = [
    ("https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi", "vtest.avi"),
    ("https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4",
     "walking_people.mp4"),
    ("https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
     "car_traffic.mp4"),
    ("https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
     "people_crossing.mp4"),
    ("https://github.com/intel-iot-devkit/sample-videos/raw/master/street-detection.mp4",
     "busy_street.mp4"),
    ("https://github.com/intel-iot-devkit/sample-videos/raw/master/object-detection.mp4",
     "mixed_traffic.mp4"),
]


def download_file(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        return
    print(f"  [DL] {dest.name} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size = dest.stat().st_size
        print(f" {size // 1024} KB")
    except Exception as e:
        print(f" FAILED: {e}")
        if dest.exists():
            dest.unlink()


def select_coco_images():
    """Select 10 multi-object COCO images per folder category."""
    if not COCO_ANN.exists():
        print("  COCO annotations not found, skipping image selection")
        return {}

    print("  Reading COCO annotations...")
    with open(COCO_ANN) as f:
        ann_data = json.load(f)

    cat_name_to_id = {c["name"]: c["id"] for c in ann_data["categories"]}
    img_id_to_file = {i["id"]: i["file_name"] for i in ann_data["images"]}

    # Build: category_id -> list of image_ids
    cat_to_images = {}
    for ann in ann_data["annotations"]:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        cat_to_images.setdefault(cat_id, set()).add(img_id)

    # For each folder, find images matching its categories
    folder_images = {}
    used_ids = set()

    for folder_name, cat_names in FOLDER_CATEGORIES.items():
        matched = []
        cat_ids = []
        for name in cat_names:
            cid = cat_name_to_id.get(name)
            if cid:
                cat_ids.append(cid)

        # Find images that contain objects from any of these categories
        candidate_ids = set()
        for cid in cat_ids:
            candidate_ids.update(cat_to_images.get(cid, set()))

        # Score by total annotation count (prefer multi-object images)
        img_ann_count = {}
        for ann in ann_data["annotations"]:
            iid = ann["image_id"]
            if iid in candidate_ids and iid not in used_ids:
                img_ann_count[iid] = img_ann_count.get(iid, 0) + 1

        # Pick top 10 by annotation count, preferring multi-object
        sorted_ids = sorted(img_ann_count, key=img_ann_count.get, reverse=True)
        picked = sorted_ids[:10]

        if len(picked) < 10:
            # Fall back to used images if needed
            remaining = [i for i in candidate_ids if i not in used_ids and i not in picked]
            picked.extend(remaining[:10 - len(picked)])

        folder_images[folder_name] = picked[:10]
        used_ids.update(picked)

        print(f"    {folder_name}: {len(picked)} images selected "
              f"({sum(1 for i in picked if i in candidate_ids)} matching cats)")

    return folder_images, img_id_to_file


def main():
    print("=" * 60)
    print("  VLMshowcase Demo Material Setup")
    print("=" * 60)

    # --- Clean up any stray files in images/ root ---
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for f in IMAGES_DIR.iterdir():
        if f.is_file():
            print(f"  [CLEAN] Removing stray file: {f.name}")
            f.unlink()

    # --- Select and copy COCO images to subfolders ---
    print("\n\u2500\u2500 COCO Image Selection \u2500\u2500")
    selection, img_id_to_file = select_coco_images()

    if selection:
        print("\n  Copying images to subfolders...")
        for folder_name, img_ids in selection.items():
            folder = IMAGES_DIR / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            count = 0
            for img_id in img_ids:
                src_name = img_id_to_file.get(img_id)
                if not src_name:
                    continue
                src = COCO_IMG / src_name
                if not src.exists():
                    continue
                dst = folder / f"coco_{src_name}"
                shutil.copy2(src, dst)
                count += 1
            print(f"    {folder_name}: {count} images")

    # --- Copy existing random_sample_set images into appropriate folders ---
    random_dir = IMAGES_DIR / "random_sample_set"
    if random_dir.exists():
        print("\n  Distributing random_sample_set images...")
        for f in sorted(random_dir.iterdir()):
            if f.is_file():
                # Guess category from filename
                name_lower = f.name.lower()
                if any(k in name_lower for k in ["dog", "cat", "bird", "horse", "cow"]):
                    target = "animals"
                elif any(k in name_lower for k in ["sport", "ball", "frisbee", "kite"]):
                    target = "sports"
                elif any(k in name_lower for k in ["person", "people", "action", "walk", "run"]):
                    target = "people_actions"
                elif any(k in name_lower for k in ["car", "bus", "truck", "street", "traffic"]):
                    target = "street_scene"
                elif any(k in name_lower for k in ["indoor", "room", "kitchen", "office"]):
                    target = "indoor_scenes"
                else:
                    # Check folder name in path
                    parent_name = f.parent.name
                    if parent_name in FOLDER_CATEGORIES:
                        target = parent_name
                    else:
                        target = "people_actions"  # default
                dst = IMAGES_DIR / target / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    print(f"    {f.name} -> {target}/")
        # Remove the old random_sample_set directory
        shutil.rmtree(random_dir)
        print("    Removed random_sample_set/")

    # --- Ensure every subfolder has at least 10 images ---
    print("\n  Filling missing images (any COCO source)...")
    for folder_name in FOLDER_CATEGORIES:
        folder = IMAGES_DIR / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        existing = len(list(folder.glob("*")))
        if existing < 10:
            needed = 10 - existing
            print(f"    {folder_name}: {existing} exist, need {needed} more")
            # Pick random COCO images not yet used
            all_coco = sorted(COCO_IMG.glob("*.jpg"))
            used_names = set(f.name for f in folder.glob("*"))
            added = 0
            for src in all_coco:
                if added >= needed:
                    break
                dst_name = f"fill_{src.name}"
                if dst_name not in used_names:
                    dst = folder / dst_name
                    shutil.copy2(src, dst)
                    added += 1
            print(f"      Added {added} images")

    # --- Download videos ---
    print("\n\u2500\u2500 Video Downloads \u2500\u2500")
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    for url, fname in VIDEOS:
        download_file(url, VIDEOS_DIR / fname)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Demo Material Summary")
    print("=" * 60)
    print(f"\n  Images ({IMAGES_DIR}):")
    for d in sorted(IMAGES_DIR.iterdir()):
        if d.is_dir():
            count = len(list(d.glob("*")))
            print(f"    {d.name:22s} {count} images")
    print(f"\n  Videos ({VIDEOS_DIR}):")
    for f in sorted(VIDEOS_DIR.iterdir()):
        size = f.stat().st_size // (1024 * 1024) if f.stat().st_size > 0 else 0
        print(f"    {f.name:35s} {size} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
