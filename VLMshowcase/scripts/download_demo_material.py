#!/usr/bin/env python3
"""Download demo images and short video clips for the VLM showcase."""
import subprocess
import sys
import urllib.request
from pathlib import Path

DEMO_DIR = Path("/mnt/HDD1/Project_Data/demoMaterial")
PUBLIC_DIR = Path("/mnt/HDD1/Project_Data/public_datasets")
TMP = Path("/mnt/HDD1/tmp")

SCENES = [
    ("street_scene", [
        ("https://raw.githubusercontent.com/ultralytics/assets/main/bus.jpg", "bus_street.jpg"),
        ("https://raw.githubusercontent.com/ultralytics/assets/main/zidane.jpg", "soccer_players.jpg"),
    ]),
    ("indoor_scenes", [
        ("https://raw.githubusercontent.com/ultralytics/assets/main/pedestrian.jpg", "crosswalk.jpg"),
    ]),
    ("people_actions", [
        ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/person.jpg", "person_action.jpg"),
    ]),
    ("animals", [
        ("https://raw.githubusercontent.com/ultralytics/assets/main/dog.jpg", "dog_park.jpg"),
    ]),
]

VIDEOS = [
    ("https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi", "vtest.avi"),
    ("https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4", "walking_people.mp4"),
    ("https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4", "car_traffic.mp4"),
]


def download_file(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [SKIP] {dest.name} exists ({dest.stat().st_size // 1024} KB)")
        return
    print(f"  [DL]   {dest.name} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size = dest.stat().st_size
        print(f" {size // 1024} KB")
    except Exception as e:
        print(f" FAILED: {e}")
        if dest.exists():
            dest.unlink()


def main():
    print("=" * 60)
    print("  Downloading demo material for VLMshowcase")
    print("=" * 60)

    print("\n── Scene Images ──")
    for category, files in SCENES:
        cat_dir = DEMO_DIR / "images" / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [{category}]")
        for url, fname in files:
            download_file(url, cat_dir / fname)

    print("\n── Video Clips ──")
    videos_dir = DEMO_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for url, fname in VIDEOS:
        download_file(url, videos_dir / fname)

    print("\n" + "=" * 60)
    print("  Done! Demo material in:")
    print(f"    Images: {DEMO_DIR / 'images'}")
    print(f"    Videos: {DEMO_DIR / 'videos'}")
    print(f"    COCO:   {PUBLIC_DIR / 'coco' / 'val2017'}")
    print(f"    DOTA:   {PUBLIC_DIR / 'dotav1' / 'images'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
