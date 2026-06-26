"""Example: process images from a live folder / named pipe / video frames.
Import qwen_detector, load once, call many times.
"""

import sys
import os
import time
import json
from glob import glob
from qwen_detector import QwenVLDetector

# Load model once (this is the only slow part)
detector = QwenVLDetector()

# Example 1: watch a folder for new images
def watch_folder(folder, prompt="Detect all objects. Return JSON with bbox_2d and label."):
    seen = set()
    print(f"Watching {folder} for new images...")
    while True:
        for path in glob(os.path.join(folder, "*.[jp][pn]g")):
            if path not in seen:
                seen.add(path)
                result = detector.detect(path, prompt)
                print(f"{os.path.basename(path)}: {json.dumps(result, indent=2)}")
        time.sleep(0.1)

# Example 2: read from a named pipe (for piping video frame paths)
def watch_pipe(pipe_path, prompt="Detect all objects. Return JSON with bbox_2d and label."):
    print(f"Listening on pipe {pipe_path}...")
    with open(pipe_path, "r") as f:
        while True:
            line = f.readline().strip()
            if line:
                result = detector.detect(line, prompt)
                print(f"{line}: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pipe":
        watch_pipe(sys.argv[2] if len(sys.argv) > 2 else "/tmp/qwen_pipe")
    else:
        watch_folder(sys.argv[1] if len(sys.argv) > 1 else ".")
