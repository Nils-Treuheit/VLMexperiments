#!/usr/bin/env python3
"""CLI wrapper for Qwen3-VL-Thinking. Called by the showcase via subprocess."""
import sys
import json
import os
from contextlib import redirect_stdout, redirect_stderr
import io

sys.path.insert(0, "/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/qwen3-vl_thinking")

_suppress = io.StringIO()
with redirect_stdout(_suppress), redirect_stderr(_suppress):
    from qwen_detector import QwenVLDetector


def main():
    if len(sys.argv) < 3:
        print("Usage: _thinking_wrapper.py <image_path> <prompt> [--describe|--detect]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "--describe"

    _suppress2 = io.StringIO()
    with redirect_stdout(_suppress2), redirect_stderr(_suppress2):
        det = QwenVLDetector(max_seq_length=2048)

    if mode == "--detect":
        result = det.detect(image_path, prompt)
    else:
        result = det.describe(image_path, prompt)

    if isinstance(result, (list, dict)):
        print(json.dumps(result))
    else:
        print(result)


if __name__ == "__main__":
    main()
