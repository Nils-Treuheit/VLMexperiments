#!/usr/bin/env python3
"""Persistent VLM model server. Loads model once, processes images via stdin/stdout.

Usage: <venv_python> _model_server.py <model_key> [--mode describe|detect|ground]

stdin: one JSON line per request  →  {"image": "/path.jpg", "prompt": "..."}
stdout: one JSON line per response →  {"result": ..., "time_sec": 0.0}
Empty line on stdin → graceful shutdown.

Model key: locate_anything | qwen3_instruct | qwen3_thinking
"""
import json
import os
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

BASE = Path("/mnt/HDD1/Project_Code/vlm_det_test")

model_key = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "describe"

_suppress = StringIO()

start = time.time()

if model_key == "locate_anything":
    sys.path.insert(0, str(BASE / "locate_anything"))
    with redirect_stdout(_suppress), redirect_stderr(_suppress):
        import infer
        worker = infer.LocateAnythingWorker(str(BASE / "locate_anything" / "model"))

    def predict(img_path, prompt):
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        text = worker.predict(img, prompt)
        return text

elif model_key == "qwen3_instruct":
    sys.path.insert(0, str(BASE / "qwen3-vl_instruct"))
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    with redirect_stdout(_suppress), redirect_stderr(_suppress):
        import infer_qwen3
        mp = infer_qwen3.resolve_model_path()
        model, processor = infer_qwen3.load_model(mp)

    def predict(img_path, prompt):
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(images=img, text=text, padding=True, return_tensors="pt")
        import torch
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        output = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return output

elif model_key == "qwen3_thinking":
    sys.path.insert(0, str(BASE / "qwen3-vl_thinking"))
    with redirect_stdout(_suppress), redirect_stderr(_suppress):
        from qwen_detector import QwenVLDetector
        worker = QwenVLDetector(max_seq_length=2048)

    def predict(img_path, prompt):
        if mode == "detect":
            return worker.detect(img_path, prompt)
        return worker.describe(img_path, prompt)

else:
    print(json.dumps({"error": f"Unknown model: {model_key}"}))
    sys.exit(1)

load_time = time.time() - start
print(json.dumps({"event": "ready", "load_time_sec": round(load_time, 2)}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        break
    req = json.loads(line)
    img_path = req["image"]
    prompt = req.get("prompt", "")
    t0 = time.time()
    result = predict(img_path, prompt)
    inf_time = time.time() - t0
    print(json.dumps({"result": result, "time_sec": round(inf_time, 2)}), flush=True)
