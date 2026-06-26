"""Lightweight FastAPI server for Qwen3-VL detection.
Keeps model loaded in memory, handles parallel requests.

Usage:
    .venv/bin/python server.py              # start on :8000
    .venv/bin/python server.py --port 8080  # custom port

Request:
    curl -X POST http://localhost:8000/detect \
      -H "Content-Type: application/json" \
      -d '{"image_url": "https://...", "prompt": "Detect cats"}'

    curl -X POST http://localhost:8000/detect \
      -H "Content-Type: application/json" \
      -d '{"image_path": "/path/to/img.jpg"}'

    # base64-encoded image
    curl -X POST http://localhost:8000/detect \
      -H "Content-Type: application/json" \
      -d '{"image_base64": "<base64>", "prompt": "..."}'
"""

import sys
import base64
import json
import argparse
from io import BytesIO
from PIL import Image
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from qwen_detector import QwenVLDetector

app = FastAPI()
detector = None

class DetectRequest(BaseModel):
    image_url: str | None = None
    image_path: str | None = None
    image_base64: str | None = None
    prompt: str | None = None

class DetectResponse(BaseModel):
    result: str | list | None = None
    error: str | None = None

@app.on_event("startup")
def load_model():
    global detector
    detector = QwenVLDetector()
    print("Model loaded. Ready.", flush=True)

@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    try:
        if req.image_url:
            img = Image.open(requests.get(req.image_url, stream=True).raw)
        elif req.image_path:
            img = Image.open(req.image_path).convert("RGB")
        elif req.image_base64:
            data = base64.b64decode(req.image_base64)
            img = Image.open(BytesIO(data)).convert("RGB")
        else:
            return DetectResponse(error="Provide image_url, image_path, or image_base64")

        result = detector.detect(img, req.prompt or "Detect all objects. Return JSON with bbox_2d and label.")
        return DetectResponse(result=result)
    except Exception as e:
        return DetectResponse(error=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
