from unsloth import FastModel
from PIL import Image
import json
import re
import torch
import os

MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"

class QwenVLDetector:
    def __init__(self, max_seq_length=2048):
        self.model, self.processor = FastModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        self.model.eval()

    def _parse_json(self, text):
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None

    def detect(self, image, prompt="Detect all objects in this image. Return bounding boxes as JSON with labels."):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        parsed = self._parse_json(text)
        return parsed if parsed else text

    def describe(self, image, prompt="Describe this image in detail."):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=512)
        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def detect_from_path(self, image_path, prompt=None):
        if prompt is None:
            prompt = (
                'Detect all objects in this image. '
                'Return ONLY a JSON array of objects with "bbox_2d" '
                '([x_min, y_min, x_max, y_max] in 0-1000 range) and "label". '
                'Example: [{"bbox_2d": [100, 200, 300, 400], "label": "person"}]'
            )
        return self.detect(image_path, prompt)


if __name__ == "__main__":
    import sys
    import requests
    d = QwenVLDetector()
    print("Model loaded. Ready.", flush=True)

    if len(sys.argv) > 1:
        urls = sys.argv[1:]
    else:
        urls = ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"]

    for url in urls:
        img = Image.open(requests.get(url, stream=True).raw)
        print(f"\n=== {url} ===")
        print("Detection:", d.detect(img))
        print()
