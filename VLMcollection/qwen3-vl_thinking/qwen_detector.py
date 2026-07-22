import os
import re
import json
import torch
from PIL import Image

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


class QwenVLDetector:
    def __init__(self, model_path=None, enable_thinking=True):
        from transformers import AutoProcessor
        from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
        if model_path is None:
            base = os.path.join(os.path.dirname(__file__), "..", "qwen3-vl_instruct")
            model_path = os.path.join(base, "model_vl")
            if not os.path.isdir(model_path):
                model_path = os.path.join(base, "model")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, device_map="cuda", torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True,
        )
        self.model.eval()
        self.enable_thinking = enable_thinking

    def _generate(self, image, prompt, max_new_tokens=512,
                  temperature=0.1, enable_thinking=None):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if enable_thinking is None:
            enable_thinking = self.enable_thinking
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        inputs = self.processor(text=chat, images=image, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        with torch.no_grad():
            gen = self.model.generate(**inputs, **gen_kwargs)
        text = self.processor.decode(
            gen[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return text

    def detect(self, image, prompt=None):
        if prompt is None:
            prompt = (
                "Analyze this image. Identify every visible object. "
                "For each object, output a bounding box and label.\n"
                "Return ONLY a JSON array:\n"
                '[{"bbox_2d": [x1,y1,x2,y2], "label": "name", "confidence": 0.95}]\n'
                "Coordinates are in pixels. Use [] if no objects found."
            )
        text = self._generate(image, prompt, max_new_tokens=1024,
                              temperature=0.1, enable_thinking=True)
        parsed = self._parse_json(text)
        return parsed if parsed else text

    def classify(self, image, categories):
        prompt = (
            f"Look at this image and classify it into exactly one of these categories:\n"
            f"{', '.join(categories)}\n"
            f"Reply with ONLY the category name, nothing else."
        )
        text = self._generate(image, prompt, max_new_tokens=32,
                              temperature=0.0, enable_thinking=False)
        return text.strip()

    def describe(self, image, prompt="Describe this image in detail."):
        return self._generate(image, prompt, max_new_tokens=512,
                              temperature=0.1, enable_thinking=False)

    def _parse_json(self, text):
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None
