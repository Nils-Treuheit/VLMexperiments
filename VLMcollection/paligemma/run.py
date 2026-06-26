import torch
import requests
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/paligemma2-3b-mix-224"

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "What is in this image?"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50)
print(f"Prompt: {prompt}")
print(f"Answer: {processor.decode(output[0], skip_special_tokens=True)}")

prompt = "caption en"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50)
print(f"Prompt: {prompt}")
print(f"Caption: {processor.decode(output[0], skip_special_tokens=True)}")
