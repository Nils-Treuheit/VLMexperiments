import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "microsoft/Phi-3.5-vision-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<|user|>\n<|image_1|>\nWhat is shown in this image? Describe it in detail.<|end|>\n<|assistant|>\n"
inputs = processor(prompt, image, return_tensors="pt").to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=500)
response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)
