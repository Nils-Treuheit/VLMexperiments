import torch
import requests
import warnings
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

warnings.filterwarnings("ignore")
model_id = "microsoft/Phi-3.5-vision-instruct"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config._attn_implementation = "eager"

model = AutoModelForCausalLM.from_pretrained(
    model_id, config=config, trust_remote_code=True,
    torch_dtype="auto", device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

prompt = "<|user|>\n<|image_1|>\nWhat is shown in this image? Describe it in detail.<|end|>\n<|assistant|>\n"
inputs = processor(prompt, image, return_tensors="pt").to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=500, use_cache=False)
response = processor.tokenizer.decode(generate_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
