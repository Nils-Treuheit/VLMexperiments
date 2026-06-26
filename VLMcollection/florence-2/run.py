import torch
import requests
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "microsoft/Florence-2-large-ft"

model = Florence2ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch_dtype, trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

task_prompt = "<CAPTION>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
)
result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed = processor.post_process_generation(result, task=task_prompt, image_size=(image.width, image.height))
print(f"<CAPTION>: {parsed}")

task_prompt = "<DETAILED_CAPTION>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
)
result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed = processor.post_process_generation(result, task=task_prompt, image_size=(image.width, image.height))
print(f"<DETAILED_CAPTION>: {parsed}")

task_prompt = "<OD>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
)
result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed = processor.post_process_generation(result, task=task_prompt, image_size=(image.width, image.height))
print(f"<OD>: {parsed}")
