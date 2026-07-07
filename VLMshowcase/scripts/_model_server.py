#!/usr/bin/env python3
"""Persistent VLM model server. Loads model once, processes images via stdin/stdout."""
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

BASE = Path("/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection")
model_key = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "describe"
_suppress = StringIO()

def fail(msg):
    print(json.dumps({"error": msg}))
    sys.exit(1)

def ready():
    print(json.dumps({"event": "ready", "load_time_sec": round(time.time() - start, 2)}), flush=True)

start = time.time()

try:
    if model_key == "locate_anything":
        sys.path.insert(0, str(BASE / "locate_anything"))
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            import infer
            worker = infer.LocateAnythingWorker(str(BASE / "locate_anything" / "model"))

        def predict(img_path, prompt):
            from PIL import Image
            return worker.predict(Image.open(img_path).convert("RGB"), prompt)

        ready()

    elif model_key == "locate_anything_trt":
        sys.path.insert(0, str(BASE / "locate_anything"))
        _trt_venv = str(BASE / "locate_anything" / "model" / "tensorRT" / ".venv")
        _libs = [
            _trt_venv + "/lib/python3.10/site-packages/tensorrt_libs",
            f"{os.path.expanduser('~')}/.local/lib/python3.10/site-packages/nvidia/cudnn/lib",
            "/usr/local/cuda-12.8/lib64",
        ]
        import ctypes as _ct
        for _d in _libs:
            if os.path.isdir(_d):
                for _f in os.listdir(_d):
                    if _f.endswith(".so") or ".so." in _f:
                        try:
                            _ct.CDLL(os.path.join(_d, _f), mode=_ct.RTLD_GLOBAL)
                        except Exception:
                            pass
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            from infer_trt import LocateAnythingWorkerTRT
            worker = LocateAnythingWorkerTRT(str(BASE / "locate_anything" / "model"))

        def predict(img_path, prompt):
            from PIL import Image
            return worker.predict(Image.open(img_path).convert("RGB"), prompt)

        ready()

    elif model_key == "qwen3_instruct":
        sys.path.insert(0, str(BASE / "qwen3-vl_instruct"))
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            import infer_qwen3
            mp = infer_qwen3.resolve_model_path()
            model, processor = infer_qwen3.load_model(mp)

        def predict(img_path, prompt):
            from PIL import Image
            import torch
            img = Image.open(img_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=img, text=text, padding=True, return_tensors="pt")
            dev = next(model.parameters()).device
            inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
            return processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        ready()

    elif model_key == "qwen3_thinking":
        sys.path.insert(0, str(BASE / "qwen3-vl_thinking"))
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            from qwen_detector import QwenVLDetector
            worker = QwenVLDetector(max_seq_length=2048)

        def predict(img_path, prompt):
            if mode == "detect":
                return worker.detect(img_path, prompt)
            return worker.describe(img_path, prompt)

        ready()

    elif model_key == "florence2":
        import torch
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            from transformers import AutoModelForCausalLM, AutoProcessor
            model_id = "microsoft/Florence-2-large-ft"
            tdtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=tdtype, trust_remote_code=True,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            if model.generation_config.decoder_start_token_id is None:
                model.generation_config.decoder_start_token_id = model.config.text_config.decoder_start_token_id

        def predict(img_path, prompt):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            task = prompt.strip() if prompt else "<CAPTION>"
            dtype = next(model.parameters()).dtype
            inputs = processor(text=task, images=img, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device).to(dtype) if (hasattr(v, "dtype") and v.dtype.is_floating_point) else (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            import torch
            with torch.no_grad():
                gids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3)
            result = processor.batch_decode(gids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(result, task=task, image_size=(img.width, img.height))
            return str(parsed)

        ready()

    elif model_key == "paligemma":
        import torch
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
            model_id = "google/paligemma2-3b-mix-224"
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
            processor = AutoProcessor.from_pretrained(model_id)

        def predict(img_path, prompt):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            p = prompt if prompt else "caption en"
            inputs = processor(img, p, return_tensors="pt").to(model.device)
            import torch
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=100)
            return processor.decode(output[0], skip_special_tokens=True)

        ready()

    elif model_key == "cosmos_nemotron":
        import torch
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            model_id = "nvidia/Cosmos-Reason1-7B"
            model = AutoModelForMultimodalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
            processor = AutoProcessor.from_pretrained(model_id)

        def predict(img_path, prompt):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            p = prompt or "Describe this scene. What physical interactions do you observe?"
            msgs = [
                {"role": "system", "content": "You are a helpful assistant with physical reasoning."},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p}]},
            ]
            inputs = processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            ).to(model.device)
            import torch
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256)
            return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        ready()

    elif model_key == "phi_vision":
        import torch
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
            model_id = "microsoft/Phi-3.5-vision-instruct"
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            config._attn_implementation = "eager"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, config=config, trust_remote_code=True, torch_dtype="auto", device_map="auto")
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model.generation_config.use_cache = False

        def predict(img_path, prompt):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            p = prompt or "What is shown in this image? Describe it in detail."
            full = f"<|user|>\n<|image_1|>\n{p}<|end|>\n<|assistant|>\n"
            inputs = processor(full, img, return_tensors="pt").to(model.device)
            import torch
            with torch.no_grad():
                gids = model.generate(**inputs, max_new_tokens=500, use_cache=False)
            return processor.decode(gids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        ready()

    elif model_key == "llama_vision":
        import torch
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            model = MllamaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto")
            processor = AutoProcessor.from_pretrained(model_id)

        def predict(img_path, prompt):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            p = prompt or "Describe this image in detail."
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            full = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inputs = processor(full, img, return_tensors="pt").to(model.device)
            import torch
            with torch.no_grad():
                output = model.generate(**inputs, temperature=0.7, top_p=0.9, max_new_tokens=256)
            return processor.decode(output[0], skip_special_tokens=True)

        ready()

    elif model_key == "phi4_multimodal":
        import torch
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            from transformers import AutoModelForCausalLM, AutoProcessor
            model_id = "microsoft/Phi-4-multimodal-instruct"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        def predict(img_path, prompt):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            p = prompt or "Describe this image in detail."
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p}]}]
            inputs = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            import torch
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=256)
            return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        ready()

    elif model_key == "diffusion_gemma_vl":
        with redirect_stdout(_suppress), redirect_stderr(_suppress):
            import sys as _sys
            _sys.path.insert(0, str(BASE / "diffusion_gemma_vl"))
            from run import run_encoder_yolo, YOLO_MODELS, YOLO
            if YOLO is None:
                fail("YOLO (ultralytics) not installed in diffusion_gemma_vl venv")

        def predict(img_path, prompt):
            return run_encoder_yolo(img_path, yolo_tasks=["aabb"])

        ready()

    else:
        fail(f"Unknown model key: {model_key}")

except Exception as e:
    tb = traceback.format_exc()[-800:]
    fail(f"{e}  [traceback: {tb}]")

for line in sys.stdin:
    line = line.strip()
    if not line:
        break
    req = json.loads(line)
    img_path = req["image"]
    prompt = req.get("prompt", "")
    t0 = time.time()
    try:
        result = predict(img_path, prompt)
        inf_time = time.time() - t0
        print(json.dumps({"result": result, "time_sec": round(inf_time, 2)}), flush=True)
    except Exception as e:
        inf_time = time.time() - t0
        tb = traceback.format_exc()[-500:]
        print(json.dumps({"error": str(e), "time_sec": round(inf_time, 2), "traceback": tb}), flush=True)
