import json
from .subprocess_utils import run_with_timer
from .config import MODELS


def describe_with_qwen3_instruct(image_path, prompt="Describe this image in detail."):
    cfg = MODELS["qwen3_instruct"]
    result = run_with_timer(
        [cfg["venv_python"], cfg["script"], image_path, prompt, "--json"],
        timeout=300, label="Loading Qwen3-VL-Instruct",
    )
    try:
        data = json.loads(result.stdout.strip())
        return data.get("text", result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return result.stdout.strip() or result.stderr


def describe_with_qwen3_thinking(image_path, prompt="Describe this image in detail with reasoning."):
    cfg = MODELS["qwen3_thinking"]
    wrapper = cfg["script_wrapper"]
    result = run_with_timer(
        [cfg["venv_python"], wrapper, image_path, prompt, "--describe"],
        timeout=300, label="Loading Qwen3-VL-Thinking",
    )
    return result.stdout.strip() or result.stderr.strip()


def detect_with_qwen3_instruct(image_path, query="Detect all objects. Output each as <box>x1,y1,x2,y2</box>."):
    cfg = MODELS["qwen3_instruct"]
    result = run_with_timer(
        [cfg["venv_python"], cfg["script"], image_path, query, "--json"],
        timeout=300, label="Loading Qwen3-VL-Instruct",
    )
    try:
        return json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return {"text": result.stdout.strip()}


def analyze_scene(image_path):
    return {
        "qwen3_instruct_description": describe_with_qwen3_instruct(image_path),
        "qwen3_thinking_description": describe_with_qwen3_thinking(image_path),
        "qwen3_instruct_detection": detect_with_qwen3_instruct(
            image_path, "List ALL objects. Be exhaustive. Output each as <box>x1,y1,x2,y2</box>."
        ),
    }
