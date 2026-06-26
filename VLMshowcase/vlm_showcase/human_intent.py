from .subprocess_utils import run_with_timer
from .config import MODELS

INTENT_PROMPTS = {
    "action": "Analyze the human(s) in this image. What are they doing? Describe their actions, body language, and any interaction with objects or other people. Be specific about what their pose suggests.",
    "intent": "Look at the person/people in this image. Based on their posture, gaze direction, hand position, and context, what do you think they intend to do next? Predict their next action.",
    "emotion": "Analyze the emotional state of the people in this image based on their facial expressions, body language, and the scene context. What emotions are being conveyed?",
    "social": "Describe the social dynamics in this image. What is the relationship between the people? Are they collaborating, competing, ignoring each other? What social cues support your analysis?",
}


def analyze_intent(image_path, aspect="action"):
    prompt = INTENT_PROMPTS.get(aspect, INTENT_PROMPTS["action"])
    cfg = MODELS["qwen3_thinking"]
    wrapper = cfg["script_wrapper"]
    result = run_with_timer(
        [cfg["venv_python"], wrapper, image_path, prompt, "--describe"],
        timeout=600, label="Analyzing intent with Qwen3-Thinking",
    )
    return result.stdout.strip() or result.stderr.strip()


def full_intent_analysis(image_path):
    results = {}
    for aspect in INTENT_PROMPTS:
        results[aspect] = analyze_intent(image_path, aspect)
    return results
