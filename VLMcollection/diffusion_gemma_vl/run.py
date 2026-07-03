"""
DiffusionGemma VLM – pluggable vision encoder + llama-diffusion-cli (text-only diffusion).

Architecture:
  Image -> vision encoder (YOLO | SigLIP2 | MoonViT) -> structured text description
       -> llama-diffusion-cli (multimodal by proxy: text describes the scene)
       -> final answer

Encoders:
  yolo      YOLO11 detection/pose/OBB -> object counts -> text
  siglip2   SigLIP2 zero-shot classification -> top labels -> text
  moonvit   MoonViT + SigLIP2 text encoder -> top labels -> text

Usage:
  python run.py --image path/to/img.jpg --task caption --encoder yolo
  python run.py --image path/to/img.jpg --task caption --encoder siglip2
  python run.py --image path/to/img.jpg --task caption --encoder moonvit
  python run.py --image path/to/img.jpg --task vqa --prompt "Any people?" --encoder siglip2
  python run.py --image path/to/img.jpg --task detect
  python run.py --image path/to/img.jpg --task caption --encoder yolo --yolo-tasks aabb,pose
  python run.py --image path/to/img.jpg --task caption --encoder yolo --yolo-tasks aabb,pose,obb
  python run.py --task chat --prompt "What is 2+2?"
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent  # VLMcollection root
LLAMA_CLI_BUILD = PROJECT_DIR / "llama.cpp" / "build2" / "bin" / "llama-diffusion-cli"
LLAMA_CLI = LLAMA_CLI_BUILD if LLAMA_CLI_BUILD.exists() else (
    PROJECT_DIR / "llama.cpp" / "build" / "bin" / "llama-diffusion-cli"
)
# CUDA runtime libs from venv pip packages
CUDA_LIB_DIR = str(PROJECT_DIR / ".venv" / "lib" / "python3.13" / "site-packages" / "nvidia" / "cu13" / "lib")
GGUF_PATH = (
    Path(os.environ["GGUF_PATH"])
    if "GGUF_PATH" in os.environ
    else Path(
        "/mnt/HDD1/unsloth_and_hugging_face_models"
        "/huggingface/hub/diffusiongemma_local"
        "/diffusiongemma-26B-A4B-it-Q8_0.gguf"
    )
)

SIGLIP2_RUN = ROOT_DIR / "siglip2" / "run.py"
MOONVIT_RUN = ROOT_DIR / "moonvit" / "run.py"

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


# ---------------------------------------------------------------------------
# YOLO feeders
# ---------------------------------------------------------------------------

def describe_detection(results) -> str:
    objects = results[0].boxes
    if objects is None or len(objects) == 0:
        return "No objects detected in the image."
    counts: dict[str, int] = {}
    for cls_id in objects.cls.int().tolist():
        name = COCO_NAMES.get(int(cls_id), f"object_{int(cls_id)}")
        counts[name] = counts.get(name, 0) + 1
    items = sorted(counts.items(), key=lambda x: -x[1])
    parts = [f"{n} (x{c})" if c > 1 else n for n, c in items]
    return "Objects detected: " + ", ".join(parts) + "."


def describe_pose(results) -> str:
    keypoints = results[0].keypoints
    if keypoints is None or len(keypoints) == 0:
        return "No poses detected."
    n_people = len(keypoints)
    lines = [f"{n_people} person(s) detected with pose keypoints."]
    for i, kp in enumerate(keypoints):
        visible = kp.data[0][kp.data[0][:, 2] > 0.5]
        n_visible = len(visible)
        lines.append(f"  Person {i+1}: {n_visible} visible keypoints")
    return "\n".join(lines)


def describe_obb(results) -> str:
    objs = results[0].obb
    if objs is None or len(objs) == 0:
        return "No oriented objects detected."
    lines = [f"{len(objs)} oriented object(s) detected:"]
    for i, obj in enumerate(objs):
        cls_id = int(obj.cls[0])
        conf = float(obj.conf[0])
        name = COCO_NAMES.get(cls_id, f"object_{cls_id}")
        xywhr = obj.xywhr[0].tolist()
        cx, cy, w, h, r = xywhr
        lines.append(
            f"  {i+1}. {name} (conf={conf:.2f}) "
            f"center=({cx:.1f},{cy:.1f}) size={w:.1f}x{h:.1f} rotation={r:.2f}rad"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# encoder dispatch
# ---------------------------------------------------------------------------

YOLO_MODELS = {
    "aabb": "yolo11n.pt",
    "pose": "yolo11n-pose.pt",
    "obb": "yolo11n-obb.pt",
}


def run_encoder_yolo(image_path: str, yolo_tasks: list[str] | None = None) -> str:
    if YOLO is None:
        return "YOLO (ultralytics) not installed."
    if yolo_tasks is None:
        yolo_tasks = ["aabb"]
    descriptions = []
    # Use CPU for YOLO to avoid GPU memory conflict with the LLM engine
    for task in yolo_tasks:
        model = YOLO(YOLO_MODELS[task])
        results = model(image_path, verbose=False, device='cpu')
        if task == "aabb":
            descriptions.append(describe_detection(results))
        elif task == "pose":
            descriptions.append(describe_pose(results))
        elif task == "obb":
            descriptions.append(describe_obb(results))
    return "\n".join(descriptions)


def run_encoder_siglip2(image_path: str, top_k: int = 8) -> str:
    if not SIGLIP2_RUN.exists():
        return f"SigLIP2 runner not found at {SIGLIP2_RUN}"
    result = subprocess.run(
        [sys.executable, str(SIGLIP2_RUN), "--image", str(image_path),
         "--task", "describe", "--top-k", str(top_k)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return f"SigLIP2 error: {result.stderr.strip()}"
    try:
        data = json.loads(result.stdout)
        return data.get("description_text", "")
    except json.JSONDecodeError:
        return f"SigLIP2 parse error: {result.stdout[:500]}"


def run_encoder_moonvit(image_path: str, top_k: int = 8) -> str:
    if not MOONVIT_RUN.exists():
        return f"MoonViT runner not found at {MOONVIT_RUN}"
    result = subprocess.run(
        [sys.executable, str(MOONVIT_RUN), "--image", str(image_path),
         "--task", "describe", "--top-k", str(top_k)],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        return f"MoonViT error: {result.stderr.strip()}"
    try:
        data = json.loads(result.stdout)
        return data.get("description_text", "")
    except json.JSONDecodeError:
        return f"MoonViT parse error: {result.stdout[:500]}"


def build_encoder(name: str, yolo_tasks: list[str] | None = None):
    if name == "yolo":
        return lambda img: run_encoder_yolo(img, yolo_tasks=yolo_tasks)
    if name == "siglip2":
        return run_encoder_siglip2
    if name == "moonvit":
        return run_encoder_moonvit
    raise ValueError(f"Unknown encoder: {name}")


# ---------------------------------------------------------------------------
# prompt building
# ---------------------------------------------------------------------------

def build_chat_prompt(user_text: str) -> str:
    bos = "<bos>"
    return f"{bos}<|turn>user\n{user_text}<turn|>\n<|turn>model\n"


def build_caption_prompt(scene_text: str) -> str:
    return (
        f"{scene_text}\n\n"
        f"Based strictly on this data, describe the scene "
        f"in one or two sentences. Be concise and factual."
    )


def build_vqa_prompt(scene_text: str, question: str) -> str:
    return (
        f"{scene_text}\n\n"
        f"Based strictly on this data, answer concisely: {question}"
    )


# ---------------------------------------------------------------------------
# persistent engine (keeps llama-diffusion-cli alive across images)
# ---------------------------------------------------------------------------

import pty as _pty
import select as _select


class PersistentEngine:
    """Manages a long-lived llama-diffusion-cli subprocess using a PTY
    for line-buffered I/O (conversation mode)."""

    def __init__(self, gguf_path, temperature=0.7, ngpu_layers=0, prompt_timeout=300, n_predict=256):
        self._proc = None
        self._master_fd = None
        self.closed = False
        self._prompt_timeout = prompt_timeout
        self._start(gguf_path, temperature, ngpu_layers, n_predict)

    def _start(self, gguf_path, temperature, ngpu_layers, n_predict=256):
        if not LLAMA_CLI.exists():
            raise FileNotFoundError(f"llama-diffusion-cli not found at {LLAMA_CLI}")
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF not found at {gguf_path}")

        cmd = [str(LLAMA_CLI), "-m", str(gguf_path), "--conversation",
               "--temp", str(temperature), "-n", str(n_predict)]
        if ngpu_layers > 0:
            cmd.extend(["-ngl", str(ngpu_layers)])

        env = os.environ.copy()
        env["LLAMA_ARG_THREADS"] = str(os.cpu_count() or 4)
        if os.path.exists(CUDA_LIB_DIR):
            env.setdefault("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{CUDA_LIB_DIR}:{env['LD_LIBRARY_PATH']}"

        master_fd, slave_fd = _pty.openpty()
        self._master_fd = master_fd
        self._proc = subprocess.Popen(
            cmd, stdin=slave_fd, stdout=slave_fd,
            stderr=subprocess.PIPE, env=env,
        )
        os.close(slave_fd)
        self._read_until_prompt()

    def _read_chunk(self, timeout=None):
        timeout = timeout or self._prompt_timeout
        r, _, _ = _select.select([self._master_fd], [], [], timeout)
        if not r:
            return None
        try:
            data = os.read(self._master_fd, 65536)
        except OSError as e:
            err = ""
            if self._proc and self._proc.poll() is not None:
                err = self._proc.stderr.read().decode("utf-8", errors="replace")[:1000]
            else:
                # Try non-blocking stderr read
                try:
                    import fcntl
                    fd = self._proc.stderr.fileno()
                    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                    err = self._proc.stderr.read().decode("utf-8", errors="replace")[:1000]
                except Exception:
                    err = "(stderr read failed)"
            raise RuntimeError(f"PTY read error: {e} (rc={self._proc.poll()}) stderr={err!r}")
        return data

    def _write(self, text):
        os.write(self._master_fd, text.encode("utf-8"))
        os.write(self._master_fd, b"\n")

    def _read_until_prompt(self):
        buf = b""
        while True:
            chunk = self._read_chunk()
            if chunk is None:
                raise TimeoutError(f"timeout waiting for prompt, buf={buf[-200:]!r}")
            if not chunk:
                raise RuntimeError("Persistent engine process died")
            buf += chunk
            buf = buf[-65536:]  # keep bounded
            if buf.endswith(b"> ") or buf.endswith(b"\n> "):
                return

    def generate(self, prompt):
        if self.closed or self._proc is None:
            raise RuntimeError("Engine is closed")
        # PTY conversation mode reads line by line; flatten multi-line prompts
        prompt = ' '.join(prompt.splitlines())
        # Clear conversation history and wait for fresh prompt
        self._write("/clear")
        self._read_until_prompt()
        # Send user prompt
        self._write(prompt)
        # Read response until the next prompt
        buf = b""
        while True:
            chunk = self._read_chunk()
            if chunk is None:
                break
            if not chunk:
                break
            buf += chunk
            # Check for trailing prompt marker
            if buf.endswith(b"\r\n> ") or buf.endswith(b"> "):
                idx = buf.rfind(b"\r\n> ")
                if idx >= 0:
                    buf = buf[:idx]
                else:
                    buf = buf.rstrip()
                    if buf.endswith(b">"):
                        buf = buf[:-1].rstrip()
                break
        result = _extract_response(buf.decode("utf-8", errors="replace"))
        # Drain any leftover prompt data so next call starts clean
        self._drain()
        return result

    def _drain(self):
        try:
            r, _, _ = _select.select([self._master_fd], [], [], 0.1)
            if r:
                os.read(self._master_fd, 65536)
        except (OSError, _select.error):
            pass

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self._write("/exit")
            except Exception:
                pass
            try:
                if self._proc:
                    self._proc.wait(timeout=10)
            except Exception:
                if self._proc:
                    self._proc.kill()
            if self._master_fd is not None:
                try:
                    os.close(self._master_fd)
                except OSError:
                    pass
            self._proc = None
            self._master_fd = None


_ENGINE = None


def _get_engine(temperature=0.7, ngpu_layers=0, n_predict=256, prompt_timeout=300):
    global _ENGINE
    if _ENGINE is None or _ENGINE.closed:
        _ENGINE = PersistentEngine(GGUF_PATH, temperature=temperature,
                                    ngpu_layers=ngpu_layers, n_predict=n_predict,
                                    prompt_timeout=prompt_timeout)
    return _ENGINE


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def run_diffusion(
    prompt: str,
    n_predict: int = 256,
    temperature: float = 0.7,
    timeout: int = 300,
    ngpu_layers: int = 0,
    persist: bool = False,
) -> str:
    if persist:
        engine = _get_engine(temperature=temperature, ngpu_layers=ngpu_layers,
                             n_predict=n_predict, prompt_timeout=timeout)
        return engine.generate(prompt)

    if not LLAMA_CLI.exists():
        raise FileNotFoundError(
            f"llama-diffusion-cli not found at {LLAMA_CLI}. "
            f"Build it first: see README.md"
        )
    if not GGUF_PATH.exists():
        raise FileNotFoundError(f"GGUF file not found at {GGUF_PATH}")

    cmd = [
        str(LLAMA_CLI),
        "-m", str(GGUF_PATH),
        "-p", prompt,
        "-n", str(n_predict),
        "-no-cnv",
        "--temp", str(temperature),
    ]
    if ngpu_layers > 0:
        cmd.extend(["-ngl", str(ngpu_layers)])

    env = os.environ.copy()
    env["LLAMA_ARG_THREADS"] = str(os.cpu_count() or 4)
    if os.path.exists(CUDA_LIB_DIR):
        env.setdefault("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{CUDA_LIB_DIR}:{env['LD_LIBRARY_PATH']}"

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
    )

    return _extract_response(result.stdout)


def _extract_response(raw_output: str) -> str:
    lines = raw_output.splitlines()
    content = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started:
            if stripped == "thought":
                started = True
            continue
        if (stripped.startswith("total time:")
                or stripped.startswith("throughput:")
                or stripped == ">"):
            break
        if stripped.startswith("diffusion step:") or stripped.startswith("0."):
            continue
        content.append(line)
    response = "\n".join(content).strip()
    if not response:
        return "(empty response)"
    return response


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DiffusionGemma VLM – pluggable vision encoder + llama-diffusion-cli"
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument(
        "--task",
        type=str,
        default="caption",
        choices=["caption", "vqa", "detect", "pose", "obb", "chat"],
        help="Task type (default: caption)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="yolo",
        choices=["yolo", "siglip2", "moonvit"],
        help="Vision encoder backend (default: yolo)",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Question for VQA or text for chat")
    parser.add_argument("--n-predict", type=int, default=256, help="Max output tokens (<=256 for single diffusion block)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=300, help="Subprocess timeout (seconds)")
    parser.add_argument(
        "--yolo-tasks",
        type=str,
        default="aabb",
        help="Comma-separated YOLO task types: aabb,pose,obb (default: aabb)",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model variant (default: yolo11n.pt, used only with --encoder yolo)",
    )
    parser.add_argument(
        "--ngpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU (requires CUDA build, default: 0 = CPU)",
    )
    parser.add_argument(
        "--persist", action="store_true",
        help="Keep llama-diffusion-cli alive across invocations (conversation mode)",
    )

    args = parser.parse_args()
    yolo_tasks_list = [t.strip() for t in args.yolo_tasks.split(",")]

    # --- YOLO-only modes (fast, no LLM) ---
    if args.task in ("detect", "pose", "obb"):
        if not args.image:
            print("--image is required for detection/pose/obb tasks", file=sys.stderr)
            sys.exit(1)
        if args.encoder != "yolo":
            print("--encoder is ignored for YOLO-only tasks (detect/pose/obb)", file=sys.stderr)
        desc = run_encoder_yolo(args.image, yolo_tasks=yolo_tasks_list)
        print(desc)
        return

    # --- Chat (text only, no image) ---
    if args.task == "chat":
        if not args.prompt:
            print("--prompt is required for chat task", file=sys.stderr)
            sys.exit(1)
        prompt = build_chat_prompt(args.prompt)
        response = run_diffusion(
            prompt,
            n_predict=args.n_predict,
            temperature=args.temperature,
            timeout=args.timeout,
            ngpu_layers=args.ngpu_layers,
            persist=args.persist,
        )
        print(response)
        return

    # --- Vision tasks (caption / vqa) ---
    if not args.image:
        print("--image is required for caption/vqa tasks", file=sys.stderr)
        sys.exit(1)

    # Start persistent engine BEFORE vision encoder to avoid CUDA/PyTorch
    # interference with the PTY subprocess.
    if args.persist:
        eng = _get_engine(
            temperature=args.temperature,
            ngpu_layers=args.ngpu_layers,
            n_predict=args.n_predict,
            prompt_timeout=args.timeout,
        )

    encoder_fn = build_encoder(args.encoder, yolo_tasks=yolo_tasks_list)
    scene_text = encoder_fn(args.image)

    if args.task == "caption":
        user_prompt = build_caption_prompt(scene_text)
    elif args.task == "vqa":
        if not args.prompt:
            print("--prompt is required for vqa task", file=sys.stderr)
            sys.exit(1)
        user_prompt = build_vqa_prompt(scene_text, args.prompt)

    if args.persist:
        send_prompt = user_prompt
        response = eng.generate(send_prompt)
    else:
        send_prompt = build_chat_prompt(user_prompt)
        response = run_diffusion(
            send_prompt,
            n_predict=args.n_predict,
            temperature=args.temperature,
            timeout=args.timeout,
            ngpu_layers=args.ngpu_layers,
            persist=False,
        )

    print(response)


if __name__ == "__main__":
    main()
