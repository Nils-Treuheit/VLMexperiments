#!/bin/bash
# Comprehensive test for all models in VLMcollection
# Tests inference/prompting in all available modes for each model
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_IMG="/mnt/HDD1/Project_Data/public_datasets/coco/val2017/000000000139.jpg"
SAMPLE_IMG2="/mnt/HDD1/Project_Data/public_datasets/coco/val2017/000000000285.jpg"
DOTA_IMG="/mnt/HDD1/Project_Data/public_datasets/dotav1/images/P0000.png"
RESULTS_DIR="/tmp/vlm_collection_test_$(date +%s)"
PASS=0
FAIL=0
FAILED_TESTS=""

mkdir -p "$RESULTS_DIR"

source ~/.bashrc 2>/dev/null || true

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

export PYTHONPATH=""
export LD_LIBRARY_PATH=""

pass() { PASS=$((PASS+1)); echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_TESTS="$FAILED_TESTS\n  $1"; echo -e "${RED}[FAIL]${NC} $1: $2"; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

report_header() {
    echo ""
    echo "============================================================"
    echo "  Testing: $1"
    echo "============================================================"
}

test_python() {
    local model="$1"
    local desc="$2"
    local cmd="$3"
    local timeout="${4:-120}"
    info "Running: $cmd"
    if timeout "$timeout" bash -c "$cmd" > "$RESULTS_DIR/${model}.log" 2>&1; then
        pass "$desc"
    else
        local ec=$?
        local tail_out=$(tail -5 "$RESULTS_DIR/${model}.log" 2>/dev/null)
        fail "$desc" "exit code $ec: $tail_out"
    fi
}

# ============================================================
# 1. YOLO11-26 — detection, pose, obb
# ============================================================
report_header "YOLO11/26 (detection, pose, OBB)"
VENV_YOLO="$SCRIPT_DIR/yolo11-26/.venv/bin/python"
YOLO_MODELS="$SCRIPT_DIR/yolo11-26/models"

test_python "yolo_detect" "YOLO11 detection" \
    "$VENV_YOLO -c \"
from ultralytics import YOLO
model = YOLO('$YOLO_MODELS/yolo11n.pt')
r = model('$SAMPLE_IMG', verbose=False)
print(f'Detected {len(r[0].boxes)} objects')
\""

test_python "yolo_pose" "YOLO11 pose estimation" \
    "$VENV_YOLO -c \"
from ultralytics import YOLO
model = YOLO('$YOLO_MODELS/yolo11n-pose.pt')
r = model('$SAMPLE_IMG', verbose=False)
print(f'Poses: {len(r[0].keypoints)}')
\""

test_python "yolo_obb" "YOLO11 OBB detection" \
    "$VENV_YOLO -c \"
from ultralytics import YOLO
model = YOLO('$YOLO_MODELS/yolo11n-obb.pt')
r = model('$DOTA_IMG', verbose=False, imgsz=640)
print(f'OBB detections: {len(r[0].obb) if r[0].obb else 0}')
\""

# ============================================================
# 2. Florence-2 — caption, detailed caption, OD
# ============================================================
report_header "Florence-2 (caption, detailed caption, OD)"
VENV_F2="$SCRIPT_DIR/florence-2/.venv/bin/python"

test_python "florence2_caption" "Florence-2 caption" \
    "cd '$SCRIPT_DIR/florence-2' && PYTHONPATH='' LD_LIBRARY_PATH='' $VENV_F2 -c \"
import torch, warnings
warnings.filterwarnings('ignore')
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tdtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = Florence2ForConditionalGeneration.from_pretrained('microsoft/Florence-2-large-ft', torch_dtype=tdtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True)
img = Image.open('$SAMPLE_IMG').convert('RGB')
for task in ['<CAPTION>', '<DETAILED_CAPTION>']:
    inputs = processor(text=task, images=img, return_tensors='pt').to(device)
    gids = model.generate(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'], max_new_tokens=200, num_beams=1)
    result = processor.batch_decode(gids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(result, task=task, image_size=(img.width, img.height))
    print(f'{task}: {parsed}')
print('Florence-2: OK')
\" 2>&1 | grep -v 'Warning\|UNEXPECTED\|MISSING\|MISMATCH\|Loading\|^$'" 180

# ============================================================
# 3. PaliGemma 2 — caption, VQA (gated)
# ============================================================
report_header "PaliGemma2-3B (caption, VQA) — gated, needs HF login"
VENV_PG="$SCRIPT_DIR/paligemma/.venv/bin/python"

test_python "paligemma" "PaliGemma2 caption+VQA" \
    "cd '$SCRIPT_DIR/paligemma' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && $VENV_PG -c \"
import torch, warnings
warnings.filterwarnings('ignore')
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
model_id = 'google/paligemma2-3b-mix-224'
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto').eval()
processor = AutoProcessor.from_pretrained(model_id)
img = Image.open('$SAMPLE_IMG').convert('RGB')
for prompt in ['What is in this image?', 'caption en']:
    inputs = processor(img, prompt, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)
    print(f'{prompt}: {processor.decode(output[0], skip_special_tokens=True)}')
print('PaliGemma: OK')
\" 2>&1 | grep -v 'Warning\|Loading\|^$'" 300

# ============================================================
# 4. Llama-Vision — describe (via Ollama)
# ============================================================
report_header "Llama-3.2-Vision (describe via Ollama)"
VENV_LLAMA="$SCRIPT_DIR/llama-vision/.venv/bin/python"

test_python "llama_vision" "Llama-3.2-Vision describe" \
    "cd '$SCRIPT_DIR/llama-vision' && PYTHONPATH='' LD_LIBRARY_PATH='' $VENV_LLAMA run.py --image '$SAMPLE_IMG' --task describe 2>/dev/null" 120

# ============================================================
# 5. Phi-3.5 Vision — caption
# ============================================================
report_header "Phi-3.5-Vision (description)"
VENV_PHI="$SCRIPT_DIR/phi-vision/.venv/bin/python"

test_python "phi_vision" "Phi-3.5-Vision" \
    "cd '$SCRIPT_DIR/phi-vision' && PYTHONPATH='' LD_LIBRARY_PATH='' $VENV_PHI run.py 2>&1 | tail -5" 300

# ============================================================
# 6. Cosmos-Nemotron — VQA
# ============================================================
report_header "Cosmos-Reason1-7B (VQA, scene description)"
VENV_COSMOS="$SCRIPT_DIR/cosmos-nemotron/.venv/bin/python"

test_python "cosmos_nemotron" "Cosmos-Nemotron inference" \
    "cd '$SCRIPT_DIR/cosmos-nemotron' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && $VENV_COSMOS -c \"
import torch, warnings
warnings.filterwarnings('ignore')
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
model_id = 'nvidia/Cosmos-Reason1-7B'
model = AutoModelForMultimodalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa')
processor = AutoProcessor.from_pretrained(model_id)
img = Image.open('$SAMPLE_IMG').convert('RGB')
msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': 'Describe this scene in one sentence.'}]}]
inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
response = processor.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
print(f'Cosmos: {response}')
print('Cosmos: OK')
\" 2>&1 | tail -5" 600

# ============================================================
# 7. LocateAnything — visual grounding
# ============================================================
report_header "LocateAnything-3B (visual grounding)"
VENV_LA="$SCRIPT_DIR/locate_anything/.venv/bin/python"

test_python "locate_anything" "LocateAnything grounding" \
    "cd '$SCRIPT_DIR/locate_anything' && PYTHONPATH='' LD_LIBRARY_PATH='' $VENV_LA infer.py '$SAMPLE_IMG' 'person' --json 2>/dev/null" 300

# ============================================================
# 8. Qwen3-VL Instruct — description, detection
# ============================================================
report_header "Qwen3-VL-8B-Instruct (description, detection)"
VENV_QWEN3="$SCRIPT_DIR/qwen3-vl_instruct/.venv/bin/python"

test_python "qwen3_instruct" "Qwen3-VL-Instruct describe" \
    "cd '$SCRIPT_DIR/qwen3-vl_instruct' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && $VENV_QWEN3 infer_qwen3.py '$SAMPLE_IMG' --json --no-thinking 2>/dev/null" 300

# ============================================================
# 9. Qwen3-VL Thinking — description, detection
# ============================================================
report_header "Qwen3-VL-8B-Thinking (description, detection)"
VENV_THINK="$SCRIPT_DIR/qwen3-vl_thinking/.venv/bin/python"

test_python "qwen3_thinking" "Qwen3-VL-Thinking describe" \
    "cd '$SCRIPT_DIR/qwen3-vl_thinking' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && $VENV_THINK -c \"
from qwen_detector import QwenVLDetector
d = QwenVLDetector()
result = d.describe('$SAMPLE_IMG', 'What do you see?')
print(f'Thinking: {result}')
\" 2>&1 | tail -10" 600

# ============================================================
# 10. DINOv3 — describe, encode
# ============================================================
report_header "DINOv3 (vision encoder)"
VENV_DINO="$SCRIPT_DIR/dinov3/.venv/bin/python"

test_python "dinov3_describe" "DINOv3 describe" \
    "cd '$SCRIPT_DIR/dinov3' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && $VENV_DINO run.py --image '$SAMPLE_IMG' --task describe 2>/dev/null" 120

test_python "dinov3_encode" "DINOv3 encode" \
    "cd '$SCRIPT_DIR/dinov3' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && $VENV_DINO run.py --image '$SAMPLE_IMG' --task encode 2>/dev/null" 120

# ============================================================
# 11. MoonViT — describe, encode
# ============================================================
report_header "MoonViT (vision encoder)"
VENV_MOON="$SCRIPT_DIR/moonvit/.venv/bin/python"

test_python "moonvit_describe" "MoonViT describe" \
    "cd '$SCRIPT_DIR/moonvit' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && $VENV_MOON run.py --image '$SAMPLE_IMG' --task describe 2>/dev/null" 120

test_python "moonvit_encode" "MoonViT encode" \
    "cd '$SCRIPT_DIR/moonvit' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && $VENV_MOON run.py --image '$SAMPLE_IMG' --task encode 2>/dev/null" 120

# ============================================================
# 12. SigLIP2 — describe, encode
# ============================================================
report_header "SigLIP2 (vision encoder)"
VENV_S2="$SCRIPT_DIR/siglip2/.venv/bin/python"

# SigLIP2 uses system python (no venv setup yet)
test_python "siglip2_describe" "SigLIP2 describe" \
    "cd '$SCRIPT_DIR/siglip2' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && python3 run.py --image '$SAMPLE_IMG' --task describe 2>/dev/null" 120

test_python "siglip2_encode" "SigLIP2 encode" \
    "cd '$SCRIPT_DIR/siglip2' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && python3 run.py --image '$SAMPLE_IMG' --task encode 2>/dev/null" 120

# ============================================================
# 13. DiffusionGemma VL — caption, detect, pose (needs GGUF)
# ============================================================
report_header "DiffusionGemma-VL (YOLO encoder tasks — no GGUF needed)"
VENV_DG="$SCRIPT_DIR/diffusion_gemma_vl/.venv/bin/python"

# YOLO-only modes (detect, pose, obb) don't need the GGUF LLM
test_python "diffusion_gemma_detect" "DiffusionGemma YOLO detection" \
    "cd '$SCRIPT_DIR/diffusion_gemma_vl' && PYTHONPATH='' LD_LIBRARY_PATH='' $VENV_DG run.py --image '$SAMPLE_IMG' --task detect 2>/dev/null" 60

test_python "diffusion_gemma_pose" "DiffusionGemma YOLO pose" \
    "cd '$SCRIPT_DIR/diffusion_gemma_vl' && PYTHONPATH='' LD_LIBRARY_PATH='' $VENV_DG run.py --image '$SAMPLE_IMG' --task pose 2>/dev/null" 60

test_python "diffusion_gemma_obb" "DiffusionGemma YOLO OBB" \
    "cd '$SCRIPT_DIR/diffusion_gemma_vl' && PYTHONPATH='' LD_LIBRARY_PATH='' $VENV_DG run.py --image '$DOTA_IMG' --task obb 2>/dev/null" 60

# ============================================================
# 14. Phi-4 Multimodal — inference
# ============================================================
report_header "Phi-4 Multimodal (inference)"
VENV_PHI4="$SCRIPT_DIR/phi-4_multimodal/.venv/bin/python"

test_python "phi4" "Phi-4 Multimodal inference" \
    "cd '$SCRIPT_DIR/phi-4_multimodal' && PYTHONPATH='' LD_LIBRARY_PATH='' source ~/.bashrc 2>/dev/null && $VENV_PHI4 run.py --image '$SAMPLE_IMG' --prompt 'Describe this image' --max-tokens 128 2>/dev/null" 300

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo -e "  TEST RESULTS: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo "============================================================"
if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Failed tests:${NC}$FAILED_TESTS"
fi
echo ""
echo "Logs: $RESULTS_DIR"
echo ""

exit $FAIL
