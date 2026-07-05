#!/usr/bin/env bash
# Don't use set -e — we handle errors per-model

SCRIPTS="/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts"
COLLECTION="/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection"
MAX_IMAGES=50
RESULTS_DIR="/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/results"
mkdir -p "$RESULTS_DIR"

clean_vram() {
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 2
}

check_vram() {
    python3 -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader']).decode()
free = int(re.sub(r'[^0-9]', '', out.strip()))
print(f'VRAM free: {free} MiB')
exit(0 if free > $1 else 1)
" 2>/dev/null
}

# Determine venv python for each model
venv_for() {
    local model="$1"
    local dir_map=(
        "dinotool:DINOtool"
        "dinov3:dinov3"
        "siglip2:siglip2"
        "moonvit:moonvit"
        "florence2:florence-2"
        "paligemma:paligemma"
        "phi_vision:phi-vision"
        "cosmos_nemotron:cosmos-nemotron"
        "llama_vision:llama-vision"
        "qwen3_native:qwen3-vl_instruct"
        "qwen3_thinking:qwen3-vl_thinking"
        "diffusion_gemma:diffusion_gemma_vl"
        "diffusion_gemma_yolo:diffusion_gemma_vl"
        "diffusion_gemma_yolo_pose:diffusion_gemma_vl"
        "diffusion_gemma_yolo_obb:diffusion_gemma_vl"
        "diffusion_gemma_siglip2:diffusion_gemma_vl"
        "diffusion_gemma_moonvit:diffusion_gemma_vl"
        "phi4_multimodal:phi-4_multimodal"
        "yolo26:yolo11-26"
        "yolo26s:yolo11-26"
        "yolo26m:yolo11-26"
        "yolo26l:yolo11-26"
        "yolo26x:yolo11-26"
        "yolo26_pose:yolo11-26"
        "yolo26s_pose:yolo11-26"
        "yolo26_obb:yolo11-26"
        "yolo26s_obb:yolo11-26"
        "yolo11:yolo11-26"
        "yolo11s:yolo11-26"
        "yolo11m:yolo11-26"
        "yolo11l:yolo11-26"
        "yolo11x:yolo11-26"
        "yolo11_pose:yolo11-26"
        "yolo11s_pose:yolo11-26"
        "yolo11_obb:yolo11-26"
        "yolo11s_obb:yolo11-26"
        "locate_anything:locate_anything"
    )
    for entry in "${dir_map[@]}"; do
        local key="${entry%%:*}"
        local dir="${entry##*:}"
        if [ "$key" = "$model" ]; then
            echo "$COLLECTION/$dir/.venv/bin/python"
            return
        fi
    done
    echo ""
}

run_benchmark() {
    local script="$1"
    local model="$2"
    local max_img="$3"
    local extra="$4"
    local venv_py
    venv_py=$(venv_for "$model")
    if [ -z "$venv_py" ] || [ ! -x "$venv_py" ]; then
        echo "  [SKIP] $model: no venv at $venv_py"
        return
    fi
    echo ""
    echo "================================================"
    echo "  $(date '+%H:%M:%S') — $script — $model"
    echo "================================================"
    clean_vram
    if [ -n "$extra" ]; then
        $venv_py "$SCRIPTS/$script" --model "$model" --max-images "$max_img" $extra 2>&1
    else
        $venv_py "$SCRIPTS/$script" --model "$model" --max-images "$max_img" 2>&1
    fi
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  [WARN] $model $script exited with code $rc"
    fi
    clean_vram
}

# ========================================
# PHASE 1: Low VRAM — Vision Encoders
# ========================================
echo ""
echo "########################################"
echo "# PHASE 1: Vision Encoders (~1-2 GB)"
echo "########################################"

for model in dinotool dinov3 siglip2 moonvit; do
    run_benchmark "benchmark_caption.py" "$model" "$MAX_IMAGES"
done

# ========================================
# PHASE 2: Low VRAM — YOLO models
# ========================================
echo ""
echo "########################################"
echo "# PHASE 2: YOLO Detection (~1-3 GB)"
echo "########################################"

for task in benchmark_od.py benchmark_pose.py benchmark_obb.py; do
    if [ "$task" = "benchmark_od.py" ]; then
        for model in yolo26 yolo11; do
            run_benchmark "$task" "$model" "$MAX_IMAGES" "--dataset coco"
        done
    elif [ "$task" = "benchmark_pose.py" ]; then
        run_benchmark "$task" "yolo26_pose" "$MAX_IMAGES"
    elif [ "$task" = "benchmark_obb.py" ]; then
        run_benchmark "$task" "yolo26_obb" "$MAX_IMAGES"
    fi
done

# ========================================
# PHASE 3: Medium VRAM (~4-10 GB)
# ========================================
echo ""
echo "########################################"
echo "# PHASE 3: Medium VRAM (~4-10 GB)"
echo "########################################"

for model in florence2 paligemma phi_vision; do
    run_benchmark "benchmark_caption.py" "$model" "$MAX_IMAGES"
done

run_benchmark "benchmark_od.py" "locate_anything" "$MAX_IMAGES" "--dataset coco"
run_benchmark "benchmark_od.py" "florence2" "$MAX_IMAGES" "--dataset coco"
run_benchmark "benchmark_od.py" "paligemma" "$MAX_IMAGES" "--dataset coco"

# Grounding
run_benchmark "benchmark_grounding.py" "locate_anything" "$MAX_IMAGES"
run_benchmark "benchmark_grounding.py" "florence2" "$MAX_IMAGES"

# ========================================
# PHASE 4: High VRAM (~10-16 GB)
# ========================================
echo ""
echo "########################################"
echo "# PHASE 4: High VRAM (~10-16 GB)"
echo "########################################"

for model in cosmos_nemotron llama_vision qwen3_native qwen3_thinking; do
    run_benchmark "benchmark_caption.py" "$model" "$MAX_IMAGES"
done

# VQA for high-VRAM models
for model in cosmos_nemotron llama_vision qwen3_native qwen3_thinking; do
    run_benchmark "benchmark_vqa.py" "$model" "$MAX_IMAGES"
done

# OD for models that support it
run_benchmark "benchmark_od.py" "qwen3_native" "$MAX_IMAGES" "--dataset coco"
run_benchmark "benchmark_od.py" "qwen3_thinking" "$MAX_IMAGES" "--dataset coco"
run_benchmark "benchmark_grounding.py" "qwen3_native" "$MAX_IMAGES"
run_benchmark "benchmark_grounding.py" "qwen3_thinking" "$MAX_IMAGES"

# ========================================
# PHASE 5: Highest VRAM (~14-20 GB)
# ========================================
echo ""
echo "########################################"
echo "# PHASE 5: Highest VRAM (~14-20 GB)"
echo "########################################"

for model in phi4_multimodal diffusion_gemma diffusion_gemma_yolo diffusion_gemma_siglip2 diffusion_gemma_moonvit; do
    run_benchmark "benchmark_caption.py" "$model" "$MAX_IMAGES"
done

# VQA for diffusion_gemma
run_benchmark "benchmark_vqa.py" "phi4_multimodal" "$MAX_IMAGES"
run_benchmark "benchmark_vqa.py" "diffusion_gemma" "$MAX_IMAGES"

echo ""
echo "================================================"
echo "  ALL BENCHMARKS COMPLETE at $(date)"
echo "================================================"
