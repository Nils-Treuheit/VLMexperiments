#!/usr/bin/env bash

SCRIPTS="/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts"
COLLECTION="/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection"
RESULTS_DIR="/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/results"
mkdir -p "$RESULTS_DIR"

clean_vram() {
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 2
}

venv_for() {
    local model="$1"
    case "$model" in
        dinotool) echo "$COLLECTION/DINOtool/.venv/bin/python" ;;
        dinov3) echo "$COLLECTION/dinov3/.venv/bin/python" ;;
        siglip2) echo "$COLLECTION/siglip2/.venv/bin/python" ;;
        moonvit) echo "$COLLECTION/moonvit/.venv/bin/python" ;;
        florence2) echo "$COLLECTION/florence-2/.venv/bin/python" ;;
        paligemma) echo "$COLLECTION/paligemma/.venv/bin/python" ;;
        phi_vision) echo "$COLLECTION/phi-vision/.venv/bin/python" ;;
        cosmos_nemotron) echo "$COLLECTION/cosmos-nemotron/.venv/bin/python" ;;
        llama_vision) echo "$COLLECTION/llama-vision/.venv/bin/python" ;;
        qwen3_native) echo "$COLLECTION/qwen3-vl_instruct/.venv/bin/python" ;;
        qwen3_thinking) echo "$COLLECTION/qwen3-vl_thinking/.venv/bin/python" ;;
        phi4_multimodal) echo "$COLLECTION/phi-4_multimodal/.venv/bin/python" ;;
        diffusion_gemma*) echo "$COLLECTION/diffusion_gemma_vl/.venv/bin/python" ;;
        locate_anything) echo "$COLLECTION/locate_anything/.venv/bin/python" ;;
        yolo26*|yolo11*) echo "$COLLECTION/yolo11-26/.venv/bin/python" ;;
        *) echo "" ;;
    esac
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

# Phase 4: High VRAM — Captioning
echo ""
echo "########################################################"
echo "# PHASE 4: High VRAM Captioning (~10-16 GB)"
echo "# $(date)"
echo "########################################################"

# Use 25 images for slower models, 50 for fast ones
for model in cosmos_nemotron 50 llama_vision 50 qwen3_native 50 qwen3_thinking 25; do
    model_name="$model"
    model_max=50
    # This double-variable trick reads pairs
done

# Actually let's just do it inline
run_benchmark "benchmark_caption.py" "cosmos_nemotron" 50
run_benchmark "benchmark_caption.py" "llama_vision" 50
run_benchmark "benchmark_caption.py" "qwen3_native" 50
run_benchmark "benchmark_caption.py" "qwen3_thinking" 25

# Phase 5: Highest VRAM
echo ""
echo "########################################################"
echo "# PHASE 5: Highest VRAM Captioning (~14-20 GB)"
echo "# $(date)"
echo "########################################################"

run_benchmark "benchmark_caption.py" "phi4_multimodal" 25
run_benchmark "benchmark_caption.py" "diffusion_gemma" 25
run_benchmark "benchmark_caption.py" "diffusion_gemma_yolo" 25
run_benchmark "benchmark_caption.py" "diffusion_gemma_siglip2" 25
run_benchmark "benchmark_caption.py" "diffusion_gemma_moonvit" 25

# VQA
echo ""
echo "########################################################"
echo "# VQA BENCHMARKS"
echo "# $(date)"
echo "########################################################"

run_benchmark "benchmark_vqa.py" "paligemma" 50
run_benchmark "benchmark_vqa.py" "florence2" 50
run_benchmark "benchmark_vqa.py" "cosmos_nemotron" 50
run_benchmark "benchmark_vqa.py" "qwen3_native" 50
run_benchmark "benchmark_vqa.py" "qwen3_thinking" 25
run_benchmark "benchmark_vqa.py" "phi4_multimodal" 25
run_benchmark "benchmark_vqa.py" "diffusion_gemma" 25

# VQA for models that have it
run_benchmark "benchmark_vqa.py" "llama_vision" 50
run_benchmark "benchmark_vqa.py" "phi_vision" 25

# Best-effort OD
echo ""
echo "########################################################"
echo "# DETECTION BENCHMARKS (models with pycocotools)"
echo "# $(date)"
echo "########################################################"

run_benchmark "benchmark_od.py" "qwen3_native" 50 "--dataset coco"
run_benchmark "benchmark_od.py" "qwen3_thinking" 25 "--dataset coco"

# Grounding
run_benchmark "benchmark_grounding.py" "qwen3_native" 50
run_benchmark "benchmark_grounding.py" "qwen3_thinking" 25

echo ""
echo "================================================"
echo "  PHASE 4+5 COMPLETE at $(date)"
echo "================================================"
