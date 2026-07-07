#!/bin/bash
# Run the new benchmark tasks
# Uses /mnt/HDD1/tmp for temp files and cleans up after

export TMPDIR=/mnt/HDD1/tmp

cd "$(dirname "$0")/scripts"

RESULTS_DIR="../results"
mkdir -p "$RESULTS_DIR"

run_model() {
    script=$1
    model=$2
    venv=$(python3 -c "
from common import PROJECT_DIR
import json
print(str(PROJECT_DIR / '$model' / '.venv' / 'bin' / 'python'))
" 2>/dev/null || true)
    
    # Use MODEL_VENV-like mapping
    case $model in
        dinotool) venv="../../VLMcollection/DINOtool/.venv/bin/python" ;;
        dinov3) venv="../../VLMcollection/dinov3/.venv/bin/python" ;;
        siglip2) venv="../../VLMcollection/siglip2/.venv/bin/python" ;;
        moonvit) venv="../../VLMcollection/moonvit/.venv/bin/python" ;;
        florence2) venv="../../VLMcollection/florence-2/.venv/bin/python" ;;
        locate_anything) venv="../../VLMcollection/locate_anything/.venv/bin/python" ;;
        paligemma) venv="../../VLMcollection/paligemma/.venv/bin/python" ;;
        llama_vision) venv="../../VLMcollection/llama-vision/.venv/bin/python" ;;
        phi_vision) venv="../../VLMcollection/phi-vision/.venv/bin/python" ;;
        cosmos_nemotron) venv="../../VLMcollection/cosmos-nemotron/.venv/bin/python" ;;
        qwen3_native) venv="../../VLMcollection/qwen3-vl_instruct/.venv/bin/python" ;;
        qwen3_thinking) venv="../../VLMcollection/qwen3-vl_thinking/.venv/bin/python" ;;
        yolo26) venv="../../VLMcollection/yolo11-26/.venv/bin/python" ;;
        yolo11) venv="../../VLMcollection/yolo11-26/.venv/bin/python" ;;
    esac
    
    if [ ! -f "$venv" ]; then
        echo "  [SKIP] No venv for $model at $venv"
        return
    fi
    
    log="$RESULTS_DIR/${model}_${script%.py}.log"
    echo "  Starting: $model ($script) -> $log"
    timeout 7200 "$venv" "$script" --model "$model" "$@" > "$log" 2>&1
    echo "  Done: $model (exit code $?)"
}

echo "=== Classification (Tiny ImageNet, 200 images) ==="
for m in dinotool dinov3 siglip2 moonvit; do
    run_model benchmark_classification.py $m --max-images 200
done

echo ""
echo "=== Scene Analysis (50 images) ==="
for m in florence2 paligemma llama_vision phi_vision cosmos_nemotron qwen3_native qwen3_thinking; do
    run_model benchmark_scene.py $m --max-images 50
done

echo ""
echo "=== Segmentation (25 images) ==="
for m in florence2 locate_anything; do
    run_model benchmark_segmentation.py $m --max-images 25
done

echo ""
echo "=== Tracking (MOT17, 2 seqs, 50 frames) ==="
for m in yolo26 yolo11; do
    run_model benchmark_tracking.py $m --max-sequences 2 --max-frames 50
done

echo ""
echo "=== 6D Pose (Linemod, 25 targets) ==="
for m in yolo26 yolo11; do
    run_model benchmark_6dpose.py $m --max-images 25
done

echo ""
echo "=== ALL NEW BENCHMARKS COMPLETE ==="
