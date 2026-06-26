#!/bin/bash
set -e

SHOWCASE_DIR="/mnt/HDD1/Project_Code/VLMshowcase"
VENV="$SHOWCASE_DIR/.venv/bin/activate"
TMPDIR="/tmp/vlm_demo"
mkdir -p "$TMPDIR"

source "$VENV"

usage() {
    echo "VLMshowcase — Quick Demo Runner"
    echo ""
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  list                          Show available models and demo material"
    echo "  scene     <image>             Analyze scene with Qwen3 + YOLO"
    echo "  detect    <image>             Object detection (YOLO + LocateAnything)"
    echo "  pose      <image>             Pose estimation with YOLO"
    echo "  intent    <image>             Human intent analysis (Qwen3-Thinking)"
    echo "  track     <video>             Object tracking (YOLO)"
    echo "  compare   <image>             Side-by-side model comparison"
    echo "  webcam    [--model yolo26n]   Live webcam detection"
    echo "  all       <image>             Run ALL capabilities on one image"
    echo ""
    echo "Images: bus.jpg, zidane.jpg, person.jpg, coco_*.jpg, or any path"
    echo "Videos: vtest.avi, walking_people.mp4, car_traffic.mp4, or any path"
    exit 1
}

CMD="${1:-help}"
shift || true

case "$CMD" in
    list)
        vlm-demo list
        ;;
    scene)
        vlm-demo scene "$@"
        ;;
    detect)
        vlm-demo detect "$@" --yolo yolo26n yolo26s --grounding person car
        ;;
    pose)
        vlm-demo pose "$@" --model yolo26n-pose yolo11n-pose
        ;;
    intent)
        vlm-demo intent "$@" --all
        ;;
    track)
        vlm-demo track "$@"
        ;;
    compare)
        vlm-demo compare "$@"
        ;;
    webcam)
        vlm-demo webcam "$@"
        ;;
    all)
        IMAGE="$1"
        if [ -z "$IMAGE" ]; then
            IMAGE="bus.jpg"
        fi
        echo "=============================================="
        echo "  Running ALL capabilities on: $IMAGE"
        echo "=============================================="
        echo ""
        echo "--- SCENE ANALYSIS ---"
        vlm-demo scene "$IMAGE" 2>/dev/null
        echo ""
        echo "--- OBJECT DETECTION ---"
        vlm-demo detect "$IMAGE" --yolo yolo26n --grounding person 2>/dev/null
        echo ""
        echo "--- POSE ESTIMATION ---"
        vlm-demo pose "$IMAGE" --model yolo26n-pose 2>/dev/null
        echo ""
        echo "--- HUMAN INTENT ---"
        vlm-demo intent "$IMAGE" --aspect action 2>/dev/null
        echo ""
        echo "--- MODEL COMPARISON ---"
        vlm-demo compare "$IMAGE" 2>/dev/null
        echo ""
        echo "All outputs saved to /tmp/vlm_demo/"
        ;;
    *)
        usage
        ;;
esac
