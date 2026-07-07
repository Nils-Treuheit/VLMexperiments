# Source this script before running TRT: source setup_env.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV_DIR="$SCRIPT_DIR/.venv"
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.10/site-packages/tensorrt_libs:$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
source "$VENV_DIR/bin/activate"
