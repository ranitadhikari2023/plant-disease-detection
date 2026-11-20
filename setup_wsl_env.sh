#!/usr/bin/env bash
# ==============================================================================
# setup_wsl_env.sh
# Production WSL Environment Creation & Isolation Script
# Plant Disease Detection V2 System
#
# Target Location: ~/venvs/plant-disease-detection (Linux Ext4 Filesystem)
# Mount Source:    /mnt/d/plant-disease-detection
# ==============================================================================

set -euo pipefail

echo "========================================================================"
echo "🌿 Initializing Isolated WSL Python Virtual Environment"
echo "========================================================================"

TARGET_DIR="$HOME/venvs/plant-disease-detection"
PROJECT_DIR="/mnt/d/plant-disease-detection"

# 1. Create target parent directory
mkdir -p "$HOME/venvs"

# 2. Check Python 3.12 availability
if ! command -v python3.12 &> /dev/null; then
    echo "⚠️ python3.12 not found directly. Checking python3..."
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python3.12"
fi

PY_VER=$($PYTHON_BIN --version)
echo "📦 Using system interpreter: $PYTHON_BIN ($PY_VER)"

# 3. Create fresh virtual environment if it doesn't already exist
if [ ! -d "$TARGET_DIR" ]; then
    echo "🚀 Creating fresh virtual environment at: $TARGET_DIR"
    $PYTHON_BIN -m venv "$TARGET_DIR"
else
    echo "ℹ️ Virtual environment already exists at: $TARGET_DIR"
fi

# 4. Activate virtual environment
source "$TARGET_DIR/bin/activate"

VENV_PY=$(which python)
echo "✅ Active Python interpreter: $VENV_PY"
echo "   Filesystem mount:"
df -T "$TARGET_DIR"

# 5. Upgrade pip
echo "🔄 Upgrading pip..."
python -m pip install --upgrade pip

# 6. Install verified, pinned dependencies
echo "📥 Installing pinned requirements from $PROJECT_DIR/requirements.txt..."
python -m pip install -r "$PROJECT_DIR/requirements.txt"

# 7. Check for dependency conflicts
echo "🔍 Running dependency integrity check..."
python -m pip check

# 8. Verify critical imports
echo "🧪 Verifying critical deep learning & ONNX imports..."
python -c "
import tensorflow as tf
import keras
import numpy as np
import PIL
import matplotlib
import psutil
import tf2onnx
import onnx
import onnxruntime as ort

print('  ✓ TensorFlow:', tf.__version__)
print('  ✓ Keras:     ', keras.__version__)
print('  ✓ NumPy:     ', np.__version__)
print('  ✓ PIL:       ', PIL.__version__)
print('  ✓ Matplotlib:', matplotlib.__version__)
print('  ✓ psutil:    ', psutil.__version__)
print('  ✓ tf2onnx:   ', tf2onnx.__version__)
print('  ✓ onnx:      ', onnx.__version__)
print('  ✓ onnxruntime:', ort.__version__)
print('✅ ALL CRITICAL IMPORTS SUCCESSFUL IN WSL ENVIRONMENT!')
"

# 9. Verify V2 Model deserialization and regression tests
cd "$PROJECT_DIR"
echo "🔍 Running model artifact verification in WSL..."
python training/verify_v2_artifacts.py

echo "🔍 Running V2 inference regression tests in WSL..."
python tests/test_v2_inference.py

echo ""
echo "========================================================================"
echo "🎉 WSL VIRTUAL ENVIRONMENT READY AND ISOLATED ON LINUX EXT4!"
echo "   Path: $TARGET_DIR"
echo "   Activation: source $TARGET_DIR/bin/activate"
echo "========================================================================"
