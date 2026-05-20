#!/usr/bin/env bash
# Phase 4 VM setup — run once on a fresh cloud GPU instance.
# Tested on: Ubuntu 22.04, CUDA 12.x, A100/H100 80GB
# Runtime: ~15-25 min (mamba-ssm compiles from source)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="nemotron-train"
PYTHON_VERSION="3.11"

echo "=================================================="
echo " Nemotron Phase 4 — VM Setup"
echo " Repo:  $REPO_DIR"
echo " Env:   $ENV_NAME"
echo "=================================================="

# ── 1. System deps ──────────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -qq && apt-get install -y -qq git wget curl build-essential

# ── 2. Conda ────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "[2/7] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
else
    echo "[2/7] Conda already installed — skipping."
    eval "$(conda shell.bash hook)"
fi

# ── 3. Conda env + PyTorch ──────────────────────────────
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[3/7] Conda env '$ENV_NAME' already exists — skipping create."
else
    echo "[3/7] Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# Detect CUDA version from nvcc or nvidia-smi
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[\d.]+" | head -1 || \
           nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[\d.]+" | head -1 || \
           echo "12.1")
CUDA_SHORT=$(echo "$CUDA_VER" | grep -oP "^\d+\.\d+")
CUDA_TAG="cu$(echo "$CUDA_SHORT" | tr -d '.')"
echo "[3/7] Detected CUDA $CUDA_SHORT → installing torch for $CUDA_TAG"

conda run -n "$ENV_NAME" pip install -q \
    "torch>=2.1" torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# ── 4. Unsloth ──────────────────────────────────────────
echo "[4/7] Installing unsloth..."
conda run -n "$ENV_NAME" pip install -q "unsloth[colab-new]"

# ── 5. mamba-ssm + causal-conv1d (compiled from source) ─
echo "[5/7] Compiling mamba-ssm + causal-conv1d from source..."
echo "      (This takes 10-20 min — nvcc must be in PATH)"
which nvcc || { echo "ERROR: nvcc not found. Install CUDA toolkit first."; exit 1; }

conda run -n "$ENV_NAME" pip install -q \
    "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git" \
    --no-build-isolation

conda run -n "$ENV_NAME" pip install -q \
    "mamba-ssm @ git+https://github.com/state-spaces/mamba.git" \
    --no-build-isolation

# ── 6. Remaining deps ───────────────────────────────────
echo "[6/7] Installing training deps..."
conda run -n "$ENV_NAME" pip install -q \
    transformers datasets peft trl accelerate \
    safetensors python-dotenv pyyaml pandas

# ── 7. Smoke check ──────────────────────────────────────
echo "[7/7] Smoke check..."
conda run -n "$ENV_NAME" python -c "
from unsloth import FastLanguageModel
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
import torch
print(f'  unsloth OK')
print(f'  mamba-ssm OK')
print(f'  torch {torch.__version__}  cuda {torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory//1024**3} GB')
"

echo ""
echo "=================================================="
echo " Setup complete. Next: run ./run_training.sh"
echo "=================================================="
