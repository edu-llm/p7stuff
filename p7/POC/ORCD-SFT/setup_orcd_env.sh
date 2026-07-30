#!/bin/bash
# One-time environment setup for the SocraTeach SFT run on MIT ORCD (Engaging).
# Run this ONCE on a login node (it only installs software; no GPU needed):
#     bash setup_orcd_env.sh
# Then submit training with:  sbatch run_sft.sbatch
set -euo pipefail

ENV_NAME="socrateach"

# --- 1. Install Miniforge into your home dir if it's not already there ---
if [ ! -d "$HOME/miniforge3" ]; then
    echo "Installing Miniforge into $HOME/miniforge3 ..."
    cd "$HOME"
    curl -L -o Miniforge3.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash Miniforge3.sh -b -p "$HOME/miniforge3"
    rm -f Miniforge3.sh
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# --- 2. Create the env (Python 3.11) ---
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# --- 3. Install deps ---
# PyTorch with CUDA (cu121 wheels work on the L40S/H100/H200 nodes).
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -U "transformers>=4.48.0" "datasets>=2.19.0" "accelerate>=0.34.0" "peft>=0.13.0"

# IMPORTANT (gotcha from the Colab run): an old torchao (0.10) breaks the latest peft's
# get_peft_model. We don't use torchao, so make sure it's not installed.
pip uninstall -y torchao 2>/dev/null || true

echo
echo "Environment '$ENV_NAME' ready."
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("CUDA available (expected False on a login node):", torch.cuda.is_available())
PY
echo "Next: sbatch run_sft.sbatch"
