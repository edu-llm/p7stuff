#!/bin/bash
# Environment for Impl 4 on MIT ORCD (Engaging). Extends ORCD-SFT/setup_orcd_env.sh
# with vLLM (PLAN §4) into the SAME `socrateach` conda env, so training stays
# bit-comparable with Impl 2 / curve_run.
#
# Run ONCE on a login node (installs software only, no GPU needed):
#     bash setup_env.sh
# Then:
#     ARM=A1 sbatch run_arm.sbatch
set -euo pipefail

ENV_NAME="${ENV_NAME:-socrateach}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. base env (idempotent; reuses the Impl 2 setup verbatim) -------------
bash "$HERE/../ORCD-SFT/setup_orcd_env.sh"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- 2. vLLM for generation --------------------------------------------------
# Optional: build_general_slot.py falls back to batched HF `generate` (~20-30 min on
# the L40S vs. a few minutes) when vLLM is absent, so a failure here is not fatal.
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM ..."
    pip install "vllm>=0.6.0" || {
        echo "WARNING: vLLM install failed. build_general_slot.py --backend hf still works."
    }
fi

# --- 3. nothing else -----------------------------------------------------------
# ROUGE-L (B2's gate) and the 13-gram decontamination check are implemented in
# impl4/gate.py and impl4/ngram.py with the stdlib — no rouge-score, no nltk.

echo
python - <<'PY'
import importlib
for m in ("torch", "transformers", "datasets", "peft", "vllm"):
    try:
        print(f"  {m:14s} {getattr(importlib.import_module(m), '__version__', '?')}")
    except Exception:
        print(f"  {m:14s} MISSING")
import torch
print("  CUDA available (expected False on a login node):", torch.cuda.is_available())
PY

echo
echo "Environment '$ENV_NAME' ready for Impl 4."
echo "Recommended next step — clone SuperNI once so every arm reads it locally:"
echo "    git clone --depth 1 https://github.com/allenai/natural-instructions \$HOME/natural-instructions"
echo "    export SUPERNI_DIR=\$HOME/natural-instructions"
echo "Then:  ARM=A1 sbatch run_arm.sbatch"
