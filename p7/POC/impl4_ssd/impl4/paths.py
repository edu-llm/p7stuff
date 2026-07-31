"""Filesystem layout for Impl 4.

Everything is resolved relative to this file so the scripts work from any cwd:

    POC/                                  <- POC_ROOT
      ORCD-SFT/train_sft.py               <- the Impl 2 trainer we reuse/patch
      socrateach_sft/prepare_socrateach_sft.py
      math_eval/math_logic_prompts.jsonl  <- decontamination target
      general_eval/general_prompts.jsonl  <- decontamination target
      impl4_ssd/                          <- IMPL4_ROOT
        data/
          pedagogy_pool/                  <- shared, built once
          superni_cache/                  <- per-task metadata + instance samples
          superni_pool.jsonl              <- shared filtered prompt pool
          tulu_reference.json             <- A1 token budget every arm matches
        shared/
          superni_train_task_ids.txt
          superni_heldout_prompts.jsonl
        runs/<arm>/
          general_slot.jsonl
          socrateach_sft_train.jsonl
          socrateach_sft_{val,test}.jsonl
          manifest.json
          ckpt-<step>/
          train.log
"""

from __future__ import annotations

import os
from pathlib import Path

IMPL4_ROOT = Path(__file__).resolve().parents[1]
POC_ROOT = IMPL4_ROOT.parent

# Reused Impl 2 assets.
ORCD_SFT_DIR = POC_ROOT / "ORCD-SFT"
TRAIN_SFT_PY = ORCD_SFT_DIR / "train_sft.py"
ORCD_DATA_DIR = ORCD_SFT_DIR / "data"
PREPARE_SOCRATEACH_PY = POC_ROOT / "socrateach_sft" / "prepare_socrateach_sft.py"

# Decontamination targets (read-only; owned by the eval team).
MATH_EVAL_PROMPTS = POC_ROOT / "math_eval" / "math_logic_prompts.jsonl"
GENERAL_EVAL_PROMPTS = POC_ROOT / "general_eval" / "general_prompts.jsonl"

# Impl 4 outputs.
DATA_DIR = IMPL4_ROOT / "data"
PEDAGOGY_POOL_DIR = DATA_DIR / "pedagogy_pool"
SUPERNI_CACHE_DIR = DATA_DIR / "superni_cache"
SUPERNI_POOL = DATA_DIR / "superni_pool.jsonl"
SUPERNI_POOL_META = DATA_DIR / "superni_pool_meta.json"
TULU_REFERENCE = DATA_DIR / "tulu_reference.json"
SUPERNI_GOLD_REFERENCE = DATA_DIR / "superni_gold_reference.json"
SHARED_DIR = IMPL4_ROOT / "shared"
RUNS_DIR = IMPL4_ROOT / "runs"


def run_dir(arm: str, runs_root: str | os.PathLike | None = None) -> Path:
    """Directory holding one arm's data, manifest and checkpoints."""
    root = Path(runs_root) if runs_root else RUNS_DIR
    return root / arm


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
