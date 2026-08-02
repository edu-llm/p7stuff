"""Filesystem layout for Impl 5.

Redefined locally rather than imported (PLAN §10): the layout is Impl 5's, and it is
trivial. Everything *shared* with Impl 4 — the pedagogy pool, the decontamination targets,
the Impl 2 trainer — is reached through ``impl4.paths`` so there is exactly one definition
of where those live.

    POC/
      impl4_ssd/data/pedagogy_pool/     <- the gold pool, built by Impl 4, read-only here
      impl5_ssd/                        <- IMPL5_ROOT
        data/
          distill/round-<r>.jsonl       <- resumable per-round rewriting output
          distilled_pool.jsonl          <- the shared distillation pass, all 22,500 rows
          distill_meta.json             <- gate rates, realised δ, sampling config
          pedagogy_reference.json       <- D0's realised pedagogy label-token total
        runs/<arm>/
          general_slot.jsonl
          socrateach_sft_train.jsonl
          socrateach_sft_{val,test}.jsonl   <- gold, never distilled
          manifest.json
          ckpt-<step>/
          train.log
"""

from __future__ import annotations

import os
from pathlib import Path

from ._impl4 import paths as paths4

IMPL5_ROOT = Path(__file__).resolve().parents[1]
POC_ROOT = IMPL5_ROOT.parent

# Shared, owned by Impl 4 / the POC. Read-only from here.
PEDAGOGY_POOL_DIR = paths4.PEDAGOGY_POOL_DIR
PEDAGOGY_POOL = PEDAGOGY_POOL_DIR / "socrateach_sft_train.jsonl"
ORCD_DATA_DIR = paths4.ORCD_DATA_DIR
MATH_EVAL_PROMPTS = paths4.MATH_EVAL_PROMPTS
GENERAL_EVAL_PROMPTS = paths4.GENERAL_EVAL_PROMPTS

# Impl 5 outputs.
DATA_DIR = IMPL5_ROOT / "data"
DISTILL_DIR = DATA_DIR / "distill"
DISTILLED_POOL = DATA_DIR / "distilled_pool.jsonl"
DISTILL_META = DATA_DIR / "distill_meta.json"
PEDAGOGY_REFERENCE = DATA_DIR / "pedagogy_reference.json"
GATE_CALIBRATION = DATA_DIR / "gate_calibration.json"
RUNS_DIR = IMPL5_ROOT / "runs"


def round_file(r: int, distill_dir: str | os.PathLike | None = None) -> Path:
    return Path(distill_dir or DISTILL_DIR) / f"round-{r}.jsonl"


def run_dir(arm: str, runs_root: str | os.PathLike | None = None) -> Path:
    return (Path(runs_root) if runs_root else RUNS_DIR) / arm


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
