"""``manifest.json`` — the per-arm deliverable that makes a run readable (PLAN §10).

The manifest is assembled incrementally: ``build_general_slot.py`` writes the
``general_slot`` section, ``mix_and_order.py`` the ``mix`` section,
``probe_loss_norm.py`` the ``loss_normalization`` section, ``acceptance_checks.py``
the ``acceptance`` section, and the trainer the ``training`` section. Each writer
merges into whatever is already on disk, so partial reruns do not lose fields.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from .config import (
    BASE_MODEL,
    GENERAL_FRAC,
    SEED,
    Arm,
    checkpoint_grid,
    priority_checkpoints,
)
from .paths import IMPL4_ROOT

MANIFEST_NAME = "manifest.json"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(IMPL4_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _version(pkg: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(pkg)
    except Exception:
        return None


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "transformers": _version("transformers"),
        "torch": _version("torch"),
        "peft": _version("peft"),
        "datasets": _version("datasets"),
        "vllm": _version("vllm"),
        "git_commit": _git_commit(),
    }


def base_manifest(arm: Arm, poc: bool = False) -> dict:
    """The fields that are known before anything is built."""
    sc = arm.sampling_config
    grid = checkpoint_grid(poc)
    return {
        "arm": arm.name,
        "aliases": list(arm.aliases),
        "block": arm.block,
        "question": arm.question,
        "poc": poc,
        "base_model": BASE_MODEL,
        "sigma": arm.sigma,
        "delta": arm.delta,
        "delta_note": (
            "Pedagogy targets are never self-distilled. No tutor-turn rewriting, no "
            "teacher-forced generation over SocraTeach, no pedagogy quality gate."
        ),
        "general_frac": GENERAL_FRAC,
        "gold_source": arm.gold_source,
        "gated": arm.gated,
        "sampling": sc.as_dict() if sc else None,
        "seed": SEED,
        "checkpoint_grid": list(grid),
        "priority_checkpoints": list(priority_checkpoints(arm, poc)),
        "priority_note": (
            "Deliberate coverage cap, NOT complete coverage: all 11 points for Block S "
            "(the curve comparison needs the trajectory), {20,160,937} for Blocks T and G "
            "(those need 'where does this arm land'). All 11 are saved regardless."
        ),
        "warmup_note": (
            "warmup_ratio=0.03 x 937 ~= 28 steps, so the 5/10/20 checkpoints sit INSIDE "
            "warmup and are not on the cosine schedule proper. That is intentional (it is "
            "where the damage happens) — do not read them as points on the schedule."
        ),
        "config_note": (
            "A4 and B2 are built at the T1 sampling config because something had to be "
            "chosen before Block T resolves. If a truncated arm wins Block T, both should "
            "ideally be re-run at the winner (+2 runs)."
        ) if arm.name in ("A4", "B2") else None,
        "environment": environment(),
    }


def load(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_NAME
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def merge(run_dir: str | Path, section: str, payload: Any) -> dict:
    """Merge one section into ``<run_dir>/manifest.json`` and write it back."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / MANIFEST_NAME
    data = load(path)
    data[section] = payload
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def init(run_dir: str | Path, arm: Arm, poc: bool = False) -> dict:
    """Create the manifest if absent, refreshing the static header either way."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / MANIFEST_NAME
    data = load(path)
    data.update(base_manifest(arm, poc=poc))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def write_jsonl(path: str | Path, rows) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
