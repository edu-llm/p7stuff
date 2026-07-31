#!/usr/bin/env python
"""Step 1 (PLAN §8.1) — regenerate the pedagogy pool, shared by every arm.

``ORCD-SFT/data/socrateach_sft_train.jsonl`` is absent from the tree (only ``_val``
and ``_test`` are present, and the train split is gitignored), so it has to be
rebuilt before anything else. This wraps:

    python socrateach_sft/prepare_socrateach_sft.py \\
        --out_dir <pool> --seed 13 --general_frac 0 --max_total 22500

``--seed 13`` is required: it reproduces Impl 2's problem-grouped split so val/test
stay comparable. ``--general_frac 0`` yields pedagogy-only — Impl 4 owns the general
slot and builds it per arm.

Usage:
    python build_pedagogy_pool.py
    python build_pedagogy_pool.py --force        # rebuild even if the pool exists
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from impl4 import manifest
from impl4.config import PED_POOL_TARGET, SEED
from impl4.paths import PEDAGOGY_POOL_DIR, PREPARE_SOCRATEACH_PY, ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=str(PEDAGOGY_POOL_DIR))
    p.add_argument("--seed", type=int, default=SEED,
                   help="Must stay 13 to reproduce Impl 2's problem-grouped split.")
    p.add_argument("--max_total", type=int, default=PED_POOL_TARGET)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    train_path = out_dir / "socrateach_sft_train.jsonl"

    if train_path.exists() and not args.force:
        n = sum(1 for _ in open(train_path, encoding="utf-8"))
        print(f"Pedagogy pool already present: {train_path} ({n} examples). "
              f"Use --force to rebuild.")
        return

    if args.seed != SEED:
        print(f"WARNING: --seed {args.seed} != {SEED}. The Impl 2 problem-grouped split "
              f"will not be reproduced and val/test are no longer comparable.",
              file=sys.stderr)

    cmd = [
        sys.executable, str(PREPARE_SOCRATEACH_PY),
        "--out_dir", str(out_dir),
        "--seed", str(args.seed),
        "--general_frac", "0",
        "--max_total", str(args.max_total),
    ]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    rows = manifest.read_jsonl(train_path)
    kinds = {}
    for r in rows:
        kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
    assert set(kinds) == {"pedagogy"}, f"pool must be pedagogy-only, got {kinds}"
    n_sys = sum(1 for r in rows if any(m["role"] == "system" for m in r["messages"]))
    assert n_sys == len(rows), (
        f"{len(rows) - n_sys} pedagogy records lack a system message "
        f"(PLAN §11 check 3)"
    )

    print(f"\nPedagogy pool ready: {len(rows)} examples, all with a system instruction.")
    if len(rows) < PED_POOL_TARGET:
        print(f"NOTE: pool is {len(rows)} < {PED_POOL_TARGET}; the full 937-block mix needs "
              f"22,488. Reduce --n_blocks in mix_and_order.py or widen the source.")
    print(f"Files: {sorted(p.name for p in Path(out_dir).glob('*.jsonl'))}")


if __name__ == "__main__":
    main()
