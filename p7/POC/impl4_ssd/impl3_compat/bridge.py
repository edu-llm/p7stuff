#!/usr/bin/env python
"""Expose Impl 4's checkpoints in the layout Impl 3's eval driver discovers.

``eval/sweep_ckpt_eval.py`` scans ``out/<run>/checkpoint-<step>/`` and reads
``trainer_state.json`` to enforce ``epoch >= 0.99`` — the guard that exists because two of
their runs crashed at ~0.28 epoch and got graded as if complete (their pitfall #5). Impl 4
writes ``runs/<arm>/ckpt-<step>/`` and its grid callback writes no ``trainer_state.json`` at
all, so the two layouts do not meet.

This builds the view they expect:

    <workdir>/out/impl4-<ARM>/
      checkpoint-<step>/           one real dir per grid point, files symlinked from runs/
      checkpoint-<final>/trainer_state.json    synthesized, see below

**The epoch guard is honoured, not defeated.** ``trainer_state.json`` is written only when
``checkpoint_index.json`` shows the run actually reached its final grid step; an arm that died
mid-training is skipped and named. Writing ``{"epoch": 1.0}`` unconditionally would silently
re-open exactly the hole their guard was added to close.

Files are symlinked so nothing is duplicated and ``runs/`` is never written to. Use ``--copy``
if the eval box cannot follow symlinks into wherever ``runs/`` lives.

Usage:
    python impl3_compat/bridge.py
    python impl3_compat/bridge.py --arms A1 --runs_root /content/drive/MyDrive/impl4_ssd/runs/full
    python impl3_compat/bridge.py --steps 1,2,3,4,8,16,32,64,128,256,512,923   # his grid only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPL4_ROOT = HERE.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_root", default=str(IMPL4_ROOT / "runs"),
                   help="Impl 4's runs root (the dir holding <arm>/ckpt-*).")
    p.add_argument("--workdir", default=str(HERE / "work"))
    p.add_argument("--arms", nargs="*", default=None,
                   help="Arms to expose (default: every arm dir with a checkpoint_index.json).")
    p.add_argument("--prefix", default="impl4-",
                   help="Run-name prefix in the results file, e.g. impl4-A3.")
    p.add_argument("--steps", default=None,
                   help="Comma-separated subset of steps to expose (default: all saved).")
    p.add_argument("--copy", action="store_true", help="Copy adapter files instead of symlinking.")
    p.add_argument("--allow_incomplete", action="store_true",
                   help="Expose runs that did not reach their final grid step. Their numbers are "
                        "not comparable — Impl 3's epoch guard exists for this case.")
    p.add_argument("--force", action="store_true", help="Rebuild checkpoint dirs that exist.")
    return p.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    work = Path(args.workdir).expanduser().resolve()
    out_root = work / "out"
    if not (work / "eval" / "sweep_ckpt_eval.py").exists():
        raise SystemExit(f"{work} is not an assembled compat workdir. Run setup_compat.py first.")
    if not runs_root.is_dir():
        raise SystemExit(f"no such runs root: {runs_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    arms = args.arms or sorted(
        d.name for d in runs_root.iterdir()
        if d.is_dir() and (d / "checkpoint_index.json").exists())
    if not arms:
        raise SystemExit(
            f"no arms with a checkpoint_index.json under {runs_root}. Train one first "
            f"(train_sft_impl4.py writes it at the end of a run).")
    want_steps = {int(s) for s in args.steps.split(",")} if args.steps else None

    print(f"runs root : {runs_root}")
    print(f"workdir   : {work}")
    print(f"arms      : {', '.join(arms)}")

    exposed, skipped = {}, {}
    for arm in arms:
        arm_dir = runs_root / arm
        idx_path = arm_dir / "checkpoint_index.json"
        if not idx_path.exists():
            skipped[arm] = "no checkpoint_index.json (has this arm finished training?)"
            continue
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        grid = list(idx.get("checkpoint_grid") or [])
        saved = set(idx.get("checkpoints_saved") or [])
        steps_run = idx.get("steps")

        # Completion test, standing in for Impl 3's epoch >= 0.99 filter: the final grid point
        # must be on disk. A crashed run has an index but not its last checkpoint.
        final = max((s for s in grid if steps_run is None or s <= steps_run), default=None)
        complete = final is not None and final in saved
        if not complete and not args.allow_incomplete:
            skipped[arm] = (f"did not reach its final grid step ({final} not in checkpoints_saved) "
                            f"— pass --allow_incomplete to grade it anyway")
            continue

        run_name = f"{args.prefix}{arm}"
        run_out = out_root / run_name
        run_out.mkdir(parents=True, exist_ok=True)
        steps = sorted(s for s in saved if want_steps is None or s in want_steps)
        linked, missing, n_new = [], [], 0
        for step in steps:
            src = arm_dir / f"ckpt-{step}"
            if not any(src.glob("adapter_model*")):
                # The index claims this step was saved but there is no adapter on disk.
                missing.append(step)
                continue
            dst = run_out / f"checkpoint-{step}"
            if not (dst.exists() and not args.force):
                dst.mkdir(parents=True, exist_ok=True)
                for f in sorted(src.iterdir()):
                    if f.is_file():
                        link_or_copy(f, dst / f.name, args.copy)
                n_new += 1
            linked.append(step)

        if not linked:
            skipped[arm] = (f"checkpoint_index.json lists {len(steps)} saved steps but none have "
                            f"adapter files on disk under {arm_dir}")
            continue

        # trainer_state.json goes only in the final checkpoint — final_epoch() reads the
        # highest-numbered one, and claiming epoch 1.0 at step 4 would be a lie on disk. It is
        # written only when that checkpoint's adapter is actually present, so an index that
        # over-reports cannot smuggle a run past Impl 3's epoch filter.
        if complete and final in linked:
            (run_out / f"checkpoint-{final}" / "trainer_state.json").write_text(json.dumps({
                "epoch": 1.0,
                "global_step": final,
                "_note": ("Synthesized by impl3_compat/bridge.py for Impl 3's epoch>=0.99 "
                          "filter. Written only because checkpoint_index.json shows this run "
                          "reached its final grid step AND that adapter is on disk."),
            }, indent=2) + "\n", encoding="utf-8")
        elif not args.allow_incomplete:
            skipped[arm] = (f"final grid step {final} has no adapter on disk, so the run cannot "
                            f"be shown to have completed")
            shutil.rmtree(run_out, ignore_errors=True)
            continue

        exposed[run_name] = linked
        note = f" | index over-reports {len(missing)} step(s): {missing[:5]}" if missing else ""
        print(f"  {run_name}: {len(linked)} checkpoints ({n_new} new) | final={final}"
              f" | steps={linked[:6]}{'...' if len(linked) > 6 else ''}{note}")

    for arm, why in skipped.items():
        print(f"  SKIP {arm}: {why}")

    if not exposed:
        raise SystemExit("nothing exposed — no completed arms found.")

    (work / "bridge_index.json").write_text(json.dumps({
        "runs_root": str(runs_root),
        "exposed": exposed,
        "skipped": skipped,
        "linked": not args.copy,
    }, indent=2) + "\n", encoding="utf-8")

    total = sum(len(v) for v in exposed.values())
    print(f"\n{total} checkpoints across {len(exposed)} runs exposed under {out_root}")
    print("\nNext:")
    print(f"  cd {work} && python eval/sweep_ckpt_eval.py --runs 'out/*' "
          f"--out out/ckpt_sweep_impl4.jsonl --batch 32")


if __name__ == "__main__":
    main()
