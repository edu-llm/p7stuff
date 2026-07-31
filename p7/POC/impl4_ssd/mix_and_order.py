#!/usr/bin/env python
"""Step 3 (PLAN §8.3) — mix the pedagogy pool with one arm's replay slot and order it.

Writes ``runs/<arm>/socrateach_sft_train.jsonl`` as repeating 32-example blocks of
**24 pedagogy then 8 general** (PLAN §6). With ``SequentialSampler`` and
``per_device_batch=8`` the micro-batches are consecutive slices, so positions 0-23
are three pedagogy micro-batches and 24-31 is the general one — the replay stream
becomes a per-step constraint instead of an in-expectation one.

Also links ``socrateach_sft_{val,test}.jsonl`` into the run dir, because
``train_sft.py``'s ``build_datasets`` requires all three files side by side.

Usage:
    python mix_and_order.py --arm A3
    python mix_and_order.py --arm A3 --poc
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

from impl4 import chat, manifest, mixing
from impl4.config import (
    ALL_ARMS,
    ARM_CHOICES,
    BASE_MODEL,
    GEN_PER_BLOCK,
    MAX_LEN,
    PED_PER_BLOCK,
    SEED,
    n_blocks,
    resolve_arm,
    slot_sizes,
)
from impl4.paths import ORCD_DATA_DIR, PEDAGOGY_POOL_DIR, run_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARM_CHOICES,
                   help=f"One of {', '.join(ALL_ARMS)} (T1 is an alias of A3).")
    p.add_argument("--pedagogy_pool", default=str(PEDAGOGY_POOL_DIR / "socrateach_sft_train.jsonl"))
    p.add_argument("--general_slot", default=None,
                   help="Defaults to runs/<arm>/general_slot.jsonl.")
    p.add_argument("--eval_data_dir", default=str(ORCD_DATA_DIR),
                   help="Where socrateach_sft_{val,test}.jsonl live.")
    p.add_argument("--runs_root", default=None)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--poc", action="store_true")
    p.add_argument("--copy_eval", action="store_true",
                   help="Copy val/test instead of symlinking them.")
    p.add_argument("--skip_token_stats", action="store_true",
                   help="Write the ordered file without loading the tokenizer. The "
                        "example ratio and block layout are still verified; the token "
                        "ratio is left null in the manifest. For login nodes with no "
                        "model cache — the real token accounting already happened in "
                        "build_general_slot.py.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> str:
    """Symlink ``src`` into the run dir, falling back to a copy.

    The relative link is computed from *resolved* paths and then checked, because a
    symlinked ancestor (``/tmp`` -> ``/private/tmp`` on macOS) makes a naive relpath
    resolve to nothing — which would only surface later as a confusing
    "missing prepared data files" from the trainer.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if not copy:
        try:
            dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))
            if dst.exists():
                return "symlink"
            dst.unlink()
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def main():
    args = parse_args()
    arm = resolve_arm(args.arm)
    nb = n_blocks(args.poc)
    need_ped, need_gen = slot_sizes(args.poc)

    out_dir = run_dir(arm.name, args.runs_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Idempotent, and it makes every stage able to seed the manifest header — otherwise
    # a run assembled with an out-of-band --general_slot would end up with a manifest
    # that records the mix but not sigma/delta/the sampling config.
    manifest.init(out_dir, arm, poc=args.poc)
    train_path = out_dir / "socrateach_sft_train.jsonl"
    if train_path.exists() and not args.force:
        print(f"{train_path} already present. Use --force to rebuild.")
        return

    ped_path = Path(args.pedagogy_pool)
    if not ped_path.exists():
        raise SystemExit(f"missing pedagogy pool {ped_path}. Run: python build_pedagogy_pool.py")
    slot_path = Path(args.general_slot) if args.general_slot else out_dir / "general_slot.jsonl"
    if not slot_path.exists():
        raise SystemExit(
            f"missing {slot_path}. Run: python build_general_slot.py --arm {arm.name}")

    pedagogy = manifest.read_jsonl(ped_path)
    general = manifest.read_jsonl(slot_path)
    print(f"pedagogy pool {len(pedagogy)} (need {need_ped}) | "
          f"replay slot {len(general)} (need {need_gen}) | {nb} blocks")

    if len(pedagogy) < need_ped:
        raise SystemExit(
            f"pedagogy pool has {len(pedagogy)}, need {need_ped}. Rebuild with a larger "
            f"--max_total, or lower the block count with --poc.")
    if len(general) != need_gen:
        raise SystemExit(
            f"replay slot has {len(general)}, need exactly {need_gen}. Rebuild it with the "
            f"same --poc setting as this call.")

    # PLAN §11 check 3, both directions.
    for r in pedagogy[:need_ped]:
        assert any(m["role"] == "system" for m in r["messages"]), \
            f"pedagogy record {r.get('dialogue_id')} has no system message"
    for r in general:
        assert all(m["role"] != "system" for m in r["messages"]), \
            f"general record {r.get('dialogue_id')} has a system message"

    ordered = mixing.block_order(pedagogy, general, nb, PED_PER_BLOCK, GEN_PER_BLOCK,
                                 seed=args.seed)
    layout = mixing.verify_block_layout(ordered, PED_PER_BLOCK, GEN_PER_BLOCK)
    print(f"Block layout verified: {layout['n_blocks']} x [{layout['layout']}]")

    if args.skip_token_stats:
        ped_tokens = gen_tokens = total_tokens = None
    else:
        tokenizer = chat.load_tokenizer(args.base_model)
        tok_fn = chat.make_tokenize_fn(tokenizer, MAX_LEN)
        counts = chat.label_token_counts(tok_fn, ordered)
        # A record whose prompt alone exceeds max_len trains on nothing but still
        # occupies a slot in its block. The trainer refuses to drop it (that would
        # break the layout), so it has to be caught here.
        dead = [i for i, c in enumerate(counts) if c == 0]
        if dead:
            kinds = Counter(ordered[i]["kind"] for i in dead)
            raise SystemExit(
                f"{len(dead)} of {len(ordered)} examples have 0 unmasked label tokens at "
                f"max_len={MAX_LEN} ({dict(kinds)}). They would train on nothing while "
                f"still consuming a block slot, diluting the stream ratio. Rebuild the "
                f"replay slot (build_general_slot.py drops these) or shorten the prompts."
            )
        ped_tokens = sum(c for c, r in zip(counts, ordered) if mixing.is_pedagogy(r))
        gen_tokens = sum(counts) - ped_tokens
        total_tokens = ped_tokens + gen_tokens

    n_written = manifest.write_jsonl(train_path, ordered)
    modes = {
        name: link_or_copy(Path(args.eval_data_dir) / f"socrateach_sft_{name}.jsonl",
                           out_dir / f"socrateach_sft_{name}.jsonl", args.copy_eval)
        for name in ("val", "test")
        if (Path(args.eval_data_dir) / f"socrateach_sft_{name}.jsonl").exists()
    }

    section = {
        "train_file": str(train_path),
        "n_train": n_written,
        "n_blocks": nb,
        "block_layout": f"{PED_PER_BLOCK} pedagogy + {GEN_PER_BLOCK} general per "
                        f"{PED_PER_BLOCK + GEN_PER_BLOCK}-example optimizer step",
        "optimizer_steps": nb,
        "pedagogy_pool": str(ped_path),
        "general_slot": str(slot_path),
        "kinds": dict(Counter(r["kind"] for r in ordered)),
        "example_ratio_general": round(need_gen / n_written, 4),
        "token_ratio_general": (round(gen_tokens / total_tokens, 4)
                                if total_tokens else None),
        "label_tokens": {"pedagogy": ped_tokens, "general": gen_tokens,
                         "total": total_tokens},
        "layout_check": layout,
        "eval_files": modes,
        "seed": args.seed,
        "sampler_note": (
            "Requires SequentialSampler + dataloader_drop_last=True + group_by_length=False. "
            "train_sft_impl4.py sets all three; the stock ORCD-SFT/train_sft.py does not and "
            "would shuffle this layout away."
        ),
    }
    manifest.merge(out_dir, "mix", section)

    print(f"\nWrote {n_written} ordered examples -> {train_path}")
    tok_share = (f"{section['token_ratio_general']:.1%} of label tokens"
                 if section["token_ratio_general"] is not None
                 else "label tokens not counted (--skip_token_stats)")
    print(f"  general share: {section['example_ratio_general']:.1%} of examples, {tok_share}")
    print(f"  optimizer steps: {nb}")
    print(f"  eval files: {modes}")

    print("\nFirst 3 blocks (PLAN §11 check 5):")
    block = PED_PER_BLOCK + GEN_PER_BLOCK
    for b in range(min(3, nb)):
        kinds = [("P" if mixing.is_pedagogy(r) else "g")
                 for r in ordered[b * block:(b + 1) * block]]
        print(f"  block {b}: {''.join(kinds)}")


if __name__ == "__main__":
    main()
