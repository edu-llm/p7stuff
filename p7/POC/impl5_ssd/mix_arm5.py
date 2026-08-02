#!/usr/bin/env python
"""Substitute the δ-fraction of distilled dialogues, attach the replay slot, order it.

δ is assigned at the **dialogue** level, not the turn level (PLAN §8): mixing rewritten and
gold turns inside one dialogue creates prefixes neither the rewriter nor the trainer ever
sees coherently, and dialogue-level assignment keeps realised δ interpretable. (Gate
fallbacks still mix gold turns into distilled dialogues — unavoidable, and precisely why
"realised δ in label tokens" is the reported quantity rather than nominal δ.)

Assignment is seeded and **nested**: D1 ⊂ D2 ⊂ D3 ⊂ D4, so a non-monotone sweep means
something.

The distilled pool is written in the gold pool's row order, and the substitution is
positional, so the seeded shuffle inside ``block_order`` selects **the same dialogues in the
same block positions** as impl4's A1. D4 block *b* and A1 block *b* teach the same problems
in the same order; only the tutor's wording differs. That is what makes the pair tight.

Ordering is Impl 4's 24-pedagogy/8-general block layout, not PLAN §6's stock Impl 2 shuffle
— see ``impl5/config5.py`` for why.

Usage:
    python mix_arm5.py --arm D4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from impl5 import chat5
from impl5._impl4 import manifest, mixing
from impl5.config5 import (
    ARM_CHOICES,
    BASE_MODEL,
    GEN_PER_BLOCK,
    MAX_LEN,
    PED_PER_BLOCK,
    SEED,
    TOKEN_MATCH_TOLERANCE,
    distilled_ids,
    n_blocks,
    resolve_arm,
    slot_sizes,
)
from impl5.paths5 import (
    DISTILL_DIR,
    DISTILL_META,
    DISTILLED_POOL,
    ORCD_DATA_DIR,
    PEDAGOGY_POOL,
    PEDAGOGY_POOL_DIR,
    PEDAGOGY_REFERENCE,
    ensure_dir,
    run_dir,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARM_CHOICES)
    p.add_argument("--gold_pool", default=str(PEDAGOGY_POOL))
    p.add_argument("--distilled_pool", default=str(DISTILLED_POOL))
    p.add_argument("--distill_dir", default=str(DISTILL_DIR))
    p.add_argument("--distill_meta", default=str(DISTILL_META))
    p.add_argument("--general_slot", default=None)
    p.add_argument("--eval_data_dir", default=None,
                   help="Where socrateach_sft_{val,test}.jsonl live. These stay GOLD.")
    p.add_argument("--runs_root", default=None)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--poc", action="store_true")
    p.add_argument("--copy_eval", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> str:
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


def per_turn_verdicts(distill_dir: Path) -> dict[str, list[bool]]:
    """``{dialogue_id: [turn 1 accepted?, turn 2 accepted?, …]}`` from the round files."""
    out: dict[str, dict[int, bool]] = {}
    for path in sorted(Path(distill_dir).glob("round-*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["dialogue_id"], {})[r["round"]] = bool(r["passed"])
    return {k: [v[r] for r in sorted(v)] for k, v in out.items()}


def main():
    args = parse_args()
    arm = resolve_arm(args.arm)
    if arm.external_run:
        raise SystemExit(f"{arm.name} is not built here — it is {arm.external_run}.")
    nb = n_blocks(args.poc)
    need_ped, need_gen = slot_sizes(args.poc)

    out_dir = ensure_dir(run_dir(arm.name, args.runs_root))
    train_path = out_dir / "socrateach_sft_train.jsonl"
    if train_path.exists() and not args.force:
        print(f"{train_path} already present. Use --force to rebuild.")
        return

    gold = manifest.read_jsonl(args.gold_pool)
    dist = manifest.read_jsonl(args.distilled_pool)
    if len(gold) != len(dist):
        raise SystemExit(f"pool mismatch: {len(gold)} gold vs {len(dist)} distilled. The "
                         f"distilled pool must be written in the gold pool's row order.")
    for g, d in zip(gold, dist):
        if g["dialogue_id"] != d["dialogue_id"]:
            raise SystemExit(f"row order diverges at {g['dialogue_id']} vs {d['dialogue_id']}; "
                             f"positional substitution would silently pair wrong dialogues.")

    slot_path = Path(args.general_slot) if args.general_slot else out_dir / "general_slot.jsonl"
    if not slot_path.exists():
        raise SystemExit(f"missing {slot_path}. Run: python build_general_slot5.py "
                         f"--arm {arm.name}")
    general = manifest.read_jsonl(slot_path)
    if len(general) != need_gen:
        raise SystemExit(f"replay slot has {len(general)}, need exactly {need_gen}")

    # -- δ assignment, nested -------------------------------------------------
    chosen = set(distilled_ids([r["dialogue_id"] for r in gold], arm.delta, args.seed))
    pedagogy = [d if g["dialogue_id"] in chosen else g for g, d in zip(gold, dist)]
    print(f"arm {arm.name} | delta={arm.delta} | {len(chosen)}/{len(gold)} dialogues distilled")
    if len(chosen) != int(round(arm.delta * len(gold))):
        raise SystemExit("δ arithmetic is wrong (PLAN §9 check 5)")

    for r in pedagogy[:need_ped]:
        assert any(m["role"] == "system" for m in r["messages"]), "pedagogy row lacks an SI"
    for r in general:
        assert all(m["role"] != "system" for m in r["messages"]), "replay row has an SI"

    ordered = mixing.block_order(pedagogy, general, nb, PED_PER_BLOCK, GEN_PER_BLOCK,
                                 seed=args.seed)
    layout = mixing.verify_block_layout(ordered, PED_PER_BLOCK, GEN_PER_BLOCK)
    print(f"Block layout verified: {layout['n_blocks']} x [{layout['layout']}]")

    # -- token accounting, arm and D0 side by side ----------------------------
    tokenizer = chat5.load_tokenizer(args.base_model)
    tok_fn = chat5.make_tokenize_fn(tokenizer, MAX_LEN)
    counts = chat5.label_token_counts(tok_fn, ordered)
    dead = [i for i, c in enumerate(counts) if c == 0]
    if dead:
        raise SystemExit(f"{len(dead)} examples have 0 unmasked label tokens at "
                         f"max_len={MAX_LEN} ({dict(Counter(ordered[i]['kind'] for i in dead))})"
                         f"; they would occupy a block slot and train on nothing.")
    ped_tokens = sum(c for c, r in zip(counts, ordered) if mixing.is_pedagogy(r))
    gen_tokens = sum(counts) - ped_tokens

    # The same 22,152 dialogues, gold — this is D0's realised pedagogy total, which is what
    # PLAN §5's ratio is defined against. Computed here rather than read from A1's manifest
    # so it is derived from the same tokenizer in the same process.
    used_ids = {r["dialogue_id"] for r in ordered if mixing.is_pedagogy(r)}
    gold_used = [g for g in gold if g["dialogue_id"] in used_ids]
    ped_tokens_gold = sum(chat5.label_token_counts(tok_fn, gold_used))
    ratio_arm = gen_tokens / ped_tokens
    ratio_d0 = gen_tokens / ped_tokens_gold
    drift = abs(ratio_arm - ratio_d0) / ratio_d0
    print(f"pedagogy label tokens: {ped_tokens_gold} gold -> {ped_tokens} this arm "
          f"({ped_tokens / ped_tokens_gold:.3f}x)")
    print(f"general/pedagogy token ratio: D0 {ratio_d0:.4f} -> {arm.name} {ratio_arm:.4f} "
          f"(drift {drift:+.1%})")
    if drift > TOKEN_MATCH_TOLERANCE:
        print(f"  WARNING: drift exceeds the ±{TOKEN_MATCH_TOLERANCE:.0%} tolerance. The δ "
              f"contrast is partly a stream-weight contrast. Report it; do not rescale it "
              f"away. (PLAN §5 would fix this with --token_match at the cost of changing "
              f"which Tulu examples are in the slot.)")

    # -- realised δ in label tokens (PLAN §4's reported quantity) -------------
    verdicts = per_turn_verdicts(Path(args.distill_dir))
    acc = tot = 0
    for row in ordered:
        if not mixing.is_pedagogy(row) or row["dialogue_id"] not in chosen:
            continue
        vs = verdicts.get(row["dialogue_id"], [])
        turns = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
        for i, t in enumerate(turns):
            n = len(tokenizer(t, add_special_tokens=False)["input_ids"]) + 1   # + EOS
            tot += n
            if i < len(vs) and vs[i]:
                acc += n
    delta_tokens = (acc / ped_tokens) if ped_tokens else 0.0
    print(f"realised δ: {len(chosen) / len(gold):.3f} of dialogues, "
          f"{delta_tokens:.3f} of pedagogy label tokens "
          f"({acc} accepted-rewrite tokens of {ped_tokens})")

    n_written = manifest.write_jsonl(train_path, ordered)
    eval_dir = Path(args.eval_data_dir) if args.eval_data_dir else (
        ORCD_DATA_DIR if (ORCD_DATA_DIR / "socrateach_sft_val.jsonl").exists()
        else PEDAGOGY_POOL_DIR)
    modes = {name: link_or_copy(eval_dir / f"socrateach_sft_{name}.jsonl",
                                out_dir / f"socrateach_sft_{name}.jsonl", args.copy_eval)
             for name in ("val", "test")
             if (eval_dir / f"socrateach_sft_{name}.jsonl").exists()}

    dmeta = json.loads(Path(args.distill_meta).read_text()) \
        if Path(args.distill_meta).exists() else {}
    manifest.merge(out_dir, "mix", {
        "train_file": str(train_path),
        "n_train": n_written,
        "n_blocks": nb,
        "optimizer_steps": nb,
        "block_layout": f"{PED_PER_BLOCK} pedagogy + {GEN_PER_BLOCK} general per "
                        f"{PED_PER_BLOCK + GEN_PER_BLOCK}-example optimizer step",
        "delta_nominal": arm.delta,
        "delta_realised_dialogues": round(len(chosen) / len(gold), 4),
        "delta_realised_label_tokens": round(delta_tokens, 4),
        "delta_note": ("Realised δ in label tokens is the fraction of pedagogy label tokens "
                       "that come from an ACCEPTED rewrite. Gate fallbacks put gold turns "
                       "back inside distilled dialogues, so it is strictly below nominal δ. "
                       "Per-turn token counts ignore the max_len truncation, which touches "
                       "<1% of pedagogy examples."),
        "n_distilled_dialogues": len(chosen),
        "nested_note": "Seeded permutation sliced by δ, so D1 ⊂ D2 ⊂ D3 ⊂ D4.",
        "kinds": dict(Counter(r["kind"] for r in ordered)),
        "example_ratio_general": round(need_gen / n_written, 4),
        "label_tokens": {"pedagogy": ped_tokens, "general": gen_tokens,
                         "total": ped_tokens + gen_tokens,
                         "pedagogy_gold_same_dialogues": ped_tokens_gold},
        "token_ratio_general": round(gen_tokens / (ped_tokens + gen_tokens), 4),
        "ratio_general_to_pedagogy": round(ratio_arm, 6),
        "ratio_general_to_pedagogy_D0": round(ratio_d0, 6),
        "ratio_drift_vs_D0": round(drift, 4),
        "ratio_within_tolerance": bool(drift <= TOKEN_MATCH_TOLERANCE),
        "pedagogy_token_ratio_vs_gold": round(ped_tokens / ped_tokens_gold, 4),
        "layout_check": layout,
        "eval_files": modes,
        "eval_files_are_gold": True,
        "gate": dmeta.get("gate"),
        "distill_sampling": dmeta.get("sampling"),
        "fallback_rate_by_turn_index": dmeta.get("fallback_rate_by_turn_index"),
        "seed": args.seed,
    })

    # D0's realised pedagogy total, for a later --token_match build (PLAN §5).
    ensure_dir(Path(PEDAGOGY_REFERENCE).parent)
    Path(PEDAGOGY_REFERENCE).write_text(json.dumps({
        "arm": "D0 (= impl4-A1)", "n_dialogues": len(gold_used),
        "pedagogy_tokens_D0": ped_tokens_gold,
        "pedagogy_tokens_this_arm": ped_tokens,
        "general_tokens": gen_tokens,
        "ratio_general_to_pedagogy": ratio_d0,
        "max_len": MAX_LEN, "base_model": args.base_model, "seed": args.seed,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {n_written} ordered examples -> {train_path}")
    print(f"  general share: {need_gen / n_written:.1%} of examples, "
          f"{gen_tokens / (ped_tokens + gen_tokens):.1%} of label tokens")
    print(f"  eval files (gold): {modes}")
    block = PED_PER_BLOCK + GEN_PER_BLOCK
    for b in range(min(3, nb)):
        kinds = "".join("P" if mixing.is_pedagogy(r) else "g"
                        for r in ordered[b * block:(b + 1) * block])
        print(f"  block {b}: {kinds}")


if __name__ == "__main__":
    main()
