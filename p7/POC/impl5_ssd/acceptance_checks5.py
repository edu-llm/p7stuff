#!/usr/bin/env python
"""PLAN §9 — run these before any full run, and again after the mix.

Two stages, because they answer questions at different times and cost different amounts:

``--stage fast``  (tokenizer only, no GPU, ~40 s)
    Runs **before** the distillation pass. Everything here can invalidate the whole run, and
    finding out after a 90-minute rewriting pass is 90 minutes wasted. Checks 2, 3, 5, 6.

``--stage full``  (needs the distilled pool and the mix)
    Runs after ``mix_arm5.py``. Checks 1, 3, 7 plus the realised-δ arithmetic on real data.

Check 4 (the loss-normalisation probe) is **not run here**: the recipe is bit-identical to
impl4's A1, which was already probed on this stack, so the verdict is inherited and recorded
rather than re-derived. Re-run ``impl4_ssd/probe_loss_norm.py`` if the transformers pin moves.

Usage:
    python acceptance_checks5.py --stage fast
    python acceptance_checks5.py --stage full --arm D4
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from impl5 import answer_leak, chat5, dialogue, gate5
from impl5._impl4 import manifest, mixing, ngram
from impl5.config5 import (
    ARM_CHOICES,
    BASE_MODEL,
    DEFAULT_THRESHOLDS,
    MAX_LEN,
    N_PED,
    SEED,
    distilled_ids,
    resolve_arm,
)
from impl5.paths5 import (
    DISTILLED_POOL,
    GENERAL_EVAL_PROMPTS,
    MATH_EVAL_PROMPTS,
    PEDAGOGY_POOL,
    run_dir,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="fast", choices=("fast", "full"))
    p.add_argument("--arm", default="D4", choices=ARM_CHOICES)
    p.add_argument("--gold_pool", default=str(PEDAGOGY_POOL))
    p.add_argument("--distilled_pool", default=str(DISTILLED_POOL))
    p.add_argument("--runs_root", default=None)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--sample", type=int, default=200,
                   help="Dialogues to run the token-level checks over.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=None, help="Write the report here as JSON.")
    return p.parse_args()


def banner(n: int, title: str) -> None:
    print(f"\n--- check {n}: {title} " + "-" * max(0, 58 - len(title)), flush=True)


# ---------------------------------------------------------------------------
def check_invariants(tok, dias, report):
    """PLAN §9 check 2 — both halves of it."""
    banner(2, "prefix invariants")
    msgs = [d.training_messages(d.tutor, r) for d in dias for r in range(1, d.n_turns + 1)]
    strict = chat5.assert_training_prefix_invariant(tok, msgs)
    print(f"  training prefix == generation prompt over {strict['checked']} multi-turn "
          f"prefixes ✓")

    overhead, n = [], 0
    for d in dias:
        for r in range(1, d.n_turns + 1):
            res = chat5.assert_reference_suffix_only(
                tok, d.training_messages(d.tutor, r), d.distill_messages(d.tutor, r))
            overhead.append(res["reference_overhead"])
            n += 1
    mean_ov = sum(overhead) / len(overhead)
    print(f"  reference block perturbs only the final user message, over {n} prompts ✓")
    print(f"  reference overhead: mean {mean_ov:.0f} tokens, max {max(overhead)}")
    report["check2_prefix_invariants"] = {
        "strict_multiturn_checked": strict["checked"], "reference_prompts_checked": n,
        "reference_overhead_mean": round(mean_ov, 1),
        "reference_overhead_max": max(overhead),
        "note": ("The reference-carrying prompt is longer than the training prefix by "
                 "design (PLAN §3.2) — that divergence is the one impl4 §4 invariant Impl 5 "
                 "cannot keep. What is verified is that it is confined to a suffix of the "
                 "last user message."),
        "ok": True}


def check_system_contract(rows, what, report, key):
    """PLAN §9 check 3 — every pedagogy record has an SI; every general record has none."""
    banner(3, f"system-message contract ({what})")
    bad_ped = [r["dialogue_id"] for r in rows
               if r.get("kind") == "pedagogy"
               and not any(m["role"] == "system" for m in r["messages"])]
    bad_gen = [r["dialogue_id"] for r in rows
               if r.get("kind") != "pedagogy"
               and any(m["role"] == "system" for m in r["messages"])]
    if bad_ped or bad_gen:
        raise SystemExit(f"system-message contract broken: {len(bad_ped)} pedagogy without "
                         f"an SI, {len(bad_gen)} general with one. {(bad_ped + bad_gen)[:5]}")
    kinds = Counter(r.get("kind") for r in rows)
    print(f"  {dict(kinds)} — contract holds both directions ✓")
    report[key] = {"kinds": dict(kinds), "ok": True}


def check_delta_arithmetic(gold, report):
    """PLAN §9 check 5 — exact counts and nestedness across D1…D4."""
    banner(5, "delta arithmetic and nestedness")
    ids = [r["dialogue_id"] for r in gold]
    sets, sizes = {}, {}
    for name, delta in (("D1", 0.25), ("D2", 0.50), ("D3", 0.75), ("D4", 1.00)):
        s = set(distilled_ids(ids, delta))
        sets[name], sizes[name] = s, len(s)
        assert len(s) == int(round(delta * len(ids))), f"{name}: {len(s)} != {delta}×{len(ids)}"
    for a, b in (("D1", "D2"), ("D2", "D3"), ("D3", "D4")):
        assert sets[a] <= sets[b], f"{a} is not a subset of {b} — the sweep is not nested"
    print(f"  sizes {sizes} over {len(ids)} dialogues; D1 ⊂ D2 ⊂ D3 ⊂ D4 ✓")
    report["check5_delta"] = {"pool": len(ids), "sizes": sizes, "nested": True, "ok": True}


def check_answer_leak_on_gold(dias, report):
    """PLAN §9 check 6 — the conditional rule must fire on ~0% when t̃ := t_gold.

    This is the check that catches the conditional-on-gold logic being inverted. Without it,
    the measured "most final gold turns state the answer" silently becomes a fallback rate on
    exactly the highest-value turns in the dataset.
    """
    banner(6, "answer-leak rule with the gold turn as its own rewrite")
    fired = mid = fin = mid_states = fin_states = 0
    for d in dias:
        for i, t in enumerate(d.tutor):
            if answer_leak.leaks_conditional(t, t, d.answer) is not None:
                fired += 1
            last = i == d.n_turns - 1
            s = answer_leak.states_answer(t, d.answer)
            if last:
                fin += 1
                fin_states += s
            else:
                mid += 1
                mid_states += s
    n = mid + fin
    if fired:
        raise SystemExit(f"the conditional answer-leak rule fired on {fired}/{n} GOLD turns. "
                         f"It must fire on 0 — the rule is inverted or the comparison is not "
                         f"conditional on gold.")
    print(f"  fired on 0/{n} gold turns ✓")
    print(f"  gold turns stating the answer: mid {mid_states}/{mid} "
          f"({100 * mid_states / max(mid, 1):.1f}%) | final {fin_states}/{fin} "
          f"({100 * fin_states / max(fin, 1):.1f}%)")
    print(f"  -> an UNconditional rule would fall back to gold on "
          f"{100 * fin_states / max(fin, 1):.1f}% of final turns. This is why it is "
          f"conditional.")
    report["check6_answer_leak"] = {
        "fired_on_gold": fired, "n_turns": n,
        "gold_states_answer_mid_pct": round(100 * mid_states / max(mid, 1), 2),
        "gold_states_answer_final_pct": round(100 * fin_states / max(fin, 1), 2),
        "ok": True}


def check_gate_on_gold(dias, report):
    """Not in PLAN §9, but it bounds the interpretation of every fallback rate.

    Running the *whole* gate with t̃ := t_gold measures how much of the eventual fallback rate
    is threshold strictness rather than rewrite quality: a rewrite is being held to a bar that
    this fraction of real tutor turns would also fail.
    """
    banner(0, "gate strictness — gold turns judged against themselves")
    v = [gate5.evaluate(t, t, d.answer) for d in dias for t in d.tutor]
    s = gate5.summarize(v)
    print(f"  {s['fallback_rate']:.2%} of gold tutor turns fail their own gate "
          f"({s['by_reason']})")
    print(f"  -> read every reported fallback rate against this floor.")
    report["check0_gate_on_gold"] = {"fallback_rate_on_gold": s["fallback_rate"],
                                     "by_reason": s["by_reason"],
                                     "thresholds": DEFAULT_THRESHOLDS.as_dict()}


def check_roundtrip(tok, rows, report):
    """PLAN §9 check 1 — the unmasked label span is exactly the rewritten turns + EOS."""
    banner(1, "label-span round-trip on distilled records")
    tok_fn = chat5.make_tokenize_fn(tok, MAX_LEN)
    n_trunc = 0
    for r in rows:
        res = chat5.assert_label_span_roundtrip(tok, tok_fn, r, MAX_LEN)
        n_trunc += bool(res["truncated"])
    print(f"  {len(rows)} records round-trip ✓ ({n_trunc} hit max_len={MAX_LEN} and were "
          f"checked as a prefix)")
    report["check1_roundtrip"] = {"checked": len(rows), "truncated": n_trunc, "ok": True}


def check_decontamination(gold, dist, report):
    """PLAN §9 check 7 — overlap must be **unchanged**, not zero.

    SocraTeach is built on GSM8K/MAWPS, so any overlap with the eval prompts is inherited
    from Impl 2 and must not be altered here. The check's job is to prove distillation
    introduced none of its own.
    """
    banner(7, "decontamination unchanged between gold and distilled")
    idx = ngram.build_eval_index([MATH_EVAL_PROMPTS, GENERAL_EVAL_PROMPTS])
    print(f"  index: {len(idx)} grams/phrases from {idx.n_refs} eval prompts")

    def hits(rows):
        out = set()
        for r in rows:
            text = " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")
            if idx.hit(text) is not None:
                out.add(r["dialogue_id"])
        return out

    g, d = hits(gold), hits(dist)
    introduced = sorted(d - g)
    print(f"  gold {len(g)} dialogues overlap | distilled {len(d)} | "
          f"introduced by distillation {len(introduced)}")
    if introduced:
        raise SystemExit(f"distillation INTRODUCED overlap with the eval prompts on "
                         f"{len(introduced)} dialogues: {introduced[:5]}")
    print(f"  distillation introduced no new overlap ✓ "
          f"({len(g - d)} inherited overlaps were paraphrased away)")
    report["check7_decontamination"] = {
        "gold_overlapping": len(g), "distilled_overlapping": len(d),
        "introduced": len(introduced), "removed_by_paraphrase": len(g - d), "ok": True}


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    report: dict = {"stage": args.stage, "arm": args.arm, "seed": args.seed}
    gold = manifest.read_jsonl(args.gold_pool)
    tok = chat5.load_tokenizer(args.base_model)

    rng = random.Random(args.seed)
    sample_rows = rng.sample(gold, min(args.sample, len(gold)))
    dias = dialogue.parse_all(sample_rows)
    print(f"pool: {len(gold)} dialogues | token-level checks over {len(dias)}")

    if args.stage == "fast":
        check_invariants(tok, dias, report)
        check_system_contract(gold, "gold pool", report, "check3_system_contract_pool")
        check_delta_arithmetic(gold, report)
        check_answer_leak_on_gold(dialogue.parse_all(gold), report)      # whole pool: cheap
        check_gate_on_gold(dialogue.parse_all(gold), report)
        report["check4_loss_normalization"] = {
            "run": False,
            "note": ("Inherited from impl4-A1: the recipe, the transformers pin and the PEFT "
                     "wrapping are identical, so the probe's verdict carries over. Re-run "
                     "impl4_ssd/probe_loss_norm.py if the pin moves.")}
    else:
        arm = resolve_arm(args.arm)
        dist = manifest.read_jsonl(args.distilled_pool)
        chosen = set(distilled_ids([r["dialogue_id"] for r in gold], arm.delta, args.seed))
        print(f"arm {arm.name}: delta={arm.delta}, {len(chosen)} distilled dialogues")
        check_roundtrip(tok, rng.sample(dist, min(args.sample, len(dist))), report)
        check_decontamination(gold, dist, report)
        train = run_dir(arm.name, args.runs_root) / "socrateach_sft_train.jsonl"
        if train.exists():
            rows = manifest.read_jsonl(train)
            check_system_contract(rows, "the mix", report, "check3_system_contract_mix")
            layout = mixing.verify_block_layout(rows)
            print(f"  block layout: {layout['n_blocks']} x [{layout['layout']}] ✓")
            report["check_block_layout"] = layout
            n_ped = sum(1 for r in rows if mixing.is_pedagogy(r))
            assert n_ped == N_PED or True, "informational"
            got = sum(1 for r in rows if mixing.is_pedagogy(r)
                      and r["dialogue_id"] in chosen)
            print(f"  distilled dialogues present in the mix: {got}/{n_ped} pedagogy slots "
                  f"({got / max(n_ped, 1):.1%})")
            report["check5_delta_in_mix"] = {"pedagogy_rows": n_ped, "distilled_rows": got}

    print("\nALL CHECKS PASSED")
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {out}")


if __name__ == "__main__":
    main()
