#!/usr/bin/env python
"""The core of Impl 5 (PLAN §3, §4): rewrite every tutor turn in π₀'s own words.

One shared pass over the whole pedagogy pool, reused by every δ arm. ~119,000 generations
for 22,500 dialogues, run as **sequential rounds over turn positions** rather than one flat
batch, because round ``r`` must condition on the turns rounds ``1…r-1`` actually accepted::

    [SI, u₀(problem), t̃₁, s₁, …, t̃_{r-1}, s_{r-1}]  ->  sample t̃_r  ->  gate  ->  keep or gold

Two properties that buys, and they are why the pass cannot be flattened:

1. **The generation context equals the training context.** At training time the example
   contains ``t̃₁…t̃_{r-1}``, so ``t̃_r`` has to have been generated conditioned on those, not
   on gold. (The one permitted exception is the reference block — PLAN §3.2.)
2. **Fallbacks compose.** A gold fallback at turn ``r`` does not abort the dialogue; later
   turns condition on the partly-gold prefix, which is exactly what training will see.

Rounds are batched across dialogues, so this is 9 large GPU calls of shrinking size, not
119,000 sequential ones. Each round is written to ``data/distill/round-<r>.jsonl`` as soon as
it finishes and is reloaded on restart — a preemption costs one round, not the whole pass.

Usage:
    python distill_pedagogy.py
    python distill_pedagogy.py --limit 200          # smoke
    python distill_pedagogy.py --batch_size 96      # if memory is tight
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from impl5 import chat5, dialogue, distill, gate5
from impl5._impl4 import manifest, ngram
from impl5.config5 import (
    BASE_MODEL,
    DEFAULT_THRESHOLDS,
    MAX_NEW_TOKENS,
    PED_POOL_EXPECTED,
    REWRITE_TEMPLATES,
    SAMPLING_DEFAULT,
    SEED,
    TEMPLATE_DEFAULT,
)
from impl5.distill import SAMPLING
from impl5.paths5 import (
    DISTILL_DIR,
    GENERAL_EVAL_PROMPTS,
    MATH_EVAL_PROMPTS,
    DISTILL_META,
    DISTILLED_POOL,
    PEDAGOGY_POOL,
    ensure_dir,
    round_file,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=str(PEDAGOGY_POOL))
    p.add_argument("--out", default=str(DISTILLED_POOL))
    p.add_argument("--distill_dir", default=str(DISTILL_DIR))
    p.add_argument("--meta", default=str(DISTILL_META))
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--sampling", default=SAMPLING_DEFAULT, choices=sorted(SAMPLING))
    p.add_argument("--template", default=TEMPLATE_DEFAULT, choices=sorted(REWRITE_TEMPLATES),
                   help="Rewriting template. 'plan' is PLAN 3.2 verbatim and yields a 2%% "
                        "keep rate on this model -- see impl5/config5.py.")
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_batch_tokens", type=int, default=262144)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--limit", type=int, default=0, help="Only the first N dialogues (smoke).")
    p.add_argument("--rouge_min", type=float, default=DEFAULT_THRESHOLDS.rouge_min)
    p.add_argument("--no_reference", action="store_true",
                   help="PLAN §8 Block R's R4: sample the next tutor turn cold, with no "
                        "reference block. The only variant where impl4's §4 invariant holds "
                        "strictly. Expect a much higher fallback rate.")
    p.add_argument("--force", action="store_true", help="Ignore cached rounds and redo them.")
    return p.parse_args()


# ---------------------------------------------------------------------------
def load_cached_round(path: Path) -> dict[str, dict] | None:
    """``{dialogue_id: row}`` for a completed round, or ``None`` if it has not run."""
    if not path.exists():
        return None
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["dialogue_id"]] = r
    return out


def main():
    args = parse_args()
    t_start = time.time()
    th = DEFAULT_THRESHOLDS.__class__(**{**DEFAULT_THRESHOLDS.as_dict(),
                                         "rouge_min": args.rouge_min})
    ensure_dir(args.distill_dir)
    ensure_dir(Path(args.out).parent)

    rows = manifest.read_jsonl(args.pool)
    if args.limit:
        rows = rows[:args.limit]
    dias = dialogue.parse_all(rows)
    schedule = dialogue.round_schedule(dias)
    n_gen = sum(schedule.values())
    if not args.limit and len(dias) != PED_POOL_EXPECTED:
        print(f"NOTE: pool has {len(dias)} dialogues, expected {PED_POOL_EXPECTED}")

    print("=" * 74)
    print(f"Impl 5 distillation pass | {len(dias)} dialogues | {n_gen} tutor turns")
    print(f"  sampling: {SAMPLING[args.sampling].as_dict()} | max_new_tokens="
          f"{args.max_new_tokens}")
    print(f"  template: {args.template}")
    print(f"  reference in context: {not args.no_reference}"
          + ("  (R4: strict §4 invariant)" if args.no_reference else "  (PLAN §3.2)"))
    print(f"  gate: {th.as_dict()}")
    print(f"  rounds: {schedule}")
    print("=" * 74, flush=True)

    tokenizer = chat5.load_tokenizer(args.base_model)
    model = None                                     # loaded lazily: a fully cached pass
    by_id = {d.dialogue_id: d for d in dias}         # needs no GPU at all
    rewritten: dict[str, list[str]] = {d.dialogue_id: [] for d in dias}
    verdicts: dict[str, list[gate5.Verdict]] = defaultdict(list)
    n_cached_rounds = 0

    for r in sorted(schedule):
        participants = [d for d in dias if d.n_turns >= r]
        path = round_file(r, args.distill_dir)
        cached = None if args.force else load_cached_round(path)

        if cached is not None and all(d.dialogue_id in cached for d in participants):
            print(f"[round {r}] {len(participants)} dialogues — cached, reusing {path.name}",
                  flush=True)
            n_cached_rounds += 1
            samples = None
        else:
            if model is None:
                model = distill.load_hf_model(args.base_model)
            build = ((lambda d: d.training_messages(rewritten[d.dialogue_id], r))
                     if args.no_reference
                     else (lambda d: d.distill_messages(rewritten[d.dialogue_id], r, args.template)))
            prompts = [build(d) for d in participants]
            print(f"[round {r}] {len(participants)} dialogues — generating ...", flush=True)
            samples = distill.generate_samples(
                prompts, model, tokenizer, sampling_name=args.sampling,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens, seed=args.seed + r)

        round_rows = []
        for i, d in enumerate(participants):
            gold = d.tutor[r - 1]
            if samples is None:
                row = cached[d.dialogue_id]
                v = gate5.Verdict(row["passed"], row["reason"], row["stage"], row["rouge"])
                text, finished, ntok = row["sample"], row["finished"], row["n_tokens"]
            else:
                s = samples[i]
                text, finished, ntok = s.text, s.finished, s.n_tokens
                v = gate5.evaluate(text, gold, d.answer, finished=finished, th=th)
            rewritten[d.dialogue_id].append(text if v.passed else gold)
            verdicts[d.dialogue_id].append(v)
            round_rows.append({"dialogue_id": d.dialogue_id, "round": r,
                               "sample": text, "finished": finished, "n_tokens": ntok,
                               **v.as_dict()})

        if samples is not None:
            manifest.write_jsonl(path, round_rows)
        summ = gate5.summarize(v for row in round_rows
                               for v in [gate5.Verdict(row["passed"], row["reason"],
                                                       row["stage"], row["rouge"])])
        print(f"[round {r}] keep {summ['keep_rate']:.1%} | "
              f"stages {summ['by_stage']} | top {list(summ['by_reason'].items())[:3]}",
              flush=True)

    # -- assemble the distilled pool -----------------------------------------
    out_rows, kept_tokens_proxy = [], 0
    for d in dias:
        rw = rewritten[d.dialogue_id]
        assert len(rw) == d.n_turns, f"{d.dialogue_id}: {len(rw)} != {d.n_turns}"
        out_rows.append(d.with_rewritten(rw))
        kept_tokens_proxy += sum(len(t.split()) for t, v in zip(rw, verdicts[d.dialogue_id])
                                 if v.passed)
    # -- decontamination fallback (PLAN §9 check 7) ---------------------------
    #
    # Told to match gold's length, the rewriter sometimes restates the problem back to the
    # student — and for a SocraTeach dialogue built on a GSM8K item, that problem statement
    # IS an eval prompt. So a rewrite can introduce a 13-gram overlap with math_eval that
    # the gold turn did not have.
    #
    # The rule PLAN §9 check 7 sets is "overlap unchanged, not zero": whatever SocraTeach
    # inherited from Impl 2 must stay, and distillation must add none of its own. The
    # remedy is the one the gate already uses everywhere else — fall back to gold — applied
    # to the whole dialogue, which keeps the pool count and the A1 pairing exact at a cost
    # of a few thousandths of realised δ.
    idx = ngram.build_eval_index([MATH_EVAL_PROMPTS, GENERAL_EVAL_PROMPTS])

    def _overlaps(rec) -> bool:
        text = " ".join(m["content"] for m in rec["messages"] if m["role"] == "assistant")
        return idx.hit(text) is not None

    reverted = []
    for i, d in enumerate(dias):
        if _overlaps(out_rows[i]) and not _overlaps(d.record):
            out_rows[i] = d.with_rewritten(d.tutor)
            verdicts[d.dialogue_id] = [gate5.Verdict(False, "decontamination_revert",
                                                     "decontamination", 0.0)
                                       for _ in d.tutor]
            reverted.append(d.dialogue_id)
    if reverted:
        print(f"\ndecontamination: reverted {len(reverted)} dialogue(s) to gold because the "
              f"rewrite introduced an eval-prompt overlap: {reverted[:5]}")

    manifest.write_jsonl(args.out, out_rows)

    # -- meta ----------------------------------------------------------------
    all_v = [v for vs in verdicts.values() for v in vs]   # post-revert
    per_round = {r: gate5.summarize(vs) for r, vs in
                 ((r, [verdicts[d.dialogue_id][r - 1] for d in dias if d.n_turns >= r])
                  for r in sorted(schedule))}
    gold_words = sum(len(t.split()) for d in dias for t in d.tutor)
    new_words = sum(len(m["content"].split()) for row in out_rows
                    for m in row["messages"] if m["role"] == "assistant")
    fully = sum(1 for d in dias if all(v.passed for v in verdicts[d.dialogue_id]))

    meta = {
        "pool": str(args.pool),
        "n_dialogues": len(dias),
        "n_tutor_turns": n_gen,
        "round_schedule": schedule,
        "cached_rounds_reused": n_cached_rounds,
        "base_model": args.base_model,
        "sampling": SAMPLING[args.sampling].as_dict(),
        "max_new_tokens": args.max_new_tokens,
        "reference_in_context": not args.no_reference,
        "template": args.template,
        "template_text": REWRITE_TEMPLATES[args.template],
        "reference_note": (
            "PLAN §3.2: the distillation prompt appends the gold turn as a reference to the "
            "content of the last user message, so it is strictly longer than the training "
            "prefix. This is the one place impl4's §4 invariant cannot hold exactly; the SDFT "
            "paper has the same gap. acceptance_checks5.py check 2 verifies the divergence is "
            "confined to that suffix."
        ) if not args.no_reference else "R4: no reference; the §4 invariant holds strictly.",
        "gate": th.as_dict(),
        "gate_overall": gate5.summarize(all_v),
        "decontamination_reverted": reverted,
        "decontamination_note": (
            "PLAN §9 check 7 requires overlap with the eval prompt sets to be UNCHANGED, "
            "not zero: SocraTeach is built on GSM8K/MAWPS so some overlap is inherited "
            "from Impl 2 and must not be altered. Dialogues whose rewrite introduced a "
            "NEW overlap are reverted to gold in full."),
        "gate_by_round": per_round,
        "fallback_rate_by_turn_index": {r: per_round[r]["fallback_rate"]
                                        for r in sorted(per_round)},
        "realised_delta_turns": round(sum(v.passed for v in all_v) / len(all_v), 4),
        "realised_delta_dialogues_fully_rewritten": round(fully / len(dias), 4),
        "tutor_words_gold": gold_words,
        "tutor_words_distilled": new_words,
        "tutor_word_ratio": round(new_words / gold_words, 4),
        "elapsed_min": round((time.time() - t_start) / 60, 1),
        "note": (
            "realised δ here is the *turn-level* keep rate of the shared pass. Each arm's "
            "manifest reports its own realised δ in dialogues and in label tokens, which is "
            "the quantity PLAN §4 asks for."
        ),
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"Distilled pool -> {args.out}  ({len(out_rows)} dialogues)")
    print(f"  turn-level keep rate : {meta['realised_delta_turns']:.1%}")
    print(f"  dialogues fully rewritten: {meta['realised_delta_dialogues_fully_rewritten']:.1%}")
    print(f"  tutor words: {gold_words} gold -> {new_words} distilled "
          f"({meta['tutor_word_ratio']:.3f}x)")
    print(f"  fallback by turn index: "
          f"{ {r: f'{v:.1%}' for r, v in meta['fallback_rate_by_turn_index'].items()} }")
    print(f"  reasons: {dict(Counter(meta['gate_overall']['by_reason']).most_common(6))}")
    print(f"  elapsed: {meta['elapsed_min']} min")
    print(f"Meta -> {args.meta}")


if __name__ == "__main__":
    main()
