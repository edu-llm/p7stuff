#!/usr/bin/env python
"""The replay slot (PLAN §5) — Tülu-3 gold, and **the same conversations impl4's A1 used**.

Impl 5's replay stream goes back to Impl 2's Tülu-3 gold, so this is far smaller than
Impl 4's version of the same script: no SuperNI, no generation, no gate.

The one substantive decision, and it is a deliberate deviation from PLAN §5.

PLAN §5 says: hold the pedagogy set fixed across δ arms and *absorb* the rewrite-induced
token drift by choosing **which** Tülu examples to include, via ``token_matched_select``,
so that ``general_tokens / pedagogy_tokens`` matches D0's in every arm. That is the right
call for a full D0…D4 sweep, where the arms are compared with each other.

This build has one trained arm and its baseline is **impl4's A1**, an existing run whose
slot was built with no matching target (A1 *is* the reference). Token-matching here would
therefore change two things at once — the pedagogy targets *and* which Tülu conversations
are in the replay stream — in the single contrast the whole run exists to make. Holding the
slot byte-identical to A1's makes the pedagogy targets the only moving part.

What that costs is the ped:gen token ratio, which now drifts by however much the rewrites
are shorter or longer than gold. So it is **measured and reported** rather than corrected:
``--check_ratio`` compares against D0's realised ratio and warns past ±5%. If a later build
runs the full sweep, pass ``--token_match`` and PLAN §5 applies again.

Reproduction is exact because every step is seed-deterministic: over-request
``ceil(n × 1.15)``, count label tokens under ``make_tokenize_fn``, drop the conversations
whose prompt eats the whole ``max_len`` budget, trim to ``n``. That is Impl 4's A1 path,
called through Impl 4's own ``tulu.load_tulu_slot``.

Usage:
    python build_general_slot5.py --arm D4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from impl5 import chat5
from impl5._impl4 import manifest, mixing, tulu
from impl5.config5 import (
    ARM_CHOICES,
    BASE_MODEL,
    MAX_LEN,
    OVERGENERATE,
    SEED,
    TOKEN_MATCH_TOLERANCE,
    resolve_arm,
    slot_sizes,
)
from impl5.paths5 import PEDAGOGY_REFERENCE, ensure_dir, run_dir

#: impl4's A1 slot, measured 2026-08-01 on the real run. Reproducing it is a hard check:
#: same seed, same shard, same filter, same trim -> same numbers, or something moved.
A1_REFERENCE = {"n": 7384, "total_label_tokens": 631395}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARM_CHOICES)
    p.add_argument("--runs_root", default=None)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--overgenerate", type=float, default=OVERGENERATE)
    p.add_argument("--poc", action="store_true")
    p.add_argument("--token_match", action="store_true",
                   help="PLAN §5: choose which Tulu examples to include so the ped:gen token "
                        "ratio matches D0's. Correct for a full sweep; NOT the default here "
                        "(see the module docstring).")
    p.add_argument("--expect_a1", action="store_true", default=True,
                   help="Check the slot reproduces impl4's A1 exactly (warns on mismatch).")
    p.add_argument("--no_expect_a1", dest="expect_a1", action="store_false")
    p.add_argument("--strict_a1", action="store_true",
                   help="Turn the A1-reproduction mismatch into a hard failure. Off by "
                        "default: the distillation pass that precedes this stage costs an "
                        "hour and a half, and a Tulu slot that came out different is a "
                        "confound to report, not a reason to throw that away.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    arm = resolve_arm(args.arm)
    if arm.external_run:
        raise SystemExit(f"{arm.name} is not trained here — it is {arm.external_run}. "
                         f"Nothing to build.")
    _, n_gen = slot_sizes(args.poc)
    out_dir = ensure_dir(run_dir(arm.name, args.runs_root))
    slot_path = out_dir / "general_slot.jsonl"
    if slot_path.exists() and not args.force:
        print(f"{slot_path} already present ({sum(1 for _ in open(slot_path))} rows). "
              f"Use --force to rebuild.")
        return

    tokenizer = chat5.load_tokenizer(args.base_model)
    tok_fn = chat5.make_tokenize_fn(tokenizer, MAX_LEN)

    want = int(math.ceil(n_gen * args.overgenerate))
    print(f"Loading {want} Tulu-3 gold conversations for a {n_gen}-example slot "
          f"(seed {args.seed}) ...")
    gold = tulu.load_tulu_slot(want, args.seed)
    counts = chat5.label_token_counts(tok_fn, gold)

    # A conversation whose prompt consumes the whole max_len budget leaves no assistant
    # turn, so it trains on nothing while still occupying one of the 8 general slots in a
    # block — it dilutes the replay stream instead of filling it. Impl 4 hit 34 of these.
    keep = [i for i, c in enumerate(counts) if c > 0]
    n_dropped = len(counts) - len(keep)
    if n_dropped:
        print(f"  dropped {n_dropped}/{len(counts)} with 0 label tokens at max_len={MAX_LEN}")
    gold = [gold[i] for i in keep]
    counts = [counts[i] for i in keep]
    if len(gold) < n_gen:
        raise SystemExit(f"only {len(gold)} usable Tulu conversations, need {n_gen}. "
                         f"Raise --overgenerate.")

    match_stats = None
    if args.token_match:
        ref = json.loads(Path(PEDAGOGY_REFERENCE).read_text())
        target = int(round(ref["ratio_general_to_pedagogy"] * ref["pedagogy_tokens_this_arm"]))
        sel, match_stats = mixing.token_matched_select(counts, n_gen, target, seed=args.seed)
        gold = [gold[i] for i in sel]
        counts = [counts[i] for i in sel]
        print(f"  token-matched to {target} label tokens: realised "
              f"{match_stats['realized_total']} ({match_stats['swaps']} swaps)")
    else:
        gold, counts = gold[:n_gen], counts[:n_gen]

    total = sum(counts)
    print(f"Slot: {len(gold)} conversations, {total} label tokens "
          f"(mean {total / len(gold):.1f})")

    reproduces = None
    if args.expect_a1 and not args.poc and not args.token_match:
        exp = A1_REFERENCE
        reproduces = (len(gold), total) == (exp["n"], exp["total_label_tokens"])
        if reproduces:
            print(f"  reproduces impl4-A1 exactly ({exp['n']} / "
                  f"{exp['total_label_tokens']} label tokens) ✓")
        else:
            msg = (f"replay slot does NOT reproduce impl4-A1: got {len(gold)} examples / "
                   f"{total} label tokens, expected {exp['n']} / "
                   f"{exp['total_label_tokens']} ({total / exp['total_label_tokens']:.4f}x).\n"
                   f"  D0 for this build is impl4-A1, so D4 vs D0 is now a two-variable "
                   f"contrast: the pedagogy targets AND which Tulu conversations are in the "
                   f"replay stream. Likely causes: a different Tulu shard, a different "
                   f"tokenizer revision, a changed max_len.")
            if args.strict_a1:
                raise SystemExit(msg)
            print(f"  WARNING: {msg}\n  Continuing anyway (--strict_a1 to stop here). This "
                  f"is recorded as reproduces_impl4_A1: false and must be quoted alongside "
                  f"any D4-vs-D0 number.")

    manifest.write_jsonl(slot_path, gold)
    manifest.merge(out_dir, "general_slot", {
        "source": tulu.TULU_ID,
        "kind": tulu.KIND,
        "n": len(gold),
        "n_requested": want,
        "n_dropped_zero_label_tokens": n_dropped,
        "total_label_tokens": total,
        "mean_label_tokens": round(total / len(gold), 2),
        "token_matched": bool(args.token_match),
        "token_match_stats": match_stats,
        "reproduces_impl4_A1": reproduces,
        "impl4_A1_reference": A1_REFERENCE,
        "deviation_note": (
            "PLAN §5 prescribes token_matched_select against D0's ratio. This build holds the "
            "slot byte-identical to impl4-A1's instead, because D0 IS impl4-A1 and matching "
            "would change the replay stream and the pedagogy targets in the same contrast. "
            "The realised ped:gen token ratio is reported by mix_arm5.py and compared against "
            "D0's; treat drift beyond +/-5% as a caveat on the comparison."
        ) if not args.token_match else None,
        "tolerance": TOKEN_MATCH_TOLERANCE,
        "seed": args.seed,
        "max_len": MAX_LEN,
    })
    print(f"Wrote {len(gold)} replay examples -> {slot_path}")


if __name__ == "__main__":
    main()
