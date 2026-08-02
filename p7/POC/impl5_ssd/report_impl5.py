#!/usr/bin/env python
"""Assemble the run's evidence into one readable summary.

Deliberately prints the caveats next to the numbers rather than in a footnote, because every
number here has a condition attached that changes what it means:

* **realised δ**, not nominal δ, is what the arm actually trained on. Gate fallbacks put gold
  turns back inside distilled dialogues, so a "δ=1" arm can be half gold.
* **ped_nll is measured on held-out gold.** D4 trained on paraphrases, so a gap is expected
  by construction and is not evidence about teaching quality.
* **Stage 4 did not run**, so "matched pedagogy quality" is unverified and no Definition-of-
  Done claim can be made from this run.
* the **ped:gen token ratio** drifts when rewrites change length, which turns part of a δ
  contrast into a stream-weight contrast.

    python report_impl5.py --results impl5_results/ --ped_nll ped_nll.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="impl5_results",
                   help="Directory holding distill_meta.json / *_manifest.json.")
    p.add_argument("--ped_nll", action="append", default=None,
                   help="ped_nll JSONL file(s), repeatable.")
    p.add_argument("--baseline", default="impl4-A1")
    return p.parse_args()


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * max(len(title), 70))


def main():
    args = parse_args()
    d = Path(args.results)
    meta = load(d / "distill_meta.json")
    man = next((load(p) for p in sorted(d.glob("*_manifest.json"))), {})
    mix = man.get("mix", {})

    print("=" * 74)
    print("IMPL 5 — self-distilled pedagogy targets")
    print("=" * 74)

    if meta:
        rule("The distillation pass (shared by every δ arm)")
        g = meta.get("gate_overall", {})
        print(f"  dialogues           {meta.get('n_dialogues')}   "
              f"tutor turns {meta.get('n_tutor_turns')}")
        print(f"  template            {meta.get('template')}   "
              f"(PLAN §3.2's own template keeps ~7% on this model)")
        print(f"  sampling            {meta.get('sampling', {}).get('name')} "
              f"T={meta.get('sampling', {}).get('T_train')} "
              f"max_new={meta.get('max_new_tokens')}")
        print(f"  turn-level keep     {g.get('keep_rate')}")
        print(f"  by stage            {g.get('by_stage')}")
        print(f"  top reasons         {dict(list(g.get('by_reason', {}).items())[:5])}")
        print(f"  tutor words         {meta.get('tutor_words_gold')} gold -> "
              f"{meta.get('tutor_words_distilled')} distilled "
              f"({meta.get('tutor_word_ratio')}x)")
        fb = meta.get("fallback_rate_by_turn_index", {})
        if fb:
            print("  fallback by turn    " +
                  "  ".join(f"r{k}:{v:.0%}" for k, v in list(fb.items())[:9]))
            print("    (PLAN §13 expects this to climb with r as the rewritten prefix drifts "
                  "from gold;\n     it caps realised δ and is reported, not hidden)")
        print(f"  elapsed             {meta.get('elapsed_min')} min")

    if mix:
        rule("What D4 actually trained on")
        print(f"  nominal δ                 {mix.get('delta_nominal')}")
        print(f"  realised δ (dialogues)    {mix.get('delta_realised_dialogues')}")
        print(f"  realised δ (LABEL TOKENS) {mix.get('delta_realised_label_tokens')}"
              f"   <- the quantity PLAN §4 asks for")
        lt = mix.get("label_tokens", {})
        print(f"  pedagogy label tokens     {lt.get('pedagogy')} "
              f"(gold, same dialogues: {lt.get('pedagogy_gold_same_dialogues')})")
        print(f"  general label tokens      {lt.get('general')}")
        print(f"  general share of tokens   {mix.get('token_ratio_general')}")
        drift = mix.get("ratio_drift_vs_D0")
        ok = mix.get("ratio_within_tolerance")
        print(f"  ped:gen ratio vs D0       {mix.get('ratio_general_to_pedagogy')} vs "
              f"{mix.get('ratio_general_to_pedagogy_D0')}  drift {drift}  "
              f"{'within ±5%' if ok else '*** OUTSIDE ±5% ***'}")
        if not ok:
            print("    -> part of the D4-vs-D0 difference is a stream-weight difference. "
                  "Say so when quoting it.")

    gs = man.get("general_slot", {})
    if gs:
        rule("Replay slot")
        rep = gs.get("reproduces_impl4_A1")
        print(f"  {gs.get('n')} Tulu-3 conversations, {gs.get('total_label_tokens')} label "
              f"tokens")
        print(f"  reproduces impl4-A1: {rep}"
              + ("" if rep else "   <- D4 vs D0 is a TWO-variable contrast; quote this"))

    if args.ped_nll:
        rule("Pedagogy NLL (held-out GOLD dialogues — never distilled)")
        runs, base = {}, None
        for path in args.ped_nll:
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("ped_nll") is None:
                    continue
                if r["run"] == "base":
                    base = r["ped_nll"]
                else:
                    runs.setdefault(r["run"], {})[int(r["step"])] = r["ped_nll"]
        if base:
            print(f"  base π₀ {base:.4f}")
        ref = runs.get(args.baseline, {})
        for name in sorted(runs):
            last = max(runs[name])
            v = runs[name][last]
            delta = (f"   {1000 * (v - ref[last]):+6.1f} millinats vs {args.baseline}"
                     if last in ref and name != args.baseline else "")
            print(f"  {name:12s} step {last:>4}  {v:.4f}{delta}")
        print("\n  D4 is scored against gold targets it was not trained on, so a positive "
              "gap is\n  expected by construction. This axis says nothing about forgetting "
              "or about\n  pedagogy quality — neither was measured in this run.")

    gate = mix.get("gate") or (meta.get("gate") if meta else None)
    if gate and not gate.get("calibrated", True):
        rule("*** The kill/go gate did not run ***")
        print("  PLAN §4 Stage 4 (blind-judge calibration) was skipped. The gate thresholds")
        print("  are the plan's provisional values. 'Matched pedagogy quality' — the "
              "condition\n  the whole Definition of Done is stated at — is UNVERIFIED.")
        print("  A forgetting win from this run alone is not a win.")
    print()


if __name__ == "__main__":
    main()
