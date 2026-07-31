#!/usr/bin/env python
"""Step 2a (PLAN §3) — build the shared Super-NaturalInstructions prompt pool.

Every SuperNI arm (A2, A3/T1, A4, T2, T3, T4, B2) draws from *this one pool* at
identical counts. Building it once is what makes A2 (gold) and A3 (self-generated)
a clean paired control rather than two independent samples.

Filters, in PLAN §3 order:
  1. English training tasks only (``splits/default/train_tasks.txt``).
  2. Contamination exclusion (mandatory): task-level Source/name blocklist for
     BIG-Bench / GSM8K / MATH / AIME, then a per-instance 13-gram overlap check
     against ``math_eval/math_logic_prompts.jsonl`` and
     ``general_eval/general_prompts.jsonl``.
  3. Gold output length >= 30 whitespace words, mean over the task's instances.
  4. Round-robin instance sampling so no single task dominates.

Also writes ``shared/superni_train_task_ids.txt`` (so the eval team can keep their
sets clean) and, with ``--split test``, ``shared/superni_heldout_prompts.jsonl``
(the untouched ``test_tasks.txt`` split, shipped unused for a possible
general-prompt KL axis).

Usage:
    # recommended on the cluster: clone once, then read locally
    git clone --depth 1 https://github.com/allenai/natural-instructions /path/ni
    python build_prompt_pool.py --superni_dir /path/ni
    python build_prompt_pool.py --superni_dir /path/ni --split test

    # no clone: stream the pinned commit over HTTP
    python build_prompt_pool.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from impl4 import manifest, ngram, superni
from impl4.config import SEED, SUPERNI_POOL_SIZE
from impl4.paths import (
    GENERAL_EVAL_PROMPTS,
    MATH_EVAL_PROMPTS,
    SHARED_DIR,
    SUPERNI_CACHE_DIR,
    SUPERNI_POOL,
    SUPERNI_POOL_META,
    ensure_dir,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--superni_dir", default=None,
                   help="Local `git clone` of allenai/natural-instructions (recommended). "
                        "Omit to stream tasks/*.json from the pinned commit over HTTP.")
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="'train' builds the pool we train on; 'test' builds the held-out "
                        "prompt file we ship unused (PLAN §10).")
    p.add_argument("--n_prompts", type=int, default=SUPERNI_POOL_SIZE,
                   help="Pool size. Needs >= 7,496 x 1.15 over-generation, plus margin "
                        "for B2 resampling.")
    p.add_argument("--instances_per_task", type=int, default=300,
                   help="Stop streaming each task file after this many instances "
                        "(0 = read them all).")
    p.add_argument("--min_gold_words", type=int, default=superni.MIN_GOLD_WORDS,
                   help="PLAN §3 filter 3. SuperNI is dominated by short-answer "
                        "classification, so this bites hard — the length profile printed "
                        "at the end shows retention at every candidate threshold.")
    p.add_argument("--max_tasks", type=int, default=0, help="Debug: cap tasks scanned.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=None)
    p.add_argument("--cache_dir", default=str(SUPERNI_CACHE_DIR),
                   help="Cache fetched task metadata + instances here ('' to disable). "
                        "Re-scanning at a different --min_gold_words is then free.")
    p.add_argument("--scan_only", action="store_true",
                   help="Report the length profile and exit without writing a pool. "
                        "Use this to choose --min_gold_words before committing.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def print_length_profile(prof: dict, chosen: int) -> None:
    if not prof.get("n_tasks_scored"):
        return
    print(f"\nGold-length profile over {prof['n_tasks_scored']} tasks that cleared the "
          f"English + contamination filters:")
    q = prof["quantiles"]
    print("  mean gold words per task: "
          + "  ".join(f"{k}={v:g}" for k, v in q.items()))
    print(f"  {'threshold':>10} {'tasks':>7} {'instances':>11}")
    for t, n in prof["tasks_retained_at_threshold"].items():
        mark = "  <- --min_gold_words" if int(t) == chosen else ""
        print(f"  {t:>10} {n:>7} "
              f"{prof['instances_available_at_threshold'][t]:>11}{mark}")


def main():
    args = parse_args()
    heldout = args.split == "test"
    out_path = Path(args.out) if args.out else (
        SHARED_DIR / "superni_heldout_prompts.jsonl" if heldout else SUPERNI_POOL
    )
    if out_path.exists() and not args.force and not args.scan_only:
        n = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"{out_path} already present ({n} rows). Use --force to rebuild.")
        return

    print("Building the decontamination index (PLAN §3 filter 2b) ...")
    idx = ngram.build_eval_index([MATH_EVAL_PROMPTS, GENERAL_EVAL_PROMPTS])
    print(f"  {idx.n_refs} eval prompts -> {len(idx)} 13-grams/short-phrases")

    src = superni.SuperNISource(
        local_dir=Path(args.superni_dir) if args.superni_dir else None,
        instances_per_task=args.instances_per_task,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    print(f"Source: {json.dumps(src.describe())}")
    print(f"Scanning the '{args.split}' split ...")
    tasks, stats = superni.scan_tasks(
        src, split=args.split, ngram_index=idx,
        min_gold_words=args.min_gold_words, max_tasks=args.max_tasks,
    )
    print(f"  {stats['tasks_listed']} listed -> {stats['tasks_retained']} retained")
    for k in ("dropped_non_english", "dropped_contaminated_source", "dropped_short_gold",
              "dropped_no_instances", "dropped_fetch_error", "instances_dropped_ngram"):
        print(f"    {k}: {stats[k]}")
    if stats["fetch_errors"]:
        print(f"    WARNING: {len(stats['fetch_errors'])} fetch errors, first few:")
        for e in stats["fetch_errors"][:5]:
            print(f"      {e}")

    print_length_profile(stats["length_profile"], args.min_gold_words)

    if args.scan_only:
        print("\n--scan_only: no pool written. Pick a --min_gold_words from the table "
              "above and re-run.")
        return

    if not tasks:
        raise SystemExit(
            "no tasks survived the filters — refusing to write an empty pool. "
            "The gold-length threshold is almost certainly the cause; see the profile above."
        )

    pool = superni.round_robin_sample(tasks, args.n_prompts, seed=args.seed)
    if len(pool) < args.n_prompts:
        print(f"  NOTE: only {len(pool)} instances available (asked for {args.n_prompts}). "
              f"Raise --instances_per_task or lower --n_prompts.")

    # Belt and braces: the index already ran per-instance during the scan, but assert
    # zero survivors here so PLAN §11 check 6 can never be quietly skipped.
    residual = [p for p in pool if idx.hit(f"{p['definition']}\n{p['input']}\n{p['gold']}")]
    if residual:
        raise SystemExit(f"{len(residual)} pool instances still hit an eval n-gram — bug")

    n = manifest.write_jsonl(out_path, pool)
    print(f"\nWrote {n} prompts -> {out_path}")

    meta = {
        "split": args.split,
        "source": src.describe(),
        "n_prompts": n,
        "n_prompts_requested": args.n_prompts,
        "seed": args.seed,
        "min_gold_words": args.min_gold_words,
        "contam_patterns": list(superni.CONTAM_PATTERNS),
        "decontam": {
            "n_reference_prompts": idx.n_refs,
            "n_grams": len(idx),
            "n": idx.n,
            "targets": [str(MATH_EVAL_PROMPTS), str(GENERAL_EVAL_PROMPTS)],
        },
        "stats": stats,
        "mean_gold_words_over_retained_tasks": round(
            sum(t.mean_gold_words for t in tasks) / len(tasks), 2),
    }
    if not heldout:
        SUPERNI_POOL_META.parent.mkdir(parents=True, exist_ok=True)
        SUPERNI_POOL_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote pool metadata -> {SUPERNI_POOL_META}")

        ensure_dir(SHARED_DIR)
        ids_path = SHARED_DIR / "superni_train_task_ids.txt"
        used = sorted({p["superni_task_id"] for p in pool})
        ids_path.write_text("\n".join(used) + "\n", encoding="utf-8")
        print(f"Wrote {len(used)} task ids -> {ids_path}")
    else:
        (SHARED_DIR / "superni_heldout_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    hist = stats["category_histogram"]
    print(f"\nCategories retained ({len(hist)} distinct across "
          f"{stats['tasks_retained']} tasks):")
    for cat, cnt in list(hist.items())[:20]:
        print(f"  {cnt:>4}  {cat}")
    if len(hist) > 20:
        print(f"  ... and {len(hist) - 20} more")
    if stats["tasks_retained"] < 30:
        print(f"\nWARNING: only {stats['tasks_retained']} tasks survived, so the pool's "
              f"domain breadth is thin and round-robin draws ~{n // max(1, stats['tasks_retained'])} "
              f"instances from each. PLAN §3 asks for a Categories histogram precisely so "
              f"this is visible — report it, and consider lowering --min_gold_words using "
              f"the profile above (--scan_only makes that free once the cache is warm).")


if __name__ == "__main__":
    main()
