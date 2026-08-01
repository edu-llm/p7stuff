#!/usr/bin/env python
"""Step 1 (PLAN §8.1) — the pedagogy pool, shared by every arm.

**Pulls the published dataset; does not regenerate the system instructions.**

The per-dialogue SIs are *baked into* ``meric533/socrateach-sft``. They were generated
once, upstream in the POC, by ``prepare_socrateach_sft.py`` — and regenerating them
locally produces *different* strings, because the generator draws phrasing variants
per dialogue. Impl 3 trains on the Hub rows as published, so a regenerated pool would
mean our pedagogy stream carries different system prompts from theirs, and pedagogy
NLL would silently stop being comparable. Nothing downstream can detect that.

So the revision is pinned:

    meric533/socrateach-sft @ 1fd0b54ab8a0d96d07471f1f7d7173666d4071b8

``--regenerate`` restores the old behaviour (shell out to
``socrateach_sft/prepare_socrateach_sft.py``) for provenance work. It prints a loud
warning, because the resulting pool is not comparable to Impl 3.

Usage:
    python build_pedagogy_pool.py
    python build_pedagogy_pool.py --force                 # rebuild even if present
    python build_pedagogy_pool.py --regenerate            # old path; NOT comparable
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from impl4 import manifest
from impl4.config import PED_POOL_TARGET, SEED
from impl4.paths import PEDAGOGY_POOL_DIR, PREPARE_SOCRATEACH_PY, ensure_dir

# The dataset Impl 3 trains on, pinned. Both projects must read the same bytes.
HF_DATASET = "meric533/socrateach-sft"
HF_REVISION = "1fd0b54ab8a0d96d07471f1f7d7173666d4071b8"

# Row fields we carry through unchanged, so the pool matches the schema the rest of
# the pipeline (and Impl 3's tagging) expects.
KEEP_FIELDS = ("messages", "problem_id", "dialogue_id", "answer", "source", "kind")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=str(PEDAGOGY_POOL_DIR))
    p.add_argument("--dataset", default=HF_DATASET)
    p.add_argument("--revision", default=HF_REVISION,
                   help="Pinned so the baked-in system instructions match Impl 3's exactly.")
    p.add_argument("--max_total", type=int, default=PED_POOL_TARGET)
    p.add_argument("--regenerate", action="store_true",
                   help="Rebuild the SIs locally instead of using the Hub rows. Produces "
                        "different system prompts -> NOT comparable to Impl 3.")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Only used by --regenerate; must stay 13 there.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
def from_hub(args, out_dir: Path) -> list[dict]:
    """Pedagogy rows from the pinned Hub revision, in dataset order.

    The repo stores plain JSONL under ``data/``, so the files are fetched by name rather
    than through ``load_dataset``: split-name inference from filenames varies across
    ``datasets`` versions, and a pool that silently came from the wrong split is exactly
    the class of bug this whole exercise is trying to avoid.

    Order is preserved rather than shuffled — ``mix_and_order.py`` does its own seeded
    shuffle within each stream, so shuffling here would only make the pool harder to diff
    against the source.
    """
    from huggingface_hub import hf_hub_download

    def fetch(split: str) -> Path:
        return Path(hf_hub_download(
            repo_id=args.dataset, repo_type="dataset", revision=args.revision,
            filename=f"data/socrateach_sft_{split}.jsonl",
        ))

    print(f"Loading {args.dataset} @ {args.revision[:12]} ...")
    train_src = fetch("train")
    all_rows = manifest.read_jsonl(train_src)
    counts: dict = {}
    for r in all_rows:
        counts[r.get("kind")] = counts.get(r.get("kind"), 0) + 1
    print(f"  train split: {len(all_rows)} rows {counts}")

    rows = []
    for r in all_rows:
        if r.get("kind") != "pedagogy":
            continue          # Impl 4 builds its own replay slot per arm
        rows.append({k: r.get(k) for k in KEEP_FIELDS})
        if args.max_total and len(rows) >= args.max_total:
            break
    print(f"  kept {len(rows)} pedagogy rows")

    # val/test come across verbatim: Impl 3's KL and pedagogy-NLL probes are the first
    # 64 / 128 rows of the validation split *in file order*, so the order is load-bearing
    # and these files must not be re-serialised through anything that reorders them.
    for split, local in (("val", "val"), ("test", "test")):
        dst = out_dir / f"socrateach_sft_{local}.jsonl"
        src = fetch(split)
        dst.write_bytes(src.read_bytes())
        n = sum(1 for _ in open(dst, encoding="utf-8"))
        print(f"  copied {split} verbatim ({n} rows) -> {dst.name}")
    return rows


def regenerate(args, out_dir: Path) -> list[dict]:
    print("WARNING: --regenerate rebuilds the per-dialogue system instructions locally.\n"
          "         They will NOT match the Hub rows Impl 3 trained on, so pedagogy NLL\n"
          "         and every new-task comparison against Impl 3 become invalid.\n",
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
    return manifest.read_jsonl(out_dir / "socrateach_sft_train.jsonl")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    train_path = out_dir / "socrateach_sft_train.jsonl"
    source_path = out_dir / "pool_source.json"

    if train_path.exists() and not args.force:
        n = sum(1 for _ in open(train_path, encoding="utf-8"))
        src = json.loads(source_path.read_text()) if source_path.exists() else {}
        print(f"Pedagogy pool already present: {train_path} ({n} examples).")
        if src:
            print(f"  built from: {src.get('mode')} {src.get('dataset', '')} "
                  f"{(src.get('revision') or '')[:12]}")
        if src.get("mode") == "regenerated":
            print("  WARNING: this pool has locally regenerated SIs and is NOT comparable to "
                  "Impl 3. Rebuild with --force to pull the pinned Hub revision.")
        print("  Use --force to rebuild.")
        return

    rows = regenerate(args, out_dir) if args.regenerate else from_hub(args, out_dir)
    if not args.regenerate:
        manifest.write_jsonl(train_path, rows)
        print(f"Wrote {len(rows)} pedagogy examples -> {train_path}")

    rows = manifest.read_jsonl(train_path)
    kinds: dict = {}
    for r in rows:
        kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
    assert set(kinds) == {"pedagogy"}, f"pool must be pedagogy-only, got {kinds}"
    n_sys = sum(1 for r in rows if any(m["role"] == "system" for m in r["messages"]))
    assert n_sys == len(rows), (
        f"{len(rows) - n_sys} pedagogy records lack a system message (PLAN §11 check 3)"
    )

    source_path.write_text(json.dumps({
        "mode": "regenerated" if args.regenerate else "hub",
        "dataset": None if args.regenerate else args.dataset,
        "revision": None if args.regenerate else args.revision,
        "n": len(rows),
        "comparable_to_impl3": not args.regenerate,
        "note": ("SIs are baked into the Hub rows; regenerating them produces different "
                 "system prompts and breaks pedagogy-NLL comparability with Impl 3."),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nPedagogy pool ready: {len(rows)} examples, all with a system instruction.")
    if len(rows) < PED_POOL_TARGET:
        print(f"NOTE: pool is {len(rows)} < {PED_POOL_TARGET}; the 923-block mix needs 22,152.")
    print(f"Files: {sorted(p.name for p in Path(out_dir).glob('*.jsonl'))}")
    print(f"Source recorded -> {source_path}")


if __name__ == "__main__":
    main()
