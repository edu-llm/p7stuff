#!/usr/bin/env python
"""PLAN §11 — acceptance checks. Run these before any full run.

    1. Round-trip a generated general example through `make_tokenize_fn`; the unmasked
       label span must decode to exactly the assistant content + EOS.
    2. The generation-time prompt string must be byte-identical to the training-time
       prefix for the same messages. This is the §4 invariant; a mismatch invalidates
       everything.
    3. No general record has a system message; every pedagogy record has one.
    4. The loss-normalisation probe (§5)  -- run separately, see --with_probe.
    5. The first 3 blocks of the ordered train file show the 24/8 layout.
    6. 13-gram overlap with `math_logic_prompts.jsonl` and `general_prompts.jsonl` is zero.
    7. A `--poc` smoke run end to end  -- run separately, see RUNBOOK.md.

Checks 1, 2 and part of 5 need a tokenizer; 3, 5, 6 are pure Python. ``--offline``
skips the tokenizer-dependent ones so the rest can run without downloading a model.

Usage:
    python acceptance_checks.py --arm A3
    python acceptance_checks.py --arm A3 --with_probe
    python acceptance_checks.py --arm A3 --offline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback

from impl4 import chat, manifest, mixing, ngram
from impl4.config import (
    ALL_ARMS,
    ARM_CHOICES,
    BASE_MODEL,
    GEN_PER_BLOCK,
    MAX_LEN,
    PED_PER_BLOCK,
    resolve_arm,
)
from impl4.paths import GENERAL_EVAL_PROMPTS, IMPL4_ROOT, MATH_EVAL_PROMPTS, run_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARM_CHOICES,
                   help=f"One of {', '.join(ALL_ARMS)} (T1 is an alias of A3).")
    p.add_argument("--runs_root", default=None)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--offline", action="store_true",
                   help="Skip the checks that need the tokenizer (1, 2, and label decoding).")
    p.add_argument("--with_probe", action="store_true",
                   help="Also run probe_loss_norm.py (check 4). Needs torch + a model.")
    p.add_argument("--sample", type=int, default=25,
                   help="How many records to round-trip for checks 1 and 2.")
    return p.parse_args()


class Checks:
    def __init__(self):
        self.results: list[dict] = []

    def run(self, num: int, name: str, fn, skip: str | None = None):
        if skip:
            print(f"[{num}] SKIP  {name}  ({skip})")
            self.results.append({"check": num, "name": name, "status": "skipped",
                                 "reason": skip})
            return None
        try:
            detail = fn()
            print(f"[{num}] PASS  {name}"
                  + (f"  {json.dumps(detail, default=str)}" if detail else ""))
            self.results.append({"check": num, "name": name, "status": "pass",
                                 "detail": detail})
            return detail
        except Exception as e:
            print(f"[{num}] FAIL  {name}\n      {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            self.results.append({"check": num, "name": name, "status": "fail",
                                 "error": f"{type(e).__name__}: {e}"})
            return None

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.results if r["status"] == "fail"]


def main():
    args = parse_args()
    arm = resolve_arm(args.arm)
    d = run_dir(arm.name, args.runs_root)
    slot_path = d / "general_slot.jsonl"
    train_path = d / "socrateach_sft_train.jsonl"

    if not slot_path.exists():
        raise SystemExit(f"missing {slot_path}. Run: python build_general_slot.py --arm {arm.name}")
    # Idempotent; guarantees the manifest carries its header (sigma, delta, sampling
    # config, checkpoint grid) even if the slot was assembled out of band.
    poc = bool(manifest.load(d).get("poc", False))
    manifest.init(d, arm, poc=poc)
    general = manifest.read_jsonl(slot_path)
    ordered = manifest.read_jsonl(train_path) if train_path.exists() else None

    print(f"Acceptance checks for arm {arm.name}  ({d})")
    print(f"  general_slot: {len(general)} records"
          + (f" | train file: {len(ordered)} records" if ordered else " | train file: absent"))
    print("-" * 74)

    c = Checks()
    tokenizer = tok_fn = None
    tok_skip = "--offline" if args.offline else None
    if not args.offline:
        try:
            tokenizer = chat.load_tokenizer(args.base_model)
            tok_fn = chat.make_tokenize_fn(tokenizer, MAX_LEN)
        except Exception as e:
            tok_skip = f"tokenizer unavailable: {type(e).__name__}: {e}"

    # --- 1. label span round-trip ------------------------------------------
    def check1():
        sample = general[:args.sample]
        out = [chat.assert_label_span_roundtrip(tokenizer, tok_fn, r, MAX_LEN) for r in sample]
        return {"records": len(out),
                "mean_label_tokens": round(sum(o["n_label_tokens"] for o in out) / len(out), 1),
                "truncated": sum(1 for o in out if o["truncated"])}

    c.run(1, "unmasked label span == assistant content + EOS", check1, tok_skip)

    # --- 2. generation prompt == training prefix ---------------------------
    def check2():
        prompts = [[m for m in r["messages"] if m["role"] != "assistant"]
                   for r in general[:args.sample]]
        return chat.assert_prompt_invariant(tokenizer, prompts)

    c.run(2, "generation prompt is byte-identical to the training prefix (§4)",
          check2, tok_skip)

    # --- 3. system-message contract ----------------------------------------
    def check3():
        bad_gen = [r.get("dialogue_id") for r in general
                   if any(m["role"] == "system" for m in r["messages"])]
        if bad_gen:
            raise AssertionError(f"{len(bad_gen)} general records carry a system message: "
                                 f"{bad_gen[:5]}")
        detail = {"general_without_system": len(general)}
        if ordered:
            ped = [r for r in ordered if mixing.is_pedagogy(r)]
            bad_ped = [r.get("dialogue_id") for r in ped
                       if not any(m["role"] == "system" for m in r["messages"])]
            if bad_ped:
                raise AssertionError(f"{len(bad_ped)} pedagogy records lack a system "
                                     f"message: {bad_ped[:5]}")
            detail["pedagogy_with_system"] = len(ped)
        return detail

    c.run(3, "system message present <=> tutor mode", check3)

    # --- 4. loss-normalisation probe ---------------------------------------
    def check4():
        cmd = [sys.executable, str(IMPL4_ROOT / "probe_loss_norm.py"),
               "--arm", arm.name, "--model", args.base_model]
        if args.runs_root:
            cmd += ["--runs_root", args.runs_root]
        print("      + " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        return manifest.load(d).get("loss_normalization", {}).get("verdict")

    c.run(4, "loss-normalisation probe (§5)", check4,
          None if args.with_probe else "pass --with_probe to run it")

    # --- 5. block layout ----------------------------------------------------
    def check5():
        if ordered is None:
            raise AssertionError(f"missing {train_path}; run mix_and_order.py --arm {arm.name}")
        layout = mixing.verify_block_layout(ordered, PED_PER_BLOCK, GEN_PER_BLOCK)
        block = PED_PER_BLOCK + GEN_PER_BLOCK
        print("      first 3 blocks (P = pedagogy, g = general):")
        for b in range(min(3, layout["n_blocks"])):
            row = "".join("P" if mixing.is_pedagogy(r) else "g"
                          for r in ordered[b * block:(b + 1) * block])
            print(f"        block {b}: {row}")
            assert row == "P" * PED_PER_BLOCK + "g" * GEN_PER_BLOCK, \
                f"block {b} layout is {row}"
        return {"n_blocks": layout["n_blocks"], "layout": layout["layout"]}

    c.run(5, "ordered train file is 24 pedagogy + 8 general per block (§6)", check5)

    # --- 6. decontamination -------------------------------------------------
    def check6():
        idx = ngram.build_eval_index([MATH_EVAL_PROMPTS, GENERAL_EVAL_PROMPTS])
        hits = []
        for r in general:
            text = "\n".join(m["content"] for m in r["messages"])
            g = idx.hit(text)
            if g is not None:
                hits.append({"dialogue_id": r.get("dialogue_id"),
                             "task": r.get("superni_task_id"), "gram": " ".join(g)})
        if hits:
            raise AssertionError(f"{len(hits)} replay records overlap an eval prompt: "
                                 f"{hits[:3]}")
        return {"records_checked": len(general), "reference_prompts": idx.n_refs,
                "grams": len(idx), "hits": 0}

    c.run(6, "zero 13-gram overlap with math_logic_prompts / general_prompts", check6)

    # --- report -------------------------------------------------------------
    print("-" * 74)
    manifest.merge(d, "acceptance", {
        "checks": c.results,
        "n_pass": sum(1 for r in c.results if r["status"] == "pass"),
        "n_fail": len(c.failed),
        "n_skipped": sum(1 for r in c.results if r["status"] == "skipped"),
        "note": "Check 7 (--poc smoke run) is run separately; see RUNBOOK.md.",
    })
    if c.failed:
        print(f"{len(c.failed)} CHECK(S) FAILED — do not start a full run:")
        for r in c.failed:
            print(f"  [{r['check']}] {r['name']}: {r['error']}")
        raise SystemExit(1)
    print(f"All non-skipped checks passed. Manifest updated: {d / manifest.MANIFEST_NAME}")


if __name__ == "__main__":
    main()
