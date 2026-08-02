#!/usr/bin/env python
"""Pedagogy-NLL only, over the bridged checkpoints. ~40s per checkpoint.

The full sweep is ~5 min per checkpoint because the math probe generates 500 completions
of up to 512 tokens. ``ped_nll`` is 128 *forward passes* and nothing else, so running it
alone is ~8x cheaper and answers the new-task-fit question on its own.

``pedagogy_nll`` and the tokenizer setup are taken from Impl 3's own files rather than
reimplemented — the function is loaded straight out of ``eval/sweep_ckpt_eval.py`` by
path, so this measures exactly what their column measures.

**These rows are deliberately not protocol-stamped like a full sweep row.** The stamp
covers the KL context rule and the math conditions, neither of which ran here; claiming
it would let an NLL-only file merge into a results file as though it were complete. They
carry ``axis: "ped_nll"`` instead, and ``compare.py`` will refuse to merge them — which is
correct. ``ped_nll`` itself is directly comparable to theirs.

Usage (from the compat workdir):
    python nll_only.py --runs 'out/*' --out out/ped_nll.jsonl
"""

from __future__ import annotations

import argparse
import gc
import glob
import importlib.util
import json
import os
import pathlib
import re
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent


def load_their_module(work: pathlib.Path):
    """Import their sweep driver by path, for pedagogy_nll and discover()."""
    for p in (str(work), str(work / "eval"), str(work / "eval" / "math_eval")):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "impl3_sweep", work / "eval" / "sweep_ckpt_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)          # main() is __main__-guarded, so nothing runs
    return mod


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=str(HERE / "work"))
    p.add_argument("--runs", default="out/*")
    p.add_argument("--out", default="out/ped_nll.jsonl")
    p.add_argument("--base_model", default="allenai/OLMo-2-0425-1B-Instruct")
    p.add_argument("--val_file", default="data/socrateach_sft_val.jsonl")
    p.add_argument("--n_nll", type=int, default=128)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--require_epoch", type=float, default=0.99)
    return p.parse_args()


def main():
    args = parse_args()
    work = pathlib.Path(args.work).resolve()
    os.chdir(work)
    sweep = load_their_module(work)

    from common.chat import make_tokenize_fn
    from common.modeling import load_for_inference

    todo = sweep.discover(args.runs, args.require_epoch)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["run"], r["step"]))
    todo = [t for t in todo if (t[0], t[1]) not in done]
    print(f"{len(todo)} checkpoints to score ({len(done)} already done)")

    base, tok, device = load_for_inference(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    val = [json.loads(l) for l in open(args.val_file, encoding="utf-8") if l.strip()]
    tok_fn = make_tokenize_fn(tok, args.max_len)
    items = []
    for r in val[:args.n_nll]:
        ex = tok_fn(r)
        if any(t != -100 for t in ex["labels"]):
            items.append((torch.tensor([ex["input_ids"]], device=device),
                          torch.tensor([ex["labels"]], device=device)))
    print(f"pedagogy NLL over {len(items)} held-out dialogues")

    def emit(rec):
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    if ("base", 0) not in done:
        v = sweep.pedagogy_nll(base, items)
        print(f"base ped_nll = {v:.6f}")
        emit({"run": "base", "step": 0, "variant": None, "temperature": None,
              "ped_nll": v, "axis": "ped_nll", "n_nll": len(items)})

    for i, (run, step, path) in enumerate(todo, 1):
        try:
            sft, _, _ = load_for_inference(args.base_model, adapter_dir=path, merge=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"  SKIP {run}@{step} (load failed): {e}")
            continue
        v = sweep.pedagogy_nll(sft, items)
        print(f"[{i}/{len(todo)}] {run} step {step:>4}  ped_nll={v:.6f}", flush=True)
        emit({"run": run, "step": step, "variant": None, "temperature": None,
              "ped_nll": v, "axis": "ped_nll", "n_nll": len(items)})
        del sft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
