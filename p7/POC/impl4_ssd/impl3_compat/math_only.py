#!/usr/bin/env python
"""Math retention (and optionally KL) over the bridged checkpoints.

The prior-task axis, run on its own. This is the expensive one: 250 GSM8K items x 2
prompt conditions x up to 512 generated tokens is ~500 autoregressive generations per
checkpoint, so budget ~4 min each against ~10 s for the NLL axis.

``--with-kl`` adds ``kl_new_SI`` / ``kl_ped_noSI``. It is cheap (the base continuations
are generated once for the whole sweep; per checkpoint it is 2 forward passes over 64
prompts x 2 conditions) and it is what makes these numbers plottable on Impl 3's
KL-forgetting plane — math alone gives a y-axis with no x.

Everything measured here comes from Impl 3's own module, loaded by path: ``math_stats``,
``with_boxed_hint``, ``generate_batched``, ``discover``, and (for KL) ``base_continuations``
/ ``mean_kl_cached``. Nothing is reimplemented.

Rows carry the real ``protocol`` stamp when --with-kl is set AND the math item hash
matches, because then every field the stamp covers was actually measured. Without KL the
stamp is withheld and the rows are marked ``axis: "math"`` — a partial file must not merge
as though it were complete.

    python math_only.py --work . --runs 'out/*' --out out/math.jsonl [--with-kl]
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent


def load_their_module(work: pathlib.Path):
    for p in (str(work), str(work / "eval"), str(work / "eval" / "math_eval")):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "impl3_sweep", work / "eval" / "sweep_ckpt_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=str(HERE / "work"))
    p.add_argument("--runs", default="out/*")
    p.add_argument("--out", default="out/math.jsonl")
    p.add_argument("--base_model", default="allenai/OLMo-2-0425-1B-Instruct")
    p.add_argument("--math_prompts", default="eval/math_eval/math_logic_prompts.jsonl")
    p.add_argument("--val_file", default="data/socrateach_sft_val.jsonl")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--gen_max", type=int, default=512)
    p.add_argument("--kl_gen_max", type=int, default=200)
    p.add_argument("--n_kl", type=int, default=64)
    p.add_argument("--with_kl", action="store_true")
    p.add_argument("--require_epoch", type=float, default=0.99)
    return p.parse_args()


def main():
    args = parse_args()
    work = pathlib.Path(args.work).resolve()
    os.chdir(work)
    sweep = load_their_module(work)
    from common.modeling import load_for_inference

    math_rows = [json.loads(l) for l in open(args.math_prompts, encoding="utf-8") if l.strip()]
    bare = [r["prompt"] for r in math_rows]
    hint = [sweep.with_boxed_hint(r) for r in math_rows]
    print(f"math probe: {len(math_rows)} prompts x 2 conditions (bare, hint), no pedagogy SI")

    class _A:                       # measurement_protocol reads only .ifeval
        ifeval = False
    protocol = sweep.measurement_protocol(_A(), math_rows)
    print(f"protocol (full-sweep equivalent): {protocol}")

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

    def gen(model, prompts):
        return sweep.generate_batched(model, tok, device, prompts,
                                      batch=args.batch, gen_max=args.gen_max)

    cached_si = cached_no = None
    if args.with_kl:
        val = [json.loads(l) for l in open(args.val_file, encoding="utf-8") if l.strip()]
        kl_items = sweep.pedagogy_contexts(val, args.n_kl)
        print(f"KL probe: {len(kl_items)} contexts; pre-generating base continuations once ...")
        cached_si = sweep.base_continuations(base, tok, kl_items, True, gen_max=args.kl_gen_max)
        cached_no = sweep.base_continuations(base, tok, kl_items, False, gen_max=args.kl_gen_max)

    def emit(rec):
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    if ("base", 0) not in done:
        print("scoring base (once) ...", flush=True)
        b = {**sweep.math_stats(gen(base, bare), math_rows, "math_bare"),
             **sweep.math_stats(gen(base, hint), math_rows, "math_hint")}
        b.update(run="base", step=0, variant=None, temperature=None,
                 prior_score=b["math_hint"], math_bare_forget=0.0, math_hint_forget=0.0)
        if args.with_kl:
            b.update(kl_new_SI=0.0, kl_ped_noSI=0.0, protocol=protocol)
        else:
            b["axis"] = "math"
        emit(b)
        base_stats = b
        print(f"base: bare={b['math_bare']:.3f} hint={b['math_hint']:.3f} "
              f"deflect={b['math_hint_deflect']:.3f}", flush=True)
    else:
        base_stats = next(json.loads(l) for l in open(args.out, encoding="utf-8")
                          if json.loads(l)["run"] == "base")

    for i, (run, step, path) in enumerate(todo, 1):
        try:
            sft, _, _ = load_for_inference(args.base_model, adapter_dir=path, merge=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"  SKIP {run}@{step} (load failed): {e}")
            continue
        rec = {"run": run, "step": step, "variant": None, "temperature": None}
        rec.update(sweep.math_stats(gen(sft, bare), math_rows, "math_bare"))
        rec.update(sweep.math_stats(gen(sft, hint), math_rows, "math_hint"))
        for cond in ("math_bare", "math_hint"):
            rec[f"{cond}_forget"] = base_stats[cond] - rec[cond]
        rec["prior_score"] = rec["math_hint"]
        if args.with_kl:
            rec["kl_new_SI"] = sweep.mean_kl_cached(base, sft, cached_si)
            rec["kl_ped_noSI"] = sweep.mean_kl_cached(base, sft, cached_no)
            rec["protocol"] = protocol
        else:
            rec["axis"] = "math"
        emit(rec)
        print(f"[{i}/{len(todo)}] {run} step {step:>4}  bare={rec['math_bare']:.3f} "
              f"hint={rec['math_hint']:.3f} deflect={rec['math_hint_deflect']:.3f}"
              + (f" kl={rec['kl_new_SI']:.3f}" if args.with_kl else ""), flush=True)
        del sft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
