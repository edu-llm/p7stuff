#!/usr/bin/env python
"""Headless driver for one Impl 5 arm on a Colab CLI runtime.

Mirrors ``impl4_ssd/run_matched.py``, with two differences that matter operationally.

**The cheap checks run before the expensive pass.** ``checks_fast`` needs a tokenizer and
~40 s; the distillation pass needs a GPU and over an hour. Ordering them the other way round
means a broken prefix invariant is discovered after the hour, not before it.

**Checkpoints are packaged per-arm the moment training ends**, with ``tar cf`` rather than
``tar czf``. Gzipping 1.1 GB of adapter safetensors takes ~15 minutes and compresses them by
almost nothing, and a runtime reclaimed during that window takes the whole run with it.

Stages, in order::

    deps, bundle, pool, checks_fast, distill, slot, mix, checks_full, train, bridge, eval

``eval`` is **pedagogy-NLL only** — ``impl3_compat/nll_only.py``, ~40 s per checkpoint. The
math and KL axes are deliberately not run here; they are ~4 min per checkpoint and this run
is budgeted for training. The rows are stamped ``axis: "ped_nll"`` so a partial file cannot
merge into a results file as though it were complete.

A stage whose output already exists is skipped, so re-running after a crash resumes.

    python run_impl5.py --arm D4
    python run_impl5.py --arm D4 --poc
    python run_impl5.py --arm D4 --stages distill,slot,mix
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC_ROOT = HERE.parent
IMPL4 = POC_ROOT / "impl4_ssd"
COMPAT = IMPL4 / "impl3_compat"

PINS = ["transformers==5.14.1", "datasets==5.0.1", "accelerate==1.14.0", "peft==0.20.0",
        "huggingface_hub==1.25.1", "numpy==2.4.6", "langdetect==1.0.9",
        "pyarrow==25.0.0", "matplotlib==3.11.1"]

ALL_STAGES = ("deps", "bundle", "pool", "checks_fast", "distill", "slot", "mix",
              "checks_full", "train", "bridge", "eval")
GPU_STAGES = {"distill", "slot", "mix", "train", "eval"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="D4")
    p.add_argument("--poc", action="store_true", help="63-block rehearsal instead of 923.")
    p.add_argument("--stages", default="all")
    p.add_argument("--runs_root", default=None)
    p.add_argument("--bundle_tar", default="/content/impl3_handoff.tar.gz")
    p.add_argument("--bundle", default="/content/impl3_handoff")
    p.add_argument("--distill_limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=0, help="0 = size from GPU memory.")
    p.add_argument("--max_batch_tokens", type=int, default=0)
    p.add_argument("--per_device_batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--artifacts_dir", default="/content")
    p.add_argument("--skip_package", action="store_true")
    return p.parse_args()


def sh(cmd: str, cwd: Path = HERE, check: bool = True, log_path: Path | None = None) -> int:
    """Run a command, streaming to stdout so a tailed log shows progress live.

    Tees in Python rather than shelling out: ``cmd | tee f`` reports *tee's* exit status, so
    a failed training run would come back 0 and the driver would sail on to eval with no
    checkpoints.
    """
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    fh = open(log_path, "a", encoding="utf-8") if log_path else None
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONUNBUFFERED="1",
                                     TOKENIZERS_PARALLELISM="false"))
    for line in proc.stdout:
        print(line, end="", flush=True)
        if fh:
            fh.write(line)
    rc = proc.wait()
    if fh:
        fh.close()
    print(f"[exit {rc} in {(time.time() - t0) / 60:.1f} min]", flush=True)
    if check and rc:
        raise SystemExit(f"stage failed (exit {rc}): {cmd}")
    return rc


def banner(text: str) -> None:
    print("\n" + "=" * 74 + f"\n== {text}\n" + "=" * 74, flush=True)


def gpu_report(required: bool) -> int:
    """Print the device and return its total memory in GiB (0 if there is none)."""
    try:
        import torch
    except ImportError:
        raise SystemExit("torch missing; run the 'deps' stage first")
    if not torch.cuda.is_available():
        if required:
            raise SystemExit("no GPU visible — provision one with `colab new --gpu A100`")
        print("GPU: none (CPU-only stages)")
        return 0
    props = torch.cuda.get_device_properties(0)
    gib = int(props.total_memory / 2**30)
    bf16 = torch.cuda.is_bf16_supported()
    print(f"GPU: {props.name} | {gib} GiB | torch {torch.__version__} | bf16={bf16}",
          flush=True)
    if not bf16:
        print("  NOTE: no bf16 — generation falls back to fp16 and these checkpoints are "
              "not bit-comparable with a bf16 (A100/H100) run.", flush=True)
    return gib


def sizes_for(gib: int, args) -> tuple[int, int]:
    """Generation batch geometry, from the card rather than from hope.

    Attention memory grows as ``B·L²`` during prefill and the KV cache as ``B·L``, and the
    rewriting prompts run to ~1,250 tokens at the tail. A row count alone does not bound
    either, so both a row cap and a padded-token cap are set, and both are scaled to the
    device: a 40 GiB A100 and an 80 GiB one want very different numbers, and guessing high
    on the smaller card costs an OOM an hour into the pass.
    """
    if args.batch_size and args.max_batch_tokens:
        return args.batch_size, args.max_batch_tokens
    if gib >= 60:
        b, t = 128, 196608
    elif gib >= 30:
        b, t = 64, 98304
    else:
        b, t = 24, 32768
    return args.batch_size or b, args.max_batch_tokens or t


def main():
    args = parse_args()
    want = set(ALL_STAGES) if args.stages == "all" else {
        s.strip() for s in args.stages.split(",")}
    unknown = want - set(ALL_STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}")

    poc = "--poc" if args.poc else ""
    runs_root = Path(args.runs_root) if args.runs_root else HERE / (
        "runs_poc" if args.poc else "runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    rr = shlex.quote(str(runs_root))
    out_dir = runs_root / args.arm
    art = Path(args.artifacts_dir)

    print(f"impl5_ssd headless run | arm={args.arm} "
          f"mode={'poc' if args.poc else 'full'} runs_root={runs_root}")
    print(f"stages: {[s for s in ALL_STAGES if s in want]}")

    if "deps" in want:
        banner("deps")
        sh(f"{sys.executable} -m pip -q install " + " ".join(shlex.quote(p) for p in PINS))
        # Colab preinstalls torchao 0.10.0 and peft 0.20.0 hard-raises on anything below
        # 0.16 from inside its LoRA dispatcher, so get_peft_model dies. Warning about it was
        # not enough last time — the probe caught the ImportError, said "could not PEFT-wrap"
        # and returned a verdict for an unwrapped model. Remove it.
        sh(f"{sys.executable} -m pip -q uninstall -y torchao", check=False)
        sh(f'{sys.executable} -c "import importlib.util as u; print('
           f"'torchao still present -- training will fail' if u.find_spec('torchao') "
           f"else 'torchao absent (good)'"
           f')"')
    gib = gpu_report(required=bool(want & GPU_STAGES))
    batch, max_batch_tokens = sizes_for(gib, args)

    if "bundle" in want:
        banner("bundle — extract + verify the Impl 3 assets")
        if not Path(args.bundle, "eval/sweep_ckpt_eval.py").exists():
            if not Path(args.bundle_tar).exists():
                raise SystemExit(f"{args.bundle_tar} not on the VM. Send it first:\n"
                                 f"    colab upload impl3_handoff.tar.gz {args.bundle_tar}")
            sh(f"tar xzf {shlex.quote(args.bundle_tar)} -C "
               f"{shlex.quote(str(Path(args.bundle).parent))}")
        sh(f"{sys.executable} impl3_compat/setup_compat.py --bundle "
           f"{shlex.quote(args.bundle)}", cwd=IMPL4)

    if "pool" in want:
        banner("stage 1 — pedagogy pool (Impl 4's, pinned Hub revision)")
        sh(f"{sys.executable} build_pedagogy_pool.py", cwd=IMPL4)
        src = json.loads((IMPL4 / "data/pedagogy_pool/pool_source.json").read_text())
        if not src.get("comparable_to_impl3"):
            raise SystemExit("pedagogy pool has regenerated SIs — not comparable")
        print(f"  pool source: {src.get('mode')} {src.get('dataset')} "
              f"{(src.get('revision') or '')[:12]}")

    if "checks_fast" in want:
        banner("acceptance checks (fast) — BEFORE the expensive pass")
        sh(f"{sys.executable} acceptance_checks5.py --stage fast "
           f"--out {shlex.quote(str(HERE / 'data/acceptance_fast.json'))}")

    if "distill" in want:
        banner(f"stage 2 — the distillation pass (batch {batch}, "
               f"{max_batch_tokens} padded tokens)")
        lim = f"--limit {args.distill_limit}" if args.distill_limit else ""
        sh(f"{sys.executable} distill_pedagogy.py --batch_size {batch} "
           f"--max_batch_tokens {max_batch_tokens} {lim}",
           log_path=HERE / "data/distill.log")

    if "slot" in want:
        banner("stage 3 — replay slot (Tulu-3 gold, reproducing impl4-A1)")
        sh(f"{sys.executable} build_general_slot5.py --arm {args.arm} --runs_root {rr} {poc}")

    if "mix" in want:
        banner("stage 4 — substitute distilled targets, order into 24/8 blocks")
        sh(f"{sys.executable} mix_arm5.py --arm {args.arm} --runs_root {rr} {poc}")

    if "checks_full" in want:
        banner("acceptance checks (full)")
        sh(f"{sys.executable} acceptance_checks5.py --stage full --arm {args.arm} "
           f"--runs_root {rr} "
           f"--out {shlex.quote(str(HERE / 'data/acceptance_full.json'))}")

    if "train" in want:
        banner("stage 5 — train")
        out_dir.mkdir(parents=True, exist_ok=True)
        sh(f"{sys.executable} train_sft_impl5.py --arm {args.arm} --runs_root {rr} {poc} "
           f"--resume auto --per_device_batch {args.per_device_batch} "
           f"--grad_accum {args.grad_accum} --save_steps 100 --save_total_limit 1",
           log_path=out_dir / "train.log")
        if not args.skip_package:
            # Immediately, and uncompressed. See the module docstring.
            #
            # `cd` into the runs root rather than using `tar -C <root> D4/ckpt-*`: the shell
            # expands the glob in the *current* directory, not in -C's, so the pattern would
            # not match, tar would be handed the literal string, and the tarball would come
            # out holding the manifest and nothing else.
            tarball = art / f"impl5_{args.arm}.tar"
            sh(f"cd {rr} && tar cf {shlex.quote(str(tarball))} "
               f"--exclude='checkpoint-*' --exclude='*.jsonl' {shlex.quote(args.arm)}",
               check=False)
            sh(f"ls -la {shlex.quote(str(tarball))}", check=False)
            print(f"  packaged -> {tarball}", flush=True)

    if "bridge" in want:
        banner("stage 6 — expose checkpoints in Impl 3's layout")
        sh(f"{sys.executable} impl3_compat/bridge.py --runs_root {rr} "
           f"--prefix impl5- --arms {args.arm}", cwd=IMPL4)

    if "eval" in want:
        banner("stage 7 — pedagogy NLL (ONLY; no math, no KL)")
        sh(f"{sys.executable} nll_only.py --runs 'out/*' --out out/ped_nll_impl5.jsonl",
           cwd=COMPAT)

    if not args.skip_package:
        banner("packaging results")
        results = COMPAT / "work" / "out" / "ped_nll_impl5.jsonl"
        keep = []
        for src, name in ((results, "ped_nll_impl5.jsonl"),
                          (HERE / "data/distill_meta.json", "distill_meta.json"),
                          (HERE / "data/acceptance_fast.json", "acceptance_fast.json"),
                          (HERE / "data/acceptance_full.json", "acceptance_full.json"),
                          (HERE / "data/pedagogy_reference.json", "pedagogy_reference.json"),
                          (out_dir / "manifest.json", f"{args.arm}_manifest.json"),
                          (out_dir / "checkpoint_index.json", f"{args.arm}_ckpt_index.json"),
                          (out_dir / "train.log", f"{args.arm}_train.log")):
            if Path(src).exists():
                dst = art / "impl5_results" / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(Path(src).read_bytes())
                keep.append(name)
        if keep:
            sh(f"tar czf {shlex.quote(str(art / 'impl5_results.tar.gz'))} "
               f"-C {shlex.quote(str(art))} impl5_results", check=False)
        for k in keep:
            print(f"  {k}")
        print(f"\ncollect with:\n    colab download {art}/impl5_results.tar.gz .\n"
              f"    colab download {art}/impl5_{args.arm}.tar .")
    print("\nIMPL5_RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
