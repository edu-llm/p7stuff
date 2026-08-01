#!/usr/bin/env python
"""Headless driver: the notebook's stages, runnable on a Colab CLI runtime.

``impl4_ssd_colab.ipynb`` is the interactive path. Two of its cells cannot run
head-less — ``drive.mount()`` and the ``files.upload()`` widget both block on a browser
— so ``colab exec -f impl4_ssd_colab.ipynb`` would hang. This is the same pipeline with
those two removed: the bundle is expected to be already on the VM (``colab upload``) and
results are pulled off with ``colab download`` rather than written to Drive.

Everything else is identical, because both are thin wrappers over the same scripts.

Intended use, from a laptop with the Colab CLI authenticated::

    colab new -s impl4 --gpu A100
    colab upload -s impl4 impl3_handoff.tar.gz /content/impl3_handoff.tar.gz
    colab exec  -s impl4 -f bootstrap.py        # clones the repo, launches this under nohup
    # poll:
    colab exec  -s impl4 <<< "print(open('/content/run.log').read()[-4000:])"
    # collect:
    colab download -s impl4 /content/artifacts.tar.gz .

Run it directly on the VM as::

    python run_matched.py --arm A1 --poc
    python run_matched.py --arm A1                     # the 923-step gate
    python run_matched.py --arm A1 --stages data,train  # partial

Stages, in order: ``deps, bundle, pool, superni, slot, mix, checks, train, bridge, eval,
compare``. ``data`` is an alias for ``pool,superni,slot,mix,checks``. A stage that has
already produced its output is skipped, so re-running after a crash resumes.
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

HERE = Path(__file__).resolve().parent          # impl4_ssd/
POC_ROOT = HERE.parent

PINS = ["transformers==5.14.1", "datasets==5.0.1", "accelerate==1.14.0", "peft==0.20.0",
        "huggingface_hub==1.25.1", "numpy==2.4.6", "langdetect==1.0.9",
        "pyarrow==25.0.0", "matplotlib==3.11.1"]

ALL_STAGES = ("deps", "bundle", "pool", "superni", "slot", "mix", "checks",
              "train", "bridge", "eval", "compare")
DATA_STAGES = ("pool", "superni", "slot", "mix", "checks")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="A1")
    p.add_argument("--poc", action="store_true", help="63-block rehearsal instead of 923.")
    p.add_argument("--stages", default="all",
                   help=f"Comma-separated subset of: {', '.join(ALL_STAGES)} ('data' = "
                        f"{','.join(DATA_STAGES)}).")
    p.add_argument("--bundle_tar", default="/content/impl3_handoff.tar.gz")
    p.add_argument("--bundle", default="/content/impl3_handoff")
    p.add_argument("--runs_root", default=None, help="Default: impl4_ssd/runs[_poc].")
    p.add_argument("--eval_steps", default=None,
                   help="Restrict the eval to these steps, e.g. '1,2,3,4,8,16,32,64,128,256,512,923'.")
    p.add_argument("--eval_batch", type=int, default=32)
    p.add_argument("--per_device_batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_batch_tokens", type=int, default=32768)
    p.add_argument("--artifacts", default="/content/artifacts.tar.gz")
    return p.parse_args()


# ---------------------------------------------------------------------------
def sh(cmd: str, cwd: Path = HERE, check: bool = True, log_path: Path | None = None) -> int:
    """Run a command, streaming to stdout so the tailed log shows progress live.

    ``log_path`` tees in Python rather than shelling out to ``tee``: a ``cmd | tee f``
    pipeline reports *tee's* exit status, so a failed training run would come back 0 and
    the driver would sail on to the eval stage with no checkpoints.
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


# Stages that put work on the accelerator. The rest (pool build, mix, bridge, compare)
# are CPU-only, so a GPU-less box can still run them — useful for re-running `compare`
# on a laptop after pulling the results.
GPU_STAGES = {"slot", "checks", "train", "eval"}


def gpu_report(required: bool) -> None:
    try:
        import torch
    except ImportError:
        raise SystemExit("torch missing; run the 'deps' stage first")
    avail = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if avail else "no CUDA"
    bf16 = avail and torch.cuda.is_bf16_supported()
    print(f"GPU: {name} | torch {torch.__version__} | bf16={bf16}", flush=True)
    if required and not avail:
        raise SystemExit("no GPU visible — provision one with `colab new --gpu A100`")
    if not bf16 and avail:
        print("  NOTE: no bf16 on this GPU, so generation falls back to fp16 and these "
              "checkpoints are not bit-comparable with a bf16 (A100/H100/H200) run.",
              flush=True)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    want = set(ALL_STAGES) if args.stages == "all" else set()
    for s in ([] if args.stages == "all" else args.stages.split(",")):
        want |= set(DATA_STAGES) if s.strip() == "data" else {s.strip()}
    unknown = want - set(ALL_STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}")

    poc = "--poc" if args.poc else ""
    mode = "poc" if args.poc else "full"
    runs_root = Path(args.runs_root) if args.runs_root else HERE / (
        "runs_poc" if args.poc else "runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    rr = shlex.quote(str(runs_root))
    compat = HERE / "impl3_compat" / "work"

    print(f"impl4_ssd headless run | arm={args.arm} mode={mode} runs_root={runs_root}")
    print(f"stages: {[s for s in ALL_STAGES if s in want]}")

    if "deps" in want:
        banner("deps")
        sh(f"{sys.executable} -m pip -q install " + " ".join(shlex.quote(p) for p in PINS))
        # Impl 3's environment note: do NOT have torchao around, an old version breaks
        # peft.get_peft_model. Colab preinstalls 0.10.0, and peft 0.20.0 hard-raises on
        # anything below 0.16 from inside its LoRA dispatcher — so training dies at
        # get_peft_model and the loss-norm probe silently falls back to an unwrapped model,
        # which is the one configuration whose verdict does not answer PLAN §5. Warning
        # about it was not enough; remove it.
        sh(f"{sys.executable} -m pip -q uninstall -y torchao", check=False)
        sh(f'{sys.executable} -c "'
           f'import importlib.util as u;'
           f'print(\'torchao still present -- training will fail\' if u.find_spec(\'torchao\') '
           f'else \'torchao absent (good)\')"')
    gpu_report(required=bool(want & GPU_STAGES))

    if "bundle" in want:
        banner("bundle — extract + verify the Impl 3 assets")
        if not Path(args.bundle, "eval/sweep_ckpt_eval.py").exists():
            if not Path(args.bundle_tar).exists():
                raise SystemExit(
                    f"{args.bundle_tar} not on the VM. Send it first:\n"
                    f"    colab upload impl3_handoff.tar.gz {args.bundle_tar}")
            sh(f"tar xzf {shlex.quote(args.bundle_tar)} -C {shlex.quote(str(Path(args.bundle).parent))}")
        # Hard-stops on any hash mismatch: SI, math item ids, val split.
        sh(f"{sys.executable} impl3_compat/setup_compat.py --bundle {shlex.quote(args.bundle)}")

    if "pool" in want:
        banner("stage 1 — pedagogy pool (pinned Hub revision)")
        sh(f"{sys.executable} build_pedagogy_pool.py")
        src = json.loads((HERE / "data/pedagogy_pool/pool_source.json").read_text())
        if not src.get("comparable_to_impl3"):
            raise SystemExit("pedagogy pool has regenerated SIs — not comparable to Impl 3")
        print(f"  pool source: {src['mode']} {src.get('dataset')} {(src.get('revision') or '')[:12]}")

    # A1's replay slot is Tulu gold, so it needs no SuperNI pool — skip the slowest data
    # stage unless an arm actually draws on it.
    needs_superni = args.arm.upper() not in ("A1",)
    if "superni" in want and needs_superni:
        banner("stage 2 — SuperNI prompt pool")
        if not (HERE / "data/superni_pool.jsonl").exists():
            sh(f"{sys.executable} build_prompt_pool.py --min_gold_words 25 "
               f"--instances_per_task 900")
        else:
            print("  pool already present")
    elif "superni" in want:
        print(f"\n(skipping the SuperNI pool: arm {args.arm} does not draw on it)")

    slot_args = (f"--runs_root {rr} --token_reference a1 {poc} --backend hf "
                 f"--batch_size 32 --max_batch_tokens {args.max_batch_tokens}")
    if "slot" in want:
        banner("stage 3 — replay slot")
        if not (HERE / "data/tulu_reference.json").exists():
            sh(f"{sys.executable} build_general_slot.py --arm A1 {slot_args}")
        if args.arm.upper() != "A1":
            sh(f"{sys.executable} build_general_slot.py --arm {args.arm} {slot_args}")

    if "mix" in want:
        banner("stage 4 — mix and order")
        sh(f"{sys.executable} mix_and_order.py --arm {args.arm} --runs_root {rr} {poc}")

    if "checks" in want:
        banner("stage 5 — acceptance checks (PLAN §11)")
        sh(f"{sys.executable} acceptance_checks.py --arm {args.arm} --runs_root {rr} --with_probe")

    if "train" in want:
        banner("stage 6 — train")
        out = runs_root / args.arm
        out.mkdir(parents=True, exist_ok=True)
        sh(f"{sys.executable} train_sft_impl4.py --arm {args.arm} --runs_root {rr} {poc} "
           f"--resume auto --per_device_batch {args.per_device_batch} "
           f"--grad_accum {args.grad_accum} --save_steps 100 --save_total_limit 1",
           log_path=out / "train.log")

    if "bridge" in want:
        banner("stage 7a — expose checkpoints in Impl 3's layout")
        cmd = (f"{sys.executable} impl3_compat/bridge.py --runs_root {rr} "
               f"--workdir {shlex.quote(str(compat))} --arms {args.arm}")
        if args.eval_steps:
            cmd += f" --steps {args.eval_steps}"
        sh(cmd)

    results = compat / "out" / "ckpt_sweep_impl4.jsonl"
    if "eval" in want:
        banner("stage 7b — Impl 3's eval driver (KL / GSM8K / ped-NLL)")
        sh(f"{sys.executable} eval/sweep_ckpt_eval.py --runs 'out/*' "
           f"--out out/ckpt_sweep_impl4.jsonl --batch {args.eval_batch}", cwd=compat)

    if "compare" in want and results.exists():
        banner("stage 8 — A1 gate + merged figure")
        their = Path(args.bundle) / "out/ckpt_sweep_bare_hint250.jsonl"
        rc = sh(f"{sys.executable} impl3_compat/compare.py "
                f"--impl3 {shlex.quote(str(their))} --impl4 {shlex.quote(str(results))} "
                f"--out {shlex.quote(str(compat / 'out/merged.jsonl'))} "
                f"--fig {shlex.quote(str(compat / 'out/impl3_vs_impl4.png'))} "
                f"--gate_arm impl4-{args.arm}", check=False)
        print("\nGATE PASSED" if rc == 0 else "\nGATE FLAGGED — read the table above")

    # One tarball to pull off the VM: results, manifests, logs, figure. Checkpoints are
    # deliberately excluded (22 x 25 MB per arm); fetch those separately if wanted.
    banner("packaging artifacts")
    keep = []
    for pat in (f"{runs_root.name}/{args.arm}/manifest.json",
                f"{runs_root.name}/{args.arm}/checkpoint_index.json",
                f"{runs_root.name}/{args.arm}/train.log",
                "impl3_compat/work/out/ckpt_sweep_impl4.jsonl",
                "impl3_compat/work/out/merged.jsonl",
                "impl3_compat/work/out/impl3_vs_impl4.png",
                "impl3_compat/work/compat_setup.json",
                "data/pedagogy_pool/pool_source.json"):
        if (HERE / pat).exists():
            keep.append(pat)
    if keep:
        sh("tar czf " + shlex.quote(args.artifacts) + " " + " ".join(shlex.quote(k) for k in keep))
        print(f"\ncollect with:\n    colab download {args.artifacts} .")
    for k in keep:
        print(f"  {k}")
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
