#!/usr/bin/env python
"""Assemble a workdir where Impl 3's own eval driver can run against Impl 4's checkpoints.

Impl 4 does not reimplement the three axes (KL / GSM8K / pedagogy NLL). It runs
``eval/sweep_ckpt_eval.py`` **from the Impl-3 comparability bundle, unmodified**, which is the
only way to be sure the arithmetic matches rather than merely resembling theirs.

The bundle is missing two modules its own code imports (``common.modeling``, ``common.chat``)
and one prompt file (``common/prompts/impl1_system_prompt.txt``). This script builds:

    <workdir>/
      common/{kl,system_instructions}.py      <- copied from the bundle, byte-identical
      common/prompts/canonical_si.txt         <- copied, hash-verified (the +SI string)
      common/prompts/impl1_system_prompt.txt  <- PLACEHOLDER (absent from the bundle, unused)
      common/{modeling,chat}.py               <- shims from ./shims/
      common/_impl4_root.txt                  <- lets the shims import impl4.*
      eval/**                                 <- copied from the bundle, incl. the 250-item set
      data/socrateach_sft_val.jsonl           <- copied, hash-verified (pins the 64/128 items)
      out/                                    <- results land here; bridge.py adds checkpoints

Everything is *copied* rather than symlinked on purpose: the driver resolves ``__file__`` to
locate its project root, and a symlinked ``eval/`` would resolve back into the bundle, making it
read the bundle's ``common/`` (no shims) and write results into the bundle's ``out/``.

Three hashes are verified after assembly and the script refuses to continue on any mismatch —
each one silently invalidates a different axis:

    canonical_si.txt   -> kl_new_SI is measured against a different string
    math item ids      -> math_bare / math_hint are a different probe
    val split          -> the 64 KL contexts and 128 NLL dialogues are different items

Usage:
    python impl3_compat/setup_compat.py
    python impl3_compat/setup_compat.py --bundle /content/impl3_handoff --workdir /content/impl3_work
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPL4_ROOT = HERE.parent

# Verified against the values Impl 3 published in BUNDLE_README.md.
EXPECT = {
    "canonical_si_sha256": "e2bde3bbfdb8d6a56856b73f393b55606b8c54af7d413412b65fdf1c6f469e12",
    "math_ids_sha1_8": "995cd590",
    "val_sha256": "23d4ee3c75384aa6d362750044abdbc9c8e97808bb8acf489e3779127c2df258",
}
EXPECT_PROTOCOL = "kl=ctx-first-turn;math=bare+hint@250/995cd590;ifeval=off"

# The bundle omits this file, and IMPL1_SYSTEM_PROMPT (which reads it at import time) is
# defined-but-never-used across the whole bundle. Rather than patch their module, supply a
# placeholder that would be unmissable if it ever reached a model.
IMPL1_PLACEHOLDER = """PLACEHOLDER -- impl1_system_prompt.txt was not part of the Impl-3 comparability bundle.

common/system_instructions.py reads this file at import time to define IMPL1_SYSTEM_PROMPT,
which nothing in the bundle uses (the KL / math / NLL paths use CANONICAL_SI only). If this
text ever appears in a model prompt or a generation, something IS using the Impl-1 prompting
artifact and you must obtain the real file from the Impl-3 repo before trusting the result.
"""

BUNDLE_COPY = [
    ("common/kl.py", "common/kl.py"),
    ("common/system_instructions.py", "common/system_instructions.py"),
    ("common/prompts/canonical_si.txt", "common/prompts/canonical_si.txt"),
    ("data/socrateach_sft_val.jsonl", "data/socrateach_sft_val.jsonl"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default=os.environ.get("IMPL3_BUNDLE"),
                   help="Extracted impl3_handoff/ (default: $IMPL3_BUNDLE, else searched for).")
    p.add_argument("--workdir", default=str(HERE / "work"),
                   help="Where to assemble the runnable tree.")
    p.add_argument("--force", action="store_true", help="Re-copy over an existing workdir.")
    return p.parse_args()


def find_bundle(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not (p / "eval" / "sweep_ckpt_eval.py").exists():
            raise SystemExit(f"{p} does not look like the bundle (no eval/sweep_ckpt_eval.py)")
        return p
    # Walk up from impl4_ssd/ looking for a sibling impl3_handoff/ at any level — the p7 tree
    # nests p7/OLMo-core/p7/POC, so a fixed parents[n] index is wrong in one layout or the other.
    candidates = [p / "impl3_handoff" for p in (IMPL4_ROOT, *IMPL4_ROOT.parents)]
    candidates += [Path("/content/impl3_handoff"), Path.cwd() / "impl3_handoff"]
    for c in candidates:
        if (c / "eval" / "sweep_ckpt_eval.py").exists():
            return c.resolve()
    raise SystemExit(
        "cannot find the Impl-3 bundle. Extract impl3_handoff.tar.gz and pass --bundle PATH "
        "(or set IMPL3_BUNDLE). Looked in:\n  " + "\n  ".join(str(c) for c in candidates))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def math_ids_hash(path: Path) -> str:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return hashlib.sha1(";".join(sorted(r["id"] for r in rows)).encode()).hexdigest()[:8], len(rows)


def main():
    args = parse_args()
    bundle = find_bundle(args.bundle)
    work = Path(args.workdir).expanduser().resolve()
    print(f"bundle : {bundle}")
    print(f"workdir: {work}")

    if work.exists() and not args.force:
        print(f"{work} already assembled. Use --force to re-copy.")
    else:
        for sub in ("common/prompts", "data", "out"):
            (work / sub).mkdir(parents=True, exist_ok=True)
        for src_rel, dst_rel in BUNDLE_COPY:
            src, dst = bundle / src_rel, work / dst_rel
            if not src.exists():
                raise SystemExit(f"bundle is missing {src_rel}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copied {src_rel}")
        # eval/ is copied wholesale so __file__ resolution stays inside the workdir.
        if (work / "eval").exists():
            shutil.rmtree(work / "eval")
        shutil.copytree(bundle / "eval", work / "eval")
        print(f"  copied eval/ ({sum(1 for _ in (work / 'eval').rglob('*') if _.is_file())} files)")

        (work / "common/prompts/impl1_system_prompt.txt").write_text(
            IMPL1_PLACEHOLDER, encoding="utf-8")
        print("  wrote common/prompts/impl1_system_prompt.txt (PLACEHOLDER — absent from bundle)")

        for name in ("modeling.py", "chat.py"):
            shutil.copy2(HERE / "shims" / name, work / "common" / name)
            print(f"  installed shim common/{name}")
        (work / "common/_impl4_root.txt").write_text(str(IMPL4_ROOT) + "\n", encoding="utf-8")

    # ---- verification -----------------------------------------------------------------
    print("\nverifying the assets that define the axes:")
    failures = []

    si = work / "common/prompts/canonical_si.txt"
    got = sha256(si)
    ok = got == EXPECT["canonical_si_sha256"]
    print(f"  canonical_si.txt  {got[:16]}  {'OK' if ok else 'MISMATCH'}  ({si.stat().st_size} bytes)")
    if not ok:
        failures.append("canonical_si.txt differs -> every kl_new_SI would be on a different axis")

    mpath = work / "eval/math_eval/math_logic_prompts.jsonl"
    ids_hash, n_items = math_ids_hash(mpath)
    ok = ids_hash == EXPECT["math_ids_sha1_8"]
    print(f"  math item ids     {ids_hash}          {'OK' if ok else 'MISMATCH'}  ({n_items} items)")
    if not ok:
        failures.append("math item ids differ -> math_bare/math_hint are a different probe")

    vpath = work / "data/socrateach_sft_val.jsonl"
    got = sha256(vpath)
    ok = got == EXPECT["val_sha256"]
    n_val = sum(1 for _ in open(vpath, encoding="utf-8"))
    print(f"  val split         {got[:16]}  {'OK' if ok else 'MISMATCH'}  ({n_val} rows)")
    if not ok:
        failures.append("val split differs -> the 64 KL / 128 NLL items are different")

    if failures:
        raise SystemExit("\nREFUSING to continue — comparability is already broken:\n  - "
                         + "\n  - ".join(failures))

    # ---- the KL-context invariant Impl 3 states -----------------------------------------
    # Their stated guarantee: the 64 KL prompts are val[:64], each truncated to a single user
    # turn. If that ever stops holding, kl_new_SI and kl_ped_noSI collapse together (their
    # pitfall #1) and the KL axis goes flat — so assert it rather than trust it.
    sys.path.insert(0, str(work))
    n_ctx = None
    try:
        from common.kl import pedagogy_contexts                    # noqa: E402
        from common.system_instructions import CANONICAL_SI        # noqa: E402
    except ImportError as e:
        print(f"\n  KL-context invariant: SKIPPED ({e}) — their kl.py imports torch at module "
              f"level. This check runs wherever the eval runs, which always has torch.")
    else:
        val = [json.loads(l) for l in open(vpath, encoding="utf-8") if l.strip()]
        ctxs = pedagogy_contexts(val, 64)
        lengths = {len(c) for c in ctxs}
        last_roles = {c[-1]["role"] for c in ctxs}
        n_ctx = len(ctxs)
        print(f"\n  pedagogy_contexts(val, 64): {n_ctx} contexts | lengths={lengths} "
              f"| last role={last_roles}")
        assert n_ctx == 64, f"expected 64 KL contexts, got {n_ctx}"
        assert lengths == {1}, f"contexts must be a single user turn, got lengths {lengths}"
        assert last_roles == {"user"}, f"contexts must end on the student turn, got {last_roles}"
        print(f"  CANONICAL_SI: {len(CANONICAL_SI)} chars, ends {CANONICAL_SI[-38:]!r}")

    (work / "compat_setup.json").write_text(json.dumps({
        "bundle": str(bundle),
        "impl4_root": str(IMPL4_ROOT),
        "expected_protocol": EXPECT_PROTOCOL,
        "verified": EXPECT,
        "n_kl_contexts": n_ctx,
        "n_math_items": n_items,
        "n_val_rows": n_val,
        "placeholder_files": ["common/prompts/impl1_system_prompt.txt"],
        "shims": ["common/modeling.py", "common/chat.py"],
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nReady. Expected protocol string:\n  {EXPECT_PROTOCOL}")
    print("\nNext:")
    print(f"  python {HERE.name}/bridge.py --workdir {work}")
    print(f"  cd {work} && python eval/sweep_ckpt_eval.py --runs 'out/*' "
          f"--out out/ckpt_sweep_impl4.jsonl")


if __name__ == "__main__":
    main()
