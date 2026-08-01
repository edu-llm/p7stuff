#!/usr/bin/env python
"""What `colab exec -f` runs on a fresh Colab CLI runtime: clone, then launch detached.

Sent to the VM with::

    colab exec -s impl4 -f colab_bootstrap.py

It clones the repo and starts ``run_matched.py`` under ``nohup``, then returns
immediately. The job therefore survives the CLI connection dropping, which matters
because a full arm is hours and ``colab exec`` is a single request/response — a blocking
exec that loses its connection at hour two loses the run.

Poll from the laptop::

    colab exec -s impl4 <<< "print(open('/content/run.log').read()[-4000:])"

Configuration comes from environment variables so this file never needs editing; the
launcher script sets them via a small prelude. Defaults are the POC smoke on arm A1.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("IMPL4_REPO", "https://github.com/edu-llm/p7stuff.git")
BRANCH = os.environ.get("IMPL4_BRANCH", "impl4-ssd")
CLONE = Path(os.environ.get("IMPL4_CLONE", "/content/p7stuff"))
ARM = os.environ.get("IMPL4_ARM", "A1")
POC = os.environ.get("IMPL4_POC", "1") == "1"
STAGES = os.environ.get("IMPL4_STAGES", "all")
EXTRA = os.environ.get("IMPL4_EXTRA", "")
LOG = Path(os.environ.get("IMPL4_LOG", "/content/run.log"))

IMPL4 = CLONE / "p7" / "POC" / "impl4_ssd"


def run(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


# --- 1. code ----------------------------------------------------------------
if (CLONE / ".git").exists():
    run(f"git -C {CLONE} fetch --depth 1 origin {BRANCH} && "
        f"git -C {CLONE} reset --hard FETCH_HEAD")
else:
    run(f"git clone --depth 1 --branch {BRANCH} {REPO} {CLONE}")
run(f"git -C {CLONE} log --oneline -1")

for need in ("run_matched.py", "impl3_compat/setup_compat.py", "build_pedagogy_pool.py"):
    if not (IMPL4 / need).exists():
        raise SystemExit(f"clone is incomplete: missing {need}")

# --- 2. the bundle has to be here already -----------------------------------
tar = Path(os.environ.get("IMPL4_BUNDLE_TAR", "/content/impl3_handoff.tar.gz"))
if not tar.exists() and not Path("/content/impl3_handoff/eval/sweep_ckpt_eval.py").exists():
    raise SystemExit(
        f"{tar} is not on this VM. From the laptop:\n"
        f"    colab upload impl3_handoff.tar.gz {tar}\n"
        f"then re-run this bootstrap.")

# --- 3. launch detached ------------------------------------------------------
cmd = [sys.executable, "run_matched.py", "--arm", ARM, "--stages", STAGES]
if POC:
    cmd.append("--poc")
if EXTRA:
    cmd += EXTRA.split()
quoted = " ".join(cmd)

# setsid + nohup so the job is not in the exec'd process group: when the CLI request
# ends, its children would otherwise be signalled.
run(f"cd {IMPL4} && setsid nohup {quoted} > {LOG} 2>&1 < /dev/null & echo $! > /content/run.pid")

pid = Path("/content/run.pid").read_text().strip()
print(f"\nlaunched pid {pid}: {quoted}")
print(f"arm={ARM} poc={POC} stages={STAGES}")
print(f"log: {LOG}")
print("\npoll with:")
print(f"    colab exec <<< \"print(open('{LOG}').read()[-4000:])\"")
print("check it is alive with:")
print(f"    colab exec <<< \"import os;print(os.path.exists('/proc/{pid}'))\"")
