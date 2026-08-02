#!/usr/bin/env python
"""What ``colab exec -f`` runs on a fresh runtime: clone, then launch ``run_impl5.py`` detached.

    colab upload impl3_handoff.tar.gz /content/impl3_handoff.tar.gz
    colab exec -f colab_bootstrap5.py

``setsid nohup`` matters: a full arm is hours and ``colab exec`` is one request/response, so
a blocking exec that loses its connection at hour two loses the run. Detached, the job
survives the CLI dropping — which it does routinely.

Poll from the laptop::

    colab exec <<< "print(open('/content/impl5.log').read()[-4000:])"

Configuration is environment variables so this file never needs editing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("IMPL5_REPO", "https://github.com/edu-llm/p7stuff.git")
BRANCH = os.environ.get("IMPL5_BRANCH", "impl5-ssd")
CLONE = Path(os.environ.get("IMPL5_CLONE", "/content/p7stuff"))
ARM = os.environ.get("IMPL5_ARM", "D4")
POC = os.environ.get("IMPL5_POC", "0") == "1"
STAGES = os.environ.get("IMPL5_STAGES", "all")
EXTRA = os.environ.get("IMPL5_EXTRA", "")
LOG = Path(os.environ.get("IMPL5_LOG", "/content/impl5.log"))

IMPL5 = CLONE / "p7" / "POC" / "impl5_ssd"


def run(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


if (CLONE / ".git").exists():
    run(f"git -C {CLONE} fetch --depth 1 origin {BRANCH} && "
        f"git -C {CLONE} reset --hard FETCH_HEAD")
else:
    run(f"git clone --depth 1 --branch {BRANCH} {REPO} {CLONE}")
run(f"git -C {CLONE} log --oneline -1")

for need in ("run_impl5.py", "distill_pedagogy.py", "impl5/gate5.py"):
    if not (IMPL5 / need).exists():
        raise SystemExit(f"clone is incomplete: missing {need}")
# Impl 5 imports Impl 4's library and reuses its compat bridge; both must be in the clone.
for need in ("impl4/config.py", "impl3_compat/nll_only.py", "impl3_compat/setup_compat.py"):
    if not (CLONE / "p7" / "POC" / "impl4_ssd" / need).exists():
        raise SystemExit(f"clone is missing impl4_ssd/{need}, which Impl 5 depends on")

tar = Path(os.environ.get("IMPL5_BUNDLE_TAR", "/content/impl3_handoff.tar.gz"))
if not tar.exists() and not Path("/content/impl3_handoff/eval/sweep_ckpt_eval.py").exists():
    raise SystemExit(f"{tar} is not on this VM. From the laptop:\n"
                     f"    colab upload impl3_handoff.tar.gz {tar}\nthen re-run.")

cmd = [sys.executable, "run_impl5.py", "--arm", ARM, "--stages", STAGES]
if POC:
    cmd.append("--poc")
if EXTRA:
    cmd += EXTRA.split()
quoted = " ".join(cmd)

run(f"cd {IMPL5} && setsid nohup {quoted} > {LOG} 2>&1 < /dev/null & "
    f"echo $! > /content/impl5.pid")
pid = Path("/content/impl5.pid").read_text().strip()
print(f"\nlaunched pid {pid}: {quoted}")
print(f"arm={ARM} poc={POC} stages={STAGES}\nlog: {LOG}")
print(f"\npoll:  colab exec <<< \"print(open('{LOG}').read()[-4000:])\"")
print(f"alive: colab exec <<< \"import os;print(os.path.exists('/proc/{pid}'))\"")
