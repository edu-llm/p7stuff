# ORCD Bootstrap and W&B Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that one MIT Engaging account can run a short OLMo2-190M GPU job, publish metrics
to `eduLLM/test`, save/resume a checkpoint, and safely transfer a small public artifact to/from S3.

**Architecture:** Keep persistent code and the Python environment in Engaging Home, active data and
outputs in Scratch, and credentials in operator-private files. Use ordinary Slurm `sbatch` jobs and
the portable OLMo example; do not depend on the future GitHub queue. The S3 test is a separate gate
after local training works.

**Tech Stack:** Bash, Slurm, MIT Engaging `mit_normal`/`mit_normal_gpu`, Miniforge, Python 3.11+,
PyTorch, OLMo-core, W&B, NumPy memmaps, optional HTTPS S3 presigned URLs.

## Global Constraints

- Use one L40S and at most 45 minutes for the generic GPU smoke.
- Never commit `WANDB_API_KEY`, AWS credentials, presigned URLs, or MIT credentials.
- Use `eduLLM/test` and W&B group `orcd-bootstrap`.
- Use OLMo2-190M and 20 steps; this is an engineering smoke, not scientific evidence.
- Do not require S3 for Stages 1–3.
- Do not install packages on a login node; submit environment setup to `mit_normal`.
- Store temporary data under `$HOME/orcd/scratch/edullm`.

---

## File Structure

Create:

```text
src/scripts/orcd/
  README.md                 Operator setup and manual acceptance procedure
  setup_env.sbatch          One-time persistent virtualenv installation
  probe.py                  CUDA, filesystem, import, and network checks
  probe.sbatch              One-L40S probe allocation
  create_tiny_data.py       Deterministic GPT-2-compatible token files
  run_generic_smoke.sh      Shared OLMo command for initial and resume runs
  generic_smoke.sbatch      One-L40S training allocation
  wandb.env.example         Secret-free environment template
  s3_transfer_pilot.py      Presigned GET/PUT measurement tool

src/test/scripts/orcd/
  probe_test.py
  create_tiny_data_test.py
  job_scripts_test.py
  s3_transfer_pilot_test.py
```

Responsibilities:

- Slurm files allocate resources and call focused scripts.
- Python files contain testable logic.
- `run_generic_smoke.sh` contains one canonical training command.
- No ORCD-specific logic enters `src/olmo_core/`.

---

### Task 1: Persistent Environment and GPU Probe

**Files:**
- Create: `src/scripts/orcd/setup_env.sbatch`
- Create: `src/scripts/orcd/probe.py`
- Create: `src/scripts/orcd/probe.sbatch`
- Test: `src/test/scripts/orcd/probe_test.py`

**Interfaces:**
- Consumes: `EDULLM_REPO_ROOT`, `EDULLM_VENV`, `EDULLM_SCRATCH`.
- Produces: persistent virtualenv and a JSON probe report.

- [ ] **Step 1: Write the failing probe tests**

```python
# src/test/scripts/orcd/probe_test.py
import importlib.util
from pathlib import Path


def load_probe_module():
    path = Path("src/scripts/orcd/probe.py")
    spec = importlib.util.spec_from_file_location("orcd_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_read_probe(tmp_path):
    probe = load_probe_module()
    result = probe.check_writable_directory(tmp_path)
    assert result == {"path": str(tmp_path), "writable": True}


def test_report_contains_required_sections(tmp_path, monkeypatch):
    probe = load_probe_module()
    monkeypatch.setattr(probe, "check_cuda", lambda: {"available": False})
    monkeypatch.setattr(probe, "check_wandb", lambda: {"importable": True, "reachable": False})
    report = probe.build_report(tmp_path)
    assert set(report) == {"python", "cuda", "wandb", "scratch"}
```

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
pytest -v src/test/scripts/orcd/probe_test.py
```

Expected: FAIL because `src/scripts/orcd/probe.py` does not exist.

- [ ] **Step 3: Implement the focused probe**

```python
# src/scripts/orcd/probe.py
import json
import platform
import tempfile
from pathlib import Path

import requests
import torch


def check_writable_directory(path: Path) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path, delete=True) as handle:
        handle.write(b"edullm")
        handle.flush()
    return {"path": str(path), "writable": True}


def check_cuda() -> dict[str, object]:
    available = torch.cuda.is_available()
    return {
        "available": available,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if available else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def check_wandb() -> dict[str, object]:
    try:
        import wandb  # noqa: F401

        importable = True
    except ImportError:
        importable = False
    try:
        response = requests.get("https://api.wandb.ai", timeout=10)
        reachable = response.status_code < 500
    except requests.RequestException:
        reachable = False
    return {"importable": importable, "reachable": reachable}


def build_report(scratch: Path) -> dict[str, object]:
    return {
        "python": {"version": platform.python_version()},
        "cuda": check_cuda(),
        "wandb": check_wandb(),
        "scratch": check_writable_directory(scratch),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.scratch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["cuda"]["available"]:
        raise SystemExit("CUDA is not available")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the one-time environment setup job**

```bash
#!/bin/bash
# src/scripts/orcd/setup_env.sbatch
#SBATCH -p mit_normal
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -J edullm-env
#SBATCH -o %x-%j.log
set -euo pipefail

: "${EDULLM_REPO_ROOT:?export EDULLM_REPO_ROOT before sbatch}"
: "${EDULLM_COMMIT_SHA:?export EDULLM_COMMIT_SHA before sbatch}"
EDULLM_VENV="${EDULLM_VENV:-$HOME/venvs/edullm}"

module load miniforge/24.3.0-0
test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"
test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"
python -m venv "$EDULLM_VENV"
source "$EDULLM_VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install "${EDULLM_REPO_ROOT}[wandb]"
python -c "import torch, wandb, olmo_core; print(torch.__version__)"
```

- [ ] **Step 5: Add the one-L40S probe job**

```bash
#!/bin/bash
# src/scripts/orcd/probe.sbatch
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -t 00:15:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -J edullm-probe
#SBATCH -o %x-%j.log
set -euo pipefail

: "${EDULLM_REPO_ROOT:?export EDULLM_REPO_ROOT before sbatch}"
: "${EDULLM_COMMIT_SHA:?export EDULLM_COMMIT_SHA before sbatch}"
EDULLM_VENV="${EDULLM_VENV:-$HOME/venvs/edullm}"
EDULLM_SCRATCH="${EDULLM_SCRATCH:-$HOME/orcd/scratch/edullm}"

source "$EDULLM_VENV/bin/activate"
export PYTHONPATH="$EDULLM_REPO_ROOT/src"
test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"
test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"
nvidia-smi
python "$EDULLM_REPO_ROOT/src/scripts/orcd/probe.py" \
  --scratch "$EDULLM_SCRATCH" \
  --output "$EDULLM_SCRATCH/probes/${SLURM_JOB_ID}.json"
```

- [ ] **Step 6: Run tests and static shell checks**

Run:

```bash
pytest -v src/test/scripts/orcd/probe_test.py
bash -n src/scripts/orcd/setup_env.sbatch
bash -n src/scripts/orcd/probe.sbatch
```

Expected: PASS and no shell syntax output.

- [ ] **Step 7: Commit**

```bash
git add src/scripts/orcd src/test/scripts/orcd/probe_test.py
git commit -m "feat: add ORCD environment probe"
```

---

### Task 2: Deterministic Tiny Data and Generic W&B Smoke

**Files:**
- Create: `src/scripts/orcd/create_tiny_data.py`
- Create: `src/scripts/orcd/run_generic_smoke.sh`
- Create: `src/scripts/orcd/generic_smoke.sbatch`
- Create: `src/scripts/orcd/wandb.env.example`
- Test: `src/test/scripts/orcd/create_tiny_data_test.py`
- Test: `src/test/scripts/orcd/job_scripts_test.py`

**Interfaces:**
- Consumes: GPT-2 padded vocabulary, run name, W&B private environment.
- Produces: deterministic token files, W&B run, logs, and local checkpoint directory.

- [ ] **Step 1: Write the failing data-generator test**

```python
# src/test/scripts/orcd/create_tiny_data_test.py
import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    path = Path("src/scripts/orcd/create_tiny_data.py")
    spec = importlib.util.spec_from_file_location("tiny_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_data_is_deterministic(tmp_path):
    module = load_module()
    first = module.create_data(tmp_path / "a", train_tokens=4096, eval_tokens=1024, seed=7)
    second = module.create_data(tmp_path / "b", train_tokens=4096, eval_tokens=1024, seed=7)
    assert np.array_equal(np.memmap(first["train"], dtype=np.uint16), np.memmap(second["train"], dtype=np.uint16))
    assert Path(first["eval"]).stat().st_size == 1024 * np.dtype(np.uint16).itemsize
```

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
pytest -v src/test/scripts/orcd/create_tiny_data_test.py
```

Expected: FAIL because the generator does not exist.

- [ ] **Step 3: Implement deterministic raw token files**

```python
# src/scripts/orcd/create_tiny_data.py
import argparse
import json
from pathlib import Path

import numpy as np


def write_tokens(path: Path, count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.memmap(path, mode="w+", dtype=np.uint16, shape=(count,))
    array[:] = rng.integers(0, 50_257, size=count, dtype=np.uint16)
    array.flush()


def create_data(root: Path, *, train_tokens: int, eval_tokens: int, seed: int) -> dict[str, str]:
    train = root / "c4-train.00000-00099.npy"
    eval_path = root / "c4-validation.00000-00008.npy"
    write_tokens(train, train_tokens, seed)
    write_tokens(eval_path, eval_tokens, seed + 1)
    manifest = {
        "format": "raw-uint16-token-stream",
        "tokenizer": "gpt2",
        "seed": seed,
        "train_tokens": train_tokens,
        "eval_tokens": eval_tokens,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"train": str(train), "eval": str(eval_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=1_000_000)
    parser.add_argument("--eval-tokens", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(create_data(args.output, train_tokens=args.train_tokens, eval_tokens=args.eval_tokens, seed=args.seed))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the secret-free W&B template**

```bash
# src/scripts/orcd/wandb.env.example
# Store the real key in ~/.config/edullm/wandb.key with mode 600.
export WANDB_API_KEY="$(cat "$HOME/.config/edullm/wandb.key")"
export WANDB_ENTITY="eduLLM"
export WANDB_PROJECT="test"
export WANDB_GROUP="orcd-bootstrap"
```

- [ ] **Step 5: Add the canonical generic smoke command**

```bash
#!/bin/bash
# src/scripts/orcd/run_generic_smoke.sh
set -euo pipefail

: "${EDULLM_REPO_ROOT:?missing EDULLM_REPO_ROOT}"
: "${EDULLM_SCRATCH:?missing EDULLM_SCRATCH}"
: "${EDULLM_COMMIT_SHA:?missing EDULLM_COMMIT_SHA}"
: "${WANDB_API_KEY:?missing WANDB_API_KEY}"

RUN_NAME="${RUN_NAME:-orcd-smoke-${SLURM_JOB_ID:-manual}}"
HARD_STOP_STEPS="${HARD_STOP_STEPS:-20}"
DATA_ROOT="${OLMO_DATA_ROOT:-$EDULLM_SCRATCH/data/generic-smoke}"
SAVE_FOLDER="$EDULLM_SCRATCH/runs/$RUN_NAME"

export OLMO_DATA_ROOT="$DATA_ROOT"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"
export WANDB_RESUME=allow
export WANDB_SYNC_DIR="$SAVE_FOLDER/wandb"
if ! python -c 'import requests; raise SystemExit(0 if requests.get("https://api.wandb.ai", timeout=5).status_code < 500 else 1)'
then
  export WANDB_MODE=offline
fi
export PYTHONPATH="$EDULLM_REPO_ROOT/src"

test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"
test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"
python -c 'import olmo_core, os; assert os.path.realpath(olmo_core.__file__).startswith(os.path.realpath(os.environ["EDULLM_REPO_ROOT"]))'

torchrun --standalone --nproc-per-node=1 \
  "$EDULLM_REPO_ROOT/src/examples/llm/train.py" "$RUN_NAME" \
  --model-factory=olmo2_190M \
  --sequence-length=512 \
  --save-folder="$SAVE_FOLDER" \
  --work-dir="$EDULLM_SCRATCH/cache/$RUN_NAME" \
  --data_loader.global_batch_size=8192 \
  --train_module.rank_microbatch_size=2048 \
  --train_module.max_sequence_length=512 \
  --trainer.hard_stop="{value: $HARD_STOP_STEPS, unit: steps}" \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.callbacks.checkpointer.save_interval=10 \
  --trainer.callbacks.wandb.enabled=true \
  --trainer.callbacks.wandb.entity="$WANDB_ENTITY" \
  --trainer.callbacks.wandb.project="$WANDB_PROJECT" \
  --trainer.callbacks.wandb.group="$WANDB_GROUP" \
  --trainer.callbacks.wandb.tags="[orcd,generic-smoke,olmo2-190m]"
```

- [ ] **Step 6: Add the Slurm allocation**

```bash
#!/bin/bash
# src/scripts/orcd/generic_smoke.sbatch
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -t 00:45:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -J edullm-smoke
#SBATCH -o %x-%j.log
set -euo pipefail

: "${EDULLM_REPO_ROOT:?missing EDULLM_REPO_ROOT}"
: "${EDULLM_COMMIT_SHA:?missing EDULLM_COMMIT_SHA}"
EDULLM_VENV="${EDULLM_VENV:-$HOME/venvs/edullm}"
EDULLM_SCRATCH="${EDULLM_SCRATCH:-$HOME/orcd/scratch/edullm}"
WANDB_ENV="${WANDB_ENV:-$HOME/.config/edullm/wandb.env}"

source "$EDULLM_VENV/bin/activate"
source "$WANDB_ENV"
export EDULLM_SCRATCH

EDULLM_DATA_MODE="${EDULLM_DATA_MODE:-synthetic}"
if [[ "$EDULLM_DATA_MODE" == "synthetic" ]]; then
  python "$EDULLM_REPO_ROOT/src/scripts/orcd/create_tiny_data.py" \
    --output "$EDULLM_SCRATCH/data/generic-smoke"
elif [[ "$EDULLM_DATA_MODE" == "staged" ]]; then
  : "${OLMO_DATA_ROOT:?staged mode requires OLMO_DATA_ROOT}"
  test -s "$OLMO_DATA_ROOT/c4-train.00000-00099.npy"
  test -s "$OLMO_DATA_ROOT/c4-validation.00000-00008.npy"
  export OLMO_DATA_ROOT
else
  echo "unknown EDULLM_DATA_MODE: $EDULLM_DATA_MODE" >&2
  exit 2
fi
bash "$EDULLM_REPO_ROOT/src/scripts/orcd/run_generic_smoke.sh"
```

- [ ] **Step 7: Add static contract tests**

```python
# src/test/scripts/orcd/job_scripts_test.py
from pathlib import Path


def test_generic_job_requests_one_l40s():
    text = Path("src/scripts/orcd/generic_smoke.sbatch").read_text()
    assert "#SBATCH -G l40s:1" in text
    assert "#SBATCH -t 00:45:00" in text


def test_generic_smoke_routes_to_wandb_and_caps_steps():
    text = Path("src/scripts/orcd/run_generic_smoke.sh").read_text()
    assert "--trainer.callbacks.wandb.enabled=true" in text
    assert "HARD_STOP_STEPS" in text
    assert "--model-factory=olmo2_190M" in text
    assert "torchrun --standalone" in text
    assert "EDULLM_COMMIT_SHA" in text
    assert "PYTHONPATH" in text
    assert "WANDB_MODE=offline" in text
    assert 'WANDB_SYNC_DIR="$SAVE_FOLDER/wandb"' in text
    assert "WANDB_API_KEY=" not in text
    job = Path("src/scripts/orcd/generic_smoke.sbatch").read_text()
    assert 'EDULLM_DATA_MODE="${EDULLM_DATA_MODE:-synthetic}"' in job
    assert 'elif [[ "$EDULLM_DATA_MODE" == "staged" ]]' in job
```

- [ ] **Step 8: Run tests and shell checks**

```bash
pytest -v src/test/scripts/orcd/create_tiny_data_test.py src/test/scripts/orcd/job_scripts_test.py
bash -n src/scripts/orcd/run_generic_smoke.sh
bash -n src/scripts/orcd/generic_smoke.sbatch
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/scripts/orcd src/test/scripts/orcd
git commit -m "feat: add generic ORCD W&B smoke"
```

---

### Task 3: Checkpoint and Resume Verification

**Files:**
- Modify: `src/scripts/orcd/run_generic_smoke.sh`
- Modify: `src/scripts/orcd/README.md`
- Test: `src/test/scripts/orcd/job_scripts_test.py`

**Interfaces:**
- Consumes: stable `RUN_NAME`, `WANDB_RUN_ID`, and `HARD_STOP_STEPS`.
- Produces: checkpoint at step 10/20 and a resumed run advancing to step 25.

- [ ] **Step 1: Add a failing resume-contract test**

```python
def test_resume_uses_stable_save_folder_and_wandb_id():
    text = Path("src/scripts/orcd/run_generic_smoke.sh").read_text()
    assert 'SAVE_FOLDER="${SAVE_FOLDER:-' in text
    assert 'WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"' in text
    assert "WANDB_RESUME=allow" in text
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -v src/test/scripts/orcd/job_scripts_test.py -k resume
```

Expected: FAIL because `SAVE_FOLDER` cannot yet be overridden.

- [ ] **Step 3: Make run identity resumable**

Change:

```bash
SAVE_FOLDER="${SAVE_FOLDER:-$EDULLM_SCRATCH/runs/$RUN_NAME}"
```

Keep the existing automatic checkpoint load from `src/examples/llm/train.py`.

- [ ] **Step 4: Document the two submissions**

Add to `src/scripts/orcd/README.md`:

```bash
# Initial run
export EDULLM_COMMIT_SHA="$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)"
export RUN_NAME=orcd-bootstrap
export WANDB_RUN_ID=orcd-bootstrap
export HARD_STOP_STEPS=20
sbatch --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH",RUN_NAME="$RUN_NAME",WANDB_RUN_ID="$WANDB_RUN_ID",HARD_STOP_STEPS="$HARD_STOP_STEPS" \
  src/scripts/orcd/generic_smoke.sbatch

# Resume after the first job reaches a checkpoint
export HARD_STOP_STEPS=25
sbatch --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH",RUN_NAME="$RUN_NAME",WANDB_RUN_ID="$WANDB_RUN_ID",HARD_STOP_STEPS="$HARD_STOP_STEPS" \
  src/scripts/orcd/generic_smoke.sbatch
```

Acceptance: logs show checkpoint loading and the second job reaches step 25.

- [ ] **Step 5: Run tests**

```bash
pytest -v src/test/scripts/orcd/job_scripts_test.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/scripts/orcd src/test/scripts/orcd/job_scripts_test.py
git commit -m "test: add ORCD checkpoint resume smoke"
```

---

### Task 4: Safe S3 Transfer Pilot

**Files:**
- Create: `src/scripts/orcd/s3_transfer_pilot.py`
- Test: `src/test/scripts/orcd/s3_transfer_pilot_test.py`
- Modify: `src/scripts/orcd/README.md`

**Interfaces:**
- Consumes: a mode-`0600` runtime JSON file containing short-lived presigned GET and PUT URLs.
- Produces: local file, uploaded result, SHA-256, bytes, elapsed time, throughput report.

- [ ] **Step 1: Write failing transfer tests**

```python
# src/test/scripts/orcd/s3_transfer_pilot_test.py
import hashlib
import importlib.util
from pathlib import Path

import pytest


def load_module():
    path = Path("src/scripts/orcd/s3_transfer_pilot.py")
    spec = importlib.util.spec_from_file_location("s3_transfer_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_file_digest(tmp_path):
    module = load_module()
    path = tmp_path / "data.bin"
    path.write_bytes(b"edullm")
    assert module.sha256_file(path) == hashlib.sha256(b"edullm").hexdigest()


def test_sanitizes_presigned_url_errors(monkeypatch):
    module = load_module()
    secret_url = "https://example.invalid/object?X-Amz-Signature=SECRET"
    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            module.requests.ConnectionError(secret_url)
        ),
    )
    with pytest.raises(RuntimeError) as exc:
        module.download(
            secret_url,
            Path("/tmp/unused"),
            expected_sha256="0" * 64,
            max_bytes=1024,
        )
    assert "SECRET" not in str(exc.value)
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -v src/test/scripts/orcd/s3_transfer_pilot_test.py
```

Expected: FAIL because the transfer tool does not exist.

- [ ] **Step 3: Implement presigned transfer measurement**

```python
# src/scripts/orcd/s3_transfer_pilot.py
import argparse
import hashlib
import json
import stat
import time
from pathlib import Path

import requests


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> tuple[int, float]:
    start = time.monotonic()
    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"S3 GET failed with HTTP {response.status_code}")
        content_length = int(response.headers.get("Content-Length", "0"))
        if content_length and content_length > max_bytes:
            raise RuntimeError("S3 GET exceeds the configured size limit")
        written = 0
        with response, destination.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise RuntimeError("S3 GET exceeded the configured size limit")
                handle.write(chunk)
    except requests.RequestException as error:
        raise RuntimeError(f"S3 GET failed: {type(error).__name__}") from None
    if sha256_file(destination) != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError("S3 GET digest mismatch")
    return destination.stat().st_size, time.monotonic() - start


def upload(url: str, source: Path) -> tuple[int, float]:
    start = time.monotonic()
    try:
        with source.open("rb") as handle:
            response = requests.put(
                url,
                data=handle,
                headers={"Content-Length": str(source.stat().st_size)},
                timeout=60,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"S3 PUT failed with HTTP {response.status_code}")
    except requests.RequestException as error:
        raise RuntimeError(f"S3 PUT failed: {type(error).__name__}") from None
    return source.stat().st_size, time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--max-download-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()
    mode = stat.S_IMODE(args.url_file.stat().st_mode)
    if mode & 0o077:
        raise SystemExit("URL file must not be readable by group or other users")
    urls = json.loads(args.url_file.read_text(encoding="utf-8"))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    local = args.work_dir / "s3-pilot-input.bin"
    download_bytes, download_seconds = download(
        urls["download_url"],
        local,
        expected_sha256=args.expected_sha256,
        max_bytes=args.max_download_bytes,
    )
    upload_bytes, upload_seconds = upload(urls["upload_url"], args.result_file)
    report = {
        "download_bytes": download_bytes,
        "upload_bytes": upload_bytes,
        "sha256": sha256_file(local),
        "download_seconds": download_seconds,
        "upload_seconds": upload_seconds,
        "download_mib_per_second": download_bytes / (1024**2) / download_seconds,
        "upload_mib_per_second": upload_bytes / (1024**2) / upload_seconds,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Document credential and data constraints**

Document that:

- Presigned URLs live only in a temporary mode-`0600` JSON file outside the repository.
- URLs must never appear in Git, Slurm logs, or W&B.
- Pilot data must be public/research-cleared.
- Current unversioned sandbox buckets are pilot-only.
- Measure AWS transfer charges before scaling.
- The downloaded object is a bounded public token shard with a known SHA-256.
- Copy it into the generic smoke data root, generate the local validation shard, and train from
  Engaging Scratch.
- Upload `step20/config.json` or a selected checkpoint archive from that run, not a synthetic digest
  payload.

- [ ] **Step 5: Run tests**

```bash
pytest -v src/test/scripts/orcd/s3_transfer_pilot_test.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/scripts/orcd src/test/scripts/orcd/s3_transfer_pilot_test.py
git commit -m "feat: add ORCD S3 transfer pilot"
```

---

### Task 5: Manual Engaging Acceptance

**Files:**
- Modify: `src/scripts/orcd/README.md`

**Interfaces:**
- Consumes: operator Engaging account and W&B membership.
- Produces: probe JSON, Slurm IDs, W&B URL, checkpoint/resume evidence, transfer report.

- [ ] **Step 1: Run local verification**

```bash
pytest -v src/test/scripts/orcd/
bash -n src/scripts/orcd/*.sh src/scripts/orcd/*.sbatch
make lint-check
make style-check
```

Expected: PASS.

- [ ] **Step 2: Install the environment through Slurm**

```bash
export EDULLM_REPO_ROOT="$HOME/OLMo-core"
export EDULLM_COMMIT_SHA="$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)"
sbatch --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA" \
  src/scripts/orcd/setup_env.sbatch
```

Expected: setup log prints Python, PyTorch, W&B, and OLMo versions.

- [ ] **Step 3: Run the GPU probe**

```bash
export EDULLM_SCRATCH="$HOME/orcd/scratch/edullm"
sbatch --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH" \
  src/scripts/orcd/probe.sbatch
```

Expected: JSON reports one L40S and writable Scratch.

- [ ] **Step 4: Configure W&B privately**

```bash
mkdir -p "$HOME/.config/edullm"
cp src/scripts/orcd/wandb.env.example "$HOME/.config/edullm/wandb.env"
umask 077
printf '%s\n' "$WANDB_API_KEY" > "$HOME/.config/edullm/wandb.key"
chmod 600 "$HOME/.config/edullm/wandb.env" "$HOME/.config/edullm/wandb.key"
```

Run the `printf` command only from a shell where `WANDB_API_KEY` is already set without shell
history exposure, then unset it. Verify another `eduLLM` member can open `eduLLM/test`.

Exercise the outage path by exporting `WANDB_MODE=offline`, completing the smoke, preserving
`$EDULLM_SCRATCH/runs/$RUN_NAME/wandb`, and then syncing:

```bash
source "$HOME/venvs/edullm/bin/activate"
source "$HOME/.config/edullm/wandb.env"
wandb sync --sync-all "$EDULLM_SCRATCH/runs/$RUN_NAME/wandb"
```

The training job must complete even when W&B is initially unreachable.

- [ ] **Step 5: Run initial and resumed smokes**

Use the commands documented in `src/scripts/orcd/README.md`. Record:

- Slurm job IDs.
- W&B URL.
- Initial final step.
- Checkpoint path.
- Resume-loaded step and final step 25.
- The final `train/CE loss` value.

Verify the required metric directly:

```python
import math
import wandb

run = wandb.Api().run("eduLLM/test/orcd-bootstrap")
history = run.history(keys=["train/CE loss"], pandas=True)
value = float(history["train/CE loss"].dropna().iloc[-1])
assert math.isfinite(value)
print(value)
```

Use the actual deterministic run ID if the bootstrap ID is changed.

- [ ] **Step 6: Run the S3 transfer pilot if access is approved**

Create a temporary mode-`0600` URL JSON outside the repository, run the staged-data training path,
and upload the selected `step20/config.json` or checkpoint archive. Record GET/PUT bytes, duration,
throughput, and estimated transfer cost. Delete the URL file immediately afterward.

- [ ] **Step 7: Write acceptance evidence**

Add a dated section to `src/scripts/orcd/README.md` containing non-secret Slurm IDs, W&B URL,
GPU model, runtime, checkpoint result, and transfer measurements.

- [ ] **Step 8: Commit**

```bash
git add src/scripts/orcd/README.md
git commit -m "docs: record ORCD bootstrap verification"
```
