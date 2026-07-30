# eduLLM Training Workspace — Setup / Bootstrap Guide

Point a fresh Cursor workspace's agent at this file to reconstruct everything
needed for **MIT ORCD**, **GitHub job submission**, and **Weights & Biases (W&B)**.

> Project context: group project training an LLM (OLMo 2 base) on the
> `edu-llm/OLMo-core` fork. Job requests are filed as GitHub Issues; an assigned
> operator runs them on the MIT ORCD Slurm cluster; metrics go to W&B
> (`eduLLM` entity). **Never act on AWS or push/commit to GitHub without the
> training lead's explicit approval. Always work on a branch, never `main`.**

---

## 0. Where the context actually lives (read this first)

| Piece | Source | How to get it in the new workspace |
|-------|--------|-------------------------------------|
| `edullm` CLI, `config/edullm/*`, docs | inside `edu-llm/OLMo-core` repo | clone the repo (§2) |
| `submit-edullm-job` skill | inside the repo (`.cursor/skills/` + `.claude/skills/`, force-tracked) | clone the repo (§2) |
| `wandb-primary` skill | public registry `wandb/skills` | reinstall (§4a) |
| `edullm-aws-training` skill | **custom, this workspace only** | copy the folder (§4b) |
| `sb-aws-readonly` skill (root `SKILL.md`) | **custom, this workspace only** | copy the file (§4b) |
| `sb-aws` MCP (AWS access) | MCP server config | configure only if doing AWS (§5d) |

The submission skill's validator (`validate_request.py`) imports `edullm.*`, so
it **must run from the OLMo-core repo root with the venv active**. Clone the repo
into (or beside) the new workspace.

---

## 1. Prerequisites

- `git`, `gh` (GitHub CLI), `python3` (3.10+), `node`/`npx`
- An **MIT ORCD account** + **Duo** enrolled (Kerberos username, e.g. `xing33`)
- A **W&B account** with access to the `eduLLM` entity
- **MIT network / GlobalProtect VPN** connectivity (ORCD login is reachable on
  MIT's network; connect the VPN before SSH if you're off-campus)

---

## 2. Clone the repo (gets the CLI + submit skill + config + docs)

```bash
git clone https://github.com/edu-llm/OLMo-core.git
cd OLMo-core
git switch main
git pull --ff-only
```

---

## 3. Python env + `edullm` CLI install

On **Linux / Apple Silicon** the normal path works:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[wandb]'
```

On **Intel macOS** this fails (`torch>=2.6.0` has no wheel). The local operator
CLI does **not** need torch — install it without deps:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e . --no-deps
python -m pip install requests pyyaml wandb
python -c "import edullm; print('edullm OK')"
```

> `python` may be missing while `python3` exists — that just means the venv
> isn't active. `source .venv/bin/activate` fixes it.

---

## 4. Install the skills into the new workspace

### 4a. `wandb-primary` (official W&B skill, from registry)

```bash
# run at the new workspace root
npx skills add wandb/skills --skill '*' --yes
```

This writes `.agents/skills/wandb-primary` + `skills-lock.json` but does **not**
populate `.cursor/skills`, so mirror it for Cursor discovery:

```bash
cp -R .agents/skills/wandb-primary .cursor/skills/wandb-primary
```

> Do **not** install `ovachiever/droid-tings/weights-and-biases` — that's the
> wrong one. The correct skill is named **`wandb-primary`**.

### 4b. Custom skills (copy from this workspace — not in any registry)

If the new workspace is on the **same machine**, copy these two directly
(replace `<NEW>` with the new workspace root):

```bash
SRC="/Users/jamesxing/Documents/MericXing/MIT/Intern/AlphaAI/Training_Team"
mkdir -p "<NEW>/.cursor/skills"
cp -R "$SRC/.cursor/skills/edullm-aws-training" "<NEW>/.cursor/skills/"
# the sb-aws access skill currently lives at the workspace root as SKILL.md:
mkdir -p "<NEW>/.cursor/skills/sb-aws-readonly"
cp "$SRC/SKILL.md" "<NEW>/.cursor/skills/sb-aws-readonly/SKILL.md"
```

- `edullm-aws-training` — AWS tags (SEC-05), region lock (`us-east-2`), GPU
  sandbox limits, S3 layout, and the SLURM job-submission workflow.
- `sb-aws-readonly` — routes all live AWS through the `sb-aws` MCP, read-only by
  default. Only needed if the new workspace does AWS work.

---

## 5. Authentication

### 5a. GitHub

```bash
gh auth login          # choose GitHub.com + HTTPS + your account
gh api user --jq .login # verify (should print your login, e.g. meric233)
```

### 5b. Weights & Biases

```bash
wandb login            # paste your W&B API key when prompted
```

### 5c. MIT ORCD SSH (ControlMaster + interactive Duo)

`edullm` reuses a persistent SSH master connection because ORCD needs an
interactive Kerberos password + Duo that can't be automated. Add this to
`~/.ssh/config` (replace the user with your MIT Kerberos username):

```
Host orcd-login
    Hostname orcd-login.mit.edu
    ControlMaster auto
    ControlPath ~/.ssh/edullm-%C
    ControlPersist 1h
    User xing33
```

Then open the master connection **interactively** (enter MIT password, then pick
the Duo push option), and provision the remote:

```bash
ssh -MNf orcd-login                      # completes password + Duo, backgrounds the master
edullm setup --orcd-username xing33      # or: python -m edullm.cli setup --orcd-username xing33
edullm jobs --mine                       # should now succeed over the reused session
```

> `edullm` runs **locally** on your Mac (not on the login node). If you get
> `edullm: command not found`, you're SSH'd into ORCD — `exit` and run it locally.

### 5d. AWS (only if needed)

Configure the `sb-aws` MCP server in the new workspace and follow the
`sb-aws-readonly` skill. Account `056956104102` (`sbsandbox`), region-locked to
`us-east-1`/`us-east-2`.

---

## 6. The job workflow (end to end)

1. **Branch + code**: create a branch off `main`, make/adjust the training
   script, commit, and **push to `edu-llm/OLMo-core`** (never submit from `main`).
   Submittable scripts must match a fixed entrypoint in
   `config/edullm/entrypoints.yaml`:
   - `generic-smoke` → `src/examples/llm/train.py` (torchrun; emits `train/CE loss`)
   - `hypothesis-smoke` → `src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py` (python)
2. **`/submit-edullm-job`** (Cursor skill, run from repo root): passes a
   fail-closed gate (clean tree, non-`main` branch, pushed 40-char SHA), validates
   via `validate_request.py`, previews the Issue, and — after your explicit
   confirmation — creates a `edullm-job` / `status:requested` Issue.
3. **Automation** validates + assigns: `requested → ready → assigned`.
4. **Review gate**: open a PR for the branch; required CI checks must pass and a
   reviewer must approve before the code should actually run.
5. **Operator** runs `edullm run` → submits to Slurm → `submitted → running →
   completed`. Monitor with `edullm jobs --mine`; logs via `edullm logs <issue>`.
6. Metrics land in W&B under entity `eduLLM`, project `test` (for smokes).

---

## 7. Gotchas

- **Intel macOS**: `torch>=2.6.0` won't install; use the `--no-deps` path (§3).
- **Clean-tree gate**: `/submit-edullm-job` fails if `git status` isn't empty —
  including untracked files (e.g., stray `.agents/`, `skills-lock.json`). In
  `OLMo-core`, `.cursor/` is gitignored so skills there don't dirty the tree.
- **Never submit from `main`**; the SHA must be pushed to `edu-llm/OLMo-core`.
- **`edullm jobs`** only lists jobs already **submitted** to Slurm (submitted/
  running/finished). `requested` and `assigned` Issues won't show — that's normal.
- **ORCD auth** is interactive (password + Duo). If a command says
  `run ssh -MNf orcd-login and retry`, the master session dropped — re-open it.
- **Approvals**: no AWS actions or GitHub pushes/commits without the lead's OK.
