# eduLLM ORCD Rollout Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement each plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a verified MIT Engaging training path first, then a safe three-operator request
pool, then end-to-end Skill-DAG and curriculum smoke runs.

**Architecture:** The approved design is intentionally split into three independently testable
subprojects. ORCD bootstrap proves the GPU, environment, W&B, checkpoint, and transfer path.
The job-pool plan adds the GitHub request queue and operator tooling without touching training
semantics. The hypothesis-wiring plan connects prepared data artifacts to OLMo training.

**Tech Stack:** Python 3.11+, PyTorch/OLMo-core, Slurm, OpenSSH ControlMaster, GitHub Issues and
Actions, Slack incoming webhooks, W&B, MIT Engaging Home/Scratch/Pool, optional S3 presigned URLs.

## Global Constraints

- Never store MIT, SSH, Kerberos, Duo, W&B, GitHub, or AWS credentials in Git.
- Run only reviewed full commit SHAs under an operator's Engaging account.
- Use one L40S and OLMo2-190M for the initial smoke.
- Keep W&B metrics/configuration small; keep datasets and checkpoints in Engaging/S3.
- Do not introduce multi-node training in this rollout.
- Do not treat the current sandbox S3 buckets as permanent storage while versioning is disabled.
- Do not automate through personal SSH credentials from GitHub Actions.

---

## Plan Order

### Plan 1: ORCD Bootstrap and W&B Smoke

Path: `docs/superpowers/plans/2026-07-22-orcd-bootstrap-implementation.md`

Delivers:

- Persistent operator virtualenv in Engaging Home.
- L40S/PyTorch/OLMo/W&B/filesystem probe.
- Tiny local token data.
- Generic 20-step OLMo2-190M W&B run under `eduLLM/test`.
- Local checkpoint/resume verification.
- Safe, measured S3 transfer pilot.

Exit gate:

```text
One L40S job completes 20 steps, another process can see the W&B run,
and a resumed run advances beyond its saved checkpoint.
```

Do not begin Plan 2 operator rollout or Plan 3 GPU runs before this gate passes.

### Plan 2: GitHub Queue and Three-Operator Pool

Path: `docs/superpowers/plans/2026-07-22-edullm-job-pool-implementation.md`

Delivers:

- `edullm` command.
- GitHub Issue request form and validation.
- Reviewed-SHA execution gate.
- Least-loaded assignment across enabled operators.
- GitHub and Slack notifications.
- SSH ControlMaster setup.
- Slurm submission, logs, cancellation, and status repair.
- W&B URL/state reconciliation.
- Shared `/submit-edullm-job` Agent Skill.

Exit gate:

```text
A test Issue is assigned to the pilot operator; `edullm run` submits it;
the Issue records Slurm/W&B identity and reaches completed without exposing secrets.
```

### Plan 3: Hypothesis Smoke Wiring

Path: `docs/superpowers/plans/2026-07-22-hypothesis-smoke-wiring-implementation.md`

Delivers:

- Skill-DAG mix weights consumed by the data loader.
- Curriculum document order consumed by the data loader.
- Dry-run followed by real training.
- Stable `eduLLM/pretraining` W&B study groups.
- Saved mix/order hashes and resolved config.
- Equal-token control/treatment smoke requests.

Exit gate:

```text
One Skill-DAG control/treatment pair and one curriculum control/treatment pair
finish on Engaging, with distinct verified data identities and shared W&B study groups.
```

## Integration Sequence

```text
Bootstrap environment
  -> generic smoke
  -> checkpoint/resume
  -> queue pilot with one operator
  -> hypothesis data wiring
  -> hypothesis GPU smokes
  -> queue pilot with three operators
```

The S3 transfer pilot branches after checkpoint/resume and may run in parallel with the
single-operator queue pilot. S3 is not a prerequisite for local Engaging smoke runs.

## Final Verification

- [ ] Run `pytest -v src/test/scripts/orcd/`.
- [ ] Run `pytest -v src/test/edullm/`.
- [ ] Run `pytest -v src/test/scripts/hypothesis/`.
- [ ] Run `make lint-check`.
- [ ] Run `make style-check`.
- [ ] Run `make type-check`.
- [ ] Run `go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 .github/workflows/edullm-*.yml`.
- [ ] Run `git diff --check`.
- [ ] Complete each plan's manual Engaging acceptance test.
- [ ] Confirm a second W&B organization member can view the pilot run.
- [ ] Confirm ORCD approves the shared project workflow before enabling three simultaneous users.
