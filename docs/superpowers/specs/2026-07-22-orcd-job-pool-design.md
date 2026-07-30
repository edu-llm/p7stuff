# eduLLM ORCD Job Pool Design

Date: 2026-07-22
Status: Approved design; implementation plans available under `docs/superpowers/plans/`

## Purpose

Provide one simple workflow through which eduLLM researchers without MIT Engaging access can
request reviewed GPU jobs, while three authorized Engaging account holders submit those jobs
through their own accounts. GitHub is the request queue, Slurm is the compute scheduler, and
Weights & Biases (W&B) is the shared live-results interface.

The initial acceptance test is one generic 20-step OLMo2-190M run on one Engaging L40S that
appears under the `eduLLM/test` W&B project.

## Design Principles

1. Never share MIT, SSH, Kerberos, Duo, W&B, GitHub, or AWS credentials.
2. Every Engaging job runs under the MIT identity of the operator who submitted it.
3. Researchers request jobs through GitHub Issues; they do not write Slurm commands.
4. Operators use a small `edullm` command-line interface; they do not rewrite requests.
5. Only reviewed code at an immutable commit SHA may run under an operator's account.
6. The training code defines scientific metrics. Queue automation only records and routes them.
7. W&B projects are durable work lanes; W&B groups are stable studies, not per-request folders.
8. Engaging Scratch is the fast working tier. S3 is introduced only after an explicit access,
  transfer, and cost test.
9. Begin with one account and one GPU. Add the other operators only after the direct path works.



## Non-Goals

- Sharing or centrally storing personal SSH credentials.
- Running an unattended service through personal MIT accounts.
- Replacing Engaging's Slurm scheduler.
- Splitting one distributed training job across different users' Slurm allocations.
- Allowing arbitrary shell commands copied from an Issue.
- Building a custom experiment tracker or storing large datasets in W&B.
- Treating the current unversioned sandbox S3 buckets as permanent production storage.
- Claiming the current Skill-DAG or curriculum scripts are end-to-end ready before their generated
mix/order is connected to the training loader.



## System Overview

```text
Researcher + Agent Skill
        |
        v
Structured GitHub Issue
        |
        v
Validation + least-loaded operator assignment
        |
        +---- GitHub mention
        +---- Slack ping
        |
        v
Operator: edullm run
        |
        v
Personal SSH ControlMaster connection
        |
        v
MIT Engaging Slurm (sbatch)
        |
        v
OLMo torchrun job
        |
        +---- W&B: metrics and run state
        +---- Engaging Scratch/Pool: data, logs, checkpoints
        +---- S3 pilot: staged data and selected output copies
```



## Roles



### Researcher

- Owns the hypothesis, comparison, code, data choice, and scientific success criterion.
- Uses the Agent Skill to create a valid request.
- Monitors the GitHub Issue and W&B run.
- Does not require Engaging access.



### Compute Operator

- Is one of the three authorized Engaging account holders.
- Reviews or relies on an approved review of the exact commit.
- Receives GitHub and Slack assignment notifications.
- Runs `edullm run` to submit the next assigned request through their own account.
- Never shares credentials with researchers or other operators.



### Queue Automation

- Validates request structure and code review state.
- Assigns ready requests to the operator with the fewest active requested GPUs.
- Reassigns an unsubmitted request after 30 minutes.
- Posts GitHub and Slack notifications.
- Polls W&B every five minutes for started-run state.
- Holds no MIT or SSH credentials.



## Engaging Resource Model

The initial design targets the documented `mit_normal_gpu` partition:

- Default GPU: L40S, 44 GB.
- Available types include L40S, H100, and H200.
- Base limit: two GPUs per user.
- Maximum normal GPU job duration: six hours.
- One GPU: `-G 1` or `-G l40s:1`.
- Two GPUs: `-G 2`.
- H200: request explicitly, for example `-G h200:1`; do not use for the initial smoke.

Multi-node jobs are supported by Engaging but excluded from the initial implementation. A
two-GPU job is one Slurm allocation submitted by one operator and launches two local `torchrun`
processes.

## W&B Organization



### Entity and Projects

Use entity `eduLLM`.


| Project         | Purpose                                                             |
| --------------- | ------------------------------------------------------------------- |
| `test`          | Temporary ORCD, environment, and integration smoke tests            |
| `pretraining`   | Architecture, data-mixture, Skill-DAG, and curriculum training      |
| `posttraining`  | SFT, preference optimization, reinforcement learning, and alignment |
| `evaluation`    | PedBench and checkpoint-only benchmark runs                         |
| `data-pipeline` | Dolma preparation, filtering, tokenization, and corpus-quality jobs |


Do not initially create separate `model-architecture` or `data-corpus` projects. Those dimensions
belong inside `pretraining` so their runs remain comparable.

### Studies, Runs, and Tags

- Project: durable workflow lane.
- Group: one stable research study, such as `skill-dag-v1` or `curriculum-v1`.
- Run: one execution, such as `natural-seed0-slurm12345`.
- Tags/config: request ID, model, condition, GPU, cluster, commit, seed, and data identity.

Example:

```text
Entity:  eduLLM
Project: pretraining
Group:   skill-dag-v1
Run:     natural-seed0-slurm12345
Tags:    issue-42, olmo2-190m, l40s, engaging
```

Issue 43 may use the same group for another arm or seed. Groups are not generated per request.

### Metric Ownership

The training/evaluation code determines which metrics exist. Queue automation may not invent a
metric that the code does not calculate.

Common OLMo metrics include:

- Training loss.
- Evaluation loss and perplexity.
- Learning rate.
- Tokens seen and tokens per second.
- GPU utilization and memory.

The Agent Skill verifies that the requested success signal is represented by code or changes the
request to an engineering-only smoke. For the generic smoke, success means 20 completed steps,
finite loss, no out-of-memory error, and a visible W&B run.

## GitHub Issue Request

Every request uses the same Issue form. There is no separate "custom job" category.

Required fields:

1. Requester identity derived from the GitHub Issue author.
2. Plain-language purpose.
3. Study/group name.
4. Condition/arm.
5. Comparison condition or explicit "engineering smoke only."
6. Repository and exact full commit SHA.
7. Protected entrypoint profile and script path inside the repository.
8. Ordered argument list represented as structured values, not a shell string.
9. Immutable data manifest, manifest SHA-256, and data classification.
10. Seed.
11. W&B project.
12. Expected success signal and metric names emitted by the code.
13. GPU count: one or two.
14. GPU preference: any/L40S, H100, or H200.
15. Maximum runtime, capped by policy.

The system generates:

- Request name: `issue-<number>-<study>-<condition>`.
- W&B run ID and URL.
- Slurm job name and resource flags.
- Operator assignment.
- Resolved request JSON consumed by the operator CLI.
- Canonical request SHA-256 and append-only attempt identity.

Students never supply a repository URL, environment path, credential, `sbatch` command, or
arbitrary shell pipeline. Restricted data is rejected by the public pilot queue.

## Code Review Gate

"Reviewed commit" means:

1. The researcher pushes a branch and opens a pull request.
2. Repository CI passes.
3. At least one authorized reviewer approves that exact SHA and no current authorized review
   requests changes.
4. The request references the approved full commit SHA.
5. The queue verifies that the SHA is the exact head of an open/merged approved pull request, all
   required non-GPU repository checks pass, and the selected script exists at that SHA.
6. Any code change creates a different SHA and invalidates the previous approval.

This gate is required because training code runs under an operator's Unix identity and may receive
W&B or scoped S3 access. A branch name alone is mutable and is not accepted as execution identity.

## Agent Skill

Create a project skill at:

```text
.cursor/skills/submit-edullm-job/
  SKILL.md
  request-reference.md
  scripts/validate_request.py
```

Trigger it when a user asks to submit, queue, schedule, or run an eduLLM GPU/Engaging job.

The skill:

1. Inspects the current repository, branch, commit, PR, and script.
2. Reads the training configuration and W&B callback.
3. Identifies metrics actually emitted by the code.
4. Asks one plain-language question at a time for missing scientific intent.
5. Recommends safe compute defaults but requires confirmation.
6. Shows a plain-language request preview.
7. Runs the same validator used by CI.
8. Creates the structured GitHub Issue only after confirmation.
9. Never reads, writes, or requests personal credentials.

The skill may default an engineering smoke to one L40S and 30 minutes. It may not choose the
hypothesis, comparison, condition, or scientific metric for the researcher.

### Metric-Instrumentation Handoff

When a requested success metric is not emitted by the training code:

1. The researcher uses the `/weights-and-biases` skill to implement and test that metric.
2. The researcher commits the change, opens a pull request, and obtains review.
3. The `/submit-edullm-job` skill verifies the approved commit and confirms that the metric is
   present before creating the request.

The W&B skill changes experiment instrumentation; it does not submit Slurm jobs. The submission
skill creates requests; it does not invent or implement scientific metrics. Large datasets and
checkpoints remain in Engaging/S3, while W&B stores metrics, configuration, comparisons, and
artifact references.

## Operator CLI

Expose only:

```text
edullm setup          One-time GitHub, W&B, Engaging, environment, and SSH setup
edullm jobs           Show all requests and refresh statuses available to this operator
edullm jobs --mine    Show requests assigned to this operator
edullm run            Validate and submit the next assigned request
edullm logs <issue>   Show logs for a request
edullm stop <issue>   Cancel a request
edullm logout         Close the SSH ControlMaster session
```

There is no user-facing `claim`, `run-next`, `sync`, or `tail` command.

### `edullm setup`

- Verifies `gh` authentication.
- Verifies W&B team/project access.
- Verifies `ssh orcd-login.mit.edu`.
- Creates or safely updates an `orcd-login` SSH alias after showing the diff.
- Configures `ControlMaster auto`, a per-host `ControlPath`, and `ControlPersist 1h`.
- Creates an operator-local configuration file with mode `0600`.
- Builds or verifies the operator's persistent virtualenv in Engaging Home.
- Verifies Scratch paths and records an environment fingerprint.



### `edullm run`

1. Refreshes current jobs belonging to the operator.
2. Selects the oldest assigned, validated request.
3. Revalidates the commit, arguments, data classification, and resource caps.
4. Opens or reuses the SSH ControlMaster connection.
5. Fetches and checks out the exact SHA in detached mode under Engaging Scratch.
6. Activates the persistent environment.
7. Sources the operator's private W&B environment.
8. Generates an `sbatch` file from structured arguments without `eval`.
9. Submits the job and records the Slurm ID in the Issue.
10. Posts the predetermined W&B URL.



### SSH Convenience and Limits

MIT documents SSH ControlMaster as the supported way to reuse one Kerberos/Duo login. The initial
connection authenticates normally. Subsequent commands multiplex over the local control socket.

`ControlPersist 1h` is the project default. Operators can close it immediately with
`edullm logout`. The private key never leaves the operator's computer. This is not an unattended
service credential and does not permit GitHub Actions to SSH to Engaging.

## Assignment and Notifications

Assignment score is active requested GPU count, then active job count, then rotation order. The
lowest score wins.

On assignment:

- Assign and @mention the operator in GitHub.
- Post a Slack message mentioning the mapped operator.
- Post a reminder after 15 minutes if the job remains unsubmitted.
- Reassign after 30 minutes.

On submission, start, completion, failure, cancellation, or reassignment:

- Update Issue labels and a machine-maintained status comment.
- Notify the requester through GitHub.
- Notify the Slack thread for terminal events.

Operator-to-Slack mappings live in a CODEOWNERS-protected repository configuration. The Slack
webhook is a GitHub Actions secret.

## Status Model

```text
requested
  -> validating
  -> ready
  -> assigned
  -> submitted
  -> running
  -> completed | failed | cancelled | preempted
```

W&B is polled every five minutes after a run starts. State transitions are monotonic; terminal
states cannot regress, and preempted requests require explicit retry approval. `edullm jobs`
queries the operator's `squeue`/`sacct` and repairs stale GitHub state.

The polling workflow uses a dedicated W&B monitoring credential stored as a GitHub Actions secret,
never an Engaging operator's key. If the W&B organization cannot provide a dedicated credential,
disable scheduled polling for the pilot and rely on `edullm jobs` plus the W&B UI until an approved
monitoring identity exists.

GitHub cannot continuously poll three personal Slurm accounts without storing personal SSH
credentials. Submitted-but-not-started jobs therefore remain `submitted` until W&B starts or an
operator runs `edullm jobs`. A future ORCD-approved shared service account could remove this
limitation.

## Engaging Files and Environment



### Initial Virtualenv

For the first operator:

- Code and persistent virtualenv: Engaging Home.
- Active data, caches, working trees, logs, and temporary checkpoints: Engaging Scratch.
- Shared/reusable datasets: PI shared Pool when available.
- Important outputs: copy to backed-up Home or approved durable storage.

Each pilot job activates the existing environment but imports OLMo from a pristine reviewed
checkout. Dependencies are not reinstalled per job. After the first smoke succeeds, produce and
verify a source-complete, digest-pinned Apptainer image before onboarding other operators.

### Current Hypothesis Branch Gaps

The branch `hypothesis/smoke-skilldag-cl` is preparation-ready but not end-to-end ready:

- `run_smoke.sh` exits after a successful `dry_run`; it does not then train.
- Skill-DAG mix JSON is printed but not connected to the dataset loader.
- Curriculum order JSONL is printed but not consumed or materialized into a training shard.
- The 190M/370M scripts use default AI2 GCS data/evaluation paths for non-AI2 clusters.
- W&B callbacks are present but disabled and point to different project names.

Correct these gaps after the generic environment smoke. Use OLMo2-190M for the first hypothesis
smokes. The 370M factory exceeds 400M total parameters with the Dolma2 vocabulary and is not part
of the initial smoke.

## S3 Pilot

The current path is:

```text
Small public dataset in S3
  -> staged once to Engaging Scratch/Pool
  -> Slurm GPU job trains from cluster storage
  -> selected checkpoint/result copied back to S3
```

Rules:

- Do not call the current sandbox buckets permanent or production-ready while versioning is off.
- Do not place interactive `sbsandbox` credentials on Engaging.
- Prefer short-lived, least-privilege credentials or presigned URLs.
- Separate read-only dataset access from write-only run-output access.
- Stage data to an approved Engaging path and verify every manifest-referenced shard's size and
  SHA-256 immediately before training.
- Confirm outbound S3 access or an ORCD-managed transfer method with ORCD.
- Use only public/research-cleared data for the pilot.
- Licensed, FERPA, PII, or restricted data requires MIT and organizational approval.
- Measure bytes, duration, throughput, S3 requests, and data-transfer cost.
- Do not have every GPU worker repeatedly stream the same corpus from S3.

If S3 access is not confirmed, the first smoke remains entirely on Engaging storage.

## Staged Verification



### Stage 1: Environment Probe

Request one L40S for at most 30 minutes.

Verify:

- `nvidia-smi`.
- PyTorch import and CUDA availability.
- OLMo import.
- W&B import and outbound connectivity.
- Write/read in Scratch.



### Stage 2: Generic OLMo + W&B Smoke

- OLMo2-190M.
- One L40S.
- 20 steps.
- Tiny local data.
- W&B entity `eduLLM`, project `test`, group `orcd-bootstrap`.
- Success: finite loss, no OOM, run visible to other organization members.



### Stage 3: Checkpoint and Resume

- Save a local checkpoint.
- Restart from it and advance beyond the saved step.
- Preserve the resolved config and metrics.



### Stage 4: S3 Transfer Pilot

- Stage one small public dataset from S3.
- Train from Engaging Scratch.
- Upload one checkpoint/result.
- Record transfer time and cost.



### Stage 5: Hypothesis Smoke

- Correct the Skill-DAG/curriculum data wiring.
- Run one control and one treatment at equal tokens.
- Use stable W&B study groups under `pretraining`.



### Stage 6: Three-Operator Rollout

- Reproduce setup for the other two operators.
- Verify least-loaded assignment and timeout reassignment.
- Verify GitHub and Slack notifications.
- Confirm ORCD policy for the shared project workflow.



## Acceptance Criteria

The initial vertical slice passes when:

1. A researcher creates a valid request through the Agent Skill.
2. GitHub validates and assigns it to the least-loaded operator.
3. The operator receives GitHub and Slack notifications.
4. `edullm run` submits the request after one Kerberos/Duo authentication.
5. A one-L40S, 20-step OLMo2-190M job runs successfully.
6. The run appears under `eduLLM/test` and is visible to another organization member.
7. The Issue contains the Slurm ID, W&B URL, and completed status.
8. `edullm jobs` and `edullm jobs --mine` report correct state.
9. No personal credential appears in Git, GitHub Actions, Slurm output, or W&B config.
10. A second operator command during the ControlPersist window does not require another Duo prompt.



## Failure Handling

- Validation failure: leave the Issue open with actionable corrections; do not assign.
- Operator timeout: remind at 15 minutes and reassign at 30 minutes.
- Slurm pending: remain submitted; do not treat queueing as failure.
- Environment failure: stop before training and attach a redacted diagnostic summary.
- W&B unavailable: retain local W&B files and sync later; do not fail training solely for tracking.
- GPU OOM: fail without automatic resource escalation; researcher must submit a revised request.
- Preemption: preserve checkpoint if available and return the request to ready only after operator or
researcher approval.
- S3 failure: preserve local outputs and mark the transfer step failed; do not discard the run.
