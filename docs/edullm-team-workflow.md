# eduLLM Team Quick Start

This handout is for a meeting where the host is ready to open the eduLLM job
workflow. Do not announce that teammates can submit until the checklist below is
complete.

## Host Pre-Meeting Checklist

- Confirm PR #14 and PR #13 have merged.
- Confirm the no-compute acceptance test has passed.
- Grant every submitter Write access to the canonical repository:
  `https://github.com/edu-llm/OLMo-core`.
- Share repository access and W&B access with the team.
- Ensure at least one enabled operator is on call and can run:

```bash
edullm jobs --mine
```

Enabled operators are `philote-dev`, `meric233`, and `alsy7009`.

## The Post-Merge Flow

```text
Teammate pushes the exact clean non-main commit to edu-llm/OLMo-core
  -> teammate invokes /submit-edullm-job in Cursor
  -> GitHub Issue validation assigns an operator and sends Slack
  -> assigned operator inspects the Issue and runs edullm run
  -> Slurm runs the job
  -> W&B and the GitHub Issue show status and results
```

No pull request, reviewer approval, or PR CI is required to run a job. PRs and
future rulesets govern merging to `main` only.

The `/submit-edullm-job` Skill creates one validated request Issue. It does not
submit compute. The assigned operator authorizes and submits compute only by
running `edullm run`.

`edullm jobs --mine` shows assigned requests under `Ready to run` and later
jobs under `Submitted and recent`. Inspect the first, lowest-numbered Ready
row: `edullm run` always submits that request. Later Ready rows wait behind it.
A `Ready to run (0)` result means there is no request to submit.

## Teammate Setup

Prerequisites for macOS or Linux:

- Git
- GitHub CLI, `gh`
- Cursor
- Python 3.10 or newer

Copy and paste:

```bash
git --version
gh --version
python3 --version

gh auth login

git clone https://github.com/edu-llm/OLMo-core.git
cd OLMo-core
git switch main
git pull --ff-only origin main

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

cursor .
```

For an existing clone, start from the repository root and run:

```bash
git switch main
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -e .
cursor .
```

The editable install is required because the Skill's validation adapter imports
the eduLLM modules from the local checkout before it can preview or create the
request Issue.

## Branch Requirements

Before invoking `/submit-edullm-job`, the teammate owns these preconditions:

- Work on a non-`main` branch.
- Use `origin` pointing at the canonical repository,
  `https://github.com/edu-llm/OLMo-core.git`.
- Commit all code, config, data choices, and metric wiring.
- Push the full commit to the canonical repository.
- Keep the local tree clean.
- Use the exact 40-character commit SHA that is already pushed.

Fork-only commits are not eligible. Direct-`main` SHAs are not eligible.

## W&B Expectations

The teammate owns metric wiring. If the selected script does not emit the
scientific metric the team wants to track, use `/weights-and-biases` first, then
commit and push the metric changes before submitting the eduLLM request.

Do not put API keys in GitHub, Slack, or Git.

Operator-side credentials are used for the actual run. Runs log to W&B entity
`eduLLM` and one of these allowed projects: `test`, `pretraining`,
`posttraining`, `evaluation`, or `data-pipeline`.

The GitHub Issue will carry the W&B link after the run is authorized and status
is recorded.

## Meeting Prompt For Teammates

After the teammate has pushed the exact clean non-main commit to
`edu-llm/OLMo-core`, they can paste this into Cursor:

```text
/submit-edullm-job

The branch is pushed to edu-llm/OLMo-core and my local tree is clean. Please run
the repository and pushed-commit gate first, then collect any missing request
fields one at a time. Do not invent scientific fields. If a requested metric is
not emitted by the selected code, stop and send me to /weights-and-biases.

Only offer generic-smoke if I explicitly choose an engineering demo request.
```

The `generic-smoke` option is for engineering demonstrations only. It is not a
stand-in for a scientific hypothesis, comparison, data choice, seed, or metric.

## Operator Demo

Safe meeting demo, with no compute:

1. Show the Slack assignment notification.
2. Run `edullm jobs --mine`.
3. Open and inspect the assigned GitHub Issue.
4. Stop there. Do not run `edullm run`.

Full operator flow, when the team is ready to submit compute:

1. Read the Slack assignment.
2. Run `edullm jobs --mine`.
3. If `Ready to run` contains assigned requests, inspect the GitHub Issue for
   the first, lowest-numbered row, including the commit, script, arguments,
   data manifest, W&B project, metrics, GPU request, and runtime.
4. Run `edullm run` only for that first request; later rows wait behind it.

Running `edullm run` is the operator's approval and compute submission. It
revalidates the request, performs the submission transaction, and sends the job
to Slurm.

## Troubleshooting

Skill is missing:

- Pull the latest `main` from `edu-llm/OLMo-core`.
- Open the repository root in Cursor with `cursor .`.
- Confirm the repository contains `.cursor/skills/submit-edullm-job/`.

`gh` is missing or not authenticated:

- Install GitHub CLI.
- Run `gh auth login`.
- Check with `gh auth status`.

Python is too old:

- Install Python 3.10 or newer.
- Recreate the virtual environment with that Python.
- Re-run `python -m pip install -e .`.

`ModuleNotFoundError: edullm`:

- Activate the repository virtual environment.
- Run `python -m pip install -e .` from the repository root.
- Retry `/submit-edullm-job`.

Repository or commit gate failure:

- Switch to a non-`main` branch.
- Commit all local changes so the tree is clean.
- Make sure `origin` is `https://github.com/edu-llm/OLMo-core.git`.
- Push the exact commit to `edu-llm/OLMo-core`.
- Retry with the full 40-character SHA.

`edullm: command not found` for an operator:

- Activate the operator's Python environment.
- From the repository root, run `python -m pip install -e '.[wandb]'`.
- Confirm the install with `edullm jobs --mine`.

## ORCD SSH session

The first `edullm setup`, `edullm jobs`, `edullm run`, `edullm logs`, or `edullm stop` command
without a healthy shared ORCD SSH session automatically starts one. Your terminal prompts for your
ORCD password and Duo approval; eduLLM does not capture either credential.

After a successful login, later eduLLM commands reuse the same SSH session for up to one hour.
`edullm logout` still closes it immediately.

If automatic login fails, start the session manually and retry the eduLLM command:

```bash
ssh -MNf orcd-login
```

## Copy/Paste Announcement

```text
eduLLM job requests are ready for the team after the host checklist is complete.
Start here: https://github.com/edu-llm/OLMo-core/blob/main/docs/edullm-team-workflow.md

To run a job, push a clean non-main branch commit to edu-llm/OLMo-core, open the
repository in Cursor, and invoke /submit-edullm-job. The Skill creates a
validated Issue; the assigned operator reviews it and runs edullm run to submit
compute.
```
