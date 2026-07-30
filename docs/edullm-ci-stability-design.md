# eduLLM CI Stability Design

## Problem

The eduLLM queue correctly rejects a requested commit unless its curated required checks have
completed successfully. On July 23, 2026, the `Lint` check changed from passing with Ruff 0.15.22
to reporting 3,830 repository-wide findings with Ruff 0.16.0, even though the submitted pull
request changed only five documentation lines in `src/examples/llm/train.py`.

The CI environment installs the unbounded `ruff` development dependency through
`uv pip install -e .[all]`. A new Ruff release therefore changed lint behavior without a repository
change. This is dependency drift, not evidence that the eduLLM training smoke path failed.

Other failing `Main` jobs are not part of the eduLLM gate. They exercise S3, Beaker, gated Hugging
Face models, or broader upstream behavior that is unavailable in the repository's current GitHub
Actions credential environment.

## Goals

- Restore deterministic lint behavior immediately.
- Keep the curated eduLLM safety checks unchanged.
- Allow a requested training commit to become eligible after its exact SHA is approved and the
  required checks pass.
- Keep unrelated credential-dependent failures outside the eduLLM eligibility decision.

## Non-goals

- Make every job in the upstream `Main` workflow green.
- Add AWS, Beaker, Hugging Face, ORCD, SSH, Duo, or W&B credentials to GitHub Actions.
- Upgrade the repository to Ruff 0.16.0 in this repair.
- Bypass exact-SHA approval or any existing eduLLM validation gate.

## Design

Pin the `ruff` development dependency in `pyproject.toml` to `ruff==0.15.22`, the exact version
that passed immediately before the unexpected 0.16.0 upgrade. The existing CI setup installs the
project's `all` extra, so the pin applies to both `Lint` and `Lint (min Python)`. Changing
`pyproject.toml` also changes the existing dependency-cache key, preventing reuse of the old
unbounded environment.

Do not change `config/edullm/policy.yaml`. The queue continues to require:

- `Lint`
- `Test transformer`
- `Test attention`
- `Test examples`
- `Test scripts`
- `Test eduLLM core`
- `Integration tests`
- `Type check`
- `Build`
- `Style`
- `Docs`

The broader `Test`, `Test checkpoint`, `Test olmo3 ladder`, credential-dependent tests, and GPU
jobs remain visible in `Main` but are not added to the eduLLM eligibility gate.

## Verification

The repair is accepted when:

1. The isolated baseline remains healthy (`src/test/edullm`: 994 tests pass).
2. The installed repair environment reports Ruff 0.15.22.
3. `make lint-check` passes locally with Ruff 0.15.22.
4. The repair pull request's `Lint` and `Lint (min Python)` jobs use Ruff 0.15.22 and pass.
5. All eleven curated eduLLM checks pass on the exact requested training commit.

## Rollout

1. Merge the reviewed Ruff pin after its CI evidence is available.
2. Update the experiment pull request onto the repaired `main`, producing a new head SHA.
3. Have a different listed approver approve that exact SHA after required CI passes.
4. Submit a new request through `/submit-edullm-job`.
5. Confirm validation, assignment, Slack notification, operator submission, Slurm status, and W&B
   completion in order.

## Alternatives rejected

- **Remove `Lint` from the gate:** fastest, but weakens every future request and hides a known
  reproducibility defect.
- **Accept Ruff 0.16.0 and fix all 3,830 findings now:** a large repository-wide migration unrelated
  to opening the queue.
- **Require every `Main` job:** blocks eduLLM on external credentials and unrelated upstream tests.

Ruff 0.16.0 can be adopted later through a separate reviewed migration with explicit rule and code
changes.
