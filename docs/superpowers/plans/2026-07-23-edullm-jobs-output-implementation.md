# eduLLM Jobs Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show assigned requests before Slurm-backed jobs and render a readable grouped operator
dashboard.

**Architecture:** The jobs service produces validated `OperatorJobSummary` records from one GitHub
Issue scan. The CLI renders those records in two deterministic plain-text groups without changing
submission or reconciliation policy.

**Tech Stack:** Python 3.10+, dataclasses, GitHub Issues, Slurm, pytest, Ruff

## Global Constraints

- Assigned requests have no synthetic Slurm or W&B attempt.
- Assigned-only listings do not query Slurm.
- Assigned requests sort oldest-first; later states sort newest-Issue-first.
- Only the lowest-numbered assigned request says `Next: edullm run`; every later assigned request
  says `Waiting behind #<lowest issue>`.
- Existing authorization, submission, reconciliation, SSH, and error behavior remain unchanged.
- Output contains only validated numeric IDs, bounded statuses, canonical Slurm IDs, and W&B URLs.
- Production behavior must be preceded by a focused failing regression test.
- Do not stage, commit, or push without an explicit user request.

---

### Task 1: Return assigned and Slurm-backed job summaries

**Files:**
- Modify: `src/edullm/jobs.py`
- Test: `src/test/edullm/jobs_test.py`

**Interfaces:**
- Produces: `OperatorJobSummary(issue, status, slurm_job_id, wandb_url)`
- Changes: `jobs(...) -> tuple[OperatorJobSummary, ...]`

- [ ] **Step 1: Write failing tests for assigned visibility**

Add a test with one operator-assigned Issue and no lifecycle attempt:

```python
def test_jobs_lists_assigned_request_without_querying_slurm(valid_request):
    github = StatefulGitHub(valid_request)
    slurm = CompletedSlurm()

    summaries = jobs(
        mine=True,
        operator="operator",
        github=github,
        configuration=_config(),
        slurm=slurm,
        now=NOW,
    )

    assert summaries == (
        OperatorJobSummary(
            issue=42,
            status="assigned",
            slurm_job_id=None,
            wandb_url=None,
        ),
    )
    assert slurm.queries == []
```

Add a second test with reversed assigned Issue order and assert ascending order matches
`run_assigned()` selection.

- [ ] **Step 2: Run the assigned tests and verify RED**

Run:

```bash
pytest -q src/test/edullm/jobs_test.py -k "jobs_lists_assigned or jobs_orders_assigned"
```

Expected: FAIL because assigned Issues are skipped and `OperatorJobSummary` does not exist.

- [ ] **Step 3: Add the summary model and assigned partition**

Add the immutable public dataclass:

```python
@dataclass(frozen=True)
class OperatorJobSummary:
    """One validated operator-facing queue record."""

    issue: int
    status: str
    slurm_job_id: str | None
    wandb_url: str | None
```

Validate that `issue` is positive, `status` is a supported lifecycle state, assigned summaries have
neither optional value, and later summaries have both validated values.

During the existing Issue scan, pass each matching assigned Issue through the existing authorization
gate without external commit revalidation or a Slurm query. Reuse the corresponding initial scan row
inside the gate while still fetching the exact current Issue and comments. Continue using the same
fresh authorization and reconciliation path for every later state.

- [ ] **Step 4: Verify assigned tests GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Update the terminal reconciliation regression**

Change the existing terminal repair assertion from `LifecycleState.current_state` to the complete
summary:

```python
assert summaries == (
    OperatorJobSummary(
        issue=42,
        status="completed",
        slurm_job_id="12345",
        wandb_url="https://wandb.ai/eduLLM/test/runs/issue-42-attempt-1-12345",
    ),
)
```

Retain its label-repair, Slurm-query, and no-review-call assertions.

- [ ] **Step 6: Implement deterministic combined ordering**

Return assigned summaries sorted by ascending Issue number followed by reconciled summaries sorted
by descending Issue number. Run:

```bash
pytest -q src/test/edullm/jobs_test.py
```

Expected: all jobs tests pass.

---

### Task 2: Render the grouped operator dashboard

**Files:**
- Modify: `src/edullm/cli.py`
- Test: `src/test/edullm/cli_test.py`
- Modify: `docs/edullm-team-workflow.md`

**Interfaces:**
- Consumes: `tuple[OperatorJobSummary, ...]`
- Produces: `_format_operator_jobs(operator, summaries) -> str`

- [ ] **Step 1: Write failing exact-output tests**

Add one mixed summary test expecting:

```text
eduLLM jobs for operator

Ready to run (1)
  #20  assigned
       Next: edullm run

Submitted and recent (1)
  #12  completed   Slurm 18653501
       W&B: https://wandb.ai/eduLLM/test/runs/example
```

Add tests for no assigned requests (`Ready to run (0)`) and no jobs
(`No eduLLM jobs assigned to operator.`).

- [ ] **Step 2: Run rendering tests and verify RED**

Run:

```bash
pytest -q src/test/edullm/cli_test.py -k "format_operator_jobs"
```

Expected: FAIL because `_format_operator_jobs` does not exist.

- [ ] **Step 3: Implement minimal deterministic rendering**

Partition summaries by `status == "assigned"` and order that group by ascending Issue number. Render
only the first row with `Next: edullm run`; render each later assigned row with
`Waiting behind #<first issue>`. Render each later lifecycle state with its status and Slurm ID on
one line and its W&B URL on the next line. Do not add colors or terminal-width detection.

- [ ] **Step 4: Wire `handle_jobs()` and verify GREEN**

Replace the existing one-line print loop with one print of `_format_operator_jobs()`. Run:

```bash
pytest -q src/test/edullm/cli_test.py -k "format_operator_jobs or handle_jobs"
```

Expected: selected tests pass.

- [ ] **Step 5: Update operator documentation**

Document that `edullm jobs --mine` shows a `Ready to run` section and that operators inspect the
first, lowest-numbered row because `edullm run` always submits that request.

- [ ] **Step 6: Run full focused verification**

Run:

```bash
pytest -q \
  src/test/edullm/jobs_test.py \
  src/test/edullm/cli_test.py \
  src/test/edullm/ssh_test.py
ruff check \
  src/edullm/jobs.py src/edullm/cli.py \
  src/test/edullm/jobs_test.py src/test/edullm/cli_test.py
black --check \
  src/edullm/jobs.py src/edullm/cli.py \
  src/test/edullm/jobs_test.py src/test/edullm/cli_test.py
isort --check-only \
  src/edullm/jobs.py src/edullm/cli.py \
  src/test/edullm/jobs_test.py src/test/edullm/cli_test.py
git diff --check
```

Expected: all tests and checks pass.

- [ ] **Step 7: Live acceptance**

Run:

```bash
edullm jobs --mine
```

Expected: any assigned request appears under `Ready to run`; submitted and terminal jobs appear
under `Submitted and recent`.
