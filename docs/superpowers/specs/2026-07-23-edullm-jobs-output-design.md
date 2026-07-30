# eduLLM Jobs Output Design

**Status:** Approved in chat on 2026-07-23

## Problem

`edullm jobs --mine` omits requests that are assigned but have not been submitted. The jobs service
explicitly skips `status:assigned` Issues because they do not yet have a Slurm attempt. As a result,
Issue #20 was invisible in `edullm jobs --mine` even though `edullm run` correctly selected and
submitted it.

The command also prints dense one-line records with long W&B URLs, making status and next actions
hard to scan.

## Goals

- Show every request assigned to the current operator, including requests waiting for
  `edullm run`.
- Put waiting requests first and clearly state the next action.
- Mark only the lowest-numbered assigned request as next; later assigned requests wait behind it.
- Keep submitted and recent jobs visible with Slurm and W&B identities.
- Use one authoritative GitHub Issue scan and avoid synthetic Slurm lifecycle records.
- Keep output deterministic, plain text, and straightforward to test.

## Non-goals

- No changes to assignment, authorization, submission order, Slurm reconciliation, or W&B behavior.
- No colors, interactive terminal UI, pagination, or new command-line options.
- No change to the current operator-only visibility boundary.

## Design

### Job summary model

Add an immutable `OperatorJobSummary` in `edullm.jobs` with:

- `issue: int`
- `status: str`
- `slurm_job_id: str | None`
- `wandb_url: str | None`

An assigned summary has no Slurm or W&B values. Every later lifecycle state has both values from its
latest validated attempt. The jobs service returns summaries rather than forcing an assigned request
into a lifecycle attempt that does not exist.

### Single-scan collection

The existing operator Issue scan will partition matching Issues. Each matching Issue still passes
the existing authorization gate against a fresh exact Issue fetch and fresh comments, while the
gate reuses the corresponding row from the initial scan instead of repeating the full queue scan:

- `assigned`: create a summary immediately and do not query Slurm.
- later states: run the existing authorization and Slurm reconciliation path, then create summaries
  from the repaired lifecycle state.

If all matching requests are assigned, no Slurm query occurs. Assigned summaries sort by ascending
Issue number, matching the oldest-first selection used by `edullm run`. Submitted and recent
summaries sort by descending Issue number.

### Rendering

`edullm jobs` and `edullm jobs --mine` render:

```text
eduLLM jobs for philote-dev

Ready to run (1)
  #20  assigned
       Next: edullm run

Submitted and recent (3)
  #12  completed   Slurm 18653501
       W&B: https://wandb.ai/...
  #8   completed   Slurm 18641420
       W&B: https://wandb.ai/...
  #6   failed      Slurm 18639510
       W&B: https://wandb.ai/...
```

Empty sections display a zero count. If the operator has no matching jobs at all, the command prints
`No eduLLM jobs assigned to <operator>.`

When more than one assigned request is present, only the first, lowest-numbered row displays
`Next: edullm run`. Every later assigned row displays `Waiting behind #<first issue>`, because
`edullm run` always submits the lowest-numbered assigned Issue.

## Error Handling

Existing fail-closed behavior remains unchanged. A malformed status, assignment, lifecycle comment,
Slurm response, or summary invariant fails the command rather than displaying untrusted or
inconsistent information.

## Testing

Focused tests will prove:

- Assigned Issue #20 appears before any submitted or terminal jobs.
- Assigned-only output does not call Slurm.
- Multiple assigned requests are ordered exactly as `edullm run` selects them.
- Assigned summaries fail closed on malformed or mismatched authorization evidence.
- One complete jobs operation performs exactly one full active-queue scan while retaining exact
  current Issue and comment fetches.
- Exact output identifies only the lowest assigned Issue as next and points later rows to it.
- Submitted and terminal reconciliation still updates labels and summaries.
- Grouped CLI output is exact for mixed, empty-ready, and no-job cases.
- Existing SSH recovery and generic error messages remain unchanged.
