# eduLLM SSH Session Reliability Design

**Status:** Approved in chat on 2026-07-23

## Problem

Operator commands such as `edullm jobs` run SSH subprocesses with a 30-second timeout. When the
one-hour OpenSSH ControlMaster has expired, SSH starts an interactive password and Duo login inside
that same timeout. Authentication time consumes the budget, the Slurm query is cancelled, and the
CLI reduces the underlying error to `edullm jobs failed`.

The observed failure is deterministic:

- No `~/.ssh/edullm-*` control socket was present.
- `edullm jobs` prompted for password and Duo.
- A traced run failed after 32.7 seconds with `authoritative Slurm query failed`.
- Explicitly running `ssh -MNf orcd-login` produced `Master running`, after which `edullm jobs`
  returned the expected job list.

## Goals

- Automatically establish a healthy personal ControlMaster before any interactive operator command
  contacts Engaging.
- Allow up to five minutes for password and Duo authentication.
- Reuse a healthy master without prompting.
- Prevent ordinary Slurm operations from starting hidden interactive authentication.
- Keep errors sanitized while telling operators how to recover.
- Preserve the existing one-hour persistence and `edullm logout` behavior.

## Non-goals

- No unattended SSH credentials or GitHub Actions access to personal Engaging accounts.
- No changes to Slurm authorization, job ownership, W&B credentials, or request validation.
- No storage of passwords, Duo responses, private keys, or raw SSH diagnostics.
- No longer ControlPersist interval; automatic reconnection removes that requirement.

## Design

### SSH session lifecycle

`SSHClient.ensure_master()` will own the session transition:

1. Probe `orcd-login` with fixed `BatchMode=yes`, a short connection timeout, and both diagnostic
   streams sent directly to `DEVNULL`.
2. If the probe succeeds, reuse the healthy master without user interaction.
3. If it fails, close any stale project master on a best-effort basis with both diagnostic streams
   sent directly to `DEVNULL`.
4. Launch the fixed command `ssh -MNf orcd-login` with inherited stdin and preserved
   controlling-terminal access so password and Duo prompts remain visible, suppress ordinary
   stdout/stderr diagnostics, and allow up to five minutes for authentication.
5. Probe again in batch mode. If verification fails, raise a sanitized session error.

Only fixed project-controlled arguments are used. User input never becomes SSH options or shell
text.

### Operational commands

Loading operator services for `jobs`, `run`, `logs`, and `stop` will call `ensure_master()` once.
All subsequent remote commands and remote writes will include `BatchMode=yes`. This guarantees that
the bounded 30-second command timeout measures the Slurm or file operation rather than human
authentication.

### Setup

After the operator approves and applies the managed SSH configuration, `edullm setup` will establish
and verify the ControlMaster before running remote environment checks. Existing configurations use
the same path. This gives first-time operators the same five-minute visible authentication window
as existing operators.

### Errors

Session bootstrap failures will produce one fixed actionable message:

`ORCD SSH login failed; run ssh -MNf orcd-login and retry.`

Raw subprocess output, usernames beyond existing public configuration, private paths, and
credentials remain suppressed. Non-SSH failures retain their current sanitized behavior.

## Testing

Focused tests will prove:

- A healthy master is reused without starting an interactive login.
- A missing or stale master starts one interactive login and is verified afterward.
- Failed or timed-out login raises only the sanitized session error.
- Remote commands and writes always use `BatchMode=yes`.
- Operator commands call the session bootstrap before Slurm access.
- Setup establishes the master before remote checks.
- `edullm logout` still closes only the project ControlMaster.

The focused SSH and CLI suites, lint checks for changed files, and a real operator test will verify
the implementation.
