# eduLLM request reference

`src/edullm/request_parser.py`, `src/edullm/policy.py`, and
`src/edullm/validation.py` are the sole request schema and policy. This page
explains how to collect values; it does not define another request model.

## Fields in authoritative order

Build one JSON object whose string keys follow `ISSUE_HEADINGS` exactly:

1. `Purpose` — the submitting team's reason for the request.
2. `Study` — the submitting team's stable study identifier.
3. `Condition` — the submitting team's condition identifier.
4. `Comparison` — the submitting team's comparison or control.
5. `Commit SHA` — the exact full pushed commit SHA from `edu-llm/OLMo-core`.
   Reject a direct-main SHA and any value other than the verified 40-character
   lowercase SHA.
6. `Entrypoint profile` — a profile accepted by the trusted policy.
7. `Script path` — the repository-relative script selected by the team.
8. `Launcher` — the launcher's policy value, not a shell command.
9. `Arguments JSON` — an ordered JSON array of strings, never a shell command.
10. `Data manifest` — `builtin://generic-smoke-v1` or a policy-controlled
    `/orcd/pool/...` manifest; no S3 URI.
11. `Data manifest SHA-256` — the exact manifest's lowercase digest.
12. `Data classification` — the team's classification.
13. `Seed` — the submitting team's seed.
14. `W&B project` — a project accepted by policy.
15. `Success signal` — the submitting team's scientific or engineering signal.
16. `Success metrics` — comma-separated names actually emitted by the selected code.
17. `GPU count` — the requested integer count within policy.
18. `GPU preference` — the requested policy value.
19. `Maximum runtime minutes` — the requested integer runtime within policy.

Do not copy these descriptions into a separate schema. Create the JSON, then run
the validation adapter so the parser, policy, and validator decide acceptance.

## Team-owned science

The submitting team owns the hypothesis, condition, comparison, seed,
model/code/config, data choice, manifest, mixture, curriculum, success signal,
and scientific metrics. Ask for missing intent one field at a time. Never infer
these values, alter a data mix, or substitute generic-smoke values into a
scientific request.

Read the selected script and configuration plus its W&B callback before
collecting `Success metrics`. If a requested metric is not emitted, stop and
direct the user to `/weights-and-biases`. Metric wiring requires a new commit,
the exact updated pushed SHA, and a passing exact-SHA gate before a request can
be created. Do not call W&B APIs from this Skill.

## Only offerable platform fixture

For an engineering generic smoke, and only when the submitter chooses it, the
Skill may offer these exact platform defaults:

- profile `generic-smoke`
- script `src/examples/llm/train.py`
- launcher `torchrun`
- arguments JSON `["orcd-bootstrap"]`
- one L40S
- 30 minutes
- W&B project `test`
- manifest `builtin://generic-smoke-v1`
- manifest digest
  `1c82abfc35b17e8a15eae8e0e1afa3dee6696aeb213d46799f204e1c4fc093d7`

The team still supplies every team-owned value. Do not offer other defaults.

## Status and execution boundary

The Skill creates one validated request Issue. GitHub Actions performs validation
and assignment, and its recorded state is authoritative. Report `requested`,
validation errors, `ready`, or the assigned operator from Issue state; do not
infer assignment.

PR review controls merging to main.
The assigned operator authorizes a job by running `edullm run`.

`edullm run` is operator-only. This Skill never runs compute, calls SSH or
Slurm, accesses W&B, handles credentials, or acts as an operator.
