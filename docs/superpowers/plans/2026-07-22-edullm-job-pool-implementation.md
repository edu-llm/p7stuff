# eduLLM GitHub Queue and Operator Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `edu-llm/OLMo-core` Issues into a validated GPU-job queue assigned across three MIT
Engaging operators, with one-command Slurm submission, W&B links/status, GitHub/Slack notifications,
and no shared personal credentials.

**Architecture:** Add a separate `edullm` Python namespace and console command in the fork; do not
add queue logic to `olmo_core`. GitHub Actions handle request validation, assignment, reminders,
and optional W&B reconciliation. Operator-local commands alone hold SSH access and invoke Slurm.
The shared Agent Skill creates requests but never accesses credentials.

**Tech Stack:** Python 3.11+, stdlib `argparse`/dataclasses/subprocess, PyYAML, requests, W&B API,
GitHub REST/Issues/Actions, Slack webhook, OpenSSH ControlMaster, Slurm `sbatch`/`squeue`/`sacct`.

## Global Constraints

- No SSH/Kerberos/Duo credentials in GitHub, W&B, Slack, configuration, or test fixtures.
- Execute only a full SHA belonging to a green, approved PR or merged `main`.
- Do not execute a shell string from an Issue; render argv from typed fields without `eval`.
- Initial policy permits one or two GPUs and at most six hours; smoke default is one L40S/30 min.
- W&B entity is `eduLLM`; allowed projects are `test`, `pretraining`, `posttraining`,
  `evaluation`, and `data-pipeline`.
- One W&B group represents a durable study, not an Issue.
- GitHub Actions have explicit least-privilege `permissions`.
- Use full commit SHA pins for new third-party GitHub Actions.
- Operators submit through their own Engaging accounts.

---

## File Structure

Create:

```text
src/edullm/
  __init__.py
  cli.py
  models.py
  policy.py
  request_parser.py
  validation.py
  assignment.py
  github.py
  notifications.py
  data_manifest.py
  ssh.py
  slurm.py
  jobs.py
  wandb_status.py

config/edullm/
  entrypoints.yaml
  main-ruleset.json
  policy.yaml
  operators.yaml
  operators.example.yaml

.github/
  CODEOWNERS
  ISSUE_TEMPLATE/config.yml
  ISSUE_TEMPLATE/edullm-job-request.yml
  workflows/edullm-validate.yml
  workflows/edullm-assign.yml
  workflows/edullm-reminders.yml
  workflows/edullm-wandb-reconcile.yml

.cursor/skills/submit-edullm-job/
  SKILL.md
  request-reference.md
  scripts/validate_request.py

src/test/edullm/
  conftest.py
  models_test.py
  request_parser_test.py
  validation_test.py
  assignment_test.py
  github_test.py
  ssh_test.py
  slurm_test.py
  jobs_test.py
  data_manifest_test.py
  wandb_status_test.py
  fixtures/
    valid_issue.md
    operators.yaml
    policy.yaml
```

Modify:

```text
pyproject.toml
.github/workflows/main.yml
docs/superpowers/specs/2026-07-22-orcd-job-pool-design.md
```

Package boundary:

- `src/edullm/` is project-specific and separate from `src/olmo_core/`.
- `edullm` may use `requests`/PyYAML, but not OLMo training internals.
- Training remains in existing scripts; the CLI only prepares and submits commands.

---

### Task 1: Package, Models, and Policy

**Files:**
- Create: `src/edullm/__init__.py`
- Create: `src/edullm/models.py`
- Create: `src/edullm/policy.py`
- Create: `config/edullm/policy.yaml`
- Create: `config/edullm/entrypoints.yaml`
- Create: `config/edullm/operators.yaml`
- Create: `config/edullm/operators.example.yaml`
- Modify: `pyproject.toml`
- Create: `src/test/edullm/conftest.py`
- Test: `src/test/edullm/models_test.py`

**Interfaces:**
- Produces: `JobRequest`, `JobStatus`, `ResolvedRequest`, `Policy`, `Operator`.
- Consumed by: all queue workflows and operator commands.

- [ ] **Step 1: Write failing model tests**

```python
# src/test/edullm/models_test.py
from edullm.models import JobRequest, JobStatus
from edullm.policy import load_policy


def test_job_request_generates_stable_name():
    request = JobRequest(
        issue_number=42,
        requester="student",
        purpose="Skill-DAG smoke",
        study="skill-dag-v1",
        condition="natural",
        comparison="fixed-uniform",
        commit_sha="a" * 40,
        entrypoint_profile="hypothesis-smoke",
        script_path="src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py",
        launcher="python",
        argv=("train_single", "skilldag-natural", "local", "--seed=0"),
        data_manifest="builtin://generic-smoke-v1",
        data_manifest_sha256="b" * 64,
        data_classification="public",
        seed=0,
        wandb_project="pretraining",
        success_signal="20 steps and finite loss",
        success_metrics=("train/loss",),
        gpu_count=1,
        gpu_preference="l40s",
        max_runtime_minutes=30,
    )
    assert request.request_name == "issue-42-skill-dag-v1-natural"
    assert request.status is JobStatus.REQUESTED


def test_policy_loads_allowed_projects(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        "wandb_entity: eduLLM\n"
        "allowed_wandb_projects: [test]\n"
        "required_checks: [Lint]\n"
    )
    (tmp_path / "entrypoints.yaml").write_text("entrypoints: {}\n")
    assert load_policy(path).wandb_entity == "eduLLM"
```

- [ ] **Step 2: Add shared test fixtures**

```python
# src/test/edullm/conftest.py
import pytest

from edullm.models import JobRequest, ResolvedRequest
from edullm.policy import Policy


@pytest.fixture
def valid_request():
    return JobRequest(
        issue_number=42,
        requester="student",
        purpose="Skill-DAG smoke",
        study="skill-dag-v1",
        condition="natural",
        comparison="fixed-uniform",
        commit_sha="a" * 40,
        entrypoint_profile="hypothesis-smoke",
        script_path="src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py",
        launcher="python",
        argv=("train_single", "skilldag-natural", "local", "--seed=0"),
        data_manifest="builtin://generic-smoke-v1",
        data_manifest_sha256="b" * 64,
        data_classification="public",
        seed=0,
        wandb_project="pretraining",
        success_signal="20 steps and finite loss",
        success_metrics=("train/loss",),
        gpu_count=1,
        gpu_preference="l40s",
        max_runtime_minutes=30,
    )


@pytest.fixture
def policy():
    return Policy(
        "eduLLM",
        ("test", "pretraining"),
        entrypoints={
            "hypothesis-smoke": {
                "script": "src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py",
                "launcher": "python",
                "positionals": 3,
                "allowed_positionals": {0: ["dry_run", "train_single", "train"], 2: ["local"]},
                "allowed_options": {"seed": {"type": "integer", "min": 0, "max": 2147483647}},
            }
        },
    )


@pytest.fixture
def valid_resolved_request(valid_request):
    return ResolvedRequest(
        request=valid_request,
        operator="operator",
        wandb_entity="eduLLM",
        wandb_run_prefix="issue-42-attempt-1",
        slurm_job_name="issue-42-skill-dag-v1-natural",
        log_pattern="logs/issue-42-attempt-1-%j.log",
        allowed_data_kinds=("skill-dag", "curriculum"),
    )
```

- [ ] **Step 3: Run and confirm failure**

```bash
pytest -v src/test/edullm/models_test.py
```

Expected: FAIL because the `edullm` package does not exist.

- [ ] **Step 4: Add package metadata**

Modify `pyproject.toml`:

```toml
[project.scripts]
edullm = "edullm.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
include = ["olmo_core*", "edullm*"]
exclude = []
```

No new base dependency is required: PyYAML and requests already exist.

- [ ] **Step 5: Implement immutable request models**

```python
# src/edullm/models.py
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:48]


class JobStatus(str, Enum):
    REQUESTED = "requested"
    VALIDATING = "validating"
    READY = "ready"
    ASSIGNED = "assigned"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


@dataclass(frozen=True)
class JobRequest:
    issue_number: int
    requester: str
    purpose: str
    study: str
    condition: str
    comparison: str
    commit_sha: str
    entrypoint_profile: str
    script_path: str
    launcher: str
    argv: tuple[str, ...]
    data_manifest: str
    data_manifest_sha256: str
    data_classification: str
    seed: int
    wandb_project: str
    success_signal: str
    success_metrics: tuple[str, ...]
    gpu_count: int
    gpu_preference: str
    max_runtime_minutes: int
    status: JobStatus = JobStatus.REQUESTED

    @property
    def request_name(self) -> str:
        return f"issue-{self.issue_number}-{slug(self.study)}-{slug(self.condition)}"

    def canonical_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True)
class Operator:
    github: str
    slack_user_id: str
    rotation_order: int
    enabled: bool = True
    apptainer_path: str | None = None
    apptainer_sha256: str | None = None


@dataclass(frozen=True)
class ResolvedRequest:
    request: JobRequest
    operator: str
    wandb_entity: str
    wandb_run_prefix: str
    slurm_job_name: str
    log_pattern: str
    allowed_data_kinds: tuple[str, ...]
    slurm_job_id: str | None = None


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    request_digest: str
    operator: str
    slurm_job_id: str
    wandb_run_id: str
    log_path: str
```

- [ ] **Step 6: Implement policy loading**

```python
# src/edullm/policy.py
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Policy:
    wandb_entity: str
    allowed_wandb_projects: tuple[str, ...]
    max_runtime_minutes: int = 360
    max_gpu_count: int = 2
    allowed_gpu_preferences: tuple[str, ...] = ("any", "l40s", "h100", "h200")
    required_checks: tuple[str, ...] = (
        "Lint",
        "Test",
        "Test checkpoint",
        "Test transformer",
        "Test attention",
        "Test examples",
        "Test scripts",
        "Integration tests",
        "Test olmo3 ladder",
        "Type check",
        "Build",
        "Style",
        "Docs",
    )
    entrypoints: dict[str, dict] = field(default_factory=dict)
    reminder_after_minutes: int = 15
    reassign_after_minutes: int = 30


def load_policy(path: Path, entrypoints_path: Path | None = None) -> Policy:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entrypoints_path = entrypoints_path or path.with_name("entrypoints.yaml")
    entrypoints = yaml.safe_load(entrypoints_path.read_text(encoding="utf-8"))["entrypoints"]
    return Policy(
        wandb_entity=data["wandb_entity"],
        allowed_wandb_projects=tuple(data["allowed_wandb_projects"]),
        max_runtime_minutes=int(data.get("max_runtime_minutes", 360)),
        max_gpu_count=int(data.get("max_gpu_count", 2)),
        allowed_gpu_preferences=tuple(
            data.get("allowed_gpu_preferences", ["any", "l40s", "h100", "h200"])
        ),
        required_checks=tuple(data["required_checks"]),
        entrypoints=entrypoints,
        reminder_after_minutes=int(data.get("reminder_after_minutes", 15)),
        reassign_after_minutes=int(data.get("reassign_after_minutes", 30)),
    )


def load_operators(path: Path):
    from edullm.models import Operator

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(
        Operator(
            github=row["github"],
            slack_user_id=row["slack_user_id"],
            rotation_order=int(row["rotation_order"]),
            enabled=bool(row.get("enabled", True)),
            apptainer_path=row.get("apptainer_path"),
            apptainer_sha256=row.get("apptainer_sha256"),
        )
        for row in data["operators"]
    )
```

Create `config/edullm/policy.yaml`:

```yaml
wandb_entity: eduLLM
allowed_wandb_projects:
  - test
  - pretraining
  - posttraining
  - evaluation
  - data-pipeline
max_runtime_minutes: 360
max_gpu_count: 2
allowed_gpu_preferences: [any, l40s, h100, h200]
required_checks:
  - Lint
  - Test
  - Test checkpoint
  - Test transformer
  - Test attention
  - Test examples
  - Test scripts
  - Integration tests
  - Test olmo3 ladder
  - Type check
  - Build
  - Style
  - Docs
reminder_after_minutes: 15
reassign_after_minutes: 30
```

Create `config/edullm/entrypoints.yaml`:

```yaml
entrypoints:
  generic-smoke:
    script: src/examples/llm/train.py
    launcher: torchrun
    wandb_callback: true
    allowed_data_kinds: [generic-smoke]
    positionals: 1
    allowed_options:
      model-factory: {type: string, values: [olmo2_190M]}
      sequence-length: {type: integer, min: 128, max: 2048}
      save-folder: {type: path, roots: ["$HOME/orcd/scratch/edullm"]}
      work-dir: {type: path, roots: ["$HOME/orcd/scratch/edullm"]}
      trainer.hard_stop: {type: duration, max_steps: 100}
      trainer.callbacks.wandb.enabled: {type: boolean}
      trainer.callbacks.wandb.entity: {type: string, values: [eduLLM]}
      trainer.callbacks.wandb.project: {type: string, values: [test]}
      trainer.callbacks.wandb.group: {type: slug}
  hypothesis-smoke:
    script: src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py
    launcher: python
    wandb_callback: true
    allowed_data_kinds: [skill-dag, curriculum]
    positionals: 3
    allowed_positionals:
      0: [dry_run, train_single, train]
      2: [local]
    allowed_options:
      trainer.hard_stop: {type: duration, max_steps: 100}
      trainer.callbacks.wandb.enabled: {type: boolean}
```

Add `entrypoint_profile: str` to `JobRequest`. The Issue form selects one profile; script and
launcher are derived from this protected file and must exactly match. A new arbitrary program
requires a reviewed change to `entrypoints.yaml`, including its positional and option schema.

Create `config/edullm/operators.example.yaml` with fixture identities only:

```yaml
operators:
  - github: alice
    slack_user_id: U11111111
    rotation_order: 0
    enabled: true
  - github: bob
    slack_user_id: U22222222
    rotation_order: 1
    enabled: false
  - github: carol
    slack_user_id: U33333333
    rotation_order: 2
    enabled: false
```

Production onboarding copies this structure to `config/edullm/operators.yaml` with real approved
operators; the example values are never used by workflows.

Commit `config/edullm/operators.yaml` fail-closed:

```yaml
operators: []
```

Assignment refuses to run until at least one reviewed operator entry is added.

- [ ] **Step 7: Run tests**

```bash
pytest -v src/test/edullm/models_test.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/edullm config/edullm src/test/edullm/models_test.py
git commit -m "feat: add eduLLM job request model"
```

---

### Task 2: Issue Parser and Request Validation

**Files:**
- Create: `src/edullm/request_parser.py`
- Create: `src/edullm/validation.py`
- Create: `src/test/edullm/fixtures/valid_issue.md`
- Test: `src/test/edullm/request_parser_test.py`
- Test: `src/test/edullm/validation_test.py`

**Interfaces:**
- Consumes: GitHub Issue-form Markdown.
- Produces: validated `JobRequest` or actionable validation errors.

On successful validation, serialize `JobRequest.canonical_json()` into the bot's
`<!-- edullm-status:v1 -->` comment with its SHA-256 digest and validation timestamp. Editing the
Issue removes `status:ready`; the edited request receives a new digest only after validation.

- [ ] **Step 1: Write parser and validation tests**

```python
# src/test/edullm/request_parser_test.py
from pathlib import Path

from edullm.request_parser import parse_issue


def test_parse_valid_issue():
    body = Path("src/test/edullm/fixtures/valid_issue.md").read_text()
    request = parse_issue(body, issue_number=42, requester="student")
    assert request.study == "skill-dag-v1"
    assert request.launcher == "python"
    assert request.argv == ("train_single", "skilldag-natural", "local", "--seed=0")
    assert request.gpu_count == 1
    assert len(request.digest) == 64
```

```python
# src/test/edullm/validation_test.py
from dataclasses import replace

from edullm.validation import validate_request


def test_rejects_mutable_sha(valid_request, policy):
    errors = validate_request(replace(valid_request, commit_sha="main"), policy)
    assert "commit SHA must be 40 lowercase hexadecimal characters" in errors


def test_rejects_shell_escape(valid_request, policy):
    errors = validate_request(
        replace(valid_request, argv=("--output=ok; curl attacker",)), policy
    )
    assert any("unsafe argument" in error for error in errors)
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -v src/test/edullm/request_parser_test.py src/test/edullm/validation_test.py
```

Expected: FAIL because parser/validator do not exist.

- [ ] **Step 3: Implement deterministic Issue parsing**

```python
# src/edullm/request_parser.py
import json
import re

from edullm.models import JobRequest

HEADING = re.compile(r"^### (?P<name>.+)$", re.MULTILINE)


def fields_from_markdown(body: str) -> dict[str, str]:
    matches = list(HEADING.finditer(body))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        fields[match.group("name").strip()] = body[start:end].strip()
    return fields


def parse_issue(body: str, *, issue_number: int, requester: str) -> JobRequest:
    fields = fields_from_markdown(body)
    return JobRequest(
        issue_number=issue_number,
        requester=requester,
        purpose=fields["Purpose"],
        study=fields["Study"],
        condition=fields["Condition"],
        comparison=fields["Comparison"],
        commit_sha=fields["Commit SHA"],
        entrypoint_profile=fields["Entrypoint profile"],
        script_path=fields["Script path"],
        launcher=fields["Launcher"].lower(),
        argv=tuple(json.loads(fields["Arguments JSON"])),
        data_manifest=fields["Data manifest"],
        data_manifest_sha256=fields["Data manifest SHA-256"],
        data_classification=fields["Data classification"],
        seed=int(fields["Seed"]),
        wandb_project=fields["W&B project"],
        success_signal=fields["Success signal"],
        success_metrics=tuple(
            metric.strip() for metric in fields["Success metrics"].split(",") if metric.strip()
        ),
        gpu_count=int(fields["GPU count"]),
        gpu_preference=fields["GPU preference"].lower(),
        max_runtime_minutes=int(fields["Maximum runtime minutes"]),
    )
```

- [ ] **Step 4: Implement policy and command safety checks**

```python
# src/edullm/validation.py
import re
from pathlib import PurePosixPath

import yaml

from edullm.models import JobRequest
from edullm.policy import Policy

SHA = re.compile(r"^[0-9a-f]{40}$")
UNSAFE = re.compile(r"[;|`]|\$\(|\n|\r")


def validate_request(request: JobRequest, policy: Policy) -> list[str]:
    errors: list[str] = []
    if not SHA.fullmatch(request.commit_sha):
        errors.append("commit SHA must be 40 lowercase hexadecimal characters")
    profile = policy.entrypoints.get(request.entrypoint_profile)
    if profile is None:
        errors.append("entrypoint profile is not allowed")
    elif request.script_path != profile["script"] or request.launcher != profile["launcher"]:
        errors.append("script and launcher do not match the entrypoint profile")
    script = PurePosixPath(request.script_path)
    if script.is_absolute() or ".." in script.parts:
        errors.append("script path must be repository-relative without '..'")
    if request.launcher not in {"python", "torchrun", "bash"}:
        errors.append("launcher must be python, torchrun, or bash")
    for value in request.argv:
        if UNSAFE.search(value):
            errors.append(f"unsafe argument value: {value}")
    if profile is not None:
        positionals = [value for value in request.argv if not value.startswith("--")]
        options = [value[2:].split("=", 1)[0] for value in request.argv if value.startswith("--")]
        if len(positionals) != int(profile["positionals"]):
            errors.append("positional arguments do not match the entrypoint profile")
        allowed = set(profile["allowed_options"])
        unknown = sorted(set(options) - allowed)
        if unknown:
            errors.append(f"options are not allowed for this entrypoint: {unknown}")
        for index, allowed_values in profile.get("allowed_positionals", {}).items():
            if int(index) < len(positionals) and positionals[int(index)] not in allowed_values:
                errors.append(f"positional argument {index} is not allowed")
        for raw in (value for value in request.argv if value.startswith("--") and "=" in value):
            name, value = raw[2:].split("=", 1)
            rule = profile["allowed_options"].get(name)
            if rule is None:
                continue
            if "values" in rule and value not in {str(item) for item in rule["values"]}:
                errors.append(f"value for --{name} is not allowed")
            if rule["type"] == "integer":
                try:
                    integer = int(value)
                except ValueError:
                    errors.append(f"value for --{name} must be an integer")
                else:
                    if integer < rule.get("min", integer) or integer > rule.get("max", integer):
                        errors.append(f"value for --{name} is outside its allowed range")
            if rule["type"] == "path":
                if ".." in PurePosixPath(value).parts or not any(
                    value.startswith(root) for root in rule["roots"]
                ):
                    errors.append(f"path for --{name} is outside allowed roots")
            if rule["type"] == "boolean" and value.lower() not in {"true", "false"}:
                errors.append(f"value for --{name} must be true or false")
            if rule["type"] == "slug" and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value):
                errors.append(f"value for --{name} must be a lowercase slug")
            if rule["type"] == "duration":
                try:
                    duration = yaml.safe_load(value)
                    steps = int(duration["value"])
                    unit = duration["unit"]
                except (TypeError, ValueError, KeyError):
                    errors.append(f"value for --{name} must be a duration mapping")
                else:
                    if unit != "steps" or steps < 1 or steps > int(rule["max_steps"]):
                        errors.append(f"value for --{name} exceeds the allowed smoke duration")
    if not re.fullmatch(r"[0-9a-f]{64}", request.data_manifest_sha256):
        errors.append("data manifest SHA-256 must be 64 lowercase hexadecimal characters")
    if not request.data_manifest.startswith(
        ("builtin://", "/orcd/pool/")
    ):
        errors.append("data manifest location is not allowed")
    if request.gpu_count < 1 or request.gpu_count > policy.max_gpu_count:
        errors.append(f"GPU count must be 1..{policy.max_gpu_count}")
    if request.max_runtime_minutes > policy.max_runtime_minutes:
        errors.append(f"runtime exceeds {policy.max_runtime_minutes} minutes")
    if request.gpu_preference not in policy.allowed_gpu_preferences:
        errors.append("GPU preference is not allowed")
    if request.wandb_project not in policy.allowed_wandb_projects:
        errors.append("W&B project is not allowed")
    if request.data_classification not in {"public", "research-cleared", "restricted"}:
        errors.append("data classification is invalid")
    if request.data_classification == "restricted":
        errors.append("restricted data is not accepted by the public pilot queue")
    if not request.success_metrics:
        errors.append("at least one emitted success metric is required")
    return errors
```

- [ ] **Step 5: Add fixtures and run tests**

Create `valid_issue.md` using the exact headings consumed above, then run:

```bash
pytest -v src/test/edullm/request_parser_test.py src/test/edullm/validation_test.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/edullm src/test/edullm
git commit -m "feat: validate eduLLM Issue requests"
```

---

### Task 3: GitHub Client and Reviewed-Commit Gate

**Files:**
- Create: `src/edullm/github.py`
- Test: `src/test/edullm/github_test.py`

**Interfaces:**
- Consumes: `GITHUB_TOKEN`, repository, SHA.
- Produces: PR review/check evidence and Issue mutations.

- [ ] **Step 1: Write failing review-gate tests**

```python
# src/test/edullm/github_test.py
from edullm.github import GitHubClient


def test_commit_requires_approved_pr():
    sha = "a" * 40
    client = GitHubClient("token", "edu-llm/OLMo-core")
    client.paginated_get = lambda path, key=None: {
        f"/repos/edu-llm/OLMo-core/commits/{sha}/pulls": [{"number": 7}],
        "/repos/edu-llm/OLMo-core/pulls/7/reviews": [
            {
                "state": "APPROVED",
                "commit_id": sha,
                "submitted_at": "2026-07-22T10:00:00Z",
                "user": {"login": "operator"},
            }
        ],
        f"/repos/edu-llm/OLMo-core/commits/{sha}/check-runs": [
            {"name": "Lint", "status": "completed", "conclusion": "success"},
            {"name": "Test", "status": "completed", "conclusion": "success"},
            {"name": "Test scripts", "status": "completed", "conclusion": "success"},
        ],
    }[path]
    client.get = lambda path: {"head": {"sha": sha}, "state": "open", "merged_at": None}
    client.file_exists = lambda path, ref: path.endswith("train.py") and ref == sha
    result = client.reviewed_commit(
        sha,
        script_path="src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py",
        allowed_reviewers={"operator"},
        required_checks={"Lint", "Test", "Test scripts"},
    )
    assert result.approved is True


def test_rejects_approval_for_previous_sha(review_client, sha):
    review_client.reviews = [
        {
            "state": "APPROVED",
            "commit_id": "b" * 40,
            "submitted_at": "2026-07-22T10:00:00Z",
            "user": {"login": "operator"},
        }
    ]
    assert not review_client.result(sha).approved


def test_latest_changes_requested_overrides_approval(review_client, sha):
    review_client.reviews = [
        {
            "state": "APPROVED",
            "commit_id": sha,
            "submitted_at": "2026-07-22T10:00:00Z",
            "user": {"login": "operator"},
        },
        {
            "state": "CHANGES_REQUESTED",
            "commit_id": sha,
            "submitted_at": "2026-07-22T11:00:00Z",
            "user": {"login": "operator"},
        },
    ]
    assert not review_client.result(sha).approved


def test_rejects_missing_required_check(review_client, sha):
    review_client.checks = [{"name": "Lint", "status": "completed", "conclusion": "success"}]
    assert not review_client.result(sha).approved


def test_paginated_get_reads_second_page(github_client):
    github_client.get = lambda path: [{"id": 1}] * 100 if "page=1" in path else [{"id": 2}]
    assert len(github_client.paginated_get("/reviews")) == 101
```

Add these fixtures to `src/test/edullm/conftest.py`:

```python
@pytest.fixture
def sha():
    return "a" * 40


@pytest.fixture
def github_client():
    from edullm.github import GitHubClient

    return GitHubClient("token", "edu-llm/OLMo-core")


@pytest.fixture
def review_client(sha):
    from edullm.github import GitHubClient

    class ReviewClient(GitHubClient):
        def __init__(self):
            super().__init__("token", "edu-llm/OLMo-core")
            self.reviews = []
            self.checks = [
                {"name": name, "status": "completed", "conclusion": "success"}
                for name in ("Lint", "Test", "Test scripts")
            ]

        def paginated_get(self, path, *, key=None):
            if path.endswith("/pulls"):
                return [{"number": 7}]
            if path.endswith("/reviews"):
                return self.reviews
            if path.endswith("/check-runs"):
                return self.checks
            raise AssertionError(path)

        def get(self, path):
            return {"head": {"sha": sha}, "state": "open", "merged_at": None}

        def file_exists(self, path, *, ref):
            return ref == sha

        def result(self, requested_sha):
            return self.reviewed_commit(
                requested_sha,
                script_path="src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py",
                allowed_reviewers={"operator"},
                required_checks={"Lint", "Test", "Test scripts"},
            )

    return ReviewClient()
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -v src/test/edullm/github_test.py
```

Expected: FAIL because `GitHubClient` does not exist.

- [ ] **Step 3: Implement a thin REST client**

```python
# src/edullm/github.py
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class ReviewResult:
    approved: bool
    pr_number: int | None
    reason: str


class GitHubClient:
    def __init__(self, token: str, repo: str, *, base_url: str = "https://api.github.com"):
        self.repo = repo
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def get(self, path: str):
        response = self.session.get(f"{self.base_url}{path}", timeout=20)
        response.raise_for_status()
        return response.json()

    def paginated_get(self, path: str, *, key: str | None = None) -> list[dict]:
        page = 1
        rows: list[dict] = []
        while True:
            separator = "&" if "?" in path else "?"
            payload = self.get(f"{path}{separator}per_page=100&page={page}")
            batch = payload[key] if key is not None else payload
            rows.extend(batch)
            if len(batch) < 100:
                return rows
            page += 1

    def file_exists(self, path: str, *, ref: str) -> bool:
        response = self.session.get(
            f"{self.base_url}/repos/{self.repo}/contents/{path}",
            params={"ref": ref},
            timeout=20,
        )
        return response.status_code == 200

    def reviewed_commit(
        self,
        sha: str,
        *,
        script_path: str,
        allowed_reviewers: set[str],
        required_checks: set[str],
    ) -> ReviewResult:
        pulls = self.paginated_get(f"/repos/{self.repo}/commits/{sha}/pulls")
        for pull in pulls:
            number = int(pull["number"])
            details = self.get(f"/repos/{self.repo}/pulls/{number}")
            if details["head"]["sha"] != sha:
                continue
            if details["state"] != "open" and details.get("merged_at") is None:
                continue
            reviews = self.paginated_get(f"/repos/{self.repo}/pulls/{number}/reviews")
            latest = {}
            for review in sorted(reviews, key=lambda row: row["submitted_at"]):
                latest[review["user"]["login"]] = review
            approved = any(
                login in allowed_reviewers
                and review["state"].upper() == "APPROVED"
                and review["commit_id"] == sha
                for login, review in latest.items()
            )
            blocked = any(
                login in allowed_reviewers
                and review["state"].upper() == "CHANGES_REQUESTED"
                for login, review in latest.items()
            )
            checks = self.paginated_get(
                f"/repos/{self.repo}/commits/{sha}/check-runs",
                key="check_runs",
            )
            successful = {
                check["name"]
                for check in checks
                if check["status"] == "completed" and check["conclusion"] == "success"
            }
            if approved and not blocked and required_checks <= successful and self.file_exists(
                script_path, ref=sha
            ):
                return ReviewResult(True, number, "approved PR with green checks")
        return ReviewResult(False, None, "SHA is not part of an approved green PR")
```

Add focused methods later for labels, assignees, comments, and listing queue Issues; all methods
must accept structured values and JSON-encode through `requests`.

- [ ] **Step 4: Run tests**

```bash
pytest -v src/test/edullm/github_test.py
```

Expected: PASS.

- [ ] **Step 5: Add the reviewed-GitHub fixture**

Append to `src/test/edullm/conftest.py`:

```python
@pytest.fixture
def reviewed_github():
    from edullm.github import ReviewResult

    class ReviewedGitHub:
        @staticmethod
        def reviewed_commit(sha, *, allowed_reviewers):
            return ReviewResult(True, 7, "approved")

    return ReviewedGitHub()
```

- [ ] **Step 6: Commit**

```bash
git add src/edullm/github.py src/test/edullm/github_test.py
git commit -m "feat: enforce reviewed commit execution"
```

---

### Task 4: Issue Form, CODEOWNERS, and Validation Workflow

**Files:**
- Create: `.github/ISSUE_TEMPLATE/config.yml`
- Create: `.github/ISSUE_TEMPLATE/edullm-job-request.yml`
- Create: `.github/CODEOWNERS`
- Create: `config/edullm/main-ruleset.json`
- Create: `.github/workflows/edullm-validate.yml`
- Create: `src/edullm/automation.py`
- Test: `src/test/edullm/validation_test.py`

**Interfaces:**
- Consumes: Issue events.
- Produces: `status:ready` or actionable validation comment.

- [ ] **Step 1: Extend tests for workflow output**

```python
def test_valid_request_is_ready(valid_request, policy, reviewed_github):
    from edullm.automation import validation_decision

    decision = validation_decision(
        valid_request, policy=policy, github=reviewed_github, allowed_reviewers={"operator"}
    )
    assert decision.status == "ready"
    assert decision.errors == ()
```

- [ ] **Step 2: Implement pure validation decision**

```python
# src/edullm/automation.py
from dataclasses import dataclass

from edullm.validation import validate_request


@dataclass(frozen=True)
class ValidationDecision:
    status: str
    errors: tuple[str, ...]


def validation_decision(request, *, policy, github, allowed_reviewers):
    errors = validate_request(request, policy)
    review = github.reviewed_commit(
        request.commit_sha,
        script_path=request.script_path,
        allowed_reviewers=allowed_reviewers,
        required_checks=set(policy.required_checks),
    )
    if not review.approved:
        errors.append(review.reason)
    return ValidationDecision("ready" if not errors else "requested", tuple(errors))
```

Add a test proving every restricted request is rejected before assignment. A future private intake
system requires a separate approved design before restricted data can be enabled.

- [ ] **Step 3: Create the Issue form**

Use dropdowns for launcher, GPU count/preference, W&B project, and data classification
(`public` or `research-cleared` only). Use
textareas for purpose/success criteria and a required JSON-array textarea for ordered arguments.
Derive requester from `github.event.issue.user.login`; do not display or trust a requester field.
Apply labels
`edullm-job` and `status:requested`.

- [ ] **Step 4: Create least-privilege validation workflow**

```yaml
name: Validate eduLLM job request
on:
  issues:
    types: [opened, edited, reopened]
permissions:
  contents: read
  issues: write
  pull-requests: read
  checks: read
jobs:
  validate:
    if: contains(github.event.issue.labels.*.name, 'edullm-job')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
      - uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1
        with:
          python-version: "3.11"
      - run: pip install -e .
      - run: python -m edullm.cli automation validate --issue "${{ github.event.issue.number }}"
        env:
          GITHUB_TOKEN: ${{ github.token }}
```

- [ ] **Step 5: Add CODEOWNERS**

Create the organization team `compute-operators` before enabling queue workflows, then add:

```text
/config/edullm/ @edu-llm/compute-operators
/.github/CODEOWNERS @edu-llm/compute-operators
/.github/workflows/edullm-* @edu-llm/compute-operators
/src/edullm/ @edu-llm/compute-operators
/.cursor/skills/submit-edullm-job/ @edu-llm/compute-operators
```

If the team cannot be created, stop this task and keep all queue workflows disabled; do not replace
the review gate with an unprotected path.

- [ ] **Step 6: Define and apply enforceable main protection**

Create `config/edullm/main-ruleset.json`:

```json
{
  "name": "Protect main and queue controls",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["~DEFAULT_BRANCH"],
      "exclude": []
    }
  },
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": true,
        "required_review_thread_resolution": true,
        "require_code_owner_review": true,
        "require_last_push_approval": true
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          {"context": "Lint"},
          {"context": "Test"},
          {"context": "Test checkpoint"},
          {"context": "Test transformer"},
          {"context": "Test attention"},
          {"context": "Test examples"},
          {"context": "Test scripts"},
          {"context": "Integration tests"},
          {"context": "Test olmo3 ladder"},
          {"context": "Type check"},
          {"context": "Build"},
          {"context": "Style"},
          {"context": "Docs"}
        ]
      }
    }
  ],
  "bypass_actors": []
}
```

An organization/repository administrator applies it:

```bash
gh api repos/edu-llm/OLMo-core/rulesets \
  --method POST \
  --input config/edullm/main-ruleset.json
```

Then verify direct pushes, stale approvals, unresolved review threads, and CODEOWNERS bypass are
rejected on a disposable test PR before enabling queue workflows.

- [ ] **Step 7: Run tests and actionlint**

```bash
pytest -v src/test/edullm/validation_test.py
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 \
  .github/workflows/edullm-validate.yml
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add .github config/edullm/main-ruleset.json \
  src/edullm/automation.py src/test/edullm/validation_test.py
git commit -m "feat: validate GitHub job requests"
```

---

### Task 5: Least-Loaded Assignment and Notifications

**Files:**
- Create: `src/edullm/assignment.py`
- Create: `src/edullm/notifications.py`
- Create: `.github/workflows/edullm-assign.yml`
- Create: `.github/workflows/edullm-reminders.yml`
- Test: `src/test/edullm/assignment_test.py`

**Interfaces:**
- Consumes: enabled operators and active queue Issues.
- Produces: assignment, GitHub mention, Slack ping, reminder/reassignment.

- [ ] **Step 1: Write assignment tests**

```python
# src/test/edullm/assignment_test.py
from edullm.assignment import OperatorLoad, select_operator


def test_selects_fewest_active_gpus():
    loads = [
        OperatorLoad("alice", active_gpus=2, active_jobs=1, rotation=0),
        OperatorLoad("bob", active_gpus=1, active_jobs=2, rotation=1),
        OperatorLoad("carol", active_gpus=1, active_jobs=1, rotation=2),
    ]
    assert select_operator(loads, incoming_gpus=1, max_gpus=2).github == "carol"


def test_excludes_timed_out_operator():
    loads = [
        OperatorLoad("alice", 0, 0, 0),
        OperatorLoad("bob", 1, 1, 1),
    ]
    assert select_operator(loads, incoming_gpus=1, max_gpus=2, exclude={"alice"}).github == "bob"


def test_rejects_operator_without_remaining_capacity():
    loads = [
        OperatorLoad("alice", 2, 1, 0),
        OperatorLoad("bob", 1, 1, 1),
    ]
    assert select_operator(loads, incoming_gpus=1, max_gpus=2).github == "bob"
```

- [ ] **Step 2: Implement scoring**

```python
# src/edullm/assignment.py
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorLoad:
    github: str
    active_gpus: int
    active_jobs: int
    rotation: int


def select_operator(
    loads: list[OperatorLoad],
    *,
    incoming_gpus: int,
    max_gpus: int,
    exclude: set[str] | None = None,
) -> OperatorLoad:
    exclude = exclude or set()
    eligible = [
        load
        for load in loads
        if load.github not in exclude and load.active_gpus + incoming_gpus <= max_gpus
    ]
    if not eligible:
        raise ValueError("no eligible operators")
    return min(eligible, key=lambda load: (load.active_gpus, load.active_jobs, load.rotation))
```

- [ ] **Step 3: Implement sanitized Slack payloads**

```python
# src/edullm/notifications.py
import requests


def slack_assignment(webhook: str, *, issue: int, operator_slack_id: str, title: str) -> None:
    safe_title = title.replace("<", "").replace(">", "")[:120]
    response = requests.post(
        webhook,
        json={"text": f"eduLLM job #{issue} assigned to <@{operator_slack_id}>: {safe_title}"},
        timeout=10,
    )
    response.raise_for_status()
```

- [ ] **Step 4: Create assign/reminder workflows**

Use explicit `issues: write`, a per-Issue concurrency group, and `SLACK_WEBHOOK_URL` only in steps
that send notifications. Reminder cron runs every five minutes, records reminder/reassignment
timestamps in the machine status comment, and is idempotent.

Assignment itself uses one repository-wide concurrency group:

```yaml
concurrency:
  group: edullm-assignment
  cancel-in-progress: false
```

Within that serialized job, refetch every active Issue immediately before computing capacity and
writing the assignment. Pass the incoming request's GPU count to `select_operator()`. If no
operator has remaining capacity, leave the Issue `status:ready` and retry on the next queue event.

- [ ] **Step 5: Run tests**

```bash
pytest -v src/test/edullm/assignment_test.py
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 \
  .github/workflows/edullm-assign.yml .github/workflows/edullm-reminders.yml
```

- [ ] **Step 6: Commit**

```bash
git add src/edullm .github/workflows src/test/edullm
git commit -m "feat: assign and notify compute operators"
```

---

### Task 6: Operator Setup and SSH ControlMaster

**Files:**
- Create: `src/edullm/cli.py`
- Create: `src/edullm/ssh.py`
- Test: `src/test/edullm/ssh_test.py`
- Test: `src/test/edullm/cli_test.py`

**Interfaces:**
- Consumes: operator-local config and SSH key.
- Produces: `edullm setup`, `jobs`, `run`, `logs`, `stop`, `logout`.

- [ ] **Step 1: Write SSH config tests**

```python
# src/test/edullm/ssh_test.py
from edullm.ssh import control_block


def test_control_block_uses_one_hour_persist():
    block = control_block("philote")
    assert "Host orcd-login" in block
    assert "Hostname orcd-login.mit.edu" in block
    assert "ControlMaster auto" in block
    assert "ControlPersist 1h" in block
    assert "User philote" in block
```

- [ ] **Step 2: Implement safe SSH helpers**

```python
# src/edullm/ssh.py
import shlex
import subprocess


def control_block(username: str) -> str:
    return "\n".join(
        [
            "Host orcd-login",
            "    Hostname orcd-login.mit.edu",
            "    ControlMaster auto",
            "    ControlPath ~/.ssh/edullm-%C",
            "    ControlPersist 1h",
            f"    User {username}",
        ]
    )


def run_remote(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "orcd-login", shlex.join(argv)],
        check=True,
        text=True,
        capture_output=True,
    )


def write_remote(path: str, content: str) -> None:
    command = f"umask 077 && cat > {shlex.quote(path)}"
    subprocess.run(
        ["ssh", "orcd-login", command],
        input=content,
        check=True,
        text=True,
        capture_output=True,
    )


def close_master() -> None:
    subprocess.run(["ssh", "-O", "exit", "orcd-login"], check=False)
```

`setup` must parse existing `~/.ssh/config`, display the exact addition/change, create a backup, and
ask for confirmation before writing. It must use `os.open(..., 0o600)` for operator configuration.
Add mocked-subprocess tests proving an argv element containing spaces remains one remote argument,
shell metacharacters are quoted, and `write_remote()` sends content through stdin without placing it
in the process argv.

- [ ] **Step 3: Implement every setup preflight**

`edullm setup` runs these checks in order and stops on the first failure:

```text
gh auth status
gh api user --jq .login
python -c "import wandb; print(wandb.Api().viewer.username)"
ssh orcd-login "hostname; command -v sbatch; command -v squeue"
ssh orcd-login "mkdir -p $HOME/orcd/scratch/edullm && test -w $HOME/orcd/scratch/edullm"
```

It then:

1. Writes `~/.config/edullm/config.yaml` with mode `0600`.
2. Shows and confirms the SSH ControlMaster diff before applying it.
3. Submits `src/scripts/orcd/setup_env.sbatch` through the remote checkout if the environment is
   missing.
4. Waits for the setup Slurm job to finish and fails if `sacct` is not `COMPLETED`.
5. Records this environment fingerprint in operator config:

```bash
source "$HOME/venvs/edullm/bin/activate"
python -m pip freeze --all | LC_ALL=C sort | sha256sum
```

6. Verifies `python -c "import torch, wandb, olmo_core"`.
7. Prompts for the operator's W&B key with `getpass.getpass()` only if a verified remote key is
   absent. Send it through `ssh.write_remote()` stdin to
   `~/.config/edullm/wandb.key`, apply mode `0600`, and write a secret-free remote `wandb.env` that
   reads that key and exports `WANDB_ENTITY=eduLLM`.
8. Verifies remote authentication without printing the key:

```bash
source "$HOME/venvs/edullm/bin/activate"
source "$HOME/.config/edullm/wandb.env"
python -c 'import wandb; print(wandb.Api().viewer.username)'
```

9. Verifies the configured GitHub username exists in `config/edullm/operators.yaml`.

Add mocked tests for every failure and success transition. Assert the literal W&B key appears only
in subprocess stdin and never in argv, output, exception text, or operator config. Repeat this exact
command on operator two and operator three only after the Apptainer rollout task; onboarding is
complete only when image fingerprints match.

- [ ] **Step 4: Implement the argparse command surface**

```python
# src/edullm/cli.py
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edullm")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup")
    jobs = sub.add_parser("jobs")
    jobs.add_argument("--mine", action="store_true")
    sub.add_parser("run")
    logs = sub.add_parser("logs")
    logs.add_argument("issue", type=int)
    stop = sub.add_parser("stop")
    stop.add_argument("issue", type=int)
    sub.add_parser("logout")
    automation = sub.add_parser("automation")
    automation.add_argument("action", choices=["validate", "assign", "remind", "reconcile"])
    automation.add_argument("--issue", type=int)
    return parser
```

Dispatch each subcommand to a focused function; do not put SSH or GitHub logic in `main()`.

- [ ] **Step 5: Test command help**

```python
# src/test/edullm/cli_test.py
from edullm.cli import build_parser


def test_commands_are_plain_language():
    parser = build_parser()
    text = parser.format_help()
    for command in ("setup", "jobs", "run", "logs", "stop", "logout"):
        assert command in text
    assert "claim" not in text
    assert "sync" not in text
```

- [ ] **Step 6: Run tests**

```bash
pytest -v src/test/edullm/ssh_test.py src/test/edullm/cli_test.py
```

- [ ] **Step 7: Commit**

```bash
git add src/edullm src/test/edullm
git commit -m "feat: add edullm operator command"
```

---

### Task 7: Slurm Rendering, Submission, Logs, and Stop

**Files:**
- Create: `src/edullm/slurm.py`
- Create: `src/edullm/jobs.py`
- Create: `src/edullm/data_manifest.py`
- Test: `src/test/edullm/slurm_test.py`
- Test: `src/test/edullm/jobs_test.py`
- Test: `src/test/edullm/data_manifest_test.py`

**Interfaces:**
- Consumes: validated `ResolvedRequest`.
- Produces: safe `sbatch`, Slurm ID, status, logs, cancellation.

- [ ] **Step 1: Write exact render tests**

```python
# src/test/edullm/slurm_test.py
from edullm.slurm import render_sbatch


def test_render_uses_structured_argv(valid_resolved_request):
    text = render_sbatch(valid_resolved_request)
    assert "#SBATCH -p mit_normal_gpu" in text
    assert "#SBATCH -G l40s:1" in text
    assert "#SBATCH -t 00:30:00" in text
    assert "#SBATCH --export=NONE" in text
    assert "#SBATCH -o logs/issue-42-attempt-1-%j.log" in text
    assert "git checkout --detach " + "a" * 40 in text
    assert "git clone --no-checkout" in text
    assert 'mkdir -p "$(dirname "$WORKTREE")"' in text
    assert 'export PYTHONPATH="$WORKTREE/src"' in text
    assert "eval " not in text
    assert "WANDB_API_KEY" not in text
```

- [ ] **Step 2: Implement shell-safe quoting and rendering**

First implement data-manifest verification:

```python
# src/edullm/data_manifest.py
import hashlib
import json
import shlex
from pathlib import Path

BUILTINS = {
    "builtin://generic-smoke-v1": {
        "kind": "generic-smoke",
        "generate_tiny_data": True,
    }
}
ALLOWED_ROOTS = (Path("/orcd/pool"),)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(uri: str, expected_sha256: str, allowed_kinds: set[str]) -> dict:
    if uri.startswith("builtin://"):
        if uri not in BUILTINS:
            raise ValueError("unknown built-in dataset")
        data = BUILTINS[uri]
        if data["kind"] not in allowed_kinds:
            raise ValueError("dataset kind is not allowed for this entrypoint")
        return data
    manifest = Path(uri).resolve(strict=True)
    if not any(manifest.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError("manifest is outside approved roots")
    if sha256_file(manifest) != expected_sha256:
        raise ValueError("manifest digest mismatch")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["kind"] not in allowed_kinds:
        raise ValueError("dataset kind is not allowed for this entrypoint")
    verified_paths = set()
    for row in data["files"]:
        path = Path(row["path"]).resolve(strict=True)
        if not any(path.is_relative_to(root) for root in ALLOWED_ROOTS):
            raise ValueError("shard is outside approved roots")
        if path.stat().st_size != int(row["size"]):
            raise ValueError(f"shard size mismatch: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"shard digest mismatch: {path}")
        verified_paths.add(path)
    required_path_keys = {
        "skill-dag": ("mix_file",),
        "curriculum": ("order_file",),
        "generic-smoke": (),
    }[data["kind"]]
    for key in required_path_keys:
        if Path(data[key]).resolve(strict=True) not in verified_paths:
            raise ValueError(f"{key} is not bound to a verified file")
    if "data_dir" in data:
        data_dir = Path(data["data_dir"]).resolve(strict=True)
        if not any(data_dir.is_relative_to(root) for root in ALLOWED_ROOTS):
            raise ValueError("data_dir is outside approved roots")
    return data


def runtime_environment(data: dict) -> dict[str, str]:
    if data["kind"] == "generic-smoke":
        root = str(Path.home() / "orcd/scratch/edullm/data/generic-smoke")
        return {
            "EDULLM_DATA_MODE": "synthetic" if data.get("generate_tiny_data") else "staged",
            "OLMO_DATA_ROOT": data.get("data_root", root),
        }
    if data["kind"] == "skill-dag":
        return {
            "SMOKE_MODE": "skill_dag",
            "SMOKE_DATA_DIR": data["data_dir"],
            "SMOKE_MIX_FILE": data["mix_file"],
        }
    if data["kind"] == "curriculum":
        return {
            "SMOKE_MODE": "curriculum",
            "SMOKE_DATA_DIR": data["data_dir"],
            "SMOKE_ORDER_FILE": data["order_file"],
        }
    raise ValueError("unsupported dataset kind")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["verify", "render-env"])
    parser.add_argument("uri")
    parser.add_argument("sha256")
    parser.add_argument("--allowed-kind", action="append", required=True)
    args = parser.parse_args()
    data = verify_manifest(args.uri, args.sha256, set(args.allowed_kind))
    if args.command == "render-env":
        for key, value in runtime_environment(data).items():
            print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
```

Tests must cover an unknown built-in, `..` traversal, a symlink escaping `/orcd/pool`, wrong shard
size, wrong shard hash, and a valid multi-shard manifest. Invoke this verifier from `edullm run`
over SSH immediately before `sbatch`, and again inside the Slurm script before training.

```python
# src/edullm/slurm.py
import base64
import json
import shlex


def duration(minutes: int) -> str:
    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}:{mins:02d}:00"


def gpu_flag(preference: str, count: int) -> str:
    return f"{preference}:{count}" if preference != "any" else str(count)


def render_sbatch(resolved) -> str:
    request = resolved.request
    request_b64 = base64.b64encode(request.canonical_json().encode()).decode()
    data_kind_args = " ".join(
        f"--allowed-kind {shlex.quote(kind)}" for kind in resolved.allowed_data_kinds
    )
    argv = [request.script_path, *request.argv]
    audit_tags = [
        f"issue-{request.issue_number}",
        resolved.wandb_run_prefix,
        request.entrypoint_profile,
        request.condition,
        request.gpu_preference,
        "engaging",
        request.commit_sha[:12],
        f"seed-{request.seed}",
        f"data-{request.data_manifest_sha256[:12]}",
    ]
    argv.extend(
        [
            "--trainer.callbacks.wandb.enabled=true",
            f"--trainer.callbacks.wandb.entity={resolved.wandb_entity}",
            f"--trainer.callbacks.wandb.project={request.wandb_project}",
            f"--trainer.callbacks.wandb.group={request.study}",
            f"--trainer.callbacks.wandb.tags={json.dumps(audit_tags, separators=(',', ':'))}",
            f"--trainer.callbacks.wandb.notes=request_digest:{request.digest}",
        ]
    )
    if request.launcher == "python":
        argv = ["python", *argv]
    elif request.launcher == "torchrun":
        argv = [
            "torchrun",
            "--standalone",
            f"--nproc-per-node={request.gpu_count}",
            *argv,
        ]
    elif request.launcher == "bash":
        argv = ["bash", *argv]
    command = " ".join(shlex.quote(part) for part in argv)
    return f"""#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -G {gpu_flag(request.gpu_preference, request.gpu_count)}
#SBATCH -t {duration(request.max_runtime_minutes)}
#SBATCH -c {max(4, request.gpu_count * 4)}
#SBATCH --mem=64G
#SBATCH -J {shlex.quote(resolved.slurm_job_name)}
#SBATCH --export=NONE
#SBATCH -o {resolved.log_pattern}
#SBATCH -e {resolved.log_pattern}
set -euo pipefail
source "$HOME/venvs/edullm/bin/activate"
source "$HOME/.config/edullm/wandb.env"
WORKTREE="$HOME/orcd/scratch/edullm/work/{request.request_name}/${{SLURM_JOB_ID}}"
mkdir -p "$(dirname "$WORKTREE")"
chmod 700 "$(dirname "$WORKTREE")"
git clone --no-checkout https://github.com/edu-llm/OLMo-core.git "$WORKTREE"
cd "$WORKTREE"
git fetch origin {request.commit_sha}
git checkout --detach {request.commit_sha}
test "$(git rev-parse HEAD)" = {request.commit_sha}
test -z "$(git status --porcelain)"
export PYTHONPATH="$WORKTREE/src"
python -c 'import olmo_core, os; assert os.path.realpath(olmo_core.__file__).startswith(os.path.realpath(os.environ["PYTHONPATH"]))'
export WANDB_ENTITY={shlex.quote(resolved.wandb_entity)}
export WANDB_PROJECT={shlex.quote(request.wandb_project)}
export WANDB_GROUP={shlex.quote(request.study)}
export WANDB_RUN_PREFIX={shlex.quote(resolved.wandb_run_prefix)}
export WANDB_RUN_ID="${{WANDB_RUN_PREFIX}}-${{SLURM_JOB_ID}}"
export EDULLM_REQUEST_ID={request.issue_number}
export EDULLM_COMMIT_SHA={request.commit_sha}
export EDULLM_SEED={request.seed}
export EDULLM_DATA_MANIFEST={shlex.quote(request.data_manifest)}
export EDULLM_DATA_MANIFEST_SHA256={request.data_manifest_sha256}
export EDULLM_REQUEST_PATH="$WORKTREE/edullm_request.json"
printf %s {shlex.quote(request_b64)} | base64 --decode > "$EDULLM_REQUEST_PATH"
DATA_ENV="$WORKTREE/edullm_data.env"
python -m edullm.data_manifest render-env \
  "$EDULLM_DATA_MANIFEST" "$EDULLM_DATA_MANIFEST_SHA256" \
  {data_kind_args} > "$DATA_ENV"
source "$DATA_ENV"
if [[ "${{EDULLM_DATA_MODE:-}}" == "synthetic" ]]; then
  python src/scripts/orcd/create_tiny_data.py --output "$OLMO_DATA_ROOT"
fi
{command}
"""
```

The final W&B ID does not exist until Slurm assigns a job ID. After `sbatch --parsable` returns,
the CLI combines `wandb_run_prefix` and the returned ID, then posts the exact URL to the Issue.

Direct `s3://` training manifests are rejected by the initial pool. Plan 1 must stage and verify
S3 data into an approved `/orcd/pool/` location before a request references it.

Every initial entrypoint profile must declare `wandb_callback: true`. The renderer adds Issue and
attempt IDs, condition, GPU/cluster, commit, seed, request digest, and data-manifest digest through
the existing OLMo `WandBCallback`; `ConfigSaverCallback` supplies the resolved training config.
For a future non-OLMo profile, the `/weights-and-biases` instrumentation must read
`EDULLM_REQUEST_PATH` and log equivalent metadata before that profile is approved.

- [ ] **Step 3: Implement submission/status operations**

Use `ssh.run_remote()` with `shlex.join()` and `ssh.write_remote()` for stdin upload. Upload the
rendered file to a restrictive remote path. Submit from the fixed log root:

```bash
mkdir -p "$HOME/orcd/scratch/edullm/logs"
cd "$HOME/orcd/scratch/edullm"
sbatch --export=NONE --parsable /private/path/request.sbatch
```

Parse only numeric job IDs, expand `%j` into the returned ID for
`AttemptRecord.log_path`, and use `squeue --json` when supported and stable
`sacct --parsable2` fields otherwise.

Expose:

```python
def submit(resolved: ResolvedRequest) -> str: ...
def jobs(*, mine: bool) -> list[SlurmJob]: ...
def logs(issue: int) -> str: ...
def stop(issue: int) -> None: ...
```

- [ ] **Step 4: Connect CLI orchestration**

`edullm run` must:

1. Read assigned Issues.
2. Select the oldest validated request and parse canonical JSON from the bot status comment.
3. Parse the current Issue and require its digest to equal the validated digest.
4. Revalidate SHA, review state, data manifest, and policy.
5. Refetch the status comment immediately before submission and require the digest and validation
   timestamp to be unchanged.
6. Render and submit.
7. Append an `AttemptRecord` with `attempt-$ATTEMPT_NUMBER`, request digest, operator, Slurm ID,
   and W&B ID.
8. Post Slurm ID and deterministic W&B URL.
9. Transition Issue to `submitted`.

Add tests that edit the Issue after validation and that alter canonical JSON after assignment; both
must abort before `sbatch`.

- [ ] **Step 5: Run tests**

```bash
pytest -v src/test/edullm/slurm_test.py src/test/edullm/jobs_test.py
```

- [ ] **Step 6: Commit**

```bash
git add src/edullm src/test/edullm
git commit -m "feat: submit eduLLM jobs to Engaging"
```

---

### Task 8: W&B Identity and Status Reconciliation

**Files:**
- Create: `src/edullm/wandb_status.py`
- Create: `.github/workflows/edullm-wandb-reconcile.yml`
- Test: `src/test/edullm/wandb_status_test.py`

**Interfaces:**
- Consumes: W&B run ID and state.
- Produces: GitHub running/completed/failed state.

- [ ] **Step 1: Write state-mapping tests**

```python
# src/test/edullm/wandb_status_test.py
import pytest

from edullm.wandb_status import issue_status


@pytest.mark.parametrize(
    ("wandb_state", "expected"),
    [
        ("running", "running"),
        ("finished", "completed"),
        ("failed", "failed"),
        ("crashed", "failed"),
        ("killed", "failed"),
        ("preempted", "preempted"),
    ],
)
def test_state_mapping(wandb_state, expected):
    assert issue_status(wandb_state, current_status="running") == expected


def test_unrequested_kill_is_failure():
    assert issue_status("killed", current_status="running") == "failed"


def test_terminal_status_never_regresses():
    assert issue_status("running", current_status="completed") == "completed"
    assert issue_status("pending", current_status="failed") == "failed"
```

- [ ] **Step 2: Implement W&B state lookup**

```python
# src/edullm/wandb_status.py
TERMINAL = {"completed", "failed", "cancelled"}
ALLOWED = {
    "submitted": {"running", "completed", "failed", "preempted"},
    "running": {"completed", "failed", "preempted"},
    "preempted": set(),
}


def issue_status(wandb_state: str, *, current_status: str) -> str:
    if current_status in TERMINAL:
        return current_status
    state = wandb_state.lower()
    if state == "killed":
        candidate = "cancelled" if current_status == "cancelled" else "failed"
    else:
        candidate = {
            "running": "running",
            "finished": "completed",
            "failed": "failed",
            "crashed": "failed",
            "preempting": "preempted",
            "preempted": "preempted",
        }.get(state, current_status)
    return candidate if candidate in ALLOWED.get(current_status, set()) else current_status


def lookup(api, *, entity: str, project: str, run_id: str):
    return api.run(f"{entity}/{project}/{run_id}")
```

A preempted Issue remains `preempted`. It returns to `ready` only after the requester or operator
adds `retry-approved`; the next submission creates a new `AttemptRecord`. Never reuse the previous
Slurm/W&B attempt identity.

- [ ] **Step 3: Create optional scheduled reconciliation**

The workflow runs every five minutes only when `EDULLM_WANDB_MONITOR_KEY` exists. Use a dedicated
monitoring identity, not an operator key. Without it, `edullm jobs` remains the supported repair
path.

- [ ] **Step 4: Run tests and actionlint**

```bash
pytest -v src/test/edullm/wandb_status_test.py
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 \
  .github/workflows/edullm-wandb-reconcile.yml
```

- [ ] **Step 5: Commit**

```bash
git add src/edullm/wandb_status.py .github/workflows/edullm-wandb-reconcile.yml \
  src/test/edullm/wandb_status_test.py
git commit -m "feat: reconcile W&B job status"
```

---

### Task 9: Shared Agent Skill

**Files:**
- Create: `.cursor/skills/submit-edullm-job/SKILL.md`
- Create: `.cursor/skills/submit-edullm-job/request-reference.md`
- Create: `.cursor/skills/submit-edullm-job/scripts/validate_request.py`
- Test: `src/test/edullm/skill_test.py`

**Interfaces:**
- Consumes: researcher intent and current Git repository.
- Produces: validated GitHub Issue; no credentials or Slurm access.

- [ ] **Step 1: Write skill structure test**

```python
# src/test/edullm/skill_test.py
from pathlib import Path

import yaml


def test_submission_skill_metadata_and_size():
    path = Path(".cursor/skills/submit-edullm-job/SKILL.md")
    text = path.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter["name"] == "submit-edullm-job"
    assert "Engaging" in frontmatter["description"]
    assert len(text.splitlines()) < 500
```

- [ ] **Step 2: Write concise skill workflow**

Use frontmatter:

```yaml
---
name: submit-edullm-job
description: Creates validated eduLLM GitHub job requests for MIT Engaging. Use when a researcher asks to submit, queue, schedule, or run an OLMo GPU experiment.
---
```

Required behavior:

1. Inspect current branch, full SHA, PR, and working-tree cleanliness.
2. Read the selected script and W&B callback.
3. List metrics that code emits; never invent metrics.
4. If a metric is missing, direct the user to `/weights-and-biases`, then require commit/review.
5. Ask one missing scientific question at a time.
6. Apply safe compute defaults only for engineering smokes.
7. Render plain-language preview.
8. Run the shared validator.
9. Create the Issue after explicit confirmation.
10. Never request or handle credentials.

- [ ] **Step 3: Add shared validator wrapper**

```python
# .cursor/skills/submit-edullm-job/scripts/validate_request.py
import argparse
import json
from pathlib import Path

from edullm.models import JobRequest
from edullm.policy import load_policy
from edullm.validation import validate_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/edullm/policy.yaml"),
    )
    args = parser.parse_args()
    request = JobRequest(**json.loads(args.input_json.read_text(encoding="utf-8")))
    errors = validate_request(request, load_policy(args.policy))
    if errors:
        raise SystemExit("\n".join(errors))
    print(request.canonical_json())


if __name__ == "__main__":
    main()
```

Before creating the Issue, the Agent Skill writes the previewed request to a mode-`0600` temporary
JSON file, runs this script with `--input-json`, displays any errors, and deletes the file. The
GitHub workflow parses the resulting Issue through the same `JobRequest` and `validate_request()`
functions.

- [ ] **Step 4: Run tests**

```bash
pytest -v src/test/edullm/skill_test.py
```

- [ ] **Step 5: Commit**

```bash
git add .cursor/skills/submit-edullm-job src/test/edullm/skill_test.py src/edullm/cli.py
git commit -m "feat: add eduLLM job submission skill"
```

---

### Task 10: CI and Pilot Acceptance

**Files:**
- Modify: `.github/workflows/main.yml`
- Modify: `CHANGELOG.md`
- Create: `docs/source/guides/edullm_engaging.rst`

**Interfaces:**
- Produces: tested package, documented operator/researcher flow, one completed queue pilot.

- [ ] **Step 1: Add explicit CI task**

Add to the existing matrix:

```yaml
- name: Test edullm queue
  run: pytest -v --color=yes --durations=3 src/test/edullm/
```

- [ ] **Step 2: Run repository checks**

```bash
pytest -v src/test/edullm/
pytest -v src/test/scripts/orcd/
make lint-check
make style-check
make type-check
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 \
  .github/workflows/edullm-*.yml
git diff --check
```

Expected: PASS.

- [ ] **Step 3: Configure pilot repository settings**

Create labels, add the real initial operator GitHub/Slack mappings through a protected config,
configure `SLACK_WEBHOOK_URL`, and leave only the pilot operator enabled. Do not add a W&B
monitoring key unless a dedicated identity exists.

- [ ] **Step 4: Exercise assignment timeout**

Temporarily use one-minute reminder/two-minute reassignment values in a staging policy fixture,
not production policy. Verify one reminder and exactly one reassignment, then restore 15/30.

- [ ] **Step 5: Submit the generic ORCD smoke through the queue**

Acceptance:

- Valid Issue created through Agent Skill.
- Approved SHA gate passes.
- Least-loaded operator receives GitHub + Slack ping.
- `edullm run` prompts for Duo once and submits.
- Issue records Slurm ID and W&B URL.
- W&B run under `eduLLM/test` completes.
- W&B history contains a finite `train/CE loss` value and the checked value is recorded in the
  Issue's completion comment.
- Issue reaches `completed`.
- No secret appears in logs/config/comments.

- [ ] **Step 6: Onboard other operators only after policy confirmation**

Keep the other two operators disabled in this plan. Confirm with ORCD that the shared project
workflow is acceptable, but perform their activation only after Plan 3 completes one Skill-DAG and
one curriculum pair through the single-operator queue.

- [ ] **Step 7: Commit acceptance documentation**

```bash
git add .github/workflows/main.yml CHANGELOG.md docs/source/guides/edullm_engaging.rst
git commit -m "docs: document eduLLM Engaging job pool"
```
