"""
Immutable records shared by eduLLM queue workflows.
"""

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum


def slug(value: str) -> str:
    """
    Convert a value into a lowercase, bounded request-name component.

    :param value: The source value.

    :returns: A slug containing at most 48 characters.
    """
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:48]


class JobStatus(str, Enum):
    """Lifecycle states for an eduLLM job request."""

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
    """A researcher's immutable, structured request for compute."""

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
        """Return the stable request name derived from Issue and study identity."""
        return f"issue-{self.issue_number}-{slug(self.study)}-{slug(self.condition)}"

    def canonical_json(self) -> str:
        """Serialize the request deterministically for audit and digest use."""
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of :meth:`canonical_json`."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True)
class Operator:
    """A reviewed compute operator eligible for queue assignment."""

    github: str
    slack_user_id: str
    rotation_order: int
    enabled: bool = True
    apptainer_path: str | None = None
    apptainer_sha256: str | None = None


@dataclass(frozen=True)
class ResolvedRequest:
    """A validated request enriched with trusted execution metadata."""

    request: JobRequest
    operator: str
    wandb_entity: str
    wandb_run_prefix: str
    slurm_job_name: str
    log_pattern: str
    allowed_data_kinds: tuple[str, ...]
    slurm_job_id: str | None = None
    model_identity: str = "olmo2-190m"
    repository_url: str = "https://github.com/edu-llm/OLMo-core.git"
    slurm_partition: str = "mit_normal_gpu"
    slurm_memory: str = "64G"
    slurm_cpus_per_gpu: int = 4
    scratch_root: str = "$HOME/orcd/scratch/edullm"
    fixed_launcher_arguments: tuple[str, ...] = ()
    fixed_arguments: tuple[str, ...] = ()
    fixed_option_names: tuple[str, ...] = ()
    derived_path_options: tuple[str, ...] = ()
    fixed_wandb_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttemptRecord:
    """Immutable identities and paths recorded for one submission attempt."""

    attempt_id: str
    request_digest: str
    operator: str
    slurm_job_id: str
    wandb_run_id: str
    log_path: str
