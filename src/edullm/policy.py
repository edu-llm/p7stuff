"""
Load trusted eduLLM queue policy and operator configuration.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from edullm.github import GitHubValidationError, normalize_actor_login
from edullm.models import Operator

_WANDB_ENTITY = "eduLLM"
_ALLOWED_WANDB_PROJECTS = (
    "test",
    "pretraining",
    "posttraining",
    "evaluation",
    "data-pipeline",
)
_ALLOWED_GPU_PREFERENCES = ("any", "l40s", "h100", "h200")
_REQUIRED_CHECKS = (
    "Lint",
    "Test transformer",
    "Test attention",
    "Test examples",
    "Test scripts",
    "Test eduLLM core",
    "Integration tests",
    "Type check",
    "Build",
    "Style",
    "Docs",
)
_POLICY_REQUIRED_FIELDS = {
    "wandb_entity",
    "allowed_wandb_projects",
    "required_checks",
    "repository_url",
    "scratch_root",
    "slurm_cpus_per_gpu",
    "slurm_memory",
    "slurm_partition",
}
_POLICY_OPTIONAL_FIELDS = {
    "max_runtime_minutes",
    "max_gpu_count",
    "allowed_gpu_preferences",
    "reminder_after_minutes",
    "reassign_after_minutes",
}
_OPERATOR_REQUIRED_FIELDS = {
    "github",
    "slack_user_id",
    "rotation_order",
    "enabled",
}
_OPERATOR_OPTIONAL_FIELDS = {"apptainer_path", "apptainer_sha256"}
_SLACK_USER_ID = re.compile(r"[UW][A-Z0-9]{8,20}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_APPROVED_ENTRYPOINTS = {
    "generic-smoke": {
        "script": "src/examples/llm/train.py",
        "launcher": "torchrun",
        "wandb_callback": True,
        "model_identity": "olmo2-190m",
        "allowed_data_kinds": ("generic-smoke",),
        "fixed_launcher_arguments": ("--standalone", "--nproc-per-node=1"),
        "fixed_options": {
            "model-factory": "olmo2_190M",
            "sequence-length": 512,
            "save-folder": {
                "type": "derived_path",
                "root_env": "EDULLM_SCRATCH",
                "relative": "runs/{run_name}",
            },
            "work-dir": {
                "type": "derived_path",
                "root_env": "EDULLM_SCRATCH",
                "relative": "runs/{run_name}",
            },
            "data_loader.global_batch_size": 8192,
            "train_module.rank_microbatch_size": 2048,
            "train_module.max_sequence_length": 512,
            "trainer.hard_stop": {"value": 20, "unit": "steps"},
            "trainer.callbacks.lm_evaluator.enabled": False,
            "trainer.callbacks.downstream_evaluator.enabled": False,
            "trainer.callbacks.checkpointer.save_interval": 10,
            "trainer.callbacks.checkpointer.ephemeral_save_interval": None,
            "trainer.callbacks.wandb.enabled": True,
            "trainer.callbacks.wandb.entity": "eduLLM",
            "trainer.callbacks.wandb.project": "test",
            "trainer.callbacks.wandb.group": {"type": "request_field", "field": "study"},
            "trainer.callbacks.wandb.tags": ("orcd", "generic-smoke", "olmo2-190m"),
        },
        "positionals": 1,
        "allowed_positionals": {0: {"type": "slug"}},
        "allowed_options": {},
    },
    "hypothesis-smoke": {
        "script": "src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py",
        "launcher": "python",
        "wandb_callback": True,
        "model_identity": "olmo2-190m",
        "allowed_data_kinds": ("skill-dag", "curriculum"),
        "positionals": 3,
        "allowed_positionals": {
            0: ("dry_run", "train_single", "train"),
            2: ("local",),
        },
        "fixed_options": {"trainer.callbacks.wandb.enabled": True},
        "allowed_options": {
            "seed": {
                "type": "integer",
                "min": 0,
                "max": 2147483647,
                "required": True,
                "request_field": "seed",
            },
            "trainer.hard_stop": {
                "type": "duration",
                "max_steps": 100,
            },
        },
    },
    "worked-examples-cpt": {
        "script": "src/scripts/hypothesis/worked_examples/train_cpt_arm.py",
        "launcher": "torchrun",
        "wandb_callback": True,
        "model_identity": "olmo2-760m",
        "allowed_data_kinds": ("worked-examples",),
        "fixed_launcher_arguments": ("--standalone", "--nproc-per-node=2"),
        "fixed_options": {
            "pack-dir": "/orcd/pool/edullm/data/worked-examples-metamath-v0",
            "load-path": "/orcd/pool/edullm/checkpoints/OLMo-Ladder-760M-0.5xC-core",
            "token-budget": 200_000_000,
            "seed": 0,
            "save-folder": {
                "type": "derived_path",
                "root_env": "EDULLM_SCRATCH",
                "relative": "runs/{run_name}",
            },
            "work-dir": {
                "type": "derived_path",
                "root_env": "EDULLM_SCRATCH",
                "relative": "runs/{run_name}",
            },
            "wandb-project": "pretraining",
            "wandb-group": {"type": "request_field", "field": "study"},
            "trainer.callbacks.wandb.enabled": True,
            "trainer.callbacks.wandb.entity": "eduLLM",
            "trainer.callbacks.wandb.project": "pretraining",
            "trainer.callbacks.wandb.group": {"type": "request_field", "field": "study"},
            "trainer.callbacks.wandb.tags": (
                "orcd",
                "worked-examples-cpt",
                "olmo2-760m",
            ),
        },
        "positionals": 1,
        "allowed_positionals": {0: {"type": "slug"}},
        "allowed_options": {
            "arm": {
                "type": "choice",
                "values": ("bare", "complete", "fade_ordered", "fade_shuffled"),
                "required": True,
                "request_field": "condition",
            },
        },
        "gpu_count": 2,
        "gpu_preference": "h100",
        "max_runtime_minutes": 360,
    },
}


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class Policy:
    """Trusted limits, allowlists, and entrypoint profiles for queue requests."""

    wandb_entity: str
    allowed_wandb_projects: tuple[str, ...]
    max_runtime_minutes: int = 360
    max_gpu_count: int = 2
    allowed_gpu_preferences: tuple[str, ...] = _ALLOWED_GPU_PREFERENCES
    required_checks: tuple[str, ...] = _REQUIRED_CHECKS
    entrypoints: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    reminder_after_minutes: int = 15
    reassign_after_minutes: int = 30
    repository_url: str = "https://github.com/edu-llm/OLMo-core.git"
    scratch_root: str = "$HOME/orcd/scratch/edullm"
    slurm_partition: str = "mit_normal_gpu"
    slurm_memory: str = "64G"
    slurm_cpus_per_gpu: int = 4

    def __post_init__(self) -> None:
        """Detach and recursively freeze every policy container."""
        object.__setattr__(self, "allowed_wandb_projects", tuple(self.allowed_wandb_projects))
        object.__setattr__(self, "allowed_gpu_preferences", tuple(self.allowed_gpu_preferences))
        object.__setattr__(self, "required_checks", tuple(self.required_checks))
        object.__setattr__(
            self,
            "entrypoints",
            cast(Mapping[str, Mapping[str, object]], _deep_freeze(self.entrypoints)),
        )


def _load_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path.name}: invalid YAML") from error


def _require_mapping(value: object, path: str) -> dict[object, object]:
    if type(value) is not dict:
        raise ValueError(f"{path} must be a mapping")
    return cast(dict[object, object], value)


def _field_names(fields: set[object]) -> str:
    return ", ".join(sorted(str(field) for field in fields))


def _validate_fields(
    value: Mapping[object, object],
    *,
    required: set[object],
    optional: set[object],
    path: str,
) -> None:
    fields = set(value)
    unknown = fields - required - optional
    if unknown:
        raise ValueError(f"{path}: unknown fields: {_field_names(unknown)}")
    missing = required - fields
    if missing:
        raise ValueError(f"{path}: missing required fields: {_field_names(missing)}")


def _require_string_list(value: object, path: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{path} must be a list")
    result = []
    for index, item in enumerate(cast(list[object], value)):
        if type(item) is not str:
            raise ValueError(f"{path}[{index}] must be a string")
        result.append(cast(str, item))
    return tuple(result)


def _validate_exact(value: object, expected: object, path: str) -> None:
    if isinstance(expected, Mapping):
        mapping = _require_mapping(value, path)
        expected_mapping = cast(Mapping[object, object], expected)
        expected_fields = set(expected_mapping)
        _validate_fields(
            mapping,
            required=expected_fields,
            optional=set(),
            path=path,
        )
        for field, expected_value in expected_mapping.items():
            _validate_exact(mapping[field], expected_value, f"{path}.{field}")
        return
    if isinstance(expected, tuple):
        if type(value) is not list:
            raise ValueError(f"{path} must be a list")
        items = cast(list[object], value)
        if len(items) != len(expected):
            raise ValueError(f"{path} must contain {len(expected)} items")
        for index, (item, expected_item) in enumerate(zip(items, expected)):
            _validate_exact(item, expected_item, f"{path}[{index}]")
        return
    if type(expected) is bool:
        if type(value) is not bool:
            raise ValueError(f"{path} must be a boolean")
        if value is not expected:
            raise ValueError(f"{path} must be {str(expected).lower()}")
        return
    if type(expected) is int:
        if type(value) is not int:
            raise ValueError(f"{path} must be an integer")
        if value != expected:
            raise ValueError(f"{path} must be {expected}")
        return
    if type(expected) is str:
        if type(value) is not str:
            raise ValueError(f"{path} must be a string")
        if value != expected:
            raise ValueError(f"{path} must be {expected!r}")
        return
    if expected is None:
        if value is not None:
            raise ValueError(f"{path} must be null")
        return
    raise TypeError(f"unsupported policy schema value at {path}")


def load_policy(path: Path, entrypoints_path: Path | None = None) -> Policy:
    """
    Load queue limits and the protected entrypoint allowlist.

    :param path: The policy YAML path.
    :param entrypoints_path: An optional entrypoint YAML path. By default the
        sibling ``entrypoints.yaml`` file is used.

    :returns: The loaded immutable policy record.
    """
    data = _require_mapping(_load_yaml(path), "policy")
    _validate_fields(
        data,
        required=set(_POLICY_REQUIRED_FIELDS),
        optional=set(_POLICY_OPTIONAL_FIELDS),
        path="policy",
    )

    wandb_entity = data["wandb_entity"]
    _validate_exact(wandb_entity, _WANDB_ENTITY, "policy.wandb_entity")

    allowed_wandb_projects = _require_string_list(
        data["allowed_wandb_projects"], "policy.allowed_wandb_projects"
    )
    if allowed_wandb_projects != _ALLOWED_WANDB_PROJECTS:
        raise ValueError("policy.allowed_wandb_projects must exactly match the reviewed projects")

    max_runtime_minutes = data.get("max_runtime_minutes", 360)
    if type(max_runtime_minutes) is not int or not 1 <= max_runtime_minutes <= 360:
        raise ValueError("policy.max_runtime_minutes must be an integer from 1 to 360")

    max_gpu_count = data.get("max_gpu_count", 2)
    if type(max_gpu_count) is not int or not 1 <= max_gpu_count <= 2:
        raise ValueError("policy.max_gpu_count must be an integer from 1 to 2")

    allowed_gpu_preferences = _require_string_list(
        data.get("allowed_gpu_preferences", list(_ALLOWED_GPU_PREFERENCES)),
        "policy.allowed_gpu_preferences",
    )
    if allowed_gpu_preferences != _ALLOWED_GPU_PREFERENCES:
        raise ValueError(
            "policy.allowed_gpu_preferences must exactly match the reviewed preferences"
        )

    required_checks = _require_string_list(data["required_checks"], "policy.required_checks")
    if required_checks != _REQUIRED_CHECKS:
        raise ValueError("policy.required_checks must exactly match the staged required checks")

    fixed_policy = {
        "repository_url": "https://github.com/edu-llm/OLMo-core.git",
        "scratch_root": "$HOME/orcd/scratch/edullm",
        "slurm_cpus_per_gpu": 4,
        "slurm_memory": "64G",
        "slurm_partition": "mit_normal_gpu",
    }
    for field_name, expected in fixed_policy.items():
        _validate_exact(data[field_name], expected, f"policy.{field_name}")

    reminder_after_minutes = data.get("reminder_after_minutes", 15)
    if type(reminder_after_minutes) is not int or reminder_after_minutes < 1:
        raise ValueError("policy.reminder_after_minutes must be a positive integer")

    reassign_after_minutes = data.get("reassign_after_minutes", 30)
    if type(reassign_after_minutes) is not int or reassign_after_minutes <= reminder_after_minutes:
        raise ValueError(
            "policy.reassign_after_minutes must be greater than reminder_after_minutes"
        )

    entrypoints_path = entrypoints_path or path.with_name("entrypoints.yaml")
    entrypoints_document = _require_mapping(_load_yaml(entrypoints_path), "entrypoints document")
    _validate_fields(
        entrypoints_document,
        required={"entrypoints"},
        optional=set(),
        path="entrypoints document",
    )
    entrypoints = entrypoints_document["entrypoints"]
    _validate_exact(entrypoints, _APPROVED_ENTRYPOINTS, "entrypoints")

    return Policy(
        wandb_entity=cast(str, wandb_entity),
        allowed_wandb_projects=allowed_wandb_projects,
        repository_url=cast(str, data["repository_url"]),
        scratch_root=cast(str, data["scratch_root"]),
        slurm_partition=cast(str, data["slurm_partition"]),
        slurm_memory=cast(str, data["slurm_memory"]),
        slurm_cpus_per_gpu=cast(int, data["slurm_cpus_per_gpu"]),
        max_runtime_minutes=cast(int, max_runtime_minutes),
        max_gpu_count=cast(int, max_gpu_count),
        allowed_gpu_preferences=allowed_gpu_preferences,
        required_checks=required_checks,
        entrypoints=cast(Mapping[str, Mapping[str, object]], entrypoints),
        reminder_after_minutes=cast(int, reminder_after_minutes),
        reassign_after_minutes=cast(int, reassign_after_minutes),
    )


def load_operators(path: Path) -> tuple[Operator, ...]:
    """
    Load the reviewed operator roster.

    :param path: The operator YAML path.

    :returns: Operators in their configured order.
    """
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("operators: invalid YAML") from None
    except OSError:
        raise ValueError("operators: configuration cannot be read") from None
    if type(document) is not dict or set(document) != {"operators"}:
        raise ValueError("operators: expected only the operators field")
    rows = cast(dict[object, object], document)["operators"]
    if type(rows) is not list:
        raise ValueError("operators: operators must be a list")

    operators: list[Operator] = []
    github_identities: set[str] = set()
    slack_identities: set[str] = set()
    enabled_rotations: set[int] = set()
    for index, value in enumerate(cast(list[object], rows)):
        if type(value) is not dict:
            raise ValueError(f"operators[{index}] must be a mapping")
        row = cast(dict[object, object], value)
        if "enabled" not in row:
            raise ValueError(f"operators[{index}].enabled must be a boolean")
        _validate_fields(
            row,
            required=set(_OPERATOR_REQUIRED_FIELDS),
            optional=set(_OPERATOR_OPTIONAL_FIELDS),
            path=f"operators[{index}]",
        )
        enabled = row["enabled"]
        if type(enabled) is not bool:
            raise ValueError(f"operators[{index}].enabled must be a boolean")
        try:
            github = normalize_actor_login(row["github"])
        except GitHubValidationError:
            raise ValueError(f"operators[{index}].github is invalid") from None
        if github != row["github"] or github.endswith("[bot]"):
            raise ValueError(f"operators[{index}].github must be a canonical user login")
        if github in github_identities:
            raise ValueError("operators: duplicate GitHub identities")
        github_identities.add(github)

        slack_user_id = row["slack_user_id"]
        if type(slack_user_id) is not str or _SLACK_USER_ID.fullmatch(slack_user_id) is None:
            raise ValueError(f"operators[{index}].slack_user_id is invalid")
        if slack_user_id in slack_identities:
            raise ValueError("operators: duplicate Slack identities")
        slack_identities.add(slack_user_id)

        rotation_order = row["rotation_order"]
        if type(rotation_order) is not int or rotation_order < 0:
            raise ValueError(f"operators[{index}].rotation_order must be a non-negative integer")
        if enabled and rotation_order in enabled_rotations:
            raise ValueError("operators: enabled rotation orders must be unique")
        if enabled:
            enabled_rotations.add(rotation_order)

        apptainer_path = row.get("apptainer_path")
        apptainer_sha256 = row.get("apptainer_sha256")
        if apptainer_path is not None and (
            type(apptainer_path) is not str
            or not apptainer_path.startswith("/")
            or ".." in apptainer_path.split("/")
        ):
            raise ValueError(f"operators[{index}].apptainer_path is invalid")
        if apptainer_sha256 is not None and (
            type(apptainer_sha256) is not str or _SHA256.fullmatch(apptainer_sha256) is None
        ):
            raise ValueError(f"operators[{index}].apptainer_sha256 is invalid")
        if (apptainer_path is None) != (apptainer_sha256 is None):
            raise ValueError(
                f"operators[{index}] must set both apptainer_path and apptainer_sha256"
            )
        operators.append(
            Operator(
                github=github,
                slack_user_id=slack_user_id,
                rotation_order=rotation_order,
                enabled=cast(bool, enabled),
                apptainer_path=cast(str | None, apptainer_path),
                apptainer_sha256=cast(str | None, apptainer_sha256),
            )
        )
    return tuple(operators)
