import dataclasses
import hashlib
import json
import operator
import re
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from edullm.models import AttemptRecord, JobRequest, JobStatus, Operator
from edullm.policy import load_operators, load_policy


def _mapping(value: object) -> Mapping[object, object]:
    assert isinstance(value, Mapping)
    return value


def _setitem(container: object, key: object, value: object) -> None:
    operator.setitem(container, key, value)  # type: ignore[call-overload]


def _policy_documents():
    policy = yaml.safe_load(Path("config/edullm/policy.yaml").read_text(encoding="utf-8"))
    entrypoints = yaml.safe_load(Path("config/edullm/entrypoints.yaml").read_text(encoding="utf-8"))
    return policy, entrypoints


def _write_policy_bundle(tmp_path, policy, entrypoints):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    (tmp_path / "entrypoints.yaml").write_text(
        yaml.safe_dump(entrypoints, sort_keys=False), encoding="utf-8"
    )
    return policy_path


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


def test_job_request_has_canonical_digest_and_is_immutable(valid_request):
    canonical = valid_request.canonical_json()

    assert canonical == json.dumps(
        dataclasses.asdict(valid_request), sort_keys=True, separators=(",", ":")
    )
    assert valid_request.digest == hashlib.sha256(canonical.encode()).hexdigest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        valid_request.status = JobStatus.READY


def test_job_status_values_are_stable():
    assert [status.value for status in JobStatus] == [
        "requested",
        "validating",
        "ready",
        "assigned",
        "submitted",
        "running",
        "completed",
        "failed",
        "cancelled",
        "preempted",
    ]


def test_supporting_records_are_immutable(valid_resolved_request):
    operator = Operator(
        github="operator",
        slack_user_id="U11111111",
        rotation_order=0,
        apptainer_path="/orcd/pool/edullm.sif",
        apptainer_sha256="c" * 64,
    )
    attempt = AttemptRecord(
        attempt_id="issue-42-attempt-1",
        request_digest=valid_resolved_request.request.digest,
        operator=operator.github,
        slurm_job_id="12345",
        wandb_run_id="issue-42-attempt-1",
        log_path="/scratch/logs/issue-42-attempt-1-12345.log",
    )

    assert valid_resolved_request.slurm_job_id is None
    assert operator.enabled is True
    assert attempt.operator == "operator"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(operator, "enabled", False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(valid_resolved_request, "slurm_job_id", "12345")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(attempt, "slurm_job_id", "54321")


def test_policy_loads_reviewed_defaults(tmp_path):
    policy_data, entrypoints = _policy_documents()
    for field in (
        "max_runtime_minutes",
        "max_gpu_count",
        "allowed_gpu_preferences",
        "reminder_after_minutes",
        "reassign_after_minutes",
    ):
        policy_data.pop(field)
    policy = load_policy(_write_policy_bundle(tmp_path, policy_data, entrypoints))

    assert policy.wandb_entity == "eduLLM"
    assert policy.allowed_wandb_projects == (
        "test",
        "pretraining",
        "posttraining",
        "evaluation",
        "data-pipeline",
    )
    assert policy.max_runtime_minutes == 360
    assert policy.max_gpu_count == 2
    assert policy.allowed_gpu_preferences == ("any", "l40s", "h100", "h200")
    assert policy.reminder_after_minutes == 15
    assert policy.reassign_after_minutes == 30


def test_production_policy_has_reviewed_limits_and_staged_required_checks():
    policy = load_policy(Path("config/edullm/policy.yaml"))

    assert policy.allowed_wandb_projects == (
        "test",
        "pretraining",
        "posttraining",
        "evaluation",
        "data-pipeline",
    )
    assert policy.max_runtime_minutes == 360
    assert policy.max_gpu_count == 2
    assert policy.allowed_gpu_preferences == ("any", "l40s", "h100", "h200")
    assert policy.reminder_after_minutes == 15
    assert policy.reassign_after_minutes == 30
    assert policy.required_checks == (
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
    # These upstream jobs require credentials that are unavailable in the pilot fork.
    assert {"Test", "Test checkpoint", "Test olmo3 ladder"}.isdisjoint(policy.required_checks)
    assert "Test edullm queue" not in policy.required_checks


def test_policy_rejects_invalid_yaml(tmp_path):
    _, entrypoints = _policy_documents()
    path = tmp_path / "policy.yaml"
    path.write_text("wandb_entity: [\n", encoding="utf-8")
    (tmp_path / "entrypoints.yaml").write_text(
        yaml.safe_dump(entrypoints, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="policy.yaml: invalid YAML"):
        load_policy(path)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "policy must be a mapping"),
        (
            {"entrypoints": []},
            "entrypoints must be a mapping",
        ),
    ],
    ids=["policy-root", "entrypoints-container"],
)
def test_policy_rejects_malformed_document_containers(tmp_path, document, message):
    policy, entrypoints = _policy_documents()
    if message.startswith("policy "):
        policy = document
    else:
        entrypoints = document

    with pytest.raises(ValueError, match=re.escape(message)):
        load_policy(_write_policy_bundle(tmp_path, policy, entrypoints))


@pytest.mark.parametrize("field", ["wandb_entity", "allowed_wandb_projects", "required_checks"])
def test_policy_rejects_missing_required_fields(tmp_path, field):
    policy, entrypoints = _policy_documents()
    policy.pop(field)

    with pytest.raises(ValueError, match=re.escape(f"policy: missing required fields: {field}")):
        load_policy(_write_policy_bundle(tmp_path, policy, entrypoints))


def test_policy_rejects_unknown_root_fields(tmp_path):
    policy, entrypoints = _policy_documents()
    policy["unreviewed"] = True

    with pytest.raises(ValueError, match="policy: unknown fields: unreviewed"):
        load_policy(_write_policy_bundle(tmp_path, policy, entrypoints))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("wandb_entity", "other", "policy.wandb_entity must be 'eduLLM'"),
        (
            "allowed_wandb_projects",
            ["test"],
            "policy.allowed_wandb_projects must exactly match the reviewed projects",
        ),
        (
            "allowed_wandb_projects",
            "test",
            "policy.allowed_wandb_projects must be a list",
        ),
        (
            "max_runtime_minutes",
            361,
            "policy.max_runtime_minutes must be an integer from 1 to 360",
        ),
        (
            "max_runtime_minutes",
            True,
            "policy.max_runtime_minutes must be an integer from 1 to 360",
        ),
        (
            "max_gpu_count",
            3,
            "policy.max_gpu_count must be an integer from 1 to 2",
        ),
        (
            "max_gpu_count",
            "2",
            "policy.max_gpu_count must be an integer from 1 to 2",
        ),
        (
            "allowed_gpu_preferences",
            ["any", "l40s", "attacker"],
            "policy.allowed_gpu_preferences must exactly match the reviewed preferences",
        ),
        (
            "required_checks",
            ["Lint"],
            "policy.required_checks must exactly match the staged required checks",
        ),
        (
            "required_checks",
            "Lint",
            "policy.required_checks must be a list",
        ),
        (
            "reminder_after_minutes",
            0,
            "policy.reminder_after_minutes must be a positive integer",
        ),
        (
            "reassign_after_minutes",
            15,
            "policy.reassign_after_minutes must be greater than reminder_after_minutes",
        ),
    ],
)
def test_policy_rejects_limit_and_allowlist_violations(tmp_path, field, value, message):
    policy, entrypoints = _policy_documents()
    policy[field] = value

    with pytest.raises(ValueError, match=re.escape(message)):
        load_policy(_write_policy_bundle(tmp_path, policy, entrypoints))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown-root", "entrypoints document: unknown fields: unreviewed"),
        ("missing-profile", "entrypoints: missing required fields: generic-smoke"),
        ("unknown-profile", "entrypoints: unknown fields: arbitrary"),
        ("profile-container", "entrypoints.generic-smoke must be a mapping"),
        ("unknown-profile-field", "entrypoints.generic-smoke: unknown fields: arbitrary"),
        ("missing-script", "entrypoints.generic-smoke: missing required fields: script"),
        (
            "script-traversal",
            "entrypoints.generic-smoke.script must be 'src/examples/llm/train.py'",
        ),
        ("launcher", "entrypoints.generic-smoke.launcher must be 'torchrun'"),
        (
            "wandb-callback-type",
            "entrypoints.generic-smoke.wandb_callback must be a boolean",
        ),
        (
            "allowed-data",
            "entrypoints.generic-smoke.allowed_data_kinds[0] must be 'generic-smoke'",
        ),
        (
            "positionals-type",
            "entrypoints.generic-smoke.positionals must be an integer",
        ),
        (
            "positional-schema",
            "entrypoints.generic-smoke.allowed_positionals.0.type must be 'slug'",
        ),
        (
            "option-schema",
            "entrypoints.hypothesis-smoke.allowed_options.trainer.hard_stop.type "
            "must be 'duration'",
        ),
        (
            "seed-min",
            "entrypoints.hypothesis-smoke.allowed_options.seed.min must be 0",
        ),
        (
            "seed-max",
            "entrypoints.hypothesis-smoke.allowed_options.seed.max must be 2147483647",
        ),
        (
            "seed-required",
            "entrypoints.hypothesis-smoke.allowed_options.seed.required must be true",
        ),
        (
            "seed-request-field",
            "entrypoints.hypothesis-smoke.allowed_options.seed.request_field must be 'seed'",
        ),
        (
            "derived-root",
            "entrypoints.generic-smoke.fixed_options.save-folder.root_env "
            "must be 'EDULLM_SCRATCH'",
        ),
        (
            "derived-relative",
            "entrypoints.generic-smoke.fixed_options.save-folder.relative "
            "must be 'runs/{run_name}'",
        ),
        (
            "fixed-option-value",
            "entrypoints.generic-smoke.fixed_options.model-factory must be 'olmo2_190M'",
        ),
        (
            "unknown-fixed-option",
            "entrypoints.generic-smoke.fixed_options: unknown fields: attacker",
        ),
        (
            "missing-fixed-option",
            "entrypoints.generic-smoke.fixed_options: missing required fields: trainer.hard_stop",
        ),
        (
            "hypothesis-wandb-false",
            "entrypoints.hypothesis-smoke.fixed_options."
            "trainer.callbacks.wandb.enabled must be true",
        ),
        (
            "hypothesis-wandb-allowed",
            "entrypoints.hypothesis-smoke.allowed_options: unknown fields: "
            "trainer.callbacks.wandb.enabled",
        ),
    ],
)
def test_policy_rejects_profile_schema_violations(tmp_path, case, message):
    policy, document = _policy_documents()
    profiles = document["entrypoints"]
    generic = profiles["generic-smoke"]
    hypothesis = profiles["hypothesis-smoke"]

    if case == "unknown-root":
        document["unreviewed"] = True
    elif case == "missing-profile":
        profiles.pop("generic-smoke")
    elif case == "unknown-profile":
        profiles["arbitrary"] = generic.copy()
    elif case == "profile-container":
        profiles["generic-smoke"] = []
    elif case == "unknown-profile-field":
        generic["arbitrary"] = True
    elif case == "missing-script":
        generic.pop("script")
    elif case == "script-traversal":
        generic["script"] = "../attacker.py"
    elif case == "launcher":
        generic["launcher"] = "bash"
    elif case == "wandb-callback-type":
        generic["wandb_callback"] = "true"
    elif case == "allowed-data":
        generic["allowed_data_kinds"] = ["attacker"]
    elif case == "positionals-type":
        generic["positionals"] = "1"
    elif case == "positional-schema":
        generic["allowed_positionals"][0]["type"] = "string"
    elif case == "option-schema":
        hypothesis["allowed_options"]["trainer.hard_stop"]["type"] = "unknown"
    elif case in {"seed-min", "seed-max", "seed-required", "seed-request-field"}:
        seed_rule = hypothesis["allowed_options"].setdefault(
            "seed",
            {
                "type": "integer",
                "min": 0,
                "max": 2147483647,
                "required": True,
                "request_field": "seed",
            },
        )
        if case == "seed-min":
            seed_rule["min"] = -1
        elif case == "seed-max":
            seed_rule["max"] = 2147483648
        elif case == "seed-required":
            seed_rule["required"] = False
        else:
            seed_rule["request_field"] = "other"
    elif case == "derived-root":
        generic["fixed_options"]["save-folder"]["root_env"] = "HOME"
    elif case == "derived-relative":
        generic["fixed_options"]["save-folder"]["relative"] = "../runs/{run_name}"
    elif case == "fixed-option-value":
        generic["fixed_options"]["model-factory"] = "attacker"
    elif case == "unknown-fixed-option":
        generic["fixed_options"]["attacker"] = True
    elif case == "missing-fixed-option":
        generic["fixed_options"].pop("trainer.hard_stop")
    elif case == "hypothesis-wandb-false":
        hypothesis["fixed_options"]["trainer.callbacks.wandb.enabled"] = False
    elif case == "hypothesis-wandb-allowed":
        hypothesis["allowed_options"]["trainer.callbacks.wandb.enabled"] = {"type": "boolean"}
    else:
        raise AssertionError(f"unhandled test case: {case}")

    with pytest.raises(ValueError, match=re.escape(message)):
        load_policy(_write_policy_bundle(tmp_path, policy, document))


def test_generic_profile_owns_the_reviewed_plan_1_smoke_arguments():
    profile = load_policy(Path("config/edullm/policy.yaml")).entrypoints["generic-smoke"]

    assert profile["script"] == "src/examples/llm/train.py"
    assert profile["launcher"] == "torchrun"
    assert profile["fixed_launcher_arguments"] == ("--standalone", "--nproc-per-node=1")
    assert profile["fixed_options"] == {
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
    }


def test_generic_profile_only_accepts_typed_non_safety_critical_values():
    profile = load_policy(Path("config/edullm/policy.yaml")).entrypoints["generic-smoke"]

    assert profile["positionals"] == 1
    assert profile["allowed_positionals"] == {0: {"type": "slug"}}
    assert profile["allowed_options"] == {}
    serialized = repr(profile)
    assert "$HOME" not in serialized
    assert "EDULLM_SCRATCH" in serialized


def test_loaded_policy_is_recursively_immutable():
    policy = load_policy(Path("config/edullm/policy.yaml"))
    generic = policy.entrypoints["generic-smoke"]
    hypothesis = policy.entrypoints["hypothesis-smoke"]
    fixed_options = _mapping(generic["fixed_options"])
    tags = fixed_options["trainer.callbacks.wandb.tags"]
    generic_positionals = _mapping(generic["allowed_positionals"])
    generic_first_positional = _mapping(generic_positionals[0])
    hypothesis_positionals = _mapping(hypothesis["allowed_positionals"])
    hypothesis_options = _mapping(hypothesis["allowed_options"])
    seed_rule = _mapping(hypothesis_options["seed"])

    with pytest.raises(TypeError):
        _setitem(policy.entrypoints, "generic-smoke", {})
    with pytest.raises(TypeError):
        _setitem(generic, "script", "src/attacker.py")
    with pytest.raises(TypeError):
        _setitem(generic, "launcher", "bash")
    with pytest.raises(TypeError):
        _setitem(fixed_options, "model-factory", "attacker")
    with pytest.raises(TypeError):
        _setitem(tags, 0, "attacker")
    with pytest.raises(TypeError):
        _setitem(generic_first_positional, "type", "string")
    with pytest.raises(TypeError):
        _setitem(hypothesis_positionals[0], 0, "attacker")
    with pytest.raises(TypeError):
        _setitem(seed_rule, "max", 99)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(policy, "max_gpu_count", 99)

    assert generic["script"] == "src/examples/llm/train.py"
    assert generic["launcher"] == "torchrun"
    assert fixed_options["model-factory"] == "olmo2_190M"
    assert fixed_options["trainer.callbacks.wandb.tags"] == (
        "orcd",
        "generic-smoke",
        "olmo2-190m",
    )
    assert generic_first_positional["type"] == "slug"
    assert hypothesis_positionals[0] == ("dry_run", "train_single", "train")
    assert seed_rule["max"] == 2147483647
    assert policy.max_gpu_count == 2


def test_hypothesis_profile_fixes_wandb_enabled():
    profile = load_policy(Path("config/edullm/policy.yaml")).entrypoints["hypothesis-smoke"]
    fixed_options = _mapping(profile["fixed_options"])
    allowed_options = _mapping(profile["allowed_options"])

    assert fixed_options["trainer.callbacks.wandb.enabled"] is True
    assert "trainer.callbacks.wandb.enabled" not in allowed_options


def test_hypothesis_profile_allows_one_authoritative_bounded_seed():
    profile = load_policy(Path("config/edullm/policy.yaml")).entrypoints["hypothesis-smoke"]
    allowed_options = _mapping(profile["allowed_options"])

    assert allowed_options["seed"] == {
        "type": "integer",
        "min": 0,
        "max": 2147483647,
        "required": True,
        "request_field": "seed",
    }


@pytest.mark.parametrize(
    "enabled",
    [pytest.param(None, id="missing"), pytest.param("false", id="quoted-false"), 0, 1],
)
def test_operator_enabled_requires_an_explicit_yaml_boolean(tmp_path, enabled):
    row = {
        "github": "operator",
        "slack_user_id": "U11111111",
        "rotation_order": 0,
    }
    if enabled is not None:
        row["enabled"] = enabled
    path = tmp_path / "operators.yaml"
    path.write_text(yaml.safe_dump({"operators": [row]}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"operators\[0\]\.enabled must be a boolean"):
        load_operators(path)


@pytest.mark.parametrize("enabled", [False, True])
def test_operator_enabled_preserves_explicit_yaml_booleans(tmp_path, enabled):
    path = tmp_path / "operators.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "operators": [
                    {
                        "github": "operator",
                        "slack_user_id": "U11111111",
                        "rotation_order": 0,
                        "enabled": enabled,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_operators(path)[0].enabled is enabled


def test_operator_files_keep_reviewed_roster_and_examples_inert():
    assert load_operators(Path("config/edullm/operators.yaml")) == (
        Operator(
            github="philote-dev",
            slack_user_id="U0BA7EHAKJR",
            rotation_order=0,
            enabled=True,
        ),
        Operator(
            github="meric233",
            slack_user_id="U0BAARXNKC2",
            rotation_order=1,
            enabled=True,
        ),
        Operator(
            github="alsy7009",
            slack_user_id="U0B9K2XTTBL",
            rotation_order=2,
            enabled=True,
        ),
    )
    examples = load_operators(Path("config/edullm/operators.example.yaml"))

    assert [operator.github for operator in examples] == ["alice", "bob", "carol"]
    assert [operator.enabled for operator in examples] == [True, False, False]
    assert [operator.rotation_order for operator in examples] == [0, 1, 2]


@pytest.mark.parametrize(
    "document",
    [
        "",
        "[]\n",
        "unknown: []\n",
        "operators: {}\n",
        "operators: []\nextra: true\n",
        "operators:\n  - github: alice\n",
        (
            "operators:\n"
            "  - github: Alice\n"
            "    slack_user_id: U11111111\n"
            "    rotation_order: 0\n"
            "    enabled: true\n"
        ),
        (
            "operators:\n"
            "  - github: alice\n"
            "    slack_user_id: invalid\n"
            "    rotation_order: 0\n"
            "    enabled: true\n"
        ),
        (
            "operators:\n"
            "  - github: alice\n"
            "    slack_user_id: U11111111\n"
            "    rotation_order: -1\n"
            "    enabled: true\n"
        ),
        (
            "operators:\n"
            "  - github: alice\n"
            "    slack_user_id: U11111111\n"
            "    rotation_order: 0\n"
            "    enabled: true\n"
            "    unknown: secret\n"
        ),
        (
            "operators:\n"
            "  - github: alice\n"
            "    slack_user_id: U11111111\n"
            "    rotation_order: 0\n"
            "    enabled: true\n"
            "  - github: Alice\n"
            "    slack_user_id: U22222222\n"
            "    rotation_order: 1\n"
            "    enabled: false\n"
        ),
        (
            "operators:\n"
            "  - github: alice\n"
            "    slack_user_id: U11111111\n"
            "    rotation_order: 0\n"
            "    enabled: true\n"
            "  - github: bob\n"
            "    slack_user_id: U11111111\n"
            "    rotation_order: 1\n"
            "    enabled: true\n"
        ),
        (
            "operators:\n"
            "  - github: alice\n"
            "    slack_user_id: U11111111\n"
            "    rotation_order: 0\n"
            "    enabled: true\n"
            "  - github: bob\n"
            "    slack_user_id: U22222222\n"
            "    rotation_order: 0\n"
            "    enabled: true\n"
        ),
    ],
)
def test_operator_roster_rejects_malformed_or_duplicate_protected_state(tmp_path, document):
    path = tmp_path / "operators.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="operators"):
        load_operators(path)


def test_disabled_operators_may_not_hide_duplicate_enabled_rotation(tmp_path):
    path = tmp_path / "operators.yaml"
    path.write_text(
        """
operators:
  - github: disabled
    slack_user_id: U00000000
    rotation_order: 0
    enabled: false
  - github: alice
    slack_user_id: U11111111
    rotation_order: 0
    enabled: true
  - github: bob
    slack_user_id: U22222222
    rotation_order: 0
    enabled: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rotation"):
        load_operators(path)


def test_package_metadata_includes_edullm_with_task_6_console_script():
    metadata = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in metadata
    assert 'include = ["olmo_core*", "edullm*"]' in metadata
    assert "[project.scripts]" in metadata
    assert 'edullm = "edullm.cli:main"' in metadata
