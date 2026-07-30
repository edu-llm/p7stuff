from __future__ import annotations

import hashlib
import json
import os
import pwd
import shlex
import stat
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from edullm.data_manifest import BUILTIN_GENERIC_SMOKE_SHA256
from edullm.jobs import SubmissionGateSnapshot, build_resolved_request
from edullm.policy import load_policy
from edullm.slurm import (
    SubmissionError,
    SubmissionReceipt,
    SubmissionSpec,
    build_submission_key,
    parse_submission_receipt,
    render_sbatch,
    stage_submission,
    submission_transaction,
)
from edullm.validation import validate_request
from olmo_core.config import Config
from olmo_core.train import Duration

NOW = datetime(2026, 7, 23, 13, 0, 0, tzinfo=timezone.utc)
REMOTE_USER = pwd.getpwuid(os.geteuid()).pw_name


@dataclass
class DurationConfig(Config):
    hard_stop: Duration | None = None


def _spec(valid_resolved_request, script: str) -> SubmissionSpec:
    request = valid_resolved_request.request
    return SubmissionSpec(
        issue=request.issue_number,
        request_digest=request.digest,
        attempt_number=1,
        operator=valid_resolved_request.operator,
        remote_user=REMOTE_USER,
        script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        manifest_uri=request.data_manifest,
        manifest_sha256=request.data_manifest_sha256,
        allowed_data_kinds=valid_resolved_request.allowed_data_kinds,
        log_pattern=valid_resolved_request.log_pattern,
    )


def _runner(
    calls: list[list[str]],
    *,
    output: str = "12345\n",
    returncode: int = 0,
):
    def run(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs["check"] is False
        assert kwargs["text"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] > 0
        assert kwargs["cwd"].startswith("/")
        return subprocess.CompletedProcess(argv, returncode, output, "private slurm diagnostics")

    return run


def test_render_uses_fixed_directives_structured_argv_and_reviewed_worktree(
    valid_resolved_request,
):
    text = render_sbatch(valid_resolved_request)

    assert "#SBATCH -p mit_normal_gpu" in text
    assert "#SBATCH -G l40s:1" in text
    assert "#SBATCH -t 00:30:00" in text
    assert "#SBATCH -c 4" in text
    assert "#SBATCH --mem=64G" in text
    assert "#SBATCH --export=NONE" in text
    assert "#SBATCH -o logs/issue-42-attempt-1-%j.log" in text
    assert "#SBATCH -e logs/issue-42-attempt-1-%j.log" in text
    assert "git clone --no-checkout https://github.com/edu-llm/OLMo-core.git" in text
    assert "git checkout --detach " + "a" * 40 in text
    assert 'mkdir -p "$(dirname "$WORKTREE")"' in text
    assert 'export PYTHONPATH="$WORKTREE/src"' in text
    assert 'source "$HOME/venvs/edullm/bin/activate"' in text
    assert 'source "$HOME/.config/edullm/wandb.env"' in text
    assert "python -m edullm.data_manifest render-env" in text
    assert "eval " not in text
    assert "pip install" not in text
    assert "WANDB_API_KEY" not in text


def test_sbatch_binds_exact_sha_wandb_identity_and_safe_argv(valid_resolved_request):
    text = render_sbatch(valid_resolved_request)
    request = valid_resolved_request.request

    assert f"git fetch --no-tags origin {request.commit_sha}" in text
    assert f'test "$(git rev-parse HEAD)" = {request.commit_sha}' in text
    assert 'WANDB_RUN_ID="${WANDB_RUN_PREFIX}-${SLURM_JOB_ID}"' in text
    assert (
        'WANDB_RUN_URL="https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}' '/runs/${WANDB_RUN_ID}"'
    ) in text
    assert request.digest in text
    assert "eval " not in text
    assert "WANDB_API_KEY" not in text


def test_generic_smoke_renders_every_protected_launcher_and_training_option(
    valid_resolved_request,
):
    policy = load_policy(Path("config/edullm/policy.yaml"))
    request = replace(
        valid_resolved_request.request,
        entrypoint_profile="generic-smoke",
        script_path="src/examples/llm/train.py",
        launcher="torchrun",
        argv=("generic-run",),
        data_manifest="builtin://generic-smoke-v1",
        data_manifest_sha256=BUILTIN_GENERIC_SMOKE_SHA256,
        wandb_project="test",
    )
    snapshot = SubmissionGateSnapshot(
        issue=request.issue_number,
        request=request,
        request_digest=request.digest,
        operator="operator",
        validated_at=NOW,
        status_comment_id=1,
        assignment_comment_id=2,
        assignment_binding="b" * 64,
        assignment_version=0,
        config_digest="c" * 64,
        lifecycle=None,
        profile=policy.entrypoints["generic-smoke"],
        repository_url=policy.repository_url,
        scratch_root=policy.scratch_root,
        slurm_partition=policy.slurm_partition,
        slurm_memory=policy.slurm_memory,
        slurm_cpus_per_gpu=policy.slurm_cpus_per_gpu,
    )

    text = render_sbatch(build_resolved_request(snapshot, attempt_number=1))

    expected = (
        "--model-factory=olmo2_190M",
        "--sequence-length=512",
        "data_loader.global_batch_size=8192",
        "train_module.rank_microbatch_size=2048",
        "train_module.max_sequence_length=512",
        'trainer.hard_stop={"value":20,"unit":"steps"}',
        "trainer.callbacks.lm_evaluator.enabled=false",
        "trainer.callbacks.downstream_evaluator.enabled=false",
        "trainer.callbacks.checkpointer.save_interval=10",
        "trainer.callbacks.checkpointer.ephemeral_save_interval=null",
        "trainer.callbacks.wandb.enabled=true",
        "trainer.callbacks.wandb.entity=eduLLM",
        "trainer.callbacks.wandb.project=test",
        "trainer.callbacks.wandb.group=skill-dag-v1",
    )
    for argument in expected:
        assert argument in text
    command_line = next(line for line in text.splitlines() if line.startswith("TRAIN_COMMAND=("))
    command = command_line.removeprefix("TRAIN_COMMAND=(").removesuffix(")")
    tokens = shlex.split(command)
    hard_stop = next(token for token in tokens if token.startswith("--trainer.hard_stop=")).split(
        "=", 1
    )[1]
    parsed = DurationConfig().merge([f"hard_stop={hard_stop}"])
    assert parsed.hard_stop == Duration.steps(20)
    assert text.count("--standalone") == 1
    assert text.count("--nproc-per-node=1") == 1
    assert 'EDULLM_RUN_DIR="$EDULLM_SCRATCH/runs/issue-42-skill-dag-v1-natural"' in text
    assert '"--save-folder=$EDULLM_RUN_DIR"' in text
    assert '"--work-dir=$EDULLM_RUN_DIR"' in text


@pytest.mark.parametrize("arm", ["bare", "complete", "fade_ordered", "fade_shuffled"])
def test_worked_examples_cpt_renders_locked_science_inputs(valid_resolved_request, arm):
    policy = load_policy(Path("config/edullm/policy.yaml"))
    request = replace(
        valid_resolved_request.request,
        purpose="Compare worked and faded scaffolds at a matched CPT token budget",
        study="worked-examples-faded-scaffolds",
        condition=arm,
        comparison="bare-vs-complete-vs-fade-ordered-vs-fade-shuffled",
        entrypoint_profile="worked-examples-cpt",
        script_path="src/scripts/hypothesis/worked_examples/train_cpt_arm.py",
        launcher="torchrun",
        argv=(f"we-cpt-{arm.replace('_', '-')}", f"--arm={arm}"),
        data_manifest="/orcd/pool/edullm/manifests/worked-examples-metamath-v0.json",
        wandb_project="pretraining",
        success_signal="Finite train/PPL and Pass@N metrics",
        success_metrics=("eval/pass_at_n", "eval/pass_ratio_at_n"),
        gpu_count=2,
        gpu_preference="h100",
        max_runtime_minutes=360,
    )
    assert validate_request(request, policy) == []
    snapshot = SubmissionGateSnapshot(
        issue=request.issue_number,
        request=request,
        request_digest=request.digest,
        operator="operator",
        validated_at=NOW,
        status_comment_id=1,
        assignment_comment_id=2,
        assignment_binding="b" * 64,
        assignment_version=0,
        config_digest="c" * 64,
        lifecycle=None,
        profile=policy.entrypoints["worked-examples-cpt"],
        repository_url=policy.repository_url,
        scratch_root=policy.scratch_root,
        slurm_partition=policy.slurm_partition,
        slurm_memory=policy.slurm_memory,
        slurm_cpus_per_gpu=policy.slurm_cpus_per_gpu,
    )

    text = render_sbatch(build_resolved_request(snapshot, attempt_number=1))

    for argument in (
        f"--arm={arm}",
        "--pack-dir=/orcd/pool/edullm/data/worked-examples-metamath-v0",
        "--load-path=/orcd/pool/edullm/checkpoints/OLMo-Ladder-760M-0.5xC-core",
        "--token-budget=200000000",
        "--seed=0",
        "--wandb-project=pretraining",
        "--wandb-group=worked-examples-faded-scaffolds",
        "--trainer.callbacks.wandb.project=pretraining",
    ):
        assert argument in text
    assert "#SBATCH -G h100:2" in text
    assert "#SBATCH -t 06:00:00" in text
    assert text.count("--standalone") == 1
    assert text.count("--nproc-per-node=2") == 1
    assert '"--save-folder=$EDULLM_RUN_DIR"' in text
    assert '"--work-dir=$EDULLM_RUN_DIR"' in text


def test_generic_smoke_merges_static_and_dynamic_audit_tags_once_shell_safely(
    valid_resolved_request,
):
    policy = load_policy(Path("config/edullm/policy.yaml"))
    request = replace(
        valid_resolved_request.request,
        entrypoint_profile="generic-smoke",
        script_path="src/examples/llm/train.py",
        launcher="torchrun",
        argv=("generic-run",),
        data_manifest="builtin://generic-smoke-v1",
        data_manifest_sha256=BUILTIN_GENERIC_SMOKE_SHA256,
        wandb_project="test",
    )
    snapshot = SubmissionGateSnapshot(
        issue=request.issue_number,
        request=request,
        request_digest=request.digest,
        operator="operator",
        validated_at=NOW,
        status_comment_id=1,
        assignment_comment_id=2,
        assignment_binding="b" * 64,
        assignment_version=0,
        config_digest="c" * 64,
        lifecycle=None,
        profile=policy.entrypoints["generic-smoke"],
        repository_url=policy.repository_url,
        scratch_root=policy.scratch_root,
        slurm_partition=policy.slurm_partition,
        slurm_memory=policy.slurm_memory,
        slurm_cpus_per_gpu=policy.slurm_cpus_per_gpu,
    )

    text = render_sbatch(build_resolved_request(snapshot, attempt_number=1))
    command_line = next(line for line in text.splitlines() if line.startswith("TRAIN_COMMAND=("))
    command = command_line.removeprefix("TRAIN_COMMAND=(").removesuffix(")")
    tokens = shlex.split(command)
    tag_arguments = [
        token for token in tokens if token.startswith("--trainer.callbacks.wandb.tags=")
    ]

    assert len(tag_arguments) == 1
    tags_json = tag_arguments[0].split("=", 1)[1]
    tags = json.loads(tags_json)
    assert tags == [
        "orcd",
        "generic-smoke",
        "olmo2-190m",
        "issue-42",
        "attempt-1",
        "natural",
        "l40s",
        "engaging",
        request.commit_sha,
        request.commit_sha[:12],
        f"seed-{request.seed}",
        request.digest,
        request.data_manifest_sha256,
    ]
    assert tag_arguments[0] == (
        "--trainer.callbacks.wandb.tags=" + json.dumps(tags, separators=(",", ":"))
    )


@pytest.mark.parametrize(
    "override",
    [
        "--model-factory=other",
        "--sequence-length=4096",
        "--save-folder=/tmp/escape",
        "--trainer.hard_stop={value:200,unit:steps}",
        "--trainer.callbacks.wandb.project=other",
        '--trainer.callbacks.wandb.tags=["requester-controlled"]',
    ],
)
def test_generic_smoke_rejects_researcher_overrides_of_protected_options(
    valid_resolved_request,
    override,
):
    policy = load_policy(Path("config/edullm/policy.yaml"))
    request = replace(
        valid_resolved_request.request,
        entrypoint_profile="generic-smoke",
        script_path="src/examples/llm/train.py",
        launcher="torchrun",
        argv=("generic-run", override),
        data_manifest="builtin://generic-smoke-v1",
        data_manifest_sha256=BUILTIN_GENERIC_SMOKE_SHA256,
        wandb_project="test",
    )

    errors = validate_request(request, policy)

    assert any("fixed by policy" in error for error in errors)


def test_generic_smoke_rejects_request_level_wandb_project_override(
    valid_resolved_request,
):
    policy = load_policy(Path("config/edullm/policy.yaml"))
    request = replace(
        valid_resolved_request.request,
        entrypoint_profile="generic-smoke",
        script_path="src/examples/llm/train.py",
        launcher="torchrun",
        argv=("generic-run",),
        data_manifest="builtin://generic-smoke-v1",
        data_manifest_sha256=BUILTIN_GENERIC_SMOKE_SHA256,
        wandb_project="pretraining",
    )

    errors = validate_request(request, policy)

    assert "W&B project is fixed by entrypoint policy" in errors


def test_render_audit_metadata_is_complete_and_wandb_id_is_slurm_deterministic(
    valid_resolved_request,
):
    text = render_sbatch(valid_resolved_request)
    request = valid_resolved_request.request

    for value in (
        "issue-42",
        "attempt-1",
        request.entrypoint_profile,
        request.condition,
        "l40s",
        request.commit_sha,
        f"seed-{request.seed}",
        request.digest,
        request.data_manifest_sha256,
        valid_resolved_request.model_identity,
        "engaging",
    ):
        assert value in text
    assert 'WANDB_RUN_ID="${WANDB_RUN_PREFIX}-${SLURM_JOB_ID}"' in text
    assert (
        'WANDB_RUN_URL="https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${WANDB_RUN_ID}"'
        in text
    )
    assert "timeout 10" in text
    assert "WANDB_MODE=offline" in text
    assert "wandb sync" in text
    assert "|| true" in text


@pytest.mark.parametrize(
    "launcher,expected",
    [
        ("python", "python src/scripts/train/smoketests/"),
        ("torchrun", "torchrun --standalone --nproc-per-node=1"),
        ("bash", "bash src/scripts/train/smoketests/"),
    ],
)
def test_render_quotes_arguments_once_for_every_launcher(
    valid_resolved_request,
    launcher,
    expected,
):
    request = replace(
        valid_resolved_request.request,
        launcher=launcher,
        argv=("value with spaces", "$(touch nope)", ";", "--option=-leading"),
    )
    resolved = replace(valid_resolved_request, request=request)

    text = render_sbatch(resolved)

    assert expected in text
    assert "'value with spaces'" in text
    assert "'$(touch nope)'" in text
    assert "';'" in text
    assert "--option=-leading" in text
    assert "eval " not in text


@pytest.mark.parametrize(
    "mutator",
    [
        lambda resolved: replace(resolved, slurm_job_name="bad\n#SBATCH --uid=root"),
        lambda resolved: replace(resolved, log_pattern="../../tmp/%j"),
        lambda resolved: replace(resolved, slurm_partition="other"),
        lambda resolved: replace(resolved, slurm_memory="64G\n#SBATCH --requeue"),
        lambda resolved: replace(resolved, repository_url="https://evil.invalid/repo"),
        lambda resolved: replace(resolved, model_identity="model\nBAD"),
        lambda resolved: replace(
            resolved,
            request=replace(resolved.request, gpu_preference="l40s\n#SBATCH --uid=0"),
        ),
        lambda resolved: replace(
            resolved,
            request=replace(resolved.request, argv=("nul\x00value",)),
        ),
        lambda resolved: replace(
            resolved,
            request=replace(resolved.request, commit_sha="A" * 40),
        ),
    ],
)
def test_render_revalidates_every_resolved_value(valid_resolved_request, mutator):
    with pytest.raises(SubmissionError):
        render_sbatch(mutator(valid_resolved_request))


def test_render_creates_private_atomic_request_and_data_environment(
    valid_resolved_request,
):
    text = render_sbatch(valid_resolved_request)

    assert "umask 077" in text
    assert 'REQUEST_TMP="$(mktemp "$WORKTREE/.edullm-request.XXXXXX")"' in text
    assert 'chmod 600 "$REQUEST_TMP"' in text
    assert 'mv -f "$REQUEST_TMP" "$EDULLM_REQUEST_PATH"' in text
    assert 'DATA_ENV_TMP="$(mktemp "$WORKTREE/.edullm-data.XXXXXX")"' in text
    assert 'chmod 600 "$DATA_ENV_TMP"' in text
    assert 'mv -f "$DATA_ENV_TMP" "$DATA_ENV"' in text
    assert 'if ! GIT_STATUS="$(git status --porcelain --untracked-files=all)"; then' in text
    assert 'test -z "$GIT_STATUS"' in text


def test_submission_key_is_deterministic_bounded_and_nonsemantic(valid_resolved_request):
    request = valid_resolved_request.request

    first = build_submission_key(42, request.digest, 1, "operator")
    second = build_submission_key(42, request.digest, 1, "operator")

    assert first == second
    assert len(first) == 64
    assert first.isalnum()
    with pytest.raises(SubmissionError):
        build_submission_key(0, request.digest, 1, "operator")
    with pytest.raises(SubmissionError):
        build_submission_key(42, request.digest, 1, "Operator")


def test_transaction_submits_once_and_persists_private_canonical_receipt(
    tmp_path,
    valid_resolved_request,
):
    script = render_sbatch(valid_resolved_request)
    spec = _spec(valid_resolved_request, script)
    key = build_submission_key(spec.issue, spec.request_digest, spec.attempt_number, spec.operator)
    stage_submission(tmp_path, key, script)
    calls: list[list[str]] = []
    manifest_calls = []

    receipt = submission_transaction(
        tmp_path,
        key,
        spec,
        sbatch_runner=_runner(calls),
        manifest_verifier=lambda *args: manifest_calls.append(args),
        now=NOW,
    )

    assert receipt.slurm_job_id == "12345"
    assert receipt.log_path == str(tmp_path.parent / "logs/issue-42-attempt-1-12345.log").replace(
        str(tmp_path.parent), str(tmp_path.parent)
    )
    assert calls == [
        [
            "sbatch",
            "--export=NONE",
            "--parsable",
            f"--job-name=edullm-{key}",
            f"--comment=edullm:{key}",
            str(tmp_path / key / "request.sbatch"),
        ]
    ]
    assert manifest_calls == []
    receipt_path = tmp_path / key / "receipt.json"
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(receipt_path.parent.stat().st_mode) == 0o700
    assert parse_submission_receipt(receipt_path.read_text(encoding="utf-8")) == receipt


def test_retry_returns_existing_receipt_without_second_submit(
    tmp_path,
    valid_resolved_request,
):
    script = render_sbatch(valid_resolved_request)
    spec = _spec(valid_resolved_request, script)
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)
    stage_submission(tmp_path, key, script)
    calls: list[list[str]] = []
    first = submission_transaction(
        tmp_path,
        key,
        spec,
        sbatch_runner=_runner(calls),
        manifest_verifier=lambda *args: None,
        now=NOW,
    )
    second = submission_transaction(
        tmp_path,
        key,
        spec,
        sbatch_runner=_runner(calls, output="99999\n"),
        manifest_verifier=lambda *args: None,
        now=NOW,
    )

    assert second == first
    assert len(calls) == 1


def test_concurrent_transaction_invocations_submit_only_once(
    tmp_path,
    valid_resolved_request,
):
    script = render_sbatch(valid_resolved_request)
    spec = _spec(valid_resolved_request, script)
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)
    stage_submission(tmp_path, key, script)
    calls: list[list[str]] = []
    barrier = threading.Barrier(2)
    results: list[SubmissionReceipt] = []

    def run_sbatch(argv, **kwargs):
        calls.append(list(argv))
        time.sleep(0.05)
        return subprocess.CompletedProcess(argv, 0, "12345\n", "")

    def invoke():
        barrier.wait()
        results.append(
            submission_transaction(
                tmp_path,
                key,
                spec,
                sbatch_runner=run_sbatch,
                manifest_verifier=lambda *args: None,
                now=NOW,
            )
        )

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1
    assert results == [results[0], results[0]]


@pytest.mark.parametrize(
    "output",
    [
        "",
        "12345;cluster\n",
        "12345_2\n",
        "12345 extra\n",
        "0\n",
        "-1\n",
        "1\n2\n",
        "x" * 1000,
    ],
)
def test_transaction_rejects_malformed_sbatch_output_without_retrying(
    tmp_path,
    valid_resolved_request,
    output,
):
    script = render_sbatch(valid_resolved_request)
    spec = _spec(valid_resolved_request, script)
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)
    stage_submission(tmp_path, key, script)
    calls: list[list[str]] = []

    with pytest.raises(SubmissionError):
        submission_transaction(
            tmp_path,
            key,
            spec,
            sbatch_runner=_runner(calls, output=output),
            manifest_verifier=lambda *args: None,
            now=NOW,
        )
    with pytest.raises(SubmissionError):
        submission_transaction(
            tmp_path,
            key,
            spec,
            sbatch_runner=_runner(calls, output="99999\n"),
            manifest_verifier=lambda *args: None,
            now=NOW,
        )
    assert len([argv for argv in calls if argv[0] == "sbatch"]) == 1


def test_transaction_rejects_failed_sbatch_without_receipt(
    tmp_path,
    valid_resolved_request,
):
    script = render_sbatch(valid_resolved_request)
    spec = _spec(valid_resolved_request, script)
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)
    stage_submission(tmp_path, key, script)
    calls: list[list[str]] = []

    with pytest.raises(SubmissionError, match="failed") as raised:
        submission_transaction(
            tmp_path,
            key,
            spec,
            sbatch_runner=_runner(calls, returncode=1),
            manifest_verifier=lambda *args: None,
            now=NOW,
        )

    assert "private" not in str(raised.value)
    assert not (tmp_path / key / "receipt.json").exists()


def test_transaction_rejects_tampered_script_receipt_and_spec_mismatch(
    tmp_path,
    valid_resolved_request,
):
    script = render_sbatch(valid_resolved_request)
    spec = _spec(valid_resolved_request, script)
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)
    stage_submission(tmp_path, key, script)
    script_path = tmp_path / key / "request.sbatch"
    script_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(SubmissionError, match="script"):
        submission_transaction(
            tmp_path,
            key,
            spec,
            sbatch_runner=_runner([]),
            manifest_verifier=lambda *args: None,
            now=NOW,
        )

    stage_submission(tmp_path, key, script)
    receipt = submission_transaction(
        tmp_path,
        key,
        spec,
        sbatch_runner=_runner([]),
        manifest_verifier=lambda *args: None,
        now=NOW,
    )
    receipt_path = tmp_path / key / "receipt.json"
    payload = json.loads(receipt_path.read_text())
    payload["operator"] = "other"
    receipt_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    receipt_path.chmod(0o600)
    with pytest.raises(SubmissionError, match="receipt"):
        submission_transaction(
            tmp_path,
            key,
            spec,
            sbatch_runner=_runner([]),
            manifest_verifier=lambda *args: None,
            now=NOW,
        )

    receipt_path.write_text(receipt.canonical_json(), encoding="utf-8")
    receipt_path.chmod(0o600)
    with pytest.raises(SubmissionError, match="receipt"):
        submission_transaction(
            tmp_path,
            key,
            replace(spec, operator="other"),
            sbatch_runner=_runner([]),
            manifest_verifier=lambda *args: None,
            now=NOW,
        )


def test_failed_receipt_persistence_leaves_durable_ambiguous_intent_and_never_resubmits(
    tmp_path,
    valid_resolved_request,
    monkeypatch,
):
    script = render_sbatch(valid_resolved_request)
    spec = _spec(valid_resolved_request, script)
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)
    stage_submission(tmp_path, key, script)
    calls: list[list[str]] = []
    import edullm.slurm as module

    real_publish = module._publish_private

    def fail_receipt(path, content):
        if path.name == "receipt.json":
            raise OSError("private persistence diagnostics")
        return real_publish(path, content)

    monkeypatch.setattr(module, "_publish_private", fail_receipt)
    with pytest.raises(SubmissionError, match="receipt") as raised:
        submission_transaction(
            tmp_path,
            key,
            spec,
            sbatch_runner=_runner(calls),
            manifest_verifier=lambda *args: None,
            now=NOW,
        )
    assert "private" not in str(raised.value)
    assert (tmp_path / key / "intent.json").exists()

    monkeypatch.setattr(module, "_publish_private", real_publish)
    with pytest.raises(SubmissionError):
        submission_transaction(
            tmp_path,
            key,
            spec,
            sbatch_runner=_runner(calls, output="99999\n"),
            manifest_verifier=lambda *args: None,
            now=NOW,
        )
    assert len([argv for argv in calls if argv[0] == "sbatch"]) == 1


def test_failed_receipt_persistence_recovers_one_exact_authoritative_job_without_resubmit(
    tmp_path,
    valid_resolved_request,
    monkeypatch,
):
    script = render_sbatch(valid_resolved_request)
    spec = _spec(valid_resolved_request, script)
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)
    stage_submission(tmp_path, key, script)
    calls: list[list[str]] = []
    import edullm.slurm as module

    real_publish = module._publish_private

    def fail_receipt(path, content):
        if path.name == "receipt.json":
            raise OSError("private persistence diagnostics")
        return real_publish(path, content)

    monkeypatch.setattr(module, "_publish_private", fail_receipt)
    with pytest.raises(SubmissionError, match="receipt"):
        submission_transaction(
            tmp_path,
            key,
            spec,
            sbatch_runner=_runner(calls),
            manifest_verifier=lambda *args: None,
            now=NOW,
        )

    monkeypatch.setattr(module, "_publish_private", real_publish)

    def discover(argv, **kwargs):
        calls.append(list(argv))
        assert argv[0] == "sacct"
        identity = f"edullm-{key}"
        output = f"12345|{identity}|PENDING|{REMOTE_USER}|edullm:{key}|" "2026-07-23T13:00:00\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    receipt = submission_transaction(
        tmp_path,
        key,
        spec,
        sbatch_runner=discover,
        manifest_verifier=lambda *args: None,
        now=NOW,
    )

    assert receipt.slurm_job_id == "12345"
    assert receipt.submitted_at == NOW
    assert len([argv for argv in calls if argv[0] == "sbatch"]) == 1
    assert len([argv for argv in calls if argv[0] == "sacct"]) == 1
    assert (
        parse_submission_receipt((tmp_path / key / "receipt.json").read_text(encoding="utf-8"))
        == receipt
    )


def test_recovery_uses_trusted_orcd_user_separately_from_github_operator(
    tmp_path,
    valid_resolved_request,
):
    script = render_sbatch(valid_resolved_request)
    spec = replace(
        _spec(valid_resolved_request, script),
        operator="github-operator",
        remote_user="orcd-user",
    )
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)
    stage_submission(tmp_path, key, script)
    intent = tmp_path / key / "intent.json"
    intent.write_text(spec.canonical_json(), encoding="utf-8")
    intent.chmod(0o600)
    calls: list[list[str]] = []

    def discover(argv, **kwargs):
        calls.append(list(argv))
        identity = f"edullm-{key}"
        output = f"12345|{identity}|PENDING|orcd-user|edullm:{key}|" "2026-07-23T13:00:00\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    receipt = submission_transaction(
        tmp_path,
        key,
        spec,
        sbatch_runner=discover,
        manifest_verifier=lambda *args: None,
        remote_user_getter=lambda: "orcd-user",
        now=NOW,
    )

    assert receipt.operator == "github-operator"
    assert receipt.remote_user == "orcd-user"
    assert len([argv for argv in calls if argv[0] == "sbatch"]) == 0
    assert len([argv for argv in calls if argv[0] == "sacct"]) == 1


def test_recovery_identity_switches_and_missing_identity_fail_without_sbatch(
    tmp_path,
    valid_resolved_request,
):
    script = render_sbatch(valid_resolved_request)
    spec = replace(
        _spec(valid_resolved_request, script),
        operator="github-operator",
        remote_user="orcd-user",
    )
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)
    calls: list[list[str]] = []

    def unexpected_runner(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "99999\n", "")

    for identity_name, authenticated_user in (("wrong-user", "other-user"), ("missing", "")):
        state_root = tmp_path / identity_name
        stage_submission(state_root, key, script)
        intent = state_root / key / "intent.json"
        intent.write_text(spec.canonical_json(), encoding="utf-8")
        intent.chmod(0o600)

        def remote_user_getter() -> str:
            return authenticated_user

        with pytest.raises(SubmissionError, match="remote user"):
            submission_transaction(
                state_root,
                key,
                spec,
                sbatch_runner=unexpected_runner,
                remote_user_getter=remote_user_getter,
                now=NOW,
            )

    switched_user_root = tmp_path / "switched-user"
    stage_submission(switched_user_root, key, script)
    switched_intent = switched_user_root / key / "intent.json"
    switched_intent.write_text(spec.canonical_json(), encoding="utf-8")
    switched_intent.chmod(0o600)
    with pytest.raises(SubmissionError, match="intent"):
        submission_transaction(
            switched_user_root,
            key,
            replace(spec, remote_user="other-user"),
            sbatch_runner=unexpected_runner,
            remote_user_getter=lambda: "other-user",
            now=NOW,
        )

    mismatched_receipt_root = tmp_path / "mismatched-receipt"
    stage_submission(mismatched_receipt_root, key, script)
    mismatched_receipt = SubmissionReceipt(
        issue=spec.issue,
        request_digest=spec.request_digest,
        attempt_number=spec.attempt_number,
        operator=spec.operator,
        remote_user="other-user",
        script_sha256=spec.script_sha256,
        manifest_sha256=spec.manifest_sha256,
        slurm_job_id="12345",
        log_path=str(tmp_path / "logs/issue-42-attempt-1-12345.log"),
        submitted_at=NOW,
    )
    receipt_path = mismatched_receipt_root / key / "receipt.json"
    receipt_path.write_text(mismatched_receipt.canonical_json(), encoding="utf-8")
    receipt_path.chmod(0o600)
    with pytest.raises(SubmissionError, match="receipt"):
        submission_transaction(
            mismatched_receipt_root,
            key,
            spec,
            sbatch_runner=unexpected_runner,
            remote_user_getter=lambda: "orcd-user",
            now=NOW,
        )

    switched_operator_root = tmp_path / "switched-operator"
    stage_submission(switched_operator_root, key, script)
    switched_operator_intent = switched_operator_root / key / "intent.json"
    switched_operator_intent.write_text(spec.canonical_json(), encoding="utf-8")
    switched_operator_intent.chmod(0o600)
    with pytest.raises(SubmissionError, match="key"):
        submission_transaction(
            switched_operator_root,
            key,
            replace(spec, operator="other-operator"),
            sbatch_runner=unexpected_runner,
            remote_user_getter=lambda: "orcd-user",
            now=NOW,
        )

    assert calls == []


@pytest.mark.parametrize(
    "output",
    [
        "",
        (
            "12345|{identity}|PENDING|{user}|edullm:{key}|2026-07-23T13:00:00\n"
            "12346|{identity}|PENDING|{user}|edullm:{key}|2026-07-23T13:00:01\n"
        ),
        "malformed\n",
        "12345|wrong-name|PENDING|{user}|edullm:{key}|2026-07-23T13:00:00\n",
        "12345|{identity}|PENDING|other|edullm:{key}|2026-07-23T13:00:00\n",
        "12345|{identity}|PENDING|{user}|wrong-comment|2026-07-23T13:00:00\n",
        "12345.batch|{identity}|PENDING|{user}|edullm:{key}|2026-07-23T13:00:00\n",
    ],
)
def test_ambiguous_intent_never_resubmits_for_zero_multiple_or_mismatched_evidence(
    tmp_path,
    valid_resolved_request,
    output,
):
    script = render_sbatch(valid_resolved_request)
    spec = _spec(valid_resolved_request, script)
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)
    stage_submission(tmp_path, key, script)
    intent = tmp_path / key / "intent.json"
    intent.write_text(spec.canonical_json(), encoding="utf-8")
    intent.chmod(0o600)
    calls: list[list[str]] = []
    rendered_output = output.format(identity=f"edullm-{key}", key=key, user=REMOTE_USER)

    def discover(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, rendered_output, "")

    with pytest.raises(SubmissionError):
        submission_transaction(
            tmp_path,
            key,
            spec,
            sbatch_runner=discover,
            manifest_verifier=lambda *args: None,
            now=NOW,
        )

    assert calls and all(argv[0] == "sacct" for argv in calls)
    assert not (tmp_path / key / "receipt.json").exists()


def test_stage_submission_is_private_atomic_and_rejects_symlinks(
    tmp_path,
    valid_resolved_request,
):
    script = render_sbatch(valid_resolved_request)
    spec = _spec(valid_resolved_request, script)
    key = build_submission_key(spec.issue, spec.request_digest, 1, spec.operator)

    path = stage_submission(tmp_path, key, script)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    target = tmp_path / "outside"
    target.write_text("unchanged", encoding="utf-8")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(SubmissionError):
        stage_submission(tmp_path, key, script)
    assert target.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text + "\n",
        lambda text: text.replace('"operator":"operator"', '"operator":"Operator"'),
        lambda text: text.replace('"slurm_job_id":"12345"', '"slurm_job_id":"12345;cluster"'),
        lambda text: text.replace('"issue":42', '"issue":0'),
        lambda text: text.replace('"attempt_number":1', '"attempt_number":true'),
        lambda text: text.replace('{"attempt_number"', '{"extra":1,"attempt_number"'),
    ],
)
def test_receipt_parser_rejects_noncanonical_or_malformed_records(mutator):
    receipt = SubmissionReceipt(
        issue=42,
        request_digest="a" * 64,
        attempt_number=1,
        operator="operator",
        remote_user=REMOTE_USER,
        script_sha256="b" * 64,
        manifest_sha256="c" * 64,
        slurm_job_id="12345",
        log_path="/home/operator/orcd/scratch/edullm/logs/issue-42-attempt-1-12345.log",
        submitted_at=NOW,
    )

    with pytest.raises(SubmissionError):
        parse_submission_receipt(mutator(receipt.canonical_json()))
