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
        data_manifest="/orcd/pool/edullm/manifests/skill-dag-v1.json",
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
                "allowed_options": {
                    "seed": {
                        "type": "integer",
                        "min": 0,
                        "max": 2147483647,
                        "required": True,
                        "request_field": "seed",
                    }
                },
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


@pytest.fixture
def reviewed_github():
    from edullm.github import ReviewResult

    class ReviewedGitHub:
        @staticmethod
        def reviewed_commit(
            sha,
            *,
            script_path,
            allowed_reviewers,
            required_checks,
        ):
            return ReviewResult(True, 7, "approved")

    return ReviewedGitHub()
