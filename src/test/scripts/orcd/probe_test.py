import importlib.util
from pathlib import Path

import pytest

from olmo_core.version import VERSION


def load_probe_module():
    path = Path("src/scripts/orcd/probe.py")
    spec = importlib.util.spec_from_file_location("orcd_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_read_probe(tmp_path):
    probe = load_probe_module()
    result = probe.check_writable_directory(tmp_path)
    assert result == {"path": str(tmp_path), "writable": True}


def test_olmo_core_probe_reports_import_and_version():
    probe = load_probe_module()
    assert probe.check_olmo_core() == {"importable": True, "version": VERSION}


def test_report_contains_required_sections(tmp_path, monkeypatch):
    probe = load_probe_module()
    monkeypatch.setattr(probe, "check_cuda", lambda: {"available": False})
    monkeypatch.setattr(probe, "check_wandb", lambda: {"importable": True, "reachable": False})
    monkeypatch.setattr(
        probe,
        "check_olmo_core",
        lambda: {"importable": True, "version": "test-version"},
    )
    report = probe.build_report(tmp_path)
    assert set(report) == {"python", "cuda", "wandb", "olmo_core", "scratch"}


@pytest.mark.parametrize("script_name", ["setup_env.sbatch", "probe.sbatch"])
def test_slurm_jobs_route_logs_to_scratch_before_clean_check(script_name):
    script = Path("src/scripts/orcd") / script_name
    contents = script.read_text(encoding="utf-8")

    scratch_requirement = ': "${EDULLM_SCRATCH:?export EDULLM_SCRATCH before sbatch}"'
    create_log_directory = 'mkdir -p "$EDULLM_SCRATCH/logs"'
    redirect_logs = 'exec >"$EDULLM_SCRATCH/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log" 2>&1'
    clean_check = "status --porcelain"

    assert "#SBATCH -o /dev/null" in contents
    assert "#SBATCH -e /dev/null" in contents
    assert (
        contents.index(scratch_requirement)
        < contents.index(create_log_directory)
        < contents.index(redirect_logs)
        < contents.index(clean_check)
    )
