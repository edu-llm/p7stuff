import argparse
import json
import platform
import tempfile
from pathlib import Path
from typing import Any

import requests
import torch


def check_writable_directory(path: Path) -> dict[str, object]:
    """
    Verify that a directory can be created and written.

    :param path: The directory to check.

    :returns: The directory path and its writable status.
    """
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path, delete=True) as handle:
        handle.write(b"edullm")
        handle.flush()
    return {"path": str(path), "writable": True}


def check_cuda() -> dict[str, object]:
    """
    Collect PyTorch and CUDA availability information.

    :returns: CUDA device information and the installed PyTorch versions.
    """
    available = torch.cuda.is_available()
    return {
        "available": available,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if available else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def check_wandb() -> dict[str, object]:
    """
    Check whether W&B imports and its API endpoint is reachable.

    :returns: W&B importability and API reachability statuses.
    """
    try:
        import wandb  # noqa: F401

        importable = True
    except ImportError:
        importable = False

    try:
        response = requests.get("https://api.wandb.ai", timeout=10)
        reachable = response.status_code < 500
    except requests.RequestException:
        reachable = False

    return {"importable": importable, "reachable": reachable}


def check_olmo_core() -> dict[str, object]:
    """
    Check whether OLMo-core imports and report its version.

    :returns: OLMo-core importability and version information.
    """
    try:
        import olmo_core  # noqa: F401
        from olmo_core.version import VERSION
    except ImportError:
        return {"importable": False, "version": None}
    return {"importable": True, "version": VERSION}


def build_report(scratch: Path) -> dict[str, Any]:
    """
    Build the ORCD environment probe report.

    :param scratch: The Engaging Scratch directory to verify.

    :returns: Python, CUDA, W&B, OLMo-core, and scratch probe results.
    """
    return {
        "python": {"version": platform.python_version()},
        "cuda": check_cuda(),
        "wandb": check_wandb(),
        "olmo_core": check_olmo_core(),
        "scratch": check_writable_directory(scratch),
    }


def main() -> None:
    """Write the environment probe report and require CUDA availability."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.scratch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["cuda"]["available"]:
        raise SystemExit("CUDA is not available")


if __name__ == "__main__":
    main()
