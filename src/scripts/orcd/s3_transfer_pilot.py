import argparse
import hashlib
import hmac
import json
import math
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

_TRANSFER_CHUNK_BYTES = 1024 * 1024
_TRANSFER_TIMEOUT_SECONDS = 60


def sha256_file(path: Path) -> str:
    """
    Calculate the SHA-256 digest of a file.

    :param path: The file to digest.
    :returns: The lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_TRANSFER_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> tuple[int, float]:
    """
    Download and verify a size-bounded object through a presigned URL.

    The destination is replaced only after the complete object has passed its
    size and digest checks.

    :param url: The short-lived presigned GET URL.
    :param destination: The final path for the verified object.
    :param expected_sha256: The expected lowercase hexadecimal SHA-256 digest.
    :param max_bytes: The maximum permitted object size.
    :returns: The downloaded byte count and elapsed seconds.
    :raises RuntimeError: If the transfer or a verification check fails.
    """
    if max_bytes <= 0:
        raise RuntimeError("S3 GET requires a positive configured size limit")

    start = time.monotonic()
    temporary_path: Path | None = None
    try:
        with requests.get(url, stream=True, timeout=_TRANSFER_TIMEOUT_SECONDS) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"S3 GET failed with HTTP {response.status_code}")

            content_length_value = response.headers.get("Content-Length")
            if content_length_value is not None:
                try:
                    content_length = int(content_length_value)
                except (TypeError, ValueError):
                    raise RuntimeError("S3 GET returned an invalid Content-Length") from None
                if content_length < 0:
                    raise RuntimeError("S3 GET returned an invalid Content-Length")
                if content_length > max_bytes:
                    raise RuntimeError("S3 GET exceeds the configured size limit")

            written = 0
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                for chunk in response.iter_content(_TRANSFER_CHUNK_BYTES):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise RuntimeError("S3 GET exceeded the configured size limit")
                    handle.write(chunk)

        observed_sha256 = sha256_file(temporary_path)
        if not hmac.compare_digest(observed_sha256, expected_sha256.lower()):
            raise RuntimeError("S3 GET digest mismatch")

        temporary_path.replace(destination)
        temporary_path = None
    except requests.RequestException as error:
        raise RuntimeError(f"S3 GET failed: {type(error).__name__}") from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination.stat().st_size, time.monotonic() - start


def upload(url: str, source: Path) -> tuple[int, float]:
    """
    Upload a local result through a presigned URL.

    :param url: The short-lived presigned PUT URL.
    :param source: The selected checkpoint result to upload.
    :returns: The uploaded byte count and elapsed seconds.
    :raises RuntimeError: If the transfer fails.
    """
    source_size = source.stat().st_size
    start = time.monotonic()
    try:
        with source.open("rb") as handle:
            with requests.put(
                url,
                data=handle,
                headers={"Content-Length": str(source_size)},
                timeout=_TRANSFER_TIMEOUT_SECONDS,
            ) as response:
                if response.status_code >= 400:
                    raise RuntimeError(f"S3 PUT failed with HTTP {response.status_code}")
    except requests.RequestException as error:
        raise RuntimeError(f"S3 PUT failed: {type(error).__name__}") from None
    return source_size, time.monotonic() - start


def _load_urls(path: Path) -> dict[str, str]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise SystemExit("URL file must not be readable by group or other users; use mode 0600")

    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise SystemExit("Unable to read S3 URL file safely") from None

    if not isinstance(value, dict) or not all(
        isinstance(value.get(key), str) and value[key] for key in ("download_url", "upload_url")
    ):
        raise SystemExit("URL file must contain download_url and upload_url strings")
    return {"download_url": value["download_url"], "upload_url": value["upload_url"]}


def _safe_elapsed(seconds: float) -> float:
    if not math.isfinite(seconds) or seconds < 0:
        return 0.0
    return seconds


def _throughput_mib_per_second(byte_count: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return byte_count / (1024**2) / seconds


def main() -> None:
    """Run the bounded presigned-URL transfer pilot and print a secret-free report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--max-download-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()

    urls = _load_urls(args.url_file)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    local = args.work_dir / "s3-pilot-input.bin"
    download_bytes, raw_download_seconds = download(
        urls["download_url"],
        local,
        expected_sha256=args.expected_sha256,
        max_bytes=args.max_download_bytes,
    )
    upload_bytes, raw_upload_seconds = upload(urls["upload_url"], args.result_file)
    download_seconds = _safe_elapsed(raw_download_seconds)
    upload_seconds = _safe_elapsed(raw_upload_seconds)
    report = {
        "download_bytes": download_bytes,
        "upload_bytes": upload_bytes,
        "sha256": sha256_file(local),
        "download_seconds": download_seconds,
        "upload_seconds": upload_seconds,
        "download_mib_per_second": _throughput_mib_per_second(download_bytes, download_seconds),
        "upload_mib_per_second": _throughput_mib_per_second(upload_bytes, upload_seconds),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
