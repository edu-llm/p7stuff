import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


class ResponseDouble:
    def __init__(self, *, status_code=200, headers=None, chunks=()):
        self.status_code = status_code
        self.headers = headers or {}
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def iter_content(self, chunk_size):
        assert chunk_size == 1024 * 1024
        yield from self.chunks


def load_module():
    path = Path("src/scripts/orcd/s3_transfer_pilot.py")
    spec = importlib.util.spec_from_file_location("s3_transfer_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_file_digest(tmp_path):
    module = load_module()
    path = tmp_path / "data.bin"
    path.write_bytes(b"edullm")
    assert module.sha256_file(path) == hashlib.sha256(b"edullm").hexdigest()


def test_sanitizes_presigned_get_url_errors(tmp_path, monkeypatch):
    module = load_module()
    secret_url = "https://example.invalid/object?X-Amz-Signature=GET_SECRET"
    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(module.requests.ConnectionError(secret_url)),
    )

    with pytest.raises(RuntimeError) as exc:
        module.download(
            secret_url,
            tmp_path / "unused",
            expected_sha256="0" * 64,
            max_bytes=1024,
        )

    assert "S3 GET failed" in str(exc.value)
    assert "GET_SECRET" not in str(exc.value)
    assert not list(tmp_path.iterdir())


def test_sanitizes_presigned_put_url_errors(tmp_path, monkeypatch):
    module = load_module()
    source = tmp_path / "result.json"
    source.write_text("{}", encoding="utf-8")
    secret_url = "https://example.invalid/object?X-Amz-Signature=PUT_SECRET"
    monkeypatch.setattr(
        module.requests,
        "put",
        lambda *args, **kwargs: (_ for _ in ()).throw(module.requests.ConnectionError(secret_url)),
    )

    with pytest.raises(RuntimeError) as exc:
        module.upload(secret_url, source)

    assert "S3 PUT failed" in str(exc.value)
    assert "PUT_SECRET" not in str(exc.value)


def test_download_rejects_declared_oversize_without_partial_file(tmp_path, monkeypatch):
    module = load_module()
    response = ResponseDouble(headers={"Content-Length": "9"})
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="configured size limit"):
        module.download(
            "https://example.invalid/download",
            tmp_path / "download.bin",
            expected_sha256="0" * 64,
            max_bytes=8,
        )

    assert not list(tmp_path.iterdir())


def test_download_rejects_streamed_oversize_without_partial_file(tmp_path, monkeypatch):
    module = load_module()
    response = ResponseDouble(chunks=(b"1234", b"5"))
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="configured size limit"):
        module.download(
            "https://example.invalid/download",
            tmp_path / "download.bin",
            expected_sha256="0" * 64,
            max_bytes=4,
        )

    assert not list(tmp_path.iterdir())


def test_download_rejects_digest_mismatch_without_partial_file(tmp_path, monkeypatch):
    module = load_module()
    response = ResponseDouble(chunks=(b"unexpected",))
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="digest mismatch"):
        module.download(
            "https://example.invalid/download",
            tmp_path / "download.bin",
            expected_sha256="0" * 64,
            max_bytes=1024,
        )

    assert not list(tmp_path.iterdir())


def test_download_writes_only_verified_content(tmp_path, monkeypatch):
    module = load_module()
    content = b"edullm"
    response = ResponseDouble(
        headers={"Content-Length": str(len(content))},
        chunks=(content[:3], content[3:]),
    )
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)
    destination = tmp_path / "download.bin"

    byte_count, elapsed = module.download(
        "https://example.invalid/download",
        destination,
        expected_sha256=hashlib.sha256(content).hexdigest(),
        max_bytes=len(content),
    )

    assert byte_count == len(content)
    assert elapsed >= 0
    assert destination.read_bytes() == content
    assert list(tmp_path.iterdir()) == [destination]


def test_upload_reports_source_size(tmp_path, monkeypatch):
    module = load_module()
    source = tmp_path / "step20-config.json"
    source.write_bytes(b"config")
    uploaded = bytearray()

    def put_double(url, *, data, headers, timeout):
        assert url == "https://example.invalid/upload"
        assert headers == {"Content-Length": str(source.stat().st_size)}
        assert timeout == 60
        uploaded.extend(data.read())
        return ResponseDouble()

    monkeypatch.setattr(module.requests, "put", put_double)

    byte_count, elapsed = module.upload("https://example.invalid/upload", source)

    assert byte_count == source.stat().st_size
    assert elapsed >= 0
    assert bytes(uploaded) == source.read_bytes()


def test_main_rejects_url_file_readable_by_group(tmp_path, monkeypatch):
    module = load_module()
    url_file = tmp_path / "urls.json"
    url_file.write_text("{}", encoding="utf-8")
    url_file.chmod(0o640)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "s3_transfer_pilot.py",
            "--url-file",
            str(url_file),
            "--work-dir",
            str(tmp_path / "work"),
            "--result-file",
            str(tmp_path / "result"),
            "--expected-sha256",
            "0" * 64,
        ],
    )

    with pytest.raises(SystemExit, match="must not be readable by group or other users"):
        module.main()


def test_main_prints_secret_free_report_and_handles_invalid_timing(tmp_path, monkeypatch, capsys):
    module = load_module()
    content = b"public token shard"
    expected_sha256 = hashlib.sha256(content).hexdigest()
    url_file = tmp_path / "urls.json"
    get_secret = "https://example.invalid/get?X-Amz-Signature=GET_SECRET"
    put_secret = "https://example.invalid/put?X-Amz-Signature=PUT_SECRET"
    url_file.write_text(
        json.dumps({"download_url": get_secret, "upload_url": put_secret}),
        encoding="utf-8",
    )
    url_file.chmod(0o600)
    result_file = tmp_path / "step20" / "config.json"
    result_file.parent.mkdir()
    result_file.write_text("{}", encoding="utf-8")

    def download_double(url, destination, *, expected_sha256, max_bytes):
        assert url == get_secret
        assert expected_sha256 == hashlib.sha256(content).hexdigest()
        assert max_bytes == 1024
        destination.write_bytes(content)
        return len(content), 0.0

    def upload_double(url, source):
        assert url == put_secret
        assert source == result_file
        return source.stat().st_size, float("nan")

    monkeypatch.setattr(module, "download", download_double)
    monkeypatch.setattr(module, "upload", upload_double)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "s3_transfer_pilot.py",
            "--url-file",
            str(url_file),
            "--work-dir",
            str(tmp_path / "work"),
            "--result-file",
            str(result_file),
            "--expected-sha256",
            expected_sha256,
            "--max-download-bytes",
            "1024",
        ],
    )

    module.main()

    output = capsys.readouterr().out
    report = json.loads(output)
    assert report == {
        "download_bytes": len(content),
        "upload_bytes": result_file.stat().st_size,
        "sha256": expected_sha256,
        "download_seconds": 0.0,
        "upload_seconds": 0.0,
        "download_mib_per_second": None,
        "upload_mib_per_second": None,
    }
    assert "GET_SECRET" not in output
    assert "PUT_SECRET" not in output
    assert "NaN" not in output
