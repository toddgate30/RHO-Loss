import gzip
import hashlib
from pathlib import Path

import pytest

from reproduction.data import _digit_resources, _download_verified, HTTPSRedirectsOnly


def digit_fixture(tmp_path):
    payload = b"valid-extracted-resource" * 5
    raw_dir = tmp_path / "indices_MNIST" / "raw"
    raw_dir.mkdir(parents=True)
    archive = raw_dir / "resource.gz"
    with gzip.open(archive, "wb") as stream:
        stream.write(payload)
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    cls = type(
        "indices_MNIST",
        (),
        {"resources": [("resource.gz", md5)], "mirrors": ["https://example.invalid/"]},
    )
    return cls, raw_dir, payload


def test_corrupt_extracted_data_is_repaired_from_verified_archive(tmp_path):
    cls, raw_dir, payload = digit_fixture(tmp_path)
    (raw_dir / "resource").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Missing/corrupt extracted"):
        _digit_resources(cls, tmp_path, allow_download=False)
    _digit_resources(cls, tmp_path, allow_download=True)
    assert (raw_dir / "resource").read_bytes() == payload
    _digit_resources(cls, tmp_path, allow_download=False)


def test_missing_or_corrupt_archive_never_downloads_in_verify_only(
    tmp_path, monkeypatch
):
    cls, raw_dir, _ = digit_fixture(tmp_path)
    (raw_dir / "resource.gz").write_bytes(b"bad archive")
    monkeypatch.setattr(
        "reproduction.data._download_verified",
        lambda *a, **k: pytest.fail("No network"),
    )
    with pytest.raises(RuntimeError, match="verified archive"):
        _digit_resources(cls, tmp_path, allow_download=False)


def test_downloads_and_redirects_cannot_downgrade_tls(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        _download_verified(
            "http://example.invalid/data", tmp_path / "archive", "checksum"
        )
    with pytest.raises(RuntimeError, match="HTTPS"):
        HTTPSRedirectsOnly().redirect_request(
            None, None, 302, "", {}, "http://example.invalid/data"
        )
