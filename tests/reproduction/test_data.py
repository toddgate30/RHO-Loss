import gzip
import hashlib
import io
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


@pytest.mark.parametrize("known_size", [True, False])
def test_download_progress_and_cached_reuse(tmp_path, monkeypatch, capsys, known_size):
    payload = b"dataset chunk" * 100000
    path = tmp_path / "dataset.gz"
    md5 = hashlib.md5(payload).hexdigest()
    bars = []

    class Bar:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.downloaded = 0
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, count):
            self.downloaded += count

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))} if known_size else {}

    class Opener:
        def open(self, *args, **kwargs):
            return Response(payload)

    monkeypatch.setattr("reproduction.data.build_opener", lambda *args: Opener())
    monkeypatch.setattr("reproduction.data.tqdm", Bar)
    _download_verified("https://example.invalid/dataset", path, md5)
    assert path.read_bytes() == payload
    assert not path.with_name(path.name + ".part").exists()
    assert bars[0].total == (len(payload) if known_size else None)
    assert bars[0].downloaded == len(payload)
    monkeypatch.setattr(
        "reproduction.data.build_opener",
        lambda *args: pytest.fail("Cached file must not download"),
    )
    _download_verified("https://example.invalid/dataset", path, md5)
    output = capsys.readouterr().out
    assert "Downloading dataset.gz" in output
    assert "Download verified" in output
    assert "Using verified archive" in output


def test_progress_does_not_bypass_failed_download_checksum(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        headers = {"Content-Length": "3"}

    class Opener:
        def open(self, *args, **kwargs):
            return Response(b"bad")

    monkeypatch.setattr("reproduction.data.build_opener", lambda *args: Opener())
    path = tmp_path / "dataset.gz"
    with pytest.raises(RuntimeError, match="checksum failed"):
        _download_verified(
            "https://example.invalid/dataset",
            path,
            hashlib.md5(b"expected").hexdigest(),
        )
    assert not path.exists()
    assert not path.with_name(path.name + ".part").exists()
