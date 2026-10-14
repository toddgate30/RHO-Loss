"""Prepare the repository's exact dataset layout on login nodes, before Slurm."""

import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class HTTPSRedirectsOnly(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise RuntimeError("Refusing a dataset download redirect away from HTTPS.")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _download_verified(url, path, md5):
    """Preserve TLS/MD5 checks without torchvision's HTTPS-to-HTTP fallback."""
    from torchvision.datasets.utils import check_integrity

    if urlparse(url).scheme != "https":
        raise ValueError("Dataset downloads must use HTTPS.")
    if check_integrity(str(path), md5):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        opener = build_opener(HTTPSRedirectsOnly())
        with opener.open(
            Request(url, headers={"User-Agent": "RHO-Loss dataset preparation"}),
            timeout=60,
        ) as response:
            with open(temporary, "wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
        if not check_integrity(str(temporary), md5):
            raise RuntimeError(f"Downloaded archive checksum failed: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digit_resources(cls, root, allow_download):
    from torchvision.datasets.utils import check_integrity, extract_archive

    resources = (
        cls.resources["test"] if cls.__name__ == "indices_QMNIST" else cls.resources
    )
    raw_dir = root / cls.__name__ / "raw"
    files = []
    for resource, md5 in resources:
        filename = resource.rsplit("/", 1)[-1]
        archive = raw_dir / filename
        raw = raw_dir / filename.removesuffix(".gz")
        if not check_integrity(str(archive), md5):
            if not allow_download:
                raise RuntimeError(
                    f"Missing/corrupt verified archive: {archive}. Prepare on a login node."
                )
            raw_dir.mkdir(parents=True, exist_ok=True)
            if cls.__name__ == "indices_QMNIST":
                urls = [resource]
            else:
                # Prefer the repository's HTTPS mirror; preserve MD5 verification.
                urls = [
                    mirror + filename
                    for mirror in cls.mirrors
                    if mirror.startswith("https://")
                ]
            for index, url in enumerate(urls):
                try:
                    _download_verified(url, archive, md5)
                    break
                except Exception:
                    if index == len(urls) - 1:
                        raise
            if not check_integrity(str(archive), md5):
                raise RuntimeError(f"Archive verification failed: {archive}")
        # Check extracted files against the checksum-verified compressed source.
        expected = hashlib.sha256()
        with gzip.open(archive, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                expected.update(chunk)
        if not raw.is_file() or sha256(raw) != expected.hexdigest():
            if not allow_download:
                raise RuntimeError(
                    f"Missing/corrupt extracted file: {raw}. Prepare on a login node."
                )
            extract_archive(str(archive), str(raw_dir))
        if sha256(raw) != expected.hexdigest():
            raise RuntimeError(f"Extraction verification failed: {raw}")
        files.extend((archive, raw))
    return files


def _prepare(dataset, root, allow_download):
    from src.datamodules.datasets import sequence_datasets as native
    from torchvision.datasets.utils import check_integrity, extract_archive

    def no_download(*args, **kwargs):
        raise RuntimeError(
            "Offline loader attempted download; prepare data on a login node."
        )

    if dataset in ("cifar10", "cifar100"):
        cls = (
            native.indices_CIFAR10 if dataset == "cifar10" else native.indices_CIFAR100
        )
        expected = [(name, md5) for name, md5 in cls.train_list + cls.test_list]
        expected.append((cls.meta["filename"], cls.meta["md5"]))
        files = [root / cls.base_folder / name for name, _ in expected]
        if not all(
            check_integrity(str(path), md5) for path, (_, md5) in zip(files, expected)
        ):
            if not allow_download:
                raise RuntimeError(
                    f"Incomplete/corrupt {dataset} data. Prepare on a login node."
                )
            archive = root / cls.filename
            _download_verified(cls.url, archive, cls.tgz_md5)
            extract_archive(str(archive), str(root))
        if not all(
            check_integrity(str(path), md5) for path, (_, md5) in zip(files, expected)
        ):
            raise RuntimeError(f"Dataset integrity failed: {dataset}")
        with patch.object(
            native, "download_and_extract_archive", side_effect=no_download
        ):
            train, test = cls(str(root), train=True), cls(str(root), train=False)
        lengths = {"train": len(train), "test": len(test)}
        if lengths != {"train": 50000, "test": 10000}:
            raise RuntimeError(f"Unexpected {dataset} lengths: {lengths}")
        train[0], test[0]
    elif dataset in ("mnist", "qmnist"):
        # The repository's QMNISTDataModule needs BOTH MNIST and QMNIST test50k.
        files = _digit_resources(native.indices_MNIST, root, allow_download)
        files += _digit_resources(native.indices_QMNIST, root, allow_download)
        with patch.object(
            native, "download_and_extract_archive", side_effect=no_download
        ):
            train = native.indices_MNIST(str(root), train=True)
            test = native.indices_MNIST(str(root), train=False)
            holdout = native.indices_QMNIST(
                str(root), "test50k", download=False, compat=True
            )
        lengths = {"train": len(train), "test": len(test), "holdout": len(holdout)}
        if lengths != {"train": 60000, "test": 10000, "holdout": 50000}:
            raise RuntimeError(f"Unexpected MNIST/QMNIST lengths: {lengths}")
        train[0], test[0], holdout[0]
    else:
        raise ValueError("Supported datasets: cifar10, cifar100, mnist/qmnist")
    return {
        "dataset": dataset,
        "root": str(root),
        "lengths": lengths,
        "files": {str(path.relative_to(root)): sha256(path) for path in files},
    }


def prepare_dataset(dataset, root, allow_download=True):
    """Serialize preparation; return a content manifest after loading all splits."""
    root = Path(root).expanduser().resolve()
    if allow_download:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise RuntimeError(f"Data root does not exist: {root}")
    # A shared lock prevents concurrent login-node jobs extracting into one cache.
    with open(root / ".rho-data.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _prepare(dataset, root, allow_download)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["cifar10", "cifar100", "mnist", "qmnist"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_dataset(args.dataset, args.data_root, not args.verify_only),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
