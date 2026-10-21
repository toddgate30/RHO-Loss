"""Offline-run sync daemon adapted from online-batch-selection/wandb-sync-daemon.py."""

import argparse
import fcntl
import subprocess
import time
from pathlib import Path


def sync(root, subdirs=(), wandb="wandb"):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Output root not found: {root}")
    with open(root / ".wandb-sync.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another sync process holds this output root; skipping.")
            return False
        roots = (
            [path for pattern in subdirs for path in root.glob(pattern)]
            if subdirs
            else [root]
        )
        runs = {}
        for folder in roots:
            for offline_dir in folder.glob("**/wandb/offline-run-*"):
                if "slurm_history" not in offline_dir.parts and offline_dir.is_dir():
                    runs.setdefault(offline_dir.parent.parent, set()).add(offline_dir)
        success = True
        for run, offline_dirs in sorted(runs.items()):
            # Snapshot completion BEFORE syncing, so completion during a partial
            # upload cannot cause a premature SYNCED marker.
            finished = (run / "FINISHED").is_file()
            if finished and (run / "SYNCED").is_file():
                continue
            # An interrupted initialization may leave an empty offline directory.
            # W&B 0.12.1 can skip it and nevertheless exit with status zero.
            if any(len(list(folder.glob("*.wandb"))) != 1 for folder in offline_dirs):
                print(
                    f"Cannot sync {run}: expected one .wandb file per offline directory.",
                    flush=True,
                )
                success = False
                continue
            upload_failed = False
            command = [
                wandb,
                "sync",
                "--include-synced",
                "--project",
                "ValidateRhoLoss",
                "--entity",
                "miller-ml-research",
                *map(str, sorted(offline_dirs)),
            ]
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    # W&B 0.12.1 does not propagate background-thread failures
                    # to the CLI exit code, including failed individual uploads.
                    lowered = line.lower()
                    if any(
                        message in lowered
                        for message in (
                            "traceback (most recent call last)",
                            "exception in thread",
                            "wandb: error",
                            "failed to upload",
                            "upload_file exception",
                            "skipping directory:",
                            ".wandb file is empty",
                            ".wandb file is incomplete",
                        )
                    ):
                        upload_failed = True
            if process.returncode == 0 and not upload_failed and finished:
                (run / "SYNCED").touch()
            elif process.returncode != 0 or upload_failed:
                success = False
        return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subdirs", nargs="*", help="Subdirectory/glob filters under --root"
    )
    parser.add_argument("--root", required=True, help="Shared reproduction output root")
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        start = time.monotonic()
        success = sync(args.root, args.subdirs)
        if args.once:
            raise SystemExit(0 if success else 1)
        time.sleep(max(0, args.interval - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
