import io

import pytest

from reproduction import sync as daemon


def test_active_final_failed_and_already_synced_runs(tmp_path, monkeypatch):
    runs = {}
    for name in ["active", "finished", "failed", "already"]:
        run = tmp_path / name
        offline = run / "wandb" / "offline-run-test"
        offline.mkdir(parents=True)
        (offline / "run-test.wandb").touch()
        runs[name] = run
    for name in ["finished", "failed", "already"]:
        (runs[name] / "FINISHED").touch()
    (runs["already"] / "SYNCED").touch()
    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdout = io.StringIO("sync output\n")
            self.returncode = 1 if "/failed/" in command[-1] else 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(daemon.subprocess, "Popen", Process)
    assert daemon.sync(tmp_path) is False
    assert len(commands) == 3
    assert all("--include-synced" in command for command in commands)
    for command in commands:
        assert command[command.index("--entity") + 1] == "miller-ml-research"
        assert command[command.index("--project") + 1] == "ValidateRhoLoss"
    assert not (runs["active"] / "SYNCED").exists()
    assert not (runs["failed"] / "SYNCED").exists()
    assert (runs["finished"] / "SYNCED").is_file()


@pytest.mark.parametrize(
    "output",
    [
        "Exception in thread Thread-1:\nTraceback (most recent call last):\n",
        'wandb: ERROR Error uploading "config.yaml": CommError, <Response [400]>\ndone.\n',
        "Skipping directory: /some/run\n",
        "Seen metric with glob (shouldnt happen)\ndone.\n",
    ],
)
def test_zero_exit_does_not_hide_upload_errors(tmp_path, monkeypatch, output):
    offline = tmp_path / "wandb" / "offline-run-test"
    offline.mkdir(parents=True)
    (offline / "run-test.wandb").touch()
    (tmp_path / "FINISHED").touch()

    class Process:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO(output)
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(daemon.subprocess, "Popen", Process)
    expected = output.startswith("Seen metric")
    assert daemon.sync(tmp_path) is expected
    assert (tmp_path / "SYNCED").exists() is expected


def test_missing_upload_file_is_not_marked_synced(tmp_path, monkeypatch):
    (tmp_path / "wandb" / "offline-run-test").mkdir(parents=True)
    (tmp_path / "FINISHED").touch()
    monkeypatch.setattr(
        daemon.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("Must not sync an empty directory"),
    )
    assert daemon.sync(tmp_path) is False
    assert not (tmp_path / "SYNCED").exists()
