import io

from reproduction import sync as daemon


def test_active_final_failed_and_already_synced_runs(tmp_path, monkeypatch):
    runs = {}
    for name in ["active", "finished", "failed", "already"]:
        run = tmp_path / name
        (run / "wandb" / "offline-run-test").mkdir(parents=True)
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
