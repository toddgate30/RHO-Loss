import json
from pathlib import Path

import pytest

from reproduction import submit as launcher
from reproduction.run_job import RunType, run_job


def example(tmp_path):
    python = tmp_path / "python"
    python.touch()
    site = {
        "python": str(python),
        "data_root": str(tmp_path / "data"),
        "output_root": str(tmp_path / "out"),
        "slurm": {"cpus": 4, "mem": "32G", "time": "1-00:00:00", "exclude_nodes": []},
    }
    recipe = {
        "dataset": "cifar10",
        "seeds": [1, 2, 3, 4, 5],
        "reference_pairs": [
            {"irr_seed": 1, "main_seed": 12, "label": "repository_defaults"}
        ],
        "reference_note": "Defaults are not confirmed published seeds",
        "methods": ["rho", "uniform"],
        "loss_source": "precomputed",
        "irr": {"experiment": "cifar10_resnet18_irred"},
        "main": {"experiment": "cifar10_resnet18_main"},
    }
    return recipe, site


def test_download_failure_prevents_all_submissions(tmp_path, monkeypatch):
    recipe, site = example(tmp_path)

    def fail(*args):
        raise RuntimeError("download failed")

    monkeypatch.setattr(launcher, "prepare_on_login_node", fail)
    monkeypatch.setattr(
        launcher, "run_job", lambda *a, **k: pytest.fail("Must not submit")
    )
    with pytest.raises(RuntimeError, match="download failed"):
        launcher.submit(recipe, site, dry_run=False)
    assert not Path(site["output_root"]).exists()


def test_six_pipelines_use_their_own_artifacts_and_dependencies(tmp_path, monkeypatch):
    recipe, site = example(tmp_path)
    events = []
    monkeypatch.setattr(
        launcher, "prepare_on_login_node", lambda *args: events.append("prepared")
    )

    def fake_job(path, *args, **kwargs):
        assert events[0] == "prepared"
        manifest = json.loads(Path(path).read_text())
        events.append((manifest, kwargs.get("dependency")))
        return str(len(events) + 100)

    monkeypatch.setattr(launcher, "run_job", fake_job)
    records = launcher.submit(recipe, site, dry_run=False)
    assert len(records) == 18
    irr_dirs = set()
    for offset in range(0, 18, 3):
        irr, rho, uniform = [event for event in events[1:][offset : offset + 3]]
        irr_dirs.add(irr[0]["run_dir"])
        assert irr[1] is None
        assert rho[1] == uniform[1] == records[offset]["job_id"]
        assert rho[0]["irr_dir"] == uniform[0]["irr_dir"] == irr[0]["run_dir"]
        assert rho[0]["irr_seed"] == irr[0]["irr_seed"]
        assert "logger.wandb.project=ValidateRhoLoss" in rho[0]["overrides"]
        assert "logger.wandb.entity=ksmiller-byu-research" in rho[0]["overrides"]
    assert len(irr_dirs) == 6
    assert events[-1][0]["irr_seed"] == 1 and events[-1][0]["main_seed"] == 12


def test_dry_run_has_p100_and_regular_job_flags_without_side_effects(tmp_path, capsys):
    run_job(
        tmp_path / "job.json", RunType.DRY, python="/env/bin/python", dependency="123"
    )
    output = capsys.readouterr().out
    assert "--constraint=pascal" in output
    assert "--dependency=afterok:123" in output
    assert "--time=1-00:00:00" in output
    assert "standby" not in output and "--requeue" not in output
    assert not list(tmp_path.iterdir())
