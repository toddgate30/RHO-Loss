import json
import sys
from types import SimpleNamespace

import pytest
import torch

from reproduction import execute as runner


def test_non_p100_gpu_is_rejected_before_any_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "NVIDIA V100")
    monkeypatch.setattr(
        torch, "zeros", lambda *a, **k: pytest.fail("Must reject before GPU operation")
    )
    with pytest.raises(RuntimeError, match="P100-only"):
        runner.execute({"run_dir": str(tmp_path)})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fail", [False, True])
def test_completion_requires_success_and_wandb_flush(tmp_path, monkeypatch, fail):
    run_dir = tmp_path / "irr"
    manifest = {
        "run_dir": str(run_dir),
        "irr_dir": str(run_dir),
        "dataset": "cifar10",
        "data_root": str(tmp_path / "data"),
        "stage": "irr",
        "irr_seed": 2,
        "recipe_sha256": "recipe",
        "overrides": [],
        "entrypoint": "run_irreducible.py",
        "experiment": "cifar10_resnet18_irred",
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *args: "Tesla P100-PCIE-16GB"
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (6, 0))
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: torch.tensor([0]))
    monkeypatch.setattr(
        runner, "prepare_dataset", lambda *a, **k: {"files": {"data": "checksum"}}
    )
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "fixture\n")
    monkeypatch.setattr(sys, "argv", sys.argv[:])
    monkeypatch.chdir(tmp_path)
    for name in ["WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR"]:
        monkeypatch.setenv(name, "fixture")
    events = []

    def finish(**kwargs):
        assert not (run_dir / "FINISHED").exists()
        events.append(kwargs)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(finish=finish))

    def fake_native_loop(*args, **kwargs):
        if fail:
            raise RuntimeError("fixture stage failure")
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "epoch_001.ckpt").write_bytes(b"fixture model")
        (checkpoint_dir / "last.ckpt").write_bytes(b"fixture last model")
        (checkpoint_dir / "irred_losses_and_checks.pt").write_bytes(b"fixture losses")

    monkeypatch.setattr(runner.runpy, "run_path", fake_native_loop)
    if fail:
        with pytest.raises(RuntimeError, match="fixture stage failure"):
            runner.execute(manifest)
        assert events == [{"exit_code": 1}]
        assert not (run_dir / "FINISHED").exists()
        assert (
            json.loads((run_dir / "FAILED.json").read_text())["type"] == "RuntimeError"
        )
        assert not (run_dir / "artifact.json").exists()
    else:
        runner.execute(manifest)
        assert events == [{}]
        assert (run_dir / "FINISHED").is_file()
        artifact = json.loads((run_dir / "artifact.json").read_text())
        assert artifact["irr_seed"] == 2
        assert artifact["checkpoint"].endswith("epoch_001.ckpt")
        assert artifact["precomputed_sha256"]
