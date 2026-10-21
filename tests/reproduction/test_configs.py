from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from reproduction.execute import artifact_overrides
from reproduction.submit import build_manifest, load_yaml, seed_pairs

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100", "qmnist"])
@pytest.mark.parametrize("stage", ["irr", "rho", "uniform"])
def test_recipes_compose_without_training(dataset, stage, monkeypatch):
    # Numeric-only random IDs must survive Hydra parsing as strings.
    monkeypatch.setattr("reproduction.submit.secrets.token_hex", lambda _: "37429928")
    recipe = load_yaml(REPO / f"configs/reproduction/{dataset}.yaml")
    site = {"data_root": "/shared/data with spaces", "slurm": {"cpus": 4}}
    job = build_manifest(
        recipe,
        site,
        Path("/shared/output with spaces/run") / stage,
        seed_pairs(recipe)[0],
        stage,
        Path("/shared/output/irr"),
    )
    overrides = ["+experiment=" + job["experiment"], *job["overrides"]]
    if stage != "irr":
        overrides += artifact_overrides(job, Path("/shared/artifact with spaces.pt"))
    with initialize_config_dir(config_dir=str(REPO / "configs")):
        config = compose(
            config_name="irreducible_training" if stage == "irr" else "config",
            overrides=overrides,
        )
    assert config.trainer.gpus == 1
    assert config.trainer.min_epochs <= config.trainer.max_epochs
    assert config.datamodule.data_dir == site["data_root"]
    assert config.logger.wandb.id == "37429928"
    assert isinstance(config.logger.wandb.id, str)
    assert config.logger.wandb.offline is True
    assert config.logger.wandb.project == "ValidateRhoLoss"
    assert config.logger.wandb.entity == "miller-ml-research"
    assert config.model.progress_logging is True
    if stage != "irr":
        assert config.model.percent_train == 0.1
        assert config.eval_set == ("val" if dataset == "cifar10" else "test")
    if stage != "irr" and dataset == "qmnist":
        assert config.model.update_irreducible is True
        assert config.datamodule.trainset_corruption.pc_corrupted == 0.1


def test_conda_base_and_frozen_research_versions():
    adopted = load_yaml(REPO / "environment-p100.yaml")
    assert adopted["name"] == "rho-loss-p100"
    assert adopted["channels"] == ["conda-forge", "nodefaults"]
    assert "python=3.9.6" in adopted["dependencies"]
    assert "pip=21.2.4" in adopted["dependencies"]
    assert adopted["dependencies"][-1]["pip"] == ["-r requirements-p100.txt"]
    requirements = (REPO / "requirements-p100.txt").read_text()
    assert "--require-hashes" in requirements
    for pin in [
        "numpy==1.20.3",
        "pillow==8.3.1",
        "pytorch-lightning==1.4.6",
        "torchmetrics==0.5.1",
        "hydra-core==1.1.1",
        "wandb==0.12.1",
    ]:
        assert pin in requirements
    assert "torch-1.9.0%2Bcu111-cp39-cp39-linux_x86_64.whl" in requirements
    assert "torchvision-0.10.0%2Bcu111-cp39-cp39-linux_x86_64.whl" in requirements
    assert "_libgcc_mutex" not in str(adopted) and "jpeg=" not in str(adopted)
