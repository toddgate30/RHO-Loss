from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from reproduction.execute import artifact_overrides
from reproduction.submit import build_manifest, load_yaml, seed_pairs

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100", "qmnist"])
@pytest.mark.parametrize("stage", ["irr", "rho", "uniform"])
def test_recipes_compose_without_training(dataset, stage):
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
    assert config.logger.wandb.offline is True
    assert config.logger.wandb.project == "ValidateRhoLoss"
    assert config.logger.wandb.entity == "ksmiller-byu-research"
    assert config.model.progress_logging is True
    if stage != "irr":
        assert config.model.percent_train == 0.1
        assert config.eval_set == ("val" if dataset == "cifar10" else "test")
    if stage != "irr" and dataset == "qmnist":
        assert config.model.update_irreducible is True
        assert config.datamodule.trainset_corruption.pc_corrupted == 0.1


def test_conda_spec_preserves_original_pins():
    original = load_yaml(REPO / "my_environment.yaml")
    adopted = load_yaml(REPO / "environment-p100.yaml")
    assert adopted["name"] == "rho-loss-p100"
    assert original["channels"] == adopted["channels"]
    assert original["dependencies"][:-1] == adopted["dependencies"][:-1]
    assert set(original["dependencies"][-1]["pip"]) <= set(
        adopted["dependencies"][-1]["pip"]
    )
