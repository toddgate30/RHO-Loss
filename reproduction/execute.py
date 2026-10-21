"""One offline P100 stage. Invoked by Slurm, not directly from a login node."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

from reproduction.data import prepare_dataset, sha256

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def artifact_for(manifest):
    path = Path(manifest["irr_dir"]) / "artifact.json"
    artifact = json.loads(path.read_text())
    for key in ("dataset", "irr_seed", "recipe_sha256"):
        if artifact[key] != manifest[key]:
            raise RuntimeError(f"Holdout artifact does not match {key}: {path}")
    artifact_path = Path(artifact[manifest["loss_source"]])
    if (
        not artifact_path.is_file()
        or sha256(artifact_path) != artifact[manifest["loss_source"] + "_sha256"]
    ):
        raise RuntimeError(f"Missing/corrupt holdout artifact: {artifact_path}")
    return artifact, artifact_path


def artifact_overrides(manifest, artifact_path):
    if manifest["loss_source"] == "precomputed":
        return [
            "irreducible_loss_generator=precomputed_loss",
            "irreducible_loss_generator.f=" + json.dumps(str(artifact_path)),
        ]
    if manifest["loss_source"] == "checkpoint":
        return [
            "irreducible_loss_generator=irreducible_loss_model",
            "irreducible_loss_generator.checkpoint_path="
            + json.dumps(str(artifact_path)),
        ]
    raise ValueError("loss_source must be precomputed or checkpoint")


def execute(manifest):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This reproduction requires exactly one allocated P100 GPU.")
    name = torch.cuda.get_device_name(0)
    if "P100" not in name or torch.cuda.get_device_capability(0) != (6, 0):
        raise RuntimeError(f"P100-only reproduction; refusing allocated GPU {name}.")
    torch.zeros(
        1, device="cuda"
    )  # Verify compiled kernels/driver before loading data or training.
    run_dir = Path(manifest["run_dir"])
    if (run_dir / "STARTED").exists():
        raise RuntimeError(
            "Run already started; use a new run directory for an explicit retry."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STARTED").touch()
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = "ValidateRhoLoss"
    os.environ["WANDB_ENTITY"] = "miller-ml-research"
    os.environ["WANDB_DIR"] = str(run_dir)
    metadata = {
        **manifest,
        "gpu_name": name,
        "compute_capability": [6, 0],
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
    }
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = "unavailable"
    metadata["nvidia_driver"] = driver
    try:
        data = prepare_dataset(
            manifest["dataset"], manifest["data_root"], allow_download=False
        )
        write_json(run_dir / "data_manifest.json", data)
        overrides = list(manifest["overrides"])
        if manifest["stage"] != "irr":
            artifact, artifact_path = artifact_for(manifest)
            if artifact["data_files"] != data["files"]:
                raise RuntimeError(
                    "Dataset files differ from the holdout preparation run."
                )
            metadata["holdout_artifact"] = artifact
            overrides += artifact_overrides(manifest, artifact_path)
        overrides += [
            "++reproduction.gpu=" + json.dumps(name),
            "++reproduction.torch=" + json.dumps(torch.__version__),
            "++reproduction.cuda=" + json.dumps(torch.version.cuda),
            "++reproduction.nvidia_driver=" + json.dumps(driver),
            "++reproduction.data_sha256="
            + json.dumps(
                hashlib.sha256(
                    json.dumps(data["files"], sort_keys=True).encode()
                ).hexdigest()
            ),
        ]
        if manifest["stage"] != "irr":
            overrides.append(
                "++reproduction.holdout_sha256=" + json.dumps(sha256(artifact_path))
            )
        write_json(run_dir / "run_info.json", metadata)
        # Training's legacy loaders must never fall back to networking on compute nodes.
        from src.datamodules.datasets import sequence_datasets as native
        from torchvision.datasets import utils as vision_utils

        def offline_only(*args, **kwargs):
            raise RuntimeError(
                "Dataset download attempted on a compute node. Prepare on a login node first."
            )

        sys.argv = [
            manifest["entrypoint"],
            "+experiment=" + manifest["experiment"],
            *overrides,
        ]
        os.chdir(REPO)
        with patch.object(
            native, "download_and_extract_archive", side_effect=offline_only
        ), patch.object(native, "download_url", side_effect=offline_only), patch.object(
            vision_utils, "download_url", side_effect=offline_only
        ):
            runpy.run_path(str(REPO / manifest["entrypoint"]), run_name="__main__")
        if manifest["stage"] == "irr":
            checkpoints = [
                path
                for path in (run_dir / "checkpoints").glob("*.ckpt")
                if path.name != "last.ckpt"
            ]
            if len(checkpoints) != 1:
                raise RuntimeError(
                    f"Expected one best holdout checkpoint, found {checkpoints}"
                )
            losses = run_dir / "checkpoints" / "irred_losses_and_checks.pt"
            if not losses.is_file():
                raise RuntimeError(
                    "Holdout training did not produce irreducible losses."
                )
            artifact = {
                key: manifest[key] for key in ("dataset", "irr_seed", "recipe_sha256")
            }
            artifact.update(
                {
                    "checkpoint": str(checkpoints[0]),
                    "checkpoint_sha256": sha256(checkpoints[0]),
                    "precomputed": str(losses),
                    "precomputed_sha256": sha256(losses),
                    "data_files": data["files"],
                }
            )
            write_json(run_dir / "artifact.json", artifact)
        # Mark completion only after local W&B has flushed and all stage artifacts exist.
        import wandb

        wandb.finish()
        (run_dir / "FINISHED").touch()
    except BaseException as error:
        write_json(
            run_dir / "FAILED.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        try:
            import wandb

            wandb.finish(exit_code=1)
        finally:
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    args = parser.parse_args()
    execute(json.loads(Path(args.manifest).read_text()))


if __name__ == "__main__":
    main()
