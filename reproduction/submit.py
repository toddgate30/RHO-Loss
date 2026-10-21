"""Build seed pipelines; stage data on the LOGIN NODE before any submission."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess

import yaml
from tqdm import tqdm

from reproduction.run_job import RunType, run_job

REPO = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def required_path(value, name):
    if not value:
        raise ValueError(
            f"Set {name} in configs/reproduction/site.yaml before submitting."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute shared path.")
    return path.resolve()


def build_manifest(recipe, site, run_dir, pair, stage, irr_dir):
    irr_seed, main_seed = pair["irr_seed"], pair["main_seed"]
    seed = irr_seed if stage == "irr" else main_seed
    data_root = required_path(site["data_root"], "data_root")
    selected = recipe["irr"] if stage == "irr" else recipe["main"]
    overrides = list(selected.get("overrides", []))
    overrides += [
        f"seed={seed}",
        f"datamodule.data_dir={json.dumps(str(data_root))}",
        f"datamodule.num_workers={site['slurm']['cpus']}",
        f"hydra.run.dir={json.dumps(str(run_dir))}",
        f"base_outdir={json.dumps(str(run_dir))}",
        "+model.progress_logging=true",
        "logger.wandb.offline=true",
        "logger.wandb.project=ValidateRhoLoss",
        "logger.wandb.entity=miller-ml-research",
        f"logger.wandb.name={recipe['dataset']}_{stage}_irr{irr_seed}_main{main_seed}",
        f"logger.wandb.id={secrets.token_hex(4)}",
        f"logger.wandb.group={json.dumps(run_dir.parent.parent.name)}",
        f"logger.wandb.job_type={stage}",
        f"logger.wandb.tags=[{recipe['dataset']},{stage},{pair['label']}]",
        "++trainer.default_root_dir=" + json.dumps(str(run_dir)),
    ]
    if stage != "irr":
        if stage == "uniform":
            overrides.append("selection_method=uniform_selection")
            if recipe.get("uniform_percent_train") is not None:
                overrides.append(
                    f"model.percent_train={recipe['uniform_percent_train']}"
                )
        # Artifact-dependent overrides are appended only on the compute node,
        # after the irr-stage dependency has succeeded.
    metadata = {
        "dataset": recipe["dataset"],
        "stage": stage,
        "irr_seed": irr_seed,
        "main_seed": main_seed,
        "seed_label": pair["label"],
        "reference_note": recipe["reference_note"],
        "recipe_sha256": hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode()
        ).hexdigest(),
    }
    # Hydra's dictionary grammar differs from JSON. Add metadata as individual leaves.
    overrides.extend(
        f"++reproduction.{key}={json.dumps(value)}" for key, value in metadata.items()
    )
    return {
        **metadata,
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "irr_dir": str(irr_dir),
        "loss_source": recipe["loss_source"],
        "entrypoint": "run_irreducible.py" if stage == "irr" else "run.py",
        "experiment": selected["experiment"],
        "overrides": overrides,
    }


def seed_pairs(recipe):
    pairs = [
        {"irr_seed": seed, "main_seed": seed, "label": "paired"}
        for seed in recipe["seeds"]
    ]
    pairs.extend(recipe.get("reference_pairs", []))
    identities = [(pair["irr_seed"], pair["main_seed"]) for pair in pairs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate seed pairs in recipe.")
    return pairs


def prepare_on_login_node(recipe, site, verify_only=False):
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Run submission/data preparation on a login node, outside a compute allocation."
        )
    python = required_path(site["python"], "python")
    root = required_path(site["data_root"], "data_root")
    command = [
        str(python),
        "-m",
        "reproduction.data",
        recipe["dataset"],
        "--data-root",
        str(root),
    ]
    if verify_only:
        command.append("--verify-only")
    # Use the training interpreter for the exact native loaders/version.
    action = (
        "Verifying cached dataset" if verify_only else "Checking/downloading dataset"
    )
    print(
        f"[data] {action} on the login node before submitting any jobs...", flush=True
    )
    subprocess.run(command, cwd=REPO, check=True)
    print(
        "[data] Dataset preparation succeeded. Starting Slurm submission.", flush=True
    )


def submit(recipe, site, *, dry_run=True, verify_only=False):
    python = required_path(site["python"], "python")
    output = required_path(site["output_root"], "output_root")
    required_path(site["data_root"], "data_root")
    if recipe.get("needs_decision"):
        raise ValueError(recipe["needs_decision"])
    pairs = seed_pairs(recipe)
    if dry_run:
        print(
            "DRY RUN: login-node dataset check/download/verification precedes every submission."
        )
        print(
            f"Would prepare {recipe['dataset']} using {python}; no downloads/jobs performed."
        )
    else:
        if not python.is_file():
            raise ValueError(f"Training interpreter not found: {python}")
        prepare_on_login_node(recipe, site, verify_only)
    token = (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + secrets.token_hex(4)
    )
    sweep_dir = output / token / recipe["dataset"]
    submitted = []
    with tqdm(
        total=len(pairs) * (1 + len(recipe["methods"])),
        desc="Submitting jobs",
        unit="job",
        disable=dry_run,
        dynamic_ncols=True,
    ) as progress:
        for pair in pairs:
            pair_dir = (
                sweep_dir
                / f"{pair['label']}_irr{pair['irr_seed']}_main{pair['main_seed']}"
            )
            irr_dir = pair_dir / "irr"
            dependency = None
            for stage in ("irr", *recipe["methods"]):
                run_dir = pair_dir / stage
                manifest = build_manifest(recipe, site, run_dir, pair, stage, irr_dir)
                manifest_path = run_dir / "job.json"
                if not dry_run:
                    run_dir.mkdir(parents=True, exist_ok=False)
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    (run_dir / "recipe.yaml").write_text(
                        yaml.safe_dump(recipe, sort_keys=False)
                    )
                else:
                    print(
                        f"{stage}: {manifest['entrypoint']} +experiment={manifest['experiment']} "
                        f"irr_seed={pair['irr_seed']} main_seed={pair['main_seed']}"
                    )
                if not dry_run:
                    progress.set_postfix_str(
                        f"{stage} irr={pair['irr_seed']} main={pair['main_seed']}"
                    )
                job_id = run_job(
                    manifest_path,
                    RunType.DRY if dry_run else RunType.SBATCH,
                    python=python,
                    **site["slurm"],
                    name=f"rho-{recipe['dataset']}-{stage}-{pair['main_seed']}",
                    dependency=(dependency or "IRR_JOB_ID") if stage != "irr" else None,
                )
                if stage == "irr":
                    dependency = job_id
                submitted.append(
                    {"job_id": job_id, "manifest": str(manifest_path), "stage": stage}
                )
                if not dry_run:
                    # Persist each success so partially submitted sweeps remain traceable.
                    (sweep_dir / "submitted.json").write_text(
                        json.dumps(submitted, indent=2) + "\n"
                    )
                    progress.update(1)
                    tqdm.write(
                        f"Submitted job {job_id}: {stage}, irr_seed={pair['irr_seed']}, main_seed={pair['main_seed']}"
                    )
    if not dry_run:
        print(
            f"Queued {len(submitted)} jobs. Submission record: {sweep_dir / 'submitted.json'}",
            flush=True,
        )
    return submitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Dataset recipe YAML")
    parser.add_argument("--site", default="configs/reproduction/site.yaml")
    parser.add_argument(
        "--submit", action="store_true", help="Prepare data, then actually submit jobs"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Require already prepared data; do not download",
    )
    args = parser.parse_args()
    submit(
        load_yaml(args.config),
        load_yaml(args.site),
        dry_run=not args.submit,
        verify_only=args.verify_only,
    )


if __name__ == "__main__":
    main()
