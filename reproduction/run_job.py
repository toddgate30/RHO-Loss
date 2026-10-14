"""Regular-job launcher adapted from online-batch-selection/run/run_utils/run_job.py."""

from enum import Enum
from pathlib import Path
import shlex
import subprocess


class RunType(Enum):
    DRY = 0
    NORMAL = 1
    SBATCH = 2
    SRUN = 3


def run_job(
    manifest_path,
    run_type=RunType.DRY,
    *,
    python,
    cpus=4,
    mem="32G",
    time="1-00:00:00",
    name="rho-loss",
    exclude_nodes=None,
    dependency=None,
):
    """Submit ONLY to P100s, without standby/requeue flags. Preflight belongs to submit.py."""
    repo = Path(__file__).resolve().parents[1]
    python_cmd = [
        str(python),
        "-m",
        "reproduction.execute",
        str(Path(manifest_path).resolve()),
    ]
    flags = [
        "--gres=gpu:1",
        "--constraint=pascal",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time}",
        f"--job-name={name}",
        f"--chdir={repo}",
    ]
    if exclude_nodes:
        if not isinstance(exclude_nodes, list):
            raise ValueError("exclude_nodes must be a list")
        flags.append("--exclude=" + ",".join(exclude_nodes))
    if dependency:
        flags.append("--dependency=afterok:" + str(dependency))
        flags.append("--kill-on-invalid-dep=yes")
    log_dir = Path(manifest_path).resolve().parent / "slurm"
    flags += [f"--output={log_dir}/%j.out", f"--error={log_dir}/%j.err"]
    command = ["sbatch", "--parsable", *flags, "--wrap=" + shlex.join(python_cmd)]
    if run_type == RunType.DRY:
        print(shlex.join(command))
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    if run_type == RunType.NORMAL:
        subprocess.run(python_cmd, cwd=repo, check=True)
        return None
    if run_type == RunType.SRUN:
        subprocess.run(["srun", *flags, *python_cmd], cwd=repo, check=True)
        return None
    if run_type != RunType.SBATCH:
        raise ValueError(f"Unknown run type: {run_type}")
    result = subprocess.run(
        command, cwd=repo, check=True, capture_output=True, text=True
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Cannot interpret sbatch job ID: {result.stdout!r}")
    return job_id
