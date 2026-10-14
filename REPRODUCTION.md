# ValidateRhoLoss on BYU P100 nodes

This implementation runs the repository's native holdout and target loops,
with regular Slurm jobs and offline W&B. **Use P100 GPUs only.** The launcher
requests `--constraint=pascal`; confirm BYU still uses that feature name before
the first submission. The stage runner also rejects GPUs other than P100.
No training experiments or cluster jobs have been run to validate this setup.

## Environment: prepare once on a login node

```bash
conda env create -f environment-p100.yaml
conda activate rho-loss-p100
python -m pip check
```

`environment-p100.yaml` preserves the original Conda channels and every
original dependency/build pin from `my_environment.yaml`. The environment is
renamed `rho-loss-p100`; pinned `python-dotenv`, `pytest`, and `sh` are added for
development checks. Python 3.9.6, PyTorch 1.9.0/CUDA 11.1, Lightning 1.4.6,
TorchMetrics 0.5.1, and NumPy 1.20.3 remain as specified by the authors.

The historical Conda builds have not been installed/resolved in this cloud
machine, and GPU support has not been validated on BYU. If creation fails,
save the solver error and review any proposed dependency change; do not silently
upgrade the training stack or use unpinned `requirements.txt`.

## Edit the site settings

Edit **`configs/reproduction/site.yaml`** before running anything. Its three
required paths are deliberately empty rather than guessed:

- `python`: absolute path to `rho-loss-p100/bin/python`, visible on compute nodes.
- `data_root`: shared absolute dataset directory.
- `output_root`: shared absolute directory for runs, checkpoints, and offline W&B.

The same file prominently sets **4 CPUs, 32 GB RAM, and one day**. Adjust those
resource values there. P100 selection and regular-job behavior are enforced
by the launcher, not optional site settings. There is no standby QoS/requeue
implementation. Run all commands below from the repository root.

## Inspect and submit a dataset sweep

```bash
# Dry run: prints commands only; no files/downloads/jobs.
python run_validate_rho_loss.py --config configs/reproduction/cifar10.yaml

# Run this on a LOGIN NODE when ready. It downloads/verifies data first,
# then submits holdout, RHO-Loss, and uniform jobs with dependencies.
python run_validate_rho_loss.py --config configs/reproduction/cifar10.yaml --submit

# Subsequent datasets use the same submission command and site settings.
python run_validate_rho_loss.py --config configs/reproduction/cifar100.yaml --submit
python run_validate_rho_loss.py --config configs/reproduction/qmnist.yaml --submit
```

These are reproduction-tool YAML files, not merged Hydra configs to pass
directly to `run.py`. They select the original Hydra experiment recipes and
record any extra overrides. Every submitted stage receives its own `job.json`,
`recipe.yaml`, native composed `.hydra` config, Slurm logs, and `run_info.json`.
Partial submission history is retained in `submitted.json`.

On the login node the submission command checks the exact repository dataset
files, downloads/extracts missing or corrupt data, verifies checksums and
loads all required splits, and **only then calls `sbatch`**. Failure prevents
submission. A shared lock prevents concurrent extraction into the same cache.
Use `--verify-only --submit` to require already prepared data without downloads.
Submission inside an existing Slurm allocation is rejected.

Digit preparation needs both `indices_MNIST/raw` (training and test data) and
`indices_QMNIST/raw` (QMNIST test50k holdout). Standard torchvision cache folder
names are insufficient. Checksum-verified compressed digit files are retained
so extracted files can be verified and repaired, rather than treating a
directory or an unverified raw file as complete. The compute-stage runner
rechecks data without downloading and blocks native download fallbacks.

## Seeds and experiment settings

All three dataset sweeps include:

| Label | Holdout/main seed pairs |
| --- | --- |
| `paired` | (1,1), (2,2), (3,3), (4,4), (5,5) |
| `repository_defaults` | (1,12) |

The `(1,12)` pair reflects the base configuration defaults, **not confirmed
published-run provenance**. Checked-in job lists also contain seeds 1 and 2.
Do not describe a configuration default as proof of the seed used in a paper
figure. The noisy `mnist_update` recipe explicitly uses main seed 1, so its
native default pair `(1,1)` is already included in the MNIST/QMNIST paired sweep.
Its extra `(1,12)` trial is labeled `base_defaults_reference` to distinguish
that requested comparison from the native MNIST recipe defaults.

Each pair trains its own holdout model and records its artifact hashes. It
then runs RHO-Loss and uniform sampling with the same main seed, selected
fraction (0.1), and epoch budget. No holdout artifact is shared across replicate
directories, including the extra reference trial. Select the seed list before
running and retain all completed, failed, and retried outcomes. A study across
these seeds measures sensitivity; it cannot establish whether the authors
cherry-picked seeds.

| Recipe | Holdout epochs | Main epochs | Main evaluation | Holdout use |
| --- | --- | --- | --- | --- |
| CIFAR-10 | 100 | 175 | Validation each epoch; final test | Precomputed losses |
| CIFAR-100 | 100 | 150 | Test set in the validation loop, as configured originally | Holdout checkpoint; losses computed during selection |
| MNIST/QMNIST | 10 | 100 | Test set in the validation loop | Holdout checkpoint, updated during main training; original 10% label noise |

The QMNIST holdout config had `max_epochs=10` while inheriting `min_epochs=30`;
its reproduction override sets `min_epochs=10` to make that range valid.
No other recipe epoch budget, optimizer, selection mode, augmentation, or
data split is changed. MNIST's configured augmentation warning remains native.
Uniform uses the same selected fraction/epochs, as requested, and preserves
the MNIST recipe's holdout-model update behavior.

Target jobs use `afterok` dependencies on their own holdout job. That stage
must create `artifact.json` before it succeeds. The main runner checks dataset,
holdout seed, recipe identity, data hashes, and artifact hash before training.
Normal best/last checkpoints remain available; automatic standby recovery is
not added. A new submission produces fresh unique run directories. Keep
manual retries explicit; an already-started stage refuses silent reuse.

## W&B and login-node sync daemon

All jobs log offline to **`ksmiller-byu-research/ValidateRhoLoss`**, using the
native Lightning logger. No compute-node internet access or W&B authentication
is required. The remote project is populated when the offline files are synced;
this implementation has not created/uploaded a project online.

From an internet-connected login node with access to the shared output root:

```bash
conda activate rho-loss-p100
wandb login
python wandb-sync-daemon.py --root /YOUR/SHARED/OUTPUT_ROOT

# Optional filters, one-shot execution, or another polling interval:
python wandb-sync-daemon.py '*/cifar10' --root /YOUR/SHARED/OUTPUT_ROOT --once
python wandb-sync-daemon.py --root /YOUR/SHARED/OUTPUT_ROOT --interval 60
```

Use the existing W&B login if available; never put keys in configs or Git.
This adapts the group's daemon: recursively discovers `wandb/offline-run-*`,
syncs active runs, polls every 15 seconds by default, and skips finished runs
with a `SYNCED` marker. `FINISHED` is written only after successful stage
completion, artifact creation, and W&B flushing. `SYNCED` is written only after
successful syncing of a run observed finished before that sync began. Failed
syncs remain eligible for retry. A root lock prevents overlapping daemon
instances. `--include-synced` allows growing active runs to upload again after
an earlier upload. Destination project/entity are explicit in the sync command.

## Added progress metrics: no new predictions

`progress_logging` is optional and defaults to false for existing callers;
the reproduction launcher enables it. Native loss/accuracy keys and aggregation
remain unchanged. New keys are:

- `train_progress_step`: fitted progress on the native selected training batch
  at Lightning's native logger emission points (default: every 50 steps).
- `train_progress_epoch`: fit on pooled native training observations throughout
  the epoch; these span changing model states and selected/augmented samples.
- `val_progress_epoch` and `test_progress_epoch`: fit on existing predictions
  from the corresponding native evaluation loop. `val_*` follows the actual
  dataset fed to that loop, which is the test set in some original recipes.

The mathematical function is ported from the group's `_geodesic_progress`.
Predictions are detached and copied; no forward pass, mode change, RNG call,
or extra BatchNorm update is added. Epoch values are fitted on pooled
predictions rather than averaging separately fitted batch values. Sanity
validation is excluded. Added CPU fitting/copy overhead is not zero; account
for it separately in runtime comparisons. No full-dataset eval-mode training
diagnostic or custom logarithmic schedule is introduced.

## Changes and validation record

| Change | Reason |
| --- | --- |
| Separate named Conda spec, original pins plus three pinned development packages | Preserve the paper's stack without modifying the group's environment |
| Login-node integrity/download preparation and offline compute checks | GPU nodes have no internet; exact native dataset layouts are required |
| HTTPS-only downloads with original archive checksums | Avoid the old torchvision downloader's automatic fallback to HTTP |
| Regular P100-only launcher, stage/seed manifests and dependencies | Reuse the group's submission approach with Hydra commands and explicit provenance |
| Base-default reference trial alongside paired 1–5 sweeps | Compare base defaults with varied complete pipelines without claiming published seed provenance |
| Offline native W&B and adapted daemon | Populate ValidateRhoLoss from login-node syncing |
| Optional prediction-only progress instrumentation | Add requested diagnostics while preserving training state |
| QMNIST holdout min_epochs 30 → 10 | Resolve original contradictory min/max bounds |
| Git provenance lookup explicitly uses checkout cwd | Hydra outputs can live outside the checkout on shared storage |

Focused checks can be run without downloading data, submitting jobs, launching
an experiment, or syncing to W&B:

```bash
python -m pytest -o addopts='' tests/reproduction -q
```

The checks exercise config composition, seed/artifact dependencies, rejection
of wrong-seed/tampered artifacts, corrupted local data repair, offline refusal
to download, mocked sync outcomes, and diagnostic invariance. Synthetic native
step tests compare forward counts, BatchNorm buffers, weights, RNG state,
accuracy, and loss with progress disabled/enabled. They do not execute a paper
experiment, `Trainer.fit`, a cluster job, or real `wandb sync`.

Current cloud result: **29 focused checks passed** in a Poetry-derived Python
3.9 environment. The exact Conda pins above have not been solved here.

Still unverified: actual historical Conda solve, full datasets downloaded into
the selected paths, P100/driver execution, BYU Slurm feature/limits, complete
paper loops, and real W&B synchronization. The old MNIST test elsewhere in the
repository imports an absent module and remains a known unrelated failure.
Do not call these unexecuted steps validated.
