# RHO-Loss reproduction plan for BYU P100 nodes

Status: implementation authorized and added to this checkout. See
`REPRODUCTION.md` for executable instructions, adopted settings, changes, and
validation limits. This document preserves the design and its rationale.
No paper experiments or Slurm jobs have been run; actual execution remains
outside the current request.

## Scope and decisions

- Reproduce CIFAR-10 first; then CIFAR-100 and the repository's MNIST/QMNIST
  workflow. Other datasets and GPU models are outside the initial scope.
- Use **P100 GPUs only**. Carry this requirement into every Slurm submission
  path, including interactive runs and examples. The group's existing
  environment specification documents `--constraint=pascal` (`-C pascal`).
  Confirm that this is still BYU's P100 selector before the first submission.
- Use regular jobs. Do not add standby QoS, standby requeue handling, or the
  group's experiment-extension machinery. Retain normal checkpoint saving.
- Keep the environment and experiment behavior as close to the paper's code
  as possible. Document every departure before interpreting results.
- Vary both irreducible-model and main-model seeds for every replicate. Never
  share one fixed-seed holdout model across the main-run seed sweep.
- GPU compute nodes have no internet access. Install dependencies and prepare
  data in advance on an approved machine with internet access and shared storage.
  Record offline W&B runs on compute nodes and sync them later.
- Added diagnostics must **only reuse predictions already computed by the
  original loops**. They must not add forward passes, change model mode, update
  BatchNorm statistics, consume training randomness, or alter optimization.

## 1. Establish and record the original experiment recipes

Inspect the README, tutorial, experiment configurations, and job lists together.
For each selected result, record the composed configuration and all CLI
overrides, rather than assuming one example YAML fully specifies the paper run.

Starting references:

| Dataset | Holdout-stage reference | Target-stage reference |
| --- | --- | --- |
| CIFAR-10 | `configs/experiment/cifar10_resnet18_irred.yaml` | `configs/experiment/cifar10_resnet18_main.yaml` |
| CIFAR-100 | `configs/experiment/cifar100_resnet18_irred.yaml` | `configs/experiment/cifar100_resnet18_main.yaml` |
| MNIST/QMNIST | `configs/experiment/qmnist_irr.yaml` | Select and document a compatible target configuration using `qmnist_datamodule`; do not substitute AmbiMNIST/infinite-MNIST experiments |

The CIFAR model references resolve to the repository's own ResNet variants;
preserve these rather than replacing them with torchvision models. Preserve
batch size, selection fraction, optimizer, epochs, augmentation, data splits,
and selection mode from the selected recipe.

Record evaluation semantics explicitly. `src/train.py` supports `eval_set=val`
and `eval_set=test`; the latter feeds the test dataset through the validation
loop and is described in the source as being used for final paper figures.
The CIFAR-10 main example inherits `eval_set=val`, while the CIFAR-100 main
example specifies `eval_set=test`. Resolve this against the intended result;
do not silently change datasets or checkpoint-selection rules to make names
consistent. Metrics named `val_*` can therefore describe test data.

Acceptance: a recorded recipe for each stage, including the actual evaluation
dataset, checkpoint monitor, and comparison baseline.

## 2. Prepare a separate legacy Conda environment for P100

Create a dedicated environment, provisionally named `rho-loss-p100`. Do not
modify or reuse the group's installed `online-bs-p100` environment.

Start from `my_environment.yaml`, whose environment name is `prio_train` and
whose key pins are:

| Component | Original Conda specification |
| --- | --- |
| Python | 3.9.6 |
| PyTorch / torchvision | 1.9.0 / 0.10.0 |
| CUDA toolkit | 11.1.74 |
| PyTorch Lightning | 1.4.6 |
| TorchMetrics | 0.5.1 |
| Hydra / OmegaConf | 1.1.1 / 2.1.1 |
| NumPy | 1.20.3 |
| W&B | 0.12.1 |

The initial creation command will be `conda env create -n rho-loss-p100 -f
my_environment.yaml`, from the repository root. This is the YAML environment
creation command; the README's `conda install --file` example is not suitable
for this YAML format.

The file is an old export containing exact build strings. If it no longer
solves, investigate package availability and prepare a separate, smaller
environment specification preserving the important versions above. Use the
Poetry lock as supporting evidence, while noting that its NumPy/CUDA artifacts
are not identical to the Conda export. Do not install today's unpinned
`requirements.txt` and assume it reproduces the old stack. Bootstrap tooling
may need pins because old packages import `pkg_resources`.

Install once on an approved internet-connected machine, outside GPU training
jobs, into a location visible to the allocated compute node. Verify checksum,
signature, and TLS checks; do not bypass them. Record a resolved environment
export, channels, package build identifiers, and installation instructions.

P100 is Pascal compute capability 6.0, supported by the original CUDA/PyTorch
generation. Actual readiness still requires a P100 allocation: confirm device
name, compute capability, driver/runtime versions, and a GPU tensor operation
before training. No cluster GPU validation has been completed here. Prior
cloud CPU smoke checks used a Poetry-derived environment and do not establish
that this exact Conda environment works on BYU.

Acceptance: reproducible installation, successful required imports, and a
successful P100 functional check. Log every version change in this document.

## 3. Download and verify datasets before submitting jobs

The submission workflow must first resolve the requested dataset and data
root, then ensure all required files are present and valid. Only after this
preflight succeeds may it invoke `sbatch` or launch training with `srun`.
An existing directory alone is not evidence that a download is complete.

Run the submission command on a BYU login node. It must synchronously perform
this sequence before transferring execution to an allocated compute node:

1. Resolve the experiment's dataset, required splits, and shared absolute data
   root on the login node.
2. Check whether the required files are already downloaded, extracted, and
   valid; reuse them when complete.
3. Download/extract any missing or corrupt data on the login node, then verify
   integrity and loading. If preparation fails, stop without submitting a job.
4. Submit the GPU job with `sbatch` (or start the requested allocation with
   `srun`) only after preparation succeeds.
5. Run training on the P100 compute node using the same shared data root,
   without downloading or otherwise accessing the internet.

Dataset preparation is part of the login-node submission command, not a
separate GPU job or an action delayed until compute-job startup. Prevent
concurrent extraction into the same dataset and reuse complete, verified
downloads. Follow BYU's policy for download and preparation work on login
nodes; that policy has not been independently verified here.

Required data:

| Workflow | Files/splits to stage |
| --- | --- |
| CIFAR-10 | Training and test batches, metadata, extraction, repository integrity checks |
| CIFAR-100 | Training and test data, metadata, extraction, repository integrity checks |
| MNIST/QMNIST | MNIST training and test images/labels; QMNIST test images/labels needed for `test50k` |

Use the actual classes in `src/datamodules/datasets/sequence_datasets.py`:
`indices_CIFAR10`, `indices_CIFAR100`, `indices_MNIST`, and `indices_QMNIST`.
The digit classes derive storage directories from their class names, so a
generic torchvision download into `MNIST/` or `QMNIST/` may not satisfy these
loaders. Verify `indices_MNIST/raw` and `indices_QMNIST/raw` layout and extraction
with the selected environment. Prefer the supported raw-data layout; inspect
legacy processed-data behavior before relying on older caches.

`QMNISTDataModule` uses MNIST training data for the target training set,
QMNIST `test50k` as holdout/validation data, and MNIST test data for testing.
Do not download only QMNIST, or change this split to a standard random MNIST
split. QMNIST's `test50k` is selected from the downloaded QMNIST test resources.

The repository loaders contain automatic download calls. Complete staging
must make those calls short-circuit locally. Before GPU submission, verify
file integrity, dataset lengths, metadata, and sample loading without network
access. If anything is missing or corrupt, fail before queueing; do not rely
on a compute node to repair the dataset. Include all stages' data needs in
preflight, not just the target stage's train split.

Audit selected recipes for any other remote dependencies. For the intended
CIFAR/digit recipes, use local model definitions rather than adding pretrained
model downloads. All checkpoints and irreducible-loss artifacts must be local.

Acceptance: a validated local data manifest and successful loading of every
required split with network unavailable. Submission refuses incomplete data.

## 4. Adapt the group's regular-job Slurm infrastructure

Preserve the structure of `online-batch-selection/run/run_utils/run_job.py`
and its examples where practical: `RunType`, dry runs, resource arguments,
node exclusions, subprocess error handling, and per-job stdout/stderr paths.

Adapt only the repository-specific pieces:

1. Construct `run_irreducible.py` or `run.py` commands with Hydra overrides,
   instead of `main.py --config`. Preserve quoting through argument lists and
   `shlex.join` for `sbatch --wrap`.
2. Explicitly use the dedicated Conda interpreter/environment and repository
   working directory; do not depend on whichever `python` happens to be active.
3. Require one GPU and the confirmed P100-only selector. Include an explicit
   note in launcher documentation/examples: **only submit to P100 nodes**.
4. Preserve configurable CPUs, RAM, wall time, job name, and node exclusions.
   Set resource defaults after a representative run; the supplied resource
   table lists a three-day maximum for P100 nodes, not a recommended request.
5. Remove standby QoS and standby-oriented requeue behavior from this workflow.
6. Run dataset preflight before submission, with a verification-only path when
   the submission machine itself is offline.
7. Give each dataset/method/stage/seed a unique persistent output directory,
   with separate Slurm logs and composed Hydra configuration. Do not rely on
   second-resolution timestamps to prevent collisions.
8. Return job IDs and use successful-completion dependencies for stages.
   Holdout jobs produce the model checkpoint and irreducible-loss artifact;
   target jobs validate the artifact before consuming it.

Data preparation completes before queueing; holdout training is itself a GPU
job. Its output does not exist at initial submission time, so artifact checks
belong both in stage coordination and at target-job startup.

Keep ordinary last/best checkpoints and document manual resume. Automated
standby recovery and copying the group's `extend_job` implementation are not
required. Avoid copying its merged-config generator unchanged: RHO-Loss uses
Hydra composition rather than the group's single merged YAML interface.

Acceptance: dry-run commands show P100 selection, regular-job flags, correct
interpreter/cwd, preflight, unique paths, and stage dependencies. Actual
scheduler validation waits for authorization and confirmed BYU requirements.

## 5. Configure offline W&B and later synchronization

Retain RHO-Loss's Lightning `WandbLogger` and existing metric names. Set offline
mode explicitly for all compute-node jobs using the supported logger setting
`logger.wandb.offline=True` and consistent W&B runtime configuration. Do not
attempt online authentication or uploads during training.

Write W&B files, configs, metrics, and checkpoints to persistent storage.
Assign unique run IDs/names and groups identifying dataset, method, stage,
seed, and reproduction recipe. Record GPU model, driver, environment versions,
commit, holdout artifact identity, evaluation dataset, and any deviations.

Emulate `online-batch-selection/wandb-sync-daemon.py` on a separate approved
internet-connected machine with access to the shared output storage. Preserve
its recursive discovery of `wandb/offline-run-*`, optional run-subdirectory
filters, configurable polling interval (CLI default: 15 seconds), and streamed
`wandb sync` output. Adapt its `./experiments` root to the chosen RHO-Loss output
root; do not assume Hydra run directories already match the group's layout.

Follow its completion-marker protocol: repeatedly sync active runs, write a
run-level `FINISHED` marker only after the experiment successfully completes
and W&B finishes flushing locally, and write `SYNCED` only after successful
sync of a finished run. Skip runs with both markers. A failed sync must remain
eligible for retry. Failed/incomplete experiments may still upload diagnostic
logs, but must not be marked successfully finished. Keep markers local to each
stage/seed run, and prevent overlapping daemon instances from racing over the
same output tree. No daemon process will run on GPU compute nodes.

The daemon may upload growing offline runs while training continues, as in the
group's implementation, and performs a final sync after completion. Authenticate
on that internet-connected machine and sync to the group's project/entity.
Inspect existing authentication before requesting credentials; never put keys
in scripts, this file, or job logs. Syncing should not require rerunning the
experiment. Verify offline logging and sync compatibility with pinned W&B
0.12.1; document any necessary W&B-only version change.

Preserve the native training/validation/final-test accuracy and loss metrics.
Do not relabel `val_*` as test results unless the recorded evaluation dataset
is actually the test split. Preserve checkpoint monitors and report which
checkpoint the final test evaluates.

Acceptance: a completed offline run can be synced with its metrics/config and
correct identity, without compute-node internet access. No sync is authorized
as part of this planning task.

## 6. Add geodesic progress using existing predictions only

RHO-Loss does not currently implement geodesic progress. Port only the
mathematical function from the group's
`methods/diagnostics/standard.py::_geodesic_progress`; do not port its
`ForwardPass` diagnostic or logarithmic evaluation schedule.

Preserve the original measurement semantics:

- `MultiModels.training_step` selects examples and computes target-model
  logits in train mode. Logged accuracy/loss use those pre-update logits on
  the selected minibatch. Selection itself also defaults to train mode.
- `OneModel.training_step` supplies the holdout-stage training predictions.
- Validation/test logits come from their original eval-mode loops.
- Native target/holdout train metrics log on step and epoch. With the default
  trainer configuration, logger emission is every 50 training steps and
  validation runs every epoch. Explicit recipe overrides take precedence.
- Some validation metrics have step and epoch forms; preserve their existing
  behavior rather than replacing all metrics with a new uniform schedule.

Attach progress computation where the corresponding logits and labels already
exist. Detach and copy only the data needed for diagnostics. Do not repeat
selection, forward a fixed training loader, switch `.train()`/`.eval()`, add
model hooks that run inference, or recompute predictions from a checkpoint.
The requirement is **no additional BatchNorm state changes caused by logging**;
the original training forwards continue updating BatchNorm as originally coded.

Compute batch training progress at the native logging opportunities. If an
epoch progress value is required, pool the existing predictions/labels and
fit the geodesic parameter over that collection. Do not average batch progress
values and call that pooled progress: the global optimization is nonlinear.
For training, label the epoch metric as progress over selected training
observations spanning the epoch, not full-dataset progress from one model state.
For validation/test, pool the existing evaluation predictions for the run of
that loop. Exclude Lightning sanity-validation observations from actual
experiment summaries.

Implement for both holdout and target models, without changing original loss,
accuracy aggregation, optimization, mode, data order, or augmentation.
Document names, prediction timing, data population, accumulation window, and
cadence. Record diagnostic overhead separately before comparing runtime.

Acceptance, when tests are authorized: function agreement with the group's
implementation; no increase in model forward-call counts; no changes in
BatchNorm buffers, RNG state, gradients, or weights attributable to enabling
logging; correct epoch aggregation and no sanity-loop contamination.

## 7. Run independent seeds with explicit holdout provenance

### Main runs versus irreducible-loss preparation

The primary experiments of interest are the main/target runs (`run.py`).
The `irr`/irreducible runs (`run_irreducible.py`) train a separate model on
holdout data and produce either a model checkpoint or precomputed losses for
the target training examples. RHO-Loss selects examples using target-model
loss minus this holdout-model loss; therefore the main RHO-Loss runs require
a compatible holdout artifact. The preparation stage is not a competing
selection method or the primary reported target-model experiment.

Reuse an existing verified artifact only when it matches the dataset, indexing,
split, model/recipe requirements, and that replicate's irreducible-model seed.
If none exists, prepare it on a P100 for that replicate. Do not train on login
nodes, substitute zeros for holdout losses, or rerun preparation needlessly.
Keep preparation runs separate from the main-run plots and groups, while
recording their provenance and compute cost.

The selected study varies the complete pipeline: each replicate changes both
the holdout-model seed and target-model seed and consumes its own corresponding
holdout artifact. Do not fix the irreducible-stage seed or share its artifact
across replicates. Keep the two stage seeds explicit in configuration and
metadata. A simple proposed mapping is replicate `s`: `irr_seed=s` and
`main_seed=s`; these values both change between replicates. This does not
require changing the repository's original dataset splits.

Keep RHO-Loss's `seed_everything(seed, workers=True)` behavior and original
dataset splits. Both stages already accept a top-level `seed` override.
Do not introduce a new deterministic-training policy without recording it.

The user selected seeds 1 through 5 and an extra base-default reference pair
`(irr=1, main=12)` for all three workflows. The native MNIST recipe explicitly
uses main seed 1, so its default pair is already in the paired sweep; the user
also requested the extra base-default reference trial. Defaults are not confirmed published-run
seeds, and this list is not a claim about the paper's seed count.
Choose and record the seed list before launching experiments. Report all
replicates rather than selecting the best seeds or best holdout model; record
failures and any same-seed retries explicitly. This evaluates sensitivity to
seeds, but cannot by itself prove whether the authors selected seeds.

Require separate complete pipelines: for each replicate, train a holdout
model, record its checkpoint/loss artifact, and train the target model using
that artifact. The CIFAR-100 example selects an on-the-fly holdout model rather
than precomputed losses; preserve the selected paper recipe or explicitly
document a switch. Do not silently switch artifact formats for convenience.

Pair RHO-Loss and uniform-comparison target runs by seed and recipe. Preserve
the paper baseline's batch/update budget and hyperparameters rather than
assuming the same wall time or nominal epoch count makes a fair comparison.
Do not substitute a fixed-holdout sweep for this complete-pipeline study.

Include seed, stage, method, and artifact identity in directory naming and W&B
metadata. Aggregate completed seeds as individual curves plus mean/standard
deviation and count. Preserve failed/incomplete outcomes; never average them
as zero-valued completed experiments. Align comparisons on the original
training coordinate and record actual update/sample counts where available.

Acceptance: every run maps unambiguously to its seed, configuration, stage,
and holdout artifact; paired baselines and aggregate summaries are traceable.

## Implementation order and validation gates

1. Confirm BYU P100 selector, shared-storage paths, environment placement,
   wall-time rules, and approved preparation/sync machines.
2. Record exact CIFAR-10 recipe and prepare the pinned P100 environment.
3. Add dataset staging/integrity preflight for CIFAR-10, CIFAR-100, MNIST/QMNIST.
4. Adapt regular-job launcher/examples, unique directories, and dependencies.
5. Configure offline W&B and add prediction-only progress instrumentation.
6. Add configurable seed pipelines and experiment/result manifests.
7. Once authorized, validate diagnostics without changing model behavior,
   offline data loading, P100 execution, and W&B sync.
8. Complete one CIFAR-10 holdout/target/baseline pipeline before submitting the
   seed sweep. Then validate CIFAR-100 and the MNIST/QMNIST recipes similarly.

GPU training, submission, downloads, environment installation, and W&B uploads
are future execution steps, not actions performed by saving this plan.

## Deviations and validation record

Update this table during implementation; record actual changes, not only plans.
Keep it in the fork alongside the reproduction instructions.

| Area | Original behavior | Adopted behavior | Reason | Validation |
| --- | --- | --- | --- | --- |
| Environment | Historical `my_environment.yaml` | Separate `environment-p100.yaml`; original pins plus pinned development tools | P100 reproduction | Pin preservation checked; Conda solve/P100 not tested |
| Scheduling | Author-specific Slurm paths/partitions | BYU P100 regular jobs; 4 CPUs, 32G, one day; required site paths | Cluster adaptation | Mocked submission/dependencies checked; Slurm not contacted |
| Data preparation | Loaders can download when data is absent | Login-node staging/integrity preflight before submission; compute verification only | Compute nodes are offline | Local corrupt-file/verification checks; real downloads not performed |
| W&B | Online logger default | Offline `ksmiller-byu-research/ValidateRhoLoss` plus adapted daemon | Compute nodes are offline | Configuration and mocked syncing checked; no real sync |
| Progress | No geodesic progress metric | Optional reuse of native predictions and pooled epoch fitting | Additional research diagnostic | State/forward/RNG invariance checked on synthetic native steps |
| Seeds | Individual recipe overrides | Explicit paired 1–5 pipelines plus documented reference trial | Replication/variability study | Seed/artifact identity and dependencies checked |

## Remaining inputs and known limits

- Confirm BYU paths, P100 selector, W&B project/entity, approved online machine,
  and desired seed count before actual submissions.
- BYU's website was inaccessible through the cloud network proxy during the
  assessment. `pascal` comes from the group's repository, not a verified current
  scheduler query. The supplied hardware table lists P100s on `marylou9g`.
- Existing RHO-Loss defaults include author-specific dataset/checkpoint paths.
  Override them with recorded local paths; holdout training must generate the
  artifacts needed by the target stage.
- The repository's MNIST unit test imports a module absent from this checkout.
  Do not treat that stale test as proof the actual QMNIST datamodule is broken,
  or count a collection failure as a passing readiness check.
- Prior cloud CPU smoke validation is distinct from a full paper reproduction,
  a P100 GPU check, Slurm execution, offline dataset readiness, and W&B sync.
