"""Prediction-only diagnostics, adapted from online-batch-selection.

No model is called here. Epoch training observations span changing model
states; this metric is not a full-dataset evaluation of one checkpoint.
"""

import numpy as np
import torch
from scipy import optimize


def geodesic_progress(log_probs, labels):
    """Fit the uniform-to-ground-truth geodesic parameter, in [0, 1]."""
    if log_probs.numel() == 0:
        return 0.0
    probabilities = torch.exp(log_probs.detach().cpu())
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(
        1e-12
    )
    predictions = np.sqrt(probabilities.numpy().astype(np.float64, copy=False))
    labels = labels.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    count, classes = predictions.shape
    if labels.size != count or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Progress requires one valid class label per prediction.")
    ground_truth = np.zeros((count, classes), dtype=np.float64)
    ground_truth[np.arange(count), labels] = 1.0
    ignorance = np.sqrt(np.full((count, classes), 1.0 / classes))
    cosine_ig = np.clip((ignorance * ground_truth).sum(axis=1), 0.0, 1.0)
    cosine_ip = np.clip((ignorance * predictions).sum(axis=1), 0.0, 1.0)
    cosine_gp = np.clip((ground_truth * predictions).sum(axis=1), 0.0, 1.0)
    angles = np.arccos(cosine_ig)
    degenerate = angles < 1e-8
    distance = float(np.arccos(cosine_ip[degenerate]).sum())
    if np.all(degenerate):
        return 0.0
    angles, cosine_ip, cosine_gp = (
        values[~degenerate] for values in (angles, cosine_ip, cosine_gp)
    )

    def objective(t):
        cosine = cosine_ip * np.sin((1.0 - t) * angles) / np.sin(
            angles
        ) + cosine_gp * np.sin(t * angles) / np.sin(angles)
        return distance + float(np.arccos(np.clip(cosine, 0.0, 1.0)).sum())

    value = optimize.minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded").x
    return float(np.clip(value, 0.0, 1.0))


class PredictionProgressMixin:
    """Reuse logits at native step/epoch logging points; never run inference."""

    def _reset_progress(self, split):
        if not hasattr(self, "_progress_observations"):
            self._progress_observations = {}
        self._progress_observations[split] = []

    def _record_progress(self, split, logits, labels):
        if not self.hparams.get("progress_logging", False):
            return
        if self.trainer.sanity_checking:
            return
        log_probs = torch.log_softmax(logits.detach(), dim=1).cpu()
        labels = labels.detach().cpu().clone()
        if not hasattr(self, "_progress_observations"):
            self._progress_observations = {}
        self._progress_observations.setdefault(split, []).append((log_probs, labels))
        # Use Lightning's own emission decision, including manual optimization.
        if split == "train" and (
            self.trainer.logger_connector.should_update_logs
            or self.trainer.fast_dev_run
        ):
            self.log(
                "train_progress_step",
                geodesic_progress(log_probs, labels),
                on_step=True,
                on_epoch=False,
                logger=True,
            )

    def _finish_progress(self, split):
        observations = getattr(self, "_progress_observations", {}).pop(split, [])
        if not observations or self.trainer.sanity_checking:
            return
        log_probs, labels = zip(*observations)
        # Refit on pooled predictions: averaging fitted batch values is incorrect.
        self.log(
            split + "_progress_epoch",
            geodesic_progress(torch.cat(log_probs), torch.cat(labels)),
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def on_train_epoch_start(self):
        self._reset_progress("train")
        super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        self._reset_progress("val")
        super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        self._reset_progress("test")
        super().on_test_epoch_start()

    def on_train_epoch_end(self, unused=None):
        self._finish_progress("train")
        super().on_train_epoch_end(unused)

    def on_validation_epoch_end(self):
        self._finish_progress("val")
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        self._finish_progress("test")
        super().on_test_epoch_end()
