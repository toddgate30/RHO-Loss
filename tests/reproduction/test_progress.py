import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from src.curricula.selection_methods import uniform_selection, reducible_loss_selection
from src.models.OneModel import OneModel
from src.models.MultiModels import MultiModels
from src.utils.geodesic_progress import geodesic_progress


@pytest.mark.parametrize("t", [0.0, 0.2, 0.7, 1.0])
def test_recovers_known_geodesic_parameter(t):
    classes = 4
    labels = torch.tensor([0, 1, 2, 3])
    uniform = np.full((4, classes), 1 / np.sqrt(classes))
    truth = np.eye(classes)
    angle = np.arccos(1 / np.sqrt(classes))
    point = (np.sin((1 - t) * angle) * uniform + np.sin(t * angle) * truth) / np.sin(
        angle
    )
    log_probs = torch.as_tensor(np.log(np.clip(point ** 2, 1e-30, 1)))
    assert geodesic_progress(log_probs, labels) == pytest.approx(t, abs=3e-5)


@pytest.mark.parametrize(
    "model_class,selection",
    [
        (OneModel, None),
        (MultiModels, uniform_selection),
        (MultiModels, reducible_loss_selection),
    ],
)
def test_logging_reuses_predictions_without_changing_training_state(
    model_class, selection
):
    torch.manual_seed(73)
    network = nn.Sequential(nn.BatchNorm1d(6), nn.Dropout(0.3), nn.Linear(6, 3))
    state = copy.deepcopy(network.state_dict())
    batch = (torch.arange(12), torch.randn(12, 6), torch.randint(0, 3, (12,)))
    rng = torch.get_rng_state()
    outcomes = []
    for enabled in [False, True]:
        net = copy.deepcopy(network)
        net.load_state_dict(state)
        if model_class is OneModel:
            module = OneModel(net, OmegaConf.create({}), None, progress_logging=enabled)
        else:
            datamodule = SimpleNamespace(
                num_classes=3, percentage_corrupted=lambda indices: None
            )
            module = MultiModels(
                net,
                selection_method=selection(),
                irreducible_loss_generator=torch.zeros(12),
                datamodule=datamodule,
                percent_train=0.5,
                progress_logging=enabled,
            )
        module.trainer = SimpleNamespace(
            sanity_checking=False,
            fast_dev_run=False,
            current_epoch=0,
            global_step=0,
            logger_connector=SimpleNamespace(should_update_logs=True),
        )
        logs = {}
        module.log = lambda key, value, **kwargs: logs.update({key: value})
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        calls = []
        handle = net.register_forward_hook(lambda *args: calls.append(1))
        module.optimizers = lambda: optimizer
        module.manual_backward = lambda loss: loss.backward()
        torch.set_rng_state(rng)
        if model_class is OneModel:
            optimizer.zero_grad()
            loss = module.training_step(batch, 0)
            loss.backward()
            optimizer.step()
        else:
            module.training_step(batch, 0)
        module._finish_progress("train")
        outcomes.append(
            (copy.deepcopy(net.state_dict()), torch.get_rng_state(), calls, logs)
        )
        handle.remove()
    native, instrumented = outcomes
    expected_calls = [1, 1] if selection is reducible_loss_selection else [1]
    assert native[2] == instrumented[2] == expected_calls
    assert torch.equal(native[1], instrumented[1])
    for key in native[
        0
    ]:  # Includes BatchNorm running mean/variance and num_batches_tracked.
        assert torch.equal(native[0][key], instrumented[0][key]), key
    for key in ["train_acc", "train_loss"]:
        assert torch.equal(native[3][key], instrumented[3][key])
    assert "train_progress_step" in instrumented[3]
    assert "train_progress_epoch" in instrumented[3]


def test_epoch_pooling_and_sanity_validation_exclusion():
    module = OneModel(
        nn.Linear(2, 3), OmegaConf.create({}), None, progress_logging=True
    )
    trainer = SimpleNamespace(
        sanity_checking=True,
        fast_dev_run=False,
        logger_connector=SimpleNamespace(should_update_logs=False),
    )
    module.trainer = trainer
    logs = {}
    module.log = lambda key, value, **kwargs: logs.update({key: value})
    logits = [torch.tensor([[4.0, 0, 0], [0, 2.0, 0]]), torch.tensor([[1.0, 2.0, 3.0]])]
    labels = [torch.tensor([0, 1]), torch.tensor([0])]
    module._reset_progress("val")
    module._record_progress("val", logits[0], labels[0])
    module._finish_progress("val")
    assert not logs
    trainer.sanity_checking = False
    module._reset_progress("val")
    for predictions, target in zip(logits, labels):
        module._record_progress("val", predictions, target)
    module._finish_progress("val")
    expected = geodesic_progress(
        torch.cat([torch.log_softmax(x, 1) for x in logits]), torch.cat(labels)
    )
    assert logs["val_progress_epoch"] == expected
    assert "val" not in module._progress_observations
